#!/usr/bin/env python3
"""Kiểm tra GPU trước khi train: PyTorch thấy GPU AMD chưa, phép tính trên GPU có ĐÚNG không (so với CPU),
train thử vài bước với cả 4 kiến trúc (ResNet-50, Swin-T, MobileNetV2, DenseNet-121) bằng dữ liệu giả.

    ./chay_trong_docker.sh python3 kiem_tra_gpu.py           # RX570 (trong Docker)
    python3 kiem_tra_gpu.py --quick                          # chỉ kiểm tra phép tính, bỏ phần 4 model

Mã thoát 0 = dùng được; khác 0 = có lỗi (in lý do).
"""
from __future__ import annotations

import argparse
import os
import sys
import time

os.environ.setdefault("MIOPEN_LOG_LEVEL", "3")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--batch", type=int, default=8)
    args = parser.parse_args()
    try:
        import torch
        import torchvision
    except ImportError as exc:
        print(f"[✖] Chưa cài PyTorch: {exc}")
        return 2
    hip = getattr(torch.version, "hip", None)
    print(f"PyTorch {torch.__version__} | torchvision {torchvision.__version__} | ROCm/HIP {hip or '-'} | CUDA {torch.version.cuda or '-'}")
    for key in ("HSA_OVERRIDE_GFX_VERSION", "ROC_ENABLE_PRE_VEGA", "PYTORCH_ROCM_ARCH"):
        print(f"  {key}={os.environ.get(key, '(chưa đặt)')}")
    if not torch.cuda.is_available():
        print("[✖] torch.cuda.is_available() = False -> PyTorch KHÔNG thấy GPU.\n"
              "    RX570 cần PyTorch biên dịch cho gfx803: môi trường .rx570-venv (cai_khong_docker_rx570.sh)\n"
              "    hoặc Docker (cai_moi_truong_rx570.sh). User phải thuộc nhóm video + render, có /dev/kfd.")
        return 1
    props = torch.cuda.get_device_properties(0)
    print(f"[+] GPU: {props.name} | VRAM {props.total_memory / 2**30:.1f} GB | arch {getattr(props, 'gcnArchName', '?')}")
    device = torch.device("cuda:0")

    # 1) Phép tính đúng? (GPU chạy nhưng ra số sai là lỗi hay gặp khi ép kiến trúc không hỗ trợ)
    torch.manual_seed(0)
    a, b = torch.randn(512, 512), torch.randn(512, 512)
    x, w = torch.randn(4, 3, 64, 64), torch.randn(16, 3, 3, 3)
    try:
        mm = (a.to(device) @ b.to(device)).cpu()
        conv = torch.nn.functional.conv2d(x.to(device), w.to(device), padding=1).cpu()
    except Exception as exc:
        print(f"[✖] Lỗi khi tính trên GPU: {type(exc).__name__}: {exc}")
        return 1
    err_mm = (mm - a @ b).abs().max().item()
    err_conv = (conv - torch.nn.functional.conv2d(x, w, padding=1)).abs().max().item()

    # Lan truyền ngược (train cần) + BatchNorm: MIOpen trong image là bản ROCm 6.4 gốc, không build lại cho gfx803.
    def grads(dev):
        xi = x.clone().to(dev).requires_grad_(True)
        wi = w.clone().to(dev).requires_grad_(True)
        bn = torch.nn.BatchNorm2d(16).to(dev).train()
        out = bn(torch.nn.functional.conv2d(xi, wi, padding=1)).relu()
        (out * out).mean().backward()
        return out.detach().cpu(), xi.grad.cpu(), wi.grad.cpu()

    try:
        gpu_out, gpu_gx, gpu_gw = grads(device)
    except Exception as exc:
        print(f"[✖] Lỗi khi lan truyền ngược trên GPU: {type(exc).__name__}: {exc}")
        return 1
    cpu_out, cpu_gx, cpu_gw = grads(torch.device("cpu"))
    rel = lambda g, c: ((g - c).abs().max() / (c.abs().max() + 1e-12)).item()  # noqa: E731
    err_bn, err_gx, err_gw = rel(gpu_out, cpu_out), rel(gpu_gx, cpu_gx), rel(gpu_gw, cpu_gw)
    print(f"  Sai khác GPU vs CPU: matmul {err_mm:.2e} | conv2d {err_conv:.2e} | conv+BatchNorm {err_bn:.2e} | "
          f"gradient ảnh {err_gx:.2e} | gradient trọng số {err_gw:.2e}")
    if not (err_mm < 1e-2 and err_conv < 1e-3 and max(err_bn, err_gx, err_gw) < 1e-3):
        print("[✖] GPU cho KẾT QUẢ SAI — không được train (thử kernel khác / image khác, xem README).")
        return 1
    print("[✔] Phép tính trên GPU đúng (cả chiều xuôi lẫn lan truyền ngược).")
    if args.quick:
        return 0

    # 2) Train thử 3 bước mỗi model (lần đầu MIOpen biên dịch kernel nên chậm; các lần sau dùng cache)
    from torchvision import models

    failed = []
    for name, build in [("MobileNetV2", models.mobilenet_v2), ("ResNet-50", models.resnet50),
                        ("DenseNet-121", models.densenet121), ("Swin-T", models.swin_t)]:
        try:
            model = build(num_classes=3).to(device)
            optimizer = torch.optim.AdamW(model.parameters(), 1e-4)
            images = torch.randn(args.batch, 3, 224, 224, device=device)
            labels = torch.randint(0, 3, (args.batch,), device=device)
            torch.cuda.reset_peak_memory_stats()
            start = time.time()
            for step in range(3):
                loss = torch.nn.functional.cross_entropy(model(images), labels)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                torch.cuda.synchronize()
                if step == 0:
                    first = time.time() - start
                    start = time.time()
            per_step = (time.time() - start) / 2
            if not torch.isfinite(loss):
                raise FloatingPointError("loss = NaN/inf")
            print(f"[✔] {name:<13} batch {args.batch}: bước đầu {first:5.1f}s (biên dịch), sau đó {per_step:.2f}s/bước, "
                  f"VRAM đỉnh {torch.cuda.max_memory_allocated() / 2**30:.2f} GB")
        except Exception as exc:
            failed.append(name)
            print(f"[✖] {name}: {type(exc).__name__}: {str(exc)[:200]}")
        finally:
            model = optimizer = images = labels = None
            torch.cuda.empty_cache()
    if failed:
        print(f"Model lỗi: {', '.join(failed)}")
        return 1
    print("\nTất cả OK — có thể train: ./train_may_linux.sh")
    return 0


if __name__ == "__main__":
    sys.exit(main())
