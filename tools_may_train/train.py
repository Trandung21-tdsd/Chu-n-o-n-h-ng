#!/usr/bin/env python3
"""Train 1 trong 4 model trên máy Linux (GPU AMD RX570 qua ROCm, hoặc NVIDIA/CPU) bằng pipeline của dự án.

Được train_may_linux.sh gọi:  python3 train.py --model ResNet-50 --batch_size 16 --accumulation 2 --precision FP32
Chạy tay cũng được (từ bất kỳ đâu):
    python3 tools_may_train/train.py --model MobileNetV2
    python3 tools_may_train/train.py --model Swin-T --plan             # chỉ xem kế hoạch, chưa train
    python3 tools_may_train/train.py --model ResNet-50 --epochs 2      # chạy thử nhanh

Dữ liệu: dùng ĐÚNG phân chia train/val/test đã tạo (data/prepared/<task>/manifest.csv, copy từ máy Mac) để kết quả
so sánh được giữa 2 máy; ảnh được tìm lại theo thư mục data/ của dự án trên máy này. Chưa có manifest thì tự chạy prepare.
Kết quả: outputs/<task>_linux_rx570/ (benchmark_results.csv, comparison.md, charts/, tato/ ...).
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parent
sys.path.insert(0, str(PROJECT / "src"))

# Cache dùng chung giữa các lần chạy (container Docker bị xoá sau mỗi lần chạy):
#   MIOpen (thư viện conv của AMD) biên dịch kernel lần đầu rất lâu -> lưu lại; trọng số pretrained tải 1 lần.
CACHE = PROJECT / ".cache"
for key, sub in {"MIOPEN_USER_DB_PATH": "miopen", "MIOPEN_CUSTOM_CACHE_DIR": "miopen", "TORCH_HOME": "torch",
                 "MPLCONFIGDIR": "matplotlib"}.items():
    os.environ.setdefault(key, str(CACHE / sub))
    Path(os.environ[key]).mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MIOPEN_LOG_LEVEL", "3")  # bớt cảnh báo hipBLAS/MIOpen

MODELS = {  # tên trong menu train_may_linux.sh -> tên trong pipeline
    "ResNet-50": "resnet50",
    "Swin-T": "swin_t",
    "MobileNetV2": "mobilenet_v2",
    "DenseNet-121": "densenet121",
}
HEAVY = {"resnet50", "swin_t", "densenet121"}


def gpu_report(allow_cpu: bool):
    import torch

    from throat_ai.models import pick_device

    device = pick_device("auto")
    hip = getattr(torch.version, "hip", None)
    print(f"[*] PyTorch {torch.__version__} | ROCm/HIP {hip or '-'} | CUDA {torch.version.cuda or '-'}")
    if device.type == "cuda":
        props = torch.cuda.get_device_properties(0)
        vram = props.total_memory / 2**30
        print(f"[+] GPU: {props.name} | VRAM {vram:.1f} GB | backend {'ROCm (AMD)' if hip else 'CUDA (NVIDIA)'}")
        return device, vram
    if device.type == "cpu" and not allow_cpu:
        sys.exit("[!] Không thấy GPU (torch.cuda.is_available() = False). Với RX570 phải chạy trong Docker gfx803 "
                 "(xem tools_may_train/README.md; kiểm tra: ./chay_trong_docker.sh python3 kiem_tra_gpu.py). Muốn train bằng CPU (rất chậm): --allow-cpu")
    print(f"[!] Train trên {device} (không phải GPU AMD/NVIDIA).")
    return device, None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", required=True, help=" | ".join(MODELS) + " (hoặc tên bất kỳ trong pipeline)")
    parser.add_argument("--batch_size", "--batch-size", type=int, default=None)
    parser.add_argument("--accumulation", type=int, default=None,
                        help="số bước cộng dồn gradient (batch hiệu dụng = batch_size x accumulation)")
    parser.add_argument("--precision", default="FP32", choices=["FP32", "AMP", "fp32", "amp"],
                        help="RX570 không tăng tốc FP16 -> mặc định FP32")
    parser.add_argument("--task", default="bagls_3class")
    parser.add_argument("--out", default=None, help="mặc định outputs/<task>_linux_rx570")
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--resolution", type=int, default=224)
    parser.add_argument("--seeds", nargs="+", default=["42"])
    parser.add_argument("--dup-policy", nargs="+", default=["dedup"])
    parser.add_argument("--allow-cpu", action="store_true")
    parser.add_argument("--plan", action="store_true", help="chỉ in kế hoạch")
    args, extra = parser.parse_known_args()  # tham số khác được chuyển nguyên cho throat_ai.benchmark

    model = MODELS.get(args.model, args.model)
    device, vram = gpu_report(args.allow_cpu or args.plan or bool(os.environ.get("THROAT_AI_ALLOW_CPU")))
    batch = args.batch_size or (32 if model not in HEAVY else 16)
    accumulation = args.accumulation or max(1, 32 // batch)
    limit = (8 if model in HEAVY else 16) if (vram is not None and vram < 6) else None  # RX570 bản 4 GB
    if limit and batch > limit:
        accumulation = max(1, (batch * accumulation) // limit)
        batch = limit
        print(f"[!] VRAM {vram:.1f} GB: giảm batch còn {batch} x accumulation {accumulation} (giữ batch hiệu dụng) để tránh hết bộ nhớ.")
    print(f"[*] Model {args.model} -> {model} | batch {batch} x {accumulation} = {batch * accumulation} | {args.precision.upper()}")

    from throat_ai import benchmark, prepare
    from throat_ai.utils import DEFAULT_PREPARED_DIR, get_task, resolve_config_path

    manifest = DEFAULT_PREPARED_DIR / args.task / "manifest.csv"
    if not manifest.exists():
        print(f"[!] Chưa có {manifest} -> tạo phân chia trên máy này. Để so sánh được với máy Mac, nên copy thư mục "
              f"data/prepared/{args.task} từ máy Mac sang (dong_goi_tu_mac.sh làm việc này).")
        prepare.main(["--task", args.task])
    # Ảnh nằm ở data/ của dự án TRÊN MÁY NÀY (manifest copy từ Mac ghi đường dẫn tuyệt đối của Mac).
    roots = [f"{s['name']}={resolve_config_path(s['root'])}" for s in get_task(args.task)["sources"]]
    out = args.out or str(PROJECT / "outputs" / f"{args.task}_linux_rx570")
    argv = ["--task", args.task, "--out", out, "--matrix", str(PROJECT / "configs" / "models_linux_rx570.json"),
            "--models", model, "--clahe", "off",
            "--dup-policy", *args.dup_policy, "--seeds", *map(str, args.seeds),
            "--batch-size", str(batch), "--accumulation", str(accumulation), "--precision", args.precision.upper(),
            "--workers", str(args.workers), "--resolution", str(args.resolution),
            "--epochs", str(args.epochs), "--patience", str(args.patience), "--device", str(device)]
    for root in roots:
        argv += ["--source-root", root]
    if args.plan:
        argv.append("--plan")
    benchmark.main(argv + extra)
    if args.plan:
        return

    import pandas as pd

    results = Path(out) / "benchmark_results.csv"
    if results.exists():
        df = pd.read_csv(results)
        mine = df[df["model"] == model]
        if not mine.empty:
            last = mine.iloc[-1]
            print(f"\n[*] {args.model}: trạng thái {last['status']} | best val macro-F1 {last.get('best_val_score')}"
                  + (f" | lỗi: {last.get('abort_reason')}" if last["status"] != "completed" else ""))
            if (mine["status"] == "completed").any():
                return
    sys.exit(1)  # để train_may_linux.sh / telemetry biết lần train này hỏng


if __name__ == "__main__":
    main()
