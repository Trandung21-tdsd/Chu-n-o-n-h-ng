"""BƯỚC 2 — Benchmark nhiều model: (ma trận cấu hình) × chính sách duplicate × seed (CLAHE mặc định TẮT).

Ví dụ:
    python -m throat_ai.benchmark --task bagls_3class --plan          # xem trước danh sách run, chưa train
    python -m throat_ai.benchmark --task bagls_3class                 # chạy toàn bộ ma trận mặc định
    python -m throat_ai.benchmark --task bagls_3class --models resnet18 efficientnet_b0 --seeds 42 43 44
    python -m throat_ai.benchmark --task bagls_3class --dup-policy dedup keep naive   # thêm minh họa leakage

    # Profiling cho TATO (máy GPU NVIDIA, ma trận C01–C08, chạy lại với --thermal-state warm / pre-stressed):
    python -m throat_ai.benchmark --task bagls_3class --matrix configs/tato_c01_c08.json \
        --out outputs/bagls_3class_tato --clahe off --dup-policy dedup --seeds 42 43 \
        --thermal-state cold --precondition cold --precondition-seconds 900 \
        --stop-on-target --target-quality 0.85 --require-gpu-telemetry --shuffle-order

Mỗi lần train ghi thêm 1 dòng vào <out>/benchmark_results.csv. Chạy lại đúng lệnh cũ sẽ BỎ QUA run đã xong
(tiếp tục sau khi bị ngắt/tắt máy); thêm --force để chạy lại. Kết thúc: tự chọn model theo validation,
đánh giá test, viết comparison.md và xuất dữ liệu TATO (tắt bằng --no-finalize).
"""
from __future__ import annotations

import argparse
import os
import random
import time
from pathlib import Path

from .data import DUP_POLICIES, attach_paths_and_targets, load_prepared, verify_files
from .evaluate import finalize
from .models import model_table, pick_device, resolve_model_name
from .report import load_summaries, write_reports
from .thermal import STATES
from .train import MONITOR_CHOICES, RunContext, RunSpec, TrainSettings, compute_run_key, train_run
from .utils import (
    CONFIG_DIR,
    DEFAULT_OUTPUT_DIR,
    DEFAULT_PREPARED_DIR,
    default_machine_id,
    iso_now,
    log,
    optional_float,
    parse_key_values,
    read_json,
    resolve_cli_path,
    setup_console,
    timestamp,
    warn,
    write_json,
)

try:
    import psutil
except ImportError:
    psutil = None


class BenchmarkLock:
    """Không cho 2 tiến trình benchmark cùng ghi vào một thư mục kết quả."""

    def __init__(self, out_dir: Path) -> None:
        self.path = Path(out_dir) / ".benchmark.lock"

    def __enter__(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if self.path.exists():
            text = self.path.read_text(encoding="utf-8").strip()
            pid = int(text) if text.isdigit() else None
            alive = psutil.pid_exists(pid) if (psutil is not None and pid) else True
            if alive:
                raise RuntimeError(f"Một benchmark khác (PID {pid}) đang ghi vào {self.path.parent}. "
                                   f"Nếu chắc chắn không còn chạy, xóa file {self.path}")
            self.path.unlink()
        with open(self.path, "x", encoding="utf-8") as handle:
            handle.write(str(os.getpid()))
        return self

    def __exit__(self, *exc) -> None:
        self.path.unlink(missing_ok=True)


def load_matrix(path: Path) -> list[dict]:
    data = read_json(path)
    configs = data.get("configs") if isinstance(data, dict) else data
    if not isinstance(configs, list) or not configs:
        raise ValueError(f"{path}: cần danh sách cấu hình (list) hoặc {{'configs': [...]}}")
    result = []
    for index, config in enumerate(configs):
        model = config.get("model") or config.get("backbone")
        if not model:
            raise ValueError(f"{path}: cấu hình #{index + 1} thiếu 'model' (hoặc 'backbone')")
        item = {
            "config_id": str(config.get("config_id") or f"C{index + 1:02d}"),
            "model": resolve_model_name(model),
            "batch_size": int(config.get("batch_size", 32)),
            "accumulation": int(config.get("accumulation", 1)),
            "precision": str(config.get("precision", "FP32")).upper(),
            "workers": int(config.get("workers", 4)),
            "resolution": int(config.get("resolution", 224)),
        }
        _validate_config(item, path)
        result.append(item)
    return result


def _validate_config(item: dict, source) -> None:
    if item["precision"] not in ("FP32", "AMP"):
        raise ValueError(f"{source}: precision phải là FP32 hoặc AMP ({item})")
    if item["batch_size"] < 1 or item["accumulation"] < 1 or item["workers"] < 0 or item["resolution"] < 32:
        raise ValueError(f"{source}: giá trị không hợp lệ ({item})")


def select_configs(configs: list[dict], args) -> list[dict]:
    if args.configs:
        unknown = set(args.configs) - {c["config_id"] for c in configs}
        if unknown:
            raise ValueError(f"Không có config_id {sorted(unknown)} trong ma trận")
        configs = [c for c in configs if c["config_id"] in args.configs]
    if args.models:
        wanted = {resolve_model_name(m) for m in args.models}
        missing = wanted - {c["model"] for c in configs}
        configs = [c for c in configs if c["model"] in wanted]
        for index, model in enumerate(sorted(missing)):  # model chưa có trong ma trận -> thêm cấu hình mặc định
            configs.append({"config_id": f"X{index + 1:02d}", "model": model, "batch_size": 32, "accumulation": 1,
                            "precision": "FP32", "workers": 4, "resolution": 224})
    overrides = {"batch_size": args.batch_size, "accumulation": args.accumulation, "precision": args.precision,
                 "workers": args.workers, "resolution": args.resolution}
    for config in configs:
        for key, value in overrides.items():
            if value is not None:
                config[key] = value.upper() if isinstance(value, str) else value
        _validate_config(config, "tham số dòng lệnh")
    if not configs:
        raise ValueError("Không còn cấu hình nào sau khi lọc --configs/--models")
    return configs


def expand_jobs(configs: list[dict], clahe_values: list[bool], policies: list[str], seeds: list[int],
                shuffle: bool) -> list[RunSpec]:
    jobs = [RunSpec(clahe=clahe, dup_policy=policy, seed=seed, **config)
            for seed in seeds for config in configs for clahe in clahe_values for policy in policies]
    if shuffle:  # trộn thứ tự để trạng thái nhiệt không trùng khớp với 1 cấu hình (khuyên dùng cho TATO)
        random.Random(2026).shuffle(jobs)
    return jobs


def settings_from_args(args) -> TrainSettings:
    return TrainSettings(
        epochs=args.epochs, patience=args.patience, min_delta=args.min_delta, monitor=args.monitor, lr=args.lr,
        head_lr_mult=args.head_lr_mult, weight_decay=args.weight_decay, warmup_epochs=args.warmup_epochs,
        label_smoothing=args.label_smoothing, balance=args.balance, grad_clip=args.grad_clip,
        augment=not args.no_augment, pretrained=not args.no_pretrained, deterministic=args.deterministic,
        amp_fallback_fp32=args.amp_fallback_fp32, target_metric=args.target_metric,
        target_quality=args.target_quality, target_consecutive=args.target_consecutive,
        stop_on_target=args.stop_on_target, thermal_state=args.thermal_state, precondition=args.precondition,
        precondition_seconds=args.precondition_seconds, pre_run_seconds=args.pre_run_seconds,
        telemetry=args.telemetry or args.require_gpu_telemetry or args.precondition != "none",
        telemetry_interval=args.telemetry_interval, ambient_temp_c=args.ambient_temp, nvml_index=args.nvml_index,
        require_gpu_telemetry=args.require_gpu_telemetry, max_gpu_temp_c=args.max_gpu_temp,
        max_cpu_temp_c=args.max_cpu_temp, max_minutes=args.max_minutes, max_energy_wh=args.max_energy_wh,
        eval_batch_size=args.eval_batch_size,
    )


def _print_plan(specs: list[RunSpec], keys: list[str], done: dict, out_dir: Path) -> None:
    pending = sum(k not in done for k in keys)
    log(f"Kế hoạch: {len(specs)} run ({len(specs) - pending} đã xong, {pending} sẽ chạy) -> {out_dir}")
    print(f"{'#':>3}  {'config':<6} {'model':<20} {'batch':>9} {'prec':<5} {'wk':>2} {'res':>4} {'CLAHE':<5} "
          f"{'duplicate':<9} {'seed':>5}  trạng thái")
    for index, (spec, key) in enumerate(zip(specs, keys), start=1):
        batch = f"{spec.batch_size}x{spec.accumulation}" if spec.accumulation > 1 else str(spec.batch_size)
        status = f"đã xong ({done[key]})" if key in done else "chờ chạy"
        print(f"{index:>3}  {spec.config_id:<6} {spec.model:<20} {batch:>9} {spec.precision:<5} {spec.workers:>2} "
              f"{spec.resolution:>4} {'on' if spec.clahe else 'off':<5} {spec.dup_policy:<9} {spec.seed:>5}  {status}")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    g = p.add_argument_group("dữ liệu & đầu ra")
    g.add_argument("--task", default="bagls_3class")
    g.add_argument("--prepared-dir", default=str(DEFAULT_PREPARED_DIR))
    g.add_argument("--out", default=None, help="thư mục kết quả (mặc định outputs/<task>)")
    g.add_argument("--source-root", action="append", default=[], metavar="TÊN=ĐƯỜNG_DẪN",
                   help="dữ liệu nằm ở chỗ khác trên máy này (vd. máy Windows của nhóm TATO)")
    g = p.add_argument_group("ma trận thí nghiệm")
    g.add_argument("--matrix", default=str(CONFIG_DIR / "models_benchmark.json"))
    g.add_argument("--configs", nargs="+", help="chỉ chạy các config_id này")
    g.add_argument("--models", nargs="+", help="chỉ chạy các model này (tên trong --list-models)")
    g.add_argument("--clahe", nargs="+", choices=["off", "on"], default=["off"],
                   help="mặc định chỉ 'off' (CLAHE không giúp ích trên BAGLS-VF); thêm 'on' nếu muốn so sánh lại")
    g.add_argument("--dup-policy", nargs="+", choices=list(DUP_POLICIES), default=["dedup", "keep"])
    g.add_argument("--seeds", nargs="+", type=int, default=[42])
    g.add_argument("--batch-size", type=int, help="ghi đè batch size cho mọi cấu hình")
    g.add_argument("--accumulation", type=int, help="ghi đè số bước tích lũy gradient")
    g.add_argument("--precision", choices=["FP32", "AMP", "fp32", "amp"], help="ghi đè precision")
    g.add_argument("--workers", type=int, help="ghi đè số DataLoader worker")
    g.add_argument("--resolution", type=int, help="ghi đè kích thước ảnh vào")
    g.add_argument("--list-models", action="store_true", help="liệt kê model có sẵn rồi thoát")
    g = p.add_argument_group("huấn luyện")
    g.add_argument("--epochs", type=int, default=40)
    g.add_argument("--patience", type=int, default=8, help="early stopping: số epoch không cải thiện")
    g.add_argument("--min-delta", type=float, default=0.001)
    g.add_argument("--monitor", choices=MONITOR_CHOICES, default="val_macro_f1")
    g.add_argument("--lr", type=float, default=1e-4)
    g.add_argument("--head-lr-mult", type=float, default=10.0)
    g.add_argument("--weight-decay", type=float, default=0.01)
    g.add_argument("--warmup-epochs", type=float, default=1.0)
    g.add_argument("--label-smoothing", type=float, default=0.0)
    g.add_argument("--balance", choices=["weights", "sampler", "none"], default="weights")
    g.add_argument("--grad-clip", type=float, default=5.0)
    g.add_argument("--eval-batch-size", type=int)
    g.add_argument("--no-augment", action="store_true")
    g.add_argument("--no-pretrained", action="store_true", help="không dùng trọng số ImageNet (không khuyến khích)")
    g.add_argument("--deterministic", action="store_true")
    g.add_argument("--amp-fallback-fp32", action="store_true", help="cho chạy cấu hình AMP bằng FP32 khi máy không hỗ trợ AMP")
    g.add_argument("--device", default="auto", help="auto | cuda | cuda:0 | mps | cpu")
    g = p.add_argument_group("TATO: chất lượng mục tiêu, trạng thái nhiệt, telemetry")
    g.add_argument("--target-metric", choices=MONITOR_CHOICES, default="val_macro_f1")
    g.add_argument("--target-quality", type=float, default=0.85, help="Q*, vd 0.85 macro-F1")
    g.add_argument("--target-consecutive", type=int, default=2)
    g.add_argument("--stop-on-target", action="store_true", help="dừng ngay khi đạt Q* (giao thức TATO)")
    g.add_argument("--thermal-state", choices=["unknown", "cold", "warm", "pre-stressed"], default="unknown")
    g.add_argument("--precondition", choices=list(STATES), default="none")
    g.add_argument("--precondition-seconds", type=float, default=0.0)
    g.add_argument("--telemetry", action="store_true",
                   help="(tùy chọn) tự đo nhiệt/công suất 1 Hz. Mặc định TẮT — việc đo nhiệt do người khác làm")
    g.add_argument("--pre-run-seconds", type=float, default=0.0, help="(chỉ khi --telemetry) đo 60 s máy đứng yên trước khi train")
    g.add_argument("--telemetry-interval", type=float, default=1.0)
    g.add_argument("--ambient-temp", type=float, help="nhiệt độ phòng (°C) đo bằng nhiệt kế")
    g.add_argument("--nvml-index", type=int)
    g.add_argument("--require-gpu-telemetry", action="store_true", help="dừng run nếu không đọc được NVML")
    g.add_argument("--max-gpu-temp", type=optional_float, default=None, help="(chỉ khi --telemetry) dừng run nếu GPU >= °C")
    g.add_argument("--max-cpu-temp", type=optional_float, default=None)
    g.add_argument("--max-minutes", type=optional_float, default=None)
    g.add_argument("--max-energy-wh", type=optional_float, default=None)
    g = p.add_argument_group("điều phối")
    g.add_argument("--session-id")
    g.add_argument("--machine-id", help="tên máy cho TATO (mặc định: mã băm ẩn danh của hostname)")
    g.add_argument("--tag", default="", help="nhãn để cố ý lặp lại cùng cấu hình (vd. rep2)")
    g.add_argument("--cooldown-seconds", type=float, default=0.0, help="nghỉ giữa 2 run")
    g.add_argument("--shuffle-order", action="store_true")
    g.add_argument("--plan", action="store_true", help="chỉ in danh sách run")
    g.add_argument("--force", action="store_true", help="chạy lại cả run đã hoàn thành")
    g.add_argument("--no-finalize", action="store_true", help="không tự chọn model/đánh giá test khi xong")
    g.add_argument("--no-test", action="store_true", help="finalize nhưng chưa tính test")
    return p


def main(argv=None) -> None:
    setup_console()
    args = build_parser().parse_args(argv)
    if args.list_models:
        print(f"{'model':<20} {'tên':<18} {'tham số (M)':>11} {'GFLOPs*':>8}  ghi chú")
        for row in model_table():
            print(f"{row['model']:<20} {row['display']:<18} {row['params_m']:>11} {row['gflops_torchvision']:>8}  {row['note']}")
        print("* theo torchvision, ở độ phân giải gốc của model (EfficientNet-B2: 288, V2-S: 384). GMACs thực tế ở "
              "--resolution của bạn được ghi vào cột gmacs / config_flops_g của mỗi run.")
        print("Model khác: timm:<tên> (cần pip install timm)")
        return
    if min(args.epochs, args.patience, args.target_consecutive) < 1 or args.lr <= 0 or args.pre_run_seconds < 0:
        raise ValueError("epochs/patience/target-consecutive phải >= 1, lr > 0, pre-run-seconds >= 0")
    if args.deterministic:
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    prepared_dir = resolve_cli_path(args.prepared_dir) / args.task
    out_dir = resolve_cli_path(args.out) if args.out else DEFAULT_OUTPUT_DIR / args.task
    configs = select_configs(load_matrix(resolve_cli_path(args.matrix)), args)
    specs = expand_jobs(configs, [v == "on" for v in args.clahe], args.dup_policy, args.seeds, args.shuffle_order)
    settings = settings_from_args(args)
    if settings.max_energy_wh and not settings.require_gpu_telemetry:
        warn("--max-energy-wh chỉ có tác dụng khi đọc được công suất GPU (NVML).")

    frame, task, manifest_sha = load_prepared(prepared_dir)
    source_roots = parse_key_values(args.source_root)
    frame = attach_paths_and_targets(frame, task["classes"], source_roots)
    machine_id = args.machine_id or default_machine_id()
    session_id = args.session_id or f"S{timestamp()}"

    def context(order: int, device=None) -> RunContext:
        return RunContext(task=task, frame=frame, manifest_sha256=manifest_sha, runs_dir=out_dir / "runs",
                          device=device, session_id=session_id, machine_id=machine_id, tag=args.tag, run_order=order)

    keys = [compute_run_key(spec, settings, context(i)) for i, spec in enumerate(specs)]
    done = {s["run_key"]: s["run_id"] for s in load_summaries(out_dir) if s.get("status") == "completed"}
    if args.plan:
        _print_plan(specs, keys, done, out_dir)
        return

    verify_files(frame["abs_path"])
    device = pick_device(args.device)
    log(f"Thiết bị: {device} | task: {args.task} | {len(specs)} run | kết quả: {out_dir}")
    interrupted = False
    with BenchmarkLock(out_dir):
        write_json(out_dir / "sessions" / f"{session_id}.json", {
            "session_id": session_id, "started_at": iso_now(), "device": str(device), "machine_id": machine_id,
            "args": vars(args), "settings": settings.__dict__,
            "jobs": [{"run_key": k, **spec.__dict__} for spec, k in zip(specs, keys)],
        })
        ran = 0
        try:
            for order, (spec, key) in enumerate(zip(specs, keys)):
                if key in done and not args.force:
                    log(f"Bỏ qua (đã xong): {spec.tag()} -> {done[key]}")
                    continue
                if ran and args.cooldown_seconds > 0:
                    log(f"Nghỉ {args.cooldown_seconds:.0f}s giữa 2 run...")
                    time.sleep(args.cooldown_seconds)
                log(f"===== Run {order + 1}/{len(specs)} =====")
                train_run(spec, settings, context(order, device))
                ran += 1
                write_reports(out_dir, task)  # cập nhật benchmark_results.csv sau mỗi run
        except KeyboardInterrupt:
            interrupted = True
            warn("Đã dừng theo yêu cầu (Ctrl+C). Chạy lại đúng lệnh này để tiếp tục các run còn lại.")
        finally:
            write_reports(out_dir, task)
    if interrupted or args.no_finalize:
        return
    finalize(out_dir, prepared_dir, args.device, not args.no_test, source_roots)
    log(f"Xong. Xem bảng so sánh: {out_dir / 'comparison.md'} | toàn bộ lần train: {out_dir / 'benchmark_results.csv'}")


if __name__ == "__main__":
    main()
