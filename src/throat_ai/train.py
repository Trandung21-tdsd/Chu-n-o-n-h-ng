"""Huấn luyện 1 cấu hình: fine-tune toàn bộ mạng, early stopping theo validation, telemetry cho TATO.

Mỗi run ghi vào <out>/runs/<run_id>/:
    config.json, summary.json (cập nhật sau mỗi epoch), history.csv (từng epoch), telemetry.csv (1 Hz),
    static_at_start.json (đặc trưng t=0 cho TATO), best.pt (checkpoint tốt nhất theo VALIDATION),
    val_predictions.csv / val_per_class.csv / val_confusion.csv, error.txt (nếu lỗi).
Tập test KHÔNG được dùng ở đây — chỉ đánh giá sau khi đã chốt model (evaluate.py).
"""
from __future__ import annotations

import gc
import math
import time
import traceback
import warnings
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import WeightedRandomSampler

from .data import ImageDataset, close_loader, make_loader, partition
from .inference import autocast_context, predict_loader, save_checkpoint, save_prediction_files
from .metrics import compute_metrics, patient_metrics
from .models import (
    FIXED_RESOLUTION,
    amp_available,
    build_model,
    count_parameters,
    display_name,
    empty_cache,
    estimate_gmacs,
    resolve_model_name,
    synchronize,
)
from .preprocessing import PreprocessConfig, build_transform
from .stopping import EarlyStopping, TargetTracker
from .telemetry import (
    TelemetryLogger,
    hardware_context,
    integrate_energy_wh,
    pre_run_features,
    run_statistics,
    window_stats,
)
from .thermal import precondition
from .utils import iso_now, is_finite, log, set_seed, stable_hash, timestamp, write_csv, write_json

SCHEMA_VERSION = "throat-ai-run-2.0"
MONITOR_CHOICES = ("val_macro_f1", "val_balanced_accuracy", "val_macro_auroc", "val_patient_macro_f1")
STATIC_PREFIXES = ("config_", "hardware_", "dataset_", "pre_run_", "task_")

# Với AMP, bước optimizer đầu có thể bị GradScaler bỏ qua (gradient inf) -> PyTorch cảnh báo vô hại này.
warnings.filterwarnings("ignore", message=r"Detected call of `lr_scheduler\.step\(\)` before `optimizer\.step\(\)`")


@dataclass
class RunSpec:
    """Một cấu hình huấn luyện (1 dòng trong ma trận benchmark × CLAHE × chính sách duplicate × seed)."""

    config_id: str
    model: str
    batch_size: int = 32
    accumulation: int = 1
    precision: str = "FP32"
    workers: int = 4
    resolution: int = 224
    clahe: bool = False
    dup_policy: str = "dedup"
    seed: int = 42

    def tag(self) -> str:
        model = self.model.replace(":", "-").replace("/", "-")
        return f"{self.config_id}_{model}_clahe{int(self.clahe)}_{self.dup_policy}_s{self.seed}"


@dataclass
class TrainSettings:
    """Siêu tham số dùng chung cho mọi run trong một lần benchmark."""

    epochs: int = 40
    patience: int = 8
    min_delta: float = 0.001
    monitor: str = "val_macro_f1"
    lr: float = 1e-4
    head_lr_mult: float = 10.0
    weight_decay: float = 0.01
    warmup_epochs: float = 1.0
    label_smoothing: float = 0.0
    balance: str = "weights"  # weights | sampler | none
    grad_clip: float = 5.0
    augment: bool = True
    pretrained: bool = True
    deterministic: bool = False
    amp_fallback_fp32: bool = False
    # Chất lượng mục tiêu Q* (TATO): đạt khi metric >= Q* trong `target_consecutive` lần validation liên tiếp.
    target_metric: str = "val_macro_f1"
    target_quality: float = 0.85
    target_consecutive: int = 2
    stop_on_target: bool = False
    # Trạng thái nhiệt + telemetry.
    thermal_state: str = "unknown"
    precondition: str = "none"
    precondition_seconds: float = 0.0
    telemetry: bool = False  # đo nhiệt/công suất do người khác đảm nhận -> mặc định TẮT
    pre_run_seconds: float = 0.0
    telemetry_interval: float = 1.0
    ambient_temp_c: float | None = None
    nvml_index: int | None = None
    require_gpu_telemetry: bool = False
    # Giới hạn an toàn (vượt => dừng run, abort_reason được ghi lại).
    max_gpu_temp_c: float | None = None
    max_cpu_temp_c: float | None = None
    max_minutes: float | None = None
    max_energy_wh: float | None = None
    eval_batch_size: int | None = None

    def key_dict(self) -> dict:
        """Các trường ảnh hưởng tới kết quả huấn luyện (dùng làm khóa để chạy tiếp/bỏ qua run đã xong)."""
        ignore = {"telemetry", "pre_run_seconds", "telemetry_interval", "nvml_index", "ambient_temp_c", "require_gpu_telemetry",
                  "eval_batch_size", "max_gpu_temp_c", "max_cpu_temp_c", "max_minutes", "max_energy_wh",
                  "precondition_seconds"}
        return {k: v for k, v in asdict(self).items() if k not in ignore}


@dataclass
class RunContext:
    task: dict
    frame: pd.DataFrame  # manifest + cột abs_path, y
    manifest_sha256: str
    runs_dir: Path
    device: torch.device
    session_id: str
    machine_id: str
    tag: str = ""
    run_order: int = 0


class AbortRun(RuntimeError):
    """Dừng run có chủ đích (vi phạm an toàn nhiệt, hết ngân sách thời gian/năng lượng, loss NaN...)."""


class _NoScaler:
    def scale(self, loss):
        return loss

    def unscale_(self, optimizer):
        pass

    def step(self, optimizer):
        optimizer.step()

    def update(self):
        pass


def compute_run_key(spec: RunSpec, settings: TrainSettings, ctx: RunContext) -> str:
    return stable_hash({"task": ctx.task.get("task"), "manifest": ctx.manifest_sha256, "spec": asdict(spec),
                        "settings": settings.key_dict(), "tag": ctx.tag}, length=16)


def _scaler(device: torch.device, enabled: bool):
    if not enabled:
        return _NoScaler()
    try:
        return torch.amp.GradScaler(device.type)
    except Exception:
        return torch.cuda.amp.GradScaler() if device.type == "cuda" else _NoScaler()


def _warmup_cosine(warmup_steps: int, total_steps: int, floor: float = 0.01):
    def factor(step: int) -> float:
        if warmup_steps > 0 and step < warmup_steps:
            return (step + 1) / warmup_steps
        progress = min(1.0, (step - warmup_steps) / max(1, total_steps - warmup_steps))
        return floor + (1.0 - floor) * 0.5 * (1.0 + math.cos(math.pi * progress))

    return factor


def _is_out_of_memory(exc: BaseException) -> bool:
    if isinstance(exc, getattr(torch.cuda, "OutOfMemoryError", ())):
        return True
    return "out of memory" in str(exc).lower()


def _peak_memory_gb(device: torch.device) -> float | None:
    try:
        if device.type == "cuda":
            return torch.cuda.max_memory_allocated(device) / 2**30
        if device.type == "mps":
            return torch.mps.driver_allocated_memory() / 2**30
    except Exception:
        pass
    return None


class _SafetyGuard:
    """Kiểm tra định kỳ trong lúc train: thời gian, nhiệt độ GPU/CPU, năng lượng, mất cảm biến."""

    def __init__(self, settings: TrainSettings, telemetry: TelemetryLogger, start: float) -> None:
        self.settings, self.telemetry, self.start = settings, telemetry, start
        self._last_energy_check = start

    def check(self) -> None:
        s, now = self.settings, self.telemetry.elapsed()
        if s.max_minutes and now - self.start >= s.max_minutes * 60:
            raise AbortRun("time_budget_exceeded")
        latest = self.telemetry.latest()
        last_sample = latest.get("elapsed_s")
        if s.require_gpu_telemetry:
            stale = last_sample is None or now - float(last_sample) > max(5 * s.telemetry_interval, 5.0)
            if stale or latest.get("gpu_temp_c") is None or latest.get("gpu_power_w") is None:
                raise AbortRun("gpu_telemetry_lost")
        gpu_temp, cpu_temp = latest.get("gpu_temp_c"), latest.get("cpu_temp_c")
        if s.max_gpu_temp_c and is_finite(gpu_temp) and gpu_temp >= s.max_gpu_temp_c:
            raise AbortRun(f"thermal_constraint_violation:gpu_temp={gpu_temp:.0f}C")
        if s.max_cpu_temp_c and is_finite(cpu_temp) and cpu_temp >= s.max_cpu_temp_c:
            raise AbortRun(f"thermal_constraint_violation:cpu_temp={cpu_temp:.0f}C")
        if s.max_energy_wh and now - self._last_energy_check >= 10:
            self._last_energy_check = now
            end = float(last_sample) if last_sample is not None else now
            energy = integrate_energy_wh(self.telemetry.rows(), self.start, end)
            if energy is not None and energy >= s.max_energy_wh:
                raise AbortRun("energy_budget_exceeded")


def _base_summary(spec: RunSpec, settings: TrainSettings, ctx: RunContext, run_id: str, run_key: str,
                  run_dir: Path, model_key: str) -> dict:
    task = ctx.task
    return {
        "schema_version": SCHEMA_VERSION,
        "run_id": run_id, "run_key": run_key, "run_order": ctx.run_order, "session_id": ctx.session_id,
        "machine_id": ctx.machine_id, "tag": ctx.tag,
        "task": task.get("task"), "task_anatomy": task.get("anatomy"), "task_modality": task.get("modality"),
        "status": "initializing", "stop_reason": None, "abort_reason": None,
        "started_at": iso_now(), "finished_at": None,
        # Cấu hình (tên ngắn) — giống cột profiling_index của TATO.
        "config_id": spec.config_id, "model": model_key, "backbone": display_name(model_key),
        "batch_size": spec.batch_size, "accumulation": spec.accumulation,
        "effective_batch_size": spec.batch_size * spec.accumulation,
        "precision": spec.precision.upper(), "precision_effective": None, "workers": spec.workers,
        "resolution": spec.resolution, "clahe": bool(spec.clahe), "dup_policy": spec.dup_policy,
        "leaky": spec.dup_policy == "naive", "seed": spec.seed,
        "lr": settings.lr, "head_lr_mult": settings.head_lr_mult, "weight_decay": settings.weight_decay,
        "warmup_epochs": settings.warmup_epochs, "label_smoothing": settings.label_smoothing,
        "balance": settings.balance, "augment": settings.augment, "pretrained": settings.pretrained,
        "epochs_max": settings.epochs, "patience": settings.patience, "min_delta": settings.min_delta,
        "monitor": settings.monitor, "monitor_effective": settings.monitor,
        "thermal_start_state": settings.thermal_state, "precondition": settings.precondition,
        # Đặc trưng tĩnh dạng TATO (biết trước khi train).
        "config_batch_size": spec.batch_size, "config_workers": spec.workers, "config_resolution": spec.resolution,
        "config_accumulation": spec.accumulation, "config_clahe": int(spec.clahe),
        "config_dedup_train": int(spec.dup_policy == "dedup"), "config_precision_amp": None,
        "task_quality_metric": settings.target_metric, "task_quality_value": settings.target_quality,
        "task_target_consecutive": settings.target_consecutive, "task_stop_on_target": settings.stop_on_target,
        "task_safe_gpu_temp_c": settings.max_gpu_temp_c, "task_safe_cpu_temp_c": settings.max_cpu_temp_c,
        "task_max_minutes": settings.max_minutes, "task_max_energy_wh": settings.max_energy_wh,
        "dataset_id": f"{task.get('task')}@{ctx.manifest_sha256[:12]}", "manifest_sha256": ctx.manifest_sha256,
        # Kết quả TATO (mặc định: chưa đạt; không đạt thì thời gian/năng lượng = trống, KHÔNG gán tổng run).
        "target_reached": 0, "target_epoch": None, "target_time_to_target_s": None,
        "target_energy_to_target_wh": None, "target_energy_scope": "GPU (NVML), không gồm CPU/hệ thống",
        "run_dir": str(run_dir),
    }


def train_run(spec: RunSpec, settings: TrainSettings, ctx: RunContext) -> dict:
    """Chạy 1 cấu hình từ đầu đến cuối. Không ném lỗi ra ngoài (trừ Ctrl+C) — lỗi được ghi vào summary."""
    model_key = resolve_model_name(spec.model)
    run_key = compute_run_key(spec, settings, ctx)
    run_id = f"{timestamp()}_{spec.tag()}_{run_key[:6]}"
    run_dir = Path(ctx.runs_dir) / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    device, classes = ctx.device, list(ctx.task["classes"])
    n_classes = len(classes)
    summary = _base_summary(spec, settings, ctx, run_id, run_key, run_dir, model_key)
    write_json(run_dir / "config.json", {"spec": asdict(spec), "settings": asdict(settings), "run_key": run_key,
                                         "task": ctx.task.get("task"), "manifest_sha256": ctx.manifest_sha256})
    write_json(run_dir / "summary.json", summary)

    telemetry = model = optimizer = train_loader = val_loader = None
    t0 = best_probs = val_df = None
    history: list[dict] = []
    try:
        # ---------------------------------------------------------------- dữ liệu theo chính sách duplicate
        if model_key in FIXED_RESOLUTION and spec.resolution != FIXED_RESOLUTION[model_key]:
            raise ValueError(f"{model_key} chỉ nhận resolution {FIXED_RESOLUTION[model_key]}")
        train_df = partition(ctx.frame, "train", spec.dup_policy)
        val_df = partition(ctx.frame, "val", spec.dup_policy)
        counts = np.bincount(train_df["y"], minlength=n_classes)
        if (counts == 0).any():
            raise ValueError(f"Tập train thiếu lớp {[c for c, n in zip(classes, counts) if n == 0]}")
        if val_df.empty:
            raise ValueError("Tập validation rỗng")
        summary.update(
            n_train_images=len(train_df), n_val_images=len(val_df),
            n_train_patients=int(train_df.loc[train_df["patient_key"] != "", "patient_key"].nunique()),
            n_val_patients=int(val_df.loc[val_df["patient_key"] != "", "patient_key"].nunique()),
            train_class_counts=";".join(f"{c}:{n}" for c, n in zip(classes, counts)),
            dataset_num_train_images=len(train_df), dataset_num_val_images=len(val_df), dataset_num_classes=n_classes,
        )

        # ---------------------------------------------------------------- độ chính xác số học
        use_amp = False
        if spec.precision.upper() == "AMP":
            if amp_available(device):
                use_amp = True
            elif not settings.amp_fallback_fp32:
                summary.update(status="skipped", stop_reason="skipped",
                               abort_reason=f"AMP không hỗ trợ trên {device.type}; không tự đổi sang FP32 "
                                            "(thêm --amp-fallback-fp32 nếu chấp nhận)")
                log(f"⏭  Bỏ qua {run_id}: {summary['abort_reason']}")
                return summary
        summary.update(precision_effective="AMP" if use_amp else "FP32", config_precision_amp=int(use_amp))

        # ---------------------------------------------------------------- telemetry + trạng thái nhiệt t=0
        telemetry = TelemetryLogger(device, settings.telemetry_interval, settings.nvml_index, enabled=settings.telemetry)
        summary.update(hardware_context(device))
        summary.update(telemetry_gpu_source=telemetry.gpu_source, telemetry_init_error=telemetry.init_error)
        if settings.require_gpu_telemetry and not telemetry.has_gpu_sensors:
            raise AbortRun(f"required_gpu_telemetry_unavailable ({telemetry.init_error})")
        telemetry.start()
        summary.update(precondition(settings.precondition, device, telemetry, settings.precondition_seconds))
        if settings.pre_run_seconds > 0:
            telemetry.set_progress(phase="pre_run")
            log(f"  Đo trạng thái nhiệt/hiệu năng trước khi train ({settings.pre_run_seconds:.0f}s, máy đứng yên)...")
            time.sleep(settings.pre_run_seconds)
        telemetry.sample()
        if settings.telemetry:
            summary.update(pre_run_features(telemetry.rows(), window_s=60.0))
            summary["pre_run_ambient_temp_c"] = settings.ambient_temp_c

        # ---------------------------------------------------------------- model + optimizer
        set_seed(spec.seed)
        torch.backends.cudnn.benchmark = not settings.deterministic
        if settings.deterministic:
            torch.use_deterministic_algorithms(True, warn_only=True)
        setup_start = time.monotonic()
        model, head_params = build_model(model_key, n_classes, pretrained=settings.pretrained)
        params_m, trainable_m = count_parameters(model)
        gmacs = estimate_gmacs(model, spec.resolution)
        summary.update(params_m=params_m, trainable_params_m=trainable_m, gmacs=gmacs,
                       config_params_m=params_m, config_flops_g=gmacs)
        write_json(run_dir / "static_at_start.json",
                   {k: v for k, v in summary.items() if k.startswith(STATIC_PREFIXES)})
        model.to(device)
        head_ids = {id(p) for p in head_params}
        backbone_params = [p for p in model.parameters() if p.requires_grad and id(p) not in head_ids]
        optimizer = torch.optim.AdamW(
            [{"params": backbone_params, "lr": settings.lr},
             {"params": head_params, "lr": settings.lr * settings.head_lr_mult}],
            weight_decay=settings.weight_decay,
        )

        # ---------------------------------------------------------------- DataLoader
        preprocess = PreprocessConfig(resolution=spec.resolution, clahe=spec.clahe,
                                      geometry=ctx.task.get("preprocess", {}).get("geometry", "letterbox"))
        augment = ctx.task.get("augment") if settings.augment else None
        train_ds = ImageDataset(train_df["abs_path"], train_df["y"], preprocess, build_transform(True, augment))
        val_ds = ImageDataset(val_df["abs_path"], val_df["y"], preprocess, build_transform(False))
        pin = device.type == "cuda"
        sampler, class_weights = None, None
        if settings.balance == "weights":
            class_weights = torch.tensor(len(train_df) / (n_classes * counts), dtype=torch.float32, device=device)
        elif settings.balance == "sampler":
            weights = torch.as_tensor((1.0 / counts)[train_df["y"].to_numpy()], dtype=torch.double)
            sampler = WeightedRandomSampler(weights, num_samples=len(train_df), replacement=True,
                                            generator=torch.Generator().manual_seed(spec.seed))
        train_loader = make_loader(train_ds, spec.batch_size, shuffle=True, workers=spec.workers, seed=spec.seed,
                                   pin_memory=pin, drop_last=len(train_ds) > spec.batch_size, sampler=sampler)
        val_loader = make_loader(val_ds, settings.eval_batch_size or spec.batch_size, shuffle=False,
                                 workers=spec.workers, seed=spec.seed, pin_memory=pin)
        criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=settings.label_smoothing)
        steps_per_epoch = max(1, math.ceil(len(train_loader) / spec.accumulation))
        scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer, _warmup_cosine(int(round(settings.warmup_epochs * steps_per_epoch)), steps_per_epoch * settings.epochs))
        scaler = _scaler(device, use_amp)
        stopper = EarlyStopping(settings.patience, settings.min_delta)
        tracker = TargetTracker(settings.target_quality, settings.target_consecutive)
        best_metrics: dict = {}
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        synchronize(device)
        summary["setup_seconds"] = time.monotonic() - setup_start

        # ---------------------------------------------------------------- huấn luyện
        t0 = telemetry.sample()["elapsed_s"]
        summary.update(status="running", training_start_elapsed_s=t0, training_start_unix=time.time(),
                       training_start_iso=iso_now())
        write_json(run_dir / "summary.json", summary)
        guard = _SafetyGuard(settings, telemetry, t0)
        log(f"▶ {run_id} | {display_name(model_key)} | CLAHE={'on' if spec.clahe else 'off'} | duplicate={spec.dup_policy} "
            f"| train={len(train_df)} val={len(val_df)} | {device} {summary['precision_effective']}")
        warned_monitor = False
        for epoch in range(1, settings.epochs + 1):
            epoch_start = time.monotonic()
            telemetry.set_progress(phase="train", epoch=epoch)
            model.train()
            optimizer.zero_grad(set_to_none=True)
            loss_total = torch.zeros((), device=device)
            correct_total = torch.zeros((), device=device, dtype=torch.long)
            seen, n_batches = 0, len(train_loader)
            for step, (images, targets) in enumerate(train_loader, start=1):
                if step % 10 == 1:
                    guard.check()
                images = images.to(device, non_blocking=pin)
                targets = targets.to(device, non_blocking=pin)
                with autocast_context(device, use_amp):
                    logits = model(images)
                loss = criterion(logits.float(), targets)
                scaler.scale(loss / spec.accumulation).backward()
                if step % spec.accumulation == 0 or step == n_batches:
                    if settings.grad_clip:
                        scaler.unscale_(optimizer)
                        nn.utils.clip_grad_norm_(model.parameters(), settings.grad_clip)
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad(set_to_none=True)
                    scheduler.step()
                batch = targets.size(0)
                loss_total += loss.detach() * batch
                correct_total += (logits.detach().argmax(1) == targets).sum()
                seen += batch
                telemetry.add_train_images(batch)
                if step % 50 == 0 and not torch.isfinite(loss_total):
                    raise AbortRun("non_finite_loss")
            synchronize(device)
            train_seconds = time.monotonic() - epoch_start
            if seen == 0:
                raise ValueError("Không có batch train nào (batch_size lớn hơn số ảnh?)")
            train_loss = float(loss_total.item()) / seen
            if not math.isfinite(train_loss):
                raise AbortRun("non_finite_loss")

            # ------------------------------------------------------------ validation (mỗi epoch)
            telemetry.set_progress(phase="val")
            val_start = time.monotonic()
            probs = predict_loader(model, val_loader, device, use_amp, guard.check)
            val_seconds = time.monotonic() - val_start
            telemetry.set_progress(phase="epoch_end")
            metrics = {f"val_{k}": v for k, v in compute_metrics(val_df["y"], probs).items()}
            # val_loss cùng công thức với train loss (cross-entropy có trọng số lớp) để vẽ chung một trục
            y_val = val_df["y"].to_numpy()
            nll = -np.log(np.clip(probs[np.arange(len(y_val)), y_val], 1e-12, 1.0))
            sample_weight = class_weights.cpu().numpy()[y_val] if class_weights is not None else np.ones(len(y_val))
            metrics["val_loss"] = float((nll * sample_weight).sum() / sample_weight.sum())
            metrics.update({f"val_patient_{k}": v for k, v in patient_metrics(val_df, probs).items()})
            monitor_value = metrics.get(settings.monitor)
            if not is_finite(monitor_value):
                if not warned_monitor:
                    log(f"  {settings.monitor} không tính được -> early stopping dùng val_macro_f1")
                    summary["monitor_effective"], warned_monitor = "val_macro_f1", True
                monitor_value = metrics["val_macro_f1"]
            sample = telemetry.sample()
            now = sample["elapsed_s"]
            rows = telemetry.rows()
            energy = integrate_energy_wh(rows, t0, now)
            improved = stopper.step(monitor_value, epoch)
            target_value = metrics.get(settings.target_metric)
            reached = tracker.step(target_value if is_finite(target_value) else metrics["val_macro_f1"], epoch)
            if improved:
                best_probs, best_metrics = probs, metrics
                save_checkpoint(run_dir / "best.pt", model, {
                    "model_name": model_key, "num_classes": n_classes, "classes": classes,
                    "display_names": ctx.task.get("display_names", {}), "preprocess": preprocess.to_dict(),
                    "task": ctx.task.get("task"), "anatomy": ctx.task.get("anatomy"),
                    "modality": ctx.task.get("modality"), "run_id": run_id, "epoch": epoch,
                    "monitor": summary["monitor_effective"], "val_score": float(monitor_value),
                    "manifest_sha256": ctx.manifest_sha256, "created_at": iso_now(),
                })
            if reached:
                summary.update(target_reached=1, target_epoch=epoch, target_time_to_target_s=now - t0,
                               target_reached_unix=time.time(), target_reached_iso=iso_now(),
                               target_energy_to_target_wh=energy)
            entry = {
                "epoch": epoch, "timestamp_unix": time.time(), "elapsed_s": now - t0,
                "epoch_seconds": time.monotonic() - epoch_start,
                "train_seconds": train_seconds, "val_seconds": val_seconds, "train_loss": train_loss,
                "train_accuracy": float(correct_total.item()) / seen, "train_images": seen,
                "train_images_per_s": seen / train_seconds if train_seconds > 0 else None,
                "lr_backbone": optimizer.param_groups[0]["lr"], "lr_head": optimizer.param_groups[1]["lr"],
                **metrics, "monitor_value": monitor_value, "improved": improved, "target_streak": tracker.streak,
                "gpu_energy_wh_cumulative": energy, **window_stats(rows, now, 60.0),
                "gpu_mem_peak_gb": _peak_memory_gb(device),
            }
            history.append(entry)
            write_csv(pd.DataFrame(history), run_dir / "history.csv")
            summary.update(
                epochs_run=epoch, best_epoch=stopper.best_epoch, best_val_score=stopper.best,
                best_macro_f1=max(h["val_macro_f1"] for h in history), final_macro_f1=metrics["val_macro_f1"],
                mean_epoch_seconds=float(np.mean([h["epoch_seconds"] for h in history])),
                train_images_per_s_mean=float(np.mean([h["train_images_per_s"] or 0 for h in history])),
                gpu_mem_peak_gb=entry["gpu_mem_peak_gb"], **best_metrics,
            )
            write_json(run_dir / "summary.json", summary)
            patient = f" | F1 bệnh nhân {metrics['val_patient_macro_f1']:.3f}" if "val_patient_macro_f1" in metrics else ""
            log(f"  epoch {epoch:>3}/{settings.epochs} | loss {train_loss:.4f} | val macro-F1 {metrics['val_macro_f1']:.4f}"
                f" | bal.acc {metrics['val_balanced_accuracy']:.4f}{patient} | {entry['epoch_seconds']:.0f}s"
                + (" ★" if improved else ""))
            if reached:
                log(f"  ◎ Đạt Q*: {settings.target_metric} >= {settings.target_quality} trong "
                    f"{settings.target_consecutive} lần validation liên tiếp (epoch {epoch}, {now - t0:.0f}s)")
            if reached and settings.stop_on_target:
                summary["stop_reason"] = "target_reached"
                break
            if stopper.should_stop:
                summary["stop_reason"] = "early_stopping"
                break
        else:
            summary["stop_reason"] = "max_epochs"
        summary["status"] = "completed"
    except AbortRun as exc:
        summary.update(status="aborted", stop_reason="aborted", abort_reason=str(exc))
        log(f"  ■ Dừng run: {exc}")
    except KeyboardInterrupt:
        summary.update(status="interrupted", stop_reason="interrupted", abort_reason="keyboard_interrupt")
        raise
    except Exception as exc:  # lỗi 1 run không làm dừng cả benchmark
        reason = "out_of_memory (giảm batch_size hoặc dùng AMP)" if _is_out_of_memory(exc) else f"{type(exc).__name__}: {exc}"
        summary.update(status="failed", stop_reason="failed", abort_reason=reason)
        (run_dir / "error.txt").write_text(traceback.format_exc(), encoding="utf-8")
        log(f"  ✖ Lỗi: {reason}")
    finally:
        if telemetry is not None:
            try:
                telemetry.set_progress(phase="end")
                end = telemetry.sample()["elapsed_s"]
                telemetry.stop()
                if settings.telemetry:
                    telemetry.save(run_dir / "telemetry.csv")
                if t0 is not None:
                    summary.update(run_statistics(telemetry.rows(), t0, end))
            except Exception as exc:
                summary["telemetry_error"] = f"{type(exc).__name__}: {exc}"
        if best_probs is not None and val_df is not None:
            try:
                save_prediction_files(run_dir, "val", val_df, best_probs, classes)
            except Exception as exc:
                summary["prediction_file_error"] = f"{type(exc).__name__}: {exc}"
        if summary["status"] in ("initializing", "running"):
            summary["status"] = "failed"
        summary["finished_at"] = iso_now()
        summary["finished_unix"] = time.time()
        write_json(run_dir / "summary.json", summary)
        # Tắt worker DataLoader + nhả tham chiếu để giải phóng bộ nhớ GPU trước run kế tiếp.
        close_loader(train_loader)
        close_loader(val_loader)
        model = optimizer = scheduler = train_loader = val_loader = train_ds = val_ds = None  # noqa: F841
        criterion = class_weights = scaler = None  # noqa: F841
        gc.collect()
        empty_cache(device)
    if summary["status"] == "completed":
        log(f"✔ Xong {run_id}: best {summary['monitor_effective']}={summary.get('best_val_score', float('nan')):.4f} "
            f"tại epoch {summary.get('best_epoch')} ({summary.get('stop_reason')})")
    return summary
