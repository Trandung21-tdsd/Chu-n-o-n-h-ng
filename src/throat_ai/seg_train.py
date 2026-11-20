"""Huấn luyện 1 cấu hình model PHÂN ĐOẠN: loss = Cross-Entropy + Dice, early stopping theo val mean Dice.

Mỗi run ghi vào <out>/runs/<run_id>/: config.json, summary.json, history.csv (train_loss, val_loss, Dice từng lớp,
timestamp_unix từng epoch), best.pt (tốt nhất theo VALIDATION). Tập test KHÔNG dùng ở đây.
Các cột summary đặt tên giống run phân loại để xuất chung định dạng TATO (chất lượng = val mean Dice).
"""
from __future__ import annotations

import gc
import math
import time
import traceback
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch import nn

from .data import close_loader, make_loader
from .inference import autocast_context, save_checkpoint
from .models import amp_available, count_parameters, empty_cache, estimate_gmacs, synchronize
from .seg_data import CLASSES, SegDataset
from .seg_models import build_seg_model, seg_display_name
from .stopping import EarlyStopping, TargetTracker
from .telemetry import hardware_context
from .train import _is_out_of_memory, _peak_memory_gb, _scaler, _warmup_cosine
from .utils import iso_now, log, set_seed, stable_hash, timestamp, write_csv, write_json

SCHEMA_VERSION = "throat-ai-seg-run-1.0"
FOREGROUND = CLASSES[1:]
STATIC_PREFIXES = ("config_", "hardware_", "dataset_", "task_")


@dataclass
class SegSpec:
    config_id: str
    model: str
    batch_size: int = 16
    precision: str = "FP32"
    workers: int = 4
    resolution: int = 256
    clahe: bool = False
    seed: int = 42

    def tag(self) -> str:
        return f"{self.config_id}_{self.model}_clahe{int(self.clahe)}_s{self.seed}"


@dataclass
class SegSettings:
    epochs: int = 40
    patience: int = 6
    min_delta: float = 0.001
    lr: float = 1e-4  # encoder pretrained; decoder/head = lr * head_lr_mult (model không pretrained: toàn bộ dùng LR lớn)
    head_lr_mult: float = 10.0
    weight_decay: float = 1e-4
    warmup_epochs: float = 1.0
    dice_weight: float = 1.0
    grad_clip: float = 5.0
    augment: bool = True
    pretrained: bool = True
    target_quality: float = 0.80  # Q* cho TATO: val mean Dice
    target_consecutive: int = 2
    stop_on_target: bool = False
    max_minutes: float | None = None
    eval_batch_size: int | None = None

    def key_dict(self) -> dict:
        return {k: v for k, v in asdict(self).items() if k not in {"max_minutes", "eval_batch_size"}}


def run_key(spec: SegSpec, settings: SegSettings, manifest_sha: str) -> str:
    return stable_hash({"manifest": manifest_sha, "spec": asdict(spec), "settings": settings.key_dict()}, length=16)


# ------------------------------------------------------------------------------ loss + metric
def soft_dice_loss(logits: torch.Tensor, target: torch.Tensor, eps: float = 1.0) -> torch.Tensor:
    probs = logits.float().softmax(1)[:, 1:]
    onehot = F.one_hot(target, logits.shape[1]).permute(0, 3, 1, 2).float()[:, 1:]
    inter = (probs * onehot).sum((0, 2, 3))
    union = probs.sum((0, 2, 3)) + onehot.sum((0, 2, 3))
    return 1 - ((2 * inter + eps) / (union + eps)).mean()


class ConfusionAccumulator:
    """Cộng dồn ma trận nhầm lẫn theo pixel + diện tích thanh môn từng ảnh (thật vs dự đoán)."""

    def __init__(self, n_classes: int = 4) -> None:
        self.n = n_classes
        self.matrix = np.zeros((n_classes, n_classes), dtype=np.int64)
        self.area_true: list[int] = []
        self.area_pred: list[int] = []
        self.image_dice: list[float] = []

    def update(self, pred: torch.Tensor, target: torch.Tensor) -> None:
        pred, target = pred.cpu(), target.cpu()
        self.matrix += torch.bincount((target * self.n + pred).flatten(), minlength=self.n ** 2) \
            .reshape(self.n, self.n).numpy()
        g_true, g_pred = (target == 3).sum((1, 2)), (pred == 3).sum((1, 2))
        self.area_true += g_true.tolist()
        self.area_pred += g_pred.tolist()
        for t, p in zip(target, pred):  # Dice trung bình 3 lớp trên từng ảnh (lớp vắng ở cả 2 => 1)
            scores = []
            for c in range(1, self.n):
                tc, pc = (t == c), (p == c)
                denom = int(tc.sum() + pc.sum())
                scores.append(1.0 if denom == 0 else 2 * int((tc & pc).sum()) / denom)
            self.image_dice.append(float(np.mean(scores)))

    def metrics(self, prefix: str = "val") -> dict:
        m = self.matrix
        tp = np.diag(m).astype(float)
        fp, fn = m.sum(0) - tp, m.sum(1) - tp
        dice = 2 * tp / np.maximum(2 * tp + fp + fn, 1)
        iou = tp / np.maximum(tp + fp + fn, 1)
        out = {f"{prefix}_mean_dice": float(dice[1:].mean()), f"{prefix}_mean_iou": float(iou[1:].mean()),
               f"{prefix}_pixel_accuracy": float(tp.sum() / max(m.sum(), 1)),
               f"{prefix}_image_mean_dice": float(np.mean(self.image_dice)) if self.image_dice else float("nan")}
        for c, name in enumerate(CLASSES[1:], start=1):
            out[f"{prefix}_dice_{name}"] = float(dice[c])
            out[f"{prefix}_iou_{name}"] = float(iou[c])
        a, b = np.asarray(self.area_true, float), np.asarray(self.area_pred, float)
        out[f"{prefix}_glottis_area_corr"] = float(np.corrcoef(a, b)[0, 1]) if len(a) > 2 and a.std() > 0 and b.std() > 0 else float("nan")
        # cùng kết luận "thanh môn đang khép" (không thấy vùng đỏ) giữa mask thật và mask dự đoán
        out[f"{prefix}_closure_agreement"] = float(((a < 1) == (b < 1)).mean()) if len(a) else float("nan")
        return out


@torch.inference_mode()
def evaluate_loader(model, loader, device, use_amp: bool = False, dice_weight: float = 1.0,
                    keep_predictions: bool = False) -> tuple[dict, float, list[np.ndarray]]:
    model.eval()
    acc = ConfusionAccumulator(len(CLASSES))
    loss_sum, seen, preds = 0.0, 0, []
    for images, masks in loader:
        images, masks = images.to(device), masks.to(device)
        with autocast_context(device, use_amp):
            logits = model(images)
        logits = logits.float()
        loss = F.cross_entropy(logits, masks) + dice_weight * soft_dice_loss(logits, masks)
        loss_sum += float(loss) * images.size(0)
        seen += images.size(0)
        pred = logits.argmax(1)
        acc.update(pred, masks)
        if keep_predictions:
            preds += [p.astype(np.uint8) for p in pred.cpu().numpy()]
    return acc.metrics("val"), loss_sum / max(seen, 1), preds


@torch.inference_mode()
def predict_masks(model, image_paths, size: int, clahe: bool, device, batch_size: int = 16) -> list[np.ndarray]:
    """Dự đoán mask (chỉ số lớp, size x size, trên ảnh đã cắt center_square) cho danh sách ảnh."""
    from .seg_data import center_square, load_rgb, to_tensor
    from .preprocessing import apply_clahe

    model.eval()
    out = []
    paths = list(image_paths)
    for start in range(0, len(paths), batch_size):
        batch = []
        for path in paths[start:start + batch_size]:
            image = center_square(np.asarray(load_rgb(path), dtype=np.uint8), size, nearest=False)
            batch.append(to_tensor(apply_clahe(image) if clahe else image))
        logits = model(torch.stack(batch).to(device)).float()
        out += [p.astype(np.uint8) for p in logits.argmax(1).cpu().numpy()]
    return out


# ------------------------------------------------------------------------------ 1 run
def _summary(spec, settings, task, manifest_sha, run_id, key, run_dir, session_id, machine_id, run_order) -> dict:
    return {
        "schema_version": SCHEMA_VERSION, "kind": "segmentation",
        "run_id": run_id, "run_key": key, "run_order": run_order, "session_id": session_id, "machine_id": machine_id,
        "task": task, "status": "initializing", "stop_reason": None, "abort_reason": None,
        "started_at": iso_now(), "finished_at": None,
        "config_id": spec.config_id, "model": spec.model, "backbone": seg_display_name(spec.model),
        "batch_size": spec.batch_size, "accumulation": 1, "effective_batch_size": spec.batch_size,
        "precision": spec.precision.upper(), "precision_effective": None, "workers": spec.workers,
        "resolution": spec.resolution, "clahe": bool(spec.clahe), "dup_policy": "video_split", "leaky": False,
        "seed": spec.seed, **{f"setting_{k}": v for k, v in asdict(settings).items()},
        "monitor": "val_mean_dice", "monitor_effective": "val_mean_dice",
        "thermal_start_state": "unknown",
        "config_batch_size": spec.batch_size, "config_workers": spec.workers, "config_resolution": spec.resolution,
        "config_accumulation": 1, "config_clahe": int(spec.clahe), "config_dedup_train": 1, "config_precision_amp": None,
        "task_quality_metric": "val_mean_dice", "task_quality_value": settings.target_quality,
        "task_target_consecutive": settings.target_consecutive, "task_stop_on_target": settings.stop_on_target,
        "dataset_id": f"{task}@{manifest_sha[:12]}", "manifest_sha256": manifest_sha,
        "target_reached": 0, "target_epoch": None, "target_time_to_target_s": None, "target_energy_to_target_wh": None,
        "run_dir": str(run_dir),
    }


def train_seg_run(spec: SegSpec, settings: SegSettings, frame: pd.DataFrame, task_info: dict, manifest_sha: str,
                  runs_dir: Path, device: torch.device, augment: dict | None, session_id: str = "",
                  machine_id: str = "", run_order: int = 0) -> dict:
    key = run_key(spec, settings, manifest_sha)
    run_id = f"{timestamp()}_{spec.tag()}_{key[:6]}"
    run_dir = Path(runs_dir) / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    task = task_info.get("task", "bagls_seg")
    summary = _summary(spec, settings, task, manifest_sha, run_id, key, run_dir, session_id, machine_id, run_order)
    write_json(run_dir / "config.json", {"spec": asdict(spec), "settings": asdict(settings), "run_key": key,
                                         "manifest_sha256": manifest_sha})
    write_json(run_dir / "summary.json", summary)
    model = optimizer = train_loader = val_loader = None
    history: list[dict] = []
    try:
        train_df, val_df = frame[frame["split"] == "train"], frame[frame["split"] == "val"]
        if train_df.empty or val_df.empty:
            raise ValueError("Tập train hoặc val của manifest phân đoạn rỗng")
        summary.update(n_train_images=len(train_df), n_val_images=len(val_df),
                       n_train_videos=int(train_df["video_id"].nunique()), n_val_videos=int(val_df["video_id"].nunique()),
                       dataset_num_train_images=len(train_df), dataset_num_val_images=len(val_df),
                       dataset_num_classes=len(CLASSES))
        use_amp = spec.precision.upper() == "AMP" and amp_available(device)
        summary.update(precision_effective="AMP" if use_amp else "FP32", config_precision_amp=int(use_amp))
        summary.update(hardware_context(device))

        set_seed(spec.seed)
        torch.backends.cudnn.benchmark = True
        model, encoder_params = build_seg_model(spec.model, len(CLASSES), pretrained=settings.pretrained)
        params_m, _ = count_parameters(model)
        gmacs = estimate_gmacs(model, spec.resolution)
        summary.update(params_m=params_m, gmacs=gmacs, config_params_m=params_m, config_flops_g=gmacs)
        write_json(run_dir / "static_at_start.json", {k: v for k, v in summary.items() if k.startswith(STATIC_PREFIXES)})
        model.to(device)
        encoder_ids = {id(p) for p in encoder_params}
        other = [p for p in model.parameters() if id(p) not in encoder_ids]
        groups = [{"params": other, "lr": settings.lr * settings.head_lr_mult}]
        if encoder_params:
            groups.insert(0, {"params": encoder_params, "lr": settings.lr})
        optimizer = torch.optim.AdamW(groups, weight_decay=settings.weight_decay)

        train_ds = SegDataset(train_df["image_path"], train_df["mask_path"], spec.resolution, spec.clahe,
                              augment if settings.augment else None)
        val_ds = SegDataset(val_df["image_path"], val_df["mask_path"], spec.resolution, spec.clahe, None)
        pin = device.type == "cuda"
        train_loader = make_loader(train_ds, spec.batch_size, shuffle=True, workers=spec.workers, seed=spec.seed,
                                   pin_memory=pin, drop_last=len(train_ds) > spec.batch_size)
        val_loader = make_loader(val_ds, settings.eval_batch_size or spec.batch_size, shuffle=False,
                                 workers=spec.workers, seed=spec.seed, pin_memory=pin)
        steps = max(1, len(train_loader))
        scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer, _warmup_cosine(int(round(settings.warmup_epochs * steps)), steps * settings.epochs))
        scaler = _scaler(device, use_amp)
        stopper = EarlyStopping(settings.patience, settings.min_delta)
        tracker = TargetTracker(settings.target_quality, settings.target_consecutive)
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)

        t0 = time.monotonic()
        summary.update(status="running", training_start_elapsed_s=0.0, training_start_unix=time.time(),
                       training_start_iso=iso_now())
        write_json(run_dir / "summary.json", summary)
        log(f"▶ {run_id} | {seg_display_name(spec.model)} | CLAHE={'on' if spec.clahe else 'off'} | "
            f"train={len(train_df)} ảnh/{summary['n_train_videos']} video, val={len(val_df)} | {device}")
        best = {}
        for epoch in range(1, settings.epochs + 1):
            start = time.monotonic()
            model.train()
            loss_sum, seen = 0.0, 0
            for images, masks in train_loader:
                if settings.max_minutes and time.monotonic() - t0 > settings.max_minutes * 60:
                    raise TimeoutError("time_budget_exceeded")
                images, masks = images.to(device, non_blocking=pin), masks.to(device, non_blocking=pin)
                with autocast_context(device, use_amp):
                    logits = model(images)
                logits = logits.float()
                loss = F.cross_entropy(logits, masks) + settings.dice_weight * soft_dice_loss(logits, masks)
                optimizer.zero_grad(set_to_none=True)
                scaler.scale(loss).backward()
                if settings.grad_clip:
                    scaler.unscale_(optimizer)
                    nn.utils.clip_grad_norm_(model.parameters(), settings.grad_clip)
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                loss_sum += float(loss.detach()) * images.size(0)
                seen += images.size(0)
            synchronize(device)
            train_seconds = time.monotonic() - start
            train_loss = loss_sum / max(seen, 1)
            if not math.isfinite(train_loss):
                raise FloatingPointError("non_finite_loss")
            metrics, val_loss, _ = evaluate_loader(model, val_loader, device, use_amp, settings.dice_weight)
            score = metrics["val_mean_dice"]
            improved = stopper.step(score, epoch)
            reached = tracker.step(score, epoch)
            now = time.monotonic() - t0
            if improved:
                best = metrics
                save_checkpoint(run_dir / "best.pt", model, {
                    "kind": "segmentation", "model_name": spec.model, "num_classes": len(CLASSES), "classes": CLASSES,
                    "display_names": task_info.get("display_names", {}), "resolution": spec.resolution,
                    "clahe": bool(spec.clahe), "geometry": task_info.get("geometry", "center_square"), "task": task,
                    "run_id": run_id, "epoch": epoch, "val_score": float(score), "manifest_sha256": manifest_sha,
                    "created_at": iso_now(),
                })
            if reached:
                summary.update(target_reached=1, target_epoch=epoch, target_time_to_target_s=now,
                               target_reached_unix=time.time(), target_reached_iso=iso_now())
            history.append({"epoch": epoch, "timestamp_unix": time.time(), "elapsed_s": now,
                            "epoch_seconds": time.monotonic() - start, "train_seconds": train_seconds,
                            "train_loss": train_loss, "val_loss": val_loss, **metrics,
                            "train_images_per_s": seen / train_seconds if train_seconds > 0 else None,
                            "lr_head": optimizer.param_groups[-1]["lr"], "improved": improved,
                            "target_streak": tracker.streak, "gpu_mem_peak_gb": _peak_memory_gb(device)})
            write_csv(pd.DataFrame(history), run_dir / "history.csv")
            summary.update(epochs_run=epoch, best_epoch=stopper.best_epoch, best_val_score=stopper.best,
                           best_macro_f1=stopper.best,  # tên cột TATO; với phân đoạn = val mean Dice tốt nhất
                           final_macro_f1=score,
                           mean_epoch_seconds=float(np.mean([h["epoch_seconds"] for h in history])),
                           train_images_per_s_mean=float(np.mean([h["train_images_per_s"] or 0 for h in history])),
                           gpu_mem_peak_gb=history[-1]["gpu_mem_peak_gb"], **best)
            write_json(run_dir / "summary.json", summary)
            log(f"  epoch {epoch:>3}/{settings.epochs} | loss {train_loss:.4f} / val {val_loss:.4f} | Dice "
                f"trái {metrics['val_dice_left_fold']:.3f} phải {metrics['val_dice_right_fold']:.3f} "
                f"thanh môn {metrics['val_dice_glottis']:.3f} | mean {score:.4f} | {history[-1]['epoch_seconds']:.0f}s"
                + (" ★" if improved else ""))
            if reached and settings.stop_on_target:
                summary["stop_reason"] = "target_reached"
                break
            if stopper.should_stop:
                summary["stop_reason"] = "early_stopping"
                break
        else:
            summary["stop_reason"] = "max_epochs"
        summary["status"] = "completed"
    except TimeoutError as exc:
        summary.update(status="aborted", stop_reason="aborted", abort_reason=str(exc))
    except KeyboardInterrupt:
        summary.update(status="interrupted", stop_reason="interrupted", abort_reason="keyboard_interrupt")
        raise
    except Exception as exc:
        reason = "out_of_memory (giảm batch_size)" if _is_out_of_memory(exc) else f"{type(exc).__name__}: {exc}"
        summary.update(status="failed", stop_reason="failed", abort_reason=reason)
        (run_dir / "error.txt").write_text(traceback.format_exc(), encoding="utf-8")
        log(f"  ✖ Lỗi: {reason}")
    finally:
        if summary["status"] in ("initializing", "running"):
            summary["status"] = "failed"
        summary.update(finished_at=iso_now(), finished_unix=time.time())
        write_json(run_dir / "summary.json", summary)
        close_loader(train_loader)
        close_loader(val_loader)
        model = optimizer = train_loader = val_loader = None  # noqa: F841
        gc.collect()
        empty_cache(device)
    if summary["status"] == "completed":
        log(f"✔ Xong {run_id}: best val mean Dice={summary.get('best_val_score', float('nan')):.4f} "
            f"tại epoch {summary.get('best_epoch')} ({summary.get('stop_reason')})")
    return summary


def load_seg_checkpoint(path, device="cpu") -> tuple[nn.Module, dict]:
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if checkpoint.get("kind") != "segmentation":
        raise ValueError(f"{path} không phải checkpoint phân đoạn.")
    model, _ = build_seg_model(checkpoint["model_name"], int(checkpoint["num_classes"]), pretrained=False)
    model.load_state_dict(checkpoint["state_dict"])
    model.to(device).eval()
    return model, {k: v for k, v in checkpoint.items() if k != "state_dict"}
