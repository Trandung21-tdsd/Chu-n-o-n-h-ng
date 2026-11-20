"""Xuất dữ liệu profiling cho TATO (Thermal-Aware Training Optimizer) từ các lần train THẬT (không tổng hợp giả).

Ghi vào <out>/tato/ (copy cả thư mục này sang repo TATO: data/profiling/):
    profiling_index.csv                    cột đúng như TATO scripts/build_dataset.py đọc (chỉ 4 backbone của C01–C08)
    static_profiling_dataset.csv           cùng định dạng data/processed/static_profiling_dataset.csv của TATO
    profiling_index_extended.csv           mọi backbone + cột bổ sung (CLAHE, duplicate, phần cứng, thời gian...)
    static_profiling_dataset_extended.csv  one-hot cho mọi backbone + đặc trưng tĩnh bổ sung
    online_observations.csv                mỗi dòng = 1 epoch: observed_* (chỉ dữ liệu tới thời điểm đó) + nhãn còn lại
    runs/<run_id>/telemetry.csv, history.csv, summary.json
    schema.json                            mô tả cột, đơn vị, quy tắc chống leakage

Ví dụ: python -m throat_ai.tato_export --task bagls_3class
"""
from __future__ import annotations

import argparse
import re
import shutil
from pathlib import Path

import numpy as np
import pandas as pd

from .report import load_summaries
from .telemetry import ols_slope, thermal_event_in_horizon
from .utils import DEFAULT_OUTPUT_DIR, is_finite, iso_now, log, resolve_cli_path, setup_console, warn, write_csv, write_json

TATO_BACKBONES = {
    "EfficientNet-B0": "config_backbone_EfficientNet_B0",
    "MobileNetV3-Small": "config_backbone_MobileNetV3_Small",
    "ResNet-18": "config_backbone_ResNet_18",
    "ResNet-50": "config_backbone_ResNet_50",
}
PRE_RUN_FEATURES = [
    "pre_run_gpu_temp_current_c", "pre_run_gpu_temp_mean_60s_c", "pre_run_gpu_temp_max_60s_c",
    "pre_run_gpu_temp_slope_60s_c_per_s", "pre_run_cpu_temp_current_c", "pre_run_ambient_temp_c",
    "pre_run_gpu_power_current_w", "pre_run_gpu_util_current", "pre_run_cpu_util_current",
    "pre_run_ram_used_gb", "pre_run_vram_used_gb",
]
STATIC_FEATURES = [
    "config_batch_size", "config_precision_amp", "config_workers", "config_resolution", "config_params_m",
    "config_flops_g", *TATO_BACKBONES.values(), "hardware_gpu_vram_gb", "hardware_ram_gb",
    "dataset_num_train_images", "dataset_num_val_images", "dataset_num_classes", *PRE_RUN_FEATURES,
]
EXTRA_STATIC_FEATURES = ["config_accumulation", "config_clahe", "config_dedup_train"]
ID_COLUMNS = ["run_id", "config_id", "thermal_start_state", "seed"]
TARGET_COLUMNS = ["target_reached", "target_energy_to_target_wh", "target_time_to_target_s",
                  "full_run_gpu_temp_max_c", "throttled", "best_macro_f1"]
INDEX_COLUMNS = ["run_id", "config_id", "backbone", "batch_size", "precision", "workers", "resolution",
                 *PRE_RUN_FEATURES, "thermal_start_state", "seed", *TARGET_COLUMNS]
INDEX_EXTRA_COLUMNS = [
    "training_start_unix", "training_start_iso", "target_reached_unix", "target_reached_iso", "finished_unix",
    "finished_at", "status", "abort_reason", "stop_reason", "session_id", "machine_id", "run_order", "dataset_id", "task", "model",
    "accumulation", "clahe", "dup_policy", "params_m", "gmacs", "epochs_run", "best_epoch", "final_macro_f1",
    "target_epoch", "task_quality_metric", "task_quality_value", "full_run_time_s", "full_run_gpu_energy_wh",
    "full_run_gpu_power_mean_w", "full_run_gpu_temp_mean_c", "throttle_seconds", "power_capped_seconds",
    "pre_run_window_s", "pre_run_samples", "precondition", "telemetry_gpu_source", "hardware_gpu_model",
    "hardware_cpu_model", "hardware_os", "hardware_device", "hardware_torch_version", "hardware_cuda_version",
    "dataset_num_train_images", "dataset_num_val_images", "dataset_num_classes", "config_params_m", "config_flops_g",
    "started_at", "finished_at",
]

# Giống src/features/validator.py của TATO (+ chặn thêm metric val/test).
FORBIDDEN_STATIC_NAMES = {
    "f1_at_epoch_1", "f1_at_epoch_3", "loss_slope_early", "images_per_sec_mean_full_run", "step_time_mean_full_run",
    "gpu_power_mean_full_run", "gpu_temp_mean_full_run", "gpu_temp_max_full_run", "throttle_count_full_run",
    "throttle_duration_full_run", "full_run_gpu_temp_max_c", "full_run_gpu_temp_mean_c", "total_time_s",
    "total_energy_wh", "final_macro_f1", "best_macro_f1", "best_epoch",
}
FORBIDDEN_PATTERNS = [re.compile(p) for p in (r"^observed_.*", r"^full_run_.*", r".*_at_epoch_.*", r".*_full_run.*",
                                              r"^val_.*", r"^test_.*", r"^target_.*")]


class DataLeakageError(ValueError):
    pass


def validate_static_feature_names(names) -> None:
    leaked = [n for n in names if n in FORBIDDEN_STATIC_NAMES or any(p.match(n) for p in FORBIDDEN_PATTERNS)]
    if leaked:
        raise DataLeakageError(f"Đặc trưng tĩnh chứa thông tin tương lai (future leakage): {leaked}")


def one_hot_name(backbone: str) -> str:
    return TATO_BACKBONES.get(backbone) or "config_backbone_" + re.sub(r"[^A-Za-z0-9]+", "_", str(backbone)).strip("_")


def _assign_folds(frame: pd.DataFrame, n_splits: int = 5) -> pd.Series:
    if len(frame) < 2:
        return pd.Series([0] * len(frame), index=frame.index)
    from sklearn.model_selection import GroupKFold

    folds = pd.Series(-1, index=frame.index)
    splitter = GroupKFold(n_splits=min(n_splits, len(frame)))
    for fold, (_, test_index) in enumerate(splitter.split(frame, groups=frame["run_id"])):
        folds.iloc[test_index] = fold
    return folds


def _static_table(runs: pd.DataFrame, backbones: list[str], extra: bool) -> pd.DataFrame:
    table = runs[ID_COLUMNS].copy()
    for column in STATIC_FEATURES:
        if column in TATO_BACKBONES.values():
            continue
        table[column] = runs[column] if column in runs else np.nan
    for backbone in backbones:
        table[one_hot_name(backbone)] = (runs["backbone"] == backbone).astype(float)
    if extra:
        for column in EXTRA_STATIC_FEATURES:
            table[column] = runs[column] if column in runs else np.nan
    ordered = ID_COLUMNS + [c for c in STATIC_FEATURES if c in table] + \
        [one_hot_name(b) for b in backbones if one_hot_name(b) not in STATIC_FEATURES] + \
        ([c for c in EXTRA_STATIC_FEATURES] if extra else [])
    table = table[list(dict.fromkeys(ordered))]
    validate_static_feature_names([c for c in table.columns if c not in ID_COLUMNS])
    for column in TARGET_COLUMNS:
        table[column] = runs[column] if column in runs else np.nan
    table["fold"] = _assign_folds(table).to_numpy()
    return table


def _online_rows(run: dict, horizon_s: float, static: dict) -> list[dict]:
    run_dir = Path(run["run_dir"])
    history_path, telemetry_path = run_dir / "history.csv", run_dir / "telemetry.csv"
    if not history_path.exists():
        return []
    history = pd.read_csv(history_path)
    telemetry = pd.read_csv(telemetry_path).to_dict("records") if telemetry_path.exists() else []
    start = run.get("training_start_elapsed_s")
    target_time, target_energy = run.get("target_time_to_target_s"), run.get("target_energy_to_target_wh")
    reached = run.get("target_reached") == 1
    rows = []
    for index, row in history.iterrows():
        last5 = history.iloc[max(0, index - 4): index + 1]
        elapsed, energy = row.get("elapsed_s"), row.get("gpu_energy_wh_cumulative")
        if reached and is_finite(target_time) and elapsed <= float(target_time) + 1e-6:
            remaining_time = float(target_time) - elapsed
            remaining_energy = (float(target_energy) - energy) if is_finite(target_energy) and is_finite(energy) else None
            within = 1
        elif reached:
            remaining_time = remaining_energy = within = None  # đã đạt trước epoch này
        else:
            remaining_time = remaining_energy = None
            within = 0
        event, censored = (None, 1)
        if is_finite(start) and telemetry:
            event, censored = thermal_event_in_horizon(telemetry, float(start) + float(elapsed), horizon_s)
        rows.append({
            **{k: run.get(k) for k in ID_COLUMNS}, "session_id": run.get("session_id"), **static,
            "observed_epoch": int(row["epoch"]),
            "observed_elapsed_time_s": elapsed,
            # chất lượng validation tại epoch e (model phân đoạn: val mean Dice — xem task_quality_metric)
            "observed_validation_macro_f1_at_epoch_e": row.get("val_macro_f1", row.get("val_mean_dice")),
            "observed_validation_balanced_accuracy_at_epoch_e": row.get("val_balanced_accuracy"),
            "observed_train_loss_mean_last_5_epochs": float(last5["train_loss"].mean()),
            "observed_loss_slope_last_5_epochs": ols_slope(last5["epoch"], last5["train_loss"]),
            "observed_gpu_power_mean_last_60s_w": row.get("gpu_power_mean_60s_w"),
            "observed_gpu_util_mean_last_60s": row.get("gpu_util_mean_60s"),
            "observed_cpu_util_mean_last_60s": row.get("cpu_util_mean_60s"),
            "observed_images_per_sec_mean_last_60s": row.get("images_per_s_60s"),
            "observed_gpu_temp_mean_last_60s_c": row.get("gpu_temp_mean_60s_c"),
            "observed_gpu_temp_max_last_60s_c": row.get("gpu_temp_max_60s_c"),
            "observed_gpu_temp_slope_last_60s_c_per_s": row.get("gpu_temp_slope_60s_c_per_s"),
            "observed_throttled_current": row.get("throttled_current"),
            "observed_cumulative_energy_wh": energy,
            "remaining_time_to_target_s": remaining_time,
            "remaining_energy_to_target_wh": remaining_energy,
            "target_reached_within_budget": within,
            "thermal_event_next_horizon": event,
            "thermal_event_horizon_s": horizon_s,
            "thermal_event_horizon_censored": censored,
        })
    return rows


def export_tato(out_dir, tato_dir=None, horizon_s: float = 60.0) -> dict:
    out_dir = Path(out_dir)
    tato_dir = Path(tato_dir) if tato_dir else out_dir / "tato"
    runs = [r for r in load_summaries(out_dir)
            if r.get("status") in ("completed", "aborted") and not r.get("leaky") and "training_start_elapsed_s" in r]
    if not runs:
        warn("Chưa có run nào đã chạy xong để xuất cho TATO.")
        return {}
    tato_dir.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(runs).sort_values("started_at").reset_index(drop=True)
    frame["precision"] = frame["precision_effective"]  # TATO suy ra config_precision_amp từ cột này

    extended_index = frame.reindex(columns=list(dict.fromkeys(INDEX_COLUMNS + INDEX_EXTRA_COLUMNS)))
    tato_only = frame[frame["backbone"].isin(TATO_BACKBONES)]
    paths = {
        "profiling_index": write_csv(tato_only.reindex(columns=INDEX_COLUMNS), tato_dir / "profiling_index.csv"),
        "profiling_index_extended": write_csv(extended_index, tato_dir / "profiling_index_extended.csv"),
    }
    if not tato_only.empty:
        paths["static_profiling_dataset"] = write_csv(
            _static_table(tato_only.reset_index(drop=True), list(TATO_BACKBONES), extra=False),
            tato_dir / "static_profiling_dataset.csv")
    all_backbones = list(TATO_BACKBONES) + sorted(set(frame["backbone"]) - set(TATO_BACKBONES))
    extended_static = _static_table(frame, all_backbones, extra=True)
    paths["static_profiling_dataset_extended"] = write_csv(extended_static, tato_dir / "static_profiling_dataset_extended.csv")

    online = []
    static_columns = [c for c in extended_static.columns if c not in ID_COLUMNS + TARGET_COLUMNS + ["fold"]]
    for position, run in enumerate(frame.to_dict("records")):
        online += _online_rows(run, horizon_s, extended_static.loc[position, static_columns].to_dict())
        target = tato_dir / "runs" / run["run_id"]
        target.mkdir(parents=True, exist_ok=True)
        for name in ("telemetry.csv", "history.csv", "summary.json", "static_at_start.json"):
            source = Path(run["run_dir"]) / name
            if source.exists():
                shutil.copy2(source, target / name)
    if online:
        paths["online_observations"] = write_csv(pd.DataFrame(online), tato_dir / "online_observations.csv")

    no_gpu = int((frame.get("telemetry_gpu_source").isna()).sum()) if "telemetry_gpu_source" in frame else len(frame)
    write_json(tato_dir / "schema.json", {
        "schema_version": "throat-ai-tato-1.0",
        "exported_at": iso_now(),
        "runs_exported": int(len(frame)),
        "runs_tato_backbones": int(len(tato_only)),
        "runs_without_gpu_telemetry": no_gpu,
        "thermal_measurement": "Đo nhiệt/công suất (pre_run_*, full_run_gpu_temp_max_c, throttled, energy) do người khác "
                               "đảm nhận; cột để trống. Ghép log nhiệt của họ vào từng run theo thời gian thực: "
                               "training_start_unix -> target_reached_unix (mốc đạt Q*) -> finished_unix; từng epoch có "
                               "timestamp_unix trong runs/<run_id>/history.csv.",
        "join_key": "run_id",
        "static_feature_columns": STATIC_FEATURES,
        "extra_static_feature_columns": EXTRA_STATIC_FEATURES,
        "target_columns": TARGET_COLUMNS,
        "units": {"temperature": "°C", "power": "W", "energy": "Wh", "time": "s", "util": "0..1", "memory": "GB",
                  "config_flops_g": "GMACs/ảnh ở config_resolution (cùng quy ước TATO, ResNet-50 ≈ 4.1)"},
        "quality_target": "target_reached=1 khi metric validation (task_quality_metric) >= Q* trong task_target_consecutive "
                          "lần liên tiếp; không đạt => target_time/energy để TRỐNG (không gán tổng thời gian run)",
        "energy_scope": "Năng lượng GPU đo bằng NVML (tích phân hình thang). Không gồm CPU/màn hình/toàn hệ thống.",
        "missing_values": "Trống = không đo được (vd. Mac/CPU không có NVML; CPU temp trên Windows/macOS). KHÔNG phải 0.",
        "leakage_rules": "Static Predictor chỉ dùng config_/hardware_/dataset_/pre_run_ (đo trong 60 s trước khi train). "
                         "observed_* chỉ dùng cho Online Predictor; full_run_*/target_*/best_* chỉ là nhãn.",
        "online_targets": {"remaining_*": "chỉ có ở các epoch trước/bằng mốc đạt mục tiêu",
                           "thermal_event_next_horizon": f"có throttling do nhiệt trong {horizon_s:.0f}s kế tiếp; "
                                                         "censored=1 nếu run kết thúc trước hết horizon"},
        "split_advice": "Khi train predictor của TATO: chia theo run_id (tối thiểu) hoặc session_id/machine_id.",
        "excluded": "Run 'naive' (leaky), run lỗi, run bị bỏ qua, run bị ngắt bằng Ctrl+C không được xuất.",
        "notes": "pre_run_ambient_temp_c chỉ có khi chạy với --ambient-temp. Cột fold = GroupKFold theo run_id.",
    })
    paths["schema"] = tato_dir / "schema.json"
    if no_gpu:
        log(f"{no_gpu}/{len(frame)} run không tự đo nhiệt (đúng mặc định) -> cột nhiệt/năng lượng để trống cho người "
            "đo nhiệt ghép vào theo training_start_unix / finished_unix.")
    log(f"Đã xuất dữ liệu TATO ({len(frame)} run) -> {tato_dir}")
    return paths


def main(argv=None) -> None:
    setup_console()
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--task", default="bagls_3class")
    parser.add_argument("--out", default=None, help="thư mục benchmark (mặc định outputs/<task>)")
    parser.add_argument("--tato-dir", default=None, help="nơi ghi (mặc định <out>/tato)")
    parser.add_argument("--horizon", type=float, default=60.0, help="horizon (giây) cho nhãn thermal_event_next_horizon")
    args = parser.parse_args(argv)
    out_dir = resolve_cli_path(args.out) if args.out else DEFAULT_OUTPUT_DIR / args.task
    export_tato(out_dir, resolve_cli_path(args.tato_dir) if args.tato_dir else None, args.horizon)


if __name__ == "__main__":
    main()
