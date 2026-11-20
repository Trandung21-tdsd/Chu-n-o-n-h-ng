"""BƯỚC 3 — Chọn model theo VALIDATION (không nhìn test) -> rồi mới đánh giá TEST -> báo cáo -> xuất dữ liệu TATO.

Thứ tự bắt buộc để tránh rò rỉ thông tin từ test:
    1) selection.json được ghi TRƯỚC khi tính bất kỳ metric test nào;
    2) hiệu chuẩn nhiệt độ (temperature scaling) chỉ fit trên validation;
    3) test của mỗi checkpoint chỉ tính 1 lần (lưu test_metrics.json, chạy lại thì đọc file cũ).

Ví dụ:
    python -m throat_ai.evaluate --task bagls_3class
    python -m throat_ai.evaluate --task bagls_3class --no-test     # chỉ chọn model + báo cáo validation
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from .data import attach_paths_and_targets, load_prepared, partition, verify_files
from .inference import load_checkpoint, predict_paths, save_prediction_files
from .metrics import apply_temperature, compute_metrics, expected_calibration_error, fit_temperature, patient_metrics
from .models import pick_device
from .report import CONFIG_KEYS, load_summaries, write_reports
from .tato_export import export_tato
from .utils import (
    DEFAULT_OUTPUT_DIR,
    DEFAULT_PREPARED_DIR,
    iso_now,
    log,
    parse_key_values,
    read_json,
    resolve_cli_path,
    setup_console,
    warn,
    write_json,
)

SELECTION_RULE = ("Gộp các run theo cấu hình (model, batch, precision, resolution, CLAHE, duplicate); chọn cấu hình có "
                  "TRUNG BÌNH metric validation (qua các seed) cao nhất — hòa thì balanced accuracy cao hơn, rồi ít tham "
                  "số hơn; trong cấu hình đó lấy checkpoint của seed có validation cao nhất. Không dùng run 'naive'.")


def _eligible(runs: list[dict], manifest_sha: str) -> pd.DataFrame:
    frame = pd.DataFrame(runs)
    if frame.empty:
        return frame
    for column in ("leaky", "best_val_score", "manifest_sha256", "finished_at"):
        if column not in frame:
            frame[column] = None
    mask = (frame["status"] == "completed") & (~frame["leaky"].fillna(False).astype(bool)) \
        & (frame["manifest_sha256"] == manifest_sha) & frame["best_val_score"].notna()
    frame = frame.loc[mask].copy()
    has_checkpoint = np.array([Path(d, "best.pt").exists() for d in frame["run_dir"]], dtype=bool)
    frame = frame.loc[has_checkpoint]
    if frame.empty:
        return frame
    return frame.sort_values("finished_at").drop_duplicates(CONFIG_KEYS + ["seed"], keep="last")


def select_best(runs: list[dict], manifest_sha: str) -> dict | None:
    frame = _eligible(runs, manifest_sha)
    if frame.empty:
        return None
    if frame["monitor_effective"].nunique() > 1:
        warn(f"Các run dùng metric early-stopping khác nhau {sorted(frame['monitor_effective'].unique())} — "
             "nên benchmark với cùng --monitor để so sánh công bằng.")
    for column in ("val_balanced_accuracy", "params_m"):
        if column not in frame:
            frame[column] = np.nan
    grouped = (frame.groupby(CONFIG_KEYS, dropna=False)
               .agg(val_mean=("best_val_score", "mean"), val_std=("best_val_score", "std"),
                    bal_mean=("val_balanced_accuracy", "mean"), params=("params_m", "first"), n=("run_id", "count"))
               .reset_index()
               .sort_values(["val_mean", "bal_mean", "params"], ascending=[False, False, True]))
    top = grouped.iloc[0]
    same_config = np.logical_and.reduce([frame[k].astype(str).to_numpy() == str(top[k]) for k in CONFIG_KEYS])
    members = frame.loc[same_config]
    best = members.sort_values(["best_val_score", "val_balanced_accuracy"], ascending=False).iloc[0]
    return {
        "run_id": best["run_id"], "run_dir": best["run_dir"], "checkpoint": str(Path(best["run_dir"]) / "best.pt"),
        "model": best["model"], "backbone": best["backbone"], "config_id": best["config_id"],
        "clahe": bool(best["clahe"]), "dup_policy": best["dup_policy"], "seed": int(best["seed"]),
        "monitor": best["monitor_effective"], "val_score": float(best["best_val_score"]),
        "config_val_mean": float(top["val_mean"]), "config_val_std": None if pd.isna(top["val_std"]) else float(top["val_std"]),
        "config_n_seeds": int(top["n"]), "candidates_compared": int(len(frame)), "rule": SELECTION_RULE,
        "manifest_sha256": manifest_sha, "test_used_for_selection": False, "selected_at": iso_now(),
    }


def calibrate(selection: dict, classes: list[str]) -> dict:
    """Fit nhiệt độ T trên dự đoán VALIDATION của run được chọn."""
    path = Path(selection["run_dir"]) / "val_predictions.csv"
    if not path.exists():
        return {"temperature": 1.0}
    table = pd.read_csv(path, dtype={"label": str})
    probs = table[[f"p_{c}" for c in classes]].to_numpy()
    y = table["label"].map({c: i for i, c in enumerate(classes)}).to_numpy()
    temperature = fit_temperature(y, probs)
    return {"temperature": temperature,
            "val_ece_before": expected_calibration_error(y, probs),
            "val_ece_after": expected_calibration_error(y, apply_temperature(probs, temperature))}


def evaluate_test(run_dir: Path, frame: pd.DataFrame, classes: list[str], device, batch_size: int = 64,
                  workers: int = 2, force: bool = False) -> dict | None:
    """Đánh giá test cho 1 checkpoint (đúng tập test theo chính sách duplicate của run đó)."""
    run_dir = Path(run_dir)
    summary_path = run_dir / "summary.json"
    summary = read_json(summary_path)
    metrics_path = run_dir / "test_metrics.json"
    if metrics_path.exists() and not force:
        return read_json(metrics_path)
    model, meta = load_checkpoint(run_dir / "best.pt", device)
    if list(meta["classes"]) != list(classes):
        warn(f"{run_dir.name}: danh sách lớp trong checkpoint khác manifest — bỏ qua.")
        return None
    test_df = partition(frame, "test", summary["dup_policy"])
    verify_files(test_df["abs_path"])
    probs = predict_paths(model, meta, test_df["abs_path"].tolist(), device, batch_size=batch_size, workers=workers)
    metrics = {f"test_{k}": v for k, v in compute_metrics(test_df["y"], probs).items()}
    metrics.update({f"test_patient_{k}": v for k, v in patient_metrics(test_df, probs).items()})
    metrics.update(n_test_images=int(len(test_df)),
                   n_test_patients=int(test_df.loc[test_df["patient_key"] != "", "patient_key"].nunique()),
                   test_evaluated_at=iso_now(), test_checkpoint_epoch=meta.get("epoch"))
    save_prediction_files(run_dir, "test", test_df, probs, classes)
    np.save(run_dir / "test_probs.npy", probs)
    write_json(metrics_path, metrics)
    summary.update(metrics)
    write_json(summary_path, summary)
    return metrics


def finalize(out_dir, prepared_dir, device_request: str = "auto", run_test: bool = True, source_roots: dict | None = None,
             batch_size: int = 64, workers: int = 2, force_test: bool = False) -> dict | None:
    out_dir, prepared_dir = Path(out_dir), Path(prepared_dir)
    frame, task, manifest_sha = load_prepared(prepared_dir)
    classes = list(task["classes"])
    frame = attach_paths_and_targets(frame, classes, source_roots)
    runs = load_summaries(out_dir)
    if not runs:
        warn(f"Chưa có run nào trong {out_dir / 'runs'}")
        return None
    other = sum(r.get("manifest_sha256") != manifest_sha for r in runs)
    if other:
        warn(f"{other} run được train trên manifest khác (split cũ) — không đưa vào lựa chọn/đánh giá.")

    # (1) Chốt lựa chọn bằng validation — ghi ra file TRƯỚC khi đụng tới test.
    selection = select_best(runs, manifest_sha)
    if selection is not None:
        selection.update(calibrate(selection, classes))
        write_json(out_dir / "selection.json", selection)
        log(f"Model được chọn (theo validation): {selection['backbone']} | CLAHE={'on' if selection['clahe'] else 'off'} "
            f"| duplicate={selection['dup_policy']} | seed={selection['seed']} | {selection['monitor']}={selection['val_score']:.4f}")
    else:
        warn("Không có run hoàn thành hợp lệ để chọn model.")

    # (2) Sau khi đã chốt: đo test cho các checkpoint hoàn thành (để có bảng so sánh), mỗi checkpoint 1 lần.
    if run_test:
        device = pick_device(device_request)
        done = [r for r in runs if r.get("status") == "completed" and r.get("manifest_sha256") == manifest_sha
                and Path(r["run_dir"], "best.pt").exists()]
        for index, run in enumerate(done, start=1):
            if not Path(run["run_dir"], "test_metrics.json").exists() or force_test:
                log(f"Đánh giá test {index}/{len(done)}: {run['run_id']}")
            try:
                evaluate_test(Path(run["run_dir"]), frame, classes, device, batch_size, workers, force_test)
            except Exception as exc:
                warn(f"Không đánh giá test được {run['run_id']}: {type(exc).__name__}: {exc}")
        if selection is not None:
            metrics_path = Path(selection["run_dir"]) / "test_metrics.json"
            if metrics_path.exists():
                selection["test_metrics"] = read_json(metrics_path)
                probs_path = Path(selection["run_dir"]) / "test_probs.npy"
                if probs_path.exists():
                    test_df = partition(frame, "test", selection["dup_policy"])
                    probs = np.load(probs_path)
                    selection["test_ece_before"] = expected_calibration_error(test_df["y"].to_numpy(), probs)
                    selection["test_ece_after"] = expected_calibration_error(
                        test_df["y"].to_numpy(), apply_temperature(probs, selection["temperature"]))
                write_json(out_dir / "selection.json", selection)
                test = selection["test_metrics"]
                log(f"TEST của model được chọn: macro-F1={test.get('test_macro_f1', float('nan')):.4f} "
                    f"| balanced acc={test.get('test_balanced_accuracy', float('nan')):.4f} "
                    f"| accuracy={test.get('test_accuracy', float('nan')):.4f}")

    prepare_summary_path = prepared_dir / "prepare_summary.json"
    prepare_summary = read_json(prepare_summary_path) if prepare_summary_path.exists() else None
    paths = write_reports(out_dir, task, selection, prepare_summary)
    try:
        export_tato(out_dir)
    except Exception as exc:
        warn(f"Xuất dữ liệu TATO lỗi: {type(exc).__name__}: {exc}")
    for name, path in paths.items():
        log(f"  {name}: {path}")
    return selection


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--task", default="bagls_3class")
    parser.add_argument("--prepared-dir", default=str(DEFAULT_PREPARED_DIR))
    parser.add_argument("--out", default=None, help="thư mục benchmark (mặc định outputs/<task>)")
    parser.add_argument("--device", default="auto", help="auto | cuda | cuda:0 | mps | cpu")
    parser.add_argument("--no-test", action="store_true", help="chỉ chọn model + báo cáo, chưa tính test")
    parser.add_argument("--force-test", action="store_true", help="tính lại test dù đã có (không ảnh hưởng lựa chọn)")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--source-root", action="append", default=[], metavar="TÊN=ĐƯỜNG_DẪN")
    return parser


def main(argv=None) -> None:
    setup_console()
    args = build_parser().parse_args(argv)
    out_dir = resolve_cli_path(args.out) if args.out else DEFAULT_OUTPUT_DIR / args.task
    finalize(out_dir, resolve_cli_path(args.prepared_dir) / args.task, args.device, not args.no_test,
             parse_key_values(args.source_root), args.batch_size, args.workers, args.force_test)


if __name__ == "__main__":
    main()
