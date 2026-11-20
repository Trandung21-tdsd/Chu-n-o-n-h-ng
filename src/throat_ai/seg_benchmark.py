"""Benchmark model PHÂN ĐOẠN thanh môn (2 dây thanh + thanh môn) trên BAGLS-VF, rồi rút dấu hiệu chẩn đoán từ mask.

Cần chạy trước:  python -m throat_ai.prepare --task bagls_3class   (dùng chung phân chia train/val theo video)

Ví dụ:
    python -m throat_ai.seg_benchmark --plan                                  # xem danh sách run
    python -m throat_ai.seg_benchmark                                         # 4 model (không CLAHE)
    python -m throat_ai.seg_benchmark --models unet_lite lraspp_mobilenet_v3  # nhanh, máy yếu
    python -m throat_ai.seg_benchmark --finalize-only                         # chỉ chọn model + đánh giá test lại

Kết quả (outputs/bagls_seg/):
    benchmark_results.csv      mỗi run 1 dòng (Dice từng lớp, IoU, thời gian, tham số...)
    model_comparison.csv       xếp hạng theo val mean Dice
    selection.json             model được chọn (theo VALIDATION) + kết quả test (đo 1 lần sau khi chọn)
    reference_ranges.json      khoảng số đo của thanh quản BÌNH THƯỜNG (từ mask gán tay của video train)
    test_video_signs.csv       mỗi video test: số đo từ mask dự đoán + dấu hiệu + nhóm bệnh được ủng hộ
    features_by_group.csv      số đo trung bình theo nhóm bệnh (train: mask thật, test: mask dự đoán)
    sign_classifier.json       chẩn đoán nhóm bệnh CHỈ từ số đo phân đoạn (hồi quy logistic, mức video)
    comparison.md + charts/    báo cáo + biểu đồ;  tato/  dữ liệu cho TATO (chất lượng = val mean Dice)
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from .glottis import VIDEO_FEATURES, assess_video, image_features, reference_ranges, video_features
from .models import pick_device
from .report import load_summaries
from .seg_data import CLASSES, center_square, load_mask, load_seg_config, load_seg_manifest
from .seg_models import SEG_MODELS, seg_display_name
from .seg_train import (
    ConfusionAccumulator, SegSettings, SegSpec, load_seg_checkpoint, predict_masks, run_key, train_seg_run,
)
from .utils import (
    DEFAULT_OUTPUT_DIR, DEFAULT_PREPARED_DIR, default_machine_id, iso_now, log, read_json, resolve_cli_path,
    setup_console, timestamp, warn, write_csv, write_json,
)

GROUPS = ["Normal", "Functional", "Organic_lesion"]


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", default=None, help="file cấu hình phân đoạn (mặc định configs/segmentation.json)")
    p.add_argument("--out", default=None, help="thư mục kết quả (mặc định outputs/bagls_seg)")
    p.add_argument("--prepared-dir", default=str(DEFAULT_PREPARED_DIR))
    p.add_argument("--models", nargs="+", default=list(SEG_MODELS), choices=list(SEG_MODELS))
    p.add_argument("--clahe", nargs="+", choices=["off", "on"], default=["off"],
                   help="mặc định chỉ 'off' (đã thử: CLAHE không cải thiện Dice); thêm 'on' nếu muốn so sánh lại")
    p.add_argument("--seeds", nargs="+", type=int, default=[42])
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--resolution", type=int, default=256, help="ảnh BAGLS-VF train là 256x256")
    p.add_argument("--precision", choices=["FP32", "AMP"], default="FP32")
    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--patience", type=int, default=6, help="early stopping theo val mean Dice")
    p.add_argument("--min-delta", type=float, default=0.001)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--head-lr-mult", type=float, default=10.0)
    p.add_argument("--dice-weight", type=float, default=1.0)
    p.add_argument("--no-augment", action="store_true")
    p.add_argument("--no-pretrained", action="store_true")
    p.add_argument("--target-quality", type=float, default=0.80, help="Q* (val mean Dice) cho TATO")
    p.add_argument("--stop-on-target", action="store_true")
    p.add_argument("--max-minutes", type=float, help="giới hạn thời gian mỗi run")
    p.add_argument("--device", default="auto")
    p.add_argument("--force", action="store_true", help="chạy lại cả run đã xong")
    p.add_argument("--plan", action="store_true", help="chỉ in danh sách run")
    p.add_argument("--no-finalize", action="store_true")
    p.add_argument("--finalize-only", action="store_true", help="không train, chỉ chọn model + đánh giá test")
    p.add_argument("--rebuild-manifest", action="store_true", help="tạo lại manifest phân đoạn")
    return p


# ------------------------------------------------------------------------------ chạy ma trận
def _done_runs(out_dir: Path) -> dict[str, str]:
    return {s["run_key"]: s["run_id"] for s in load_summaries(out_dir) if s.get("status") == "completed"}


def _results_table(out_dir: Path, selection: dict | None = None) -> pd.DataFrame:
    frame = pd.DataFrame(load_summaries(out_dir))
    if frame.empty:
        return frame
    frame["selected"] = frame["run_id"] == (selection or {}).get("run_id")
    front = ["run_id", "status", "selected", "backbone", "model", "clahe", "seed", "batch_size", "resolution",
             "precision_effective", "params_m", "gmacs", "best_val_score", "val_mean_dice", "val_dice_left_fold",
             "val_dice_right_fold", "val_dice_glottis", "val_mean_iou", "val_glottis_area_corr", "test_mean_dice",
             "best_epoch", "epochs_run", "stop_reason", "mean_epoch_seconds", "target_reached",
             "target_time_to_target_s", "training_start_unix", "finished_unix", "abort_reason"]
    cols = [c for c in front if c in frame] + [c for c in frame.columns if c not in front]
    return frame[cols]


def run_matrix(args, out_dir: Path, frame: pd.DataFrame, task_info: dict, sha: str, device, cfg: dict) -> None:
    settings = SegSettings(epochs=args.epochs, patience=args.patience, min_delta=args.min_delta, lr=args.lr,
                           head_lr_mult=args.head_lr_mult, dice_weight=args.dice_weight, augment=not args.no_augment,
                           pretrained=not args.no_pretrained, target_quality=args.target_quality,
                           stop_on_target=args.stop_on_target, max_minutes=args.max_minutes)
    specs = [SegSpec(config_id=f"S{i + 1:02d}", model=m, batch_size=args.batch_size, precision=args.precision,
                     workers=args.workers, resolution=args.resolution, clahe=c == "on", seed=s)
             for s in args.seeds for i, m in enumerate(args.models) for c in args.clahe]
    done = {} if args.force else _done_runs(out_dir)
    keys = [run_key(spec, settings, sha) for spec in specs]
    log(f"Kế hoạch: {len(specs)} run ({sum(k in done for k in keys)} đã xong) -> {out_dir}")
    for index, (spec, key) in enumerate(zip(specs, keys), start=1):
        status = f"đã xong ({done[key]})" if key in done else "chờ chạy"
        print(f"{index:>3}  {seg_display_name(spec.model):<24} CLAHE {'on ' if spec.clahe else 'off'} seed {spec.seed}  {status}")
    if args.plan:
        return
    session = f"seg-{timestamp()}"
    for order, (spec, key) in enumerate(zip(specs, keys), start=1):
        if key in done:
            continue
        train_seg_run(spec, settings, frame, task_info, sha, out_dir / "runs", device, cfg.get("augment"),
                      session_id=session, machine_id=default_machine_id(), run_order=order)
        write_csv(_results_table(out_dir), out_dir / "benchmark_results.csv")


# ------------------------------------------------------------------------------ chọn model + test + dấu hiệu
def _gt_masks(paths, size: int) -> list[np.ndarray]:
    return [center_square(load_mask(p), size, nearest=True) for p in paths]


def _video_table(rows: pd.DataFrame, masks: list[np.ndarray]) -> pd.DataFrame:
    feats = pd.DataFrame([image_features(m) for m in masks])
    feats["video_id"] = rows["video_id"].to_numpy()
    meta = rows.drop_duplicates("video_id").set_index("video_id")[["label", "histology"]]
    table = pd.DataFrame([{"video_id": v, **video_features(g)} for v, g in feats.groupby("video_id")])
    return table.join(meta, on="video_id")


def _sign_classifier(train_videos: pd.DataFrame, evals: dict[str, pd.DataFrame]) -> dict:
    """Hồi quy logistic: số đo video -> nhóm bệnh. Fit trên video TRAIN (mask thật), đo trên val/test (mask dự đoán)."""
    try:
        from sklearn.linear_model import LogisticRegression
        from sklearn.metrics import balanced_accuracy_score, f1_score
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler
    except ImportError:
        return {"error": "cần scikit-learn"}
    data = train_videos[train_videos["label"].isin(GROUPS)].dropna(subset=VIDEO_FEATURES)
    if data["label"].nunique() < 2:
        return {"error": "train thiếu nhóm bệnh"}
    clf = make_pipeline(StandardScaler(), LogisticRegression(class_weight="balanced", max_iter=2000, C=0.5))
    clf.fit(data[VIDEO_FEATURES], data["label"])
    scaler, logistic = clf[0], clf[-1]
    result = {"features": VIDEO_FEATURES, "fit_on": f"{len(data)} video train (mask gán tay)",
              # tham số đầy đủ (JSON, không pickle) để demo/predict dùng lại: xem glottis.sign_classifier_proba
              "model": {"classes": [str(c) for c in logistic.classes_], "mean": scaler.mean_.tolist(),
                        "scale": scaler.scale_.tolist(), "coef": logistic.coef_.tolist(),
                        "intercept": logistic.intercept_.tolist()}}
    for name, table in evals.items():
        part = table[table["label"].isin(clf.classes_)].dropna(subset=VIDEO_FEATURES)
        if part.empty:
            continue
        pred = clf.predict(part[VIDEO_FEATURES])
        result[name] = {"videos": int(len(part)),
                        "macro_f1": float(f1_score(part["label"], pred, average="macro")),
                        "balanced_accuracy": float(balanced_accuracy_score(part["label"], pred)),
                        "confusion": pd.crosstab(part["label"], pred).to_dict()}
        table.loc[part.index, "sign_classifier_prediction"] = pred
    return result


def finalize(out_dir: Path, frame: pd.DataFrame, device, clahe_allowed=(False,), manifest_sha: str | None = None) -> dict | None:
    """Chọn model trong các run đã xong; chỉ xét run có CLAHE thuộc `clahe_allowed` (mặc định: chỉ run không CLAHE)
    và train trên ĐÚNG manifest hiện tại (run train trên phân chia cũ bị bỏ qua — tránh dùng lại dữ liệu đã rò rỉ)."""
    runs = [s for s in load_summaries(out_dir) if s.get("status") == "completed" and (Path(s["run_dir"]) / "best.pt").exists()
            and bool(s.get("clahe")) in set(clahe_allowed)]
    stale = [s["run_id"] for s in runs if manifest_sha and s.get("manifest_sha256") != manifest_sha]
    if stale:
        warn(f"Bỏ qua {len(stale)} run train trên manifest CŨ (phân chia đã thay đổi): {', '.join(stale[:4])}"
             + (" ..." if len(stale) > 4 else ""))
        runs = [s for s in runs if s["run_id"] not in stale]
    if not runs:
        warn("Chưa có run phân đoạn nào hoàn thành.")
        return None
    best = max(runs, key=lambda s: (round(float(s.get("best_val_score") or 0), 4), -float(s.get("gmacs") or 1e9)))
    model, meta = load_seg_checkpoint(Path(best["run_dir"]) / "best.pt", device)
    size, clahe = int(meta["resolution"]), bool(meta["clahe"])
    log(f"Chọn theo VALIDATION: {best['run_id']} ({seg_display_name(best['model'])}, val mean Dice "
        f"{best['best_val_score']:.4f}). Giờ mới đánh giá TEST (1 lần).")

    results = {}
    for split in ("val", "test"):
        rows = frame[frame["split"] == split].reset_index(drop=True)
        pred = predict_masks(model, rows["image_path"], size, clahe, device)
        truth = _gt_masks(rows["mask_path"], size)
        acc = ConfusionAccumulator(len(CLASSES))
        for p, t in zip(pred, truth):
            acc.update(torch.from_numpy(p[None].astype(np.int64)), torch.from_numpy(t[None].astype(np.int64)))
        results[split] = (rows, pred, truth, {k.replace("val_", f"{split}_", 1): v for k, v in acc.metrics("val").items()})

    train_rows = frame[frame["split"] == "train"].reset_index(drop=True)
    train_videos = _video_table(train_rows, _gt_masks(train_rows["mask_path"], size))
    ranges = reference_ranges(train_videos)
    write_json(out_dir / "reference_ranges.json", {"source": "video BÌNH THƯỜNG của tập train, mask gán tay",
                                                   "percentiles": "p05/p50/p95", "ranges": ranges})
    tables = {split: _video_table(results[split][0], results[split][1]) for split in ("val", "test")}
    gt_test = _video_table(results["test"][0], results["test"][2])
    classifier = _sign_classifier(train_videos, tables)
    write_json(out_dir / "sign_classifier.json", classifier)

    signs = []
    for _, row in tables["test"].iterrows():
        assessment = assess_video(row.to_dict(), ranges)
        signs.append({"video_id": row["video_id"], "label": row["label"], "histology": row["histology"],
                      **{f: row[f] for f in VIDEO_FEATURES}, "n_frames": row["n_frames"],
                      "leaning": assessment["leaning"], "sign_classifier_prediction": row.get("sign_classifier_prediction"),
                      "signs": " | ".join(assessment["signs"])})
    write_csv(pd.DataFrame(signs), out_dir / "test_video_signs.csv")
    sources = pd.concat([train_videos.assign(source="train (mask thật)"), gt_test.assign(source="test (mask thật)"),
                         tables["test"].assign(source="test (mask dự đoán)")])
    sources = sources[sources["label"].isin(GROUPS)].groupby(["source", "label"])
    by_group = sources[VIDEO_FEATURES].mean().round(4)
    by_group.insert(0, "videos", sources.size())
    write_csv(by_group.reset_index(), out_dir / "features_by_group.csv")

    test_metrics = results["test"][3]
    selection = {
        "kind": "segmentation", "run_id": best["run_id"], "model": best["model"], "backbone": seg_display_name(best["model"]),
        "checkpoint": str(Path(best["run_dir"]) / "best.pt"), "resolution": size, "clahe": clahe,
        "val_score": best["best_val_score"], "monitor": "val_mean_dice", "val_metrics": results["val"][3],
        "test_metrics": test_metrics, "test_used_for_selection": False,
        "reference_ranges": str(out_dir / "reference_ranges.json"), "sign_classifier": classifier,
        "candidates": len(runs), "selected_at": iso_now(),
    }
    write_json(out_dir / "selection.json", selection)
    for summary_path in Path(best["run_dir"]).glob("summary.json"):
        summary = read_json(summary_path)
        summary.update(test_metrics)
        write_json(summary_path, summary)
    table = _results_table(out_dir, selection)
    write_csv(table, out_dir / "benchmark_results.csv")
    comparison = table[table["run_id"].isin({s["run_id"] for s in runs})].sort_values("best_val_score", ascending=False)
    write_csv(comparison, out_dir / "model_comparison.csv")
    charts = write_seg_charts(out_dir, comparison, selection, tables["test"], train_videos)
    _write_report(out_dir, comparison, selection, classifier, by_group, charts)
    try:
        from .tato_export import export_tato

        export_tato(out_dir)
    except Exception as exc:  # không để lỗi xuất TATO làm mất kết quả chính
        warn(f"Xuất TATO lỗi: {type(exc).__name__}: {exc}")
    log(f"TEST: mean Dice {test_metrics['test_mean_dice']:.4f} | dây thanh trái {test_metrics['test_dice_left_fold']:.3f} "
        f"| phải {test_metrics['test_dice_right_fold']:.3f} | thanh môn {test_metrics['test_dice_glottis']:.3f}")
    if "test" in classifier:
        log(f"Chẩn đoán CHỈ từ số đo phân đoạn (mức video, test): macro-F1 {classifier['test']['macro_f1']:.3f}")
    return selection


# ------------------------------------------------------------------------------ biểu đồ + báo cáo
def write_seg_charts(out_dir: Path, comparison: pd.DataFrame, selection: dict, test_videos: pd.DataFrame,
                     train_videos: pd.DataFrame) -> dict[str, Path]:
    try:
        from .charts import SERIES, TEXT_2, _pyplot, _save
        plt = _pyplot()
    except ImportError:
        return {}
    folder = out_dir / "charts"
    folder.mkdir(parents=True, exist_ok=True)
    paths = {}
    label = lambda r: f"{r['backbone']} · {'CLAHE' if r['clahe'] else 'không CLAHE'}"  # noqa: E731

    table = comparison.head(16).iloc[::-1]
    fig, ax = plt.subplots(figsize=(9, 0.45 * len(table) + 1.6))
    y = np.arange(len(table))
    ax.barh(y, table["best_val_score"], color=[SERIES[1] if s else SERIES[0] for s in table["selected"]])
    for yi, v in zip(y, table["best_val_score"]):
        ax.text(v + 0.005, yi, f"{v:.3f}", va="center", fontsize=8, color=TEXT_2)
    ax.set_yticks(y, [label(r) for _, r in table.iterrows()])
    ax.set_xlim(0, 1.05)
    ax.set_xlabel("Val mean Dice (2 dây thanh + thanh môn) — cam = model được chọn")
    ax.set_title("Xếp hạng model phân đoạn theo validation")
    ax.grid(axis="y", visible=False)
    paths["xep_hang"] = _save(fig, folder / "1_xep_hang_phan_doan.png")

    best_runs = comparison.drop_duplicates("model")
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.6))
    for i, (_, run) in enumerate(best_runs.iterrows()):
        history_path = Path(run["run_dir"]) / "history.csv"
        if not history_path.exists():
            continue
        h = pd.read_csv(history_path)
        color = SERIES[i % len(SERIES)]
        axes[0].plot(h["epoch"], h["train_loss"], color=color, lw=1.6, label=f"{run['backbone']} train")
        axes[0].plot(h["epoch"], h["val_loss"], color=color, lw=1.6, ls="--", label=f"{run['backbone']} val")
        axes[1].plot(h["epoch"], h["val_mean_dice"], color=color, lw=1.8, label=run["backbone"])
        axes[1].plot(h["epoch"], h["val_dice_glottis"], color=color, lw=1, ls=":")
    axes[0].set_title("Train loss (liền) vs val loss (đứt) — CE + Dice")
    axes[0].set_xlabel("epoch")
    axes[0].legend(fontsize=7, ncol=2)
    axes[1].set_title("Val mean Dice (liền) và Dice thanh môn (chấm)")
    axes[1].set_xlabel("epoch")
    axes[1].set_ylim(0, 1)
    axes[1].legend(fontsize=8)
    paths["loss"] = _save(fig, folder / "2_loss_va_dice.png")

    fig, ax = plt.subplots(figsize=(7.5, 4.6))
    for i, (_, run) in enumerate(comparison.iterrows()):
        ax.scatter(run.get("gmacs") or np.nan, run["best_val_score"], s=40 + 8 * float(run.get("params_m") or 1),
                   color=SERIES[list(SEG_MODELS).index(run["model"]) % len(SERIES)], alpha=0.85,
                   edgecolor="black" if run["selected"] else "none")
    for i, model in enumerate(SEG_MODELS):
        ax.scatter([], [], color=SERIES[i], label=seg_display_name(model))
    ax.set_xlabel("GMACs / ảnh (chi phí tính toán)")
    ax.set_ylabel("Val mean Dice")
    ax.set_title("Chất lượng vs chi phí (cỡ chấm = số tham số)")
    ax.legend(fontsize=8)
    paths["chi_phi"] = _save(fig, folder / "3_dice_vs_chi_phi.png")

    tm = selection["test_metrics"]
    fig, ax = plt.subplots(figsize=(7.5, 3.8))
    names = ["left_fold", "right_fold", "glottis"]
    vi = ["Dây thanh trái ảnh", "Dây thanh phải ảnh", "Thanh môn"]
    x = np.arange(3)
    ax.bar(x - 0.2, [selection["val_metrics"][f"val_dice_{n}"] for n in names], 0.4, color=SERIES[0], label="Validation")
    ax.bar(x + 0.2, [tm[f"test_dice_{n}"] for n in names], 0.4, color=SERIES[1], label="Test")
    ax.set_xticks(x, vi)
    ax.set_ylim(0, 1)
    ax.set_ylabel("Dice")
    ax.set_title(f"Dice từng vùng — {selection['backbone']}")
    ax.legend()
    paths["dice_vung"] = _save(fig, folder / "4_dice_tung_vung.png")

    show = ["min_glottis_ratio", "closed_fraction", "amplitude_asymmetry", "hourglass_fraction"]
    fig, axes = plt.subplots(1, len(show), figsize=(15, 3.8))
    for ax, feature in zip(axes, show):
        data_train = [train_videos.loc[train_videos["label"] == g, feature].dropna() for g in GROUPS]
        data_test = [test_videos.loc[test_videos["label"] == g, feature].dropna() for g in GROUPS]
        positions = np.arange(len(GROUPS))
        ax.boxplot(data_train, positions=positions - 0.18, widths=0.3, patch_artist=True,
                   boxprops={"facecolor": SERIES[0], "alpha": 0.6}, medianprops={"color": "black"}, showfliers=False)
        ax.boxplot(data_test, positions=positions + 0.18, widths=0.3, patch_artist=True,
                   boxprops={"facecolor": SERIES[1], "alpha": 0.6}, medianprops={"color": "black"}, showfliers=False)
        ax.set_xticks(positions, ["Bình thường", "Chức năng", "Thực thể"], fontsize=8)
        ax.set_title(feature, fontsize=10)
    axes[0].plot([], [], color=SERIES[0], lw=6, label="train (mask thật)")
    axes[0].plot([], [], color=SERIES[1], lw=6, label="test (mask dự đoán)")
    axes[0].legend(fontsize=7)
    fig.suptitle("Số đo thanh môn theo nhóm bệnh (mỗi điểm = 1 video)", fontsize=12, fontweight="bold")
    paths["dau_hieu"] = _save(fig, folder / "5_dau_hieu_theo_nhom.png")
    return paths


def _write_report(out_dir: Path, comparison: pd.DataFrame, selection: dict, classifier: dict, by_group: pd.DataFrame,
                  charts: dict) -> None:
    tm, vm = selection["test_metrics"], selection["val_metrics"]
    cols = ["backbone", "clahe", "params_m", "gmacs", "best_val_score", "val_dice_left_fold", "val_dice_right_fold",
            "val_dice_glottis", "val_glottis_area_corr", "best_epoch", "mean_epoch_seconds"]
    lines = [
        "# Phân đoạn thanh môn BAGLS-VF — so sánh model", "",
        f"Tạo lúc {iso_now()}. Chọn model theo **validation** (val mean Dice), test chỉ đo 1 lần sau khi chọn.", "",
        f"**Model được chọn:** {selection['backbone']} (CLAHE {'on' if selection['clahe'] else 'off'}) — `{selection['run_id']}`", "",
        "| | Dây thanh trái ảnh (xanh lá) | Dây thanh phải ảnh (xanh dương) | Thanh môn (đỏ) | Mean Dice | Tương quan diện tích thanh môn |",
        "|---|---|---|---|---|---|",
        f"| Validation | {vm['val_dice_left_fold']:.3f} | {vm['val_dice_right_fold']:.3f} | {vm['val_dice_glottis']:.3f} | "
        f"{vm['val_mean_dice']:.3f} | {vm['val_glottis_area_corr']:.3f} |",
        f"| Test | {tm['test_dice_left_fold']:.3f} | {tm['test_dice_right_fold']:.3f} | {tm['test_dice_glottis']:.3f} | "
        f"{tm['test_mean_dice']:.3f} | {tm['test_glottis_area_corr']:.3f} |", "",
        "## Bảng xếp hạng", "", _markdown(comparison[[c for c in cols if c in comparison]]), "",
        "## Từ mask đến chẩn đoán", "",
        "Số đo mức video (gộp frame) so với khoảng tham chiếu của **video bình thường trong train** (`reference_ranges.json`).",
        "Ý nghĩa theo bài giảng: thanh quản bình thường khép kín khi phát âm, hai dây thanh cân đối (tr.35); tổn thương "
        "thực thể (tr.37) làm khe thanh môn hình đồng hồ cát / bờ không đều.", "",
        _markdown(by_group.reset_index()), "",
    ]
    if "test" in classifier:
        lines += ["Chẩn đoán nhóm bệnh **chỉ từ số đo phân đoạn** (hồi quy logistic mức video, fit trên mask thật của "
                  f"train): val macro-F1 {classifier.get('val', {}).get('macro_f1', float('nan')):.3f}, "
                  f"test macro-F1 {classifier['test']['macro_f1']:.3f} ({classifier['test']['videos']} video).", ""]
    for name, path in charts.items():
        lines += [f"![{name}](charts/{Path(path).name})", ""]
    lines += ["Ngưỡng dấu hiệu là thống kê tham khảo, chưa thẩm định lâm sàng; không thay thế chẩn đoán của bác sĩ."]
    (out_dir / "comparison.md").write_text("\n".join(lines), encoding="utf-8")


def _markdown(frame: pd.DataFrame) -> str:
    def cell(value):
        if isinstance(value, float):
            return "—" if not np.isfinite(value) else f"{value:.4g}"
        return str(value)

    lines = ["| " + " | ".join(map(str, frame.columns)) + " |", "|" + "|".join("---" for _ in frame.columns) + "|"]
    lines += ["| " + " | ".join(cell(v) for v in row) + " |" for row in frame.itertuples(index=False)]
    return "\n".join(lines)


def main(argv=None) -> None:
    setup_console()
    args = build_parser().parse_args(argv)
    cfg = load_seg_config(resolve_cli_path(args.config)) if args.config else load_seg_config()
    out_dir = resolve_cli_path(args.out) if args.out else DEFAULT_OUTPUT_DIR / cfg["task"]
    out_dir.mkdir(parents=True, exist_ok=True)
    prepared = resolve_cli_path(args.prepared_dir)
    if args.rebuild_manifest:
        from .seg_data import build_seg_manifest

        build_seg_manifest(cfg, prepared, overwrite=True)
    frame, task_info, sha = load_seg_manifest(prepared, cfg)
    device = pick_device(args.device)
    if not args.finalize_only:
        run_matrix(args, out_dir, frame, task_info, sha, device, cfg)
    if not (args.plan or args.no_finalize):
        finalize(out_dir, frame, device, clahe_allowed=tuple(c == "on" for c in args.clahe), manifest_sha=sha)


if __name__ == "__main__":
    os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
    main()
