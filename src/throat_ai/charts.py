"""Biểu đồ PNG cho báo cáo benchmark (ghi vào <out>/charts/, nhúng vào comparison.md).

    1_xep_hang.png        val vs test macro-F1 theo cấu hình (trung bình ± std qua seed)
    2_learning_curves.png val macro-F1 + train accuracy theo epoch, mỗi kiến trúc 1 ô (run tốt nhất)
    3_f1_vs_thoi_gian.png chất lượng vs chi phí (phút/epoch) — chọn model cân bằng
    4_ablation.png        Δ macro-F1 do CLAHE và do duplicate (so sánh cặp)
    5_confusion_test.png  confusion matrix test (chuẩn hóa theo hàng) của model được chọn
    6_loss_curves.png     train loss vs val loss theo epoch, mỗi kiến trúc 1 ô (run tốt nhất)
"""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd

SURFACE = "#fcfcfb"
TEXT = "#0b0b0b"
TEXT_2 = "#52514e"
GRID = "#e4e3df"
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
NEUTRAL = "#8a8984"


def _pyplot():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "axes.edgecolor": GRID, "axes.labelcolor": TEXT_2, "axes.titlecolor": TEXT, "axes.titlesize": 12,
        "axes.titleweight": "bold", "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8,
        "axes.axisbelow": True, "axes.spines.top": False, "axes.spines.right": False,
        "xtick.color": TEXT_2, "ytick.color": TEXT_2, "font.size": 10, "legend.frameon": False,
    })
    return plt


def _label(row) -> str:
    clahe = "CLAHE" if row["clahe"] else "không CLAHE"
    return f"{row['backbone']} · {clahe} · {row['dup_policy']}"


def _save(fig, path: Path) -> Path:
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    import matplotlib.pyplot as plt

    plt.close(fig)
    return path


def leaderboard(comparison: pd.DataFrame, path: Path):
    plt = _pyplot()
    table = comparison[~comparison["leaky"]].head(20).iloc[::-1]
    if table.empty:
        return None
    labels = [_label(r) for _, r in table.iterrows()]
    y = np.arange(len(table))
    has_test = "test_macro_f1_mean" in table and table["test_macro_f1_mean"].notna().any()
    height = 0.38 if has_test else 0.6
    fig, ax = plt.subplots(figsize=(9, 0.45 * len(table) + 1.6))
    ax.barh(y + (height / 2 if has_test else 0), table["val_macro_f1_mean"], height=height, color=SERIES[0],
            xerr=table.get("val_macro_f1_std"), error_kw={"ecolor": TEXT_2, "lw": 1}, label="Validation (dùng để chọn)")
    if has_test:
        ax.barh(y - height / 2, table["test_macro_f1_mean"], height=height, color=SERIES[1],
                xerr=table.get("test_macro_f1_std"), error_kw={"ecolor": TEXT_2, "lw": 1}, label="Test (chỉ báo cáo)")
    for yi, value in zip(y, table["val_macro_f1_mean"]):
        ax.text(value + 0.01, yi + (height / 2 if has_test else 0), f"{value:.3f}", va="center", fontsize=8, color=TEXT_2)
    ax.set_yticks(y, labels)
    ax.set_xlim(0, 1.05)
    ax.set_xlabel("Macro-F1 (trung bình qua seed, thanh lỗi = độ lệch chuẩn)")
    ax.set_title("Xếp hạng cấu hình theo validation macro-F1")
    ax.grid(axis="y", visible=False)
    ax.legend(loc="lower right")
    return _save(fig, path)


def learning_curves(df: pd.DataFrame, epochs: pd.DataFrame, path: Path):
    plt = _pyplot()
    done = df[(df["status"] == "completed") & (~df["leaky"].fillna(False).astype(bool))]
    if done.empty or epochs.empty:
        return None
    best = done.sort_values("best_val_score", ascending=False).drop_duplicates("model")
    n = len(best)
    cols = min(3, n)
    rows = math.ceil(n / cols)
    from matplotlib.ticker import MaxNLocator

    fig, axes = plt.subplots(rows, cols, figsize=(max(4.2 * cols, 9), 3.2 * rows + 0.6), squeeze=False, sharey=True)
    for ax, (_, run) in zip(axes.flat, best.iterrows()):
        hist = epochs[epochs["run_id"] == run["run_id"]]
        ax.plot(hist["epoch"], hist["val_macro_f1"], color=SERIES[0], lw=2, marker="o", ms=3, label="val macro-F1")
        if pd.notna(run.get("best_epoch")):
            ax.axvline(run["best_epoch"], color=NEUTRAL, lw=1, ls="--")
        ax.plot(hist["epoch"], hist["train_accuracy"], color=SERIES[1], lw=1.5, label="train accuracy")
        ax.set_title(f"{run['backbone']} ({'CLAHE' if run['clahe'] else 'không CLAHE'})", fontsize=10)
        ax.set_ylim(0, 1)
        ax.set_xlabel("epoch")
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    for ax in list(axes.flat)[n:]:
        ax.set_visible(False)
    for row_axes in axes:
        row_axes[0].set_ylabel("điểm (0–1)")
    handles = [plt.Line2D([], [], color=SERIES[0], lw=2, marker="o", ms=3),
               plt.Line2D([], [], color=SERIES[1], lw=1.5),
               plt.Line2D([], [], color=NEUTRAL, lw=1, ls="--")]
    fig.legend(handles, ["val macro-F1", "train accuracy (cách xa val = overfit)", "epoch tốt nhất (early stopping)"],
               loc="lower center", ncol=3, bbox_to_anchor=(0.5, -0.01))
    fig.suptitle("Learning curve — run tốt nhất của mỗi kiến trúc", fontweight="bold", color=TEXT)
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def quality_vs_cost(comparison: pd.DataFrame, path: Path):
    plt = _pyplot()
    table = comparison[~comparison["leaky"]].dropna(subset=["mean_epoch_seconds", "val_macro_f1_mean"])
    if table.empty:
        return None
    best = table.sort_values("best_val_score_mean", ascending=False).drop_duplicates("model")
    fig, ax = plt.subplots(figsize=(8, 5))
    minutes = best["mean_epoch_seconds"] / 60
    size = 40 + 8 * best["params_m"].fillna(1)
    ax.scatter(minutes, best["val_macro_f1_mean"], s=size, color=SERIES[0], edgecolor=SURFACE, linewidth=2, zorder=3)
    for x, y, name in zip(minutes, best["val_macro_f1_mean"], best["backbone"]):
        ax.annotate(name, (x, y), xytext=(6, 4), textcoords="offset points", fontsize=8, color=TEXT)
    ax.set_xlabel("Phút / epoch (trung bình)")
    ax.set_ylabel("Val macro-F1")
    ax.set_title("Chất lượng vs chi phí\ncấu hình tốt nhất của mỗi kiến trúc · cỡ chấm = số tham số", fontsize=11)
    ax.set_ylim(0, 1.05)
    ax.set_xlim(0, float(minutes.max()) * 1.25 + 1e-6)
    return _save(fig, path)


def ablation(clahe: pd.DataFrame, dup: pd.DataFrame, path: Path):
    panels = [(p, t) for p, t in ((clahe, "CLAHE: bật − tắt"), (dup, "Duplicate: dedup − keep")) if not p.empty]
    if not panels:
        return None
    plt = _pyplot()
    fig, axes = plt.subplots(1, len(panels), figsize=(5.5 * len(panels), 3.8), squeeze=False)
    for ax, (pairs, title) in zip(axes[0], panels):
        summary = pairs.groupby("model")["delta_val_macro_f1"].agg(["mean", "std", "count"]).sort_values("mean")
        y = np.arange(len(summary))
        colors = [SERIES[0] if v >= 0 else SERIES[1] for v in summary["mean"]]
        ax.barh(y, summary["mean"], color=colors, xerr=summary["std"].fillna(0), error_kw={"ecolor": TEXT_2, "lw": 1})
        ax.axvline(0, color=TEXT_2, lw=1)
        for yi, value in zip(y, summary["mean"]):
            ax.text(value, yi, f" {value:+.3f} ", va="center", ha="left" if value >= 0 else "right", fontsize=8, color=TEXT)
        limit = max(0.02, float(summary["mean"].abs().max() + summary["std"].fillna(0).max()) * 1.4)
        ax.set_xlim(-limit, limit)
        ax.set_yticks(y, [f"{m} (n={int(c)})" for m, c in zip(summary.index, summary["count"])])
        ax.set_xlabel("Δ val macro-F1  (>0: tốt hơn)")
        ax.set_title(title)
        ax.grid(axis="y", visible=False)
    return _save(fig, path)


def confusion(selection: dict | None, path: Path):
    if not selection:
        return None
    source = Path(selection["run_dir"]) / "test_confusion.csv"
    if not source.exists():
        return None
    plt = _pyplot()
    table = pd.read_csv(source).set_index("true\\pred")
    counts = table.to_numpy(dtype=float)
    normalized = counts / np.maximum(counts.sum(1, keepdims=True), 1)
    n = len(table)
    fig, ax = plt.subplots(figsize=(max(1.0 * n + 3, 7), max(0.8 * n + 2, 5)))
    image = ax.imshow(normalized, cmap="Blues", vmin=0, vmax=1)
    for i in range(n):
        for j in range(n):
            ax.text(j, i, f"{int(counts[i, j])}", ha="center", va="center", fontsize=8,
                    color=SURFACE if normalized[i, j] > 0.55 else TEXT)
    ax.set_xticks(range(n), table.columns, rotation=45, ha="right")
    ax.set_yticks(range(n), table.index)
    ax.set_xlabel("Dự đoán")
    ax.set_ylabel("Nhãn thật")
    ax.grid(False)
    ax.set_title(f"Confusion matrix (test) — {selection.get('backbone')}\nsố = số ảnh, màu = tỷ lệ trong mỗi nhãn thật", fontsize=11)
    fig.colorbar(image, ax=ax, fraction=0.04, label="tỷ lệ trong nhãn thật")
    return _save(fig, path)


def loss_curves(df: pd.DataFrame, epochs: pd.DataFrame, path: Path):
    """Train loss vs val loss: val loss tăng trong khi train loss giảm = overfit."""
    plt = _pyplot()
    from matplotlib.ticker import MaxNLocator

    done = df[(df["status"] == "completed") & (~df["leaky"].fillna(False).astype(bool))]
    if done.empty or epochs.empty or "train_loss" not in epochs:
        return None
    val_column = "val_loss" if "val_loss" in epochs and epochs["val_loss"].notna().any() else "val_log_loss"
    best = done.sort_values("best_val_score", ascending=False).drop_duplicates("model")
    n = len(best)
    cols = min(3, n)
    rows = math.ceil(n / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(max(4.2 * cols, 9), 3.2 * rows + 0.6), squeeze=False)
    for ax, (_, run) in zip(axes.flat, best.iterrows()):
        hist = epochs[epochs["run_id"] == run["run_id"]].sort_values("epoch")
        ax.plot(hist["epoch"], hist["train_loss"], color=SERIES[0], lw=2, marker="o", ms=3)
        ax.plot(hist["epoch"], hist[val_column], color=SERIES[1], lw=2, marker="o", ms=3)
        if pd.notna(run.get("best_epoch")):
            ax.axvline(run["best_epoch"], color=NEUTRAL, lw=1, ls="--")
        lowest = hist.loc[hist[val_column].idxmin()] if hist[val_column].notna().any() else None
        if lowest is not None:
            ax.annotate(f"val thấp nhất {lowest[val_column]:.3f}", (lowest["epoch"], lowest[val_column]),
                        xytext=(8, 10), textcoords="offset points", ha="left", fontsize=8, color=TEXT_2)
        ax.set_title(f"{run['backbone']} ({'CLAHE' if run['clahe'] else 'không CLAHE'})", fontsize=10)
        top = float(np.nanmax(hist[["train_loss", val_column]].to_numpy()))
        ax.set_ylim(0, top * 1.25 if np.isfinite(top) and top > 0 else 1)
        ax.set_xlim(hist["epoch"].min() - 0.3, hist["epoch"].max() + 0.3)
        ax.set_xlabel("epoch")
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    for ax in list(axes.flat)[n:]:
        ax.set_visible(False)
    for row_axes in axes:
        row_axes[0].set_ylabel("loss")
    handles = [plt.Line2D([], [], color=SERIES[0], lw=2, marker="o", ms=3),
               plt.Line2D([], [], color=SERIES[1], lw=2, marker="o", ms=3),
               plt.Line2D([], [], color=NEUTRAL, lw=1, ls="--")]
    labels = ["train loss", "val loss (cùng công thức cross-entropy có trọng số lớp)"
              if val_column == "val_loss" else "val log-loss (không trọng số — run cũ)", "epoch tốt nhất (early stopping)"]
    fig.legend(handles, labels, loc="lower center", ncol=3, bbox_to_anchor=(0.5, -0.01))
    fig.suptitle("Train loss vs val loss — run tốt nhất của mỗi kiến trúc", fontweight="bold", color=TEXT)
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def write_charts(out_dir, df, epochs, comparison, clahe, dup, selection) -> dict[str, Path]:
    """Vẽ mọi biểu đồ có đủ dữ liệu. Thiếu matplotlib -> bỏ qua (trả về {})."""
    try:
        _pyplot()
    except ImportError:
        return {}
    folder = Path(out_dir) / "charts"
    folder.mkdir(parents=True, exist_ok=True)
    jobs = {
        "Xếp hạng cấu hình": lambda: leaderboard(comparison, folder / "1_xep_hang.png") if not comparison.empty else None,
        "Learning curve": lambda: learning_curves(df, epochs, folder / "2_learning_curves.png"),
        "Chất lượng vs chi phí": lambda: quality_vs_cost(comparison, folder / "3_f1_vs_thoi_gian.png") if not comparison.empty else None,
        "Ablation CLAHE / duplicate": lambda: ablation(clahe, dup, folder / "4_ablation.png"),
        "Confusion matrix (test)": lambda: confusion(selection, folder / "5_confusion_test.png"),
        "Train loss vs val loss": lambda: loss_curves(df, epochs, folder / "6_loss_curves.png"),
    }
    result = {}
    for title, job in jobs.items():
        try:
            path = job()
        except Exception as exc:  # 1 biểu đồ lỗi không làm hỏng báo cáo
            print(f"Không vẽ được '{title}': {type(exc).__name__}: {exc}", flush=True)
            path = None
        if path:
            result[title] = path
    return result
