"""Tổng hợp kết quả benchmark.

    benchmark_results.csv  — 1 dòng cho MỖI lần train (mọi model/seed/CLAHE/duplicate, kể cả run lỗi/bỏ qua)
    all_epochs.csv         — lịch sử từng epoch của mọi run
    model_comparison.csv   — gộp theo cấu hình (trung bình ± độ lệch chuẩn qua các seed), xếp hạng theo VALIDATION
    ablation_clahe.csv     — so sánh cặp: cùng model/seed/duplicate, CLAHE bật vs tắt
    ablation_duplicates.csv— so sánh cặp: dedup vs keep (cùng val/test) và naive (leaky) vs dedup
    comparison.md          — báo cáo đọc nhanh (tiếng Việt)
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

from .utils import iso_now, read_json, write_csv

FRONT_COLUMNS = [
    "run_id", "status", "stop_reason", "abort_reason", "selected", "task", "config_id", "model", "backbone",
    "clahe", "dup_policy", "leaky", "seed", "batch_size", "accumulation", "effective_batch_size", "precision",
    "precision_effective", "workers", "resolution", "params_m", "gmacs", "lr", "head_lr_mult", "weight_decay",
    "epochs_max", "patience", "monitor_effective", "epochs_run", "best_epoch", "best_val_score",
    "val_macro_f1", "val_balanced_accuracy", "val_accuracy", "val_weighted_f1", "val_macro_precision",
    "val_macro_recall", "val_macro_auroc", "val_log_loss", "val_ece",
    "val_patient_macro_f1", "val_patient_accuracy", "val_patient_balanced_accuracy",
    "test_macro_f1", "test_balanced_accuracy", "test_accuracy", "test_weighted_f1", "test_macro_precision",
    "test_macro_recall", "test_macro_auroc", "test_log_loss", "test_ece",
    "test_patient_macro_f1", "test_patient_accuracy", "test_patient_balanced_accuracy",
    "n_train_images", "n_val_images", "n_test_images", "n_train_patients", "n_val_patients", "n_test_patients",
    "setup_seconds", "full_run_time_s", "mean_epoch_seconds", "train_images_per_s_mean", "gpu_mem_peak_gb",
    "full_run_gpu_energy_wh", "full_run_gpu_power_mean_w", "full_run_gpu_temp_max_c", "throttled", "throttle_seconds",
    "task_quality_metric", "task_quality_value", "target_reached", "target_epoch", "target_time_to_target_s",
    "target_energy_to_target_wh", "thermal_start_state", "hardware_device", "hardware_gpu_model",
    "hardware_torch_version", "session_id", "machine_id", "started_at", "finished_at", "run_dir",
]
CONFIG_KEYS = ["config_id", "model", "backbone", "batch_size", "accumulation", "precision_effective", "resolution",
               "clahe", "dup_policy"]
PAIR_KEYS = CONFIG_KEYS + ["seed"]
COMPARE_METRICS = ["best_val_score", "val_macro_f1", "val_balanced_accuracy", "val_macro_auroc", "val_patient_macro_f1",
                   "test_macro_f1", "test_balanced_accuracy", "test_accuracy", "test_macro_auroc", "test_patient_macro_f1"]


def load_summaries(out_dir) -> list[dict]:
    summaries = []
    for path in sorted(Path(out_dir).glob("runs/*/summary.json")):
        try:
            summaries.append(read_json(path))
        except (json.JSONDecodeError, OSError):
            continue
    return summaries


def runs_frame(out_dir, selection: dict | None = None) -> pd.DataFrame:
    frame = pd.DataFrame(load_summaries(out_dir))
    if frame.empty:
        return frame
    frame["selected"] = frame["run_id"] == (selection or {}).get("run_id")
    front = [c for c in FRONT_COLUMNS if c in frame.columns]
    rest = sorted(c for c in frame.columns if c not in front)
    return frame[front + rest]


def epochs_frame(runs: pd.DataFrame) -> pd.DataFrame:
    parts = []
    for row in runs.itertuples():
        path = Path(row.run_dir) / "history.csv"
        if path.exists():
            history = pd.read_csv(path)
            for column in ("run_id", "model", "clahe", "dup_policy", "seed", "config_id"):
                history.insert(0, column, getattr(row, column))
            parts.append(history)
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()


def _completed(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty or "status" not in df:
        return df.iloc[0:0]
    done = df[df["status"] == "completed"].copy()
    if "finished_at" in done:
        done = done.sort_values("finished_at")
    present = [k for k in PAIR_KEYS if k in done]
    return done.drop_duplicates(present, keep="last")  # chạy lại cùng cấu hình (--force): giữ lần mới nhất


def model_comparison(df: pd.DataFrame) -> pd.DataFrame:
    done = _completed(df)
    if done.empty:
        return pd.DataFrame()
    spec = {"n_runs": ("run_id", "count")}
    for metric in COMPARE_METRICS:
        if metric in done:
            spec[f"{metric}_mean"] = (metric, "mean")
            spec[f"{metric}_std"] = (metric, "std")
    for column, how in (("params_m", "first"), ("gmacs", "first"), ("epochs_run", "mean"), ("best_epoch", "mean"),
                        ("mean_epoch_seconds", "mean"), ("full_run_time_s", "mean"), ("full_run_gpu_energy_wh", "mean"),
                        ("full_run_gpu_temp_max_c", "max"), ("target_reached", "mean")):
        if column in done:
            spec[column if how != "max" else "peak_gpu_temp_c"] = (column, how)
    table = done.groupby(CONFIG_KEYS, dropna=False).agg(**spec).reset_index()
    table["leaky"] = table["dup_policy"] == "naive"
    table = table.sort_values(["leaky", "best_val_score_mean"], ascending=[True, False]).reset_index(drop=True)
    rank = np.where(~table["leaky"], np.arange(1, len(table) + 1), np.nan)
    table.insert(0, "rank_by_validation", rank)
    return table


def paired_effect(df: pd.DataFrame, factor: str, treated, control, metrics=("val_macro_f1", "test_macro_f1")) -> pd.DataFrame:
    """So sánh cặp cùng cấu hình/seed, chỉ khác `factor`. delta = treated - control."""
    done = _completed(df)
    if done.empty or factor not in done:
        return pd.DataFrame()
    if factor != "dup_policy" and "leaky" in done:
        done = done[~done["leaky"].fillna(False).astype(bool)]  # run leaky không dùng để đánh giá CLAHE/khác
    keys = [k for k in PAIR_KEYS if k != factor and k in done]
    a = done[done[factor] == treated].set_index(keys)
    b = done[done[factor] == control].set_index(keys)
    common = a.index.intersection(b.index)
    if len(common) == 0:
        return pd.DataFrame()
    pairs = pd.DataFrame(index=common).reset_index()
    for metric in metrics:
        if metric in done:
            pairs[f"{metric}_{treated}"] = a.loc[common, metric].to_numpy()
            pairs[f"{metric}_{control}"] = b.loc[common, metric].to_numpy()
            pairs[f"delta_{metric}"] = pairs[f"{metric}_{treated}"] - pairs[f"{metric}_{control}"]
    pairs.insert(0, "comparison", f"{factor}: {treated} - {control}")
    return pairs


# ------------------------------------------------------------------------------------------ markdown
def _fmt(value, digits: int = 3) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return "" if value is None else str(value)
    return "" if not math.isfinite(number) else f"{number:.{digits}f}"


def _pm(mean, std, digits: int = 3) -> str:
    text = _fmt(mean, digits)
    if text and _fmt(std, digits):
        text += f" ± {_fmt(std, digits)}"
    return text or "—"


def _table(headers: list[str], rows: list[list]) -> str:
    lines = ["| " + " | ".join(headers) + " |", "|" + "|".join("---" for _ in headers) + "|"]
    lines += ["| " + " | ".join(str(c) for c in row) + " |" for row in rows]
    return "\n".join(lines)


def _minutes(seconds) -> str:
    return _fmt(float(seconds) / 60.0, 1) if seconds is not None and _fmt(seconds) else ""


def render_markdown(df: pd.DataFrame, comparison: pd.DataFrame, clahe: pd.DataFrame, dup: pd.DataFrame,
                    leaky: pd.DataFrame, selection: dict | None, task: dict | None, prepare_summary: dict | None,
                    out_dir: Path) -> str:
    task = task or {}
    lines = [f"# So sánh model — task `{task.get('task', '?')}`", "",
             f"Cập nhật: {iso_now()} · {task.get('description', '')}", ""]
    counts = df["status"].value_counts().to_dict() if not df.empty else {}
    lines += [f"Tổng số lần train: **{len(df)}** — " + ", ".join(f"{k}: {v}" for k, v in counts.items()), ""]
    lines += ["> Xếp hạng & chọn model **chỉ theo VALIDATION**. Cột test chỉ để báo cáo sau khi đã chốt; "
              "không đổi lựa chọn dựa trên test (nếu đổi, test không còn là đánh giá độc lập).", ""]

    real = comparison[~comparison["leaky"]] if not comparison.empty else comparison
    if not real.empty:
        lines += ["## 1. Bảng xếp hạng cấu hình (trung bình ± độ lệch chuẩn qua các seed)", ""]
        rows = []
        for r in real.itertuples():
            rows.append([int(r.rank_by_validation), r.backbone, "bật" if r.clahe else "tắt", r.dup_policy, int(r.n_runs),
                         _pm(getattr(r, "val_macro_f1_mean", None), getattr(r, "val_macro_f1_std", None)),
                         _pm(getattr(r, "val_patient_macro_f1_mean", None), getattr(r, "val_patient_macro_f1_std", None)),
                         _pm(getattr(r, "test_macro_f1_mean", None), getattr(r, "test_macro_f1_std", None)),
                         _fmt(getattr(r, "test_balanced_accuracy_mean", None)), _fmt(getattr(r, "test_macro_auroc_mean", None)),
                         _fmt(getattr(r, "params_m", None), 1), _minutes(getattr(r, "mean_epoch_seconds", None)),
                         _fmt(getattr(r, "full_run_gpu_energy_wh", None), 1)])
        lines += [_table(["#", "Model", "CLAHE", "Duplicate", "Số run", "Val macro-F1", "Val F1 bệnh nhân",
                          "Test macro-F1", "Test bal.acc", "Test AUROC", "Tham số (M)", "Phút/epoch", "GPU Wh"], rows), ""]
        best_per_model = real.sort_values("best_val_score_mean", ascending=False).drop_duplicates("model")
        lines += ["## 2. Cấu hình tốt nhất của từng kiến trúc", ""]
        lines += [_table(["Model", "CLAHE", "Duplicate", "Val macro-F1", "Test macro-F1", "Tham số (M)", "GMACs"],
                         [[r.backbone, "bật" if r.clahe else "tắt", r.dup_policy,
                           _pm(r.val_macro_f1_mean, r.val_macro_f1_std),
                           _pm(getattr(r, "test_macro_f1_mean", None), getattr(r, "test_macro_f1_std", None)),
                           _fmt(r.params_m, 1), _fmt(getattr(r, "gmacs", None), 2)] for r in best_per_model.itertuples()]), ""]

    def effect_section(title: str, pairs: pd.DataFrame, treated: str, control: str, note: str) -> list[str]:
        out = [title, "", note, ""]
        if pairs.empty:
            return out + ["_Chưa có cặp run nào để so sánh (cần chạy cả hai lựa chọn với cùng model/seed)._", ""]
        rows = []
        for model, group in pairs.groupby("model"):
            rows.append([model, len(group), _fmt(group["delta_val_macro_f1"].mean(), 4),
                         int((group["delta_val_macro_f1"] > 0).sum()),
                         _fmt(group["delta_test_macro_f1"].mean(), 4) if "delta_test_macro_f1" in group else ""])
        rows.append(["**Tất cả**", len(pairs), _fmt(pairs["delta_val_macro_f1"].mean(), 4),
                     int((pairs["delta_val_macro_f1"] > 0).sum()),
                     _fmt(pairs["delta_test_macro_f1"].mean(), 4) if "delta_test_macro_f1" in pairs else ""])
        return out + [_table(["Model", "Số cặp", f"Δ val macro-F1 ({treated} − {control})", "Số cặp tốt lên (val)",
                              "Δ test macro-F1"], rows), ""]

    lines += effect_section("## 3. Ảnh hưởng của CLAHE", clahe, "bật", "tắt",
                            "Cùng model, seed, chính sách duplicate; chỉ khác bật/tắt CLAHE (kênh L, clip 2.0, lưới 8×8).")
    lines += effect_section("## 4. Ảnh hưởng của duplicate (công bằng)", dup, "dedup", "keep",
                            "`dedup` bỏ bản trùng trong train; `keep` giữ bản trùng. Val/test GIỐNG HỆT nhau nên so sánh được.")
    if prepare_summary:
        lines += [f"Số liệu dữ liệu: {prepare_summary.get('exact_duplicate_groups', 0)} nhóm ảnh trùng chính xác cùng nhãn "
                  f"({prepare_summary.get('exact_duplicate_extra_copies', 0)} bản thừa); "
                  f"{prepare_summary.get('conflicting_duplicate_images_quarantined', 0)} ảnh trùng điểm ảnh nhưng khác nhãn đã bị cách ly; "
                  f"{prepare_summary.get('near_duplicate_pairs', 0)} cặp gần trùng (đã gom cùng tập).", ""]
    lines += ["### 4b. Không xử lý gì cả (naive, CÓ LEAKAGE — chỉ minh họa)", ""]
    if leaky.empty:
        lines += ["_Chưa chạy chính sách `naive`. Thêm `--dup-policy dedup keep naive` để xem điểm bị thổi phồng thế nào._", ""]
    else:
        lines += ["Chia ngẫu nhiên theo ảnh + giữ bản trùng: ảnh cùng bệnh nhân/ảnh trùng nằm ở cả train và test, "
                  "nên điểm test **cao giả tạo**. Δ dưới đây là phần 'điểm ảo' so với cách làm đúng.", ""]
        lines += [_table(["Model", "Số cặp", "Test macro-F1 naive", "Test macro-F1 dedup", "Điểm ảo (Δ)"],
                         [[m, len(g), _fmt(g.get("test_macro_f1_naive", pd.Series(dtype=float)).mean()),
                           _fmt(g.get("test_macro_f1_dedup", pd.Series(dtype=float)).mean()),
                           _fmt(g.get("delta_test_macro_f1", pd.Series(dtype=float)).mean(), 4)]
                          for m, g in leaky.groupby("model")]), ""]
        if prepare_summary:
            naive = prepare_summary.get("leakage_demo_naive_split", {})
            lines += [f"Trong split naive: {naive.get('test_images_same_patient_in_train', '?')}/{naive.get('test_images', '?')} "
                      f"ảnh test có cùng bệnh nhân trong train; {naive.get('test_images_exact_duplicate_in_train', '?')} ảnh test "
                      "trùng y hệt 1 ảnh train.", ""]

    lines += ["## 5. Model được chọn", ""]
    if not selection:
        lines += ["_Chưa có run hoàn thành để chọn._", ""]
    else:
        lines += [f"- Run: `{selection['run_id']}` — {selection.get('backbone')} · CLAHE {'bật' if selection.get('clahe') else 'tắt'}"
                  f" · duplicate `{selection.get('dup_policy')}` · seed {selection.get('seed')}",
                  f"- Quy tắc: {selection.get('rule')}",
                  f"- Validation {selection.get('monitor')}: {_fmt(selection.get('val_score'))} "
                  f"(trung bình cấu hình {_pm(selection.get('config_val_mean'), selection.get('config_val_std'))}, "
                  f"{selection.get('config_n_seeds')} seed)",
                  f"- Hiệu chuẩn nhiệt độ (fit trên validation): T = {_fmt(selection.get('temperature'))}, "
                  f"ECE val {_fmt(selection.get('val_ece_before'))} → {_fmt(selection.get('val_ece_after'))}",
                  f"- Checkpoint: `{selection.get('checkpoint')}`", ""]
        test = selection.get("test_metrics") or {}
        if test:
            lines += ["**Kết quả TEST (đo 1 lần sau khi chốt model):**", "",
                      _table(["Macro-F1", "Balanced acc", "Accuracy", "Macro AUROC", "F1 bệnh nhân", "ECE (sau hiệu chuẩn)"],
                             [[_fmt(test.get("test_macro_f1")), _fmt(test.get("test_balanced_accuracy")),
                               _fmt(test.get("test_accuracy")), _fmt(test.get("test_macro_auroc")),
                               _fmt(test.get("test_patient_macro_f1")), _fmt(selection.get("test_ece_after"))]]), ""]
            per_class = Path(selection["run_dir"]) / "test_per_class.csv"
            if per_class.exists():
                table = pd.read_csv(per_class)
                table = table[~table["class"].isin(["accuracy", "macro avg", "weighted avg"])]
                lines += ["Theo từng lớp (test):", "",
                          _table(["Lớp", "Precision", "Recall", "F1", "Số ảnh"],
                                 [[r["class"], _fmt(r["precision"]), _fmt(r["recall"]), _fmt(r["f1-score"]),
                                   int(r["support"])] for _, r in table.iterrows()]), ""]
                lines += [f"Confusion matrix: `{Path(selection['run_dir']) / 'test_confusion.csv'}`", ""]

    problems = df[df["status"].isin(["failed", "aborted", "skipped", "interrupted"])] if not df.empty else df
    if not problems.empty:
        lines += ["## 6. Run không hoàn thành", "",
                  _table(["Run", "Trạng thái", "Lý do"],
                         [[r.run_id, r.status, str(r.abort_reason)[:160]] for r in problems.itertuples()]), ""]
    lines += ["## Ghi chú", "",
              "- Macro-F1 = trung bình F1 các lớp (lớp hiếm quan trọng ngang lớp phổ biến) — metric chính.",
              "- 'F1 bệnh nhân': trung bình xác suất các ảnh của cùng bệnh nhân rồi mới dự đoán.",
              "- Kết quả là nghiên cứu, chưa thẩm định lâm sàng; cần tập kiểm định ngoài (bệnh viện/thiết bị khác).",
              f"- File chi tiết: `{out_dir / 'benchmark_results.csv'}`, `{out_dir / 'model_comparison.csv'}`, "
              f"`{out_dir / 'all_epochs.csv'}`.", ""]
    return "\n".join(lines)


def write_reports(out_dir, task: dict | None = None, selection: dict | None = None,
                  prepare_summary: dict | None = None) -> dict:
    out_dir = Path(out_dir)
    if selection is None and (out_dir / "selection.json").exists():
        selection = read_json(out_dir / "selection.json")
    df = runs_frame(out_dir, selection)
    paths = {}
    if df.empty:
        return paths
    paths["benchmark_results"] = write_csv(df, out_dir / "benchmark_results.csv")
    epochs = epochs_frame(df)
    if not epochs.empty:
        paths["all_epochs"] = write_csv(epochs, out_dir / "all_epochs.csv")
    comparison = model_comparison(df)
    if not comparison.empty:
        paths["model_comparison"] = write_csv(comparison, out_dir / "model_comparison.csv")
    clahe = paired_effect(df, "clahe", True, False)
    dup = paired_effect(df, "dup_policy", "dedup", "keep")
    leaky = paired_effect(df, "dup_policy", "naive", "dedup")
    if not clahe.empty:
        clahe["comparison"] = "clahe: bật - tắt"
        paths["ablation_clahe"] = write_csv(clahe, out_dir / "ablation_clahe.csv")
    if not dup.empty or not leaky.empty:
        paths["ablation_duplicates"] = write_csv(pd.concat([dup, leaky], ignore_index=True), out_dir / "ablation_duplicates.csv")
    markdown = render_markdown(df, comparison, clahe, dup, leaky, selection, task, prepare_summary, out_dir)
    from .charts import write_charts

    charts = write_charts(out_dir, df, epochs, comparison, clahe, dup, selection)
    if charts:
        block = ["## Biểu đồ", ""] + [f"### {title}\n\n![{title}](charts/{path.name})\n" for title, path in charts.items()]
        head, sep, rest = markdown.partition("## 1.")
        markdown = head + "\n".join(block) + "\n" + sep + rest if sep else markdown + "\n" + "\n".join(block)
        paths["charts"] = out_dir / "charts"
    else:
        markdown += "\n_Không vẽ được biểu đồ: cài thêm `pip install matplotlib`._\n"
    (out_dir / "comparison.md").write_text(markdown, encoding="utf-8")
    paths["comparison_md"] = out_dir / "comparison.md"
    return paths
