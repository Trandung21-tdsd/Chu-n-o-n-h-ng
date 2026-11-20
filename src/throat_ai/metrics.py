"""Metric phân loại (mức ảnh + mức bệnh nhân), báo cáo từng lớp, confusion matrix, hiệu chuẩn nhiệt độ."""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
    roc_auc_score,
)

METRIC_NAMES = ["accuracy", "balanced_accuracy", "macro_f1", "weighted_f1", "macro_precision", "macro_recall",
                "macro_auroc", "log_loss", "ece"]


def _normalize(probs: np.ndarray) -> np.ndarray:
    probs = np.clip(np.asarray(probs, dtype=np.float64), 1e-12, 1.0)
    return probs / probs.sum(axis=1, keepdims=True)


def expected_calibration_error(y_true: np.ndarray, probs: np.ndarray, bins: int = 15) -> float:
    confidence = probs.max(1)
    correct = probs.argmax(1) == y_true
    edges = np.linspace(0.0, 1.0, bins + 1)
    ece = 0.0
    for low, high in zip(edges[:-1], edges[1:]):
        mask = (confidence > low) & (confidence <= high)
        if mask.any():
            ece += mask.mean() * abs(correct[mask].mean() - confidence[mask].mean())
    return float(ece)


def compute_metrics(y_true, probs) -> dict:
    """Macro-F1 là metric chính (lớp hiếm có trọng số ngang lớp phổ biến)."""
    with warnings.catch_warnings():  # tập nhỏ thiếu lớp -> sklearn cảnh báo lặp lại mỗi epoch
        warnings.simplefilter("ignore", UserWarning)
        return _compute_metrics(y_true, probs)


def _compute_metrics(y_true, probs) -> dict:
    y_true = np.asarray(y_true, dtype=int)
    probs = _normalize(probs)
    n_classes = probs.shape[1]
    labels = np.arange(n_classes)
    pred = probs.argmax(1)
    result = {
        "accuracy": float(accuracy_score(y_true, pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, pred)),
        "macro_f1": float(f1_score(y_true, pred, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_true, pred, average="weighted", zero_division=0)),
        "macro_precision": float(precision_score(y_true, pred, average="macro", zero_division=0)),
        "macro_recall": float(recall_score(y_true, pred, average="macro", zero_division=0)),
        "macro_auroc": float("nan"),
        "log_loss": float(log_loss(y_true, probs, labels=labels)),
        "ece": expected_calibration_error(y_true, probs),
        "n": int(len(y_true)),
    }
    try:
        if n_classes == 2:
            result["macro_auroc"] = float(roc_auc_score(y_true, probs[:, 1]))
        else:
            result["macro_auroc"] = float(roc_auc_score(y_true, probs, multi_class="ovr", average="macro", labels=labels))
    except ValueError:
        pass  # có lớp không xuất hiện trong tập này
    return result


def aggregate_cases(case_ids, y_true, probs) -> tuple[np.ndarray, np.ndarray, list]:
    """Trung bình xác suất các ảnh của cùng (bệnh nhân, nhãn) -> 1 dự đoán / ca."""
    frame = pd.DataFrame(np.asarray(probs, dtype=np.float64))
    frame["case"] = list(case_ids)
    frame["y"] = np.asarray(y_true, dtype=int)
    grouped = frame.groupby(["case", "y"], sort=True).mean()
    return grouped.index.get_level_values("y").to_numpy(), grouped.to_numpy(), list(grouped.index.get_level_values("case"))


def patient_metrics(frame: pd.DataFrame, probs) -> dict:
    """Metric mức bệnh nhân — chỉ tính khi mọi ảnh đều có mã bệnh nhân."""
    if "patient_key" not in frame or (frame["patient_key"] == "").any() or frame.empty:
        return {}
    y, p, _ = aggregate_cases(frame["patient_key"], frame["y"], probs)
    result = compute_metrics(y, p)
    return {k: result[k] for k in ("accuracy", "balanced_accuracy", "macro_f1", "macro_auroc", "n")}


def evaluation_tables(y_true, probs, classes: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(báo cáo từng lớp, confusion matrix có tên hàng = nhãn thật, cột = dự đoán)."""
    y_true = np.asarray(y_true, dtype=int)
    pred = np.asarray(probs).argmax(1)
    labels = list(range(len(classes)))
    report = classification_report(y_true, pred, labels=labels, target_names=classes, output_dict=True, zero_division=0)
    per_class = pd.DataFrame(report).T.reset_index().rename(columns={"index": "class"})
    confusion = pd.DataFrame(confusion_matrix(y_true, pred, labels=labels), index=classes, columns=classes)
    confusion.index.name = "true\\pred"
    return per_class, confusion.reset_index()


def apply_temperature(probs, temperature: float) -> np.ndarray:
    """softmax(log p / T): tương đương chia logits cho T."""
    log_p = np.log(np.clip(np.asarray(probs, dtype=np.float64), 1e-12, 1.0)) / float(temperature)
    log_p -= log_p.max(1, keepdims=True)
    scaled = np.exp(log_p)
    return scaled / scaled.sum(1, keepdims=True)


def fit_temperature(y_true, probs) -> float:
    """Tìm T > 0 giảm NLL trên VALIDATION (không dùng test). T > 1 = model đang quá tự tin."""
    y_true = np.asarray(y_true, dtype=int)

    def nll(temperature: float) -> float:
        scaled = apply_temperature(probs, temperature)
        return float(-np.mean(np.log(np.clip(scaled[np.arange(len(y_true)), y_true], 1e-12, 1.0))))

    grid = np.exp(np.linspace(np.log(0.05), np.log(20.0), 241))
    best = float(grid[int(np.argmin([nll(t) for t in grid]))])
    low, high = max(0.05, best / 1.1), min(20.0, best * 1.1)  # tinh chỉnh golden-section quanh điểm tốt nhất
    golden = (np.sqrt(5) - 1) / 2
    for _ in range(40):
        a, b = high - golden * (high - low), low + golden * (high - low)
        if nll(a) < nll(b):
            high = b
        else:
            low = a
    return float((low + high) / 2)
