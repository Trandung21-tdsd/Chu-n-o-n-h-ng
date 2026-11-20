"""Chia train/val/test theo NHÓM (bệnh nhân + ảnh trùng/gần trùng), phân tầng theo lớp — chống leakage."""
from __future__ import annotations

import math

import numpy as np
import pandas as pd

SPLITS = ("train", "val", "test")


def parse_ratios(values) -> tuple[float, float, float]:
    """Nhận '70 15 15' hoặc '0.8 0.1 0.1'. Tổng phải đúng 100%."""
    numbers = [float(v) for v in values]
    if len(numbers) != 3:
        raise ValueError("Cần đúng 3 tỷ lệ train val test, ví dụ: --split 70 15 15")
    total = sum(numbers)
    if total > 1.5:  # nhập theo phần trăm
        numbers = [v / 100.0 for v in numbers]
        total = total / 100.0
    if min(numbers[:2]) <= 0 or numbers[2] < 0 or not math.isclose(total, 1.0, abs_tol=1e-6):
        raise ValueError(
            f"Tỷ lệ train:val:test phải dương (test = 0 chỉ khi đã có tập test cố định) và có tổng 100% (đang là {total * 100:g}%). "
            "Dùng 70 15 15 hoặc 80 10 10 — '80:20:10' không hợp lệ vì tổng là 110%."
        )
    return tuple(numbers)  # type: ignore[return-value]


def grouped_stratified_split(units: pd.DataFrame, ratios, seed: int = 42, n_iter: int = 400,
                             names=SPLITS) -> tuple[dict, dict]:
    """Gán từng NHÓM vào train/val/test sao cho tỷ lệ từng lớp gần `ratios` nhất.

    units: mỗi dòng là 1 đơn vị độc lập (bệnh nhân, hoặc 1 ảnh nếu không có mã bệnh nhân) với
           cột group_id, label, n_images. Mọi đơn vị cùng group_id luôn vào cùng 1 tập.
    Thuật toán: tham lam có ngẫu nhiên (nhóm lớn xếp trước, nhóm vào tập đang "thiếu" lớp của nó nhất),
    lặp n_iter lần với thứ tự khác nhau, giữ phương án lệch ít nhất. Chỉ dùng nhãn + kích thước nhóm,
    không bao giờ dùng ảnh hay kết quả model.
    """
    ratio = np.asarray(ratios, dtype=float)
    n_splits = len(names)
    if len(ratio) != n_splits:
        raise ValueError("Số tỷ lệ phải bằng số tập")
    classes = sorted(units["label"].unique())
    class_index = {c: i for i, c in enumerate(classes)}
    group_ids = sorted(units["group_id"].unique())
    group_index = {g: i for i, g in enumerate(group_ids)}
    n_groups, n_classes = len(group_ids), len(classes)
    if n_groups < n_splits:
        raise ValueError(f"Chỉ có {n_groups} nhóm độc lập — không thể chia {n_splits} tập.")
    counts = np.zeros((n_groups, n_classes))
    images = np.zeros(n_groups)
    for row in units.itertuples(index=False):
        g = group_index[row.group_id]
        counts[g, class_index[row.label]] += 1
        images[g] += row.n_images
    total_class = counts.sum(0)
    total_images = images.sum()
    target_class = ratio[:, None] * total_class[None, :]
    target_images = ratio * total_images
    groups_per_class = (counts > 0).sum(0)
    coverable = groups_per_class >= n_splits
    size = counts.sum(1)
    rng = np.random.default_rng(seed)

    best = None
    for iteration in range(n_iter):
        if iteration == 0:
            order = np.lexsort((rng.random(n_groups), -size))
        else:
            order = np.argsort(-size * rng.uniform(0.3, 1.7, size=n_groups), kind="stable")
        current_class = np.zeros((n_splits, n_classes))
        current_images = np.zeros(n_splits)
        assignment = np.empty(n_groups, dtype=int)
        for g in order:
            class_need = (target_class - current_class) / np.maximum(target_class, 1e-9)
            score = (class_need * counts[g][None, :]).sum(1) / max(counts[g].sum(), 1.0)
            score += 0.25 * (target_images - current_images) / np.maximum(target_images, 1e-9)
            choice = int(np.argmax(score + 1e-6 * rng.random(n_splits)))
            assignment[g] = choice
            current_class[choice] += counts[g]
            current_images[choice] += images[g]
        fraction = current_class / np.maximum(total_class, 1e-9)
        error = float(((fraction - ratio[:, None]) ** 2).mean()) + float(((current_images / total_images - ratio) ** 2).sum())
        missing = int(((current_class == 0) & coverable[None, :]).sum())
        empty = int((current_images == 0).sum())
        total_score = error + 10.0 * missing + 100.0 * empty
        if best is None or total_score < best[0] - 1e-12:
            best = (total_score, assignment.copy(), current_class.copy(), current_images.copy(), missing, empty)

    _, assignment, current_class, current_images, missing, empty = best
    if empty:
        raise ValueError("Không chia được: có tập rỗng. Dữ liệu quá ít nhóm độc lập.")
    mapping = {group_ids[g]: names[assignment[g]] for g in range(n_groups)}
    info = {
        "units_per_split": {names[s]: int(current_class[s].sum()) for s in range(n_splits)},
        "images_per_split": {names[s]: int(current_images[s]) for s in range(n_splits)},
        "achieved_ratio_images": {names[s]: round(float(current_images[s] / total_images), 4) for s in range(n_splits)},
        "achieved_ratio_units": {names[s]: round(float(current_class[s].sum() / total_class.sum()), 4) for s in range(n_splits)},
        "classes_with_fewer_than_3_groups": [c for c, ok in zip(classes, coverable) if not ok],
        "class_missing_in_some_split": int(missing),
    }
    return mapping, info


def naive_split(labels: pd.Series, ratios, seed: int = 42) -> np.ndarray:
    """Chia NGẪU NHIÊN THEO ẢNH (bỏ qua bệnh nhân & duplicate) — chỉ để minh họa leakage, không dùng thật."""
    from sklearn.model_selection import train_test_split

    y = labels.to_numpy()
    index = np.arange(len(y))

    def safe_split(idx, size, random_state):
        values, counts = np.unique(y[idx], return_counts=True)
        stratify = y[idx] if counts.min() >= 2 else None
        try:
            return train_test_split(idx, test_size=size, random_state=random_state, stratify=stratify)
        except ValueError:
            return train_test_split(idx, test_size=size, random_state=random_state, stratify=None)

    rest, test = safe_split(index, ratios[2], seed)
    train, val = safe_split(rest, ratios[1] / (ratios[0] + ratios[1]), seed + 1)
    result = np.empty(len(y), dtype=object)
    result[train], result[val], result[test] = "train", "val", "test"
    return result


def leakage_report(frame: pd.DataFrame, split_col: str) -> dict:
    """Đếm khóa (bệnh nhân / nhóm / pixel / file) xuất hiện ở > 1 tập, và ảnh test 'nhìn thấy' trong train."""
    report = {}
    for key in ("patient_key", "group_id", "pixel_hash", "sha256"):
        values = frame[frame[key].astype(str) != ""]
        report[f"{key}_crossing_splits"] = int((values.groupby(key)[split_col].nunique() > 1).sum())
    train = frame[frame[split_col] == "train"]
    test = frame[frame[split_col] == "test"]
    with_patient = test[test["patient_key"] != ""]
    report["test_images"] = int(len(test))
    report["test_images_same_patient_in_train"] = int(with_patient["patient_key"].isin(set(train["patient_key"])).sum())
    report["test_images_exact_duplicate_in_train"] = int(test["pixel_hash"].isin(set(train["pixel_hash"])).sum())
    report["test_images_same_group_in_train"] = int(test["group_id"].isin(set(train["group_id"])).sum())
    return report
