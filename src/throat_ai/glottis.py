"""Từ mask phân đoạn -> số đo + dấu hiệu nội soi thanh quản (giải thích được, đối chiếu bài giảng PDF).

Bài giảng "Kĩ năng khám TMH" tr.35: thanh quản bình thường — hai dây thanh cân đối, hít vào mở hình tam giác,
phát âm khép kín. tr.37: tổn thương thực thể (hạt dây thanh, u hạt, u nhú, ung thư) làm bờ dây thanh không đều.

Số đo trên 1 ẢNH (1 thời điểm của chu kỳ rung):
    glottis_ratio      diện tích thanh môn (đỏ) / (2 dây thanh + thanh môn)       — độ mở thanh môn
    closed             thanh môn gần như không thấy (đang khép)
    fold_asymmetry     |trái - phải| / (trái + phải) theo diện tích dây thanh   — cân đối hai bên
    gap_parts          số khe hở rời nhau của thanh môn (>= 2 => khe hình ĐỒNG HỒ CÁT, gợi ý khối trên bờ dây thanh)
    midline_offset     thanh môn lệch về một bên so với đường giữa 2 dây thanh (0 = chính giữa)
Số đo trên 1 VIDEO (gộp các frame được lấy ngẫu nhiên trong lúc phát âm — BAGLS lấy frame ngẫu nhiên):
    closed_fraction    tỉ lệ frame thanh môn khép (≈ tỉ lệ thời gian khép, "closed quotient")
    min_glottis_ratio  độ mở nhỏ nhất (bách phân vị 5) — > 0 rõ rệt => KHÔNG KHÉP KÍN khi phát âm
    max_glottis_ratio  độ mở lớn nhất (bách phân vị 95) — biên độ mở
    amplitude_asymmetry chênh lệch biên độ rung trái/phải (độ dao động diện tích mỗi dây thanh)
    hourglass_fraction tỉ lệ frame có khe đồng hồ cát
    mean_fold_asymmetry, mean_midline_offset

Ngưỡng "bất thường" = vượt khoảng tham chiếu tính từ các video BÌNH THƯỜNG của tập TRAIN (mask gán tay), không
dùng val/test => không rò rỉ. Đây là ngưỡng thống kê tham khảo, CHƯA được thẩm định lâm sàng.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .seg_data import GLOTTIS, LEFT, RIGHT

CLOSED_RATIO = 0.01  # thanh môn < 1% vùng thanh quản => coi là đang khép
MIN_PART_SHARE = 0.10  # khe hở phụ phải >= 10% diện tích thanh môn mới tính là 1 khe riêng

IMAGE_FEATURES = ["glottis_ratio", "closed", "fold_asymmetry", "gap_parts", "midline_offset",
                  "left_area", "right_area", "glottis_area"]
VIDEO_FEATURES = ["closed_fraction", "min_glottis_ratio", "max_glottis_ratio", "amplitude_asymmetry",
                  "hourglass_fraction", "mean_fold_asymmetry", "mean_midline_offset"]
# Hướng "bất thường" của từng số đo video: high = giá trị lớn là bất thường, low = giá trị nhỏ là bất thường.
DIRECTION = {"closed_fraction": "low", "min_glottis_ratio": "high", "max_glottis_ratio": "both",
             "amplitude_asymmetry": "high", "hourglass_fraction": "high", "mean_fold_asymmetry": "high",
             "mean_midline_offset": "high"}


def _components(binary: np.ndarray) -> list[int]:
    import cv2

    count, labels, stats, _ = cv2.connectedComponentsWithStats(binary.astype(np.uint8), connectivity=8)
    return sorted((int(a) for a in stats[1:, cv2.CC_STAT_AREA]), reverse=True)


def image_features(mask: np.ndarray) -> dict:
    left, right, glottis = (mask == LEFT), (mask == RIGHT), (mask == GLOTTIS)
    nl, nr, ng = int(left.sum()), int(right.sum()), int(glottis.sum())
    larynx = nl + nr + ng
    ratio = ng / larynx if larynx else float("nan")
    parts = _components(glottis) if ng else []
    gap_parts = sum(1 for a in parts if a >= MIN_PART_SHARE * ng) if ng else 0
    offset = float("nan")
    if nl and nr and ng:
        xl, xr, xg = (np.nonzero(m)[1].mean() for m in (left, right, glottis))
        width = abs(xr - xl)
        offset = abs(xg - (xl + xr) / 2) / width if width > 0 else float("nan")
    return {
        "left_area": nl, "right_area": nr, "glottis_area": ng,
        "glottis_ratio": ratio,
        "closed": bool(larynx and ratio < CLOSED_RATIO),
        "fold_asymmetry": abs(nl - nr) / (nl + nr) if (nl + nr) else float("nan"),
        "gap_parts": gap_parts,
        "midline_offset": offset,
        "folds_visible": bool(nl and nr),
    }


def video_features(frames: list[dict] | pd.DataFrame) -> dict:
    df = pd.DataFrame(frames)
    df = df[df["left_area"] + df["right_area"] + df["glottis_area"] > 0]
    if df.empty:
        return {k: float("nan") for k in VIDEO_FEATURES} | {"n_frames": 0}
    ratio = df["glottis_ratio"].astype(float)
    total = (df["left_area"] + df["right_area"]).replace(0, np.nan)
    left_share, right_share = df["left_area"] / total, df["right_area"] / total  # chuẩn hóa theo cỡ ảnh/zoom
    sl, sr = float(np.nanstd(df["left_area"] / df["left_area"].mean())), float(np.nanstd(df["right_area"] / df["right_area"].mean()))
    return {
        "n_frames": int(len(df)),
        "closed_fraction": float(df["closed"].astype(float).mean()),
        "min_glottis_ratio": float(np.nanpercentile(ratio, 5)),
        "max_glottis_ratio": float(np.nanpercentile(ratio, 95)),
        "amplitude_asymmetry": abs(sl - sr) / (sl + sr) if (sl + sr) > 0 else 0.0,
        "hourglass_fraction": float((df["gap_parts"] >= 2).mean()),
        "mean_fold_asymmetry": float(np.nanmean(np.abs(left_share - right_share))),
        "mean_midline_offset": float(np.nanmean(df["midline_offset"])) if df["midline_offset"].notna().any() else float("nan"),
    }


def reference_ranges(video_table: pd.DataFrame, label_col: str = "label", normal: str = "Normal") -> dict:
    """Khoảng tham chiếu (bách phân vị 5–95) của từng số đo video trên các video BÌNH THƯỜNG."""
    normal_rows = video_table[video_table[label_col] == normal]
    ranges = {}
    for feature in VIDEO_FEATURES:
        values = normal_rows[feature].dropna().to_numpy(dtype=float)
        if len(values) >= 5:
            ranges[feature] = {"p05": float(np.percentile(values, 5)), "p50": float(np.percentile(values, 50)),
                               "p95": float(np.percentile(values, 95)), "n_videos": int(len(values))}
    return ranges


def _flag(feature: str, value: float, ranges: dict) -> str | None:
    ref = ranges.get(feature)
    if ref is None or value is None or not np.isfinite(value):
        return None
    direction = DIRECTION[feature]
    if direction in ("high", "both") and value > ref["p95"]:
        return "high"
    if direction in ("low", "both") and value < ref["p05"]:
        return "low"
    return None


def describe_image(feat: dict) -> list[str]:
    """Mô tả 1 frame (chỉ là 1 thời điểm — không đủ để kết luận khép kín hay không)."""
    if not feat.get("folds_visible"):
        return ["Không thấy rõ đủ 2 dây thanh trong ảnh (che khuất/góc soi) — số đo kém tin cậy."]
    lines = []
    if feat["closed"]:
        lines.append("Thanh môn đang KHÉP ở thời điểm này (hai dây thanh chạm nhau).")
    else:
        lines.append(f"Thanh môn đang MỞ: khe hở chiếm {feat['glottis_ratio']:.0%} vùng thanh quản.")
    if feat["gap_parts"] >= 2:
        lines.append(f"Khe thanh môn chia thành {feat['gap_parts']} phần rời nhau (hình ĐỒNG HỒ CÁT) — gợi ý có "
                     "khối trên bờ tự do dây thanh chạm nhau ở giữa (hạt/polyp/nang, bài giảng tr.37).")
    asym = feat["fold_asymmetry"]
    if np.isfinite(asym):
        side = "cân đối" if asym < 0.2 else ("lệch nhẹ" if asym < 0.35 else "LỆCH RÕ")
        lines.append(f"Diện tích hai dây thanh {side} (chênh {asym:.0%}).")
    return lines


SIGN_TEXT = {
    ("min_glottis_ratio", "high"): ("Thanh môn KHÔNG KHÉP KÍN khi phát âm (luôn còn khe hở)", ["Functional", "Organic_lesion"]),
    ("closed_fraction", "low"): ("Thời gian khép thanh môn ngắn bất thường", ["Functional", "Organic_lesion"]),
    ("max_glottis_ratio", "high"): ("Thanh môn mở rộng bất thường", ["Functional"]),
    ("max_glottis_ratio", "low"): ("Biên độ mở thanh môn nhỏ bất thường (rung kém / co thắt)", ["Functional", "Organic_lesion"]),
    ("amplitude_asymmetry", "high"): ("Hai dây thanh rung KHÔNG ĐỀU (một bên giảm biên độ — gợi ý liệt/xơ cứng)", ["Functional", "Organic_lesion"]),
    ("hourglass_fraction", "high"): ("Khe thanh môn hình ĐỒNG HỒ CÁT lặp lại (gợi ý hạt/polyp/nang ở bờ dây thanh)", ["Organic_lesion"]),
    ("mean_fold_asymmetry", "high"): ("Hai dây thanh không cân đối về kích thước (gợi ý khối/phù một bên)", ["Organic_lesion"]),
    ("mean_midline_offset", "high"): ("Khe thanh môn lệch hẳn về một bên (một dây thanh di động kém/lệch)", ["Functional", "Organic_lesion"]),
}


def assess_video(feat: dict, ranges: dict) -> dict:
    """So số đo video với khoảng tham chiếu của thanh quản bình thường -> danh sách dấu hiệu + nhóm được ủng hộ."""
    signs, support = [], {"Normal": 0, "Organic_lesion": 0, "Functional": 0}
    for feature in VIDEO_FEATURES:
        flag = _flag(feature, feat.get(feature), ranges)
        if flag and (feature, flag) in SIGN_TEXT:
            text, groups = SIGN_TEXT[(feature, flag)]
            ref = ranges[feature]
            signs.append(f"{text} — {feature}={feat[feature]:.3f} (bình thường {ref['p05']:.3f}–{ref['p95']:.3f})")
            for group in groups:
                support[group] += 1
    if not signs and ranges:
        signs.append("Các số đo nằm trong khoảng của thanh quản bình thường (tr.35: hai dây thanh cân đối, khép kín "
                     "khi phát âm).")
        support["Normal"] += 1
    leaning = max(support, key=support.get) if any(support.values()) else None
    return {"signs": signs, "support": support, "leaning": leaning}


def consistency(classifier_label: str | None, assessment: dict) -> str | None:
    if not classifier_label or not assessment.get("leaning"):
        return None
    support = assessment["support"]
    if classifier_label == "Disorder":
        agree = support["Organic_lesion"] + support["Functional"] > 0
    else:
        agree = support.get(classifier_label, 0) > 0
    if agree:
        return "Số đo từ phân đoạn PHÙ HỢP với nhóm bệnh model phân loại dự đoán."
    return ("Số đo từ phân đoạn KHÔNG ủng hộ nhóm bệnh model phân loại dự đoán — nên để bác sĩ xem lại "
            "(một số bệnh, vd. viêm/bạch sản, thay đổi màu niêm mạc mà mask hình dạng không thấy).")


def sign_classifier_proba(model: dict | None, feat: dict) -> dict[str, float] | None:
    """Xác suất nhóm bệnh từ số đo video (tham số hồi quy logistic lưu trong sign_classifier.json)."""
    if not model:
        return None
    x = np.array([feat.get(f, np.nan) for f in VIDEO_FEATURES], dtype=float)
    if not np.isfinite(x).all():
        return None
    z = (x - np.asarray(model["mean"])) / np.asarray(model["scale"])
    coef, intercept = np.asarray(model["coef"]), np.asarray(model["intercept"])
    if len(model["classes"]) == 2:  # sklearn nhị phân: 1 hàng hệ số cho lớp thứ 2
        p = 1 / (1 + np.exp(-(z @ coef[0] + intercept[0])))
        return {model["classes"][0]: float(1 - p), model["classes"][1]: float(p)}
    logits = coef @ z + intercept
    e = np.exp(logits - logits.max())
    return {c: float(v) for c, v in zip(model["classes"], e / e.sum())}
