"""Adapter bộ BAGLS-VF (nội soi tốc độ cao thanh môn) + gán nhãn bệnh từ metadata BAGLS gốc.

BAGLS-VF chỉ có ẢNH + MASK phân đoạn (không có nhãn bệnh). Nhãn được lấy bằng cách ghép mỗi ảnh với file
`.meta` của bộ BAGLS gốc (thư mục archive-2), trường "Subject disorder status" (văn bản tự do, >40 kiểu):
    train: BAGLS_image/ID<VideoId>_<frame>_img_old_<idx>.png  ->  archive-2/training/training/<idx>.meta
    test : test_images/<idx>.png                               ->  archive-2/test/test/<idx>.meta

Văn bản chẩn đoán được quy về các nhóm theo bài giảng "Kĩ năng khám Tai Mũi Họng" (ĐH Y Dược TP.HCM):
    - Thanh quản bình thường (tr.35: dây thanh trắng, cân đối, mở hình chữ V khi hít vào, khép kín khi phát âm)
    - Tổn thương thực thể (tr.37: hạt dây thanh, u hạt mấu thanh, u nhú thanh quản, ung thư thanh quản;
      cùng nhóm: polyp, nang, phù Reinke, bạch sản, viêm thanh quản, sẹo/rãnh dây thanh)
    - Rối loạn chức năng / vận động (không thấy khối tổn thương: căng cơ, đóng thanh môn không kín, liệt, teo)
Mỗi bệnh cụ thể chỉ có 1–12 video nên KHÔNG đủ để train từng bệnh; tên bệnh chi tiết được giữ ở cột `histology`.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import pandas as pd

from .sources import AuditLog, apply_label_map, flag, normalize_rel, scan_images

# (từ khóa trong văn bản gốc, mã chi tiết) — xét theo thứ tự, bệnh nặng/đặc hiệu trước.
ORGANIC_RULES = [
    (r"carcinom|cancer", "carcinoma"),
    (r"papillom", "papilloma"),
    (r"granulom", "granuloma"),
    (r"nodul", "nodules"),
    (r"polyp(?!oid)", "polyp"),
    (r"cyst", "cyst"),
    (r"reinke|edema|oedema|polipoid|polypoid", "reinke_edema"),
    (r"leuco|leuko", "leukoplakia"),
    (r"laryngitis", "laryngitis"),
    (r"scar|sulcus|vergeture|fibrotic", "scar_sulcus"),
    (r"swelling|mucosal irregularit|teleang", "mucosal_lesion"),
]
FUNCTIONAL_RULES = [
    (r"spasmodic", "spasmodic_dysphonia"),
    (r"paresis|paralys", "paresis"),
    (r"muscle tension", "muscle_tension_dysphonia"),
    (r"atroph|bowed", "atrophy"),
    (r"insufficien|glottic closure", "glottic_insufficiency"),
    (r"functional|irregular vibration|stiff", "functional_other"),
]

DETAIL_VI = {
    "normal": "Thanh quản bình thường",
    "carcinoma": "Ung thư thanh quản", "papilloma": "U nhú thanh quản", "granuloma": "U hạt mấu thanh",
    "nodules": "Hạt dây thanh", "polyp": "Polyp dây thanh", "cyst": "Nang dây thanh",
    "reinke_edema": "Phù Reinke / phù nề dây thanh", "leukoplakia": "Bạch sản dây thanh",
    "laryngitis": "Viêm thanh quản", "scar_sulcus": "Sẹo / rãnh / xơ cứng dây thanh",
    "mucosal_lesion": "Sưng nề / bất thường niêm mạc dây thanh",
    "spasmodic_dysphonia": "Loạn trương lực thanh quản (co thắt)", "paresis": "Liệt / yếu dây thanh",
    "muscle_tension_dysphonia": "Rối loạn giọng do căng cơ", "atrophy": "Teo / cong dây thanh",
    "glottic_insufficiency": "Đóng thanh môn không kín", "functional_other": "Rối loạn chức năng khác",
}


def classify_status(text) -> tuple[str, str]:
    """Văn bản 'Subject disorder status' -> (nhóm, mã chi tiết). Nhóm '' = không xác định (bị loại)."""
    raw = str(text or "").strip()
    low = raw.lower()
    if not raw:
        return "", "unknown"
    if low == "healthy":
        return "Normal", "normal"
    if low.startswith("none"):
        return "", "post_treatment_none"  # "None (sau điều trị...)" -> không rõ hiện trạng, loại
    for pattern, detail in ORGANIC_RULES:
        if re.search(pattern, low):
            return "Organic_lesion", detail
    for pattern, detail in FUNCTIONAL_RULES:
        if re.search(pattern, low):
            return "Functional", detail
    return "", "unmapped"


def load_bagls_vf(source: str, root: Path, cfg: dict, audit: AuditLog) -> pd.DataFrame:
    root = Path(root)
    part = cfg.get("part", "train")
    if part not in ("train", "test"):
        raise ValueError("part của BAGLS-VF phải là 'train' hoặc 'test'")
    meta_root = Path(cfg["meta_root"])
    if not meta_root.is_absolute():
        from .utils import resolve_config_path

        meta_root = resolve_config_path(meta_root)
    meta_dir = meta_root / ("training/training" if part == "train" else "test/test")
    if not meta_dir.is_dir():
        raise FileNotFoundError(f"[{source}] Không thấy metadata BAGLS gốc {meta_dir} (cần thư mục archive-2)")
    image_dir = cfg.get("image_dir", "BAGLS_image" if part == "train" else "test_images")
    rows, missing = [], 0
    for path in scan_images(root / image_dir):
        rel = normalize_rel(path.relative_to(root).as_posix())
        if part == "train":
            match = re.search(r"ID(\d+)_\d+_img_old_(\d+)", path.stem)
            if not match:
                audit.add(source, "<tên_file>", rel, "unexpected_filename", "ignored")
                continue
            file_video, index = int(match.group(1)), match.group(2)
        else:
            file_video, index = None, path.stem
        meta_path = meta_dir / f"{index}.meta"
        if not meta_path.exists():
            missing += 1
            audit.add(source, meta_path.name, rel, "metadata_missing", "quarantine_image")
            rows.append({"source": source, "rel_path": rel, "image_path": str(path), "label_raw": "",
                         "patient_id": "", "histology": "", "exclude_reason": "quarantine:metadata_missing"})
            continue
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        video = int(meta["Video Id"])
        reason = ""
        if file_video is not None and file_video != video:
            audit.add(source, meta_path.name, rel, "video_id_mismatch", "quarantine_image", f"file={file_video} meta={video}")
            reason = "quarantine:video_id_mismatch"
        status = str(meta.get("Subject disorder status") or "").strip()
        group, detail = classify_status(status)
        rows.append({"source": source, "rel_path": rel, "image_path": str(path), "label_raw": group or f"?{detail}",
                     "disorder_text": status, "histology": detail, "patient_id": f"V{video}",
                     "exclude_reason": reason})
    if not rows:
        raise FileNotFoundError(f"[{source}] Không có ảnh trong {root / image_dir}")
    frame = pd.DataFrame(rows)
    frame["lesion_type"] = ""
    unknown = frame["label_raw"].str.startswith("?") & (frame["exclude_reason"] == "")
    for (detail, text), group in frame[unknown].groupby([frame["histology"], frame["disorder_text"].fillna("")]):
        audit.add(source, "metadata", text or "(trống)", f"status_{detail}", "excluded",
                  f"{group['patient_id'].nunique()} video, {len(group)} ảnh")
    flag(frame, unknown, "excluded:disorder_status_unknown_or_ambiguous")
    frame.loc[unknown, "label_raw"] = ""
    binary = cfg.get("label_scheme", "3class") == "binary"
    mapping = {"Normal": "Normal", "Organic_lesion": "Disorder" if binary else "Organic_lesion",
               "Functional": "Disorder" if binary else "Functional", "": None}
    frame = apply_label_map(frame, {"label_map": mapping, "unmapped": "error"}, source, audit)
    frame["fixed_split"] = "test" if part == "test" else ""
    audit.files.append({"source": source, "file": str(meta_dir), "rows": int(len(frame)), "status": "checked",
                        "metadata_missing": missing, "videos": int(frame["patient_id"].nunique())})
    return frame.drop(columns=["disorder_text"])
