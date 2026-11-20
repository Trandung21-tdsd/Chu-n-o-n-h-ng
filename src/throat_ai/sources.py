"""Adapter đọc từng nguồn ảnh -> bảng chuẩn hóa, kèm AUDIT FILE NHÃN.

Mỗi adapter trả về DataFrame có các cột:
    source, rel_path, image_path, label_raw, label, patient_id, exclude_reason
`exclude_reason` khác rỗng => ảnh bị loại khỏi train/val/test (ghi vào removed_images.csv kèm lý do).

Xử lý duplicate trong file nhãn (label file):
    * dòng trùng hoàn toàn                 -> giữ 1 dòng, bỏ phần thừa (ghi log)
    * cùng ảnh xuất hiện nhiều lần, CÙNG nhãn -> giữ 1 dòng (ghi log)
    * cùng ảnh xuất hiện nhiều lần, KHÁC nhãn -> cách ly ảnh (không tự chọn nhãn)
    * nhãn trong file khác nhãn suy ra từ thư mục -> cách ly ảnh
    * cột `split` có sẵn trong file nhãn     -> bỏ qua (pipeline tự chia theo bệnh nhân)
"""
from __future__ import annotations

import re
import unicodedata
from pathlib import Path

import pandas as pd

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}

# Ontology 10 lớp của CE-NBI — trùng với cột `label` trong manifests/larynx_histology.csv của bộ dữ liệu.
CENBI_10CLASS_MAP = {
    "Carcinoma_in_situ": "Carcinoma_in_situ",
    "Cyst": "Cyst",
    "High_grade_dysplasia": "High_grade_dysplasia",
    "Hyperkeratosis": "Hyperkeratosis",
    "Low_grade_dysplasia": "Low_grade_dysplasia",
    "Papillomatosis": "Papilloma",
    "Polyp": "Polyp",
    "Reinkes_edema": "Reinkes_edema",
    "SCC": "SCC",
    "Amyloidosis": "Other_benign",
    "Granuloma": "Other_benign",
    "Hemangioma": "Other_benign",
    "Hyperplasia": "Other_benign",
    "Inflammation": "Other_benign",
    "Namboo_node": "Other_benign",
    "Nodule": "Other_benign",
}

# Tên trong bảng Excel bệnh nhân khác tên thư mục -> quy về cùng một khóa.
_CENBI_ALIASES = {
    "squamous_hyperplasia": "hyperplasia",
    "papilloma": "papillomatosis",
    "bamboo_node": "namboo_node",
    "bamboo_nodes": "namboo_node",
}


class AuditLog:
    """Gom mọi phát hiện khi kiểm tra dữ liệu: mỗi dòng là một phát hiện + hành động đã làm."""

    ISSUE_COLUMNS = ["source", "file", "key", "issue", "action", "detail"]

    def __init__(self) -> None:
        self.issues: list[dict] = []
        self.files: list[dict] = []

    def add(self, source, file, key, issue, action, detail="") -> None:
        self.issues.append(
            {"source": source, "file": file, "key": str(key), "issue": issue, "action": action, "detail": str(detail)}
        )

    def issues_frame(self) -> pd.DataFrame:
        return pd.DataFrame(self.issues, columns=self.ISSUE_COLUMNS)

    def files_frame(self) -> pd.DataFrame:
        return pd.DataFrame(self.files)


def nfc(text) -> str:
    return unicodedata.normalize("NFC", str(text))


def normalize_rel(path_text) -> str:
    """Chuẩn hóa đường dẫn tương đối: '/', NFC (macOS hay lưu tên file dạng NFD)."""
    text = nfc(str(path_text)).replace("\\", "/").strip()
    while text.startswith("./"):
        text = text[2:]
    return text.lstrip("/")


def scan_images(root: Path) -> list[Path]:
    skip = (".", "~$", "._")
    files = [
        p for p in Path(root).rglob("*")
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS and not p.name.startswith(skip)
    ]
    return sorted(files, key=lambda p: normalize_rel(p.relative_to(root).as_posix()))


def flag(frame: pd.DataFrame, mask, reason: str) -> None:
    """Đánh dấu loại ảnh (in-place); giữ lý do đầu tiên nếu ảnh đã bị đánh dấu trước đó."""
    target = pd.Series(mask, index=frame.index).astype(bool) & (frame["exclude_reason"] == "")
    frame.loc[target, "exclude_reason"] = reason


def apply_label_map(frame: pd.DataFrame, cfg: dict, source: str, audit: AuditLog) -> pd.DataFrame:
    """Đổi nhãn gốc -> nhãn train theo `label_map` (giá trị null = loại khỏi bài toán)."""
    mapping = cfg.get("label_map") or {}
    unmapped = str(cfg.get("unmapped", "error" if mapping else "keep")).lower()
    labels, unknown = [], set()
    for raw in frame["label_raw"].astype(str):
        if raw in mapping:
            labels.append(mapping[raw])
        elif unmapped == "keep":
            labels.append(raw)
        elif unmapped == "drop":
            labels.append(None)
        else:
            unknown.add(raw)
            labels.append(None)
    if unknown:
        raise ValueError(
            f"[{source}] Nhãn gốc chưa khai báo trong label_map: {sorted(unknown)}. "
            "Thêm vào configs/tasks.json (giá trị null = loại bỏ) hoặc đặt \"unmapped\": \"keep\"."
        )
    frame = frame.copy()
    frame["label"] = labels
    dropped = frame["label"].isna()
    for raw, count in frame.loc[dropped, "label_raw"].value_counts().items():
        audit.add(source, "label_map", raw, "excluded_by_label_map", "excluded", f"{count} ảnh")
    flag(frame, dropped, "excluded:label_map")
    frame["label"] = frame["label"].fillna("").astype(str)
    return frame


def audit_label_table(
    table: pd.DataFrame,
    file_name: str,
    source: str,
    frame: pd.DataFrame,
    audit: AuditLog,
    path_col: str = "image_path",
    label_col: str = "label",
    expected: dict[str, pd.Series] | None = None,
    unlisted_action: str = "kept_folder_label",
) -> dict:
    """Kiểm tra 1 file nhãn. Ảnh có nhãn mâu thuẫn bị đánh dấu cách ly trực tiếp trong `frame`.

    expected: {tên cột trong file nhãn: Series(rel_path -> giá trị kỳ vọng suy từ thư mục)}
    """
    summary = {"source": source, "file": file_name, "rows": int(len(table))}
    if path_col not in table.columns:
        audit.add(source, file_name, "", "missing_path_column", "ignored_file", f"thiếu cột {path_col}")
        summary["status"] = f"ignored (thiếu cột {path_col})"
        audit.files.append(summary)
        return summary
    table = table.copy()
    original_columns = list(table.columns)
    table["_path"] = table[path_col].map(normalize_rel)

    duplicate_rows = table.duplicated(subset=original_columns, keep="first")
    for key in table.loc[duplicate_rows, "_path"]:
        audit.add(source, file_name, key, "duplicate_row", "dropped_duplicate_row")
    table = table[~duplicate_rows]

    conflicting, same_label = [], []
    repeated = table[table.duplicated("_path", keep=False)]
    for key, group in repeated.groupby("_path", sort=True):
        if label_col in group.columns and group[label_col].astype(str).str.strip().nunique() > 1:
            conflicting.append(key)
            labels = "; ".join(sorted(group[label_col].astype(str).str.strip().unique()))
            audit.add(source, file_name, key, "duplicate_path_conflicting_labels", "quarantine_image", labels)
        else:
            same_label.append(key)
            audit.add(source, file_name, key, "duplicate_path_same_label", "dropped_duplicate_row", f"{len(group)} dòng")
    table = table.drop_duplicates("_path", keep="first")

    on_disk, listed = set(frame["rel_path"]), set(table["_path"])
    missing, unlisted = sorted(listed - on_disk), sorted(on_disk - listed)
    for key in missing:
        audit.add(source, file_name, key, "listed_file_missing", "none")
    for key in unlisted:
        audit.add(source, file_name, key, "file_not_in_label_file", unlisted_action)

    bad_paths, mismatches = set(conflicting), 0
    for column, expected_values in (expected or {}).items():
        if column not in table.columns:
            continue
        subset = table[table["_path"].isin(expected_values.index) & ~table["_path"].isin(conflicting)]
        wanted = subset["_path"].map(expected_values).astype(str)
        given = subset[column].astype(str).str.strip()
        differs = given != wanted
        for key, value, target in zip(subset.loc[differs, "_path"], given[differs], wanted[differs]):
            audit.add(source, file_name, key, f"{column}_mismatch", "quarantine_image", f"file_nhãn={value} | thư_mục={target}")
            bad_paths.add(key)
            mismatches += 1
    if bad_paths:
        flag(frame, frame["rel_path"].isin(bad_paths), "quarantine:label_file_conflict")
    if "split" in table.columns:
        audit.add(source, file_name, "split", "provided_split_ignored", "ignored",
                  "pipeline tự chia theo bệnh nhân/nhóm trùng; không dùng cột split có sẵn")

    summary.update(
        status="checked",
        duplicate_rows=int(duplicate_rows.sum()),
        duplicate_paths_same_label=len(same_label),
        duplicate_paths_conflicting_labels=len(conflicting),
        listed_files_missing=len(missing),
        files_not_in_label_file=len(unlisted),
        label_mismatches=mismatches,
        quarantined_images=int(frame["rel_path"].isin(bad_paths).sum()),
        has_split_column="split" in table.columns,
    )
    audit.files.append(summary)
    return summary


# --------------------------------------------------------------------------------------- CE-NBI
def _cenbi_norm(text) -> str:
    value = nfc(text).strip().lower()
    for char in ("’", "‘", "'", "`"):
        value = value.replace(char, "")
    value = re.sub(r"[\s\-]+", "_", value)
    return _CENBI_ALIASES.get(value, value)


def load_cenbi(source: str, root: Path, cfg: dict, audit: AuditLog) -> pd.DataFrame:
    """Bộ CE-NBI: <root>/{Benign,Malignant}/<Histology>/<PatientXXX>/<ảnh>.jpg + manifests/*.csv + bảng Excel."""
    root = Path(root)
    rows = []
    for path in scan_images(root):
        rel = normalize_rel(path.relative_to(root).as_posix())
        parts = rel.split("/")
        if len(parts) != 4:
            audit.add(source, "<thư_mục>", rel, "unexpected_layout", "ignored",
                      "cần dạng Benign|Malignant/<Histology>/<Patient>/<ảnh>")
            continue
        lesion_type, histology, patient, _ = parts
        rows.append({"source": source, "rel_path": rel, "image_path": str(path), "lesion_type": lesion_type,
                     "histology": histology, "patient_id": patient})
    if not rows:
        raise FileNotFoundError(f"[{source}] Không tìm thấy ảnh CE-NBI trong {root}")
    frame = pd.DataFrame(rows)
    frame["exclude_reason"] = ""
    label_from = cfg.get("label_from", "histology")
    if label_from not in ("histology", "lesion_type"):
        raise ValueError("label_from của CE-NBI phải là 'histology' hoặc 'lesion_type'")
    frame["label_raw"] = frame[label_from]

    # (1) File nhãn có sẵn: kiểm tra trùng dòng / trùng ảnh / lệch nhãn so với thư mục.
    by_path = frame.set_index("rel_path")
    default_label = by_path["histology"].map(lambda h: CENBI_10CLASS_MAP.get(h, h))
    label_dir = root / cfg.get("label_files_dir", "manifests")
    split_columns = {}
    if label_dir.is_dir():
        for file in sorted(label_dir.glob("*.csv")):
            table = pd.read_csv(file, dtype=str, keep_default_na=False)
            expected = {"histology": by_path["histology"], "lesion_type": by_path["lesion_type"],
                        "patient_id": by_path["patient_id"]}
            if "label" in table.columns:
                values = set(table["label"].astype(str).str.strip())
                expected["label"] = by_path["lesion_type"] if values <= {"Benign", "Malignant"} else default_label
            audit_label_table(table, f"{label_dir.name}/{file.name}", source, frame, audit, expected=expected)
            if {"image_path", "split"} <= set(table.columns):
                split_columns[file.name] = table.drop_duplicates("image_path").set_index("image_path")["split"]
    else:
        audit.add(source, str(label_dir), "", "label_files_missing", "warning", "không có file nhãn; dùng nhãn theo thư mục")
    if len(split_columns) >= 2:
        names = sorted(split_columns)
        for other in names[1:]:
            a, b = split_columns[names[0]], split_columns[other]
            common = a.index.intersection(b.index)
            disagree = int((a.loc[common] != b.loc[common]).sum())
            if disagree:
                audit.add(source, f"{names[0]} vs {other}", "split", "provided_splits_disagree", "ignored",
                          f"{disagree}/{len(common)} ảnh có split khác nhau giữa 2 file -> càng không dùng split có sẵn")

    # (2) Bảng bệnh nhân (Excel): đối chiếu chẩn đoán từng bệnh nhân với thư mục.
    table_name = cfg.get("patient_table", "Patients_List_Updated_Final.xlsx")
    if table_name:
        _audit_cenbi_patient_table(frame, root / table_name, source, audit, cfg)
    return apply_label_map(frame, cfg, source, audit)


def _audit_cenbi_patient_table(frame: pd.DataFrame, path: Path, source: str, audit: AuditLog, cfg: dict) -> None:
    if not path.exists():
        if cfg.get("patient_table_required", True):
            raise FileNotFoundError(f"[{source}] Thiếu bảng bệnh nhân {path} (đặt \"patient_table\": null để bỏ qua)")
        audit.add(source, path.name, "", "patient_table_missing", "warning")
        return
    max_rows = int(cfg.get("patient_table_max_rows", 50000))
    table = pd.read_excel(path, nrows=max_rows, dtype=str)
    needed = {"Patient", "Histopathology", "Type of Lesion"}
    if needed - set(table.columns):
        raise ValueError(f"[{source}] {path.name} thiếu cột {sorted(needed - set(table.columns))}")
    patient = table["Patient"].fillna("").astype(str).str.strip()
    valid = table[patient.str.fullmatch(r"Patient\d+")].copy()
    valid["Patient"] = patient[valid.index]
    skipped = len(table) - len(valid)
    if skipped:
        audit.add(source, path.name, "", "blank_or_non_patient_rows", "ignored", f"{skipped} dòng trống/không phải bệnh nhân")
    if len(table) == max_rows and patient.iloc[-1]:
        audit.add(source, path.name, "", "patient_table_maybe_truncated", "warning", f"chỉ đọc {max_rows} dòng đầu")

    bad: dict[str, str] = {}
    duplicated = valid[valid.duplicated("Patient", keep=False)]
    for pid, group in duplicated.groupby("Patient"):
        if group[["Histopathology", "Type of Lesion"]].drop_duplicates().shape[0] > 1:
            bad[pid] = "patient_table_conflicting_rows"
            audit.add(source, path.name, pid, "patient_table_conflicting_rows", "quarantine_patient")
        else:
            audit.add(source, path.name, pid, "patient_table_duplicate_row", "dropped_duplicate_row", f"{len(group)} dòng")
    reference = valid.drop_duplicates("Patient").set_index("Patient")

    for pid, group in frame.groupby("patient_id"):
        if pid in bad:
            continue
        if group["histology"].nunique() > 1 or group["lesion_type"].nunique() > 1:
            bad[pid] = "patient_in_multiple_classes"
            audit.add(source, path.name, pid, "patient_in_multiple_classes", "quarantine_patient",
                      ", ".join(sorted(group["histology"].unique())))
            continue
        if pid not in reference.index:
            bad[pid] = "patient_missing_from_reference_table"
            audit.add(source, path.name, pid, "patient_missing_from_reference_table", "quarantine_patient")
            continue
        row = reference.loc[pid]
        histology, lesion = group["histology"].iloc[0], group["lesion_type"].iloc[0]
        if _cenbi_norm(row["Histopathology"]) != _cenbi_norm(histology):
            bad[pid] = "histology_mismatch"
            audit.add(source, path.name, pid, "histology_mismatch", "quarantine_patient",
                      f"bảng={row['Histopathology']} | thư_mục={histology}")
        elif _cenbi_norm(row["Type of Lesion"]) != _cenbi_norm(lesion):
            bad[pid] = "lesion_type_mismatch"
            audit.add(source, path.name, pid, "lesion_type_mismatch", "quarantine_patient",
                      f"bảng={row['Type of Lesion']} | thư_mục={lesion} (chẩn đoán {row['Histopathology']})")
        count = row.get("Number of Images")
        try:
            expected_count = int(float(count)) if count not in (None, "", "nan") else None
        except ValueError:
            expected_count = None
        if expected_count is not None and expected_count != len(group):
            audit.add(source, path.name, pid, "image_count_mismatch", "warning_only",
                      f"bảng={expected_count} | thư_mục={len(group)}")
    for pid in sorted(set(reference.index) - set(frame["patient_id"])):
        audit.add(source, path.name, pid, "patient_without_images", "none")
    for pid, reason in bad.items():
        flag(frame, frame["patient_id"] == pid, f"quarantine:{reason}")
    audit.files.append({"source": source, "file": path.name, "rows": int(len(valid)), "status": "checked",
                        "duplicate_rows": int(valid.duplicated("Patient").sum()), "quarantined_patients": len(bad),
                        "quarantined_images": int(frame["patient_id"].isin(list(bad)).sum())})


# --------------------------------------------------------------------------------- ImageFolder
def load_imagefolder(source: str, root: Path, cfg: dict, audit: AuditLog) -> pd.DataFrame:
    """Thư mục dạng <root>/<lớp>/.../<ảnh>. Mã bệnh nhân lấy từ `patient_level` hoặc `patient_regex` nếu có.

    class_level: vị trí thư mục chứa tên lớp (0 = thư mục con đầu tiên). Ví dụ bộ có sẵn
    <root>/Training/<lớp>/... thì đặt class_level = 1 (thư mục Training/Testing bị bỏ qua, pipeline tự chia lại).
    """
    root = Path(root)
    level = int(cfg.get("class_level", 0))
    patient_level = cfg.get("patient_level")
    pattern = re.compile(cfg["patient_regex"]) if cfg.get("patient_regex") else None
    rows = []
    for path in scan_images(root):
        rel = normalize_rel(path.relative_to(root).as_posix())
        parts = rel.split("/")
        if len(parts) - 1 <= level:
            audit.add(source, "<thư_mục>", rel, "no_class_folder", "ignored")
            continue
        patient = ""
        if patient_level is not None and len(parts) - 1 > int(patient_level):
            patient = parts[int(patient_level)]
        elif pattern is not None:
            match = pattern.search(rel)
            patient = (match.group(1) if match.groups() else match.group(0)) if match else ""
        rows.append({"source": source, "rel_path": rel, "image_path": str(path), "label_raw": parts[level],
                     "patient_id": patient})
    if not rows:
        raise FileNotFoundError(f"[{source}] Không tìm thấy ảnh trong {root}")
    frame = pd.DataFrame(rows)
    frame["exclude_reason"] = ""
    if not (frame["patient_id"] != "").any():
        audit.add(source, "<thư_mục>", "", "no_patient_ids", "warning",
                  "không có mã bệnh nhân: chỉ gom được ảnh trùng/gần trùng; ảnh khác nhau của cùng người vẫn có thể rơi vào 2 tập")
    return apply_label_map(frame, cfg, source, audit)


# ----------------------------------------------------------------------------------------- CSV
def load_csv_source(source: str, root: Path, cfg: dict, audit: AuditLog) -> pd.DataFrame:
    """File nhãn dạng bảng (CSV/Excel): cột ảnh (tương đối so với root), cột nhãn, (tùy chọn) cột bệnh nhân."""
    root = Path(root)
    labels_file = Path(cfg["labels_file"])
    labels_file = labels_file if labels_file.is_absolute() else root / labels_file
    if labels_file.suffix.lower() in (".xlsx", ".xls"):
        table = pd.read_excel(labels_file, dtype=str).fillna("")
    else:
        table = pd.read_csv(labels_file, dtype=str, keep_default_na=False)
    image_col = cfg.get("image_column", "image_path")
    label_col = cfg.get("label_column", "label")
    patient_col = cfg.get("patient_column")
    for column in [image_col, label_col] + ([patient_col] if patient_col else []):
        if column not in table.columns:
            raise ValueError(f"[{source}] {labels_file.name} thiếu cột {column!r}")
    disk = {normalize_rel(p.relative_to(root).as_posix()): p for p in scan_images(root)}
    on_disk = pd.DataFrame({"rel_path": sorted(disk), "exclude_reason": ""})
    audit_label_table(table, labels_file.name, source, on_disk, audit, path_col=image_col, label_col=label_col,
                      unlisted_action="not_used_no_label")
    quarantined = set(on_disk.loc[on_disk["exclude_reason"] != "", "rel_path"])
    table = table.assign(_path=table[image_col].map(normalize_rel)).drop_duplicates("_path", keep="first")
    table = table[table["_path"].isin(disk.keys())]
    frame = pd.DataFrame({
        "source": source,
        "rel_path": table["_path"].to_numpy(),
        "image_path": [str(disk[p]) for p in table["_path"]],
        "label_raw": table[label_col].astype(str).str.strip().to_numpy(),
        "patient_id": table[patient_col].astype(str).str.strip().to_numpy() if patient_col else "",
    })
    frame["exclude_reason"] = ["quarantine:label_file_conflict" if p in quarantined else "" for p in frame["rel_path"]]
    if frame.empty:
        raise FileNotFoundError(f"[{source}] Không có ảnh nào trong {labels_file} tồn tại dưới {root}")
    return apply_label_map(frame, cfg, source, audit)


def _load_bagls_vf(source, root, cfg, audit):
    from .bagls import load_bagls_vf  # import trễ để tránh vòng import

    return load_bagls_vf(source, root, cfg, audit)


ADAPTERS = {"bagls_vf": _load_bagls_vf, "cenbi": load_cenbi, "imagefolder": load_imagefolder, "csv": load_csv_source}
