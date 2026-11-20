"""Bộ dữ liệu giả lập kiểu CE-NBI, cài sẵn các lỗi để kiểm tra audit/duplicate/leakage (không dùng dữ liệu thật)."""
from __future__ import annotations

import shutil

import numpy as np
import pandas as pd
import pytest
from PIL import Image

CLASSES = {("Benign", "Cyst"): (200, 70, 70), ("Benign", "Polyp"): (70, 200, 70), ("Malignant", "SCC"): (70, 70, 200)}
PATIENTS_PER_CLASS = 8
IMAGES_PER_PATIENT = 3


def _image(color, seed: int) -> Image.Image:
    rng = np.random.default_rng(seed)
    base = np.array(color, dtype=np.float32)[None, None, :]
    array = np.clip(base + rng.normal(0, 30, size=(80, 96, 3)), 0, 255).astype(np.uint8)
    return Image.fromarray(array)


@pytest.fixture(scope="session")
def cenbi_root(tmp_path_factory):
    root = tmp_path_factory.mktemp("data") / "CENBI"
    histology_rows, patients, pid = [], [], 0
    for (lesion, histology), color in CLASSES.items():
        for _ in range(PATIENTS_PER_CLASS):
            pid += 1
            name = f"Patient{pid:03d}"
            folder = root / lesion / histology / name
            folder.mkdir(parents=True)
            for k in range(1, IMAGES_PER_PATIENT + 1):
                _image(color, pid * 100 + k).save(folder / f"P{pid:03d} ({k}).jpg", quality=95)
            patients.append({"Patient": name, "Histopathology": histology, "Type of Lesion": lesion,
                             "Leukoplakia": "No", "Number of Images": IMAGES_PER_PATIENT, "Image Size": "96*80"})
    # (a) trùng y hệt, cùng bệnh nhân, cùng nhãn
    shutil.copy(root / "Benign/Cyst/Patient001/P001 (1).jpg", root / "Benign/Cyst/Patient001/P001 (99).jpg")
    # (b) trùng y hệt giữa 2 bệnh nhân cùng nhãn -> phải gom 2 bệnh nhân vào cùng 1 tập
    shutil.copy(root / "Benign/Cyst/Patient002/P002 (1).jpg", root / "Benign/Cyst/Patient003/P003 (97).jpg")
    # (c) trùng y hệt nhưng KHÁC nhãn -> cách ly cả 2 bản
    shutil.copy(root / "Benign/Polyp/Patient009/P009 (1).jpg", root / "Malignant/SCC/Patient017/P017 (98).jpg")
    # (d) Excel ghi sai loại tổn thương cho Patient024 -> cách ly cả bệnh nhân
    patients[-1]["Type of Lesion"] = "Benign"
    patients.append({"Patient": "Ghi chú: tổng 24 bệnh nhân"})  # dòng không phải bệnh nhân trong Excel

    ten_class = {"Cyst": "Cyst", "Polyp": "Polyp", "SCC": "SCC"}
    for path in sorted(root.rglob("*.jpg")):
        rel = path.relative_to(root).as_posix()
        lesion, histology, patient, _ = rel.split("/")
        histology_rows.append({"image_path": rel, "label": ten_class[histology], "patient_id": patient,
                               "split": "train", "lesion_type": lesion, "histology": histology})
    table = pd.DataFrame(histology_rows)
    extra = [
        table.iloc[5].to_dict(),  # (e1) dòng trùng hoàn toàn -> bỏ bản thừa
        {**table[table.image_path == "Benign/Polyp/Patient010/P010 (2).jpg"].iloc[0].to_dict(), "label": "SCC"},  # (e2) mâu thuẫn
    ]
    table = pd.concat([table, pd.DataFrame(extra)], ignore_index=True)
    table.loc[table.image_path == "Benign/Cyst/Patient004/P004 (3).jpg", "label"] = "Polyp"  # (e3) lệch nhãn thư mục
    (root / "manifests").mkdir()
    table.to_csv(root / "manifests" / "larynx_histology.csv", index=False)
    binary = table.drop_duplicates("image_path")[["image_path", "patient_id", "lesion_type"]].rename(columns={"lesion_type": "label"})
    binary["split"] = "test"
    binary.to_csv(root / "manifests" / "larynx_binary.csv", index=False)
    pd.DataFrame(patients).to_excel(root / "Patients_List_Updated_Final.xlsx", index=False)
    return root


@pytest.fixture(scope="session")
def cenbi_task(cenbi_root):
    return {
        "description": "test",
        "anatomy": "larynx",
        "modality": "CE-NBI",
        "sources": [{"name": "cenbi", "adapter": "cenbi", "root": str(cenbi_root), "label_from": "histology",
                     "label_map": {"Cyst": "Cyst", "Polyp": "Polyp", "SCC": "SCC"}}],
        "augment": {"hflip": True, "vflip": True, "rotate": 10},
        "display_names": {"Cyst": "Nang", "Polyp": "Polyp", "SCC": "Ung thư"},
    }


@pytest.fixture(scope="session")
def prepared(cenbi_task, tmp_path_factory):
    from throat_ai.prepare import prepare_task

    root = tmp_path_factory.mktemp("prepared")
    summary = prepare_task("toy", cenbi_task, root, ratios=(0.7, 0.15, 0.15), seed=7, phash_radius=4, workers=2)
    return root, summary
