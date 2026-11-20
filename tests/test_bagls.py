"""BAGLS-VF: ghép metadata -> nhãn, test cố định, bớt frame, cắt ảnh giống cách tạo BAGLS-VF."""
import json

import numpy as np
import pandas as pd
import pytest
from PIL import Image

from throat_ai.bagls import classify_status
from throat_ai.data import load_prepared, partition
from throat_ai.prepare import prepare_task
from throat_ai.preprocessing import PreprocessConfig, preprocess_image

STATUSES = ["healthy"] * 6 + ["Muscle tension dysphonia"] * 4 + ["Vocal fold nodules", "Polyp", "carcinoma", "cyst"]


def _frame(seed, shape):
    rng = np.random.default_rng(seed)
    return Image.fromarray(rng.integers(0, 255, size=shape, dtype=np.uint8)).convert("RGB")


@pytest.fixture(scope="module")
def bagls(tmp_path_factory):
    root = tmp_path_factory.mktemp("TATO")
    train_dir, test_dir = root / "BAGLS-VF_train" / "BAGLS_image", root / "BAGLS-VF_test" / "test_images"
    meta_train, meta_test = root / "archive-2/training/training", root / "archive-2/test/test"
    for folder in (train_dir, test_dir, meta_train, meta_test):
        folder.mkdir(parents=True)
    index = 0
    for video, status in enumerate(STATUSES + ["", "None (after surgery)"]):
        for frame in range(30):  # 30 frame / video
            index += 1
            _frame(index, (256, 256)).save(train_dir / f"ID{video}_{frame}_img_old_{index}.png")
            (meta_train / f"{index}.meta").write_text(json.dumps({"Video Id": video, "Subject disorder status": status}))
    test_status = ["healthy", "healthy", "Muscle tension dysphonia", "Polyp", "Vocal fold nodules (high phonation)"]
    for k, status in enumerate(test_status):
        for frame in range(5):
            idx = k * 5 + frame
            _frame(10_000 + idx, (512, 256)).save(test_dir / f"{idx}.png")  # ảnh test gốc 256 rộng x 512 cao
            (meta_test / f"{idx}.meta").write_text(json.dumps({"Video Id": 900 + k, "Subject disorder status": status}))
    task = {
        "description": "test", "anatomy": "larynx", "modality": "HSV",
        "sources": [
            {"name": "bagls_vf_train", "adapter": "bagls_vf", "part": "train", "root": str(root / "BAGLS-VF_train"),
             "meta_root": str(root / "archive-2"), "label_scheme": "3class"},
            {"name": "bagls_vf_test", "adapter": "bagls_vf", "part": "test", "root": str(root / "BAGLS-VF_test"),
             "meta_root": str(root / "archive-2"), "label_scheme": "3class"},
        ],
        "preprocess": {"geometry": "center_square"}, "phash_radius": -1, "max_images_per_unit": 10,
        "display_names": {"Normal": "BT", "Organic_lesion": "TT", "Functional": "CN"},
    }
    prepared = tmp_path_factory.mktemp("prep")
    summary = prepare_task("bagls_toy", task, prepared, ratios=(0.8, 0.2, 0.0), seed=1, workers=2)
    return root, prepared, summary


def test_status_text_mapped_by_lecture_groups():
    assert classify_status("healthy") == ("Normal", "normal")
    assert classify_status("Muscle tension dysphonia with nodules") == ("Organic_lesion", "nodules")
    assert classify_status("Muscle tension dysphonia") == ("Functional", "muscle_tension_dysphonia")
    assert classify_status("Vocal insufficiency and contact granuloma") == ("Organic_lesion", "granuloma")
    assert classify_status("Extreme polipoid pendulating edema")[1] == "reinke_edema"
    assert classify_status("Posterior insufficient glottic closure") == ("Functional", "glottic_insufficiency")
    assert classify_status("")[0] == "" and classify_status("None (half year after)")[0] == ""


def test_fixed_test_subsample_and_no_leakage(bagls):
    _, prepared, summary = bagls
    frame, task, _ = load_prepared(prepared / "bagls_toy")
    assert task["classes"] == ["Functional", "Normal", "Organic_lesion"]
    test = frame[frame["split"] == "test"]
    assert set(test["source"]) == {"bagls_vf_test"} and len(test) == 25  # test giữ nguyên mọi frame
    assert set(frame.loc[frame["split"] != "test", "source"]) == {"bagls_vf_train"}
    per_video = frame[frame["split"] != "test"].groupby("patient_key").size()
    assert per_video.max() == 10  # bớt còn 10 frame / video
    assert (frame.groupby("patient_key")["split"].nunique() == 1).all()  # 1 video chỉ ở 1 tập
    assert set(frame["split"]) == {"train", "val", "test"}
    removed = pd.read_csv(prepared / "bagls_toy" / "removed_images.csv")
    assert (removed["exclude_reason"] == "excluded:disorder_status_unknown_or_ambiguous").sum() == 60  # 2 video
    detail = frame.drop_duplicates("patient_key").set_index("patient_id")["histology"]
    assert detail["V10"] == "nodules" and detail["V12"] == "carcinoma"
    assert task["preprocess"] == {"geometry": "center_square"}
    assert len(partition(frame, "test", "dedup")) == 25


def test_center_square_reproduces_bagls_vf_crop():
    original = _frame(5, (512, 256))  # cao 512, rộng 256 (giống ảnh BAGLS gốc)
    vf_train_style = original.crop((0, 128, 256, 384))  # cách BAGLS-VF tạo ảnh train
    a = preprocess_image(original, PreprocessConfig(resolution=256, geometry="center_square"))
    b = preprocess_image(vf_train_style, PreprocessConfig(resolution=256, geometry="center_square"))
    assert a.shape == (256, 256, 3) and np.array_equal(a, b)
