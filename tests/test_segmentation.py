"""Phân đoạn BAGLS-VF: mask màu -> nhãn, lật ngang đổi trái/phải, số đo thanh môn, manifest không rò rỉ, chạy thử toàn bộ."""
import json

import numpy as np
import pandas as pd
import pytest
from PIL import Image

from throat_ai import demo, seg_benchmark
from throat_ai.glottis import assess_video, image_features, sign_classifier_proba, video_features
from throat_ai.prepare import prepare_task
from throat_ai.seg_data import (
    GLOTTIS, LEFT, RIGHT, SEG_CONFIG, augment_pair, build_seg_manifest, decode_mask, encode_mask, mask_path_for,
)

STATUSES = ["healthy"] * 6 + ["Muscle tension dysphonia"] * 4 + ["Vocal fold nodules", "Polyp", "carcinoma", "cyst"]


def _draw(seed: int, size=(64, 64), gap: int = 4, hourglass: bool = False, video: int = 0):
    """Ảnh xám + mask màu: dây thanh trái (xanh lá), phải (xanh dương), thanh môn đỏ ở giữa (gap=0 => khép).
    Mỗi video có nền riêng (giống thật: video khác nhau thì cảnh khác nhau)."""
    h, w = size
    rng = np.random.default_rng(seed)
    texture = np.asarray(Image.fromarray(np.random.default_rng(video + 7).normal(0, 1, (4, 4)).astype(np.float32))
                         .resize((w, h), Image.Resampling.BILINEAR))
    index = np.zeros((h, w), dtype=np.uint8)
    mid = w // 2
    index[8:h - 8, mid - 14:mid - gap // 2] = LEFT
    index[8:h - 8, mid + (gap + 1) // 2:mid + 14] = RIGHT
    if gap:
        index[8:h - 8, mid - gap // 2:mid + (gap + 1) // 2] = GLOTTIS
        if hourglass:  # khối chạm nhau ở giữa -> 2 khe rời
            index[h // 2 - 3:h // 2 + 3, mid - gap // 2:mid + (gap + 1) // 2] = LEFT
    gray = np.choose(index, [90, 160, 160, 25]).astype(np.float32) + 45 * texture + rng.normal(0, 8, (h, w))
    image = Image.fromarray(np.clip(gray, 0, 255).astype(np.uint8)).convert("RGB")
    return image, Image.fromarray(encode_mask(index))


@pytest.fixture(scope="module")
def seg_data(tmp_path_factory):
    root = tmp_path_factory.mktemp("bagls_seg")
    dirs = {k: root / v for k, v in {"ti": "BAGLS-VF_train/BAGLS_image", "tm": "BAGLS-VF_train/BAGLS_mask",
                                     "ei": "BAGLS-VF_test/test_images", "em": "BAGLS-VF_test/test_masks",
                                     "mt": "archive-2/training/training", "me": "archive-2/test/test"}.items()}
    for folder in dirs.values():
        folder.mkdir(parents=True)
    index = 0
    for video, status in enumerate(STATUSES + ["", "None (after surgery)"]):
        for frame in range(6):
            index += 1
            image, mask = _draw(index, gap=(frame % 3) * 3, hourglass=status == "Vocal fold nodules", video=video)
            name = f"ID{video}_{frame}_img_old_{index}.png"
            image.save(dirs["ti"] / name)
            mask.save(dirs["tm"] / f"mask_{name}")
            (dirs["mt"] / f"{index}.meta").write_text(json.dumps({"Video Id": video, "Subject disorder status": status}))
    for k, status in enumerate(["healthy", "healthy", "Muscle tension dysphonia", "Polyp", "Vocal fold nodules"]):
        for frame in range(5):
            idx = k * 5 + frame
            image, mask = _draw(10_000 + idx, size=(96, 64), gap=(frame % 3) * 3, video=900 + k)  # ảnh test cao hơn rộng
            image.save(dirs["ei"] / f"{idx}.png")
            mask.save(dirs["em"] / f"{idx}.png")
            if k == 0:  # video train 50 = CÙNG cảnh quay với video test 900 (khác mã) -> phải bị loại khỏi train/val
                index += 1
                name = f"ID50_{frame}_img_old_{index}.png"
                image.crop((0, 16, 64, 80)).save(dirs["ti"] / name)
                mask.crop((0, 16, 64, 80)).save(dirs["tm"] / f"mask_{name}")
                (dirs["mt"] / f"{index}.meta").write_text(json.dumps({"Video Id": 50, "Subject disorder status": "healthy"}))
            (dirs["me"] / f"{idx}.meta").write_text(json.dumps({"Video Id": 900 + k, "Subject disorder status": status}))
    task = {
        "description": "test", "anatomy": "larynx", "modality": "HSV",
        "sources": [
            {"name": "bagls_vf_train", "adapter": "bagls_vf", "part": "train", "root": str(root / "BAGLS-VF_train"),
             "meta_root": str(root / "archive-2"), "label_scheme": "3class"},
            {"name": "bagls_vf_test", "adapter": "bagls_vf", "part": "test", "root": str(root / "BAGLS-VF_test"),
             "meta_root": str(root / "archive-2"), "label_scheme": "3class"},
        ],
        "preprocess": {"geometry": "center_square"}, "phash_radius": -1, "max_images_per_unit": 6,
        "scene_link_threshold": 0.99,
        "display_names": {"Normal": "BT", "Organic_lesion": "TT", "Functional": "CN"},
    }
    prepared = tmp_path_factory.mktemp("prep")
    prepare_task("bagls_toy", task, prepared, ratios=(0.75, 0.25, 0.0), seed=1, workers=2)
    cfg = json.loads(SEG_CONFIG.read_text(encoding="utf-8"))
    cfg.update(classifier_task="bagls_toy", meta_root=str(root / "archive-2"), max_frames_train=6, max_frames_val=6)
    cfg["train"]["root"], cfg["test"]["root"] = str(root / "BAGLS-VF_train"), str(root / "BAGLS-VF_test")
    cfg_path = root / "segmentation.json"
    cfg_path.write_text(json.dumps(cfg), encoding="utf-8")
    return root, prepared, cfg, cfg_path


def test_mask_colors_decode_and_flip_swaps_sides():
    index = np.array([[0, 1, 2, 3]], dtype=np.uint8)
    assert np.array_equal(decode_mask(encode_mask(index)), index)
    assert np.array_equal(decode_mask(np.array([[0, 255]], dtype=np.uint8)), [[0, GLOTTIS]])  # mask nhị phân

    class Flip:  # luôn lật, không xoay/co giãn
        def random(self):
            return 0.0

        def uniform(self, a, b):
            return 1.0 if (a, b) == (0.9, 1.1) else 0.0

    image = np.zeros((8, 8, 3), np.uint8)
    mask = np.zeros((8, 8), np.uint8)
    mask[:, :3], mask[:, 5:] = LEFT, RIGHT
    _, flipped = augment_pair(image, mask, {"hflip": True, "rotate": 0, "scale": (0.9, 1.1), "translate": 0,
                                            "brightness": 0, "contrast": 0}, Flip())
    assert (flipped[:, :3] == LEFT).all() and (flipped[:, 5:] == RIGHT).all()  # vẫn: xanh lá ở bên trái ảnh


def test_glottis_features_and_video_signs():
    _, open_mask = _draw(1, gap=6)
    _, closed_mask = _draw(2, gap=0)
    _, hourglass = _draw(3, gap=6, hourglass=True)
    f_open = image_features(decode_mask(np.asarray(open_mask)))
    f_closed = image_features(decode_mask(np.asarray(closed_mask)))
    f_hour = image_features(decode_mask(np.asarray(hourglass)))
    assert not f_open["closed"] and f_open["glottis_ratio"] > 0.1 and f_open["gap_parts"] == 1
    assert f_closed["closed"] and f_closed["glottis_ratio"] == 0
    assert f_hour["gap_parts"] == 2 and f_open["fold_asymmetry"] < 0.1
    normal = video_features([f_open, f_closed] * 3)
    never_closes = video_features([f_open] * 6)
    assert normal["closed_fraction"] == 0.5 and never_closes["min_glottis_ratio"] > normal["min_glottis_ratio"]
    ranges = {k: {"p05": 0.0, "p50": 0.0, "p95": 0.05} for k in ("min_glottis_ratio",)}
    assert "KHÔNG KHÉP KÍN" in assess_video(never_closes, ranges)["signs"][0]
    model = {"classes": ["A", "B", "C"], "mean": [0] * 7, "scale": [1] * 7, "coef": [[0] * 7, [0, 50] + [0] * 5, [0] * 7],  # trọng số trên min_glottis_ratio
             "intercept": [0, 0, 0]}
    proba = sign_classifier_proba(model, never_closes)
    assert abs(sum(proba.values()) - 1) < 1e-9 and max(proba, key=proba.get) == "B"


def test_seg_manifest_reuses_video_split_without_leakage(seg_data):
    root, prepared, cfg, _ = seg_data
    path = build_seg_manifest(cfg, prepared, overwrite=True)
    seg = pd.read_csv(path, dtype=str, keep_default_na=False)
    assert (seg.groupby("video_id")["split"].nunique() == 1).all()
    clf = pd.read_csv(prepared / "bagls_toy" / "manifest.csv", dtype=str, keep_default_na=False)
    clf_split = clf.drop_duplicates("patient_id").set_index("patient_id")["split"]
    for video, split in seg.drop_duplicates("video_id").set_index("video_id")["split"].items():
        if video in clf_split:
            assert clf_split[video] == split  # cùng phân chia với model phân loại
        else:
            assert split == "train"  # video không có nhãn bệnh chỉ dùng để train phân đoạn
    assert set(seg.loc[seg["split"] == "test", "video_id"]) == {f"V{900 + k}" for k in range(5)}
    # video train 50 cùng cảnh quay với video test 900 -> bị loại ở CẢ 2 manifest (phân loại + phân đoạn)
    assert "V50" not in set(seg["video_id"]) and "V50" not in set(clf["patient_id"])
    scene = pd.read_csv(path.parent / "duplicates_same_scene_videos.csv")
    assert {tuple(sorted(r)) for r in scene[["video_a", "video_b"]].values} == {("V50", "V900")}
    removed = pd.read_csv(prepared / "bagls_toy" / "removed_images.csv")
    assert (removed.loc[removed["patient_id"] == "V50", "exclude_reason"] == "quarantine:duplicate_of_fixed_test_image").all()
    assert (seg.loc[seg["label"] == "", "split"] == "train").all() and (seg["label"] == "").sum() == 12
    sample = seg.iloc[0]
    assert str(mask_path_for(sample["image_path"], cfg)) == sample["mask_path"]


def test_seg_benchmark_end_to_end_and_demo(seg_data, tmp_path, capsys):
    root, prepared, cfg, cfg_path = seg_data
    out = tmp_path / "seg"
    seg_benchmark.main(["--config", str(cfg_path), "--prepared-dir", str(prepared), "--out", str(out),
                        "--models", "unet_lite", "--clahe", "off", "on", "--epochs", "2", "--patience", "1",
                        "--batch-size", "4", "--workers", "0", "--resolution", "64", "--no-pretrained",
                        "--device", "cpu", "--target-quality", "0.0"])
    results = pd.read_csv(out / "benchmark_results.csv")
    assert len(results) == 2 and set(results["status"]) == {"completed"} and results["selected"].sum() == 1
    selection = json.loads((out / "selection.json").read_text(encoding="utf-8"))
    assert selection["test_used_for_selection"] is False and 0 <= selection["test_metrics"]["test_mean_dice"] <= 1
    history = pd.read_csv(next(out.glob("runs/*/history.csv")))
    assert {"train_loss", "val_loss", "val_mean_dice", "val_dice_glottis", "timestamp_unix"} <= set(history.columns)
    for name in ("reference_ranges.json", "test_video_signs.csv", "features_by_group.csv", "sign_classifier.json",
                 "comparison.md", "model_comparison.csv", "tato/static_profiling_dataset_extended.csv",
                 "tato/online_observations.csv"):
        assert (out / name).exists(), name
    assert len(pd.read_csv(out / "test_video_signs.csv")) == 5
    online = pd.read_csv(out / "tato/online_observations.csv")
    assert online["observed_validation_macro_f1_at_epoch_e"].notna().all()  # = val mean Dice
    # chạy lại: run đã xong được bỏ qua
    seg_benchmark.main(["--config", str(cfg_path), "--prepared-dir", str(prepared), "--out", str(out),
                        "--models", "unet_lite", "--clahe", "off", "--epochs", "2", "--patience", "1",
                        "--batch-size", "4", "--workers", "0", "--resolution", "64", "--no-pretrained",
                        "--device", "cpu", "--target-quality", "0.0", "--no-finalize"])
    assert len(list(out.glob("runs/*/summary.json"))) == 2

    capsys.readouterr()
    demo.main(["--no-classifier", "--seg-model", str(out), "--prepared-dir", str(prepared), "--n", "2", "--seed", "0",
               "--device", "cpu"])
    printed = capsys.readouterr().out
    assert "BAGLS-VF_test" in printed and "Phân đoạn – frame này" in printed
    record = json.loads(sorted((out / "demo").glob("*.json"))[0].read_text(encoding="utf-8"))
    assert record["segmentation"]["dice_vs_ground_truth"] is not None
    assert record["segmentation"]["video"]["n_frames"] == 5
