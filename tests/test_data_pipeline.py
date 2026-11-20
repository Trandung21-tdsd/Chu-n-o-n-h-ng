import numpy as np
import pandas as pd
import pytest
from PIL import Image

from throat_ai.audit import UnionFind, build_groups, near_duplicate_pairs
from throat_ai.data import load_prepared, partition
from throat_ai.preprocessing import PreprocessConfig, build_transform, preprocess_image
from throat_ai.sources import AuditLog, load_csv_source
from throat_ai.split import grouped_stratified_split, leakage_report, parse_ratios


def test_ratios_accept_valid_and_reject_110_percent():
    assert parse_ratios(["70", "15", "15"]) == pytest.approx((0.7, 0.15, 0.15))
    assert parse_ratios(["0.8", "0.1", "0.1"]) == pytest.approx((0.8, 0.1, 0.1))
    with pytest.raises(ValueError, match="110"):
        parse_ratios(["80", "20", "10"])


def test_grouped_split_keeps_groups_together_and_covers_classes():
    rows = []
    for c in "ABC":
        for i in range(20):
            rows.append({"group_id": f"{c}{i // 2}", "unit_id": f"{c}{i}", "label": c, "n_images": 1 + i % 3})
    units = pd.DataFrame(rows)
    mapping, info = grouped_stratified_split(units, (0.7, 0.15, 0.15), seed=3)
    assert set(mapping.values()) == {"train", "val", "test"}
    units["split"] = units["group_id"].map(mapping)
    for split, group in units.groupby("split"):
        assert set(group["label"]) == {"A", "B", "C"}, split
    assert mapping == grouped_stratified_split(units.drop(columns="split"), (0.7, 0.15, 0.15), seed=3)[0]
    assert info["class_missing_in_some_split"] == 0


def test_near_duplicate_pigeonhole_search():
    base = 0x0123456789ABCDEF

    def flip(bits):
        return f"{base ^ sum(1 << b for b in bits):016x}"

    pairs = near_duplicate_pairs([f"{base:016x}", flip([0, 17, 33, 60]), flip([1, 2, 3, 4, 5])], radius=4)
    assert (0, 1, 4) in pairs
    assert not any(set(p[:2]) == {0, 2} for p in pairs)
    assert near_duplicate_pairs([f"{base:016x}"] * 2, radius=-1) == []


def test_images_without_patient_are_not_merged():
    frame = pd.DataFrame({"patient_key": ["", "", "s::P1", "s::P1"], "pixel_hash": ["a", "b", "c", "d"]})
    groups = build_groups(frame, pairs=[])
    assert groups[0] != groups[1] and groups[2] == groups[3]
    union = UnionFind(4)
    union.union(0, 1)
    union.union(2, 3)
    union.union(1, 3)
    assert len({union.find(i) for i in range(4)}) == 1


def test_prepare_audits_duplicates_label_files_and_patient_table(prepared):
    root, summary = prepared
    reasons = summary["removed_by_reason"]
    assert reasons["quarantine:same_pixels_different_labels"] == 2  # (c) cả 2 bản
    assert reasons["quarantine:lesion_type_mismatch"] == 3  # (d) Patient024
    assert reasons["quarantine:label_file_conflict"] == 2  # (e2) + (e3)
    assert summary["exact_duplicate_groups"] == 2 and summary["exact_duplicate_extra_copies"] == 2
    issues = pd.read_csv(root / "toy" / "label_issues.csv")
    found = set(issues["issue"])
    for expected in ("duplicate_row", "duplicate_path_conflicting_labels", "label_mismatch",
                     "provided_split_ignored", "provided_splits_disagree", "lesion_type_mismatch",
                     "same_pixels_different_labels", "blank_or_non_patient_rows", "image_count_mismatch"):
        assert expected in found, expected
    proper = summary["leakage_check_proper_split"]
    assert all(proper[k] == 0 for k in proper if k.endswith("_crossing_splits"))


def test_cross_patient_duplicate_forces_same_split(prepared):
    root, _ = prepared
    frame, task, sha = load_prepared(root / "toy")
    splits = frame[frame["patient_id"].isin(["Patient002", "Patient003"])]["split"]
    assert splits.nunique() == 1
    assert len(sha) == 64 and task["classes"] == ["Cyst", "Polyp", "SCC"]


def test_duplicate_policies_share_identical_val_and_test(prepared):
    root, _ = prepared
    frame, _, _ = load_prepared(root / "toy")
    for split in ("val", "test"):
        pd.testing.assert_frame_equal(partition(frame, split, "dedup"), partition(frame, split, "keep"))
    assert len(partition(frame, "train", "keep")) >= len(partition(frame, "train", "dedup"))
    assert partition(frame, "train", "keep")["pixel_hash"].duplicated().any() or \
        not frame.loc[frame["split"] == "train", "pixel_hash"].duplicated().any()
    naive = leakage_report(frame, "split_naive")
    assert naive["test_images"] > 0


def test_preprocessing_clahe_and_letterbox():
    gradient = np.tile(np.linspace(60, 160, 120, dtype=np.uint8), (60, 1))
    image = Image.fromarray(np.stack([gradient] * 3, axis=-1))
    plain = preprocess_image(image, PreprocessConfig(resolution=64))
    clahe = preprocess_image(image, PreprocessConfig(resolution=64, clahe=True))
    assert plain.shape == clahe.shape == (64, 64, 3)
    assert not np.array_equal(plain, clahe)
    assert plain[:8].max() == 0 and plain[-8:].max() == 0  # ảnh 2:1 -> đệm đen trên/dưới
    tensor = build_transform(train=False)(plain)
    assert tuple(tensor.shape) == (3, 64, 64)
    augmented = build_transform(train=True, augment={"hflip": True, "rotate": 10})(plain)
    assert tuple(augmented.shape) == (3, 64, 64)


def test_csv_source_handles_duplicate_and_conflicting_rows(tmp_path):
    for name in ("a.png", "b.png", "c.png"):
        Image.fromarray(np.zeros((70, 70, 3), dtype=np.uint8)).save(tmp_path / name)
    pd.DataFrame({
        "image_path": ["a.png", "a.png", "b.png", "b.png", "c.png", "missing.png"],
        "label": ["x", "x", "x", "y", "y", "x"],
        "patient_id": ["p1", "p1", "p2", "p2", "p3", "p4"],
    }).to_csv(tmp_path / "labels.csv", index=False)
    audit = AuditLog()
    frame = load_csv_source("s", tmp_path, {"labels_file": "labels.csv", "patient_column": "patient_id"}, audit)
    assert len(frame) == 3
    assert frame.set_index("rel_path").loc["b.png", "exclude_reason"] == "quarantine:label_file_conflict"
    issues = set(audit.issues_frame()["issue"])
    assert {"duplicate_row", "duplicate_path_conflicting_labels", "listed_file_missing"} <= issues


def test_near_duplicate_search_skips_pairs_already_in_same_group():
    same = "0123456789abcdef"
    assert near_duplicate_pairs([same, same, same], radius=2, group_keys=["v1", "v1", "v2"]) == [(0, 2, 0), (1, 2, 0)]
