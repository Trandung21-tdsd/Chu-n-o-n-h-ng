"""BƯỚC 1 — Audit dữ liệu, xử lý duplicate, chia train/val/test theo bệnh nhân (chống data leakage).

Ví dụ:
    python -m throat_ai.prepare --task bagls_3class                    # train: BAGLS-VF_train, test: BAGLS-VF_test
    python -m throat_ai.prepare --task bagls_3class --split 80 20 0 --overwrite   # train:val = 80:20 trong BAGLS-VF_train
    python -m throat_ai.prepare --task bagls_3class --source-root bagls_vf_train=D:/data/BAGLS-VF_train

Không sửa/xóa bất kỳ file nào trong thư mục dữ liệu gốc. Kết quả ghi vào data/prepared/<task>/.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from .audit import build_groups, hash_images, mark_exact_duplicates, near_duplicate_pairs, scene_links
from .sources import ADAPTERS, AuditLog, flag
from .split import grouped_stratified_split, leakage_report, naive_split, parse_ratios
from .utils import (
    DEFAULT_PREPARED_DIR,
    DEFAULT_TASKS_FILE,
    get_task,
    iso_now,
    log,
    parse_key_values,
    resolve_cli_path,
    resolve_config_path,
    setup_console,
    sha256_file,
    warn,
    write_csv,
    write_json,
)

MANIFEST_COLUMNS = [
    "source", "rel_path", "image_path", "label", "label_raw", "patient_id", "patient_key", "unit_id",
    "lesion_type", "histology", "sha256", "pixel_hash", "phash", "width", "height",
    "dup_count", "canonical", "group_id", "fixed_split", "split", "split_naive",
]
REMOVED_COLUMNS = ["source", "rel_path", "image_path", "label_raw", "label", "patient_id", "exclude_reason", "error"]


def _join_unique(values) -> str:
    return ";".join(sorted({str(v) for v in values if str(v)}))


def prepare_task(
    task_name: str,
    task: dict,
    prepared_root: Path = DEFAULT_PREPARED_DIR,
    ratios=(0.70, 0.15, 0.15),
    seed: int = 42,
    phash_radius: int | None = None,
    source_roots: dict | None = None,
    max_images_per_unit: int | None = None,
    workers: int = 8,
    overwrite: bool = False,
) -> dict:
    out_dir = Path(prepared_root) / task_name
    manifest_path = out_dir / "manifest.csv"
    if manifest_path.exists() and not overwrite:
        raise FileExistsError(
            f"Đã có {manifest_path} (split đã cố định). Thêm --overwrite nếu thật sự muốn chia lại — "
            "khi đó các kết quả benchmark cũ không còn so sánh được với kết quả mới."
        )
    out_dir.mkdir(parents=True, exist_ok=True)
    source_roots = source_roots or {}

    # ------------------------------------------------------------------ 1. Đọc nguồn + audit file nhãn
    audit, frames, sources_used = AuditLog(), [], []
    for source in task["sources"]:
        if not source.get("enabled", True):
            continue
        name, adapter = source["name"], source["adapter"]
        if adapter not in ADAPTERS:
            raise ValueError(f"[{name}] adapter {adapter!r} không tồn tại. Có: {', '.join(ADAPTERS)}")
        root = resolve_cli_path(source_roots[name]) if name in source_roots else resolve_config_path(source["root"])
        if not root.exists():
            raise FileNotFoundError(
                f"[{name}] Không thấy thư mục dữ liệu {root}. Tải dữ liệu về rồi chỉ đường dẫn: "
                f"--source-root {name}=<đường_dẫn> (hoặc sửa 'root' trong configs/tasks.json)."
            )
        log(f"Đọc nguồn '{name}' ({adapter}) tại {root}")
        frames.append(ADAPTERS[adapter](name, root, source, audit))
        sources_used.append({"name": name, "adapter": adapter, "root": str(root)})
    if not frames:
        raise ValueError("Task không có nguồn dữ liệu nào đang bật (enabled).")
    data = pd.concat(frames, ignore_index=True)
    for column in ("lesion_type", "histology", "patient_id", "fixed_split"):
        if column not in data.columns:
            data[column] = ""
        data[column] = data[column].fillna("").astype(str)
    if data.duplicated(["source", "rel_path"]).any():
        raise ValueError("Trùng (source, rel_path) — tên nguồn trong task phải khác nhau.")
    data["patient_key"] = np.where(data["patient_id"] != "", data["source"] + "::" + data["patient_id"], "")
    n_found = len(data)
    if phash_radius is None:
        phash_radius = int(task.get("phash_radius", 4))
    if max_images_per_unit is None:
        max_images_per_unit = int(task.get("max_images_per_unit", 0))

    # Video tốc độ cao: hàng trăm frame gần giống nhau/1 video -> lấy đều N frame mỗi video cho train/val
    # (giảm chi phí train nhiều lần, gần như không mất thông tin). Tập test cố định giữ nguyên.
    if max_images_per_unit and max_images_per_unit > 0:
        pool = data[(data["exclude_reason"] == "") & (data["fixed_split"] != "test") & (data["patient_key"] != "")]
        drop = []
        for _, group in pool.sort_values("rel_path").groupby("patient_key"):
            if len(group) > max_images_per_unit:
                chosen = set(np.linspace(0, len(group) - 1, max_images_per_unit).round().astype(int))
                drop += [idx for pos, idx in enumerate(group.index) if pos not in chosen]
        flag(data, data.index.isin(drop), "excluded:frame_subsample")
        log(f"Lấy tối đa {max_images_per_unit} frame/video cho train/val: bỏ bớt {len(drop)} frame gần giống nhau.")

    # ------------------------------------------------------------------ 2. Đọc ảnh: lỗi, SHA-256, pixel hash, pHash
    data = data.reset_index(drop=True)
    needed = data["exclude_reason"] == ""  # ảnh đã bị loại (vd. bớt frame) không cần băm
    hashes = pd.DataFrame({"sha256": "", "pixel_hash": "", "phash": "", "width": 0, "height": 0, "error": ""},
                          index=data.index)
    computed = hash_images(data.loc[needed, "image_path"].tolist(), Path(prepared_root) / "_cache" / "image_hashes.csv", workers)
    computed.index = data.index[needed]
    hashes.loc[needed, computed.columns] = computed
    data = pd.concat([data, hashes], axis=1)
    unreadable = data["error"] != ""
    for row in data[unreadable].itertuples():
        audit.add(row.source, "<ảnh>", row.rel_path, "unreadable_image", "quarantine_image", row.error)
    flag(data, unreadable, "quarantine:unreadable_image")
    removed = [data[data["exclude_reason"] != ""]]
    keep = data[data["exclude_reason"] == ""].copy()

    # ------------------------------------------------------------------ 3. Trùng điểm ảnh nhưng KHÁC nhãn -> cách ly
    conflict = keep.groupby("pixel_hash")["label"].transform("nunique") > 1
    conflicting = keep[conflict].sort_values(["pixel_hash", "source", "rel_path"])
    for row in conflicting.itertuples():
        audit.add(row.source, "<pixel>", row.rel_path, "same_pixels_different_labels", "quarantine_image",
                  f"pixel_hash={row.pixel_hash[:12]} nhãn={row.label}")
    keep.loc[conflict, "exclude_reason"] = "quarantine:same_pixels_different_labels"
    removed.append(keep[conflict])
    keep = keep[~conflict].reset_index(drop=True)
    if keep.empty:
        raise ValueError("Không còn ảnh hợp lệ sau khi audit.")

    # ------------------------------------------------------------------ 4. Trùng chính xác (cùng nhãn): đánh dấu bản gốc
    keep = mark_exact_duplicates(keep)
    exact = keep[keep["dup_count"] > 1]
    exact_report = (
        exact.groupby("pixel_hash")
        .agg(copies=("rel_path", "size"), label=("label", "first"), sources=("source", _join_unique),
             patients=("patient_id", _join_unique), paths=("rel_path", " | ".join))
        .reset_index()
    )
    exact_report["cross_patient"] = exact_report["patients"].str.contains(";")
    exact_report["cross_source"] = exact_report["sources"].str.contains(";")

    # ------------------------------------------------------------------ 5. Gần trùng (pHash) + gom nhóm chống leakage
    pairs = near_duplicate_pairs(keep["phash"].tolist(), phash_radius, keep["patient_key"].tolist())
    near_rows = []
    for a, b, distance in pairs:
        ra, rb = keep.iloc[a], keep.iloc[b]
        if ra["pixel_hash"] == rb["pixel_hash"]:
            continue
        near_rows.append({
            "path_a": ra["rel_path"], "path_b": rb["rel_path"], "distance": distance,
            "source_a": ra["source"], "source_b": rb["source"], "label_a": ra["label"], "label_b": rb["label"],
            "patient_a": ra["patient_id"], "patient_b": rb["patient_id"],
            "cross_patient": ra["patient_key"] != rb["patient_key"] or ra["patient_key"] == "",
            "label_differs": ra["label"] != rb["label"],
        })
    near_report = pd.DataFrame(near_rows, columns=[
        "path_a", "path_b", "distance", "source_a", "source_b", "label_a", "label_b",
        "patient_a", "patient_b", "cross_patient", "label_differs",
    ])
    # Cùng cảnh quay nhưng khác mã video/bệnh nhân (dữ liệu chia sẵn vẫn có thể lọt) -> gộp chung 1 nhóm
    scene_threshold = task.get("scene_link_threshold")
    scene_report = pd.DataFrame(columns=["unit_a", "unit_b", "matches", "example_a", "example_b"])
    if scene_threshold:
        # Quét MỌI frame (kể cả frame bị bớt) của các video còn giữ: frame trùng có thể không nằm trong 20 frame được chọn.
        kept_units = set(keep.loc[keep["patient_key"] != "", "patient_key"])
        scan = data[data["patient_key"].isin(kept_units)
                    & data["exclude_reason"].isin(["", "excluded:frame_subsample"])]
        links = scene_links(scan["image_path"].tolist(), scan["patient_key"].tolist(), float(scene_threshold),
                            int(task.get("scene_link_min_matches", 2)), workers)
        first = {key: int(idx[0]) for key, idx in keep.groupby("patient_key").indices.items() if key}
        pairs = list(pairs) + [(first[l["unit_a"]], first[l["unit_b"]], -1) for l in links]
        scene_report = pd.DataFrame([{"unit_a": l["unit_a"], "unit_b": l["unit_b"], "matches": l["matches"],
                                      "example_a": scan.iloc[l["index_a"]]["rel_path"],
                                      "example_b": scan.iloc[l["index_b"]]["rel_path"]} for l in links],
                                    columns=scene_report.columns)
        for l in links:
            audit.add("*", "<cảnh quay>", f"{l['unit_a']} ~ {l['unit_b']}", "same_scene_different_video_id",
                      "group_together", f"{l['matches']} cặp frame gần trùng")
    keep["group_id"] = build_groups(keep, pairs)
    # Nhóm vừa có ảnh của tập test cố định vừa có ảnh train -> cách ly phần train (không để lọt sang test)
    is_test = keep["fixed_split"] == "test"
    if is_test.any():
        test_groups = set(keep.loc[is_test, "group_id"])
        leak = keep["group_id"].isin(test_groups) & ~is_test
        if leak.any():
            for row in keep[leak].itertuples():
                audit.add(row.source, "<nhóm>", row.rel_path, "duplicate_of_fixed_test_image", "quarantine_image")
            keep.loc[leak, "exclude_reason"] = "quarantine:duplicate_of_fixed_test_image"
            removed.append(keep[leak])
            keep = keep[~leak].reset_index(drop=True)
    keep["unit_id"] = np.where(keep["patient_key"] != "", keep["patient_key"], "img::" + keep["pixel_hash"])
    group_sizes = keep[keep["canonical"]].groupby("group_id").size()
    largest_share = float(group_sizes.max() / group_sizes.sum())

    # ------------------------------------------------------------------ 6. Chia tập theo nhóm, phân tầng theo lớp
    classes = sorted(keep["label"].unique())
    if len(classes) < 2:
        raise ValueError(f"Chỉ có {len(classes)} lớp sau khi lọc — không thể huấn luyện phân loại.")
    units = keep[keep["canonical"]].groupby(["group_id", "unit_id", "label"]).size().rename("n_images").reset_index()
    fixed_test = keep["fixed_split"] == "test"
    if fixed_test.any():
        # Tập test cố định (vd. BAGLS-VF_test): phần còn lại chia train/val theo tỷ lệ train:val yêu cầu.
        val_share = ratios[1] / (ratios[0] + ratios[1])
        pool_units = units[~units["group_id"].isin(set(keep.loc[fixed_test, "group_id"]))]
        mapping, split_info = grouped_stratified_split(pool_units, (1 - val_share, val_share), seed, names=("train", "val"))
        split_info["fixed_test_images"] = int(fixed_test.sum())
        keep["split"] = np.where(fixed_test, "test", keep["group_id"].map(mapping))
        naive = np.array(["test"] * len(keep), dtype=object)
        pool = ~fixed_test.to_numpy()
        naive[pool] = naive_split(keep.loc[pool, "label"], (1 - val_share, val_share * 0.5, val_share * 0.5), seed)
        naive[pool & (naive == "test")] = "val"  # tập test luôn là test cố định
        keep["split_naive"] = naive
    else:
        if ratios[2] <= 0:
            raise ValueError("Task không có tập test cố định -> tỷ lệ test phải > 0 (vd. --split 70 15 15).")
        mapping, split_info = grouped_stratified_split(units, ratios, seed)
        keep["split"] = keep["group_id"].map(mapping)
        keep["split_naive"] = naive_split(keep["label"], ratios, seed)

    proper = leakage_report(keep, "split")
    crossing = {k: v for k, v in proper.items() if k.endswith("_crossing_splits") and v}
    if crossing:
        raise RuntimeError(f"Lỗi nội bộ: vẫn có leakage sau khi chia {crossing}")
    naive = leakage_report(keep, "split_naive")

    # ------------------------------------------------------------------ 7. Thống kê + cảnh báo
    distribution = (
        keep.groupby(["split", "label"])
        .agg(images=("canonical", "sum"), images_incl_duplicates=("rel_path", "size"),
             patients=("patient_key", lambda s: int(s[s != ""].nunique())), groups=("group_id", "nunique"))
        .reset_index()
    )
    warnings = []
    canonical = keep[keep["canonical"]]
    for split in ("train", "val", "test"):
        missing = sorted(set(classes) - set(canonical.loc[canonical["split"] == split, "label"]))
        if missing:
            warnings.append(f"Tập {split} thiếu lớp {missing} (lớp quá hiếm) — metric của lớp này trên {split} không đo được.")
    if split_info["classes_with_fewer_than_3_groups"]:
        warnings.append(f"Lớp có < 3 bệnh nhân/nhóm độc lập: {split_info['classes_with_fewer_than_3_groups']} — nên gộp lớp hoặc thêm dữ liệu.")
    if largest_share > 0.15:
        warnings.append(f"Nhóm lớn nhất chiếm {largest_share:.0%} ảnh (pHash gom bắc cầu) — cân nhắc giảm --phash-radius.")
    crosstab = pd.crosstab(keep["label"], keep["source"])
    if crosstab.shape[1] > 1:
        single = [label for label, row in crosstab.iterrows() if (row > 0).sum() == 1]
        if single:
            warnings.append(f"Lớp chỉ đến từ 1 nguồn: {single} — model có thể học 'nguồn/thiết bị' thay vì bệnh (shortcut).")
    for message in warnings:
        warn(message)

    # ------------------------------------------------------------------ 8. Ghi kết quả
    manifest = keep[MANIFEST_COLUMNS].sort_values(["source", "rel_path"]).reset_index(drop=True)
    write_csv(manifest, manifest_path)
    removed_frame = pd.concat(removed, ignore_index=True)
    removed_frame = removed_frame.reindex(columns=REMOVED_COLUMNS).sort_values(["source", "rel_path"])
    write_csv(removed_frame, out_dir / "removed_images.csv")
    write_csv(audit.issues_frame(), out_dir / "label_issues.csv")
    write_csv(audit.files_frame(), out_dir / "label_file_audit.csv")
    write_csv(exact_report, out_dir / "duplicates_exact.csv")
    write_csv(conflicting[["pixel_hash", "source", "rel_path", "label", "patient_id"]], out_dir / "duplicates_conflicting_labels.csv")
    write_csv(near_report, out_dir / "duplicates_near.csv")
    write_csv(scene_report, out_dir / "duplicates_same_scene_videos.csv")
    write_csv(distribution, out_dir / "split_distribution.csv")
    write_csv(crosstab.reset_index(), out_dir / "source_label_crosstab.csv")

    manifest_sha = sha256_file(manifest_path)
    task_info = {
        "task": task_name,
        "description": task.get("description", ""),
        "anatomy": task.get("anatomy", ""),
        "modality": task.get("modality", ""),
        "classes": classes,
        "display_names": {c: task.get("display_names", {}).get(c, c) for c in classes},
        "augment": task.get("augment", {}),
        "sources": sources_used,
        "ratios": list(ratios),
        "seed": seed,
        "phash_radius": phash_radius,
        "preprocess": task.get("preprocess", {}),
        "fixed_test": bool((keep["fixed_split"] == "test").any()),
        "max_images_per_unit": max_images_per_unit,
        "manifest_sha256": manifest_sha,
        "created_at": iso_now(),
    }
    write_json(out_dir / "task.json", task_info)

    reasons = removed_frame["exclude_reason"].value_counts().to_dict()
    issues = audit.issues_frame()
    summary = {
        "task": task_name,
        "images_found": n_found,
        "images_kept": int(len(keep)),
        "images_removed": int(len(removed_frame)),
        "removed_by_reason": reasons,
        "label_file_issue_counts": issues["issue"].value_counts().to_dict() if len(issues) else {},
        "exact_duplicate_groups": int(len(exact_report)),
        "exact_duplicate_extra_copies": int((~keep["canonical"]).sum()),
        "conflicting_duplicate_images_quarantined": int(len(conflicting)),
        "near_duplicate_pairs": int(len(near_report)),
        "same_scene_video_pairs": int(len(scene_report)),
        "near_duplicate_pairs_cross_patient": int(near_report["cross_patient"].sum()) if len(near_report) else 0,
        "near_duplicate_pairs_label_differs": int(near_report["label_differs"].sum()) if len(near_report) else 0,
        "phash_radius": phash_radius,
        "groups": int(keep["group_id"].nunique()),
        "largest_group_share": round(largest_share, 4),
        "patients": int(keep.loc[keep["patient_key"] != "", "patient_key"].nunique()),
        "classes": classes,
        "ratios_requested": list(ratios),
        "split": split_info,
        "split_images_canonical": canonical["split"].value_counts().to_dict(),
        "split_patients": keep[keep["patient_key"] != ""].drop_duplicates("patient_key")["split"].value_counts().to_dict(),
        "leakage_check_proper_split": proper,
        "leakage_demo_naive_split": naive,
        "warnings": warnings,
        "manifest_sha256": manifest_sha,
        "outputs": str(out_dir),
    }
    write_json(out_dir / "prepare_summary.json", summary)
    _print_summary(summary)
    return summary


def _print_summary(s: dict) -> None:
    log(f"===== TÓM TẮT PREPARE: {s['task']} =====")
    log(f"Ảnh tìm thấy {s['images_found']} | dùng được {s['images_kept']} | bị loại {s['images_removed']}")
    for reason, count in s["removed_by_reason"].items():
        log(f"   - {reason}: {count}")
    log(f"Duplicate chính xác cùng nhãn: {s['exact_duplicate_groups']} nhóm, {s['exact_duplicate_extra_copies']} bản thừa "
        f"| trùng pixel khác nhãn (đã cách ly): {s['conflicting_duplicate_images_quarantined']} ảnh")
    log(f"Gần trùng pHash<= {s['phash_radius']}: {s['near_duplicate_pairs']} cặp "
        f"({s['near_duplicate_pairs_cross_patient']} khác bệnh nhân, {s['near_duplicate_pairs_label_differs']} khác nhãn)")
    log(f"Nhóm độc lập: {s['groups']} | bệnh nhân: {s['patients']} | lớp: {len(s['classes'])}")
    log(f"Ảnh mỗi tập (không tính bản trùng): {s['split_images_canonical']} | bệnh nhân: {s['split_patients']}")
    log(f"Tỷ lệ đạt được theo ảnh: {s['split']['achieved_ratio_images']}")
    proper = s["leakage_check_proper_split"]
    log(f"Kiểm tra leakage (split chính): bệnh nhân/nhóm/pixel/file vắt qua 2 tập = "
        f"{proper['patient_key_crossing_splits']}/{proper['group_id_crossing_splits']}/"
        f"{proper['pixel_hash_crossing_splits']}/{proper['sha256_crossing_splits']} -> OK")
    naive = s["leakage_demo_naive_split"]
    if naive["test_images"]:
        log(f"(Minh họa) chia ngẫu nhiên theo ảnh: {naive['test_images_same_patient_in_train']}/{naive['test_images']} "
            f"ảnh test có cùng bệnh nhân trong train, {naive['test_images_exact_duplicate_in_train']} ảnh test trùng y hệt ảnh train.")
    log(f"Kết quả: {s['outputs']}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--task", default="bagls_3class", help="tên task trong configs/tasks.json")
    parser.add_argument("--tasks-file", default=str(DEFAULT_TASKS_FILE))
    parser.add_argument("--split", nargs=3, default=["70", "15", "15"], metavar=("TRAIN", "VAL", "TEST"),
                        help="tỷ lệ train val test, vd 70 15 15 hoặc 80 10 10")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--phash-radius", type=int, default=None,
                        help="ngưỡng Hamming pHash để coi là gần trùng (gom cùng tập); -1 = tắt; mặc định theo task (4)")
    parser.add_argument("--max-images-per-unit", type=int, default=None,
                        help="số frame tối đa mỗi video/bệnh nhân cho train/val (0 = tất cả); mặc định theo task")
    parser.add_argument("--source-root", action="append", default=[], metavar="TÊN=ĐƯỜNG_DẪN",
                        help="ghi đè thư mục dữ liệu của 1 nguồn (lặp lại được)")
    parser.add_argument("--prepared-dir", default=str(DEFAULT_PREPARED_DIR))
    parser.add_argument("--workers", type=int, default=8, help="số luồng đọc/băm ảnh")
    parser.add_argument("--overwrite", action="store_true", help="cho phép chia lại khi đã có manifest")
    return parser


def main(argv=None) -> None:
    setup_console()
    args = build_parser().parse_args(argv)
    ratios = parse_ratios(args.split)
    task = get_task(args.task, args.tasks_file)
    prepare_task(
        args.task, task, resolve_cli_path(args.prepared_dir), ratios=ratios, seed=args.seed,
        phash_radius=args.phash_radius, source_roots=parse_key_values(args.source_root),
        max_images_per_unit=args.max_images_per_unit,
        workers=args.workers, overwrite=args.overwrite,
    )


if __name__ == "__main__":
    main()
