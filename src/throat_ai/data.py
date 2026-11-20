"""Đọc manifest đã chuẩn bị, chọn tập theo chính sách duplicate, tạo Dataset/DataLoader (an toàn trên Windows)."""
from __future__ import annotations

import os
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

from .preprocessing import PreprocessConfig, load_rgb, preprocess_image
from .split import leakage_report
from .utils import read_json, resolve_cli_path, sha256_file, warn

DUP_POLICIES = {
    "dedup": "Có xử lý duplicate: bỏ bản trùng chính xác ở mọi tập; chia theo nhóm bệnh nhân/ảnh trùng",
    "keep": "Không bỏ bản trùng trong train (val/test giống hệt 'dedup'); vẫn chia theo nhóm => không leakage",
    "naive": "Không xử lý gì: giữ bản trùng + chia ngẫu nhiên theo ảnh => CÓ LEAKAGE, chỉ để minh họa",
}
_STRING_COLUMNS = ["source", "rel_path", "image_path", "label", "label_raw", "patient_id", "patient_key", "unit_id",
                   "lesion_type", "histology", "sha256", "pixel_hash", "phash", "group_id", "split", "split_naive"]


class DataLeakageError(RuntimeError):
    pass


def load_manifest(path) -> pd.DataFrame:
    frame = pd.read_csv(path, dtype={c: str for c in _STRING_COLUMNS}, keep_default_na=False)
    frame["canonical"] = frame["canonical"].astype(str).str.lower().isin(["true", "1"])
    return frame


def load_prepared(prepared_dir) -> tuple[pd.DataFrame, dict, str]:
    """Trả về (manifest, task_info, sha256 của manifest)."""
    prepared_dir = Path(prepared_dir)
    manifest_path = prepared_dir / "manifest.csv"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Chưa có {manifest_path}. Chạy trước: python -m throat_ai.prepare --task {prepared_dir.name}")
    task = read_json(prepared_dir / "task.json")
    sha = sha256_file(manifest_path)
    if task.get("manifest_sha256") and task["manifest_sha256"] != sha:
        warn("manifest.csv đã bị sửa sau bước prepare — kết quả có thể không còn tái lập được.")
    frame = load_manifest(manifest_path)
    check_manifest(frame, task["classes"])
    return frame, task, sha


def check_manifest(frame: pd.DataFrame, classes: list[str]) -> None:
    """Chặn huấn luyện nếu split chính có leakage (bệnh nhân/nhóm/ảnh trùng vắt qua 2 tập)."""
    report = leakage_report(frame, "split")
    crossing = {k: v for k, v in report.items() if k.endswith("_crossing_splits") and v}
    if crossing:
        raise DataLeakageError(f"Manifest có leakage giữa train/val/test: {crossing}. Chạy lại prepare.")
    unknown = sorted(set(frame["label"]) - set(classes))
    if unknown:
        raise ValueError(f"Manifest có nhãn không có trong task.json: {unknown}")


def attach_paths_and_targets(frame: pd.DataFrame, classes: list[str], source_roots: dict | None = None) -> pd.DataFrame:
    """Thêm cột abs_path (có thể đổi thư mục gốc khi chạy trên máy khác) và y (chỉ số lớp)."""
    frame = frame.copy()
    source_roots = {k: resolve_cli_path(v) for k, v in (source_roots or {}).items()}
    frame["abs_path"] = [
        str(source_roots[s] / r) if s in source_roots else p
        for s, r, p in zip(frame["source"], frame["rel_path"], frame["image_path"])
    ]
    index = {c: i for i, c in enumerate(classes)}
    frame["y"] = frame["label"].map(index).astype(int)
    return frame


def verify_files(paths, show: int = 5) -> None:
    missing = [p for p in paths if not os.path.exists(p)]
    if missing:
        sample = "\n  ".join(missing[:show])
        raise FileNotFoundError(
            f"Thiếu {len(missing)} file ảnh, ví dụ:\n  {sample}\n"
            "Nếu dữ liệu nằm chỗ khác trên máy này, thêm --source-root <tên_nguồn>=<thư_mục>."
        )


def partition(frame: pd.DataFrame, split: str, policy: str) -> pd.DataFrame:
    """Lấy 1 tập theo chính sách duplicate. Val/test của 'dedup' và 'keep' GIỐNG HỆT nhau để so sánh công bằng."""
    if policy not in DUP_POLICIES:
        raise ValueError(f"dup_policy phải là một trong {list(DUP_POLICIES)}")
    if policy == "naive":
        subset = frame[frame["split_naive"] == split]
    else:
        subset = frame[frame["split"] == split]
        if policy == "dedup" or split != "train":
            subset = subset[subset["canonical"]]
    return subset.reset_index(drop=True)


class ImageDataset(Dataset):
    """Dataset chỉ giữ danh sách đường dẫn + nhãn (picklable => chạy được DataLoader nhiều worker trên Windows/macOS)."""

    def __init__(self, paths, labels, preprocess: PreprocessConfig, transform) -> None:
        self.paths = [str(p) for p in paths]
        self.labels = [int(y) for y in labels]
        self.preprocess = preprocess
        self.transform = transform

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, index: int):
        image = load_rgb(self.paths[index])
        array = preprocess_image(image, self.preprocess)
        tensor = self.transform(array).as_subclass(torch.Tensor)
        return tensor, self.labels[index]


def seed_worker(_worker_id: int) -> None:
    seed = torch.initial_seed() % 2**32
    np.random.seed(seed)
    random.seed(seed)
    try:
        import cv2

        cv2.setNumThreads(1)  # tránh mỗi worker lại mở thêm nhiều luồng OpenCV
    except Exception:
        pass


def close_loader(loader, join_timeout: float = 0.25) -> None:
    """Tắt worker của DataLoader (persistent) ngay khi xong việc.

    Trên macOS, mỗi worker đã thoát nhưng PyTorch vẫn chờ join tới 5 s/worker (sentinel bị tiến trình
    chia sẻ bộ nhớ giữ) => 20–40 s phí mỗi run. Hạ tạm timeout join khi tắt; worker vẫn thoát bình thường.
    """
    iterator = getattr(loader, "_iterator", None) if loader is not None else None
    if iterator is None or not hasattr(iterator, "_shutdown_workers"):
        return
    try:
        from torch.utils.data import _utils as data_utils

        previous = data_utils.MP_STATUS_CHECK_INTERVAL
        data_utils.MP_STATUS_CHECK_INTERVAL = join_timeout
        try:
            iterator._shutdown_workers()
        finally:
            data_utils.MP_STATUS_CHECK_INTERVAL = previous
        loader._iterator = None
    except Exception:
        pass  # API nội bộ của PyTorch thay đổi -> để PyTorch tự dọn như bình thường


def make_loader(dataset, batch_size: int, *, shuffle: bool, workers: int, seed: int, pin_memory: bool,
                drop_last: bool = False, sampler=None) -> DataLoader:
    generator = torch.Generator()
    generator.manual_seed(int(seed))
    kwargs = dict(
        batch_size=int(batch_size),
        shuffle=shuffle if sampler is None else False,
        sampler=sampler,
        num_workers=int(workers),
        pin_memory=pin_memory,
        worker_init_fn=seed_worker,
        generator=generator,
        drop_last=drop_last,
    )
    if workers > 0:
        kwargs.update(persistent_workers=True, prefetch_factor=2)
    return DataLoader(dataset, **kwargs)
