"""Tiền xử lý ảnh — DÙNG CHUNG cho train, validation, test và predict để tránh lệch xử lý.

Thứ tự: đọc ảnh (xoay theo EXIF, RGB) -> đưa về hình vuông -> [CLAHE trên kênh L của LAB]
-> [augmentation, CHỈ tập train] -> tensor -> chuẩn hóa ImageNet.
Đưa về hình vuông theo `geometry`:
    letterbox     : thu nhỏ giữ tỉ lệ + đệm viền đen (mặc định)
    center_square : cắt hình vuông chính giữa (cạnh = cạnh ngắn) rồi resize — ĐÚNG cách BAGLS-VF tạo ảnh train
                    256x256 từ ảnh gốc 256x512/256x320..., nên áp dụng y hệt cho ảnh test để không lệch hình học.

CLAHE chỉ tác động độ sáng (kênh L), giữ nguyên thông tin màu (quan trọng với ảnh NBI/mạch máu).
Không có bước nào "học" thống kê từ dữ liệu (mean/std cố định theo ImageNet) => không rò rỉ từ val/test.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass

import cv2
import numpy as np
import torch
from PIL import Image, ImageOps
from torchvision.transforms import v2

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
PREPROCESS_VERSION = "rgb-letterbox-labclahe-v2"


@dataclass(frozen=True)
class PreprocessConfig:
    resolution: int = 224
    clahe: bool = False
    clahe_clip: float = 2.0
    clahe_grid: int = 8
    geometry: str = "letterbox"

    def to_dict(self) -> dict:
        return {**asdict(self), "version": PREPROCESS_VERSION}

    @classmethod
    def from_dict(cls, data: dict) -> "PreprocessConfig":
        return cls(
            resolution=int(data["resolution"]),
            clahe=bool(data["clahe"]),
            clahe_clip=float(data.get("clahe_clip", 2.0)),
            clahe_grid=int(data.get("clahe_grid", 8)),
            geometry=str(data.get("geometry", "letterbox")),
        )


def load_rgb(path) -> Image.Image:
    """Đọc ảnh bằng PIL (an toàn với đường dẫn tiếng Việt/Unicode trên Windows)."""
    with Image.open(path) as image:
        image = ImageOps.exif_transpose(image)
        return image.convert("RGB")


def apply_clahe(rgb: np.ndarray, clip: float = 2.0, grid: int = 8) -> np.ndarray:
    lab = cv2.cvtColor(np.ascontiguousarray(rgb), cv2.COLOR_RGB2LAB)
    clahe = cv2.createCLAHE(clipLimit=float(clip), tileGridSize=(int(grid), int(grid)))
    lab[:, :, 0] = clahe.apply(lab[:, :, 0])
    return cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)


def preprocess_image(image: Image.Image, cfg: PreprocessConfig) -> np.ndarray:
    """PIL RGB -> mảng uint8 (resolution, resolution, 3) đã letterbox (+ CLAHE nếu bật)."""
    size = int(cfg.resolution)
    if cfg.geometry == "center_square":
        width, height = image.size
        side = min(width, height)
        left, top = (width - side) // 2, (height - side) // 2
        square = image.crop((left, top, left + side, top + side)).resize((size, size), Image.Resampling.BILINEAR)
        array = np.asarray(square, dtype=np.uint8)
        return apply_clahe(array, cfg.clahe_clip, cfg.clahe_grid) if cfg.clahe else np.ascontiguousarray(array)
    if cfg.geometry != "letterbox":
        raise ValueError(f"geometry không hợp lệ: {cfg.geometry}")
    fitted = ImageOps.contain(image, (size, size), Image.Resampling.BILINEAR)
    array = np.asarray(fitted, dtype=np.uint8)
    if cfg.clahe:  # CLAHE trước khi đệm viền để vùng đen đệm không làm lệch histogram
        array = apply_clahe(array, cfg.clahe_clip, cfg.clahe_grid)
    canvas = np.zeros((size, size, 3), dtype=np.uint8)
    height, width = array.shape[:2]
    top, left = (size - height) // 2, (size - width) // 2
    canvas[top:top + height, left:left + width] = array
    return canvas


def build_transform(train: bool, augment: dict | None = None) -> v2.Compose:
    """Augmentation nhẹ cho train (lật/xoay/co giãn/độ sáng-tương phản, KHÔNG đổi màu sắc); val/test/predict không augment."""
    ops: list = [v2.ToImage()]
    if train and augment:
        if augment.get("hflip", True):
            ops.append(v2.RandomHorizontalFlip())
        if augment.get("vflip", False):
            ops.append(v2.RandomVerticalFlip())
        degrees = float(augment.get("rotate", 10))
        scale = tuple(float(x) for x in augment.get("scale", (0.9, 1.1)))
        translate = float(augment.get("translate", 0.05))
        if degrees or translate or scale != (1.0, 1.0):
            ops.append(v2.RandomAffine(
                degrees=degrees,
                translate=(translate, translate) if translate else None,
                scale=scale if scale != (1.0, 1.0) else None,
            ))
        brightness, contrast = float(augment.get("brightness", 0.1)), float(augment.get("contrast", 0.1))
        if brightness or contrast:
            ops.append(v2.ColorJitter(brightness=brightness, contrast=contrast))
    ops += [v2.ToDtype(torch.float32, scale=True), v2.Normalize(IMAGENET_MEAN, IMAGENET_STD)]
    return v2.Compose(ops)
