"""Dữ liệu PHÂN ĐOẠN BAGLS-VF: ảnh + mask màu (xanh lá/xanh dương = 2 dây thanh, đỏ = thanh môn).

Mask -> chỉ số lớp: 0 nền, 1 dây thanh bên trái ảnh (xanh lá), 2 dây thanh bên phải ảnh (xanh dương), 3 thanh môn (đỏ).
Ảnh và mask đi qua CÙNG phép cắt/biến đổi hình học (mask dùng nội suy nearest để không sinh nhãn lai).
Lật ngang => trái/phải đổi chỗ => ĐỔI nhãn 1 <-> 2 (nếu không, model học sai "xanh lá luôn ở bên trái").

Manifest phân đoạn (data/prepared/bagls_seg/manifest.csv): mỗi dòng 1 ảnh, cột video_id/split/label/histology.
Chia train/val theo VIDEO và dùng lại phân chia của task phân loại => không có video nào nằm ở 2 tập.
"""
from __future__ import annotations

import json
import random
import re
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset

from .bagls import classify_status
from .preprocessing import IMAGENET_MEAN, IMAGENET_STD, apply_clahe, load_rgb
from .sources import scan_images
from .utils import CONFIG_DIR, DEFAULT_PREPARED_DIR, log, read_json, resolve_config_path, sha256_file, write_csv, write_json

SEG_CONFIG = CONFIG_DIR / "segmentation.json"
CLASSES = ["background", "left_fold", "right_fold", "glottis"]
COLORS = np.array([[0, 0, 0], [0, 255, 0], [0, 0, 255], [255, 0, 0]], dtype=np.uint8)
LEFT, RIGHT, GLOTTIS = 1, 2, 3


def load_seg_config(path=SEG_CONFIG) -> dict:
    return read_json(path)


# ------------------------------------------------------------------------------ mask màu <-> chỉ số lớp
def decode_mask(rgb: np.ndarray) -> np.ndarray:
    """Mask RGB (H, W, 3) -> chỉ số lớp uint8. Kênh nào > 127 thì tính kênh đó (mask BAGLS-VF chỉ có 4 màu thuần)."""
    if rgb.ndim == 2:  # mask nhị phân kiểu BAGLS gốc (chỉ có thanh môn)
        return np.where(rgb > 127, GLOTTIS, 0).astype(np.uint8)
    r, g, b = (rgb[..., i] > 127 for i in range(3))
    out = np.zeros(rgb.shape[:2], dtype=np.uint8)
    out[g & ~r & ~b] = LEFT
    out[b & ~r & ~g] = RIGHT
    out[r & ~g & ~b] = GLOTTIS
    return out


def encode_mask(index: np.ndarray) -> np.ndarray:
    return COLORS[np.clip(index, 0, len(COLORS) - 1)]


def overlay(image: np.ndarray, index: np.ndarray, alpha: float = 0.45) -> np.ndarray:
    color = encode_mask(index).astype(np.float32)
    out = image.astype(np.float32).copy()
    fg = index > 0
    out[fg] = (1 - alpha) * out[fg] + alpha * color[fg]
    return out.clip(0, 255).astype(np.uint8)


def load_mask(path) -> np.ndarray:
    with Image.open(path) as image:
        array = np.asarray(image.convert("L") if image.mode in ("L", "1") else image.convert("RGB"))
    return decode_mask(array)


def mask_path_for(image_path, cfg: dict | None = None) -> Path | None:
    """BAGLS_image/X.png -> BAGLS_mask/mask_X.png ; test_images/X.png -> test_masks/X.png (None nếu không có)."""
    cfg = cfg or load_seg_config()
    path = Path(image_path)
    for part in ("train", "test"):
        info = cfg[part]
        if path.parent.name == info["image_dir"]:
            candidate = path.parent.parent / info["mask_dir"] / f"{info.get('mask_prefix', '')}{path.name}"
            return candidate if candidate.exists() else None
    return None


# ------------------------------------------------------------------------------ hình học dùng chung ảnh + mask
def center_square(array: np.ndarray, size: int, nearest: bool) -> np.ndarray:
    """Cắt hình vuông chính giữa rồi resize — giống PreprocessConfig(geometry='center_square')."""
    height, width = array.shape[:2]
    side = min(height, width)
    top, left = (height - side) // 2, (width - side) // 2
    crop = np.ascontiguousarray(array[top:top + side, left:left + side])
    if side == size:
        return crop
    if nearest:
        return cv2.resize(crop, (size, size), interpolation=cv2.INTER_NEAREST)
    # PIL bilinear để KHỚP 100% với preprocess_image() của model phân loại
    return np.asarray(Image.fromarray(crop).resize((size, size), Image.Resampling.BILINEAR), dtype=np.uint8)


def prepare_pair(image_path, mask_path, size: int, clahe: bool = False) -> tuple[np.ndarray, np.ndarray | None]:
    image = center_square(np.asarray(load_rgb(image_path), dtype=np.uint8), size, nearest=False)
    if clahe:
        image = apply_clahe(image)
    mask = center_square(load_mask(mask_path), size, nearest=True) if mask_path else None
    return image, mask


def augment_pair(image: np.ndarray, mask: np.ndarray, aug: dict, rng=random) -> tuple[np.ndarray, np.ndarray]:
    if aug.get("hflip", True) and rng.random() < 0.5:
        image, mask = image[:, ::-1], mask[:, ::-1].copy()
        left = mask == LEFT
        mask[mask == RIGHT] = LEFT
        mask[left] = RIGHT
    size = image.shape[0]
    degrees, translate = float(aug.get("rotate", 10)), float(aug.get("translate", 0.05))
    low, high = (float(x) for x in aug.get("scale", (0.9, 1.1)))
    matrix = cv2.getRotationMatrix2D((size / 2, size / 2), rng.uniform(-degrees, degrees), rng.uniform(low, high))
    matrix[:, 2] += [rng.uniform(-translate, translate) * size, rng.uniform(-translate, translate) * size]
    image = cv2.warpAffine(np.ascontiguousarray(image), matrix, (size, size), flags=cv2.INTER_LINEAR,
                           borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    mask = cv2.warpAffine(np.ascontiguousarray(mask), matrix, (size, size), flags=cv2.INTER_NEAREST,
                          borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    brightness, contrast = float(aug.get("brightness", 0.15)), float(aug.get("contrast", 0.15))
    if brightness or contrast:
        b = 1 + rng.uniform(-brightness, brightness)
        c = 1 + rng.uniform(-contrast, contrast)
        mean = image.mean()
        image = np.clip((image.astype(np.float32) * b - mean) * c + mean, 0, 255).astype(np.uint8)
    return image, mask


_MEAN = np.array(IMAGENET_MEAN, dtype=np.float32)[:, None, None]
_STD = np.array(IMAGENET_STD, dtype=np.float32)[:, None, None]


def to_tensor(image: np.ndarray) -> torch.Tensor:
    array = np.ascontiguousarray(image.transpose(2, 0, 1), dtype=np.float32) / 255.0
    return torch.from_numpy((array - _MEAN) / _STD)


class SegDataset(Dataset):
    """Chỉ giữ đường dẫn (picklable) => chạy DataLoader nhiều worker trên Windows/macOS."""

    def __init__(self, image_paths, mask_paths, size: int, clahe: bool = False, augment: dict | None = None,
                 seed: int = 0) -> None:
        self.images = [str(p) for p in image_paths]
        self.masks = [str(p) for p in mask_paths]
        self.size, self.clahe, self.augment, self.seed = int(size), bool(clahe), augment, int(seed)

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, index: int):
        image, mask = prepare_pair(self.images[index], self.masks[index], self.size, self.clahe)
        if self.augment:
            image, mask = augment_pair(image, mask, self.augment, random)  # module random: đã seed theo worker/seed run
        return to_tensor(image), torch.from_numpy(mask.astype(np.int64))


# ------------------------------------------------------------------------------ manifest phân đoạn
def _video_of_test(meta_dir: Path, stem: str) -> tuple[int | None, str]:
    path = meta_dir / f"{stem}.meta"
    if not path.exists():
        return None, ""
    meta = json.loads(path.read_text(encoding="utf-8"))
    return int(meta["Video Id"]), str(meta.get("Subject disorder status") or "")


def _scan_part(cfg: dict, part: str) -> pd.DataFrame:
    info = cfg[part]
    root = resolve_config_path(info["root"])
    image_dir, mask_dir = root / info["image_dir"], root / info["mask_dir"]
    if not image_dir.is_dir() or not mask_dir.is_dir():
        raise FileNotFoundError(f"Không thấy {image_dir} hoặc {mask_dir} (sửa configs/segmentation.json)")
    meta_root = resolve_config_path(cfg["meta_root"])
    meta_dir = meta_root / ("training/training" if part == "train" else "test/test")
    rows, missing_mask = [], 0
    for path in scan_images(image_dir):
        mask = mask_dir / f"{info.get('mask_prefix', '')}{path.name}"
        if not mask.exists():
            missing_mask += 1
            continue
        if part == "train":
            match = re.search(r"ID(\d+)_\d+_img_old_(\d+)", path.stem)
            if not match:
                continue
            video, index = int(match.group(1)), int(match.group(2))
            meta_path = meta_dir / f"{index}.meta"
            status = json.loads(meta_path.read_text(encoding="utf-8")).get("Subject disorder status", "") \
                if meta_path.exists() else ""
        else:
            index = int(path.stem) if path.stem.isdigit() else -1
            video, status = _video_of_test(meta_dir, path.stem)
            if video is None:
                continue
        group, detail = classify_status(status or "")
        rows.append({"image_path": str(path), "mask_path": str(mask), "part": part, "video_id": f"V{video}",
                     "frame_index": index, "label": group, "histology": detail if group else ""})
    if missing_mask:
        log(f"[{part}] {missing_mask} ảnh không có mask -> bỏ qua")
    return pd.DataFrame(rows)


def _subsample(frame: pd.DataFrame, max_frames: int) -> pd.DataFrame:
    if not max_frames or max_frames <= 0:
        return frame
    keep = []
    for _, group in frame.sort_values("frame_index").groupby("video_id", sort=False):
        positions = np.unique(np.linspace(0, len(group) - 1, min(max_frames, len(group))).round().astype(int))
        keep.append(group.iloc[positions])
    return pd.concat(keep) if keep else frame


def _resolve_same_scene(manifest: pd.DataFrame, cfg: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Video khác mã nhưng cùng cảnh quay: dính test -> bỏ phần train/val; dính val -> chuyển cả cụm sang val."""
    columns = ["video_a", "video_b", "matches", "action"]
    threshold = cfg.get("scene_link_threshold")
    if not threshold:
        return manifest, pd.DataFrame(columns=columns)
    from .audit import UnionFind, scene_links

    manifest = manifest.reset_index(drop=True)
    links = scene_links(manifest["image_path"].tolist(), manifest["video_id"].tolist(), float(threshold),
                        int(cfg.get("scene_link_min_matches", 2)))
    videos = list(dict.fromkeys(manifest["video_id"]))
    position = {v: i for i, v in enumerate(videos)}
    union = UnionFind(len(videos))
    for link in links:
        union.union(position[link["unit_a"]], position[link["unit_b"]])
    split_of = manifest.drop_duplicates("video_id").set_index("video_id")["split"]
    clusters: dict[int, list[str]] = {}
    for video in videos:
        clusters.setdefault(union.find(position[video]), []).append(video)
    drop, to_val, actions = set(), set(), {}
    for members in clusters.values():
        if len(members) < 2:
            continue
        splits = {split_of[v] for v in members}
        if "test" in splits:
            drop |= {v for v in members if split_of[v] != "test"}
            actions.update({v: "bỏ khỏi train/val (cùng cảnh quay với video test)" for v in members})
        elif "val" in splits:
            to_val |= {v for v in members if split_of[v] == "train"}
            actions.update({v: "chuyển sang val (cùng cảnh quay với video val)" for v in members})
    report = pd.DataFrame([{"video_a": l["unit_a"], "video_b": l["unit_b"], "matches": l["matches"],
                            "action": actions.get(l["unit_a"], "gộp chung 1 tập")} for l in links], columns=columns)
    if drop or to_val:
        log(f"Cùng cảnh quay: bỏ {len(drop)} video train/val trùng với test {sorted(drop)}; "
            f"chuyển {len(to_val)} video sang val {sorted(to_val)}")
    manifest.loc[manifest["video_id"].isin(to_val), "split"] = "val"
    return manifest[~manifest["video_id"].isin(drop)], report


def build_seg_manifest(cfg: dict | None = None, prepared_dir=DEFAULT_PREPARED_DIR, overwrite: bool = False) -> Path:
    """Tạo data/prepared/<task>/manifest.csv. Phân chia train/val lấy từ manifest của task phân loại."""
    cfg = cfg or load_seg_config()
    out_dir = Path(prepared_dir) / cfg["task"]
    manifest_path = out_dir / "manifest.csv"
    classifier_manifest = Path(prepared_dir) / cfg["classifier_task"] / "manifest.csv"
    if manifest_path.exists() and not overwrite:
        recorded = read_json(out_dir / "task.json").get("classifier_manifest_sha256") if (out_dir / "task.json").exists() else None
        if not classifier_manifest.exists() or recorded == sha256_file(classifier_manifest):
            return manifest_path
        log("Manifest phân loại đã đổi (chạy lại prepare) -> tạo lại manifest phân đoạn cho khớp phân chia mới.")
    if not classifier_manifest.exists():
        raise FileNotFoundError(
            f"Chưa có {classifier_manifest}. Chạy trước: python -m throat_ai.prepare --task {cfg['classifier_task']}\n"
            "(model phân đoạn dùng lại đúng phân chia train/val theo video của model phân loại).")
    clf = pd.read_csv(classifier_manifest, dtype=str, keep_default_na=False)
    clf_train = clf[clf["source"] == "bagls_vf_train"]
    video_split = clf_train.groupby("patient_id")["split"].agg(lambda s: s.iloc[0]).to_dict()
    if (clf_train.groupby("patient_id")["split"].nunique() > 1).any():
        raise RuntimeError("Manifest phân loại có video nằm ở 2 tập — chạy lại prepare.")

    train = _scan_part(cfg, "train")
    test = _scan_part(cfg, "test")
    train["split"] = train["video_id"].map(video_split).fillna("train")  # video không nhãn -> chỉ train
    if (train["split"] == "test").any():
        raise RuntimeError("Video của BAGLS-VF_train bị gán vào test — kiểm tra lại manifest phân loại.")
    test["split"] = "test"
    overlap = set(train["video_id"]) & set(test["video_id"])
    if overlap:
        raise RuntimeError(f"Video vừa ở train vừa ở test: {sorted(overlap)[:5]} — dữ liệu bị rò rỉ.")
    # quét cùng cảnh quay trên MỌI frame (trước khi bớt frame) rồi mới bớt
    full, scene = _resolve_same_scene(pd.concat([train, test], ignore_index=True), cfg)
    write_csv(scene, out_dir / "duplicates_same_scene_videos.csv")
    train, test = full[full["part"] == "train"], full[full["part"] == "test"]
    parts = [_subsample(train[train["split"] == "train"], int(cfg.get("max_frames_train", 20))),
             _subsample(train[train["split"] == "val"], int(cfg.get("max_frames_val", 10))), test]
    manifest = pd.concat(parts, ignore_index=True)
    manifest = manifest.sort_values(["split", "video_id", "frame_index"]).reset_index(drop=True)
    write_csv(manifest, manifest_path)
    summary = manifest.groupby("split").agg(images=("image_path", "size"), videos=("video_id", "nunique"))
    write_json(out_dir / "task.json", {
        "task": cfg["task"], "classes": CLASSES, "display_names": cfg.get("display_names", {}),
        "classifier_task": cfg["classifier_task"], "classifier_manifest_sha256": sha256_file(classifier_manifest),
        "manifest_sha256": sha256_file(manifest_path), "geometry": cfg.get("geometry", "center_square"),
        "counts": summary.reset_index().to_dict("records"),
    })
    log(f"Manifest phân đoạn -> {manifest_path}\n{summary.to_string()}")
    return manifest_path


def load_seg_manifest(prepared_dir=DEFAULT_PREPARED_DIR, cfg: dict | None = None) -> tuple[pd.DataFrame, dict, str]:
    cfg = cfg or load_seg_config()
    path = build_seg_manifest(cfg, prepared_dir)
    frame = pd.read_csv(path, dtype={"video_id": str, "label": str, "histology": str, "split": str},
                        keep_default_na=False)
    crossing = frame.groupby("video_id")["split"].nunique()
    if (crossing > 1).any():
        raise RuntimeError(f"Manifest phân đoạn có video ở 2 tập: {list(crossing[crossing > 1].index[:5])}")
    return frame, read_json(path.parent / "task.json"), sha256_file(path)
