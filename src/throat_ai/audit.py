"""Kiểm tra ảnh: lỗi đọc, SHA-256 (trùng file), pixel hash (trùng nội dung), pHash (gần trùng) và gom nhóm chống leakage."""
from __future__ import annotations

import hashlib
import io
import math
import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import imagehash
import pandas as pd
from PIL import Image, ImageOps

from .utils import log, write_csv

HASH_FIELDS = ["sha256", "pixel_hash", "phash", "width", "height", "error"]


def inspect_image(path: str) -> dict:
    """Đọc 1 ảnh: băm file, băm điểm ảnh sau khi xoay EXIF + RGB, pHash 64 bit. Lỗi -> ghi vào `error`."""
    try:
        raw = Path(path).read_bytes()
        with Image.open(io.BytesIO(raw)) as image:
            image = ImageOps.exif_transpose(image).convert("RGB")
            width, height = image.size
            pixel_hash = hashlib.sha256(f"{width}x{height}:".encode() + image.tobytes()).hexdigest()
            phash = str(imagehash.phash(image))
        return {"sha256": hashlib.sha256(raw).hexdigest(), "pixel_hash": pixel_hash, "phash": phash,
                "width": width, "height": height, "error": ""}
    except Exception as exc:  # ảnh hỏng/cắt cụt/không phải ảnh
        return {"sha256": "", "pixel_hash": "", "phash": "", "width": 0, "height": 0,
                "error": f"{type(exc).__name__}: {exc}"}


def hash_images(paths: list[str], cache_file: Path | None = None, workers: int = 8) -> pd.DataFrame:
    """Băm toàn bộ ảnh (song song), có cache theo (kích thước, mtime) để chạy lại nhanh."""
    cache: dict[str, dict] = {}
    if cache_file is not None and Path(cache_file).exists():
        for row in pd.read_csv(cache_file, dtype=str, keep_default_na=False).to_dict("records"):
            cache[row["image_path"]] = row
    results: list[dict | None] = [None] * len(paths)
    todo = []
    for index, path in enumerate(paths):
        try:
            stat = os.stat(path)
        except OSError as exc:
            results[index] = {"sha256": "", "pixel_hash": "", "phash": "", "width": 0, "height": 0,
                              "error": f"FileNotFoundError: {exc}"}
            continue
        signature = f"{stat.st_size}:{stat.st_mtime_ns}"
        hit = cache.get(path)
        if hit is not None and hit.get("signature") == signature:
            results[index] = {key: hit.get(key, "") for key in HASH_FIELDS}
        else:
            todo.append((index, path, signature))
    if todo:
        log(f"Băm {len(todo)} ảnh ({len(paths) - len(todo)} ảnh lấy từ cache)...")
        with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
            for done, ((index, path, signature), result) in enumerate(
                zip(todo, pool.map(inspect_image, [item[1] for item in todo])), start=1
            ):
                results[index] = result
                cache[path] = {"image_path": path, "signature": signature, **result}
                if done % 1000 == 0 or done == len(todo):
                    log(f"  đã băm {done}/{len(todo)}")
        if cache_file is not None:
            write_csv(pd.DataFrame(list(cache.values())), cache_file)
    frame = pd.DataFrame(results, columns=HASH_FIELDS)
    for column in ("width", "height"):
        frame[column] = pd.to_numeric(frame[column], errors="coerce").fillna(0).astype(int)
    for column in ("sha256", "pixel_hash", "phash", "error"):
        frame[column] = frame[column].fillna("").astype(str)
    return frame


class UnionFind:
    def __init__(self, size: int) -> None:
        self.parent = list(range(size))

    def find(self, item: int) -> int:
        while self.parent[item] != item:
            self.parent[item] = self.parent[self.parent[item]]
            item = self.parent[item]
        return item

    def union(self, a: int, b: int) -> None:
        root_a, root_b = self.find(a), self.find(b)
        if root_a != root_b:
            self.parent[max(root_a, root_b)] = min(root_a, root_b)


def near_duplicate_pairs(phashes: list[str], radius: int, group_keys: list[str] | None = None) -> list[tuple[int, int, int]]:
    """Mọi cặp ảnh có khoảng cách Hamming pHash <= radius (radius < 0: tắt).

    Nguyên lý chuồng bồ câu: chia 64 bit thành radius+1 dải; 2 hash cách nhau <= radius bit
    thì chắc chắn trùng nhau hoàn toàn ở ít nhất 1 dải -> chỉ so sánh ảnh cùng "xô".
    group_keys: (tùy chọn) mã nhóm sẵn có (vd. bệnh nhân/video) — bỏ qua cặp cùng mã vì đã chắc chắn cùng tập.
    """
    if radius < 0:
        return []
    bands = radius + 1
    width = math.ceil(64 / bands)
    mask = (1 << width) - 1
    values = [int(h, 16) if h else None for h in phashes]
    buckets: dict[tuple[int, int], list[int]] = {}
    pairs = []
    for i, value in enumerate(values):
        if value is None:
            continue
        keys = [(band, (value >> (band * width)) & mask) for band in range(bands)]
        candidates: set[int] = set()
        for key in keys:
            candidates.update(buckets.get(key, ()))
        for j in sorted(candidates):
            if group_keys is not None and group_keys[i] and group_keys[i] == group_keys[j]:
                continue
            distance = (value ^ values[j]).bit_count()
            if distance <= radius:
                pairs.append((j, i, distance))
        for key in keys:
            buckets.setdefault(key, []).append(i)
    return pairs


def mark_exact_duplicates(frame: pd.DataFrame) -> pd.DataFrame:
    """canonical=True cho đúng 1 bản (theo thứ tự nguồn/đường dẫn) trong mỗi nhóm cùng pixel_hash."""
    frame = frame.copy()
    order = frame.sort_values(["pixel_hash", "source", "rel_path"]).index
    canonical = ~frame.loc[order].duplicated("pixel_hash", keep="first")
    frame["canonical"] = canonical.reindex(frame.index)
    frame["dup_count"] = frame.groupby("pixel_hash")["pixel_hash"].transform("size").astype(int)
    return frame


def build_groups(frame: pd.DataFrame, pairs: list[tuple[int, int, int]]) -> list[str]:
    """Gom ảnh cùng bệnh nhân + cùng pixel + gần trùng (bắc cầu) thành 1 nhóm; cả nhóm luôn vào cùng 1 tập."""
    frame = frame.reset_index(drop=True)
    union = UnionFind(len(frame))
    for column in ("patient_key", "pixel_hash"):
        for value, indices in frame.groupby(column).indices.items():
            if value == "":
                continue  # ảnh không có mã bệnh nhân KHÔNG được gộp với nhau
            first = int(indices[0])
            for other in indices[1:]:
                union.union(first, int(other))
    for a, b, _ in pairs:
        union.union(a, b)
    ids: dict[int, str] = {}
    result = []
    for index in range(len(frame)):
        root = union.find(index)
        if root not in ids:
            ids[root] = f"G{len(ids) + 1:05d}"
        result.append(ids[root])
    return result


# ------------------------------------------------------------------------------ cùng cảnh quay nhưng khác mã video
def _scene_thumbnail(path: str, size: int = 16):
    """Ảnh xám, cắt vuông giữa, thu về size x size, chuẩn hóa (trừ trung bình / chia độ lệch chuẩn)."""
    import numpy as np

    try:
        with Image.open(path) as image:
            gray = ImageOps.exif_transpose(image).convert("L")
            width, height = gray.size
            side = min(width, height)
            left, top = (width - side) // 2, (height - side) // 2
            small = gray.crop((left, top, left + side, top + side)).resize((size, size), Image.Resampling.BILINEAR)
        array = np.asarray(small, dtype=np.float32).ravel()
        return (array - array.mean()) / (array.std() + 1e-6) / math.sqrt(array.size)
    except Exception:
        return None


def scene_links(paths: list[str], units: list[str], threshold: float = 0.99, min_matches: int = 2,
                workers: int = 8) -> list[dict]:
    """Tìm các cặp VIDEO/bệnh nhân khác mã nhưng có >= min_matches cặp frame gần như giống hệt (tương quan ảnh thu nhỏ
    >= threshold) => cùng 1 người/1 lần quay bị lưu dưới 2 mã khác nhau (vd. BAGLS: video test 7 ~ video train 399).

    Trả về [{unit_a, unit_b, matches, index_a, index_b}] (index = vị trí 1 cặp frame ví dụ trong `paths`).
    """
    import numpy as np

    with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
        thumbs = list(pool.map(_scene_thumbnail, paths))
    valid = [i for i, t in enumerate(thumbs) if t is not None]
    if len(valid) < 2:
        return []
    matrix = np.stack([thumbs[i] for i in valid]).astype(np.float32)
    unit = np.array([units[i] for i in valid], dtype=object)
    codes = pd.factorize(unit)[0]
    found: dict[tuple, list] = {}
    for start in range(0, len(matrix), 2048):
        corr = matrix[start:start + 2048] @ matrix.T
        rows, cols = np.nonzero(corr >= threshold)
        for r, c in zip(rows + start, cols):
            if r < c and codes[r] != codes[c] and unit[r] and unit[c]:
                key = tuple(sorted((unit[r], unit[c])))
                entry = found.setdefault(key, [0, valid[r], valid[c]])
                entry[0] += 1
    links = [{"unit_a": a, "unit_b": b, "matches": n, "index_a": ia, "index_b": ib}
             for (a, b), (n, ia, ib) in found.items() if n >= min_matches]
    if links:
        log(f"Phát hiện {len(links)} cặp video khác mã nhưng CÙNG CẢNH QUAY (>= {min_matches} frame gần trùng): "
            + ", ".join(f"{l['unit_a']}~{l['unit_b']}" for l in links[:8]))
    return links
