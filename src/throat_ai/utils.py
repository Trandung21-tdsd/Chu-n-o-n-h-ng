"""Tiện ích dùng chung: ghi JSON/CSV an toàn, băm, seed, log, đường dẫn."""
from __future__ import annotations

import hashlib
import json
import math
import os
import platform
import random
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

PROJECT_ROOT = Path(os.environ.get("THROAT_AI_ROOT", Path(__file__).resolve().parents[2]))
CONFIG_DIR = PROJECT_ROOT / "configs"
DEFAULT_TASKS_FILE = CONFIG_DIR / "tasks.json"
DEFAULT_PREPARED_DIR = PROJECT_ROOT / "data" / "prepared"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "outputs"


def setup_console() -> None:
    """Tránh lỗi UnicodeEncodeError khi in tiếng Việt trên console Windows / khi chuyển hướng log ra file."""
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is not None:
            try:
                reconfigure(encoding="utf-8", errors="replace")
            except (ValueError, OSError):
                pass


def log(message: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)


def warn(message: str) -> None:
    log(f"CẢNH BÁO: {message}")


def to_jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): to_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [to_jsonable(v) for v in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return to_jsonable(value.tolist())
    if isinstance(value, np.generic):
        return to_jsonable(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return None  # NaN/inf -> null (không đo được), không phải 0
    return value


def write_json(path, value) -> None:
    """Ghi JSON nguyên tử (ghi file tạm rồi đổi tên) để không hỏng file khi bị ngắt giữa chừng."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(to_jsonable(value), ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(tmp, path)


def read_json(path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_csv(frame, path) -> Path:
    """Ghi CSV nguyên tử. Nếu file đang bị khóa (vd. đang mở bằng Excel trên Windows) thì ghi sang file khác."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    frame.to_csv(tmp, index=False)
    try:
        os.replace(tmp, path)
        return path
    except PermissionError:
        alternative = path.with_name(f"{path.stem}.{time.strftime('%Y%m%d-%H%M%S')}{path.suffix}")
        os.replace(tmp, alternative)
        warn(f"Không ghi đè được {path.name} (file đang mở?). Đã ghi sang {alternative.name}.")
        return alternative


def sha256_file(path, chunk_size: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stable_hash(value: Any, length: int = 12) -> str:
    text = json.dumps(to_jsonable(value), sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:length]


def set_seed(seed: int) -> None:
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def timestamp() -> str:
    return datetime.now().strftime("%Y%m%d-%H%M%S")


def iso_now() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def default_machine_id() -> str:
    """ID máy ẩn danh (băm hostname) — không ghi tên máy thật vào dữ liệu chia sẻ."""
    return "machine-" + hashlib.sha256(platform.node().encode("utf-8")).hexdigest()[:8]


def is_finite(value) -> bool:
    try:
        return value is not None and math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def resolve_config_path(value, base: Path = PROJECT_ROOT) -> Path:
    """Đường dẫn trong file cấu hình: tương đối thì tính từ thư mục dự án."""
    path = Path(os.path.expanduser(str(value)))
    return path if path.is_absolute() else (base / path).resolve()


def resolve_cli_path(value) -> Path:
    """Đường dẫn nhập từ dòng lệnh: tương đối thì tính từ thư mục hiện tại."""
    return Path(os.path.expanduser(str(value))).resolve()


def optional_float(text: str):
    """Kiểu cho argparse: số thực, hoặc none/off để tắt."""
    if str(text).strip().lower() in {"none", "off", "no", "false", ""}:
        return None
    return float(text)


def parse_key_values(items) -> dict[str, str]:
    """['cenbi=/data/CE-NBI', ...] -> {'cenbi': '/data/CE-NBI'}"""
    result: dict[str, str] = {}
    for item in items or []:
        if "=" not in item:
            raise ValueError(f"Cần dạng TÊN_NGUỒN=ĐƯỜNG_DẪN, nhận được: {item!r}")
        key, value = item.split("=", 1)
        result[key.strip()] = value.strip()
    return result


def load_tasks(path=DEFAULT_TASKS_FILE) -> dict[str, dict]:
    data = read_json(path)
    return {name: task for name, task in data.items() if not name.startswith("_")}


def get_task(name: str, path=DEFAULT_TASKS_FILE) -> dict:
    tasks = load_tasks(path)
    if name not in tasks:
        raise KeyError(f"Không có task {name!r} trong {path}. Các task: {', '.join(tasks)}")
    return tasks[name]
