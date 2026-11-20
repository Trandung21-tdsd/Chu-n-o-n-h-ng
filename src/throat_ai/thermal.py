"""(Tùy chọn) Tạo trạng thái nhiệt khởi đầu cho profiling TATO: cold / warm / pre-stressed.

Nhãn trạng thái chỉ là metadata; đặc trưng thật sự dùng cho TATO là các giá trị 'pre_run_*' đo được
trong 60 giây ngay trước khi train. Mức nhiệt cụ thể phụ thuộc từng máy (TATO v2, mục 12.2).
"""
from __future__ import annotations

import time

import torch

from .models import empty_cache, synchronize
from .telemetry import ols_slope
from .utils import is_finite, log

STATES = ("none", "cold", "warm", "pre-stressed")


def gpu_load(device: torch.device, seconds: float, duty: float = 1.0) -> None:
    """Nhân ma trận liên tục để làm nóng thiết bị. duty = tỷ lệ thời gian chạy tải trong mỗi chu kỳ 1 giây."""
    if seconds <= 0:
        return
    size = 4096 if device.type == "cuda" else 2048 if device.type == "mps" else 1024
    a = torch.randn(size, size, device=device)
    b = torch.randn(size, size, device=device)
    end = time.monotonic() + seconds
    while time.monotonic() < end:
        busy_until = time.monotonic() + max(0.05, duty)
        while time.monotonic() < busy_until:
            _ = a @ b
            synchronize(device)
        if duty < 1.0:
            time.sleep(1.0 - duty)
    del a, b
    empty_cache(device)


def cooldown(telemetry, max_seconds: float, slope_threshold: float = 0.01, min_seconds: float = 60.0) -> float:
    """Nghỉ cho máy nguội: dừng sớm khi nhiệt GPU ổn định (|độ dốc 60 s| <= ngưỡng), tối đa max_seconds."""
    start = time.monotonic()
    while time.monotonic() - start < max_seconds:
        time.sleep(5)
        waited = time.monotonic() - start
        if waited < min_seconds or not telemetry.has_gpu_sensors:
            continue
        rows = [r for r in telemetry.rows() if is_finite(r.get("gpu_temp_c"))]
        recent = [r for r in rows if r["elapsed_s"] >= rows[-1]["elapsed_s"] - 60] if rows else []
        slope = ols_slope([r["elapsed_s"] for r in recent], [r["gpu_temp_c"] for r in recent])
        if slope is not None and abs(slope) <= slope_threshold:
            break
    return time.monotonic() - start


def precondition(state: str, device: torch.device, telemetry, seconds: float) -> dict:
    if state not in STATES:
        raise ValueError(f"precondition phải là một trong {STATES}")
    if state == "none" or seconds <= 0:
        return {"precondition": "none", "precondition_seconds_actual": 0.0}
    telemetry.set_progress(phase=f"precondition_{state}")
    log(f"  Điều hòa nhiệt '{state}' tối đa {seconds:.0f}s ...")
    start = time.monotonic()
    if state == "cold":
        cooldown(telemetry, seconds)
    elif state == "warm":
        gpu_load(device, seconds, duty=0.4)
    else:
        gpu_load(device, seconds, duty=1.0)
    return {"precondition": state, "precondition_seconds_actual": time.monotonic() - start}
