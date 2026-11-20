"""Telemetry 1 Hz cho TATO: GPU NVIDIA qua NVML (công suất, nhiệt, xung, bitmask throttling), CPU/RAM qua psutil.

Quy tắc (TATO v2, mục 9): ghi timestamp thật; ghi nhiệt độ TỨC THỜI (không thay bằng trung bình);
phân biệt throttling do nhiệt (bit 0x20/0x40) với chạm trần công suất (0x04); cảm biến lỗi/không có => để
trống (None), KHÔNG ghi 0. Năng lượng ở đây là năng lượng GPU (NVML), không phải năng lượng toàn hệ thống.
"""
from __future__ import annotations

import os
import platform
import subprocess
import threading
import time
from pathlib import Path

import numpy as np
import pandas as pd

from .utils import is_finite

try:
    import psutil
except ImportError:  # psutil là dependency bắt buộc trong pyproject, nhưng vẫn chạy được nếu thiếu
    psutil = None

THERMAL_THROTTLE_BITS = 0x20 | 0x40  # SW_THERMAL_SLOWDOWN | HW_THERMAL_SLOWDOWN
POWER_CAP_BIT = 0x04  # SW_POWER_CAP (không phải quá nhiệt)

FIELDS = [
    "timestamp", "elapsed_s", "phase", "epoch", "train_images",
    "gpu_power_w", "gpu_temp_c", "gpu_util", "gpu_mem_util", "vram_used_gb", "gpu_sm_clock_mhz",
    "gpu_throttle_bitmask", "gpu_thermal_throttled", "gpu_power_capped",
    "cpu_util", "cpu_temp_c", "ram_used_gb", "sensor_error",
]


def _cpu_temperature_reader():
    """Linux: psutil.sensors_temperatures. macOS/Windows: psutil không đọc được -> None."""
    if psutil is None or not hasattr(psutil, "sensors_temperatures"):
        return None
    try:
        if not psutil.sensors_temperatures():
            return None
    except Exception:
        return None

    def read() -> float | None:
        values = []
        for name, entries in (psutil.sensors_temperatures() or {}).items():
            for entry in entries:
                label = (entry.label or "").lower()
                if name in {"coretemp", "k10temp", "zenpower", "cpu_thermal", "cpu-thermal"} or "package" in label \
                        or label in {"tctl", "tdie"}:
                    if entry.current is not None:
                        values.append(float(entry.current))
        return max(values) if values else None

    return read


class TelemetryLogger:
    def __init__(self, device, interval: float = 1.0, nvml_index: int | None = None, enabled: bool = True) -> None:
        """enabled=False: chỉ ghi đồng hồ/tiến độ (không đọc cảm biến nhiệt/công suất, không chạy luồng nền)."""
        self.enabled = bool(enabled)
        self.device = device
        self.interval = max(float(interval), 0.1)
        self._origin = time.monotonic()
        self._rows: list[dict] = []
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._progress = {"phase": "idle", "epoch": 0, "train_images": 0}
        self._nvml = None
        self._handle = None
        self._reasons = None
        self.gpu_source: str | None = None
        self.init_error: str | None = None
        self._cpu_temp = _cpu_temperature_reader() if enabled else None
        if psutil is not None and enabled:
            psutil.cpu_percent(interval=None)  # lần gọi đầu luôn trả 0 -> "mồi" trước
        if enabled and getattr(device, "type", "") == "cuda":
            self._init_nvml(nvml_index if nvml_index is not None else (device.index or 0))

    def _init_nvml(self, index: int) -> None:
        try:
            import pynvml  # gói nvidia-ml-py

            pynvml.nvmlInit()
            self._handle = pynvml.nvmlDeviceGetHandleByIndex(int(index))
            self._nvml = pynvml
            self._reasons = (getattr(pynvml, "nvmlDeviceGetCurrentClocksEventReasons", None)
                             or getattr(pynvml, "nvmlDeviceGetCurrentClocksThrottleReasons", None))
            self.gpu_source = "NVML"
        except Exception as exc:
            self.init_error = f"NVML không dùng được ({type(exc).__name__}: {exc}). Cài: pip install nvidia-ml-py"
            self._nvml = self._handle = None

    @property
    def has_gpu_sensors(self) -> bool:
        return self._handle is not None

    def elapsed(self) -> float:
        return time.monotonic() - self._origin

    def set_progress(self, **values) -> None:
        with self._lock:
            self._progress.update(values)

    def add_train_images(self, count: int) -> None:
        with self._lock:
            self._progress["train_images"] += int(count)

    def sample(self) -> dict:
        row = dict.fromkeys(FIELDS)
        row["timestamp"] = time.time()
        row["elapsed_s"] = self.elapsed()
        with self._lock:
            row.update(self._progress)
        errors = []
        if psutil is not None and self.enabled:
            try:
                row["cpu_util"] = psutil.cpu_percent(interval=None) / 100.0
                row["ram_used_gb"] = psutil.virtual_memory().used / 2**30
            except Exception as exc:
                errors.append(f"psutil: {exc}")
        if self._cpu_temp is not None:
            try:
                row["cpu_temp_c"] = self._cpu_temp()
            except Exception as exc:
                errors.append(f"cpu_temp: {exc}")
        if self._handle is not None:
            nv, handle = self._nvml, self._handle

            def read(key, function):
                try:
                    row[key] = function()
                except Exception as exc:
                    errors.append(f"{key}: {exc}")

            def utilization():
                rates = nv.nvmlDeviceGetUtilizationRates(handle)
                row["gpu_mem_util"] = rates.memory / 100.0
                return rates.gpu / 100.0

            read("gpu_power_w", lambda: nv.nvmlDeviceGetPowerUsage(handle) / 1000.0)
            read("gpu_temp_c", lambda: float(nv.nvmlDeviceGetTemperature(handle, nv.NVML_TEMPERATURE_GPU)))
            read("gpu_util", utilization)
            read("vram_used_gb", lambda: nv.nvmlDeviceGetMemoryInfo(handle).used / 2**30)
            read("gpu_sm_clock_mhz", lambda: float(nv.nvmlDeviceGetClockInfo(handle, nv.NVML_CLOCK_SM)))
            if self._reasons is not None:
                read("gpu_throttle_bitmask", lambda: int(self._reasons(handle)))
            bits = row["gpu_throttle_bitmask"]
            if bits is not None:
                row["gpu_thermal_throttled"] = int(bool(bits & THERMAL_THROTTLE_BITS))
                row["gpu_power_capped"] = int(bool(bits & POWER_CAP_BIT))
        if errors:
            row["sensor_error"] = "; ".join(errors)
        with self._lock:
            self._rows.append(row)
        return row

    def start(self) -> None:
        self.sample()
        if not self.enabled:
            return

        def loop() -> None:
            while not self._stop.wait(self.interval):
                try:
                    self.sample()
                except Exception as exc:
                    with self._lock:
                        self._rows.append({"timestamp": time.time(), "elapsed_s": self.elapsed(),
                                           "sensor_error": f"{type(exc).__name__}: {exc}"})

        self._thread = threading.Thread(target=loop, name="telemetry", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=5)
        try:
            self.sample()
        except Exception:
            pass
        if self._nvml is not None:
            try:
                self._nvml.nvmlShutdown()
            except Exception:
                pass
        self._nvml = self._handle = None

    def rows(self) -> list[dict]:
        with self._lock:
            return [dict(r) for r in self._rows]

    def latest(self) -> dict:
        with self._lock:
            return dict(self._rows[-1]) if self._rows else {}

    def save(self, path) -> None:
        pd.DataFrame(self.rows(), columns=FIELDS).to_csv(path, index=False)


# ------------------------------------------------------------------------------- phân tích chuỗi
def ols_slope(times, values) -> float | None:
    """Độ dốc (đơn vị/giây) bằng hồi quy tuyến tính trên timestamp thật — không giả định bước thời gian đều."""
    t, y = np.asarray(times, dtype=float), np.asarray(values, dtype=float)
    ok = np.isfinite(t) & np.isfinite(y)
    t, y = t[ok], y[ok]
    if len(t) < 2 or np.ptp(t) == 0:
        return None
    return float(np.polyfit(t - t[0], y, 1)[0])


def _series(rows, key):
    pairs = [(float(r["elapsed_s"]), float(r[key])) for r in rows if is_finite(r.get(key)) and is_finite(r.get("elapsed_s"))]
    return [p[0] for p in pairs], [p[1] for p in pairs]


def _mean(values):
    return float(np.mean(values)) if values else None


def _max(values):
    return float(np.max(values)) if values else None


def _window(rows, start, end):
    return [r for r in rows if is_finite(r.get("elapsed_s")) and start <= r["elapsed_s"] <= end]


def pre_run_features(rows: list[dict], window_s: float = 60.0) -> dict:
    """Trạng thái máy tại t=0 (trước khi train) — đặc trưng 'pre_run_*' duy nhất được phép cho Static Predictor."""
    rows = [r for r in rows if is_finite(r.get("elapsed_s"))]
    if not rows:
        return {}
    latest = rows[-1]
    window = _window(rows, latest["elapsed_s"] - window_s, latest["elapsed_s"])
    t_gpu, gpu_temp = _series(window, "gpu_temp_c")
    _, cpu_temp = _series(window, "cpu_temp_c")
    _, gpu_power = _series(window, "gpu_power_w")
    return {
        "pre_run_gpu_temp_current_c": latest.get("gpu_temp_c"),
        "pre_run_gpu_temp_mean_60s_c": _mean(gpu_temp),
        "pre_run_gpu_temp_max_60s_c": _max(gpu_temp),
        "pre_run_gpu_temp_slope_60s_c_per_s": ols_slope(t_gpu, gpu_temp),
        "pre_run_cpu_temp_current_c": latest.get("cpu_temp_c"),
        "pre_run_cpu_temp_mean_60s_c": _mean(cpu_temp),
        "pre_run_gpu_power_current_w": latest.get("gpu_power_w"),
        "pre_run_gpu_power_mean_60s_w": _mean(gpu_power),
        "pre_run_gpu_util_current": latest.get("gpu_util"),
        "pre_run_cpu_util_current": latest.get("cpu_util"),
        "pre_run_ram_used_gb": latest.get("ram_used_gb"),
        "pre_run_vram_used_gb": latest.get("vram_used_gb"),
        "pre_run_window_s": float(latest["elapsed_s"] - window[0]["elapsed_s"]),
        "pre_run_samples": len(window),
    }


def integrate_energy_wh(rows: list[dict], start: float, end: float, key: str = "gpu_power_w",
                        max_gap_s: float = 5.0) -> float | None:
    """Năng lượng (Wh) = tích phân hình thang công suất trong [start, end]. Thiếu dữ liệu/khoảng trống lớn -> None."""
    if end <= start:
        return 0.0
    times, power = _series(rows, key)
    if not times or min(power) < 0:
        return None
    order = np.argsort(times)
    t, p = np.asarray(times)[order], np.asarray(power)[order]
    if t[0] > start + max_gap_s or t[-1] < end - max_gap_s:
        return None
    grid = np.concatenate([[start], t[(t > start) & (t < end)], [end]])
    if np.diff(grid).max() > max_gap_s:
        return None
    values = np.interp(grid, t, p)
    integrate = getattr(np, "trapezoid", None) or np.trapz
    return float(integrate(values, grid) / 3600.0)


def window_stats(rows: list[dict], end: float, window_s: float = 60.0) -> dict:
    """Thống kê cửa sổ 60 s gần nhất (đặc trưng 'observed_*' cho Online Predictor của TATO)."""
    window = _window(rows, end - window_s, end)
    t_gpu, gpu_temp = _series(window, "gpu_temp_c")
    _, power = _series(window, "gpu_power_w")
    _, gpu_util = _series(window, "gpu_util")
    _, cpu_util = _series(window, "cpu_util")
    _, cpu_temp = _series(window, "cpu_temp_c")
    t_img, images = _series(window, "train_images")
    throughput = None
    if len(t_img) >= 2 and t_img[-1] > t_img[0]:
        throughput = (images[-1] - images[0]) / (t_img[-1] - t_img[0])
    flags = [int(r["gpu_thermal_throttled"]) for r in window if is_finite(r.get("gpu_thermal_throttled"))]
    return {
        "gpu_temp_mean_60s_c": _mean(gpu_temp),
        "gpu_temp_max_60s_c": _max(gpu_temp),
        "gpu_temp_slope_60s_c_per_s": ols_slope(t_gpu, gpu_temp),
        "gpu_power_mean_60s_w": _mean(power),
        "gpu_util_mean_60s": _mean(gpu_util),
        "cpu_util_mean_60s": _mean(cpu_util),
        "cpu_temp_max_60s_c": _max(cpu_temp),
        "images_per_s_60s": throughput,
        "throttled_any_60s": max(flags) if flags else None,
        "throttled_current": flags[-1] if flags else None,
    }


def run_statistics(rows: list[dict], start: float, end: float, max_gap_s: float = 5.0) -> dict:
    """Thống kê toàn run (full_run_*) — CHỈ dùng làm nhãn/phân tích sau run, cấm làm đặc trưng tĩnh."""
    window = _window(rows, start, end)
    _, gpu_temp = _series(window, "gpu_temp_c")
    _, power = _series(window, "gpu_power_w")
    _, gpu_util = _series(window, "gpu_util")
    _, cpu_temp = _series(window, "cpu_temp_c")
    flagged = [r for r in window if is_finite(r.get("gpu_thermal_throttled"))]
    throttle_s = capped_s = 0.0
    for current, following in zip(flagged, flagged[1:]):
        dt = following["elapsed_s"] - current["elapsed_s"]
        if dt <= max_gap_s:
            throttle_s += dt if current["gpu_thermal_throttled"] == 1 else 0.0
            capped_s += dt if current.get("gpu_power_capped") == 1 else 0.0
    throttled = None
    if flagged:
        throttled = int(any(r["gpu_thermal_throttled"] == 1 for r in flagged))
    return {
        "full_run_time_s": float(end - start),
        "full_run_gpu_energy_wh": integrate_energy_wh(rows, start, end, max_gap_s=max_gap_s),
        "full_run_gpu_power_mean_w": _mean(power),
        "full_run_gpu_temp_max_c": _max(gpu_temp),
        "full_run_gpu_temp_mean_c": _mean(gpu_temp),
        "full_run_gpu_util_mean": _mean(gpu_util),
        "full_run_cpu_temp_max_c": _max(cpu_temp),
        "throttled": throttled,
        "throttle_seconds": throttle_s if flagged else None,
        "power_capped_seconds": capped_s if flagged else None,
        "telemetry_samples": len(window),
    }


def thermal_event_in_horizon(rows: list[dict], start: float, horizon_s: float = 60.0) -> tuple[int | None, int]:
    """(có throttling do nhiệt trong (start, start+horizon] hay không, cờ bị cắt cụt do run kết thúc sớm)."""
    later = [r for r in rows if is_finite(r.get("elapsed_s")) and start < r["elapsed_s"] <= start + horizon_s]
    flags = [r["gpu_thermal_throttled"] for r in later if is_finite(r.get("gpu_thermal_throttled"))]
    last = max((r["elapsed_s"] for r in rows if is_finite(r.get("elapsed_s"))), default=start)
    censored = int(last < start + horizon_s)
    if not flags:
        return None, censored
    return int(any(f == 1 for f in flags)), censored


def _cpu_model() -> str:
    try:
        system = platform.system()
        if system == "Darwin":
            out = subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"], capture_output=True, text=True, timeout=3)
            if out.stdout.strip():
                return out.stdout.strip()
        elif system == "Linux":
            for line in Path("/proc/cpuinfo").read_text(errors="ignore").splitlines():
                if line.lower().startswith("model name"):
                    return line.split(":", 1)[1].strip()
    except Exception:
        pass
    return platform.processor() or platform.machine()


def hardware_context(device) -> dict:
    import torch

    context = {
        "hardware_os": platform.platform(),
        "hardware_python": platform.python_version(),
        "hardware_cpu_model": _cpu_model(),
        "hardware_cpu_logical_cores": os.cpu_count(),
        "hardware_ram_gb": psutil.virtual_memory().total / 2**30 if psutil is not None else None,
        "hardware_device": str(device),
        "hardware_torch_version": torch.__version__,
        "hardware_cuda_version": torch.version.cuda,
        "hardware_rocm_hip_version": getattr(torch.version, "hip", None),  # GPU AMD (ROCm): torch.cuda = HIP
        "hardware_gpu_model": None,
        "hardware_gpu_vram_gb": None,
    }
    if device.type == "cuda":
        props = torch.cuda.get_device_properties(device)
        context.update(hardware_gpu_model=props.name, hardware_gpu_vram_gb=props.total_memory / 2**30)
    elif device.type == "mps":
        context.update(hardware_gpu_model="Apple GPU (MPS, bộ nhớ dùng chung)")
    return context
