"""
TATO Standalone Telemetry Logger for Remote / Dedicated Training Machines.
Self-contained single-file script: works on Windows and Linux (Ubuntu / Debian).
Captures 1Hz GPU power (W), core temp, SM clocks, utilization, and throttling flags.
Zero project dependencies: requires only python and nvidia-ml-py (or pynvml).
GPU AMD (vd. RX570, driver amdgpu tren Linux): doc truc tiep sysfs /sys/class/drm/card*/device (khong can NVML).
"""
import glob
import os
import re
import sys
import time
import csv
from datetime import datetime, timezone

# Check NVML availability
HAS_NVML = False
try:
    import pynvml
    pynvml.nvmlInit()
    HAS_NVML = True
except Exception:
    try:
        import nvidia_ml_py as pynvml
        pynvml.nvmlInit()
        HAS_NVML = True
    except Exception:
        pass

# Check psutil availability
try:
    import psutil
    HAS_PSUTIL = True
except ImportError:
    HAS_PSUTIL = False


# ---------------------------------------------------------------- GPU AMD (amdgpu) qua sysfs — NVML chi co tren NVIDIA
def find_amd_gpu():
    """Tim GPU AMD (vendor 0x1002) dau tien trong /sys/class/drm. Tra ve {'device', 'hwmon'} hoac None."""
    seen = set()
    for card in sorted(glob.glob("/sys/class/drm/card[0-9]*")):
        dev = os.path.realpath(os.path.join(card, "device"))
        if dev in seen or not os.path.exists(os.path.join(dev, "vendor")):
            continue
        seen.add(dev)
        try:
            with open(os.path.join(dev, "vendor")) as f:
                if f.read().strip().lower() != "0x1002":
                    continue
        except OSError:
            continue
        hwmons = sorted(glob.glob(os.path.join(dev, "hwmon", "hwmon*")))
        return {"device": dev, "hwmon": hwmons[0] if hwmons else None}
    return None


def _read_num(path, scale=1.0):
    try:
        with open(path) as f:
            return float(f.read().strip()) / scale
    except (OSError, ValueError, TypeError):
        return None


def _dpm_current_mhz(path):
    """pp_dpm_sclk / pp_dpm_mclk: dong co dau '*' la muc xung nhip hien tai, vd '6: 1244Mhz *'."""
    try:
        with open(path) as f:
            for line in f:
                if "*" in line:
                    match = re.search(r"(\d+)\s*mhz", line, re.IGNORECASE)
                    return float(match.group(1)) if match else None
    except OSError:
        pass
    return None


def get_amd_sample(gpu):
    """Mau tuc thoi tu GPU AMD. amdgpu khong bao co throttle qua sysfs -> gpu_throttled de trong (khong phai 0)."""
    dev, hw = gpu["device"], gpu["hwmon"]
    data = {"timestamp": time.time(), "time_iso": datetime.now(timezone.utc).isoformat()}
    data["gpu_temp_current_c"] = _read_num(os.path.join(hw, "temp1_input"), 1000.0) if hw else None
    power = _read_num(os.path.join(hw, "power1_average"), 1e6) if hw else None
    if power is None and hw:
        power = _read_num(os.path.join(hw, "power1_input"), 1e6)
    data["gpu_power_w"] = round(power, 2) if power is not None else None
    sclk = _read_num(os.path.join(hw, "freq1_input"), 1e6) if hw else None
    data["gpu_clock_mhz"] = sclk if sclk is not None else _dpm_current_mhz(os.path.join(dev, "pp_dpm_sclk"))
    mclk = _read_num(os.path.join(hw, "freq2_input"), 1e6) if hw else None
    data["gpu_mem_clock_mhz"] = mclk if mclk is not None else _dpm_current_mhz(os.path.join(dev, "pp_dpm_mclk"))
    busy = _read_num(os.path.join(dev, "gpu_busy_percent"))
    data["gpu_util"] = busy / 100.0 if busy is not None else None
    mem_busy = _read_num(os.path.join(dev, "mem_busy_percent"))
    data["gpu_mem_util"] = mem_busy / 100.0 if mem_busy is not None else None
    used = _read_num(os.path.join(dev, "mem_info_vram_used"), 1024 ** 3)
    total = _read_num(os.path.join(dev, "mem_info_vram_total"), 1024 ** 3)
    data["gpu_mem_used_gb"] = round(used, 3) if used is not None else None
    data["gpu_mem_total_gb"] = round(total, 3) if total is not None else None
    data["gpu_throttled"] = None
    data["gpu_throttle_bitmask"] = None
    return data


def cpu_temperature():
    """Nhiet do CPU (Linux: k10temp cho AMD, coretemp cho Intel). Khong doc duoc -> None."""
    if not HAS_PSUTIL or not hasattr(psutil, "sensors_temperatures"):
        return None
    try:
        sensors = psutil.sensors_temperatures()
    except Exception:
        return None
    for name in ("k10temp", "coretemp", "zenpower", "cpu_thermal", "acpitz"):
        values = [t.current for t in sensors.get(name, []) if t.current]
        if values:
            return float(max(values))
    return None


def get_gpu_sample(handle):
    """Samples instantaneous metrics from NVIDIA GPU."""
    if not HAS_NVML or handle is None:
        return {}

    now = time.time()
    data = {
        "timestamp": now,
        "time_iso": datetime.now(timezone.utc).isoformat()
    }

    # GPU Temp
    try:
        data["gpu_temp_current_c"] = float(pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU))
    except Exception:
        data["gpu_temp_current_c"] = None

    # GPU Power
    try:
        data["gpu_power_w"] = float(pynvml.nvmlDeviceGetPowerUsage(handle)) / 1000.0
    except Exception:
        data["gpu_power_w"] = None

    # GPU Clocks
    try:
        data["gpu_clock_mhz"] = float(pynvml.nvmlDeviceGetClockInfo(handle, pynvml.NVML_CLOCK_SM))
    except Exception:
        data["gpu_clock_mhz"] = None

    try:
        data["gpu_mem_clock_mhz"] = float(pynvml.nvmlDeviceGetClockInfo(handle, pynvml.NVML_CLOCK_MEM))
    except Exception:
        data["gpu_mem_clock_mhz"] = None

    # GPU Util
    try:
        rates = pynvml.nvmlDeviceGetUtilizationRates(handle)
        data["gpu_util"] = rates.gpu / 100.0
        data["gpu_mem_util"] = rates.memory / 100.0
    except Exception:
        data["gpu_util"] = None
        data["gpu_mem_util"] = None

    # Memory
    try:
        mem = pynvml.nvmlDeviceGetMemoryInfo(handle)
        data["gpu_mem_used_gb"] = round(mem.used / (1024 ** 3), 3)
        data["gpu_mem_total_gb"] = round(mem.total / (1024 ** 3), 3)
    except Exception:
        data["gpu_mem_used_gb"] = None
        data["gpu_mem_total_gb"] = None

    # Throttle reasons
    try:
        reasons = pynvml.nvmlDeviceGetCurrentClocksThrottleReasons(handle)
        # HW slowdown bitmask: 0x40, SW slowdown: 0x20
        data["gpu_throttled"] = int(bool(reasons & (0x40 | 0x20)))
        data["gpu_throttle_bitmask"] = int(reasons)
    except Exception:
        data["gpu_throttled"] = 0
        data["gpu_throttle_bitmask"] = 0

    return data


def main():
    import argparse
    import json

    parser = argparse.ArgumentParser(description="TATO Standalone Telemetry Logger (1Hz)")
    parser.add_argument("--model", type=str, default="ResNet-18", help="Ten model AI (vd: ResNet-18, MobileNetV3-Small, EfficientNet-B0)")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size (vd: 16, 32, 64)")
    parser.add_argument("--precision", type=str, default="AMP", help="Precision: FP32 hoac AMP")
    parser.add_argument("--tag", type=str, default="", help="Ma lan chay (vd: RUN_001)")
    args = parser.parse_args()

    print("=" * 65)
    print("   TATO Standalone Hardware Telemetry Logger (1Hz)")
    print("=" * 65)
    if args.model:
        print(f"[*] Thong tin model: {args.model} | Batch: {args.batch_size} | Precision: {args.precision}")

    handle = None
    if HAS_NVML:
        try:
            device_count = pynvml.nvmlDeviceGetCount()
            handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            name = pynvml.nvmlDeviceGetName(handle)
            if isinstance(name, bytes):
                name = name.decode("utf-8")
            print(f"[+] GPU tim thay: {name} ({device_count} card)")
        except Exception as e:
            print(f"[!] Khong the lay handle GPU: {e}")
    amd = None
    if handle is None:
        amd = find_amd_gpu()
        if amd:
            print(f"[+] GPU AMD (driver amdgpu, doc sysfs): {amd['device']} | hwmon: {amd['hwmon'] or 'khong co'}")
        elif not HAS_NVML:
            print("[!] Canh bao: Khong tim thay NVML (NVIDIA) va khong thay GPU AMD trong /sys/class/drm.")
            print("    NVIDIA: pip install nvidia-ml-py psutil")

    tag = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_file = f"telemetry_{tag}.csv"
    print(f"[*] File ket qua se luu tai: {os.path.abspath(out_file)}")
    print("-" * 65)
    print(" DANG GHI TELEMETRY... (Nhan Ctrl+C tren cua so nay de DUNG)")
    print("-" * 65)

    records = []
    fieldnames = [
        "timestamp", "time_iso", "gpu_temp_current_c", "gpu_power_w",
        "gpu_clock_mhz", "gpu_mem_clock_mhz", "gpu_util", "gpu_mem_util",
        "gpu_mem_used_gb", "gpu_mem_total_gb", "gpu_throttled", "gpu_throttle_bitmask",
        "cpu_temp_current_c", "cpu_util", "ram_used_gb"
    ]

    with open(out_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        try:
            while True:
                loop_t0 = time.time()
                if handle is not None:
                    sample = get_gpu_sample(handle)
                elif amd:
                    sample = get_amd_sample(amd)
                else:
                    sample = {"timestamp": time.time(), "time_iso": datetime.now(timezone.utc).isoformat()}

                # CPU / RAM
                if HAS_PSUTIL:
                    sample["cpu_util"] = round(psutil.cpu_percent(interval=None) / 100.0, 3)
                    sample["ram_used_gb"] = round(psutil.virtual_memory().used / (1024 ** 3), 2)
                    sample["cpu_temp_current_c"] = cpu_temperature()
                else:
                    sample["cpu_util"] = None
                    sample["ram_used_gb"] = None
                    sample["cpu_temp_current_c"] = None

                writer.writerow(sample)
                f.flush()

                # Live terminal update
                gpu_t = sample.get("gpu_temp_current_c") or "--"
                pwr = sample.get("gpu_power_w") or "--"
                clk = sample.get("gpu_clock_mhz") or "--"
                util = f"{sample.get('gpu_util', 0) * 100:.0f}%" if sample.get("gpu_util") is not None else "--"
                thr = " [THROTTLED!]" if sample.get("gpu_throttled") else ""

                print(f"\r  [Live] GPU: {gpu_t}°C | {pwr}W | {clk}MHz | Util: {util}{thr}   ", end="", flush=True)

                elapsed = time.time() - loop_t0
                time.sleep(max(0.01, 1.0 - elapsed))

        except (KeyboardInterrupt, SystemExit):
            pass

    print(f"\n\n[+] Da ket thuc va luu telemetry vao: {out_file}")

    # Save companion model info JSON
    info_file = f"telemetry_{tag}_info.json"
    run_meta = {
        "run_tag": args.tag or f"RUN_{tag}",
        "telemetry_file": out_file,
        "backbone": args.model,
        "batch_size": args.batch_size,
        "precision": args.precision,
        "target_reached": None,
        "best_macro_f1": None,
        "time_to_target_s": None,
        "notes": "Dien ket qua F1/Target sau khi train"
    }
    with open(info_file, "w", encoding="utf-8") as f:
        json.dump(run_meta, f, indent=2, ensure_ascii=False)
    print(f"[+] File thong tin model tao tai: {info_file} (co the mo dien them F1)")


if __name__ == "__main__":
    main()
