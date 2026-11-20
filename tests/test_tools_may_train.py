"""Bộ công cụ máy train Linux (tools_may_train): 4 model có trong pipeline, đọc telemetry GPU AMD qua sysfs, script hợp lệ."""
import importlib.util
import shutil
import subprocess
from pathlib import Path

import pytest

from throat_ai.models import build_model, resolve_model_name

TOOLS = Path(__file__).resolve().parents[1] / "tools_may_train"


def _load(name):
    spec = importlib.util.spec_from_file_location(name, TOOLS / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_four_linux_models_exist_in_pipeline():
    train = _load("train")
    assert set(train.MODELS) == {"ResNet-50", "Swin-T", "MobileNetV2", "DenseNet-121"}
    for menu_name, key in train.MODELS.items():
        assert resolve_model_name(menu_name) == key
        model, head = build_model(key, 3, pretrained=False)
        assert head[0].shape[0] == 3


def test_amd_sysfs_telemetry(tmp_path):
    logger = _load("telemetry_logger_standalone")
    dev, hw = tmp_path / "device", tmp_path / "device" / "hwmon" / "hwmon2"
    hw.mkdir(parents=True)
    for path, value in {dev / "vendor": "0x1002", hw / "temp1_input": "71000", hw / "power1_average": "120500000",
                        dev / "pp_dpm_sclk": "0: 300Mhz\n7: 1244Mhz *\n", dev / "gpu_busy_percent": "88",
                        dev / "mem_info_vram_used": str(2 * 1024 ** 3), dev / "mem_info_vram_total": str(8 * 1024 ** 3)}.items():
        path.write_text(value)
    sample = logger.get_amd_sample({"device": str(dev), "hwmon": str(hw)})
    assert sample["gpu_temp_current_c"] == 71.0 and sample["gpu_power_w"] == 120.5
    assert sample["gpu_clock_mhz"] == 1244.0 and sample["gpu_util"] == 0.88
    assert sample["gpu_mem_used_gb"] == 2.0 and sample["gpu_mem_total_gb"] == 8.0
    assert sample["gpu_throttled"] is None  # amdgpu không báo cờ throttle -> để trống, không ghi 0
    assert sample["gpu_mem_clock_mhz"] is None and "timestamp" in sample


@pytest.mark.skipif(shutil.which("bash") is None, reason="cần bash")
def test_shell_scripts_syntax():
    for script in TOOLS.glob("*.sh"):
        subprocess.run(["bash", "-n", str(script)], check=True)
