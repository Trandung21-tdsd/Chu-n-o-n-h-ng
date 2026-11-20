import numpy as np
import pandas as pd
import pytest
import torch

from throat_ai.metrics import apply_temperature, compute_metrics, expected_calibration_error, fit_temperature, patient_metrics
from throat_ai.models import REGISTRY, build_model, estimate_gmacs, resolve_model_name
from throat_ai.stopping import EarlyStopping, TargetTracker
from throat_ai.tato_export import STATIC_FEATURES, DataLeakageError, validate_static_feature_names
from throat_ai.telemetry import integrate_energy_wh, ols_slope, pre_run_features, thermal_event_in_horizon


def test_early_stopping_and_target_confirmation():
    stopper = EarlyStopping(patience=2, min_delta=0.01)
    assert stopper.step(0.50, 1) and not stopper.step(0.505, 2) and not stopper.should_stop
    assert not stopper.step(0.40, 3) and stopper.should_stop and stopper.best_epoch == 1
    tracker = TargetTracker(target=0.85, consecutive=2)
    assert not tracker.step(0.86, 1) and not tracker.step(0.80, 2) and not tracker.step(0.87, 3)
    assert tracker.step(0.85, 4) and tracker.reached_epoch == 4
    assert not tracker.step(0.90, 5)  # chỉ báo 1 lần


def test_energy_integration_units_and_gaps():
    rows = [{"elapsed_s": float(t), "gpu_power_w": 100.0} for t in range(0, 37)]
    assert integrate_energy_wh(rows, 0, 36) == pytest.approx(1.0)  # 100 W x 36 s = 1 Wh
    assert integrate_energy_wh(rows, 5.5, 23.5) == pytest.approx(0.5)
    rows_gap = [r for r in rows if not 10 <= r["elapsed_s"] <= 20]
    assert integrate_energy_wh(rows_gap, 0, 36) is None  # không tự lấp khoảng trống
    assert integrate_energy_wh([{"elapsed_s": 0.0, "gpu_power_w": None}], 0, 10) is None


def test_slope_and_pre_run_features_use_real_timestamps():
    assert ols_slope([0, 2, 4], [50, 51, 52]) == pytest.approx(0.5)
    assert ols_slope([1], [50]) is None
    rows = [{"elapsed_s": float(t), "gpu_temp_c": 40 + 0.1 * t, "cpu_util": 0.1} for t in range(0, 120)]
    features = pre_run_features(rows)
    assert features["pre_run_gpu_temp_slope_60s_c_per_s"] == pytest.approx(0.1)
    assert features["pre_run_gpu_temp_max_60s_c"] == pytest.approx(40 + 0.1 * 119)
    mac_rows = [{"elapsed_s": 0.0, "gpu_temp_c": None}, {"elapsed_s": 1.0, "gpu_temp_c": None}]
    assert pre_run_features(mac_rows)["pre_run_gpu_temp_mean_60s_c"] is None  # không có cảm biến -> trống, không 0
    flags = [{"elapsed_s": float(t), "gpu_thermal_throttled": 1 if t == 30 else 0} for t in range(0, 100)]
    assert thermal_event_in_horizon(flags, 0, 60) == (1, 0)
    assert thermal_event_in_horizon(flags, 35, 60) == (0, 0)
    assert thermal_event_in_horizon(flags, 60, 60) == (0, 1)  # run kết thúc trước khi hết horizon -> censored
    assert thermal_event_in_horizon([{"elapsed_s": 1.0, "gpu_thermal_throttled": float("nan")}], 0, 60)[0] is None


def test_metrics_patient_aggregation_and_temperature():
    y = np.array([0, 0, 1, 1, 2, 2])
    probs = np.array([[.8, .1, .1], [.4, .5, .1], [.1, .8, .1], [.2, .7, .1], [.1, .1, .8], [.3, .3, .4]])
    metrics = compute_metrics(y, probs)
    assert metrics["accuracy"] == pytest.approx(5 / 6) and 0 < metrics["macro_f1"] < 1
    frame = pd.DataFrame({"patient_key": ["a", "a", "b", "b", "c", "c"], "y": y})
    assert patient_metrics(frame, probs)["accuracy"] == pytest.approx(1.0)
    assert patient_metrics(frame.assign(patient_key=""), probs) == {}
    rng = np.random.default_rng(0)
    labels = rng.integers(0, 3, 400)
    overconfident = np.full((400, 3), 0.005)
    noisy = np.where(rng.random(400) < 0.6, labels, rng.integers(0, 3, 400))
    overconfident[np.arange(400), noisy] = 0.99
    temperature = fit_temperature(labels, overconfident)
    assert temperature > 1.5
    assert expected_calibration_error(labels, apply_temperature(overconfident, temperature)) < \
        expected_calibration_error(labels, overconfident)


@pytest.mark.parametrize("name", sorted(REGISTRY))
def test_every_registry_model_gets_new_head(name):
    model, head = build_model(name, num_classes=5, pretrained=False)
    assert head and head[0].shape[0] == 5
    if name in ("mobilenet_v3_small", "resnet18", "efficientnet_b0", "shufflenet_v2_x1_0"):
        model.eval()
        with torch.no_grad():
            assert model(torch.zeros(2, 3, 64, 64)).shape == (2, 5)


def test_model_aliases_and_gmacs():
    assert resolve_model_name("ResNet-50") == "resnet50"
    assert resolve_model_name("MobileNetV3-Small") == "mobilenet_v3_small"
    with pytest.raises(ValueError):
        resolve_model_name("alexnet-xyz")
    model, _ = build_model("resnet18", 1000, pretrained=False)
    assert estimate_gmacs(model, 224) == pytest.approx(1.81, abs=0.05)


def test_tato_static_features_reject_future_information():
    validate_static_feature_names(STATIC_FEATURES)
    assert len(STATIC_FEATURES) == 26
    for leaked in ("best_macro_f1", "observed_gpu_temp_mean_last_60s_c", "full_run_gpu_temp_max_c",
                   "val_macro_f1", "test_accuracy", "f1_at_epoch_3"):
        with pytest.raises(DataLeakageError):
            validate_static_feature_names(["config_batch_size", leaked])


@pytest.mark.parametrize("name", ["mobilenet_v3_small", "resnet18", "efficientnet_b0", "densenet121", "convnext_tiny", "swin_t"])
def test_grad_cam_produces_normalized_map(name):
    from throat_ai.demo import grad_cam

    model, _ = build_model(name, num_classes=3, pretrained=False)
    cam = grad_cam(model, torch.rand(1, 3, 96, 96), class_index=1)
    assert cam is not None and cam.shape == (96, 96)
    assert 0.0 <= cam.min() and cam.max() <= 1.0 + 1e-6
