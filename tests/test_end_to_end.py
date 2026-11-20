"""Chạy thử toàn pipeline trên dữ liệu giả lập rất nhỏ, CPU, không tải trọng số (vài chục giây)."""
import json

import pandas as pd

from throat_ai import benchmark, demo
from throat_ai.predict import Diagnoser


def test_benchmark_evaluate_predict_and_tato_export(prepared, cenbi_root, tmp_path, capsys):
    prepared_root, _ = prepared
    out = tmp_path / "bench"
    common = ["--task", "toy", "--prepared-dir", str(prepared_root), "--out", str(out), "--models", "mobilenet_v3_small",
              "--clahe", "off", "on", "--dup-policy", "dedup", "keep", "naive", "--epochs", "2", "--patience", "1",
              "--batch-size", "8", "--workers", "0", "--resolution", "64", "--no-pretrained", "--device", "cpu",
              "--pre-run-seconds", "0", "--target-quality", "0.0"]
    benchmark.main(common + ["--plan"])
    assert "chờ chạy" in capsys.readouterr().out
    benchmark.main(common)

    results = pd.read_csv(out / "benchmark_results.csv")
    assert len(results) == 6 and set(results["status"]) == {"completed"}
    assert results["test_macro_f1"].notna().all()  # test đo SAU khi chọn
    assert results["target_reached"].eq(1).all()  # Q*=0 => luôn đạt, có thời gian tới mục tiêu
    assert results["target_time_to_target_s"].notna().all()
    selection = json.loads((out / "selection.json").read_text(encoding="utf-8"))
    assert selection["test_used_for_selection"] is False and selection["dup_policy"] != "naive"
    assert results.loc[results["selected"], "run_id"].tolist() == [selection["run_id"]]
    for name in ("model_comparison.csv", "all_epochs.csv", "comparison.md", "ablation_clahe.csv", "ablation_duplicates.csv"):
        assert (out / name).exists(), name
    try:
        import matplotlib  # noqa: F401
        for name in ("1_xep_hang.png", "2_learning_curves.png", "3_f1_vs_thoi_gian.png", "4_ablation.png", "5_confusion_test.png", "6_loss_curves.png"):
            assert (out / "charts" / name).exists(), name
        assert "![" in (out / "comparison.md").read_text(encoding="utf-8")
    except ImportError:
        pass
    tato = out / "tato"
    static = pd.read_csv(tato / "static_profiling_dataset.csv")
    assert len(static) == 4  # 2 CLAHE x 2 duplicate (run naive bị loại)
    assert {"config_backbone_MobileNetV3_Small", "pre_run_gpu_temp_current_c", "target_reached", "fold"} <= set(static.columns)
    online = pd.read_csv(tato / "online_observations.csv")
    assert online["observed_epoch"].max() == 2
    index = pd.read_csv(tato / "profiling_index.csv")
    extended = pd.read_csv(tato / "profiling_index_extended.csv")
    assert extended["training_start_unix"].notna().all() and extended["finished_unix"].notna().all()
    assert extended["target_reached_unix"].notna().all()  # Q*=0 -> mọi run đều có mốc đạt mục tiêu
    assert not list(out.glob("runs/*/telemetry.csv"))  # mặc định không tự đo nhiệt
    assert "timestamp_unix" in pd.read_csv(next(out.glob("runs/*/history.csv"))).columns
    assert index["backbone"].eq("MobileNetV3-Small").all() and set(index["precision"]) == {"FP32"}

    # Chạy lại: mọi run đã xong -> bỏ qua, không train lại; cột 'selected' vẫn giữ theo selection.json
    benchmark.main(common + ["--no-finalize"])
    rerun = pd.read_csv(out / "benchmark_results.csv")
    assert len(rerun) == 6 and rerun["selected"].sum() == 1

    image = sorted(cenbi_root.rglob("*.jpg"))[0]
    diagnosis = Diagnoser.from_benchmark(out, device="cpu").diagnose(image)
    assert diagnosis["status"] in ("ok", "uncertain")
    assert abs(sum(diagnosis["probabilities"].values()) - 1.0) < 1e-3
    assert diagnosis["top_k"][0]["label"] in {"Cyst", "Polyp", "SCC"}

    # Demo: ảnh ngẫu nhiên từ tập test -> dự đoán + dấu hiệu + Grad-CAM
    demo.main(["--model", str(out), "--no-seg", "--prepared-dir", str(prepared_root), "--n", "2", "--seed", "1", "--device", "cpu"])
    results = sorted((out / "demo").glob("*.json"))
    assert len(results) == 2
    record = json.loads(results[0].read_text(encoding="utf-8"))
    assert record["true_label"] in {"Cyst", "Polyp", "SCC"} and record["signs_of_predicted_disease"]
    assert "Model dựa" in record["attention"] or "Không có vùng" in record["attention"]
