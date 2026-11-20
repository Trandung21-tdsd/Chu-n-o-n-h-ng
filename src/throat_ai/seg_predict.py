"""Dùng model phân đoạn đã chọn (outputs/bagls_seg/selection.json): vẽ 2 dây thanh + thanh môn, đo, rút dấu hiệu.

    from throat_ai.seg_predict import SegAnalyzer
    seg = SegAnalyzer.from_benchmark()                 # None nếu chưa chạy seg_benchmark
    seg.analyse_image("anh.png")                       # mask + số đo + mô tả 1 frame
    seg.analyse_video(["f1.png", "f2.png", ...])       # gộp nhiều frame CÙNG 1 video -> dấu hiệu + nhóm bệnh gợi ý
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from .glottis import assess_video, describe_image, image_features, sign_classifier_proba, video_features
from .models import pick_device
from .seg_data import center_square, load_mask, load_seg_config, mask_path_for
from .seg_models import seg_display_name
from .seg_train import load_seg_checkpoint, predict_masks
from .utils import DEFAULT_OUTPUT_DIR, read_json


def default_seg_dir() -> Path:
    return DEFAULT_OUTPUT_DIR / load_seg_config()["task"]


class SegAnalyzer:
    def __init__(self, out_dir, device="auto") -> None:
        out_dir = Path(out_dir)
        self.selection = read_json(out_dir / "selection.json")
        self.device = pick_device(device) if isinstance(device, str) else device
        self.model, self.meta = load_seg_checkpoint(self.selection["checkpoint"], self.device)
        self.size, self.clahe = int(self.meta["resolution"]), bool(self.meta["clahe"])
        ranges_path = out_dir / "reference_ranges.json"
        self.ranges = read_json(ranges_path)["ranges"] if ranges_path.exists() else {}
        self.sign_model = (self.selection.get("sign_classifier") or {}).get("model")
        self.cfg = load_seg_config()

    @classmethod
    def from_benchmark(cls, out_dir=None, device="auto") -> "SegAnalyzer | None":
        out_dir = Path(out_dir) if out_dir else default_seg_dir()
        if not (out_dir / "selection.json").exists():
            return None
        return cls(out_dir, device)

    @property
    def info(self) -> dict:
        test = self.selection.get("test_metrics", {})
        return {"run_id": self.selection.get("run_id"), "architecture": seg_display_name(self.meta["model_name"]),
                "test_mean_dice": test.get("test_mean_dice"), "test_dice_glottis": test.get("test_dice_glottis")}

    def predict(self, paths) -> list[np.ndarray]:
        return predict_masks(self.model, [str(p) for p in paths], self.size, self.clahe, self.device)

    def analyse_image(self, path, mask: np.ndarray | None = None) -> dict:
        from .preprocessing import load_rgb

        image = center_square(np.asarray(load_rgb(path), dtype=np.uint8), self.size, nearest=False)
        pred = mask if mask is not None else self.predict([path])[0]
        truth_path = mask_path_for(path, self.cfg)
        truth = center_square(load_mask(truth_path), self.size, nearest=True) if truth_path else None
        feat = image_features(pred)
        dice = None
        if truth is not None:
            scores = []
            for c in (1, 2, 3):
                denom = int((truth == c).sum() + (pred == c).sum())
                scores.append(1.0 if denom == 0 else 2 * int(((truth == c) & (pred == c)).sum()) / denom)
            dice = {"left_fold": scores[0], "right_fold": scores[1], "glottis": scores[2], "mean": float(np.mean(scores))}
        return {"features": feat, "description": describe_image(feat), "dice_vs_ground_truth": dice,
                "_image": image, "_pred": pred, "_truth": truth}

    def analyse_video(self, paths, masks: list[np.ndarray] | None = None) -> dict:
        masks = masks if masks is not None else self.predict(paths)
        feat = video_features([image_features(m) for m in masks])
        assessment = assess_video(feat, self.ranges)
        proba = sign_classifier_proba(self.sign_model, feat)
        return {"n_frames": len(masks), "features": feat, "signs": assessment["signs"], "support": assessment["support"],
                "rule_leaning": assessment["leaning"], "sign_classifier_proba": proba,
                "sign_classifier_prediction": max(proba, key=proba.get) if proba else None}
