"""Lưu/nạp checkpoint và suy luận theo lô — dùng chung cho train, evaluate và predict."""
from __future__ import annotations

import contextlib
import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from .data import ImageDataset, close_loader, make_loader
from .metrics import evaluation_tables
from .models import build_model
from .preprocessing import PreprocessConfig, build_transform
from .utils import write_csv

CHECKPOINT_FORMAT = "throat_ai.checkpoint.v2"


def save_checkpoint(path, model: torch.nn.Module, meta: dict) -> None:
    """Chỉ lưu state_dict + metadata kiểu cơ bản => nạp lại an toàn bằng torch.load(weights_only=True)."""
    path = Path(path)
    state = {key: value.detach().to("cpu") for key, value in model.state_dict().items()}
    payload = {"format": CHECKPOINT_FORMAT, **meta, "state_dict": state}
    tmp = path.with_name(path.name + ".tmp")
    torch.save(payload, tmp)
    os.replace(tmp, path)


def load_checkpoint(path, device: torch.device | str = "cpu") -> tuple[torch.nn.Module, dict]:
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if checkpoint.get("format") != CHECKPOINT_FORMAT:
        raise ValueError(f"{path} không phải checkpoint của pipeline này (format={checkpoint.get('format')!r}).")
    model, _ = build_model(checkpoint["model_name"], int(checkpoint["num_classes"]), pretrained=False)
    model.load_state_dict(checkpoint["state_dict"])
    model.to(device).eval()
    meta = {key: value for key, value in checkpoint.items() if key != "state_dict"}
    return model, meta


def autocast_context(device: torch.device, enabled: bool):
    if not enabled:
        return contextlib.nullcontext()
    return torch.autocast(device_type=device.type, dtype=torch.float16)


@torch.inference_mode()
def predict_loader(model: torch.nn.Module, loader, device: torch.device, use_amp: bool = False, check=None) -> np.ndarray:
    """Xác suất softmax (float64) cho toàn bộ loader, đúng thứ tự."""
    model.eval()
    outputs = []
    for step, (images, _) in enumerate(loader):
        if check is not None and step % 10 == 0:
            check()
        images = images.to(device, non_blocking=device.type == "cuda")
        with autocast_context(device, use_amp):
            logits = model(images)
        outputs.append(torch.softmax(logits.float(), dim=1).cpu())
    if not outputs:
        return np.zeros((0, 0))
    return torch.cat(outputs).numpy().astype(np.float64)


def predict_paths(model, meta: dict, paths, device: torch.device, batch_size: int = 32, workers: int = 0,
                  use_amp: bool = False) -> np.ndarray:
    preprocess = PreprocessConfig.from_dict(meta["preprocess"])
    dataset = ImageDataset(paths, [0] * len(paths), preprocess, build_transform(train=False))
    loader = make_loader(dataset, batch_size, shuffle=False, workers=workers, seed=0, pin_memory=device.type == "cuda")
    try:
        return predict_loader(model, loader, device, use_amp)
    finally:
        close_loader(loader)


def save_prediction_files(run_dir, split: str, frame: pd.DataFrame, probs: np.ndarray, classes: list[str]) -> None:
    """<split>_predictions.csv (từng ảnh), <split>_per_class.csv, <split>_confusion.csv."""
    run_dir = Path(run_dir)
    table = frame[["source", "rel_path", "patient_id", "label"]].copy()
    table["prediction"] = [classes[i] for i in probs.argmax(1)]
    table["confidence"] = probs.max(1)
    table["correct"] = table["prediction"] == table["label"]
    for index, name in enumerate(classes):
        table[f"p_{name}"] = probs[:, index]
    write_csv(table, run_dir / f"{split}_predictions.csv")
    per_class, confusion = evaluation_tables(frame["y"].to_numpy(), probs, classes)
    write_csv(per_class, run_dir / f"{split}_per_class.csv")
    write_csv(confusion, run_dir / f"{split}_confusion.csv")
