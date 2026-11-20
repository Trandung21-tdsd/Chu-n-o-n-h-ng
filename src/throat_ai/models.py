"""Danh sách kiến trúc để benchmark (torchvision; tùy chọn timm) + thay lớp phân loại + đếm tham số/GMACs."""
from __future__ import annotations

import re
from dataclasses import dataclass

import torch
from torch import nn
from torchvision import models as tvm


@dataclass(frozen=True)
class ModelSpec:
    key: str
    weights: str
    display: str
    note: str = ""


REGISTRY: dict[str, ModelSpec] = {spec.key: spec for spec in [
    ModelSpec("mobilenet_v3_small", "MobileNet_V3_Small_Weights.IMAGENET1K_V1", "MobileNetV3-Small", "rất nhẹ, có trong ma trận TATO"),
    ModelSpec("mobilenet_v3_large", "MobileNet_V3_Large_Weights.IMAGENET1K_V2", "MobileNetV3-Large", "nhẹ"),
    ModelSpec("mobilenet_v2", "MobileNet_V2_Weights.IMAGENET1K_V2", "MobileNetV2", "nhẹ, có trong bộ máy train Linux"),
    ModelSpec("shufflenet_v2_x1_0", "ShuffleNet_V2_X1_0_Weights.IMAGENET1K_V1", "ShuffleNetV2-x1.0", "rất nhẹ"),
    ModelSpec("efficientnet_b0", "EfficientNet_B0_Weights.IMAGENET1K_V1", "EfficientNet-B0", "nhẹ, có trong ma trận TATO"),
    ModelSpec("efficientnet_b2", "EfficientNet_B2_Weights.IMAGENET1K_V1", "EfficientNet-B2", "trung bình"),
    ModelSpec("efficientnet_v2_s", "EfficientNet_V2_S_Weights.IMAGENET1K_V1", "EfficientNetV2-S", "nặng vừa"),
    ModelSpec("regnet_y_800mf", "RegNet_Y_800MF_Weights.IMAGENET1K_V2", "RegNetY-800MF", "nhẹ"),
    ModelSpec("resnet18", "ResNet18_Weights.IMAGENET1K_V1", "ResNet-18", "có trong ma trận TATO"),
    ModelSpec("resnet34", "ResNet34_Weights.IMAGENET1K_V1", "ResNet-34", "trung bình"),
    ModelSpec("resnet50", "ResNet50_Weights.IMAGENET1K_V2", "ResNet-50", "có trong ma trận TATO"),
    ModelSpec("densenet121", "DenseNet121_Weights.IMAGENET1K_V1", "DenseNet-121", "hay dùng trong ảnh y tế"),
    ModelSpec("convnext_tiny", "ConvNeXt_Tiny_Weights.IMAGENET1K_V1", "ConvNeXt-Tiny", "CNN hiện đại"),
    ModelSpec("swin_t", "Swin_T_Weights.IMAGENET1K_V1", "Swin-T", "transformer cửa sổ"),
    ModelSpec("vit_b_16", "ViT_B_16_Weights.IMAGENET1K_V1", "ViT-B/16", "rất nặng (86M), chỉ nhận ảnh 224"),
]}

FIXED_RESOLUTION = {"vit_b_16": 224}


def _canon(text: str) -> str:
    return re.sub(r"[^a-z0-9]", "", str(text).lower())


_ALIASES = {_canon(s.key): s.key for s in REGISTRY.values()}
_ALIASES.update({_canon(s.display): s.key for s in REGISTRY.values()})


def resolve_model_name(name: str) -> str:
    """Chấp nhận tên registry ('resnet50'), tên hiển thị kiểu TATO ('ResNet-50') hoặc 'timm:<tên>'."""
    if str(name).startswith("timm:"):
        return str(name)
    key = _ALIASES.get(_canon(name))
    if key is None:
        raise ValueError(f"Không biết model {name!r}. Có sẵn: {', '.join(REGISTRY)} — hoặc timm:<tên> nếu đã cài timm.")
    return key


def display_name(name: str) -> str:
    key = resolve_model_name(name)
    return REGISTRY[key].display if key in REGISTRY else key


def _replace_last_linear(model: nn.Module, num_classes: int) -> nn.Linear:
    last = None
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            last = (name, module)
    if last is None:
        raise ValueError("Không tìm thấy lớp Linear cuối để thay bằng lớp phân loại mới.")
    name, old = last
    parent = model
    parts = name.split(".")
    for part in parts[:-1]:
        parent = getattr(parent, part)
    head = nn.Linear(old.in_features, num_classes)
    setattr(parent, parts[-1], head)
    return head


def build_model(name: str, num_classes: int, pretrained: bool = True) -> tuple[nn.Module, list[nn.Parameter]]:
    """Tạo model với lớp phân loại `num_classes` lớp. Trả về (model, tham số của lớp phân loại)."""
    key = resolve_model_name(name)
    if key.startswith("timm:"):
        try:
            import timm
        except ImportError as exc:
            raise ImportError("Model timm:* cần cài thêm: pip install timm") from exc
        model = timm.create_model(key[5:], pretrained=pretrained, num_classes=num_classes)
        return model, list(model.get_classifier().parameters())
    weights = tvm.get_weight(REGISTRY[key].weights) if pretrained else None
    model = tvm.get_model(key, weights=weights)
    head = _replace_last_linear(model, num_classes)
    return model, list(head.parameters())


def count_parameters(model: nn.Module) -> tuple[float, float]:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total / 1e6, trainable / 1e6


def estimate_gmacs(model: nn.Module, resolution: int) -> float | None:
    """GMACs cho 1 ảnh (cùng quy ước 'GFLOPs' của torchvision/TATO, vd ResNet-50 ~ 4.1)."""
    try:
        from torch.utils.flop_counter import FlopCounterMode

        was_training = model.training
        model.eval()
        with torch.no_grad(), FlopCounterMode(display=False) as counter:
            model(torch.zeros(1, 3, int(resolution), int(resolution)))
        model.train(was_training)
        return counter.get_total_flops() / 2e9
    except Exception:
        return None


def model_table() -> list[dict]:
    rows = []
    for spec in REGISTRY.values():
        meta = tvm.get_weight(spec.weights).meta
        rows.append({"model": spec.key, "display": spec.display, "params_m": round(meta["num_params"] / 1e6, 2),
                     "gflops_torchvision": meta.get("_ops"), "note": spec.note})
    return rows


def pick_device(request: str = "auto") -> torch.device:
    if request == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda:0")
        if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    device = torch.device(request)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("Yêu cầu --device cuda nhưng PyTorch không thấy GPU CUDA (kiểm tra driver / bản torch có CUDA).")
    if device.type == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("Yêu cầu --device mps nhưng máy không hỗ trợ Apple MPS.")
    return device


def amp_available(device: torch.device) -> bool:
    """AMP float16 hỗ trợ trên CUDA và Apple MPS (PyTorch mới). CPU không tính là AMP."""
    if device.type == "cuda":
        return True
    if device.type == "mps":
        try:
            return bool(torch.amp.autocast_mode.is_autocast_available("mps"))
        except Exception:
            return False
    return False


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elif device.type == "mps":
        torch.mps.synchronize()


def empty_cache(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.empty_cache()
    elif device.type == "mps":
        torch.mps.empty_cache()
