"""Model PHÂN ĐOẠN nhẹ (4 lớp: nền, dây thanh trái ảnh, dây thanh phải ảnh, thanh môn).

    unet_lite              U-Net nhỏ tự viết (~1.9 triệu tham số), train từ đầu — nhẹ nhất, chạy tốt trên CPU/Mac
    unet_resnet18          U-Net với encoder ResNet-18 pretrained ImageNet (~14 triệu) — thường chính xác nhất
    lraspp_mobilenet_v3    LR-ASPP + MobileNetV3-Large pretrained (~3.2 triệu) — model phân đoạn di động của torchvision
    deeplabv3_mobilenet_v3 DeepLabV3 + MobileNetV3-Large pretrained (~11 triệu)

Mọi model trả về logits (N, 4, H, W) cùng kích thước ảnh vào.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn
from torchvision import models as tvm
from torchvision.models import segmentation as tvseg

SEG_MODELS = {
    "unet_lite": "U-Net lite",
    "unet_resnet18": "U-Net ResNet-18",
    "lraspp_mobilenet_v3": "LR-ASPP MobileNetV3",
    "deeplabv3_mobilenet_v3": "DeepLabV3 MobileNetV3",
}


def seg_display_name(name: str) -> str:
    return SEG_MODELS.get(name, name)


def _block(cin: int, cout: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(cin, cout, 3, padding=1, bias=False), nn.BatchNorm2d(cout), nn.ReLU(inplace=True),
        nn.Conv2d(cout, cout, 3, padding=1, bias=False), nn.BatchNorm2d(cout), nn.ReLU(inplace=True),
    )


class _Up(nn.Module):
    def __init__(self, cin: int, cskip: int, cout: int) -> None:
        super().__init__()
        self.conv = _block(cin + cskip, cout)

    def forward(self, x, skip):
        x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        return self.conv(torch.cat([x, skip], dim=1))


class UNetLite(nn.Module):
    def __init__(self, num_classes: int = 4, widths=(16, 32, 64, 128, 256)) -> None:
        super().__init__()
        self.downs = nn.ModuleList()
        cin = 3
        for width in widths:
            self.downs.append(_block(cin, width))
            cin = width
        self.ups = nn.ModuleList(_Up(widths[i + 1], widths[i], widths[i]) for i in reversed(range(len(widths) - 1)))
        self.head = nn.Conv2d(widths[0], num_classes, 1)

    def forward(self, x):
        skips = []
        for index, down in enumerate(self.downs):
            x = down(x if index == 0 else F.max_pool2d(x, 2))
            skips.append(x)
        x = skips.pop()
        for up in self.ups:
            x = up(x, skips.pop())
        return self.head(x)


class UNetResNet18(nn.Module):
    def __init__(self, num_classes: int = 4, pretrained: bool = True) -> None:
        super().__init__()
        weights = tvm.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        r = tvm.resnet18(weights=weights)
        self.encoder = nn.ModuleDict({
            "stem": nn.Sequential(r.conv1, r.bn1, r.relu),  # 1/2, 64
            "layer1": nn.Sequential(r.maxpool, r.layer1),  # 1/4, 64
            "layer2": r.layer2, "layer3": r.layer3, "layer4": r.layer4,  # 1/8 128, 1/16 256, 1/32 512
        })
        self.up4 = _Up(512, 256, 256)
        self.up3 = _Up(256, 128, 128)
        self.up2 = _Up(128, 64, 64)
        self.up1 = _Up(64, 64, 32)
        self.head = nn.Sequential(_block(32, 16), nn.Conv2d(16, num_classes, 1))

    def forward(self, x):
        size = x.shape[-2:]
        s0 = self.encoder["stem"](x)
        s1 = self.encoder["layer1"](s0)
        s2 = self.encoder["layer2"](s1)
        s3 = self.encoder["layer3"](s2)
        s4 = self.encoder["layer4"](s3)
        d = self.up1(self.up2(self.up3(self.up4(s4, s3), s2), s1), s0)
        return self.head(F.interpolate(d, size=size, mode="bilinear", align_corners=False))


class _TorchvisionSeg(nn.Module):
    def __init__(self, model: nn.Module) -> None:
        super().__init__()
        self.model = model
        self.encoder = model.backbone

    def forward(self, x):
        return self.model(x)["out"]


def build_seg_model(name: str, num_classes: int = 4, pretrained: bool = True) -> tuple[nn.Module, list[nn.Parameter]]:
    """Trả về (model, tham_số_encoder_pretrained). Encoder pretrained học với LR nhỏ hơn phần còn lại."""
    backbone_weights = tvm.MobileNet_V3_Large_Weights.IMAGENET1K_V1 if pretrained else None
    if name == "unet_lite":
        model = UNetLite(num_classes)
        return model, []
    if name == "unet_resnet18":
        model = UNetResNet18(num_classes, pretrained)
    elif name == "lraspp_mobilenet_v3":
        model = _TorchvisionSeg(tvseg.lraspp_mobilenet_v3_large(weights=None, weights_backbone=backbone_weights,
                                                               num_classes=num_classes))
    elif name == "deeplabv3_mobilenet_v3":
        model = _TorchvisionSeg(tvseg.deeplabv3_mobilenet_v3_large(weights=None, weights_backbone=backbone_weights,
                                                                   num_classes=num_classes, aux_loss=False))
    else:
        raise ValueError(f"Không có model phân đoạn {name!r}. Có: {', '.join(SEG_MODELS)}")
    return model, list(model.encoder.parameters()) if pretrained else []
