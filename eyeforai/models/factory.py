"""Model factory: custom CNN baseline and transfer-learning backbones (ResNet50, EfficientNet-B0/B2)."""

from __future__ import annotations

from torch import nn
from torchvision import models as tvm

from .custom_cnn import CustomCNN

SUPPORTED_MODELS = ("custom_cnn", "resnet50", "efficientnet_b0", "efficientnet_b2")

# Name of the attribute that holds the (replaced) classification head.
_HEAD_ATTR = {"custom_cnn": "classifier", "resnet50": "fc", "efficientnet_b0": "classifier", "efficientnet_b2": "classifier"}


def build_model(name: str, num_classes: int = 2, pretrained: bool = True, dropout: float = 0.3) -> nn.Module:
    """Build a model. ``pretrained`` loads ImageNet weights (downloaded once, then cached)."""
    if name == "custom_cnn":
        return CustomCNN(num_classes=num_classes)

    if name == "resnet50":
        model = tvm.resnet50(weights=tvm.ResNet50_Weights.IMAGENET1K_V2 if pretrained else None)
        model.fc = nn.Sequential(nn.Dropout(dropout), nn.Linear(model.fc.in_features, num_classes))
        return model

    if name in ("efficientnet_b0", "efficientnet_b2"):
        ctor, weights = {
            "efficientnet_b0": (tvm.efficientnet_b0, tvm.EfficientNet_B0_Weights.IMAGENET1K_V1),
            "efficientnet_b2": (tvm.efficientnet_b2, tvm.EfficientNet_B2_Weights.IMAGENET1K_V1),
        }[name]
        model = ctor(weights=weights if pretrained else None)
        in_features = model.classifier[-1].in_features
        model.classifier = nn.Sequential(nn.Dropout(dropout), nn.Linear(in_features, num_classes))
        return model

    raise ValueError(f"Unknown model '{name}'. Choose one of {SUPPORTED_MODELS}")


def get_gradcam_layer(model: nn.Module, name: str) -> nn.Module:
    """Last convolutional stage: the best trade-off between semantics and spatial detail."""
    if name == "custom_cnn":
        return model.blocks[-1]
    if name == "resnet50":
        return model.layer4[-1]
    if name.startswith("efficientnet"):
        return model.features[-1]
    raise ValueError(f"Unknown model '{name}'")


def split_param_groups(model: nn.Module, name: str) -> tuple[list[nn.Parameter], list[nn.Parameter]]:
    """Return ``(backbone_params, head_params)``."""
    head = getattr(model, _HEAD_ATTR[name])
    head_ids = {id(p) for p in head.parameters()}
    backbone = [p for p in model.parameters() if id(p) not in head_ids]
    return backbone, list(head.parameters())


def set_backbone_trainable(model: nn.Module, name: str, trainable: bool) -> None:
    """Freeze/unfreeze everything except the classification head (transfer-learning stage 1/2)."""
    backbone, _ = split_param_groups(model, name)
    for p in backbone:
        p.requires_grad = trainable
