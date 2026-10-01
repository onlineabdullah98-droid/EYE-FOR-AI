"""Grad-CAM (Selvaraju et al., 2017) for highlighting the regions that drive a prediction."""

from __future__ import annotations

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn


class GradCAM:
    """Usage::

        with GradCAM(model, target_layer) as cam:
            heatmap, logits = cam(input_tensor, class_idx=1)
    """

    def __init__(self, model: nn.Module, target_layer: nn.Module):
        self.model = model
        self.activations: torch.Tensor | None = None
        self.gradients: torch.Tensor | None = None
        self._handle = target_layer.register_forward_hook(self._forward_hook)

    def _forward_hook(self, _module, _inputs, output: torch.Tensor) -> None:
        self.activations = output.detach()
        if output.requires_grad:
            output.register_hook(self._save_grad)

    def _save_grad(self, grad: torch.Tensor) -> None:
        self.gradients = grad.detach()

    def __call__(self, x: torch.Tensor, class_idx: int | None = None) -> tuple[np.ndarray, torch.Tensor]:
        """Return ``(cam, logits)`` where ``cam`` is an HxW float map in [0, 1] at the input resolution."""
        self.model.eval()
        self.model.zero_grad(set_to_none=True)
        with torch.enable_grad():
            x = x.clone().requires_grad_(True)  # guarantees a graph even if weights are frozen
            logits = self.model(x)
            if class_idx is None:
                class_idx = int(logits.argmax(dim=1)[0])
            logits[0, class_idx].backward()

        if self.activations is None or self.gradients is None:
            raise RuntimeError("Grad-CAM hooks did not fire - check the target layer.")
        weights = self.gradients.mean(dim=(2, 3), keepdim=True)          # GAP over gradients
        cam = F.relu((weights * self.activations).sum(dim=1, keepdim=True))
        cam = F.interpolate(cam, size=x.shape[-2:], mode="bilinear", align_corners=False)[0, 0]
        cam = cam - cam.min()
        cam = cam / cam.max() if cam.max() > 0 else cam
        return cam.cpu().numpy().astype(np.float32), logits.detach()

    def remove(self) -> None:
        self._handle.remove()

    def __enter__(self) -> "GradCAM":
        return self

    def __exit__(self, *exc) -> None:
        self.remove()


def overlay_heatmap(rgb: np.ndarray, cam: np.ndarray, alpha: float = 0.45, colormap: int = cv2.COLORMAP_JET) -> np.ndarray:
    """Blend a [0,1] heatmap (any size) over an RGB uint8 image."""
    h, w = rgb.shape[:2]
    cam = cv2.resize(cam, (w, h), interpolation=cv2.INTER_LINEAR)
    heat = cv2.cvtColor(cv2.applyColorMap(np.uint8(255 * cam), colormap), cv2.COLOR_BGR2RGB)
    return cv2.addWeighted(rgb, 1 - alpha, heat, alpha, 0)


def cam_to_regions(cam: np.ndarray, image_shape: tuple[int, int], threshold: float = 0.6,
                   min_area_ratio: float = 0.01) -> list[tuple[int, int, int, int]]:
    """Threshold the CAM and return bounding boxes ``(x, y, w, h)`` of suspicious regions in image coords."""
    h, w = image_shape[:2]
    cam = cv2.resize(cam, (w, h), interpolation=cv2.INTER_LINEAR)
    mask = np.uint8(cam >= threshold) * 255
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    min_area = min_area_ratio * h * w
    boxes = [cv2.boundingRect(c) for c in contours if cv2.contourArea(c) >= min_area]
    return sorted(boxes, key=lambda b: b[2] * b[3], reverse=True)


def draw_regions(rgb: np.ndarray, boxes: list[tuple[int, int, int, int]], color=(255, 40, 40)) -> np.ndarray:
    out = rgb.copy()
    thickness = max(2, round(max(rgb.shape[:2]) / 300))
    font_scale = 0.35 * thickness
    for i, (x, y, bw, bh) in enumerate(boxes, start=1):
        cv2.rectangle(out, (x, y), (x + bw, y + bh), color, thickness)
        cv2.putText(out, f"#{i}", (x + thickness + 2, y + int(25 * font_scale) + thickness),
                    cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, max(1, thickness // 2))
    return out
