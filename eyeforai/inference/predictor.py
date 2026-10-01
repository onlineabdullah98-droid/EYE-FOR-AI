"""High-level inference API: classification + confidence + Grad-CAM + ELA in one call."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch

from ..config import CLASS_NAMES, FAKE_LABEL, MIN_RELIABLE_SIZE, MODELS_DIR, UNCERTAIN
from ..data.transforms import EvalTransform
from ..explain import GradCAM, cam_to_regions, draw_regions, error_level_analysis, overlay_heatmap
from ..explain.ela import ELAResult
from ..models import build_model, get_gradcam_layer
from ..utils import ImageInput, get_device, load_checkpoint, read_image_rgb


@dataclass
class AnalysisResult:
    label: str
    confidence: float          # probability of the predicted label, 0-1
    prob_fake: float
    prob_real: float
    inference_ms: float
    model_name: str
    image: np.ndarray                          # original RGB
    cam: np.ndarray | None = None              # Grad-CAM map in [0,1] (model input resolution)
    gradcam_overlay: np.ndarray | None = None  # heatmap blended over the original image
    regions: list[tuple[int, int, int, int]] = field(default_factory=list)
    regions_overlay: np.ndarray | None = None  # original image with suspicious boxes
    ela: ELAResult | None = None
    low_resolution: bool = False               # image too small for a confident verdict

    @property
    def is_fake(self) -> bool:
        return self.label == CLASS_NAMES[FAKE_LABEL]

    @property
    def is_uncertain(self) -> bool:
        return self.label == UNCERTAIN

    def summary(self) -> dict:
        return {
            "label": self.label,
            "ai_probability_percent": round(self.prob_fake * 100, 1),
            "confidence": round(self.confidence, 4),
            "prob_fake": round(self.prob_fake, 4),
            "prob_real": round(self.prob_real, 4),
            "inference_ms": round(self.inference_ms, 1),
            "model_name": self.model_name,
            "low_resolution": self.low_resolution,
            "suspicious_regions": [list(map(int, b)) for b in self.regions],
            "ela_mean_error": round(self.ela.mean_error, 3) if self.ela else None,
        }


def _val_f1(path: Path) -> float:
    try:
        return float(torch.load(str(path), map_location="cpu", weights_only=False, mmap=True)["val_metrics"]["f1"])
    except Exception:
        return -1.0


def find_checkpoints(models_dir: str | Path = MODELS_DIR) -> list[Path]:
    """Checkpoints in ``models_dir``, best validation F1 first (newest first on ties)."""
    paths = sorted(Path(models_dir).glob("*.pt"), key=lambda p: p.stat().st_mtime, reverse=True)
    return sorted(paths, key=_val_f1, reverse=True)


class Predictor:
    def __init__(self, checkpoint_path: str | Path, device: str | None = None, threshold: float | None = None,
                 real_threshold: float | None = None):
        """``threshold``: AI likelihood at or above which the verdict is AI (FAKE).
        ``real_threshold``: AI likelihood below which the verdict is REAL. In between: UNCERTAIN.
        Both default to the values calibrated on validation data at training time (0.5 for older checkpoints).
        """
        self.device = get_device(device)
        ckpt = load_checkpoint(checkpoint_path, self.device)
        self.model_name: str = ckpt["model_name"]
        self.img_size: int = ckpt["img_size"]
        self.class_names: list[str] = ckpt.get("class_names", list(CLASS_NAMES))
        self.val_metrics: dict = ckpt.get("val_metrics", {})
        self.calibrated: dict = ckpt.get("thresholds", {"real": 0.5, "ai": 0.5})
        self.threshold = self.calibrated["ai"] if threshold is None else threshold
        self.real_threshold = self.calibrated["real"] if real_threshold is None else real_threshold

        self.model = build_model(self.model_name, num_classes=ckpt["num_classes"], pretrained=False)
        self.model.load_state_dict(ckpt["state_dict"])
        self.model.to(self.device).eval()
        self.transform = EvalTransform(self.img_size, ckpt.get("mean"), ckpt.get("std"))
        self.target_layer = get_gradcam_layer(self.model, self.model_name)

    def preprocess(self, rgb: np.ndarray) -> torch.Tensor:
        return self.transform(rgb).unsqueeze(0).to(self.device)

    def _make_label(self, prob_fake: float, image_shape: tuple[int, ...] | None = None) -> tuple[str, float]:
        if image_shape is not None and max(image_shape[:2]) < MIN_RELIABLE_SIZE:
            return UNCERTAIN, max(prob_fake, 1.0 - prob_fake)
        if prob_fake >= self.threshold:
            return self.class_names[FAKE_LABEL], prob_fake
        if prob_fake < min(self.real_threshold, self.threshold):
            return self.class_names[1 - FAKE_LABEL], 1.0 - prob_fake
        return UNCERTAIN, max(prob_fake, 1.0 - prob_fake)

    @torch.inference_mode()
    def predict(self, image: ImageInput) -> AnalysisResult:
        """Classification only (fast path, no explanations)."""
        rgb = read_image_rgb(image)
        t0 = time.perf_counter()
        probs = torch.softmax(self.model(self.preprocess(rgb)).float(), dim=1)[0].cpu().numpy()
        elapsed = (time.perf_counter() - t0) * 1000
        label, conf = self._make_label(float(probs[FAKE_LABEL]), rgb.shape)
        return AnalysisResult(label, conf, float(probs[FAKE_LABEL]), float(probs[1 - FAKE_LABEL]), elapsed,
                              self.model_name, rgb, low_resolution=max(rgb.shape[:2]) < MIN_RELIABLE_SIZE)

    def analyze(
        self,
        image: ImageInput,
        cam_target: str = "fake",        # "fake" -> evidence for FAKE, "predicted" -> evidence for the predicted class
        cam_alpha: float = 0.45,
        region_threshold: float = 0.6,
        ela_quality: int = 90,
        with_ela: bool = True,
    ) -> AnalysisResult:
        """Classification + Grad-CAM heatmap + suspicious regions + ELA."""
        rgb = read_image_rgb(image)
        x = self.preprocess(rgb)
        t0 = time.perf_counter()
        with GradCAM(self.model, self.target_layer) as gradcam:
            with torch.no_grad():
                probs = torch.softmax(self.model(x).float(), dim=1)[0].cpu().numpy()
            prob_fake = float(probs[FAKE_LABEL])
            label, conf = self._make_label(prob_fake, rgb.shape)
            target = FAKE_LABEL if (cam_target == "fake" or prob_fake >= 0.5) else 1 - FAKE_LABEL
            cam, _ = gradcam(x, class_idx=target)
        elapsed = (time.perf_counter() - t0) * 1000

        result = AnalysisResult(label, conf, prob_fake, 1.0 - prob_fake, elapsed, self.model_name, rgb, cam=cam,
                                low_resolution=max(rgb.shape[:2]) < MIN_RELIABLE_SIZE)
        result.gradcam_overlay = overlay_heatmap(rgb, cam, alpha=cam_alpha)
        if result.is_fake:
            result.regions = cam_to_regions(cam, rgb.shape, threshold=region_threshold)
        result.regions_overlay = draw_regions(rgb, result.regions)
        if with_ela:
            result.ela = error_level_analysis(rgb, quality=ela_quality)
        return result
