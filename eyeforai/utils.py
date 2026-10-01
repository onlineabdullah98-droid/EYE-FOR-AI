"""Shared helpers: seeding, devices, robust image I/O and checkpoints."""

from __future__ import annotations

import random
from pathlib import Path
from typing import Any, Union

import cv2
import numpy as np
import torch

ImageInput = Union[str, Path, bytes, np.ndarray, "PIL.Image.Image"]  # noqa: F821


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device(preferred: str | None = None) -> torch.device:
    if preferred:
        return torch.device(preferred)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def read_image_rgb(image: ImageInput) -> np.ndarray:
    """Load an image as an RGB uint8 array from a path, raw bytes, PIL image or array.

    Uses ``cv2.imdecode`` instead of ``cv2.imread`` so non-ASCII Windows paths work.
    """
    if isinstance(image, np.ndarray):
        arr = image
        if arr.ndim == 2:
            arr = cv2.cvtColor(arr, cv2.COLOR_GRAY2RGB)
        elif arr.shape[2] == 4:
            arr = cv2.cvtColor(arr, cv2.COLOR_RGBA2RGB)
        return np.ascontiguousarray(arr.astype(np.uint8))

    if hasattr(image, "convert"):  # PIL.Image
        return np.asarray(image.convert("RGB"), dtype=np.uint8).copy()

    if isinstance(image, (bytes, bytearray)):
        buf = np.frombuffer(image, dtype=np.uint8)
    else:
        buf = np.fromfile(str(image), dtype=np.uint8)

    bgr = cv2.imdecode(buf, cv2.IMREAD_COLOR)
    if bgr is None:
        raise ValueError(f"Could not decode image: {image if not isinstance(image, (bytes, bytearray)) else '<bytes>'}")
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def resize_rgb(rgb: np.ndarray, size: int) -> np.ndarray:
    """Resize to ``size x size`` with the interpolation suited to the scale direction."""
    h, w = rgb.shape[:2]
    interp = cv2.INTER_AREA if (h > size or w > size) else cv2.INTER_CUBIC
    return cv2.resize(rgb, (size, size), interpolation=interp)


def save_checkpoint(path: str | Path, **payload: Any) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)
    return path


def load_checkpoint(path: str | Path, device: torch.device | str = "cpu") -> dict:
    return torch.load(str(path), map_location=device, weights_only=False)
