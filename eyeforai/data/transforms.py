"""OpenCV + torchvision preprocessing and augmentation pipelines.

Every transform takes an RGB ``uint8`` numpy array (as returned by
``read_image_rgb``) and returns a normalized ``float32`` tensor of shape (3, H, W).

Augmentations stay moderate: fake-image detectors learn from subtle pixel-level
artifacts, and heavy blur or noise would destroy exactly that signal. The
"social media" ones (resizing, beauty smoothing, sharpening, repeated JPEG) are
applied to both classes, because shared real photos almost always go through
them and a model that never saw them tends to call such photos AI.
"""

from __future__ import annotations

import random

import cv2
import numpy as np
import torch
from torchvision.transforms import v2

from ..config import IMAGENET_MEAN, IMAGENET_STD
from ..utils import resize_rgb


class RandomJPEGCompression:
    def __init__(self, p: float = 0.3, quality: tuple[int, int] = (60, 95)):
        self.p, self.quality = p, quality

    def __call__(self, rgb: np.ndarray) -> np.ndarray:
        if random.random() >= self.p:
            return rgb
        q = random.randint(*self.quality)
        ok, enc = cv2.imencode(".jpg", cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, q])
        if not ok:
            return rgb
        return cv2.cvtColor(cv2.imdecode(enc, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)


class RandomGaussianBlur:
    def __init__(self, p: float = 0.1, kernel_sizes: tuple[int, ...] = (3, 5)):
        self.p, self.kernel_sizes = p, kernel_sizes

    def __call__(self, rgb: np.ndarray) -> np.ndarray:
        if random.random() >= self.p:
            return rgb
        k = random.choice(self.kernel_sizes)
        return cv2.GaussianBlur(rgb, (k, k), 0)


class RandomResample:
    """Shrink and re-enlarge, like a photo that was resized by a chat app or website and then viewed larger."""

    def __init__(self, p: float = 0.25, scale: tuple[float, float] = (0.4, 0.85)):
        self.p, self.scale = p, scale

    def __call__(self, rgb: np.ndarray) -> np.ndarray:
        if random.random() >= self.p:
            return rgb
        h, w = rgb.shape[:2]
        s = random.uniform(*self.scale)
        small = cv2.resize(rgb, (max(8, int(w * s)), max(8, int(h * s))), interpolation=cv2.INTER_AREA)
        return cv2.resize(small, (w, h), interpolation=random.choice((cv2.INTER_LINEAR, cv2.INTER_CUBIC)))


class RandomBeautify:
    """Edge-preserving smoothing, the core of phone "beauty" filters and photo-enhancer apps."""

    def __init__(self, p: float = 0.15):
        self.p = p

    def __call__(self, rgb: np.ndarray) -> np.ndarray:
        if random.random() >= self.p:
            return rgb
        return cv2.bilateralFilter(rgb, random.choice((5, 7, 9)), random.uniform(30, 75), random.uniform(5, 15))


class RandomSharpen:
    """Unsharp masking, applied by many phones, editors and "enhance" buttons."""

    def __init__(self, p: float = 0.15, amount: tuple[float, float] = (0.5, 1.5)):
        self.p, self.amount = p, amount

    def __call__(self, rgb: np.ndarray) -> np.ndarray:
        if random.random() >= self.p:
            return rgb
        blurred = cv2.GaussianBlur(rgb, (0, 0), random.uniform(1.0, 2.0))
        a = random.uniform(*self.amount)
        return cv2.addWeighted(rgb, 1 + a, blurred, -a, 0)


def _to_normalized_tensor(mean, std) -> v2.Compose:
    return v2.Compose([v2.ToImage(), v2.ToDtype(torch.float32, scale=True), v2.Normalize(mean, std)])


class EvalTransform:
    """Deterministic preprocessing: OpenCV resize -> [0,1] -> ImageNet normalization."""

    def __init__(self, img_size: int = 224, mean=IMAGENET_MEAN, std=IMAGENET_STD):
        self.img_size = img_size
        self.to_tensor = _to_normalized_tensor(mean, std)

    def __call__(self, rgb: np.ndarray) -> torch.Tensor:
        return self.to_tensor(resize_rgb(rgb, self.img_size))


class TrainTransform:
    """Training augmentation: OpenCV JPEG/blur, then crop/flip/color jitter in torchvision."""

    def __init__(self, img_size: int = 224, mean=IMAGENET_MEAN, std=IMAGENET_STD):
        self.img_size = img_size
        # "Social media" processing, applied to real and AI images alike so the model learns to ignore it:
        # resizing, beauty smoothing, sharpening and (repeated) JPEG re-compression.
        self.cv_augs = [
            RandomResample(p=0.25),
            RandomBeautify(p=0.15),
            RandomSharpen(p=0.15),
            RandomGaussianBlur(p=0.1),
            RandomJPEGCompression(p=0.5, quality=(50, 95)),
            RandomJPEGCompression(p=0.25, quality=(60, 95)),  # shared again
        ]
        self.tv_augs = v2.Compose(
            [
                v2.ToImage(),
                v2.RandomResizedCrop(img_size, scale=(0.8, 1.0), ratio=(0.9, 1.1), antialias=True),
                v2.RandomHorizontalFlip(p=0.5),
                v2.RandomApply([v2.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1)], p=0.3),
                v2.ToDtype(torch.float32, scale=True),
                v2.Normalize(mean, std),
            ]
        )

    def __call__(self, rgb: np.ndarray) -> torch.Tensor:
        # Bring very large images down first so augmentation stays fast.
        if max(rgb.shape[:2]) > 2 * self.img_size:
            rgb = resize_rgb(rgb, 2 * self.img_size)
        for aug in self.cv_augs:
            rgb = aug(rgb)
        return self.tv_augs(rgb)
