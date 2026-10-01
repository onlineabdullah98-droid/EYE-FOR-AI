import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def _make_image(rng: np.random.Generator, fake: bool, size: int = 64) -> np.ndarray:
    """Synthetic, easily separable data: smooth gradients (REAL) vs. gradients with a periodic artifact (FAKE)."""
    yy, xx = np.mgrid[0:size, 0:size].astype(np.float32) / size
    base = np.stack([xx, yy, (xx + yy) / 2], axis=-1) * rng.uniform(80, 200) + rng.normal(0, 4, (size, size, 3))
    if fake:
        base += 60 * ((np.indices((size, size)).sum(0) % 4) < 2)[..., None]
    return np.clip(base, 0, 255).astype(np.uint8)


@pytest.fixture(scope="session")
def tiny_dataset(tmp_path_factory) -> Path:
    root = tmp_path_factory.mktemp("dataset")
    rng = np.random.default_rng(0)
    for folder, fake in (("REAL", False), ("FAKE", True)):
        (root / "train" / folder).mkdir(parents=True)
        for i in range(30):
            cv2.imwrite(str(root / "train" / folder / f"{i}.png"), _make_image(rng, fake))
    return root
