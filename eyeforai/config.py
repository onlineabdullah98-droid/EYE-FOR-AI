"""Project-wide constants and paths."""

from __future__ import annotations

import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
SPLITS_DIR = DATA_DIR / "splits"
MODELS_DIR = PROJECT_ROOT / "models"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"

# Label convention used everywhere: 0 = REAL, 1 = FAKE (FAKE is the positive class).
CLASS_NAMES: tuple[str, str] = ("REAL", "FAKE")
REAL_LABEL, FAKE_LABEL = 0, 1
# Shown when the AI likelihood falls between the calibrated REAL and AI thresholds.
UNCERTAIN = "UNCERTAIN"

# Folder names that map to each label when scanning public datasets.
#   CIFAKE: REAL / FAKE      CASIA 2.0: Au / Tp      140k faces: real / fake
REAL_ALIASES = {"real", "au", "authentic", "original", "pristine", "genuine"}
FAKE_ALIASES = {"fake", "tp", "tampered", "manipulated", "forged", "synthetic", "ai", "generated"}

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

# Recommended input resolution per architecture.
DEFAULT_IMG_SIZE = {
    "custom_cnn": 128,
    "resnet50": 224,
    "efficientnet_b0": 224,
    "efficientnet_b2": 260,
}

# SQLite by default; set e.g. postgresql+psycopg2://user:pass@localhost:5432/eyeforai
DATABASE_URL = os.getenv("EYEFORAI_DATABASE_URL", f"sqlite:///{(OUTPUTS_DIR / 'eyeforai.db').as_posix()}")
