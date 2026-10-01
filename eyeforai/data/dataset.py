"""Dataset discovery, train/val/test split manifests and DataLoaders."""

from __future__ import annotations

import csv
from collections import Counter
from pathlib import Path
from typing import Callable, Sequence

import torch
from torch.utils.data import DataLoader, Dataset

from ..config import FAKE_ALIASES, FAKE_LABEL, IMAGE_EXTENSIONS, REAL_ALIASES, REAL_LABEL
from ..utils import read_image_rgb
from .transforms import EvalTransform, TrainTransform

Sample = tuple[str, int]


def label_from_path(path: Path) -> int | None:
    """Infer the label from the nearest parent folder whose name is a known alias."""
    for part in reversed(path.parent.parts):
        name = part.lower()
        if name in REAL_ALIASES:
            return REAL_LABEL
        if name in FAKE_ALIASES:
            return FAKE_LABEL
    return None


def scan_directory(root: str | Path) -> list[Sample]:
    """Recursively collect ``(path, label)`` pairs from any real/fake folder layout."""
    samples: list[Sample] = []
    for path in sorted(Path(root).rglob("*")):
        if path.suffix.lower() in IMAGE_EXTENSIONS and path.is_file():
            label = label_from_path(path)
            if label is not None:
                samples.append((str(path.resolve()), label))
    return samples


def write_split(samples: Sequence[Sample], csv_path: str | Path) -> None:
    csv_path = Path(csv_path)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["path", "label"])
        writer.writerows(samples)


def load_split(csv_path: str | Path) -> list[Sample]:
    with Path(csv_path).open(newline="", encoding="utf-8") as f:
        return [(row["path"], int(row["label"])) for row in csv.DictReader(f)]


class FakeImageDataset(Dataset):
    def __init__(self, samples: Sequence[Sample], transform: Callable):
        self.samples = list(samples)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        path, label = self.samples[idx]
        return self.transform(read_image_rgb(path)), label

    @property
    def labels(self) -> list[int]:
        return [label for _, label in self.samples]

    def class_weights(self) -> torch.Tensor:
        """Inverse-frequency weights for CrossEntropyLoss (useful for imbalanced CASIA)."""
        counts = Counter(self.labels)
        total = len(self.samples)
        return torch.tensor([total / (2 * max(counts.get(c, 0), 1)) for c in (0, 1)], dtype=torch.float32)


def build_dataloaders(
    splits_dir: str | Path,
    img_size: int,
    batch_size: int = 32,
    num_workers: int = 2,
    augment: bool = True,
) -> dict[str, DataLoader]:
    """Create loaders for every ``{train,val,test}.csv`` present in ``splits_dir``."""
    splits_dir = Path(splits_dir)
    loaders: dict[str, DataLoader] = {}
    for split in ("train", "val", "test"):
        csv_path = splits_dir / f"{split}.csv"
        if not csv_path.exists():
            continue
        is_train = split == "train"
        transform = TrainTransform(img_size) if (is_train and augment) else EvalTransform(img_size)
        ds = FakeImageDataset(load_split(csv_path), transform)
        loaders[split] = DataLoader(
            ds,
            batch_size=batch_size,
            shuffle=is_train,
            num_workers=num_workers,
            pin_memory=torch.cuda.is_available(),
            drop_last=is_train and len(ds) > batch_size,
            persistent_workers=num_workers > 0,
        )
    if "train" not in loaders or "val" not in loaders:
        raise FileNotFoundError(f"Expected train.csv and val.csv in {splits_dir}. Run scripts/prepare_dataset.py first.")
    return loaders
