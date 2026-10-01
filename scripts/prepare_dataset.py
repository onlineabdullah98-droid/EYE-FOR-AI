"""Scan a dataset folder and write stratified train/val/test CSV manifests.

Works with any layout where images sit (at any depth) under folders named
real/fake (CIFAKE, 140k faces) or Au/Tp (CASIA). Images are not copied.

Examples:
  python scripts/prepare_dataset.py --source data/raw/cifake
  python scripts/prepare_dataset.py --source data/raw/cifake --max-per-class 10000   # quick experiments
  python scripts/prepare_dataset.py --source data/raw/cifake data/raw/faces140k --max-per-class 6000  # combine
  python scripts/prepare_dataset.py --source data/raw/openfake@5000 data/raw/genimage@3500 data/raw/cifake@2000
"""

from __future__ import annotations

import argparse
import random
from collections import Counter

import _bootstrap  # noqa: F401
from sklearn.model_selection import train_test_split

from eyeforai.config import CLASS_NAMES, SPLITS_DIR
from eyeforai.data.dataset import scan_directory, write_split


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--source", nargs="+", required=True,
                        help="Dataset root folders; append @N to cap images per class for that source")
    parser.add_argument("--out", default=str(SPLITS_DIR))
    parser.add_argument("--val-size", type=float, default=0.15)
    parser.add_argument("--test-size", type=float, default=0.15)
    parser.add_argument("--max-per-class", type=int, default=None,
                        help="Max images per class from EACH source (faster experiments, balanced mixing)")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    rng = random.Random(args.seed)
    samples = []
    for src in args.source:
        # "path@N" overrides --max-per-class for that source.
        src, _, cap = src.rpartition("@") if "@" in src else (src, "", "")
        limit = int(cap) if cap else args.max_per_class
        found = scan_directory(src)
        if limit:  # cap each source separately so a small dataset is not drowned out when combining
            by_label: dict[int, list] = {0: [], 1: []}
            for s in found:
                by_label[s[1]].append(s)
            found = [s for lbl in by_label
                     for s in rng.sample(by_label[lbl], min(limit, len(by_label[lbl])))]
        print(f"{src}: {len(found)} images")
        samples += found
    if not samples:
        raise SystemExit("No labelled images found. Folder names must be real/fake or Au/Tp (case-insensitive).")

    labels = [lbl for _, lbl in samples]
    holdout = args.val_size + args.test_size
    train, rest = train_test_split(samples, test_size=holdout, stratify=labels, random_state=args.seed)
    val, test = train_test_split(rest, test_size=args.test_size / holdout, stratify=[l for _, l in rest],
                                 random_state=args.seed)

    for name, split in (("train", train), ("val", val), ("test", test)):
        write_split(split, f"{args.out}/{name}.csv")
        counts = Counter(l for _, l in split)
        print(f"{name:5s}: {len(split):6d} images | " + " | ".join(f"{CLASS_NAMES[c]}={counts[c]}" for c in (0, 1)))
    print(f"\nManifests written to {args.out}. Next step:\n  python scripts/train.py --model efficientnet_b0")


if __name__ == "__main__":
    main()
