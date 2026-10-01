"""Evaluate a saved checkpoint on a split (Accuracy, Precision, Recall, F1, ROC-AUC + plots).

  python scripts/evaluate.py --checkpoint models/efficientnet_b0_v3.pt --split test
"""

from __future__ import annotations

import argparse
from pathlib import Path

import _bootstrap  # noqa: F401
from torch.utils.data import DataLoader

from eyeforai.config import OUTPUTS_DIR, SPLITS_DIR
from eyeforai.data import EvalTransform, FakeImageDataset, load_split
from eyeforai.inference import Predictor
from eyeforai.training import evaluate


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--split", default="test")
    p.add_argument("--splits-dir", default=str(SPLITS_DIR))
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--out", default=None)
    a = p.parse_args()

    predictor = Predictor(a.checkpoint)
    ds = FakeImageDataset(load_split(Path(a.splits_dir) / f"{a.split}.csv"), EvalTransform(predictor.img_size))
    loader = DataLoader(ds, batch_size=a.batch_size, num_workers=a.num_workers)
    out = Path(a.out or OUTPUTS_DIR / "eval" / Path(a.checkpoint).stem)
    m = evaluate(predictor.model, loader, predictor.device, out, prefix=a.split)
    print(m["report"])
    print(f"Accuracy {m['accuracy']:.4f} | Precision {m['precision']:.4f} | Recall {m['recall']:.4f} | "
          f"F1 {m['f1']:.4f} | ROC-AUC {m['roc_auc']:.4f} | Loss {m['loss']:.4f}")
    print(f"Saved metrics and plots to {out}")


if __name__ == "__main__":
    main()
