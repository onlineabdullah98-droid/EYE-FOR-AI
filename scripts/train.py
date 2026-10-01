"""Train the baseline CNN or a transfer-learning model.

Examples:
  python scripts/train.py --model custom_cnn --epochs 20 --lr 1e-3
  python scripts/train.py --model efficientnet_b0 --epochs 15 --freeze-epochs 3
  python scripts/train.py --model resnet50 --batch-size 16
  python scripts/train.py --model efficientnet_b2 --class-weights      # imbalanced data (CASIA)
  python scripts/train.py --model efficientnet_b0 --init-from models/efficientnet_b0_v3.pt --freeze-epochs 0
"""

from __future__ import annotations

import argparse

import _bootstrap  # noqa: F401

from eyeforai.models import SUPPORTED_MODELS
from eyeforai.training import TrainConfig, fit


def main() -> None:
    d = TrainConfig()
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", choices=SUPPORTED_MODELS, default=d.model_name)
    p.add_argument("--splits-dir", default=d.splits_dir)
    p.add_argument("--img-size", type=int, default=None, help="Default depends on the model (128/224/260)")
    p.add_argument("--batch-size", type=int, default=d.batch_size)
    p.add_argument("--epochs", type=int, default=d.epochs)
    p.add_argument("--freeze-epochs", type=int, default=d.freeze_epochs)
    p.add_argument("--lr", type=float, default=d.lr)
    p.add_argument("--backbone-lr-mult", type=float, default=d.backbone_lr_mult)
    p.add_argument("--weight-decay", type=float, default=d.weight_decay)
    p.add_argument("--label-smoothing", type=float, default=d.label_smoothing)
    p.add_argument("--patience", type=int, default=d.patience)
    p.add_argument("--num-workers", type=int, default=d.num_workers)
    p.add_argument("--seed", type=int, default=d.seed)
    p.add_argument("--device", default=None, help="cuda / cpu / mps (auto by default)")
    p.add_argument("--no-pretrained", action="store_true", help="Train transfer models from scratch")
    p.add_argument("--no-augment", action="store_true")
    p.add_argument("--no-amp", action="store_true")
    p.add_argument("--class-weights", action="store_true")
    p.add_argument("--init-from", default=None, help="Continue training from an existing checkpoint (.pt)")
    p.add_argument("--out-name", default=None, help="File name in models/ (default: <model>_best.pt)")
    a = p.parse_args()

    cfg = TrainConfig(
        model_name=a.model, splits_dir=a.splits_dir, img_size=a.img_size, batch_size=a.batch_size,
        epochs=a.epochs, freeze_epochs=a.freeze_epochs, lr=a.lr, backbone_lr_mult=a.backbone_lr_mult,
        weight_decay=a.weight_decay, label_smoothing=a.label_smoothing, patience=a.patience,
        num_workers=a.num_workers, seed=a.seed, device=a.device, pretrained=not a.no_pretrained,
        augment=not a.no_augment, amp=not a.no_amp, class_weights=a.class_weights,
        init_from=a.init_from, out_name=a.out_name,
    )
    fit(cfg)


if __name__ == "__main__":
    main()
