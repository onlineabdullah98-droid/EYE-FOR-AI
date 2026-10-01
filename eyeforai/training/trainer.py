"""Training loop with two-stage transfer learning, AMP, cosine LR, early stopping and checkpointing."""

from __future__ import annotations

import json
import shutil
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from torch import nn
from tqdm.auto import tqdm

from ..config import CLASS_NAMES, DEFAULT_IMG_SIZE, IMAGENET_MEAN, IMAGENET_STD, MODELS_DIR, OUTPUTS_DIR, SPLITS_DIR
from ..data import build_dataloaders
from ..models import build_model, set_backbone_trainable, split_param_groups
from ..utils import get_device, load_checkpoint, save_checkpoint, set_seed
from .metrics import compute_metrics, plot_confusion_matrix, plot_history, plot_roc_curve


@dataclass
class TrainConfig:
    model_name: str = "efficientnet_b0"
    splits_dir: str = str(SPLITS_DIR)
    output_root: str = str(OUTPUTS_DIR / "runs")
    img_size: int | None = None  # None -> DEFAULT_IMG_SIZE[model_name]
    batch_size: int = 32
    epochs: int = 15
    freeze_epochs: int = 3  # stage 1: train only the new head; stage 2: fine-tune everything
    lr: float = 1e-3
    backbone_lr_mult: float = 0.1  # backbone LR = lr * mult during fine-tuning
    weight_decay: float = 1e-4
    label_smoothing: float = 0.05
    patience: int = 5
    num_workers: int = 2
    seed: int = 42
    pretrained: bool = True
    class_weights: bool = False
    augment: bool = True
    amp: bool = True
    device: str | None = None
    init_from: str | None = None  # continue from an existing checkpoint of the same architecture
    out_name: str | None = None   # file name in models/, default "<model>_best.pt"
    run_name: str = field(default_factory=lambda: datetime.now().strftime("%Y%m%d-%H%M%S"))

    def __post_init__(self) -> None:
        if self.img_size is None:
            self.img_size = DEFAULT_IMG_SIZE.get(self.model_name, 224)
        if self.model_name == "custom_cnn":
            self.freeze_epochs = 0  # nothing pretrained to freeze
            self.backbone_lr_mult = 1.0
        if not self.pretrained:
            self.freeze_epochs = 0


def _forward_step(model, x, y, criterion, device, use_amp):
    x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
    with torch.autocast(device_type=device.type, enabled=use_amp):
        logits = model(x)
        loss = criterion(logits, y)
    return logits, loss, y


def train_one_epoch(model, loader, criterion, optimizer, scaler, device, use_amp) -> tuple[float, float]:
    model.train()
    total_loss, correct, seen = 0.0, 0, 0
    for x, y in tqdm(loader, desc="train", leave=False):
        optimizer.zero_grad(set_to_none=True)
        logits, loss, y = _forward_step(model, x, y, criterion, device, use_amp)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        total_loss += loss.item() * y.size(0)
        correct += (logits.argmax(1) == y).sum().item()
        seen += y.size(0)
    return total_loss / max(seen, 1), correct / max(seen, 1)


@torch.no_grad()
def predict_loader(model, loader, device, criterion=None, use_amp=False) -> tuple[float, np.ndarray, np.ndarray]:
    """Return ``(mean_loss, y_true, prob_fake)`` over a loader."""
    model.eval()
    total_loss, seen, ys, probs = 0.0, 0, [], []
    for x, y in tqdm(loader, desc="eval", leave=False):
        logits, loss, y = _forward_step(model, x, y, criterion or nn.CrossEntropyLoss(), device, use_amp)
        total_loss += loss.item() * y.size(0)
        seen += y.size(0)
        ys.append(y.cpu().numpy())
        probs.append(torch.softmax(logits.float(), dim=1)[:, 1].cpu().numpy())
    return total_loss / max(seen, 1), np.concatenate(ys), np.concatenate(probs)


def evaluate(model, loader, device, out_dir: str | Path | None = None, prefix: str = "test") -> dict:
    """Compute metrics on a loader and optionally save JSON, report and plots to ``out_dir``."""
    loss, y_true, prob_fake = predict_loader(model, loader, device)
    metrics = compute_metrics(y_true, prob_fake)
    metrics["loss"] = loss
    if out_dir is not None:
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / f"{prefix}_metrics.json").write_text(
            json.dumps({k: v for k, v in metrics.items() if k != "report"}, indent=2)
        )
        (out_dir / f"{prefix}_classification_report.txt").write_text(metrics["report"])
        plot_confusion_matrix(metrics["confusion_matrix"], out_dir / f"{prefix}_confusion_matrix.png")
        plot_roc_curve(y_true, prob_fake, out_dir / f"{prefix}_roc_curve.png")
    return metrics


def fit(cfg: TrainConfig) -> Path:
    """Train a model and return the path to the best checkpoint (also copied into ``models/``)."""
    set_seed(cfg.seed)
    device = get_device(cfg.device)
    use_amp = cfg.amp and device.type == "cuda"
    run_dir = Path(cfg.output_root) / f"{cfg.model_name}-{cfg.run_name}"
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "config.json").write_text(json.dumps(asdict(cfg), indent=2))
    print(f"[EYE FOR AI] device={device} amp={use_amp} run_dir={run_dir}")

    loaders = build_dataloaders(cfg.splits_dir, cfg.img_size, cfg.batch_size, cfg.num_workers, cfg.augment)
    train_ds = loaders["train"].dataset
    print(f"[data] train={len(train_ds)} val={len(loaders['val'].dataset)} "
          f"test={len(loaders['test'].dataset) if 'test' in loaders else 0}")

    model = build_model(cfg.model_name, num_classes=len(CLASS_NAMES), pretrained=cfg.pretrained and not cfg.init_from)
    if cfg.init_from:
        init = load_checkpoint(cfg.init_from)
        if init["model_name"] != cfg.model_name:
            raise ValueError(f"--init-from is a {init['model_name']} checkpoint, not {cfg.model_name}")
        model.load_state_dict(init["state_dict"])
        print(f"[init] weights loaded from {cfg.init_from}")
    model = model.to(device)
    backbone_params, head_params = split_param_groups(model, cfg.model_name)
    optimizer = torch.optim.AdamW(
        [
            {"params": backbone_params, "lr": cfg.lr * cfg.backbone_lr_mult},
            {"params": head_params, "lr": cfg.lr},
        ],
        weight_decay=cfg.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.epochs, eta_min=cfg.lr * 1e-3)
    weights = train_ds.class_weights().to(device) if cfg.class_weights else None
    criterion = nn.CrossEntropyLoss(weight=weights, label_smoothing=cfg.label_smoothing)
    scaler = torch.amp.GradScaler(device.type, enabled=use_amp)

    if cfg.freeze_epochs > 0:
        set_backbone_trainable(model, cfg.model_name, False)
        print(f"[stage 1] backbone frozen for {cfg.freeze_epochs} epoch(s) - training classification head only")

    history = {k: [] for k in ("train_loss", "train_acc", "val_loss", "val_acc", "val_f1", "lr")}
    best_f1, best_epoch, bad_epochs = -1.0, 0, 0
    best_path = run_dir / "best_model.pt"

    for epoch in range(1, cfg.epochs + 1):
        if cfg.freeze_epochs > 0 and epoch == cfg.freeze_epochs + 1:
            set_backbone_trainable(model, cfg.model_name, True)
            print("[stage 2] backbone unfrozen - fine-tuning the whole network")

        t0 = time.time()
        train_loss, train_acc = train_one_epoch(model, loaders["train"], criterion, optimizer, scaler, device, use_amp)
        val_loss, y_val, p_val = predict_loader(model, loaders["val"], device, criterion, use_amp)
        val_m = compute_metrics(y_val, p_val)
        scheduler.step()

        for k, v in zip(history, (train_loss, train_acc, val_loss, val_m["accuracy"], val_m["f1"],
                                  optimizer.param_groups[1]["lr"])):
            history[k].append(v)
        print(f"Epoch {epoch:02d}/{cfg.epochs} | train loss {train_loss:.4f} acc {train_acc:.4f} | "
              f"val loss {val_loss:.4f} acc {val_m['accuracy']:.4f} f1 {val_m['f1']:.4f} "
              f"auc {val_m['roc_auc']:.4f} | {time.time() - t0:.0f}s")

        if val_m["f1"] > best_f1:
            best_f1, best_epoch, bad_epochs = val_m["f1"], epoch, 0
            save_checkpoint(
                best_path,
                model_name=cfg.model_name,
                num_classes=len(CLASS_NAMES),
                class_names=list(CLASS_NAMES),
                img_size=cfg.img_size,
                mean=list(IMAGENET_MEAN),
                std=list(IMAGENET_STD),
                epoch=epoch,
                val_metrics={k: v for k, v in val_m.items() if k != "report"},
                state_dict=model.state_dict(),
                created_at=datetime.now().isoformat(timespec="seconds"),
            )
            print(f"  -> saved new best model (val F1 {best_f1:.4f})")
        else:
            bad_epochs += 1
            if bad_epochs >= cfg.patience and epoch > cfg.freeze_epochs:
                print(f"[early stopping] no improvement for {cfg.patience} epochs")
                break

    (run_dir / "history.json").write_text(json.dumps(history, indent=2))
    plot_history(history, run_dir / "training_curves.png")

    # Final evaluation of the best checkpoint.
    model.load_state_dict(load_checkpoint(best_path, device)["state_dict"])
    eval_split = "test" if "test" in loaders else "val"
    metrics = evaluate(model, loaders[eval_split], device, run_dir, prefix=eval_split)
    print(f"\n[best epoch {best_epoch}] {eval_split} results:\n{metrics['report']}")
    print(f"ROC-AUC: {metrics['roc_auc']:.4f}")

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    final_path = MODELS_DIR / (cfg.out_name or f"{cfg.model_name}_best.pt")
    shutil.copy2(best_path, final_path)
    print(f"[done] artifacts in {run_dir}\n[done] deployable checkpoint: {final_path}")
    return final_path
