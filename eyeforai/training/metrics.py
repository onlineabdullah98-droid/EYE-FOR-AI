"""Evaluation metrics (Accuracy, Precision, Recall, F1, ROC-AUC) and plots."""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.ticker import MaxNLocator  # noqa: E402
from sklearn.metrics import (  # noqa: E402
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

from ..config import CLASS_NAMES  # noqa: E402


def compute_metrics(y_true: Sequence[int], prob_fake: Sequence[float], threshold: float = 0.5) -> dict:
    """Metrics with FAKE (label 1) as the positive class."""
    y_true = np.asarray(y_true, dtype=int)
    prob_fake = np.asarray(prob_fake, dtype=float)
    y_pred = (prob_fake >= threshold).astype(int)
    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=[0, 1]).tolist(),
        "report": classification_report(
            y_true, y_pred, labels=[0, 1], target_names=list(CLASS_NAMES), digits=4, zero_division=0
        ),
    }
    metrics["roc_auc"] = float(roc_auc_score(y_true, prob_fake)) if len(np.unique(y_true)) == 2 else float("nan")
    return metrics


def plot_history(history: dict[str, list[float]], out_path: str | Path) -> Path:
    """Side-by-side loss and accuracy curves for train vs validation."""
    epochs = range(1, len(history["train_loss"]) + 1)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))
    ax1.plot(epochs, history["train_loss"], "o-", label="Train")
    ax1.plot(epochs, history["val_loss"], "o-", label="Validation")
    ax1.set(title="Loss", xlabel="Epoch", ylabel="Cross-entropy loss")
    ax2.plot(epochs, history["train_acc"], "o-", label="Train")
    ax2.plot(epochs, history["val_acc"], "o-", label="Validation")
    ax2.set(title="Accuracy", xlabel="Epoch", ylabel="Accuracy", ylim=(0, 1.02))
    for ax in (ax1, ax2):
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax.grid(alpha=0.3)
        ax.legend()
    fig.tight_layout()
    return _save(fig, out_path)


def plot_confusion_matrix(cm: Sequence[Sequence[int]], out_path: str | Path, normalize: bool = False) -> Path:
    cm = np.asarray(cm, dtype=float)
    if normalize:
        cm = cm / np.clip(cm.sum(axis=1, keepdims=True), 1, None)
    fig, ax = plt.subplots(figsize=(5, 4.5))
    im = ax.imshow(cm, cmap="Blues")
    fig.colorbar(im, ax=ax, fraction=0.046)
    ax.set(
        xticks=[0, 1], yticks=[0, 1], xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES,
        xlabel="Predicted", ylabel="Actual", title="Confusion Matrix",
    )
    for i in range(2):
        for j in range(2):
            text = f"{cm[i, j]:.2%}" if normalize else f"{int(cm[i, j])}"
            ax.text(j, i, text, ha="center", va="center", color="white" if cm[i, j] > cm.max() / 2 else "black")
    fig.tight_layout()
    return _save(fig, out_path)


def plot_roc_curve(y_true: Sequence[int], prob_fake: Sequence[float], out_path: str | Path) -> Path | None:
    if len(np.unique(y_true)) < 2:
        return None
    fpr, tpr, _ = roc_curve(y_true, prob_fake)
    auc = roc_auc_score(y_true, prob_fake)
    fig, ax = plt.subplots(figsize=(5, 4.5))
    ax.plot(fpr, tpr, lw=2, label=f"AUC = {auc:.4f}")
    ax.plot([0, 1], [0, 1], "--", color="gray")
    ax.set(title="ROC Curve (FAKE = positive)", xlabel="False positive rate", ylabel="True positive rate")
    ax.grid(alpha=0.3)
    ax.legend(loc="lower right")
    fig.tight_layout()
    return _save(fig, out_path)


def _save(fig, out_path: str | Path) -> Path:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path
