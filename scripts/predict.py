"""Run inference on one or more images and save a Grad-CAM + ELA report panel.

  python scripts/predict.py path/to/image.jpg --checkpoint models/efficientnet_b0_best.pt
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import _bootstrap  # noqa: F401
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from eyeforai.config import OUTPUTS_DIR  # noqa: E402
from eyeforai.inference import Predictor, find_checkpoints  # noqa: E402


def save_panel(result, out_path: Path) -> None:
    panels = [
        ("Original", result.image),
        ("Grad-CAM", result.gradcam_overlay),
        ("Suspicious regions", result.regions_overlay),
        ("Error Level Analysis", result.ela.heatmap),
    ]
    fig, axes = plt.subplots(1, 4, figsize=(18, 5))
    for ax, (title, img) in zip(axes, panels):
        ax.imshow(img)
        ax.set_title(title)
        ax.axis("off")
    color = "#d62828" if result.is_fake else "#2a9d8f"
    fig.suptitle(f"{result.label}  -  {result.confidence:.1%} confidence", fontsize=16, color=color, weight="bold")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=130)
    plt.close(fig)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("images", nargs="+")
    p.add_argument("--checkpoint", default=None, help="Defaults to the newest file in models/")
    p.add_argument("--threshold", type=float, default=0.5)
    p.add_argument("--out", default=str(OUTPUTS_DIR / "predictions"))
    a = p.parse_args()

    ckpt = a.checkpoint or next(iter(find_checkpoints()), None)
    if ckpt is None:
        raise SystemExit("No checkpoint found in models/. Train one with scripts/train.py first.")
    predictor = Predictor(ckpt, threshold=a.threshold)

    for img_path in map(Path, a.images):
        result = predictor.analyze(img_path)
        out_file = Path(a.out) / f"{img_path.parent.name}_{img_path.stem}_analysis.png"
        save_panel(result, out_file)
        print(json.dumps({"image": str(img_path), **result.summary(), "panel": str(out_file)}, indent=2))


if __name__ == "__main__":
    main()
