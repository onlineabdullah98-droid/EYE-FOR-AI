"""Download public fake-image datasets from Kaggle.

Setup (one time):
  1. Create a Kaggle account -> Settings -> API -> "Create New Token" (downloads kaggle.json)
  2. Put it at ~/.kaggle/kaggle.json   (Windows: C:\\Users\\<you>\\.kaggle\\kaggle.json)
  3. pip install kaggle

Usage:
  python scripts/download_data.py --dataset cifake
  python scripts/download_data.py --dataset faces140k
  python scripts/download_data.py --dataset casia2
"""

from __future__ import annotations

import argparse
import subprocess
import sys

import _bootstrap  # noqa: F401

from eyeforai.config import RAW_DATA_DIR

DATASETS = {
    # 60k real (CIFAR-10) + 60k Stable-Diffusion fakes, 32x32. Best starting point: AI-generated detection.
    "cifake": "birdy654/cifake-real-and-ai-generated-synthetic-images",
    # 70k real (FFHQ) + 70k StyleGAN faces, 256x256. Deepfake / GAN face detection.
    "faces140k": "xhlulu/140k-real-and-fake-faces",
    # CASIA v2.0: 7.4k authentic (Au) + 5.1k tampered (Tp: splicing / copy-move). Manipulation detection.
    "casia2": "divg07/casia-20-image-tamperingdetection-dataset",
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", choices=DATASETS, required=True)
    args = parser.parse_args()

    target = RAW_DATA_DIR / args.dataset
    target.mkdir(parents=True, exist_ok=True)
    cmd = ["kaggle", "datasets", "download", "-d", DATASETS[args.dataset], "-p", str(target), "--unzip"]
    print("Running:", " ".join(cmd))
    try:
        subprocess.run(cmd, check=True)
    except FileNotFoundError:
        sys.exit("The 'kaggle' CLI is not installed. Run: pip install kaggle")
    except subprocess.CalledProcessError as e:
        sys.exit(f"Download failed ({e}). Check ~/.kaggle/kaggle.json, or download manually from "
                 f"https://www.kaggle.com/datasets/{DATASETS[args.dataset]} and unzip into {target}")
    print(f"\nDone. Next step:\n  python scripts/prepare_dataset.py --source {target}")


if __name__ == "__main__":
    main()
