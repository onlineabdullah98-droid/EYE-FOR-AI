"""Download public fake-image datasets.

Two sources:

* HuggingFace (default, no account needed): downloads parquet shards and writes the images
  into data/raw/<dataset>/<split>/<REAL|FAKE>/ with their original encoding (no re-compression).
* Kaggle (``--source kaggle``): needs an API token.
    1. Kaggle -> Settings -> API -> "Create New Token" (downloads kaggle.json)
    2. Put it at ~/.kaggle/kaggle.json   (Windows: C:\\Users\\<you>\\.kaggle\\kaggle.json)
    3. pip install kaggle

Usage:
  python scripts/download_data.py --dataset cifake
  python scripts/download_data.py --dataset faces140k --max-shards 1 --max-per-class 3000
  python scripts/download_data.py --dataset casia2 --source kaggle
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import urllib.request
from collections import Counter
from pathlib import Path

import _bootstrap  # noqa: F401

from eyeforai.config import RAW_DATA_DIR

HF_DATASETS = {
    # 60k real (CIFAR-10) + 60k Stable Diffusion fakes, 32x32. Best starting point for AI-generated detection.
    "cifake": "dragonintelligence/CIFAKE-image-dataset",
    # 70k real (FFHQ) + 70k StyleGAN faces, 256x256. GAN / deepfake face detection.
    "faces140k": "TheKernel01/140k-Real-and-Fake-Faces",
}

KAGGLE_DATASETS = {
    "cifake": "birdy654/cifake-real-and-ai-generated-synthetic-images",
    "faces140k": "xhlulu/140k-real-and-fake-faces",
    # CASIA v2.0: 7.4k authentic (Au) + 5.1k tampered (Tp: splicing / copy-move). Manipulation detection.
    "casia2": "divg07/casia-20-image-tamperingdetection-dataset",
}

HF_BASE = "https://huggingface.co"


def _image_ext(data: bytes, path: str | None) -> str:
    if path and Path(path).suffix:
        return Path(path).suffix.lower()
    if data[:3] == b"\xff\xd8\xff":
        return ".jpg"
    if data[:8] == b"\x89PNG\r\n\x1a\n":
        return ".png"
    if data[:4] == b"RIFF" and data[8:12] == b"WEBP":
        return ".webp"
    return ".png"


def _download(url: str, dest: Path) -> None:
    print(f"  downloading {url.rsplit('/', 1)[-1]} ...", flush=True)
    tmp = dest.with_suffix(".part")
    with urllib.request.urlopen(url) as resp, tmp.open("wb") as f:
        shutil.copyfileobj(resp, f, length=1 << 20)
    tmp.rename(dest)


def _extract_parquet(parquet_path: Path, split: str, out_root: Path, counts: Counter, max_per_class: int | None) -> None:
    import pyarrow.parquet as pq

    pf = pq.ParquetFile(parquet_path)
    meta = json.loads(pf.schema_arrow.metadata[b"huggingface"])
    # Map label ids by NAME: datasets disagree on whether 0 means REAL or FAKE.
    names = [n.upper() for n in meta["info"]["features"]["label"]["names"]]
    for batch in pf.iter_batches(batch_size=1000, columns=["image", "label"]):
        for img, label in zip(batch.column("image").to_pylist(), batch.column("label").to_pylist()):
            name = names[label]
            if max_per_class and counts[(split, name)] >= max_per_class:
                continue
            data = img["bytes"]
            out_dir = out_root / split / name
            out_dir.mkdir(parents=True, exist_ok=True)
            (out_dir / f"{counts[(split, name)]:06d}{_image_ext(data, img.get('path'))}").write_bytes(data)
            counts[(split, name)] += 1


def download_hf(dataset: str, target: Path, max_shards: int | None, max_per_class: int | None) -> None:
    repo = HF_DATASETS[dataset]
    with urllib.request.urlopen(f"{HF_BASE}/api/datasets/{repo}/tree/main/data") as resp:
        files = sorted(f["path"] for f in json.load(resp) if f["path"].endswith(".parquet"))

    by_split: dict[str, list[str]] = {}
    for f in files:
        by_split.setdefault(Path(f).name.split("-")[0], []).append(f)

    cache = target / "_parquet"
    cache.mkdir(parents=True, exist_ok=True)
    counts: Counter = Counter()
    for split, shards in by_split.items():
        for shard in shards[:max_shards] if max_shards else shards:
            local = cache / Path(shard).name
            if not local.exists():
                _download(f"{HF_BASE}/datasets/{repo}/resolve/main/{shard}", local)
            _extract_parquet(local, split, target, counts, max_per_class)
    shutil.rmtree(cache)
    for (split, name), n in sorted(counts.items()):
        print(f"  {split:10s} {name:5s} {n:6d} images")


def download_kaggle(dataset: str, target: Path) -> None:
    cmd = ["kaggle", "datasets", "download", "-d", KAGGLE_DATASETS[dataset], "-p", str(target), "--unzip"]
    print("Running:", " ".join(cmd))
    try:
        subprocess.run(cmd, check=True)
    except FileNotFoundError:
        sys.exit("The 'kaggle' CLI is not installed. Run: pip install kaggle")
    except subprocess.CalledProcessError as e:
        sys.exit(f"Download failed ({e}). Check ~/.kaggle/kaggle.json, or download manually from "
                 f"https://www.kaggle.com/datasets/{KAGGLE_DATASETS[dataset]} and unzip into {target}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", choices=sorted(KAGGLE_DATASETS), required=True)
    parser.add_argument("--source", choices=("huggingface", "kaggle"), default="huggingface")
    parser.add_argument("--max-shards", type=int, default=None, help="HuggingFace: parquet shards per split")
    parser.add_argument("--max-per-class", type=int, default=None, help="HuggingFace: images per class per split")
    args = parser.parse_args()

    source = args.source
    if source == "huggingface" and args.dataset not in HF_DATASETS:
        print(f"'{args.dataset}' is only available on Kaggle - switching source.")
        source = "kaggle"

    target = RAW_DATA_DIR / args.dataset
    target.mkdir(parents=True, exist_ok=True)
    if source == "huggingface":
        download_hf(args.dataset, target, args.max_shards, args.max_per_class)
    else:
        download_kaggle(args.dataset, target)
    print(f"\nDone. Next step:\n  python scripts/prepare_dataset.py --source {target}")


if __name__ == "__main__":
    main()
