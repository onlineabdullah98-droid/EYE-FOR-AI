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
  python scripts/download_data.py --dataset genimage --splits train --max-shards 4
  python scripts/download_data.py --dataset openfake --splits test --max-shards 1 --max-per-class 3000
  python scripts/download_data.py --dataset coco --max-per-class 5000        # real everyday photos only
  python scripts/download_data.py --dataset unsplash --max-per-class 4000    # real camera/phone photos only
  python scripts/download_data.py --dataset wikiart --max-shards 4          # real paintings / artworks only
  python scripts/download_data.py --dataset diffusiondb --max-shards 3      # Stable Diffusion art only
  python scripts/download_data.py --dataset wider                          # real event / press photos only
  python scripts/download_data.py --dataset fashionpedia --max-shards 1     # real fashion photos only
  python scripts/download_data.py --dataset personsd --max-shards 5         # AI images of people only
  python scripts/download_data.py --dataset casia2 --source kaggle
"""

from __future__ import annotations

import argparse
import json
import random
import shutil
import subprocess
import sys
import time
import urllib.request
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

import _bootstrap  # noqa: F401

from eyeforai.config import RAW_DATA_DIR


@dataclass(frozen=True)
class HFDataset:
    repo: str
    data_dir: str = "data"
    # Re-encode both classes with the same random JPEG/PNG policy. Needed when real and fake images come in
    # different file formats (GenImage: every real is JPEG, every fake is PNG), otherwise the model learns
    # "PNG = fake" instead of learning what AI images look like.
    normalize_format: bool = False


HF_DATASETS = {
    # 60k real (CIFAR-10) + 60k Stable Diffusion fakes, 32x32. Best starting point for AI-generated detection.
    "cifake": HFDataset("dragonintelligence/CIFAKE-image-dataset"),
    # 70k real (FFHQ) + 70k StyleGAN faces, 256x256. GAN / deepfake face detection.
    "faces140k": HFDataset("TheKernel01/140k-Real-and-Fake-Faces"),
    # ImageNet photos vs Midjourney, SD 1.4/1.5, ADM, GLIDE, BigGAN, VQDM, Wukong (35k images, ~8 GB).
    "genimage": HFDataset("TheKernel01/Tiny-GenImage", normalize_format=True),
    # LAION photos vs modern generators: Flux, SD 3.5, Midjourney v6, DALL-E 3, GPT-Image, Imagen... (~5 GB/shard).
    "openfake": HFDataset("ComplexDataLab/OpenFake", data_dir="core", normalize_format=True),
}

REAL_NAMES = ("real", "human", "authentic")
FAKE_NAMES = ("fake", "ai", "gen", "synthetic")

KAGGLE_DATASETS = {
    "cifake": "birdy654/cifake-real-and-ai-generated-synthetic-images",
    "faces140k": "xhlulu/140k-real-and-fake-faces",
    # CASIA v2.0: 7.4k authentic (Au) + 5.1k tampered (Tp: splicing / copy-move). Manipulation detection.
    "casia2": "divg07/casia-20-image-tamperingdetection-dataset",
}

HF_BASE = "https://huggingface.co"

# Real-photo-only sources. They balance the AI-heavy datasets with everyday camera and phone photos, which a
# detector trained mostly on ImageNet photos otherwise tends to call AI.
COCO_ZIP = "http://images.cocodataset.org/zips/val2017.zip"  # 5,000 Flickr photos, ~800 MB
UNSPLASH_REPO = "1aurent/unsplash-lite"                      # 25,000 photos with camera EXIF (metadata + URLs)
UNSPLASH_BEFORE = "2021-07"  # only photos submitted before AI image generators became widespread
WIKIART_REPO, WIKIART_SHARDS = "huggan/wikiart", 72        # 81k artworks by 129 artists, ~520 MB per shard
DIFFUSIONDB_REPO, DIFFUSIONDB_PARTS = "poloclub/diffusiondb", 2000  # 1,000 Stable Diffusion images per part
EXTRA_DATASETS = ("coco", "unsplash", "wikiart", "diffusiondb", "wider", "fashionpedia", "personsd")  # single-class sources with their own downloaders
WIDER_REPO = "CUHK-CSE/wider_face"  # 32k event photos in 61 categories; only people/event scenes are kept
WIDER_EVENTS = ("Press_Conference", "Award_Ceremony", "Interview", "Meeting", "Election_Campain", "Dresses",
                "Greeting", "Handshaking", "Couple", "Family_Group", "Group", "Ceremony", "Celebration_Or_Party",
                "Photographers", "Festival", "Shoppers", "Waiter_Waitress", "Students_Schoolkids", "Voter",
                "Parade", "People_Marching", "Concerts", "Cheering", "Dancing")
FASHIONPEDIA_REPO = "detection-datasets/fashionpedia"  # 46k real photos of dressed people, ~480 MB per shard
PERSONSD_REPO = "LGirrbach/person-centric-images-stable-diffusion-v1-4"  # 300+ shards of ~1,200 AI people
PHONE_MAKES = ("APPLE", "SAMSUNG", "GOOGLE", "HUAWEI", "XIAOMI", "ONEPLUS", "OPPO", "VIVO", "MOTOROLA", "LG", "NOKIA")


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


def _download(url: str, dest: Path, retries: int = 4) -> None:
    """Download with resume: large shards (5 GB) often get cut off mid-transfer."""
    print(f"  downloading {url.rsplit('/', 1)[-1]} ...", flush=True)
    tmp = dest.with_suffix(".part")
    for attempt in range(retries + 1):
        done = tmp.stat().st_size if tmp.exists() else 0
        req = urllib.request.Request(url, headers={"Range": f"bytes={done}-"} if done else {})
        try:
            with urllib.request.urlopen(req) as resp:
                if done and resp.status != 206:  # server ignored the range: start over
                    done = 0
                total = done + int(resp.headers.get("Content-Length", 0))
                with tmp.open("ab" if done else "wb") as f:
                    shutil.copyfileobj(resp, f, length=1 << 20)
            if tmp.stat().st_size >= total:
                tmp.rename(dest)
                return
        except OSError as e:
            print(f"    interrupted ({e})", flush=True)
        if attempt < retries:
            time.sleep(2 ** (attempt + 1))
            print(f"    resuming (attempt {attempt + 2})...", flush=True)
    raise RuntimeError(f"Download of {url} did not complete after {retries + 1} attempts")


def canonical_label(name: str) -> str:
    """Map a dataset's class name ('real', 'RealArt', 'ai_gen', 'FAKE', ...) to REAL or FAKE."""
    n = name.lower()
    if any(k in n for k in REAL_NAMES):
        return "REAL"
    if any(k in n for k in FAKE_NAMES):
        return "FAKE"
    raise ValueError(f"Cannot tell whether class '{name}' is real or fake")


def normalize_image(data: bytes, rng: random.Random, max_side: int = 512) -> tuple[bytes, str] | None:
    """Downscale to ``max_side`` and re-encode: 80% JPEG (quality 70-98), 20% PNG, the same for both classes."""
    img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        return None
    h, w = img.shape[:2]
    if max(h, w) > max_side:
        scale = max_side / max(h, w)
        img = cv2.resize(img, (round(w * scale), round(h * scale)), interpolation=cv2.INTER_AREA)
    if rng.random() < 0.8:
        ok, enc = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, rng.randint(70, 98)])
        ext = ".jpg"
    else:
        ok, enc = cv2.imencode(".png", img)
        ext = ".png"
    return (enc.tobytes(), ext) if ok else None


def _extract_parquet(parquet_path: Path, split: str, out_root: Path, counts: Counter, max_per_class: int | None,
                     normalize: bool = False, seed: int = 0) -> None:
    import pyarrow.parquet as pq

    rng = random.Random(f"{seed}-{parquet_path.name}")
    pf = pq.ParquetFile(parquet_path)
    features = json.loads(pf.schema_arrow.metadata[b"huggingface"])["info"]["features"]
    # Map labels by NAME: datasets disagree on whether 0 means REAL or FAKE, and some store strings.
    label_names = features["label"].get("names")
    gen_col = next((c for c in ("generator", "model") if c in features), None)
    gen_names = features[gen_col].get("names") if gen_col else None
    columns = ["image", "label"] + ([gen_col] if gen_col else [])

    for batch in pf.iter_batches(batch_size=200, columns=columns):
        gens = batch.column(gen_col).to_pylist() if gen_col else [None] * batch.num_rows
        for img, label, gen in zip(batch.column("image").to_pylist(), batch.column("label").to_pylist(), gens):
            name = canonical_label(label_names[label] if label_names else str(label))
            if max_per_class and counts[(split, name)] >= max_per_class:
                continue
            data, ext = img["bytes"], _image_ext(img["bytes"], img.get("path"))
            if normalize:
                converted = normalize_image(data, rng)
                if converted is None:
                    continue
                data, ext = converted
            source = gen_names[gen] if (gen_names and gen is not None) else gen
            # Keep the generator in the file name so results can be broken down per generator later.
            prefix = f"{str(source).replace('/', '-').replace(' ', '-')}_" if source else ""
            out_dir = out_root / split / name
            out_dir.mkdir(parents=True, exist_ok=True)
            (out_dir / f"{prefix}{counts[(split, name)]:06d}{ext}").write_bytes(data)
            counts[(split, name)] += 1


def download_hf(dataset: str, target: Path, max_shards: int | None, max_per_class: int | None,
                splits: list[str] | None = None, keep_parquet: bool = False) -> None:
    spec = HF_DATASETS[dataset]
    with urllib.request.urlopen(f"{HF_BASE}/api/datasets/{spec.repo}/tree/main/{spec.data_dir}") as resp:
        files = sorted(f["path"] for f in json.load(resp) if f["path"].endswith(".parquet"))

    by_split: dict[str, list[str]] = {}
    for f in files:
        by_split.setdefault(Path(f).name.split("-")[0], []).append(f)
    if splits:
        by_split = {k: v for k, v in by_split.items() if k in splits}

    cache = target / "_parquet"
    cache.mkdir(parents=True, exist_ok=True)
    counts: Counter = Counter()
    for split, shards in by_split.items():
        for shard in shards[:max_shards] if max_shards else shards:
            local = cache / Path(shard).name
            if not local.exists():
                _download(f"{HF_BASE}/datasets/{spec.repo}/resolve/main/{shard}", local)
            _extract_parquet(local, split, target, counts, max_per_class, spec.normalize_format)
    if not keep_parquet:
        shutil.rmtree(cache)
    for (split, name), n in sorted(counts.items()):
        print(f"  {split:10s} {name:5s} {n:6d} images")


def _write_real(out_dir: Path, name: str, data: bytes, rng: random.Random) -> bool:
    converted = normalize_image(data, rng)
    if converted is None:
        return False
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f"{name}{converted[1]}").write_bytes(converted[0])
    return True


def download_coco(target: Path, max_images: int | None) -> None:
    import zipfile

    cache = target / "_cache"
    cache.mkdir(parents=True, exist_ok=True)
    zip_path = cache / "val2017.zip"
    if not zip_path.exists():
        _download(COCO_ZIP, zip_path)
    rng, n = random.Random(0), 0
    with zipfile.ZipFile(zip_path) as zf:
        names = sorted(f for f in zf.namelist() if f.endswith(".jpg"))
        for f in names[:max_images] if max_images else names:
            n += _write_real(target / "REAL", f"coco_{Path(f).stem}", zf.read(f), rng)
    shutil.rmtree(cache)
    print(f"  REAL {n:6d} images")


def download_unsplash(target: Path, max_images: int | None, workers: int = 16) -> None:
    """Fetch camera photos (EXIF make present, submitted before 2021-07), every phone photo first."""
    from concurrent.futures import ThreadPoolExecutor

    import pyarrow.parquet as pq

    cache = target / "_cache"
    cache.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(f"{HF_BASE}/api/datasets/{UNSPLASH_REPO}/tree/main/data") as resp:
        shards = sorted(f["path"] for f in json.load(resp) if f["path"].endswith(".parquet"))
    photos = []
    for shard in shards:
        local = cache / Path(shard).name
        if not local.exists():
            _download(f"{HF_BASE}/datasets/{UNSPLASH_REPO}/resolve/main/{shard}", local)
        for row in pq.read_table(local, columns=["photo", "exif"]).to_pylist():
            make = (row["exif"]["camera_make"] or "").split(" ")[0].upper()
            if make and make != "NAN" and (row["photo"]["submitted_at"] or "9999") < UNSPLASH_BEFORE:
                photos.append((row["photo"]["id"], row["photo"]["image_url"], make))

    rng = random.Random(0)
    rng.shuffle(photos)
    photos.sort(key=lambda p: p[2] not in PHONE_MAKES)  # stable: phones first, the rest stays shuffled
    photos = photos[:max_images] if max_images else photos

    def fetch(item: tuple[str, str, str]) -> bool:
        pid, url, make = item
        try:
            with urllib.request.urlopen(f"{url}?w=768&fm=jpg&q=90", timeout=60) as resp:
                data = resp.read()
        except OSError:
            return False
        return _write_real(target / "REAL", f"unsplash-{make.lower()}_{pid}", data, random.Random(pid))

    with ThreadPoolExecutor(workers) as pool:
        n = sum(pool.map(fetch, photos))
    shutil.rmtree(cache)
    print(f"  REAL {n:6d} images ({sum(p[2] in PHONE_MAKES for p in photos)} from phones)")


def _spread(n_wanted: int | None, total: int, default: int) -> list[int]:
    """Evenly spaced shard indices, so a few shards still cover many artists / time periods."""
    n = min(n_wanted or default, total)
    return sorted({round(i * total / n) for i in range(n)})


def download_wikiart(target: Path, n_shards: int | None, max_images: int | None) -> None:
    """Real paintings and artworks, so that "looks like a painting" is not learned as "AI"."""
    import pyarrow.parquet as pq

    cache = target / "_cache"
    cache.mkdir(parents=True, exist_ok=True)
    rng, n = random.Random(0), 0
    for i in _spread(n_shards, WIKIART_SHARDS, 4):
        name = f"train-{i:05d}-of-{WIKIART_SHARDS:05d}.parquet"
        local = cache / name
        if not local.exists():
            _download(f"{HF_BASE}/datasets/{WIKIART_REPO}/resolve/main/data/{name}", local)
        pf = pq.ParquetFile(local)
        meta = pf.schema_arrow.metadata or {}
        styles = (json.loads(meta[b"huggingface"])["info"]["features"]["style"].get("names")
                  if b"huggingface" in meta else None)
        for batch in pf.iter_batches(batch_size=100, columns=["image", "style"]):
            for img, style in zip(batch.column("image").to_pylist(), batch.column("style").to_pylist()):
                if max_images and n >= max_images:
                    break
                tag = styles[style].lower() if styles else f"style{style}"
                n += _write_real(target / "REAL", f"wikiart-{tag}_{n:06d}", img["bytes"], rng)
    shutil.rmtree(cache)
    print(f"  REAL {n:6d} artworks")


def download_diffusiondb(target: Path, n_parts: int | None, max_images: int | None) -> None:
    """AI-generated art (Stable Diffusion 1.x, 2022), the counterpart of the WikiArt paintings."""
    import zipfile

    cache = target / "_cache"
    cache.mkdir(parents=True, exist_ok=True)
    rng, n = random.Random(0), 0
    for i in _spread(n_parts, DIFFUSIONDB_PARTS, 3):
        name = f"part-{max(i, 1):06d}.zip"
        local = cache / name
        if not local.exists():
            _download(f"{HF_BASE}/datasets/{DIFFUSIONDB_REPO}/resolve/main/images/{name}", local)
        with zipfile.ZipFile(local) as zf:
            for f in sorted(x for x in zf.namelist() if x.endswith((".png", ".webp", ".jpg"))):
                if max_images and n >= max_images:
                    break
                converted = normalize_image(zf.read(f), rng)
                if converted:
                    (target / "FAKE").mkdir(parents=True, exist_ok=True)
                    (target / "FAKE" / f"diffusiondb_{Path(f).stem}{converted[1]}").write_bytes(converted[0])
                    n += 1
    shutil.rmtree(cache)
    print(f"  FAKE {n:6d} images")


def _hf_parquet_files(repo: str) -> list[str]:
    with urllib.request.urlopen(f"{HF_BASE}/api/datasets/{repo}/tree/main/data") as resp:
        return sorted(f["path"] for f in json.load(resp) if f["path"].endswith(".parquet"))


def _single_class_parquet(repo: str, target: Path, label: str, prefix: str, n_shards: int | None, default: int,
                          max_images: int | None) -> None:
    """Images from a one-class parquet dataset, from evenly spread shards, normalised like the other sources."""
    import pyarrow.parquet as pq

    files = _hf_parquet_files(repo)
    cache = target / "_cache"
    cache.mkdir(parents=True, exist_ok=True)
    rng, n = random.Random(0), 0
    for i in _spread(n_shards, len(files), default):
        local = cache / Path(files[i]).name
        if not local.exists():
            _download(f"{HF_BASE}/datasets/{repo}/resolve/main/{files[i]}", local)
        for batch in pq.ParquetFile(local).iter_batches(batch_size=100, columns=["image"]):
            for img in batch.column("image").to_pylist():
                if max_images and n >= max_images:
                    break
                converted = normalize_image(img["bytes"], rng)
                if converted:
                    (target / label).mkdir(parents=True, exist_ok=True)
                    (target / label / f"{prefix}_{n:06d}{converted[1]}").write_bytes(converted[0])
                    n += 1
    shutil.rmtree(cache)
    print(f"  {label} {n:6d} images")


def download_wider(target: Path, max_images: int | None) -> None:
    """Real event photos (press conferences, award ceremonies, interviews...) from WIDER FACE train + val."""
    import zipfile

    cache = target / "_cache"
    cache.mkdir(parents=True, exist_ok=True)
    rng, n = random.Random(0), 0
    for part in ("WIDER_train", "WIDER_val"):
        local = cache / f"{part}.zip"
        if not local.exists():
            _download(f"{HF_BASE}/datasets/{WIDER_REPO}/resolve/main/data/{part}.zip", local)
        with zipfile.ZipFile(local) as zf:
            for f in sorted(zf.namelist()):
                category = f.split("/")[-2].split("--", 1)[-1] if f.count("/") >= 2 else ""
                if not f.endswith(".jpg") or category not in WIDER_EVENTS:
                    continue
                if max_images and n >= max_images:
                    break
                n += _write_real(target / "REAL", f"wider-{category.lower()}_{n:06d}", zf.read(f), rng)
    shutil.rmtree(cache)
    print(f"  REAL {n:6d} images")


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
    parser.add_argument("--dataset", choices=sorted({*HF_DATASETS, *KAGGLE_DATASETS, *EXTRA_DATASETS}),
                        required=True)
    parser.add_argument("--source", choices=("huggingface", "kaggle"), default="huggingface")
    parser.add_argument("--max-shards", type=int, default=None, help="HuggingFace: parquet shards per split")
    parser.add_argument("--max-per-class", type=int, default=None, help="HuggingFace: images per class per split")
    parser.add_argument("--splits", nargs="+", default=None, help="HuggingFace: only these splits, e.g. train test")
    parser.add_argument("--keep-parquet", action="store_true", help="Keep downloaded parquet files")
    args = parser.parse_args()

    source = args.source
    if source == "huggingface" and args.dataset not in {*HF_DATASETS, *EXTRA_DATASETS}:
        print(f"'{args.dataset}' is only available on Kaggle - switching source.")
        source = "kaggle"

    target = RAW_DATA_DIR / args.dataset
    target.mkdir(parents=True, exist_ok=True)
    if args.dataset == "coco":
        download_coco(target, args.max_per_class)
    elif args.dataset == "unsplash":
        download_unsplash(target, args.max_per_class)
    elif args.dataset == "wikiart":
        download_wikiart(target, args.max_shards, args.max_per_class)
    elif args.dataset == "diffusiondb":
        download_diffusiondb(target, args.max_shards, args.max_per_class)
    elif args.dataset == "wider":
        download_wider(target, args.max_per_class)
    elif args.dataset == "fashionpedia":
        _single_class_parquet(FASHIONPEDIA_REPO, target, "REAL", "fashionpedia", args.max_shards, 1, args.max_per_class)
    elif args.dataset == "personsd":
        _single_class_parquet(PERSONSD_REPO, target, "FAKE", "personsd", args.max_shards, 5, args.max_per_class)
    elif source == "huggingface":
        download_hf(args.dataset, target, args.max_shards, args.max_per_class, args.splits, args.keep_parquet)
    else:
        download_kaggle(args.dataset, target)
    print(f"\nDone. Next step:\n  python scripts/prepare_dataset.py --source {target}")


if __name__ == "__main__":
    main()
