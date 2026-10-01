# 👁️ EYE FOR AI — AI System for Detecting Manipulated and Fake Images

Final Year Project (BSCS/BSAI/BSSE/BSCY). You upload an image and the system:

1. tells you whether it is a **real photo**, **AI-generated / manipulated**, or **uncertain**,
2. reports **how likely it is AI**, as a percentage with a meter (e.g. *96% AI*),
3. explains the decision with a **Grad-CAM heatmap**, **suspicious-region boxes**, and **Error Level Analysis (ELA)**,
4. stores every prediction in a **history database** (SQLite by default, PostgreSQL optional).

| Layer | Technology |
|---|---|
| Preprocessing & augmentation | OpenCV + torchvision |
| Baseline model | Custom 4-block CNN (trained from scratch) |
| Main model | Transfer learning: **EfficientNet-B0 / B2** or **ResNet50** (ImageNet weights) |
| Explainability | Grad-CAM, ELA |
| Web UI | Streamlit (`app.py`) |
| REST API | Flask (`api.py`) |
| Database | SQLAlchemy → SQLite / PostgreSQL |

---

## Quick start (one click)

After cloning the repo:

- **Windows:** double-click **`run_app.bat`** in the project folder.
- **macOS / Linux:** run `./run_app.sh` in a terminal.

The first run creates a virtual environment and installs the libraries, which takes 5–10 minutes. Every later run
starts the app straight away. The browser opens at **http://localhost:8501**. To stop the app, close the window or
press Ctrl+C.

---

## Trained models and results

The app uses **`models/efficientnet_b0_v3.pt`**. Older models are kept in `models/archive/` for comparison.

### v3: fewer real photos called AI (default model)

v2 caught modern AI images well, but on everyday photos it was wrong far too often. It called **36% of COCO
photos, 43% of Unsplash camera photos and 37% of phone photos "AI"**. Its training set had almost no ordinary
camera or phone photos. v3 continues from v2 with 9,000 more real photos, plus one more OpenFake shard so the
classes stay balanced:

- **COCO val2017:** 5,000 everyday Flickr photos.
- **Unsplash Lite:** 4,000 photos with camera EXIF, submitted before July 2021, including 839 taken with phones.

v3 also gives a **three-way verdict**. After training, two cut-offs are calibrated on validation data and stored
in the checkpoint:
- **AI** from an AI likelihood of 66%. At most about 5% of real photos score this high.
- **REAL** below 49%. At most about 10% of AI images score this low.
- **UNCERTAIN** in between.

You can move both cut-offs with the *Uncertain zone* slider in the app's sidebar.

**Clean test set:** 2,937 images that none of v1, v2 or v3 saw during training or model selection.

| Real photos wrongly called AI | v2 (cut-off 50%) | **v3** | v3 uncertain |
|---|---|---|---|
| COCO everyday photos (628) | 36.5% | **6.1%** | 4.5% |
| Unsplash camera photos (393) | 43.3% | **6.9%** | 3.6% |
| Unsplash phone photos (111) | 36.9% | **3.6%** | 6.3% |
| All real photos (1,729) | 28.3% | **5.8%** | 4.5% |

| AI images caught | v2 | **v3** | v3 wrongly called real |
|---|---|---|---|
| OpenFake, 2025–26 generators (907) | 94.6% | 88.0% | 9.2% |
| GenImage (120) | 63.3% | 60.0% | 30.0% |
| All AI images (1,208) | 92.1% | 86.7% | 9.9% |

v3 overall: accuracy 90.0% at a 50% cut-off, ROC-AUC 0.964. With the three-way verdict, 4.0% of images come out
UNCERTAIN, and the model is right on **92.2%** of the images it does decide.

The trade-off is deliberate. v3 catches about 5 points fewer AI images than v2, but wrongly accuses five times
fewer real photos. AI images caught by generator: Sora-2 100%, Flux.2 97%, Illustrious 96%, GPT-Image-1.5 95%,
Wan-2.5 95%, Z-Image 89%, Veo-3 78%, Midjourney-7 66%, Recraft-v3 65%. Older GenImage generators ADM (28%) and VQDM
(21%) remain weak. Full numbers are in `results/efficientnet_b0_v3/`.

| AI image (GPT-Image-1.5) | iPhone photo (v2 said 93% AI) | Uncertain case |
|---|---|---|
| ![ai](docs/screenshots/analyze_fake.png) | ![real](docs/screenshots/analyze_real.png) | ![uncertain](docs/screenshots/analyze_uncertain.png) |

To reproduce v3 (after the v2 downloads below):
```bash
python scripts/download_data.py --dataset coco                       # ~800 MB, 5,000 real photos
python scripts/download_data.py --dataset unsplash --max-per-class 4000
# a third OpenFake shard (test-00002) adds ~3,500 more AI images
python scripts/prepare_dataset.py --out data/splits_v3 --source data/raw/openfake@4000:10000 data/raw/coco@4000:0 \
    data/raw/unsplash@3500:0 data/raw/genimage@2500:3500 data/raw/cifake@1000:1250 data/raw/faces140k@1000:1250
python scripts/train.py --model efficientnet_b0 --splits-dir data/splits_v3 --init-from models/archive/efficientnet_b0_v2.pt \
    --img-size 160 --epochs 4 --freeze-epochs 0 --lr 3e-4 --out-name efficientnet_b0_v3.pt
```
For the published numbers, validation and test images that v1 or v2 had trained on were moved into the training
split (3,743 images), so the test set is clean.

### v2: modern AI generators

The first model (v1) only saw CIFAKE and StyleGAN faces. On images from current generators it said REAL almost
every time: only 12% of OpenFake AI images were caught. v2 continues training from v1 on four datasets.

| Data used for v2 | Images | Generators |
|---|---|---|
| OpenFake (2 shards) | 10,000 | GPT-Image 1.5/2, Midjourney 7, Flux.2, Nano Banana Pro, Sora-2, Veo-3, Seedream, Recraft, Ideogram, Z-Image... vs real ImageNet/DOCCI photos |
| Tiny-GenImage (4 shards) | 7,000 | Midjourney, SD 1.5, ADM, GLIDE, BigGAN, VQDM, Wukong vs real ImageNet photos |
| CIFAKE | 4,000 | Stable Diffusion 1.4 vs CIFAR-10 |
| 140k Faces | 4,000 | StyleGAN faces vs FFHQ |

That is 25,000 images, split 70/15/15 into 17,500 train, 3,750 validation and 3,750 test images. Training used
EfficientNet-B0 at 160 px for 6 epochs on a 4-core CPU.

**Test set results (3,750 images never used for training):**

| Test data | v1 accuracy | **v2 accuracy** | v1 ROC-AUC | **v2 ROC-AUC** |
|---|---|---|---|---|
| All | 62.3% | **88.9%** | 0.673 | **0.956** |
| OpenFake (modern generators) | 43.6% | **93.5%** | 0.380 | **0.977** |
| GenImage | 52.1% | **79.2%** | 0.587 | **0.897** |
| CIFAKE | 95.0% | 91.2% | 0.994 | 0.995 |
| 140k Faces | 94.9% | 91.8% | 0.985 | 0.989 |

v2 overall: precision 90.7%, recall 86.6%, F1 0.886.

Share of AI images caught, by generator (v1 → v2): GPT-Image-1.5 14% → 93%, Midjourney-7 24% → 84%,
Flux.2 11% → 94%, Veo-3 7% → 87%, Midjourney (GenImage) 19% → 76%, SD 1.5 33% → 83%. Real photos correctly
kept: 98% of ImageNet and 90% of DOCCI. Weak spots: ADM (21%), VQDM (52%) and Recraft-v3 (50%, only 12 test images).
The full table is in `results/efficientnet_b0_v2/per_dataset_and_generator.txt` and in the app's
*Model Performance* tab.

Notes for the report:
- v1 had been trained on part of CIFAKE and 140k Faces, so its v1 numbers on those two rows are optimistic.
  OpenFake and GenImage are clean comparisons, because neither model saw those test images.
- In GenImage every real image is a JPEG and every AI image is a PNG. Without a fix, a model learns "PNG = AI".
  `download_data.py` therefore re-encodes both classes the same way (max 512 px, random JPEG quality 70–98 or PNG).

To reproduce v2:
```bash
python scripts/download_data.py --dataset openfake --splits test --max-shards 2    # ~10 GB download
python scripts/download_data.py --dataset genimage --splits train --max-shards 4   # ~1.9 GB download
python scripts/prepare_dataset.py --out data/splits_v2 --source data/raw/openfake@5000 data/raw/genimage@3500 \
    data/raw/cifake@2000 data/raw/faces140k@2000
python scripts/train.py --model efficientnet_b0 --splits-dir data/splits_v2 --init-from models/archive/efficientnet_b0_v1.pt \
    --img-size 160 --epochs 6 --freeze-epochs 0 --lr 5e-4 --out-name efficientnet_b0_v2.pt
```

### v1: baseline comparison (CIFAKE + 140k Faces only)

| Model | Input | Test accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---|---|---|---|---|---|
| Custom CNN (from scratch) | 64 px | 81.9% | 80.6% | 84.1% | 0.823 | 0.908 |
| EfficientNet-B0 (transfer learning) | 128 px | 91.3% | 88.4% | 95.0% | 0.916 | 0.976 |

These were trained on 21,000 CIFAKE and 140k Faces images (14,700 train / 3,150 val / 3,150 test). The comparison
shows that transfer learning beats a CNN trained from scratch by about 9 points on the same data. Plots and logs are
in `results/custom_cnn_v1/` and `results/efficientnet_b0_v1/`.

| **Grad-CAM** (v1 model, StyleGAN face) | **Error Level Analysis** |
|---|---|
| ![gradcam](docs/screenshots/gradcam.png) | ![ela](docs/screenshots/ela.png) |

![Model performance tab](docs/screenshots/model_performance.png)

---

## 1. Project structure

```
EYE-FOR-AI/
├── app.py                       # Streamlit web app (drag & drop, verdict, heatmaps, history)
├── api.py                       # Flask REST API  (POST /api/predict)
├── run_app.bat / run_app.sh     # one-click setup + launch (Windows / macOS-Linux)
├── requirements.txt
├── eyeforai/                    # core Python package
│   ├── config.py                # paths, labels (0=REAL, 1=FAKE), defaults, DB URL
│   ├── utils.py                 # seeding, device selection, OpenCV image I/O, checkpoints
│   ├── data/
│   │   ├── transforms.py        # OpenCV resize/JPEG/blur + torchvision crop/flip/jitter/normalize
│   │   └── dataset.py           # dataset scanning, CSV split manifests, DataLoaders
│   ├── models/
│   │   ├── custom_cnn.py        # baseline CNN
│   │   └── factory.py           # ResNet50 / EfficientNet-B0/B2 + freezing + Grad-CAM layers
│   ├── training/
│   │   ├── trainer.py           # 2-stage fine-tuning, AMP, cosine LR, early stopping
│   │   └── metrics.py           # Accuracy, Precision, Recall, F1, ROC-AUC + plots
│   ├── explain/
│   │   ├── gradcam.py           # Grad-CAM, heatmap overlay, suspicious-region boxes
│   │   └── ela.py               # Error Level Analysis
│   ├── inference/predictor.py   # Predictor: classify + explain in one call
│   └── db/history.py            # prediction history (SQLite / PostgreSQL)
├── scripts/
│   ├── download_data.py         # CIFAKE, Faces, GenImage, OpenFake, COCO, Unsplash, CASIA (Kaggle)
│   ├── prepare_dataset.py       # stratified train/val/test CSV manifests
│   ├── train.py                 # train any model
│   ├── evaluate.py              # evaluate a saved checkpoint
│   └── predict.py               # CLI inference + saved analysis panel
├── tests/                       # pytest suite (runs on a synthetic mini-dataset)
├── results/                     # metrics, plots and logs of the included trained models
├── docs/screenshots/            # app screenshots
├── data/                        # datasets (git-ignored)
├── models/                      # efficientnet_b0_v3.pt (used by the app) + archive/ with v1/v2 models
└── outputs/                     # training runs, plots, predictions, history DB (git-ignored)
```

---

## 2. Setup

```bash
git clone <your-repo-url> EYE-FOR-AI && cd EYE-FOR-AI
python -m venv .venv
# Windows: .venv\Scripts\activate      Linux/macOS: source .venv/bin/activate

# Optional, for an NVIDIA GPU: install the CUDA build of PyTorch first, e.g.
# pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt

pytest          # quick check that everything works (about 20 s on CPU)
```

Python 3.10 or newer is required.

---

## 3. Datasets

| Dataset | Content | Detects | Download |
|---|---|---|---|
| **CIFAKE** (recommended start) | 60k real (CIFAR-10) + 60k Stable Diffusion images, 32×32 | AI-generated images | HuggingFace (no account) or Kaggle |
| **140k Real & Fake Faces** | 70k FFHQ + 70k StyleGAN faces, 256×256 | GAN / deepfake faces | HuggingFace (no account) or Kaggle |
| **OpenFake** | Real LAION/ImageNet/DOCCI photos vs 2025–26 generators (GPT-Image, MJ 7, Flux.2, Sora...) | Modern AI images | HuggingFace, ~5 GB per shard |
| **Tiny-GenImage** | ImageNet photos vs Midjourney, SD, ADM, GLIDE, BigGAN, VQDM, Wukong | AI images | HuggingFace, ~475 MB per shard |
| **CASIA v2.0** | 7.4k authentic (`Au`) + 5.1k tampered (`Tp`) | Splicing / copy-move edits | Kaggle only |

OpenFake (CC BY-NC 4.0) and Tiny-GenImage (CC BY-NC-SA 4.0) are licensed for non-commercial use, which covers an
academic project. For these two datasets, images are downscaled to at most 512 px and both classes are re-encoded
the same way. The generator's name is kept in each file name, for example `midjourney-7_000123.jpg`.

By default the images are downloaded from HuggingFace with no account needed, and labels are mapped by class
**name** (CIFAKE on HuggingFace uses `0 = FAKE`, the opposite of this project's convention).

```bash
python scripts/download_data.py --dataset cifake                                    # ~50 MB
python scripts/download_data.py --dataset faces140k --max-shards 1 --max-per-class 1500   # ~1 GB download, 9k images kept
python scripts/prepare_dataset.py --source data/raw/cifake data/raw/faces140k --max-per-class 6000
```
`--max-per-class` in `prepare_dataset.py` caps each source separately, so a small dataset is not drowned out
when you combine datasets.

**CASIA (Kaggle only):** go to kaggle.com → *Settings* → *API* → *Create New Token*, put `kaggle.json` in
`~/.kaggle/` (Windows: `C:\Users\<you>\.kaggle\`), then run
`python scripts/download_data.py --dataset casia2 --source kaggle`.

You can also download a dataset manually and unzip it into `data/raw/<name>`. Any folder layout works as long as
images sit under folders named `real`/`fake` or `Au`/`Tp` (case-insensitive). `prepare_dataset.py` writes
`data/splits/{train,val,test}.csv` with a 70/15/15 stratified split. Images are not copied.

---

## 4. Training (FYP-1)

```bash
# Baseline CNN trained from scratch
python scripts/train.py --model custom_cnn --epochs 20 --lr 1e-3

# Transfer learning (main model)
python scripts/train.py --model efficientnet_b0 --epochs 15 --freeze-epochs 3
python scripts/train.py --model resnet50        --epochs 15 --batch-size 16
python scripts/train.py --model efficientnet_b2 --epochs 15 --class-weights   # imbalanced data such as CASIA
```

How transfer learning is done:
- **Stage 1** (`--freeze-epochs`, default 3): the ImageNet backbone is frozen and only the new classifier head trains.
- **Stage 2**: the whole network is fine-tuned. The backbone uses a 10× smaller learning rate (`--backbone-lr-mult`).
- AdamW optimiser, cosine LR schedule, label smoothing, mixed precision on GPU, and early stopping on validation F1.

Each run writes to `outputs/runs/<model>-<timestamp>/`:

| File | Content |
|---|---|
| `training_curves.png` | loss and accuracy curves (train vs validation) |
| `test_confusion_matrix.png`, `test_roc_curve.png` | evaluation plots |
| `test_metrics.json`, `test_classification_report.txt` | Accuracy, Precision, Recall, F1, ROC-AUC |
| `history.json`, `config.json`, `best_model.pt` | logs and the best checkpoint |

The best checkpoint is also copied to `models/<model>_best.pt`, which the app and API load.

To re-evaluate a checkpoint later:
```bash
python scripts/evaluate.py --checkpoint models/efficientnet_b0_v3.pt --split test
```

**Compute:** training transfer models on the full CIFAKE set is slow on a CPU. Use a GPU (for example Google Colab,
below) or prepare a smaller split with `--max-per-class`.

### Training on Google Colab (free GPU)
```python
!git clone <your-repo-url> EYE-FOR-AI
%cd EYE-FOR-AI
!pip install -q -r requirements.txt
!python scripts/download_data.py --dataset cifake
!python scripts/download_data.py --dataset faces140k --max-shards 2 --max-per-class 5000
!python scripts/prepare_dataset.py --source data/raw/cifake data/raw/faces140k --max-per-class 20000
!python scripts/train.py --model efficientnet_b0 --epochs 10 --batch-size 64 --num-workers 2
# download models/efficientnet_b0_best.pt and put it in models/ on your laptop
```

---

## 5. Inference and explainability (FYP-2)

```bash
python scripts/predict.py my_photo.jpg --checkpoint models/efficientnet_b0_v3.pt
```
This prints a JSON result and saves `outputs/predictions/<folder>_<name>_analysis.png`, a four-panel image:
Original · Grad-CAM · Suspicious regions · ELA.

From Python:
```python
from eyeforai.inference import Predictor
p = Predictor("models/efficientnet_b0_v3.pt")
r = p.analyze("photo.jpg")
print(r.label, f"{r.prob_fake:.1%} AI")    # FAKE / REAL / UNCERTAIN, e.g. FAKE 96.3% AI
r.gradcam_overlay, r.regions_overlay, r.ela.heatmap   # RGB numpy arrays
```

- **Grad-CAM** takes the gradients of the FAKE score with respect to the last convolutional stage and shows which
  regions pushed the model toward FAKE. Thresholding the map gives the *suspicious region* boxes.
- **ELA** re-saves the image as JPEG at a known quality and maps the per-pixel error. Pasted or edited areas usually
  compress differently from the rest of the image and appear brighter. ELA works best on JPEG photos.

---

## 6. Web app (Streamlit)

```bash
streamlit run app.py        # opens http://localhost:8501
```
- Drag-and-drop upload (JPG/PNG/WEBP/BMP)
- REAL/FAKE verdict card with confidence % and probability bars
- Tabs for the Grad-CAM heatmap, suspicious regions, ELA, and a side-by-side view
- Sidebar settings: model checkpoint, decision threshold, Grad-CAM target, heatmap opacity, region sensitivity, ELA quality
- **History** tab (stored in the database), **Model Performance** tab (metrics and training plots), downloadable JSON report

The app picks the checkpoint with the best validation F1 by default. If `models/` is empty, it still runs and shows ELA only.

## 7. REST API (Flask)

```bash
python api.py
curl -F "image=@photo.jpg" http://localhost:5000/api/predict
curl -F "image=@photo.jpg" "http://localhost:5000/api/predict?explain=1"   # adds base64 PNG heatmaps
curl http://localhost:5000/api/history
```

## 8. Database (PostgreSQL)

SQLite (`outputs/eyeforai.db`) is used by default and needs no setup. To use PostgreSQL:
```bash
createdb eyeforai
export EYEFORAI_DATABASE_URL="postgresql+psycopg2://postgres:password@localhost:5432/eyeforai"
# Windows PowerShell: $env:EYEFORAI_DATABASE_URL="postgresql+psycopg2://..."
streamlit run app.py
```
The `predictions` table is created automatically.

---

## 9. FYP roadmap mapping

| Proposal item | Where it is implemented |
|---|---|
| Dataset & preprocessing | `scripts/download_data.py`, `scripts/prepare_dataset.py`, `eyeforai/data/` |
| Baseline CNN | `eyeforai/models/custom_cnn.py` |
| Transfer learning (ResNet / EfficientNet) | `eyeforai/models/factory.py`, `eyeforai/training/trainer.py` |
| Accuracy / Precision / Recall / F1 + plots | `eyeforai/training/metrics.py` |
| Grad-CAM / ELA / fake-region detection | `eyeforai/explain/` |
| Web interface with confidence score | `app.py` |
| Backend API + database | `api.py`, `eyeforai/db/history.py` |

**Suggested experiments for the report:** compare custom CNN vs ResNet50 vs EfficientNet-B0 on the same split; run
an ablation with and without augmentation (`--no-augment`) and with and without ImageNet weights (`--no-pretrained`);
test cross-dataset generalisation (train on CIFAKE, evaluate on CASIA with `evaluate.py`).

## 10. Limitations

- A model only knows the kinds of fakes it was trained on. New generators, heavy compression, or screenshots
  reduce accuracy. Present results as decision support, not proof.
- The v3 model is right on about 92% of the images it decides. About 6% of real photos are still called AI, and
  about 10% of AI images are called real. It is weakest on Midjourney-7, Recraft, ADM and VQDM images and on edited
  (rather than fully generated) photos. Screenshots, heavy filters and very small images also reduce accuracy.
- Training was done on a CPU at 160 px. Training on a GPU with more OpenFake shards at 224 px should improve results.
- ELA is not meaningful on PNGs or images that have been re-compressed many times.
