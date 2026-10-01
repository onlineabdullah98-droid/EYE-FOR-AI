# 👁️ EYE FOR AI — AI System for Detecting Manipulated and Fake Images

Final Year Project (BSCS/BSAI/BSSE/BSCY). You upload an image and the system:

1. classifies it as **REAL** or **FAKE** (AI-generated or manipulated),
2. reports a **confidence score** (e.g. *94.5% FAKE*),
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

## 1. Project structure

```
EYE-FOR-AI/
├── app.py                       # Streamlit web app (drag & drop, verdict, heatmaps, history)
├── api.py                       # Flask REST API  (POST /api/predict)
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
│   ├── download_data.py         # download CIFAKE / 140k Faces / CASIA 2.0 from Kaggle
│   ├── prepare_dataset.py       # stratified train/val/test CSV manifests
│   ├── train.py                 # train any model
│   ├── evaluate.py              # evaluate a saved checkpoint
│   └── predict.py               # CLI inference + saved analysis panel
├── tests/                       # pytest suite (runs on a synthetic mini-dataset)
├── data/                        # datasets (git-ignored)
├── models/                      # trained checkpoints *.pt (git-ignored)
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

| Dataset | Content | Detects | Kaggle id |
|---|---|---|---|
| **CIFAKE** (recommended start) | 60k real (CIFAR-10) + 60k Stable Diffusion images, 32×32 | AI-generated images | `birdy654/cifake-real-and-ai-generated-synthetic-images` |
| **140k Real & Fake Faces** | 70k FFHQ + 70k StyleGAN faces, 256×256 | GAN / deepfake faces | `xhlulu/140k-real-and-fake-faces` |
| **CASIA v2.0** | 7.4k authentic (`Au`) + 5.1k tampered (`Tp`) | Splicing / copy-move edits | `divg07/casia-20-image-tamperingdetection-dataset` |

**Kaggle API setup (one time):** go to kaggle.com → *Settings* → *API* → *Create New Token*, then put
`kaggle.json` in `~/.kaggle/` (Windows: `C:\Users\<you>\.kaggle\`).

```bash
python scripts/download_data.py --dataset cifake
python scripts/prepare_dataset.py --source data/raw/cifake
# faster experiments on a laptop:
python scripts/prepare_dataset.py --source data/raw/cifake --max-per-class 10000
# combine datasets (AI-generated + manipulated):
python scripts/prepare_dataset.py --source data/raw/cifake data/raw/casia2 --max-per-class 8000
```

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
python scripts/evaluate.py --checkpoint models/efficientnet_b0_best.pt --split test
```

**Compute:** training transfer models on the full CIFAKE set is slow on a CPU. Use a GPU (for example Google Colab,
below) or prepare a smaller split with `--max-per-class`.

### Training on Google Colab (free GPU)
```python
!git clone <your-repo-url> EYE-FOR-AI
%cd EYE-FOR-AI
!pip install -q -r requirements.txt
# upload kaggle.json, then:
!mkdir -p ~/.kaggle && cp kaggle.json ~/.kaggle/ && chmod 600 ~/.kaggle/kaggle.json
!python scripts/download_data.py --dataset cifake
!python scripts/prepare_dataset.py --source data/raw/cifake --max-per-class 20000
!python scripts/train.py --model efficientnet_b0 --epochs 10 --batch-size 64 --num-workers 2
# download models/efficientnet_b0_best.pt and put it in models/ on your laptop
```

---

## 5. Inference and explainability (FYP-2)

```bash
python scripts/predict.py my_photo.jpg --checkpoint models/efficientnet_b0_best.pt
```
This prints a JSON result and saves `outputs/predictions/<folder>_<name>_analysis.png`, a four-panel image:
Original · Grad-CAM · Suspicious regions · ELA.

From Python:
```python
from eyeforai.inference import Predictor
p = Predictor("models/efficientnet_b0_best.pt")
r = p.analyze("photo.jpg")
print(r.label, f"{r.confidence:.1%}")      # FAKE 94.5%
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

If no model has been trained yet, the app still runs and shows ELA only.

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
- CIFAKE images are 32×32. A model trained only on CIFAKE will not generalise well to high-resolution photos.
  Combine datasets for a stronger demo.
- ELA is not meaningful on PNGs or images that have been re-compressed many times.
