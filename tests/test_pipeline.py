import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

from eyeforai.config import FAKE_LABEL, REAL_LABEL
from eyeforai.data import EvalTransform, TrainTransform, scan_directory
from eyeforai.data.dataset import label_from_path
from eyeforai.db import HistoryStore
from eyeforai.explain import GradCAM, error_level_analysis
from eyeforai.models import SUPPORTED_MODELS, build_model, get_gradcam_layer
from eyeforai.training import TrainConfig, compute_metrics, fit

ROOT = Path(__file__).resolve().parent.parent


@pytest.mark.parametrize("path,label", [
    ("cifake/train/REAL/1.jpg", REAL_LABEL), ("cifake/test/FAKE/1.jpg", FAKE_LABEL),
    ("CASIA2/Au/a.jpg", REAL_LABEL), ("CASIA2/Tp/b.tif", FAKE_LABEL), ("misc/other/c.png", None),
])
def test_label_from_path(path, label):
    assert label_from_path(Path(path)) == label


def test_transforms_shapes():
    rgb = np.random.randint(0, 255, (300, 200, 3), dtype=np.uint8)
    assert EvalTransform(96)(rgb).shape == (3, 96, 96)
    assert TrainTransform(96)(rgb).shape == (3, 96, 96)


@pytest.mark.parametrize("name", SUPPORTED_MODELS)
def test_models_forward_and_gradcam(name):
    model = build_model(name, pretrained=False).eval()
    x = torch.randn(1, 3, 96, 96)
    assert model(x).shape == (1, 2)
    with GradCAM(model, get_gradcam_layer(model, name)) as cam:
        heat, _ = cam(x, class_idx=1)
    assert heat.shape == (96, 96) and 0.0 <= heat.min() and heat.max() <= 1.0


def test_ela_outputs():
    rgb = np.random.randint(0, 255, (80, 120, 3), dtype=np.uint8)
    res = error_level_analysis(rgb, quality=90)
    assert res.heatmap.shape == rgb.shape and res.ela_image.dtype == np.uint8
    assert res.mean_error > 0


def test_metrics():
    m = compute_metrics([0, 0, 1, 1], [0.1, 0.6, 0.8, 0.9])
    assert m["accuracy"] == 0.75 and m["recall"] == 1.0 and m["roc_auc"] == 1.0


def test_history_store(tmp_path):
    store = HistoryStore(f"sqlite:///{tmp_path / 'h.db'}")
    store.add(filename="a.jpg", image_bytes=b"x", label="FAKE", confidence=0.9, prob_fake=0.9,
              model_name="m", inference_ms=1.0)
    assert store.stats() == {"total": 1, "REAL": 0, "FAKE": 1}
    assert store.recent()[0]["filename"] == "a.jpg"


def test_end_to_end(tiny_dataset, tmp_path, monkeypatch):
    assert len(scan_directory(tiny_dataset)) == 60
    splits = tmp_path / "splits"
    subprocess.run([sys.executable, str(ROOT / "scripts/prepare_dataset.py"), "--source", str(tiny_dataset),
                    "--out", str(splits)], check=True)

    import eyeforai.training.trainer as trainer
    monkeypatch.setattr(trainer, "MODELS_DIR", tmp_path / "models")
    cfg = TrainConfig(model_name="custom_cnn", splits_dir=str(splits), output_root=str(tmp_path / "runs"),
                      img_size=64, batch_size=8, epochs=6, lr=3e-3, num_workers=0, augment=False)
    ckpt = fit(cfg)
    assert ckpt.exists()
    run_dir = next((tmp_path / "runs").iterdir())
    for f in ("training_curves.png", "test_confusion_matrix.png", "test_metrics.json", "history.json"):
        assert (run_dir / f).exists(), f

    from eyeforai.inference import Predictor
    predictor = Predictor(ckpt)
    fake_img = next((tiny_dataset / "train" / "FAKE").glob("*.png"))
    result = predictor.analyze(fake_img)
    assert result.label in ("REAL", "FAKE") and 0.5 <= result.confidence <= 1.0
    assert result.gradcam_overlay.shape == result.image.shape
    assert result.ela is not None
    assert predictor.predict(fake_img.read_bytes()).label == result.label
