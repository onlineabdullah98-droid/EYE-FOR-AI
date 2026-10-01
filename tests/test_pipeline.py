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
    assert store.stats() == {"total": 1, "REAL": 0, "FAKE": 1, "UNCERTAIN": 0}
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


def test_hf_parquet_labels_mapped_by_name(tmp_path):
    """CIFAKE on HuggingFace uses 0=FAKE, 1=REAL; folders must follow the names, not the ids."""
    import json
    from collections import Counter

    import cv2
    import pyarrow as pa
    import pyarrow.parquet as pq

    sys.path.insert(0, str(ROOT / "scripts"))
    from download_data import _extract_parquet

    ok, png = cv2.imencode(".png", np.zeros((8, 8, 3), np.uint8))
    table = pa.table({"image": [{"bytes": png.tobytes(), "path": None}] * 3, "label": [0, 1, 1]})
    meta = {"info": {"features": {"image": {"_type": "Image"},
                                  "label": {"names": ["FAKE", "REAL"], "_type": "ClassLabel"}}}}
    table = table.replace_schema_metadata({"huggingface": json.dumps(meta)})
    pq.write_table(table, tmp_path / "train.parquet")

    _extract_parquet(tmp_path / "train.parquet", "train", tmp_path / "out", Counter(), None)
    labels = Counter(label for _, label in scan_directory(tmp_path / "out"))
    assert labels == {FAKE_LABEL: 1, REAL_LABEL: 2}


@pytest.mark.parametrize("name,expected", [
    ("real", "REAL"), ("RealArt", "REAL"), ("human", "REAL"), ("FAKE", "FAKE"), ("ai_gen", "FAKE"), ("AiArtData", "FAKE"),
])
def test_canonical_label(name, expected):
    sys.path.insert(0, str(ROOT / "scripts"))
    from download_data import canonical_label

    assert canonical_label(name) == expected


def test_normalize_image_same_policy_for_any_input_format():
    """PNG and JPEG inputs both come out downscaled and re-encoded, so file format cannot leak the label."""
    import random

    import cv2

    sys.path.insert(0, str(ROOT / "scripts"))
    from download_data import normalize_image

    img = np.random.randint(0, 255, (800, 600, 3), dtype=np.uint8)
    for ext in (".png", ".jpg"):
        data = cv2.imencode(ext, img)[1].tobytes()
        out, out_ext = normalize_image(data, random.Random(0))
        decoded = cv2.imdecode(np.frombuffer(out, np.uint8), cv2.IMREAD_COLOR)
        assert out_ext in (".jpg", ".png") and max(decoded.shape[:2]) == 512


def test_calibrate_thresholds():
    from eyeforai.training import calibrate_thresholds

    rng = np.random.default_rng(0)
    y = np.r_[np.zeros(1000, int), np.ones(1000, int)]
    p = np.r_[rng.beta(2, 5, 1000), rng.beta(5, 2, 1000)]  # overlapping real / AI score distributions
    t = calibrate_thresholds(y, p, max_fpr=0.05, max_fnr=0.10)
    assert t["real"] <= 0.5 <= t["ai"]
    assert (p[y == 0] >= t["ai"]).mean() <= 0.051
    assert (p[y == 1] < t["real"]).mean() <= 0.101


def test_three_way_verdict(tmp_path):
    from eyeforai.config import UNCERTAIN
    from eyeforai.inference import Predictor
    from eyeforai.utils import save_checkpoint

    model = build_model("custom_cnn", pretrained=False)
    ckpt = save_checkpoint(tmp_path / "m.pt", model_name="custom_cnn", num_classes=2, img_size=32,
                           mean=[0.5] * 3, std=[0.5] * 3, state_dict=model.state_dict(),
                           thresholds={"real": 0.3, "ai": 0.8})
    p = Predictor(ckpt)
    assert (p.real_threshold, p.threshold) == (0.3, 0.8)
    assert p._make_label(0.9)[0] == "FAKE"
    assert p._make_label(0.5)[0] == UNCERTAIN
    assert p._make_label(0.1)[0] == "REAL"
