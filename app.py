"""EYE FOR AI - Streamlit web interface.

Run:  streamlit run app.py
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import streamlit as st

from eyeforai.config import MIN_RELIABLE_SIZE, OUTPUTS_DIR, PROJECT_ROOT
from eyeforai.db import HistoryStore
from eyeforai.explain import error_level_analysis
from eyeforai.inference import Predictor, find_checkpoints
from eyeforai.utils import read_image_rgb

st.set_page_config(page_title="EYE FOR AI - Fake Image Detector", page_icon="👁️", layout="wide")

CSS = """
<style>
.block-container {padding-top: 1.6rem; max-width: 1250px;}
.hero {padding: 1.4rem 1.8rem; border-radius: 18px; color: #fff; margin-bottom: 1.2rem;
       background: linear-gradient(120deg, #0f172a 0%, #1e3a8a 55%, #7c3aed 100%);}
.hero h1 {margin: 0; font-size: 2.1rem; letter-spacing: .5px;}
.hero p {margin: .35rem 0 0; opacity: .85;}
.verdict {border-radius: 16px; padding: 1.3rem 1.5rem; color: #fff; text-align: center; margin-bottom: .8rem;}
.verdict.fake {background: linear-gradient(135deg, #b91c1c, #ef4444);}
.verdict.real {background: linear-gradient(135deg, #047857, #10b981);}
.verdict.uncertain {background: linear-gradient(135deg, #b45309, #f59e0b);}
.verdict .label {font-size: 2.4rem; font-weight: 800; letter-spacing: 2px;}
.verdict .conf {font-size: 1.15rem; opacity: .95;}
.verdict .pct {font-size: 3.2rem; font-weight: 800; line-height: 1.1; margin-top: .3rem;}
.meter {position: relative; height: 18px; border-radius: 9px; margin: .5rem 0 .3rem;
        background: linear-gradient(90deg, #10b981 0%, #facc15 50%, #ef4444 100%);}
.meter .pin {position: absolute; top: -6px; width: 6px; height: 30px; margin-left: -3px; border-radius: 3px;
             background: #111827; box-shadow: 0 0 0 2px #fff;}
.meter-scale {display: flex; justify-content: space-between; font-size: .8rem; opacity: .75;}
.card {border: 1px solid rgba(128,128,128,.25); border-radius: 14px; padding: .9rem 1.1rem; margin-bottom: .7rem;}
.bar-wrap {background: rgba(128,128,128,.18); border-radius: 8px; height: 14px; overflow: hidden; margin: .25rem 0 .6rem;}
.bar {height: 100%; border-radius: 8px;}
.small {font-size: .85rem; opacity: .75;}
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)


# ---------- cached resources ----------
@st.cache_resource(show_spinner="Loading model...")
def load_predictor(path: str, mtime: float) -> Predictor:  # mtime busts the cache when the file changes
    return Predictor(path)


@st.cache_resource
def load_history() -> HistoryStore | None:
    try:
        return HistoryStore()
    except Exception as e:  # DB is optional - the detector must still work
        st.warning(f"History database unavailable: {e}")
        return None


def prob_bar(label: str, value: float, color: str) -> str:
    return (f"<div><b>{label}</b> <span style='float:right'>{value:.1%}</span></div>"
            f"<div class='bar-wrap'><div class='bar' style='width:{value*100:.1f}%;background:{color}'></div></div>")


def verdict_text(prob_ai: float, real_t: float, ai_t: float) -> str:
    if prob_ai >= max(ai_t, 0.9):
        return "Very likely AI-generated / manipulated"
    if prob_ai >= ai_t:
        return "Likely AI-generated / manipulated"
    if prob_ai < min(real_t, 0.1):
        return "Very likely real - a photo or artwork not made by AI"
    if prob_ai < real_t:
        return "Likely real - not made by AI"
    return "The model is not sure - check the heatmaps and the image source"


def ai_meter(prob_ai: float, real_t: float, ai_t: float) -> str:
    lo, hi = real_t * 100, ai_t * 100
    gradient = (f"linear-gradient(90deg, #10b981 0%, #10b981 {lo:.1f}%, #f59e0b {lo:.1f}%, #f59e0b {hi:.1f}%, "
                f"#ef4444 {hi:.1f}%, #ef4444 100%)")
    return (f"<div><b>AI likelihood</b> <span style='float:right'><b>{prob_ai:.1%}</b></span></div>"
            f"<div class='meter' style='background:{gradient}'><div class='pin' style='left:{prob_ai*100:.1f}%'>"
            "</div></div><div class='meter-scale'><span>Real</span><span>Uncertain</span><span>AI</span></div>")


def results_dir(checkpoint: Path, model_name: str) -> Path | None:
    """Plots for a checkpoint: results/<checkpoint name> (shipped with the repo), else the latest local run."""
    shipped = PROJECT_ROOT / "results" / checkpoint.stem
    if shipped.is_dir():
        return shipped
    runs = sorted((OUTPUTS_DIR / "runs").glob(f"{model_name}-*"), key=lambda p: p.stat().st_mtime, reverse=True)
    return runs[0] if runs else None


# ---------- sidebar ----------
with st.sidebar:
    st.markdown("## ⚙️ Settings")
    checkpoints = find_checkpoints()
    predictor = None
    if checkpoints:
        ckpt = st.selectbox("Model checkpoint", checkpoints, format_func=lambda p: p.name)
        predictor = load_predictor(str(ckpt), ckpt.stat().st_mtime)
        vm = predictor.val_metrics
        st.caption(f"**{predictor.model_name}** · input {predictor.img_size}px · {predictor.device}")
        if vm:
            st.caption(f"Validation: Acc {vm.get('accuracy', 0):.1%} · F1 {vm.get('f1', 0):.1%} · "
                       f"AUC {vm.get('roc_auc', 0):.3f}")
    else:
        st.error("No trained model found in `models/`.\n\nTrain one with `python scripts/train.py` — "
                 "until then only ELA analysis is available.")

    st.markdown("### Detection")
    calibrated = predictor.calibrated if predictor else {"real": 0.5, "ai": 0.5}
    real_t, threshold = st.slider(
        "Uncertain zone (AI likelihood)", 0.0, 1.0, (float(calibrated["real"]), float(calibrated["ai"])), 0.01,
        help="Below the left handle: REAL. At or above the right handle: AI-GENERATED. In between: UNCERTAIN. "
             "The defaults were calibrated on validation data so that about 5% of real photos are called AI.")
    st.markdown("### Explainability")
    cam_target = st.radio("Grad-CAM explains", ["fake", "predicted"], horizontal=True,
                          format_func=lambda v: "FAKE evidence" if v == "fake" else "Predicted class")
    cam_alpha = st.slider("Heatmap opacity", 0.1, 0.9, 0.45, 0.05)
    region_thr = st.slider("Region sensitivity", 0.3, 0.9, 0.6, 0.05,
                           help="Grad-CAM activation needed to mark a region as suspicious (lower = more boxes).")
    ela_quality = st.slider("ELA JPEG quality", 70, 98, 90, 1)
    save_history = st.toggle("Save results to history", value=True)

history = load_history()

st.markdown("<div class='hero'><h1>👁️ EYE FOR AI</h1>"
            "<p>AI system for detecting manipulated and AI-generated images · CNN + Transfer Learning · "
            "Grad-CAM &amp; Error Level Analysis</p></div>", unsafe_allow_html=True)

tab_analyze, tab_history, tab_perf, tab_about = st.tabs(["🔍 Analyze", "📜 History", "📊 Model Performance", "ℹ️ About"])

# ---------- analyze ----------
with tab_analyze:
    uploaded = st.file_uploader("Drag and drop an image here", type=["jpg", "jpeg", "png", "webp", "bmp"],
                                help="Max 200 MB. JPEG photos give the most informative ELA results.")
    if uploaded is None:
        st.info("⬆️ Upload an image to start the analysis.")
    else:
        data = uploaded.getvalue()
        try:
            rgb = read_image_rgb(data)
        except ValueError:
            st.error("This file could not be decoded as an image.")
            st.stop()

        col_img, col_res = st.columns([1.25, 1], gap="large")
        with col_img:
            st.image(rgb, caption=f"{uploaded.name} · {rgb.shape[1]}×{rgb.shape[0]} px · {len(data)/1024:.0f} KB",
                     width="stretch")

        if predictor is None:
            with col_res:
                st.warning("No model loaded — showing Error Level Analysis only.")
            ela = error_level_analysis(rgb, quality=ela_quality)
            st.image(ela.heatmap, caption=f"ELA heatmap · mean error {ela.mean_error:.2f}", width="stretch")
            st.stop()

        predictor.threshold, predictor.real_threshold = threshold, real_t
        with st.spinner("Analyzing image..."):
            result = predictor.analyze(rgb, cam_target=cam_target, cam_alpha=cam_alpha,
                                       region_threshold=region_thr, ela_quality=ela_quality)

        digest = hashlib.sha256(data).hexdigest()
        if save_history and history and st.session_state.get("last_saved") != (digest, threshold, real_t):
            history.add(filename=uploaded.name, image_bytes=data, label=result.label, confidence=result.confidence,
                        prob_fake=result.prob_fake, model_name=result.model_name, inference_ms=result.inference_ms)
            st.session_state["last_saved"] = (digest, threshold, real_t)

        with col_res:
            kind = "fake" if result.is_fake else "uncertain" if result.is_uncertain else "real"
            title = {"fake": "🤖 AI-GENERATED", "uncertain": "🤔 UNCERTAIN", "real": "✅ REAL (NOT AI)"}[kind]
            verdict = ("Image too small for a reliable verdict" if result.low_resolution
                       else verdict_text(result.prob_fake, real_t, threshold))
            st.markdown(f"<div class='verdict {kind}'><div class='label'>{title}</div>"
                        f"<div class='pct'>{result.prob_fake:.0%} AI</div>"
                        f"<div class='conf'>{verdict}</div></div>",
                        unsafe_allow_html=True)
            if result.low_resolution:
                st.warning(f"This image is only {rgb.shape[1]}×{rgb.shape[0]} px. Images smaller than "
                           f"{MIN_RELIABLE_SIZE} px lose the fine details the model relies on, so no verdict is "
                           "given. Try a larger version of the image.")
            st.markdown("<div class='card'>" + ai_meter(result.prob_fake, real_t, threshold)
                        + prob_bar("Real / authentic", result.prob_real, "#10b981") + "</div>",
                        unsafe_allow_html=True)
            c1, c2, c3 = st.columns(3)
            c1.metric("Inference", f"{result.inference_ms:.0f} ms")
            c2.metric("Regions", len(result.regions))
            c3.metric("ELA error", f"{result.ela.mean_error:.2f}")
            st.download_button("⬇️ Download JSON report", json.dumps({"file": uploaded.name, **result.summary()},
                               indent=2), file_name=f"{Path(uploaded.name).stem}_report.json",
                               mime="application/json", width="stretch")

        st.markdown("### 🔬 Visual breakdown")
        v1, v2, v3, v4 = st.tabs(["Grad-CAM heatmap", "Suspicious regions", "Error Level Analysis", "Side by side"])
        with v1:
            st.image(result.gradcam_overlay, width="stretch")
            st.caption("Red/yellow areas influenced the model's decision the most "
                       f"({'evidence for FAKE' if cam_target == 'fake' else 'evidence for ' + result.label}).")
        with v2:
            st.image(result.regions_overlay, width="stretch")
            st.caption("Boxes mark high-activation Grad-CAM regions (shown only for FAKE verdicts)."
                       if result.regions else "No suspicious regions above the current sensitivity.")
        with v3:
            e1, e2 = st.columns(2)
            e1.image(result.ela.ela_image, caption="Amplified ELA difference", width="stretch")
            e2.image(result.ela.heatmap, caption="ELA heatmap", width="stretch")
            st.caption(f"Mean error {result.ela.mean_error:.2f} · max {result.ela.max_error:.0f} · "
                       f"hotspots {result.ela.hotspot_ratio:.2%}. Edited regions often appear brighter than the "
                       "rest of the image because they were compressed a different number of times.")
        with v4:
            st.image(np.hstack([result.image, result.gradcam_overlay, result.ela.heatmap]),
                     caption="Original · Grad-CAM · ELA", width="stretch")

# ---------- history ----------
with tab_history:
    if history is None:
        st.info("History database is not configured.")
    else:
        s = history.stats()
        h1, h2, h3, h4 = st.columns(4)
        h1.metric("Images analysed", s["total"])
        h2.metric("AI-generated", s["FAKE"])
        h3.metric("Real", s["REAL"])
        h4.metric("Uncertain", s["UNCERTAIN"])
        rows = history.recent(100)
        if rows:
            st.dataframe(rows, width="stretch", hide_index=True)
            if st.button("🗑️ Clear history"):
                history.clear()
                st.rerun()
        else:
            st.caption("No predictions yet.")

# ---------- model performance ----------
with tab_perf:
    run = results_dir(ckpt, predictor.model_name) if predictor else None
    if run is None:
        st.info("Training plots appear here after running `scripts/train.py`.")
    else:
        st.caption(f"Run: `{run.name}`")
        metrics_file = next(iter(sorted(run.glob("*_metrics.json"))), None)
        if metrics_file:
            m = json.loads(metrics_file.read_text())
            cols = st.columns(5)
            for col, key in zip(cols, ("accuracy", "precision", "recall", "f1", "roc_auc")):
                col.metric(key.replace("_", "-").upper() if key == "roc_auc" else key.title(), f"{m[key]:.2%}"
                           if key != "roc_auc" else f"{m[key]:.4f}")
        three_way = run / "test_three_way.json"
        if three_way.exists():
            t = json.loads(three_way.read_text())
            st.caption(f"With the REAL / UNCERTAIN / AI zone (REAL below {t['thresholds']['real']:.0%}, "
                       f"AI from {t['thresholds']['ai']:.0%}):")
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Accuracy when decided", f"{t['accuracy_when_decided']:.1%}")
            c2.metric("Real photos called AI", f"{t['real_called_ai']:.1%}")
            c3.metric("AI images called real", f"{t['ai_called_real']:.1%}")
            c4.metric("Uncertain", f"{t['uncertain_share']:.1%}")
        if (run / "training_curves.png").exists():
            st.image(str(run / "training_curves.png"), caption="Loss / accuracy curves")
        p1, p2 = st.columns(2)
        for col, pattern in ((p1, "*_confusion_matrix.png"), (p2, "*_roc_curve.png")):
            img = next(iter(run.glob(pattern)), None)
            if img:
                col.image(str(img))
        breakdown = run / "per_dataset_and_generator.txt"
        if breakdown.exists():
            with st.expander("Accuracy per dataset and per AI generator"):
                st.code(breakdown.read_text(), language=None)

# ---------- about ----------
with tab_about:
    st.markdown("""
**EYE FOR AI** tells you whether an uploaded image is a **REAL** photo, **AI-generated / manipulated**, or
**UNCERTAIN**. It shows how likely the image is to be AI as a percentage and explains the result with heatmaps.

| Component | Technique |
|---|---|
| Preprocessing | OpenCV decoding, resize, ImageNet normalisation; JPEG/blur/crop/flip augmentation |
| Baseline | Custom 4-block CNN trained from scratch |
| Main model | Transfer learning — EfficientNet-B0/B2 or ResNet50 (ImageNet), two-stage fine-tuning |
| Explainability | Grad-CAM heatmaps + suspicious-region boxes, Error Level Analysis (ELA) |
| Storage | SQLAlchemy — SQLite by default, PostgreSQL via `EYEFORAI_DATABASE_URL` |

⚠️ *No detector is perfect. Treat results as decision support, not proof — especially for images
very different from the training data (new generators, heavy compression, screenshots).*
""")
