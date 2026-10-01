"""EYE FOR AI - Flask REST API (for mobile apps / other front-ends).

Run:   python api.py            (or: flask --app api run)
Call:  curl -F "image=@photo.jpg" http://localhost:5000/api/predict
       curl -F "image=@photo.jpg" "http://localhost:5000/api/predict?explain=1"
"""

from __future__ import annotations

import base64
import os

import cv2
import numpy as np
from flask import Flask, jsonify, request

from eyeforai.db import HistoryStore
from eyeforai.inference import Predictor, find_checkpoints

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 20 * 1024 * 1024  # 20 MB uploads

_ckpt = os.getenv("EYEFORAI_CHECKPOINT") or next(iter(find_checkpoints()), None)
predictor = Predictor(_ckpt) if _ckpt else None
history = HistoryStore()


def _png_b64(rgb: np.ndarray) -> str:
    ok, buf = cv2.imencode(".png", cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
    return base64.b64encode(buf.tobytes()).decode() if ok else ""


@app.get("/api/health")
def health():
    return jsonify(status="ok", model=predictor.model_name if predictor else None)


@app.post("/api/predict")
def predict():
    if predictor is None:
        return jsonify(error="No trained model available. Train one with scripts/train.py."), 503
    file = request.files.get("image")
    if file is None:
        return jsonify(error="Send the image as multipart form field 'image'."), 400
    data = file.read()
    explain = request.args.get("explain", "0") == "1"
    try:
        result = predictor.analyze(data, with_ela=explain) if explain else predictor.predict(data)
    except ValueError:
        return jsonify(error="Could not decode image."), 400

    history.add(filename=file.filename or "upload", image_bytes=data, label=result.label,
                confidence=result.confidence, prob_fake=result.prob_fake, model_name=result.model_name,
                inference_ms=result.inference_ms)
    payload = result.summary()
    if explain:
        payload["gradcam_png"] = _png_b64(result.gradcam_overlay)
        payload["regions_png"] = _png_b64(result.regions_overlay)
        payload["ela_png"] = _png_b64(result.ela.heatmap)
    return jsonify(payload)


@app.get("/api/history")
def get_history():
    return jsonify(history.recent(int(request.args.get("limit", 50))))


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.getenv("PORT", 5000)))
