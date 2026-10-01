"""Error Level Analysis (ELA).

Re-saves the image as JPEG at a known quality and measures the per-pixel
difference. Regions that were pasted/edited after the last save usually
compress differently from the rest of the image and therefore "light up".

Notes: ELA is most informative for JPEG photos. PNGs, screenshots and
heavily re-compressed social-media images give weaker, more uniform signals.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass
class ELAResult:
    ela_image: np.ndarray   # amplified RGB difference image (uint8)
    heatmap: np.ndarray     # colour-mapped single-channel ELA (uint8 RGB)
    error_map: np.ndarray   # raw mean absolute error per pixel (float32, 0-255)
    mean_error: float
    max_error: float
    hotspot_ratio: float    # fraction of pixels whose error is far above the image mean


def error_level_analysis(rgb: np.ndarray, quality: int = 90, amplify: float | None = None) -> ELAResult:
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    ok, enc = cv2.imencode(".jpg", bgr, [cv2.IMWRITE_JPEG_QUALITY, int(quality)])
    if not ok:
        raise RuntimeError("JPEG encoding failed during ELA")
    resaved = cv2.imdecode(enc, cv2.IMREAD_COLOR)

    diff = cv2.absdiff(bgr, resaved).astype(np.float32)
    error_map = diff.mean(axis=2)
    max_error = float(diff.max())
    # Scale by a high percentile rather than the max so a few outlier pixels don't leave the map almost black.
    scale = amplify if amplify is not None else (255.0 / max(float(np.percentile(diff, 99.5)), 1.0))
    ela_bgr = np.clip(diff * scale, 0, 255).astype(np.uint8)

    gray = np.clip(error_map * scale, 0, 255).astype(np.uint8)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    heat_bgr = cv2.applyColorMap(gray, cv2.COLORMAP_INFERNO)

    mean_error = float(error_map.mean())
    hotspot_ratio = float((error_map > mean_error + 3 * error_map.std()).mean()) if error_map.std() > 0 else 0.0

    return ELAResult(
        ela_image=cv2.cvtColor(ela_bgr, cv2.COLOR_BGR2RGB),
        heatmap=cv2.cvtColor(heat_bgr, cv2.COLOR_BGR2RGB),
        error_map=error_map,
        mean_error=mean_error,
        max_error=max_error,
        hotspot_ratio=hotspot_ratio,
    )
