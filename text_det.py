"""Neural scene-text detector (PP-OCRv3 DB, OpenCV zoo, Apache-2.0, 2.4 MB).

Used by auto_qc.frame_reasons() for both clip screening and final QC, so title
cards ("THIS NOVEMBER"), studio logos (LIONSGATE), rating/credit blocks and
date cards are cut out of the edit. Replaces the old hand-made residual/glyph
heuristics that fired on faces and dim warm shots and missed big titles.
"""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

_MODEL = Path(__file__).resolve().parent / "models" / "text_det_ppocrv3.onnx"
_NET = None
IN_W, IN_H = 736, 320  # multiples of 32, roughly the 2.39:1 / 16:9 plate aspect


def _net():
    global _NET
    if _NET is None:
        m = cv2.dnn_TextDetectionModel_DB(str(_MODEL))
        m.setBinaryThreshold(0.3)
        m.setPolygonThreshold(0.55)
        m.setMaxCandidates(200)
        m.setUnclipRatio(1.8)
        m.setInputParams(1.0 / 255.0, (IN_W, IN_H), (122.67891434, 116.66876762, 104.00698793), True)
        _NET = m
    return _NET


def text_boxes(rgb: np.ndarray) -> list[dict]:
    """Return text boxes in normalised coords: x, y, w, h (0..1) + score."""
    if rgb is None or not getattr(rgb, "size", 0):
        return []
    h, w = rgb.shape[:2]
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR) if rgb.ndim == 3 else cv2.cvtColor(rgb, cv2.COLOR_GRAY2BGR)
    img = cv2.resize(bgr, (IN_W, IN_H), interpolation=cv2.INTER_AREA)
    try:
        polys, confs = _net().detect(img)
    except Exception:
        return []
    out = []
    for p, c in zip(polys, confs if len(confs) else [1.0] * len(polys)):
        p = np.asarray(p, dtype=np.float32).reshape(-1, 2)
        x0, y0 = p.min(axis=0)
        x1, y1 = p.max(axis=0)
        out.append({"x": float(x0 / IN_W), "y": float(y0 / IN_H),
                    "w": float((x1 - x0) / IN_W), "h": float((y1 - y0) / IN_H),
                    "score": float(c)})
    return out


def _edge_ratio(rgb: np.ndarray, b: dict) -> float:
    g = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY).astype(np.float32) if rgb.ndim == 3 else rgb.astype(np.float32)
    H, W = g.shape
    lap = np.abs(cv2.Laplacian(g, cv2.CV_32F))
    m = np.zeros((H, W), bool)
    m[max(0, int(b["y"] * H)):max(0, int((b["y"] + b["h"]) * H)),
      max(0, int(b["x"] * W)):max(0, int((b["x"] + b["w"]) * W))] = True
    if not m.any():
        return 0.0
    if not (~m).any():
        return 10.0
    return float(lap[m].mean()) / max(0.5, float(lap[~m].mean()))


def has_overlay_text(rgb: np.ndarray) -> tuple[bool, dict]:
    """True for title/logo/date/credit typography; ignores tiny incidental or corner text."""
    boxes = text_boxes(rgb)
    aspect = rgb.shape[1] / max(1, rgb.shape[0])
    sig = []
    for b in boxes:
        cx, cy = b["x"] + b["w"] / 2, b["y"] + b["h"] / 2
        corner = (cx < 0.16 or cx > 0.84) and (cy < 0.2 or cy > 0.8) and b["w"] < 0.2
        if corner:
            continue  # network / channel bug
        if b["h"] < 0.045 or b["w"] < 0.04:
            continue  # tiny incidental text (signs, screens)
        # Typography lines are wide and short; fire, lamps, drones are blob-shaped.
        # Aspect in pixels of the source frame (boxes are normalised per axis).
        ar = (b["w"] * aspect) / max(1e-6, b["h"])
        # Stylized movie-title logo: big box, and the letters are much sharper
        # than the backdrop (fire / wrestlers / crowds are not: ratio < 1).
        huge_title = b["w"] >= 0.35 and b["h"] >= 0.25 and ar >= 1.3 and _edge_ratio(rgb, b) >= 2.0
        if ar < 2.6 and not huge_title:
            continue
        sig.append(b)
    area = sum(b["w"] * b["h"] for b in sig)
    big = any(b["h"] >= 0.08 and b["w"] >= 0.12 for b in sig)
    lines = len(sig)
    hit = big or (lines >= 2 and area >= 0.012) or area >= 0.02
    return bool(hit), {"n": lines, "area": round(area, 4), "big": big}
