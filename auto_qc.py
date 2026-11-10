#!/usr/bin/env python3
"""Per-second QC of a finished Short, plus a bounded re-slice loop hook.

Checks:
  - resolution is 1080x1920 or 2160x3840
  - fps >= 30
  - audio is not silent
  - zero frames fail the empty-dark / logo / overlay-text / title / date detectors
  - hook banner and lower-third captions are visible
On failure, timestamps are blacklisted so the caller can re-slice and
re-render (bounded loop lives in render_item / pipeline).
"""
from __future__ import annotations

import json
import subprocess
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

import scene_filter

W1080, H1080 = 1080, 1920
W4K, H4K = 2160, 3840


def _probe(path: Path) -> dict:
    raw = subprocess.check_output([
        "ffprobe", "-v", "error", "-select_streams", "v:0",
        "-show_entries", "stream=width,height,r_frame_rate,avg_frame_rate",
        "-show_entries", "format=duration",
        "-of", "json", str(path),
    ], text=True)
    info = json.loads(raw)
    st = (info.get("streams") or [{}])[0]
    fmt = info.get("format") or {}
    def fps(s):
        if not s or s in ("0/0", "N/A"):
            return 0.0
        if "/" in str(s):
            a, b = str(s).split("/", 1)
            return float(a) / float(b or 1)
        return float(s)
    return {
        "w": int(st.get("width") or 0),
        "h": int(st.get("height") or 0),
        "fps": fps(st.get("avg_frame_rate") or st.get("r_frame_rate")),
        "dur": float(fmt.get("duration") or 0),
    }


def _mean_volume(path: Path) -> float | None:
    r = subprocess.run(
        ["ffmpeg", "-i", str(path), "-af", "volumedetect", "-vn", "-f", "null", "-"],
        capture_output=True, text=True,
    )
    for ln in (r.stderr or "").splitlines():
        if "mean_volume" in ln:
            try:
                return float(ln.split("mean_volume:")[1].split()[0])
            except Exception:
                return None
    return None


def _frame_at(path: Path, t: float) -> np.ndarray | None:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        return None
    cap.set(cv2.CAP_PROP_POS_MSEC, max(0.0, t) * 1000.0)
    ok, fr = cap.read()
    cap.release()
    if not ok or fr is None:
        return None
    return cv2.cvtColor(fr, cv2.COLOR_BGR2RGB)


def _banner_visible(frame: np.ndarray, layout: dict | None) -> bool:
    if not layout:
        return True  # nothing to check
    h, w = frame.shape[:2]
    sx, sy = w / 1080.0, h / 1920.0
    y = int(layout.get("bar_y", 0) * sy)
    bh = int(layout.get("bar_h", 0) * sy)
    if bh <= 0:
        return True
    bar = frame[max(0, y):min(h, y + bh), :]
    if bar.size == 0:
        return False
    L = bar.mean(axis=2) if bar.ndim == 3 else bar
    white = float((L > 230).mean())
    ink = float((L < 70).mean())
    return white > 0.5 and 0.008 < ink < 0.5


def _caption_visible(frame: np.ndarray, y_hint: int = 1420) -> bool:
    """Look for stroked lower-third glyphs near Y=1420 (scaled)."""
    h, w = frame.shape[:2]
    y = int(y_hint * (h / 1920.0))
    band = frame[max(0, y - 20):min(h, y + 90), int(w * 0.08):int(w * 0.92)]
    if band.size == 0:
        return False
    L = band.mean(axis=2) if band.ndim == 3 else band
    # thick black stroke around bright type
    bright = float((L > 180).mean())
    dark = float((L < 40).mean())
    return bright > 0.01 and dark > 0.01


def check(path: Path, layout: dict | None = None, caption_y: int = 1420) -> dict:
    path = Path(path)
    issues = []
    blacklist = []  # output timestamps that failed a detector
    meta = _probe(path)
    res_ok = (meta["w"], meta["h"]) in {(W1080, H1080), (W4K, H4K)}
    if not res_ok:
        issues.append(f"resolution {meta['w']}x{meta['h']} not 1080x1920 or 2160x3840")
    if meta["fps"] < 29.5:
        issues.append(f"fps {meta['fps']:.2f} < 30")
    vol = _mean_volume(path)
    if vol is None or vol <= -35:
        issues.append(f"audio silent or missing (mean_volume={vol})")
    dur = meta["dur"] or 0
    # 4 fps so short typography / black flashes between 1s ticks cannot slip through.
    try:
        step = 1.0 / float(scene_filter.QC_FPS)
    except Exception:
        step = 0.25
    n = max(1, int((dur / step) + 0.5)) if dur > 0 else 1
    banner_hits = 0
    cap_hits = 0
    checked = 0
    cap = cv2.VideoCapture(str(path))
    for i in range(n):
        t = min(dur - 0.04, i * step + 0.04) if dur > 0.2 else 0.0
        fr = None
        if cap.isOpened():
            cap.set(cv2.CAP_PROP_POS_MSEC, max(0.0, float(t)) * 1000.0)
            ok, raw = cap.read()
            if ok and raw is not None:
                fr = cv2.cvtColor(raw, cv2.COLOR_BGR2RGB)
        if fr is None:
            fr = _frame_at(path, t)
        if fr is None:
            issues.append(f"missing frame at {t:.2f}s")
            continue
        checked += 1
        # scale the picture region only? detectors must ignore the white bar
        # and the black canvas. Use the plate if layout is known.
        plate = fr
        if layout and layout.get("plate_y") is not None:
            sy = fr.shape[0] / 1920.0
            y = int(layout["plate_y"] * sy)
            ph = int(layout.get("plate_h", 0) * sy)
            if ph > 0:
                plate = fr[y:y + ph, :]
        # Hard cards only: logo / title / date / overlay typography / empty near-black.
        # Cinematic darkness or saturated lighting in real story footage is NOT a fail.
        # (Fool Night and other thrillers are full of dark story frames.)
        HARD = {
            "centered_logo", "logo_animation", "graphic_blob", "text_card",
            "flat_card", "title_card", "title_logo", "text_overlay", "text_page",
            "date_card", "logo", "near_black", "empty_dark",
        }
        reasons = []
        if plate.size:
            try:
                import pipeline
                rs2, _ = pipeline.classify_shot(plate)
                reasons.extend(r for r in rs2 if r in HARD)
            except Exception:
                rs2 = []
            try:
                sf = scene_filter.detect_frame(plate)
                # typography / empty-dark hits; ignore lone cinematic "dark" / "red_solid"
                reasons.extend(r for r in sf if r in HARD)
            except Exception:
                pass
            reasons = list(dict.fromkeys(reasons))
        if reasons:
            blacklist.append({"t": round(float(t), 2), "reasons": reasons})
        if _banner_visible(fr, layout):
            banner_hits += 1
        if _caption_visible(fr, caption_y):
            cap_hits += 1
    if blacklist:
        issues.append(f"{len(blacklist)} frame(s) failed empty-dark/logo/text detectors")
    if layout and checked and banner_hits < max(2, checked // 5):
        issues.append("hook banner not visible often enough")
    # captions start at 0.4s and cover ~16-20s; require visibility on a portion
    if checked and cap_hits < 2:
        issues.append("lower-third captions not visible")
    try:
        cap.release()
    except Exception:
        pass
    return {
        "ok": not issues,
        "issues": issues,
        "blacklist": blacklist,
        "meta": meta,
        "mean_volume_db": vol,
        "banner_hits": banner_hits,
        "caption_hits": cap_hits,
        "checked": checked,
    }


def write_report(path: Path, report: dict) -> Path:
    path = Path(path)
    path.write_text(json.dumps(report, indent=2, default=str))
    return path
