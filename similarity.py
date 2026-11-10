#!/usr/bin/env python3
"""Shot uniqueness helpers: source-range padding + perceptual / histogram similarity.

Once a source range [s, e] is used, anything overlapping [s-2, e+2] is blocked.
Keyframes of used shots are compared with a 64-bit DCT pHash and an HSV histogram
so visually similar angles (same car, same table, same booth) cannot be re-used
even if they sit far apart in the trailer.
"""
from __future__ import annotations

from typing import Iterable

import cv2
import numpy as np

RANGE_PAD = 2.0
# Tight pHash = same frame / same angle. Combined with histogram so two
# different dark interiors are not automatically collapsed together.
PHASH_HARD = 8
PHASH_SOFT = 14
HIST_HARD = 0.96
HIST_SOFT = 0.88
HIST_COMBO = 0.93


def phash64(rgb: np.ndarray) -> int:
    """Classic 8x8 DCT perceptual hash (64 bits)."""
    if rgb is None or getattr(rgb, "size", 0) == 0:
        return 0
    if rgb.ndim == 3:
        gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY) if rgb.shape[2] >= 3 else rgb[:, :, 0]
    else:
        gray = rgb
    small = cv2.resize(gray, (32, 32), interpolation=cv2.INTER_AREA)
    dct = cv2.dct(np.float32(small))
    core = dct[:8, :8]
    # Exclude DC so overall brightness does not dominate the hash.
    med = float(np.median(core.reshape(-1)[1:]))
    bits = (core > med).astype(np.uint8).flatten()
    h = 0
    for b in bits:
        h = (h << 1) | int(b)
    return int(h)


def hamming64(a: int, b: int) -> int:
    return int((int(a) ^ int(b)).bit_count())


def hsv_hist(rgb: np.ndarray) -> np.ndarray:
    """L2-normalized HSV histogram (16 hue + 8 sat + 8 val)."""
    vec = np.zeros(32, dtype=np.float32)
    if rgb is None or getattr(rgb, "size", 0) == 0:
        return vec
    if rgb.ndim == 2:
        rgb = cv2.cvtColor(rgb, cv2.COLOR_GRAY2RGB)
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    h, s, v = cv2.split(hsv)
    hh, _ = np.histogram(h, bins=16, range=(0, 180), density=False)
    ss, _ = np.histogram(s, bins=8, range=(0, 256), density=False)
    vv, _ = np.histogram(v, bins=8, range=(0, 256), density=False)
    vec = np.concatenate([hh, ss, vv]).astype(np.float32)
    n = float(np.linalg.norm(vec))
    if n > 0:
        vec /= n
    return vec


def hist_corr(a: np.ndarray, b: np.ndarray) -> float:
    if a is None or b is None:
        return 0.0
    a = np.asarray(a, dtype=np.float32).reshape(-1)
    b = np.asarray(b, dtype=np.float32).reshape(-1)
    n = min(a.size, b.size)
    if n == 0:
        return 0.0
    return float(np.dot(a[:n], b[:n]))


def fingerprint_frame(rgb: np.ndarray) -> dict:
    return {"phash": phash64(rgb), "hist": hsv_hist(rgb).tolist()}


def fingerprints_similar(a: dict | None, b: dict | None) -> bool:
    if not a or not b:
        return False
    dist = hamming64(int(a.get("phash") or 0), int(b.get("phash") or 0))
    corr = hist_corr(a.get("hist") or [], b.get("hist") or [])
    # Flat color fields share a near-empty DCT, so a hard pHash match alone
    # is not enough — the histogram must also agree.
    if dist <= PHASH_HARD and corr >= 0.70:
        return True
    if corr >= HIST_HARD and dist <= 22:
        return True
    if dist <= PHASH_SOFT and corr >= HIST_SOFT:
        return True
    if corr >= HIST_COMBO and dist <= 18:
        return True
    return False


def frames_similar(fa: np.ndarray, fb: np.ndarray) -> bool:
    return fingerprints_similar(fingerprint_frame(fa), fingerprint_frame(fb))


def ranges_overlap(a0: float, a1: float, b0: float, b1: float) -> bool:
    return (a0 < b1) and (a1 > b0)


def range_blocked(start: float, end: float, used: Iterable[tuple[float, float]], pad: float = RANGE_PAD) -> bool:
    """True if [start, end] overlaps any used [s-pad, e+pad]."""
    for s, e in used:
        if ranges_overlap(float(start), float(end), float(s) - pad, float(e) + pad):
            return True
    return False


def edl_used_ranges(edl: list[dict]) -> list[tuple[float, float]]:
    out = []
    for it in edl or []:
        if it.get("scene") in (None, "end"):
            continue
        a = float(it.get("src_start") or 0)
        vis = float(it.get("visible") or 0)
        b = a + vis
        if b > a:
            out.append((a, b))
    return out


def scene_fp(scene: dict) -> dict | None:
    if not scene:
        return None
    if scene.get("phash") or scene.get("hist"):
        return {"phash": int(scene.get("phash") or 0), "hist": scene.get("hist") or []}
    return None


def similar_to_used(scene: dict, used_scenes: list[dict]) -> bool:
    fp = scene_fp(scene)
    if not fp:
        return False
    for u in used_scenes:
        if fingerprints_similar(fp, scene_fp(u)):
            return True
    return False


def edl_duplicate_pairs(edl: list[dict], scenes_by_id: dict | None = None) -> list[dict]:
    """Return pairs of EDL items that violate range padding or look alike."""
    body = [it for it in (edl or []) if it.get("scene") not in (None, "end")]
    hits = []
    for i, a in enumerate(body):
        a0 = float(a.get("src_start") or 0)
        a1 = a0 + float(a.get("visible") or 0)
        sa = (scenes_by_id or {}).get(a.get("scene")) if scenes_by_id else None
        for b in body[i + 1:]:
            b0 = float(b.get("src_start") or 0)
            b1 = b0 + float(b.get("visible") or 0)
            reasons = []
            if ranges_overlap(a0, a1, b0 - RANGE_PAD, b1 + RANGE_PAD):
                reasons.append("range_pad")
            sb = (scenes_by_id or {}).get(b.get("scene")) if scenes_by_id else None
            if sa and sb and fingerprints_similar(scene_fp(sa), scene_fp(sb)):
                reasons.append("visual")
            # also compare fingerprints stored on the EDL items themselves
            if a.get("phash") and b.get("phash"):
                if fingerprints_similar(
                    {"phash": a.get("phash"), "hist": a.get("hist") or []},
                    {"phash": b.get("phash"), "hist": b.get("hist") or []},
                ):
                    reasons.append("visual")
            if reasons:
                hits.append({
                    "a": a.get("i"), "b": b.get("i"),
                    "a_src": [round(a0, 3), round(a1, 3)],
                    "b_src": [round(b0, 3), round(b1, 3)],
                    "reasons": list(dict.fromkeys(reasons)),
                })
    return hits


def shot_duplicate_pairs(fingerprints: list[dict], times: list[float] | None = None) -> list[dict]:
    """Compare a list of shot-midpoint fingerprints (output-side QC)."""
    hits = []
    for i in range(len(fingerprints)):
        for j in range(i + 1, len(fingerprints)):
            if fingerprints_similar(fingerprints[i], fingerprints[j]):
                hits.append({
                    "i": i, "j": j,
                    "t_i": None if not times else round(float(times[i]), 2),
                    "t_j": None if not times else round(float(times[j]), 2),
                })
    return hits
