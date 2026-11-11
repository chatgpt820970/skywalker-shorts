#!/usr/bin/env python3
"""Hard rejection of logo / dark / red / text / title / date cards.

There is NO fixed opening/ending time lock. Usable story footage may be
taken from anywhere in a trailer. A rejected subclip may be retried after
shifting +4.5s (auto-shift) if enough duration remains.

Each chosen segment is sampled at QC_FPS (4) and rejected if ANY frame
fails:

  - empty_dark: very low mean luma AND a high share of near-black pixels
    (short black flashes count; cinematic night footage with visible
    detail does not)
  - text / title / date overlay: stylized typography and readable text
    over live footage, dense text pages, large title glyphs, and
    "In theaters" / date-style cards — not only flat studio cards
  - existing flat text/title/logo cards, plus optional OCR when installed

A small persistent corner network bug (e.g. NETFLIX / FOOL NIGHT in the
top-right or top-left) is ignored and is never cropped out.
"""
from __future__ import annotations

import os
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

SHIFT_SEC = 4.5
DARK_LUMA = 25
DARK_FRAC = 0.45
QC_FPS = 4.0
HARD = (
    "empty_dark", "near_black",
    "text_card", "title_card", "title_logo", "text_overlay", "text_page",
    "date_card", "logo", "centered_logo", "logo_animation", "graphic_blob",
    "flat_card",
)
# "dark" / "red_solid" are computed for compatibility but are NOT hard
# rejects on their own (cinematic night / saturated lighting).

_OCR = None  # lazy


def _bgr(frame: np.ndarray) -> np.ndarray:
    a = np.asarray(frame)
    if a.ndim == 2:
        return cv2.cvtColor(a.astype(np.uint8), cv2.COLOR_GRAY2BGR)
    if a.shape[2] == 3:
        # incoming frames from the pipeline are RGB
        return cv2.cvtColor(a.astype(np.uint8), cv2.COLOR_RGB2BGR)
    return a.astype(np.uint8)


def _rgb(frame: np.ndarray) -> np.ndarray:
    a = np.asarray(frame)
    if a.ndim == 2:
        return np.stack([a, a, a], axis=2).astype(np.uint8)
    return a.astype(np.uint8)


def _resize(frame: np.ndarray, width: int = 480) -> np.ndarray:
    a = np.asarray(frame)
    if a.ndim == 2:
        a = np.stack([a, a, a], axis=2)
    h, w = a.shape[:2]
    if w != width:
        nh = max(2, int(round(h * width / w)))
        a = cv2.resize(a.astype(np.uint8), (width, nh), interpolation=cv2.INTER_AREA)
    return a


def _corner_ignore(h: int, w: int) -> np.ndarray:
    """Pixels to ignore: small top-corner network/title bugs. Do not crop."""
    m = np.zeros((h, w), dtype=bool)
    m[: int(h * 0.14), : int(w * 0.28)] = True   # top-left (FOOL NIGHT etc.)
    m[: int(h * 0.14), int(w * 0.78) :] = True  # top-right (NETFLIX etc.)
    m[int(h * 0.90) :, int(w * 0.84) :] = True  # tiny bottom-right bugs
    return m


def active_picture(frame: np.ndarray, luma_thr: float = 16.0,
                   min_bar: float = 0.04, max_bar: float = 0.28) -> np.ndarray:
    """Drop uniform black letterbox/pillarbox for MEASUREMENT only.

    Never used to crop the delivered frame. Source-baked bars inflate
    dark-pixel share and make a compact subject look like a centered logo.
    """
    a = np.asarray(frame)
    if a.size == 0:
        return a
    L = a.mean(axis=2) if a.ndim == 3 else a.astype(np.float32)
    h, w = L.shape
    row = L.mean(axis=1)
    col = L.mean(axis=0)
    top = 0
    while top < int(h * max_bar) and row[top] < luma_thr:
        top += 1
    bot = h - 1
    while bot > int(h * (1.0 - max_bar)) and row[bot] < luma_thr:
        bot -= 1
    left = 0
    while left < int(w * max_bar) and col[left] < luma_thr:
        left += 1
    right = w - 1
    while right > int(w * (1.0 - max_bar)) and col[right] < luma_thr:
        right -= 1
    bar_h = top + (h - 1 - bot)
    bar_w = left + (w - 1 - right)
    if bar_h < h * min_bar and bar_w < w * min_bar:
        return a
    if bot - top < 8 or right - left < 8:
        return a
    interior = L[top:bot + 1, left:right + 1]
    if float(interior.mean()) < float(L.mean()) + 4:
        return a
    return a[top:bot + 1, left:right + 1]


def is_dark(frame: np.ndarray) -> bool:
    """>45% of pixels have luma < 25. Cinematic night often trips this;
    do not use it alone as a hard reject. Prefer is_empty_dark()."""
    a = _resize(frame)
    if a.ndim == 3:
        L = a.mean(axis=2)
    else:
        L = a.astype(np.float32)
    return float((L < DARK_LUMA).mean()) > DARK_FRAC


def is_empty_dark(frame: np.ndarray) -> bool:
    """Near-fully-black filler / short black flash, not a dark story beat.

    Rejects when mean luma is very low AND almost every pixel is dark
    (e.g. a 0.3s fade-to-black between shots). Letterbox bars are ignored
    so a letterboxed night shot with a visible subject is kept.
    """
    a = _resize(active_picture(frame), 480)
    L = a.mean(axis=2) if a.ndim == 3 else a.astype(np.float32)
    mean = float(L.mean())
    dark16 = float((L < 16).mean())
    dark10 = float((L < 10).mean())
    if mean < 14 and dark16 >= 0.85:
        return True
    if mean < 10 and dark10 >= 0.75:
        return True
    if dark10 >= 0.95:
        return True
    return False


def is_red_or_solid(frame: np.ndarray) -> bool:
    """A single hue (often brand-red) or a nearly solid color field."""
    a = _resize(frame).astype(np.float32)
    L = a.mean(axis=2)
    # ignore near-black when judging "solid"
    mask = L >= 20
    if mask.mean() < 0.08:
        return False
    pix = a[mask]
    # low color variance => solid
    if float(pix.std()) < 12 and float((L < 28).mean()) > 0.35:
        return True
    bgr = cv2.cvtColor(a.astype(np.uint8), cv2.COLOR_RGB2BGR)
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    h, s, v = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]
    # red wraps 0 and 180
    red = ((h <= 10) | (h >= 170)) & (s >= 80) & (v >= 40)
    red_frac = float(red.mean())
    if red_frac >= 0.22 and float((L < 28).mean()) >= 0.40:
        return True
    # one hue bin dominates the non-dark pixels
    if mask.any():
        hh = h[mask]
        hist, _ = np.histogram(hh, bins=12, range=(0, 180))
        if hist.sum() and float(hist.max() / hist.sum()) >= 0.72 and float(s[mask].mean()) > 70:
            if float((L < 28).mean()) >= 0.35 or float(pix.std()) < 22:
                return True
    return False


def is_text_card(frame: np.ndarray) -> bool:
    """Flat outer region + sharp center (title / date / credit typography)."""
    a = _resize(active_picture(frame)).astype(np.float32)
    L = a.mean(axis=2)
    h, w = L.shape
    outer = np.ones((h, w), bool)
    outer[int(h * 0.22):int(h * 0.78), int(w * 0.18):int(w * 0.82)] = False
    inner = ~outer
    lap = cv2.Laplacian(L.astype(np.float32), cv2.CV_32F, ksize=3)
    # sharpness
    outer_sharp = float(lap[outer].var()) if outer.any() else 0.0
    inner_sharp = float(lap[inner].var()) if inner.any() else 0.0
    outer_flat = float(L[outer].std()) if outer.any() else 0.0
    # title cards: calm border, busy center letterforms
    if outer_flat < 18 and inner_sharp > outer_sharp * 2.2 and inner_sharp > 80:
        return True
    if outer_sharp < 40 and inner_sharp > 200 and float((L < 28).mean()) > 0.35:
        return True
    return False


def _ocr_words(frame: np.ndarray) -> list[str]:
    """Best-effort OCR. Returns [] if no engine is installed."""
    global _OCR
    if _OCR is False:
        return []
    a = _resize(frame, 640).astype(np.uint8)
    words: list[str] = []
    # pytesseract
    try:
        import pytesseract
        text = pytesseract.image_to_string(Image.fromarray(a), config="--psm 6") or ""
        words = [w.strip().lower() for w in text.split() if len(w.strip()) >= 3]
        _OCR = "pytesseract"
        return words
    except Exception:
        pass
    try:
        import easyocr
        if not isinstance(_OCR, easyocr.Reader):
            _OCR = easyocr.Reader(["en"], gpu=False, verbose=False)
        out = _OCR.readtext(a, detail=0) or []
        words = [str(w).strip().lower() for w in out if len(str(w).strip()) >= 3]
        return words
    except Exception:
        _OCR = False
        return []


_TITLE_WORDS = {
    "netflix", "coming", "soon", "only", "theaters", "theatres", "theater",
    "theatre", "rated", "pg-13", "pg13", "copyright", "warner", "lionsgate",
    "marvel", "disney", "universal", "presents", "produced", "subscribe",
    "january", "february", "march", "april", "may", "june", "july",
    "august", "september", "october", "november", "december",
    "jan", "feb", "mar", "apr", "jun", "jul", "aug", "sep", "sept",
    "oct", "nov", "dec",
    "trailer", "official", "exclusively", "shaft", "bandai", "namco",
    "a24", "in",  # "in" alone is too short; checked only as phrase below
}

# Corner-bug labels we never treat as a title hit by themselves.
_CORNER_BUGS = {"netflix", "fool", "night", "disney+", "disney", "hulu", "max"}

_DATE_WORDS = {
    "january", "february", "march", "april", "june", "july", "august",
    "september", "october", "november", "december",
    "jan", "feb", "mar", "apr", "jun", "jul", "aug", "sep", "sept",
    "oct", "nov", "dec", "theaters", "theatres", "theater", "theatre",
}


def ocr_title_hit(frame: np.ndarray) -> tuple[bool, str]:
    """Return (hit, reason) from optional OCR. reason is title_card or date_card."""
    words = _ocr_words(frame)
    if not words:
        return False, ""
    cleaned = [w.strip(".,!?:;").lower() for w in words]
    # ignore a lone corner-network word
    meaningful = [w for w in cleaned if w not in _CORNER_BUGS and len(w) >= 3]
    if any(w in _DATE_WORDS for w in meaningful):
        return True, "date_card"
    if any(w in (_TITLE_WORDS - {"in", "may"}) for w in meaningful):
        return True, "title_card"
    # significant readable typography: several mid-length words (not the bug)
    if len(meaningful) >= 5 and sum(len(w) for w in meaningful) >= 24:
        return True, "text_overlay"
    return False, ""


def _typo_residual(rgb: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Luma morphological residual (dark-on-light and light-on-dark type)."""
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    hats = []
    for k in (7, 9, 13, 25, 41):
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
        opened = cv2.morphologyEx(gray, cv2.MORPH_OPEN, kernel)
        closed = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, kernel)
        hats.append(cv2.subtract(closed, gray))
        hats.append(cv2.subtract(gray, opened))
    return gray, np.max(np.stack(hats, 0), 0)


def _letter_score(ink: np.ndarray) -> tuple[int, float, float, bool]:
    """Column-run letter count, coverage, and vertical band fraction."""
    h, w = ink.shape
    if h < 6 or w < 12:
        return 0, 0.0, 0.0, False
    row = (ink > 0).mean(axis=1)
    if row.max() < 0.08:
        return 0, 0.0, 0.0, False
    ys = np.where(row >= max(0.12, 0.25 * row.max()))[0]
    if len(ys) == 0:
        return 0, 0.0, 0.0, False
    band = (ys[-1] - ys[0] + 1) / h
    col = (ink > 0).mean(axis=0)
    thr = max(0.10, 0.35 * col.max())
    on = col >= thr
    n = 0
    i = 0
    while i < w:
        if on[i]:
            j = i
            while j < w and on[j]:
                j += 1
            n += 1
            i = j
        else:
            i += 1
    cov = float((ink > 0).mean())
    return n, cov, float(band), True


def _glyph_count_in_box(gray_roi: np.ndarray, res_roi: np.ndarray) -> int:
    """Letter-sized components on a baseline inside a candidate text line."""
    h, w = gray_roi.shape
    best = 0
    blur = cv2.GaussianBlur(gray_roi, (3, 3), 0)
    _, m0 = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    for m in (m0, 255 - m0):
        if float((m > 0).mean()) > 0.5:
            continue
        k = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
        m = cv2.morphologyEx(m, cv2.MORPH_OPEN, k)
        n, _lab, st, _ = cv2.connectedComponentsWithStats((m > 0).astype(np.uint8), 8)
        gs = 0
        for i in range(1, n):
            _x, _y, bw, bh, area = st[i]
            if bh < 0.40 * h or bh > 1.2 * h or bw < 3:
                continue
            ar = bw / max(bh, 1)
            if 0.12 <= ar <= 2.8 and area >= 12:
                gs += 1
        best = max(best, gs)
    for t in (40, 55, 80):
        m = (res_roi >= t).astype(np.uint8)
        k = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
        m = cv2.morphologyEx(m, cv2.MORPH_OPEN, k)
        n, _lab, st, _ = cv2.connectedComponentsWithStats(m, 8)
        gs = 0
        for i in range(1, n):
            _x, _y, bw, bh, area = st[i]
            if bh < 0.45 * h or bh > 1.15 * h or bw < 3:
                continue
            ar = bw / max(bh, 1)
            if 0.12 <= ar <= 2.8 and area >= 12:
                gs += 1
        best = max(best, gs)
    return best


def _residual_text_lines(gray: np.ndarray, res: np.ndarray, ign: np.ndarray) -> list[dict]:
    """Typography lines from luma residual, verified by in-box glyph count."""
    H, W = gray.shape
    hits = []
    seen: set[tuple[int, int, int, int]] = set()
    for thr in (16, 22, 28, 36, 55, 80, 110):
        raw = ((res >= thr) & (~ign)).astype(np.uint8) * 255
        k = cv2.getStructuringElement(cv2.MORPH_RECT, (max(11, W // 36), 2))
        closed = cv2.morphologyEx(raw, cv2.MORPH_CLOSE, k)
        n, _lab, st, _ = cv2.connectedComponentsWithStats(closed, 8)
        for i in range(1, n):
            x, y, bw, bh, area = st[i]
            afrac = area / (H * W)
            if afrac < 0.003 or bh < 8 or bw < 10 or afrac > 0.35:
                continue
            ar = bw / max(bh, 1)
            hfrac = bh / H
            wfrac = bw / W
            if hfrac < 0.035 or ar > 10.5:
                continue
            # body/ground edges along the very bottom of the plate
            if y > 0.78 * H and hfrac < 0.12:
                continue
            if not (ar >= 2.0 and hfrac <= 0.24 and wfrac >= 0.16):
                continue
            key = (int(x) // 4, int(y) // 4, int(bw) // 4, int(bh) // 4)
            if key in seen:
                continue
            ink = raw[y:y + bh, x:x + bw]
            nrun, cov, band, ok = _letter_score(ink)
            if not ok or nrun < 3 or not (0.10 <= cov <= 0.55) or band < 0.40:
                continue
            glyphs = _glyph_count_in_box(gray[y:y + bh, x:x + bw], res[y:y + bh, x:x + bw])
            # require real letterforms, not ropes / arms / foliage.
            # Thin serif captions can fragment into fewer boxes; a high
            # column-run count still means readable type.
            if glyphs < 4 and not (glyphs >= 3 and nrun >= 6):
                continue
            seen.add(key)
            hits.append({
                "box": (int(x), int(y), int(bw), int(bh)),
                "glyphs": int(glyphs), "thr": int(thr),
            })
    return hits


def _sat_glyph(rgb: np.ndarray, ign: np.ndarray) -> list[dict]:
    """Large solid typographic mark (e.g. a teal title 'M')."""
    H, W = rgb.shape[:2]
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    s, v = hsv[:, :, 1], hsv[:, :, 2]
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    mask = ((s >= 90) & (v > 70) & (v < 230) & (~ign)).astype(np.uint8) * 255
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 7))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k)
    n, lab, st, _ = cv2.connectedComponentsWithStats(mask, 8)
    hits = []
    for i in range(1, n):
        x, y, bw, bh, area = st[i]
        afrac = area / (H * W)
        ar = bw / max(bh, 1)
        hfrac = bh / H
        wfrac = bw / W
        fill = area / max(bw * bh, 1)
        if not (
            0.006 <= afrac <= 0.08
            and 0.16 <= hfrac <= 0.48
            and 0.08 <= wfrac <= 0.40
            and 0.25 <= ar <= 1.15
            and 0.35 <= fill <= 0.75
        ):
            continue
        m = lab == i
        dil = cv2.dilate(m.astype(np.uint8), np.ones((5, 5), np.uint8), iterations=2)
        ring = (dil > 0) & (~m) & (~ign)
        if not ring.any():
            continue
        contrast = abs(float(gray[m].mean()) - float(gray[ring].mean()))
        sat = float(s[m].mean())
        if (sat >= 100 and contrast >= 50) or (sat >= 80 and contrast >= 70 and fill >= 0.40):
            hits.append({
                "box": (int(x), int(y), int(bw), int(bh)),
                "afrac": round(float(afrac), 4),
                "sat": round(sat, 0),
                "contrast": round(contrast, 1),
            })
    return hits


def _sat_text_lines(rgb: np.ndarray, ign: np.ndarray) -> list[dict]:
    """Colored typography over footage: same-ink letters sharing a baseline."""
    H, W = rgb.shape[:2]
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    hue, s, v = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]
    hits = []
    seen: set[tuple[int, int, int, int]] = set()
    for sat_min in (90, 110):
        mask = ((s >= sat_min) & (v > 60) & (v < 230) & (~ign)).astype(np.uint8) * 255
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k)
        n, lab, st, _ = cv2.connectedComponentsWithStats(mask, 8)
        comps = []
        for i in range(1, n):
            x, y, bw, bh, area = st[i]
            if area < 20 or bh < 8 or bh > int(H * 0.28):
                continue
            ar = bw / max(bh, 1)
            if ar > 4.5 or ar < 0.12:
                continue
            m = lab == i
            comps.append({
                "x": int(x), "y": int(y), "bw": int(bw), "bh": int(bh),
                "base": y + bh, "hue": float(hue[m].mean()),
                "sat": float(s[m].mean()),
            })
        comps.sort(key=lambda c: c["base"])
        used = [False] * len(comps)
        for i, c in enumerate(comps):
            if used[i]:
                continue
            group = [c]
            used[i] = True
            for j, d in enumerate(comps):
                if used[j]:
                    continue
                if abs(d["base"] - c["base"]) <= max(8, 0.45 * c["bh"]) and abs(d["bh"] - c["bh"]) <= 0.5 * c["bh"]:
                    dh = min(abs(d["hue"] - c["hue"]), 180 - abs(d["hue"] - c["hue"]))
                    if dh <= 18 and abs(d["sat"] - c["sat"]) <= 50:
                        group.append(d)
                        used[j] = True
            if len(group) < 6:
                used[i] = False
                continue
            xs = min(g["x"] for g in group)
            xe = max(g["x"] + g["bw"] for g in group)
            ys = min(g["y"] for g in group)
            ye = max(g["y"] + g["bh"] for g in group)
            span = (xe - xs) / W
            hfrac = (ye - ys) / H
            hues = np.array([g["hue"] for g in group], dtype=np.float32)
            sats = np.array([g["sat"] for g in group], dtype=np.float32)
            pack = sum(g["bw"] for g in group) / max(xe - xs, 1)
            # Tight, same-ink title line — not a row of similarly-colored crowd clothes.
            if (
                len(group) >= 6
                and 0.16 <= span <= 0.65
                and 0.07 <= hfrac <= 0.18
                and float(hues.std()) <= 1.2
                and float(sats.mean()) >= 115
                and pack >= 0.35
            ):
                key = (xs // 4, ys // 4, (xe - xs) // 4, (ye - ys) // 4)
                if key in seen:
                    continue
                seen.add(key)
                hits.append({
                    "box": (int(xs), int(ys), int(xe - xs), int(ye - ys)),
                    "n": len(group),
                    "span": round(float(span), 3),
                    "sat": round(float(sats.mean()), 1),
                })
    return hits


def _is_text_page(gray: np.ndarray, ign: np.ndarray) -> bool:
    """Dense typeset page (e.g. a Japanese document close-up)."""
    H, W = gray.shape
    g = cv2.GaussianBlur(gray, (3, 3), 0)
    for inv in (cv2.THRESH_BINARY_INV, cv2.THRESH_BINARY):
        bw = cv2.adaptiveThreshold(
            g, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, inv, 25, 8,
        )
        bw[ign] = 0
        n, _lab, st, _ = cv2.connectedComponentsWithStats((bw > 0).astype(np.uint8), 8)
        glyphs = []
        for i in range(1, n):
            x, y, bw_, bh, area = st[i]
            if area < 15 or bh < 6 or bw_ < 3:
                continue
            if bh > int(H * 0.16) or bw_ > int(W * 0.20):
                continue
            ar = bw_ / max(bh, 1)
            if 0.15 <= ar <= 3.5:
                glyphs.append((x, y, bw_, bh, area))
        if len(glyphs) < 28:
            continue
        heights = np.array([g_[3] for g_ in glyphs], dtype=np.float32)
        medh = float(np.median(heights))
        if medh < 6:
            continue
        similar = sum(1 for h in heights if 0.55 * medh <= h <= 1.6 * medh)
        if similar < 24:
            continue
        bases = np.array([g_[1] + g_[3] for g_ in glyphs], dtype=np.float32)
        order = np.argsort(bases)
        rows: list[list] = []
        cur = [glyphs[order[0]]]
        last = bases[order[0]]
        for idx in order[1:]:
            b = bases[idx]
            if b - last > medh * 0.75:
                rows.append(cur)
                cur = [glyphs[idx]]
            else:
                cur.append(glyphs[idx])
            last = b
        rows.append(cur)
        good = 0
        wide = 0
        for r in rows:
            if len(r) < 6:
                continue
            span = (max(c[0] + c[2] for c in r) - min(c[0] for c in r)) / W
            hs = [c[3] for c in r]
            if span >= 0.28 and (np.std(hs) / max(np.mean(hs), 1)) <= 0.45:
                good += 1
            if span >= 0.50 and len(r) >= 8:
                wide += 1
        area = sum(c[4] for c in glyphs) / (H * W)
        if good >= 5 and area >= 0.03:
            return True
        # close-up document (hand-marked Japanese page): many full-width
        # lines even if glyph heights are a bit irregular
        # Close-up document (hand-marked Japanese page): a few very wide
        # lines of relatively tall glyphs. Park foliage has many tiny
        # "glyphs" (low median height) and must not match.
        if wide >= 3 and area >= 0.10 and similar >= 40 and medh >= 14:
            return True
    return False


def _baseline_glyph_lines(gray: np.ndarray, ign: np.ndarray) -> list[dict]:
    """One or two typographic lines (title / date / 'A R RAHMAN MUSICAL').

    Clusters letter-sized components on a shared baseline. Works on gold or
    white type over a busy or dark field, where residual-close can merge the
    whole card into one blob.
    """
    H, W = gray.shape
    hits = []
    blur = cv2.GaussianBlur(gray, (3, 3), 0)
    _, otsu = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    for m in (otsu, 255 - otsu):
        ink = (m > 0) & (~ign)
        if not (0.01 <= float(ink.mean()) <= 0.45):
            continue
        n, _lab, st, _ = cv2.connectedComponentsWithStats(ink.astype(np.uint8), 8)
        glyphs = []
        for i in range(1, n):
            x, y, bw, bh, area = st[i]
            if area < 18 or bh < 8 or bw < 3:
                continue
            if bh > int(H * 0.28) or bw > int(W * 0.45):
                continue
            ar = bw / max(bh, 1)
            if 0.12 <= ar <= 3.2:
                glyphs.append((x, y, bw, bh, area))
        if len(glyphs) < 4:
            continue
        bases = np.array([g[1] + g[3] for g in glyphs], dtype=np.float32)
        heights = np.array([g[3] for g in glyphs], dtype=np.float32)
        order = np.argsort(bases)
        rows: list[list] = []
        cur = [glyphs[order[0]]]
        last = bases[order[0]]
        medh = float(np.median(heights))
        for idx in order[1:]:
            b = bases[idx]
            if b - last > max(medh * 0.70, 8):
                rows.append(cur)
                cur = [glyphs[idx]]
            else:
                cur.append(glyphs[idx])
            last = b
        rows.append(cur)
        for r in rows:
            if not (6 <= len(r) <= 18):
                continue
            hs = np.array([c[3] for c in r], dtype=np.float32)
            if float(np.std(hs) / max(float(np.mean(hs)), 1)) > 0.40:
                continue
            xs = min(c[0] for c in r)
            xe = max(c[0] + c[2] for c in r)
            span = (xe - xs) / W
            ys = min(c[1] for c in r)
            ye = max(c[1] + c[3] for c in r)
            hfrac = (ye - ys) / H
            pack = sum(c[2] for c in r) / max(xe - xs, 1)
            # single typographic baseline, not a band of hair / windows / crowd heads
            if (ye - ys) > 2.1 * float(np.median(hs)):
                continue
            if span >= 0.28 and 0.035 <= hfrac <= 0.20 and pack >= 0.32:
                hits.append({
                    "box": (int(xs), int(ys), int(xe - xs), int(ye - ys)),
                    "n": len(r),
                    "span": round(float(span), 3),
                })
    return hits



def _bright_serif_lines(gray: np.ndarray, ign: np.ndarray) -> list[dict]:
    """Thin light-on-dark serif captions (e.g. 'In exchange for 10 million yen').

    Residual-close can miss low-contrast serifs; bright ink on a darker
    field still forms a long baseline of similar-height glyphs.
    """
    H, W = gray.shape
    hits = []
    # Unsharp residual boosts thin strokes without needing a huge hat kernel.
    blur = cv2.GaussianBlur(gray, (0, 0), 1.15)
    boost = cv2.subtract(gray, blur)
    ink = ((gray >= 168) | (boost >= 18)) & (gray >= 140) & (~ign)
    if not (0.004 <= float(ink.mean()) <= 0.28):
        return hits
    n, _lab, st, _ = cv2.connectedComponentsWithStats(ink.astype(np.uint8), 8)
    glyphs = []
    for i in range(1, n):
        x, y, bw, bh, area = st[i]
        if area < 10 or bh < 5 or bw < 2:
            continue
        if bh > int(H * 0.16) or bw > int(W * 0.40):
            continue
        ar = bw / max(bh, 1)
        if 0.10 <= ar <= 3.6:
            glyphs.append((x, y, bw, bh, area))
    if len(glyphs) < 8:
        return hits
    bases = np.array([g[1] + g[3] for g in glyphs], dtype=np.float32)
    heights = np.array([g[3] for g in glyphs], dtype=np.float32)
    order = np.argsort(bases)
    rows: list[list] = []
    cur = [glyphs[order[0]]]
    last = bases[order[0]]
    medh = float(np.median(heights))
    for idx in order[1:]:
        b = bases[idx]
        if b - last > max(medh * 0.70, 6):
            rows.append(cur)
            cur = [glyphs[idx]]
        else:
            cur.append(glyphs[idx])
        last = b
    rows.append(cur)
    for r in rows:
        if len(r) < 8:
            continue
        hs = np.array([c[3] for c in r], dtype=np.float32)
        if float(np.std(hs) / max(float(np.mean(hs)), 1)) > 0.48:
            continue
        xs = min(c[0] for c in r)
        xe = max(c[0] + c[2] for c in r)
        span = (xe - xs) / W
        ys = min(c[1] for c in r)
        ye = max(c[1] + c[3] for c in r)
        hfrac = (ye - ys) / H
        pack = sum(c[2] for c in r) / max(xe - xs, 1)
        if (ye - ys) > 2.4 * float(np.median(hs)):
            continue
        # Long thin caption line — not a speckled foliage band.
        if span >= 0.34 and 0.018 <= hfrac <= 0.14 and pack >= 0.28:
            hits.append({
                "box": (int(xs), int(ys), int(xe - xs), int(ye - ys)),
                "n": len(r),
                "span": round(float(span), 3),
            })
    return hits


def clean_intervals(start: float, end: float, samples: list[tuple[float, bool]],
                    min_dur: float = 0.55) -> list[tuple[float, float]]:
    """Keep only the clean stretches of a shot.

    `samples` is [(t, is_dirty), ...] inside [start, end]. Each sample owns
    the half-gap to its neighbors, so a logo flash drops a slice instead of
    the whole scene-detect cut.
    """
    start, end = float(start), float(end)
    if end - start < float(min_dur):
        if samples and not any(d for _t, d in samples):
            return [(start, end)]
        return []
    if not samples:
        return [(start, end)]
    samples = sorted(((float(t), bool(d)) for t, d in samples), key=lambda x: x[0])
    bounds = [start]
    for i in range(len(samples) - 1):
        mid = 0.5 * (samples[i][0] + samples[i + 1][0])
        bounds.append(min(end, max(start, mid)))
    bounds.append(end)
    runs: list[tuple[float, float]] = []
    run_a = None
    run_b = start
    for i, (_t, dirty) in enumerate(samples):
        a = max(start, min(bounds[i], end))
        b = max(start, min(bounds[i + 1], end))
        if b <= a + 1e-4:
            continue
        if not dirty:
            if run_a is None:
                run_a = a
            run_b = b
        elif run_a is not None:
            if run_b - run_a >= float(min_dur):
                runs.append((round(run_a, 3), round(run_b, 3)))
            run_a = None
    if run_a is not None and run_b - run_a >= float(min_dur):
        runs.append((round(run_a, 3), round(run_b, 3)))
    return runs


def detect_overlay_text(frame: np.ndarray) -> list[str]:
    """Stylized titles, date cards, and significant readable text over footage.

    Small corner network bugs are ignored. Anime linework, faces, buildings,
    and crowds should not trip this.
    """
    rgb = _resize(_rgb(frame), 640)
    H, W = rgb.shape[:2]
    ign = _corner_ignore(H, W)
    gray, res = _typo_residual(rgb)
    reasons: list[str] = []
    if _residual_text_lines(gray, res, ign):
        reasons.append("text_overlay")
    if _sat_text_lines(rgb, ign):
        if "text_overlay" not in reasons:
            reasons.append("text_overlay")
    if _sat_glyph(rgb, ign):
        reasons.append("title_logo")
    if _is_text_page(gray, ign):
        reasons.append("text_page")
    if _baseline_glyph_lines(gray, ign):
        if "text_overlay" not in reasons:
            reasons.append("text_overlay")
    if _bright_serif_lines(gray, ign):
        if "text_overlay" not in reasons:
            reasons.append("text_overlay")
    return list(dict.fromkeys(reasons))


def detect_frame(frame: np.ndarray) -> list[str]:
    """Return rejection reasons for a single frame (may be empty).

    Hard: empty_dark (black/fade filler), text_overlay (any title / logo
    wordmark / date / rating / credits typography, via the PP-OCRv3 DB
    detector in text_det.py), OCR date/title words when an OCR engine exists.
    Soft (never a reject on their own): dark, red_solid.
    """
    reasons: list[str] = []
    if is_empty_dark(frame):
        reasons.append("empty_dark")
    # keep loose "dark" as a soft flag only (not a hard reject on its own)
    if is_dark(frame) and "empty_dark" not in reasons:
        reasons.append("dark")
    if is_red_or_solid(frame):
        reasons.append("red_solid")
    try:
        import text_det
        hit, _info = text_det.has_overlay_text(_rgb(active_picture(frame)))
        if hit:
            reasons.append("text_overlay")
    except Exception as e:  # model missing must never silently pass text
        print(f"  text_det unavailable: {e}")
        reasons.extend(detect_overlay_text(frame))
    try:
        hit, why = ocr_title_hit(frame)
        if hit and why:
            reasons.append(why)
    except Exception:
        pass
    return list(dict.fromkeys(reasons))


def _video_fps(cap: cv2.VideoCapture) -> float:
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    return fps if fps > 1.0 else 30.0


def grab_times(path: Path, times: list[float]) -> list[tuple[float, np.ndarray]]:
    """Frame-accurate RGB grabs. Seek a little early, then read forward.

    H.264 cap.set(POS_MSEC) lands on the previous keyframe, so typography
    flashes and title cards were being missed at score time.
    """
    if not times:
        return []
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        return []
    fps = _video_fps(cap)
    want = sorted({max(0.0, float(t)) for t in times})
    first = want[0]
    start_f = max(0, int(round(first * fps)) - int(round(fps * 0.85)) - 1)
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_f)
    out: list[tuple[float, np.ndarray]] = []
    # After a seek, POS_FRAMES is the index of the *next* frame that will be read.
    safety = 0
    max_reads = int((want[-1] - start_f / fps) * fps) + int(fps * 2) + 40
    while want and safety < max_reads:
        ok, fr = cap.read()
        safety += 1
        if not ok or fr is None:
            break
        nxt = cap.get(cv2.CAP_PROP_POS_FRAMES)
        t = (float(nxt) - 1.0) / fps if nxt else (start_f + safety - 1) / fps
        # Assign every due sample that this decoded frame covers.
        while want and want[0] <= t + 0.51 / fps:
            out.append((want.pop(0), cv2.cvtColor(fr, cv2.COLOR_BGR2RGB)))
    cap.release()
    return out


def _grab(path: Path, t: float) -> np.ndarray | None:
    got = grab_times(path, [t])
    return got[0][1] if got else None


def sample_times(start: float, end: float, fps: float = QC_FPS) -> list[float]:
    """Timestamps at `fps` inside [start, end], always including both ends."""
    dur = max(0.01, end - start)
    n = max(3, int(np.ceil(dur * fps)) + 1)
    times = [start + 0.02]
    if n > 2:
        times += [start + dur * i / (n - 1) for i in range(1, n - 1)]
    times.append(max(start, end - 0.02))
    out = []
    for t in times:
        t = float(min(max(start, t), max(start, end - 0.001)))
        rt = round(t, 3)
        if not out or abs(rt - out[-1]) >= 0.04:
            out.append(rt)
    return out


def inspect_clip(path: Path, start: float, end: float) -> tuple[bool, list[str], dict]:
    """4 fps check of a segment. ok=False if ANY sampled frame fails a hard rule."""
    times = sample_times(start, end, QC_FPS)
    hits: list[str] = []
    n_dark = 0
    sampled = 0
    hard = set(HARD)
    grabbed = {round(t, 3): fr for t, fr in grab_times(path, times)}
    for t in times:
        fr = grabbed.get(round(t, 3))
        if fr is None:
            # closest decoded sample
            if grabbed:
                fr = min(grabbed.items(), key=lambda kv: abs(kv[0] - t))[1]
            else:
                fr = _grab(path, t)
        if fr is None:
            continue
        sampled += 1
        rs = detect_frame(fr)
        if "empty_dark" in rs or "near_black" in rs:
            n_dark += 1
        for r in rs:
            if r in hard:
                hits.append(r)
    if hits:
        return False, list(dict.fromkeys(hits)), {"sampled": sampled, "dark": n_dark, "fps": QC_FPS}
    return True, [], {"sampled": sampled, "dark": n_dark, "fps": QC_FPS}


def auto_shift(start: float, end: float, trailer_dur: float, delta: float = SHIFT_SEC) -> tuple[float, float] | None:
    """Shift a rejected window +delta seconds. None if not enough room left."""
    ns, ne = start + delta, end + delta
    if ne > trailer_dur:
        # keep the same length if possible, otherwise shrink to the tail
        length = end - start
        ns = min(start + delta, max(0.0, trailer_dur - 0.8))
        ne = min(trailer_dur, ns + length)
    if ne - ns < 0.70:
        return None
    if abs(ns - start) < 0.2:
        return None
    return ns, ne


def sharpness(frame: np.ndarray) -> float:
    a = _resize(frame)
    L = a.mean(axis=2).astype(np.float32) if a.ndim == 3 else a.astype(np.float32)
    return float(cv2.Laplacian(L, cv2.CV_32F, ksize=3).var())


def brightness_score(luma: float) -> float:
    """Prefer mid-bright story frames; penalise near-black and blown-out."""
    # peak around luma 90-130
    return float(max(0.0, 1.0 - abs(luma - 110.0) / 110.0))


def visual_interest(frame: np.ndarray) -> float:
    a = _resize(frame).astype(np.float32)
    L = a.mean(axis=2)
    contrast = float(L.std())
    # unique coarse colors
    q = (a // 32).astype(np.int32)
    uniq = len(np.unique(q[:, :, 0] * 64 + q[:, :, 1] * 8 + q[:, :, 2]))
    gy = float(np.abs(np.diff(L, axis=0)).mean())
    gx = float(np.abs(np.diff(L, axis=1)).mean())
    return contrast * 0.55 + min(uniq, 120) * 0.25 + (gx + gy) * 0.8


_YUNET = None  # cached FaceDetectorYN
_YUNET_WH = None

def face_or_character_score(frame: np.ndarray, yunet: Path | None = None) -> float:
    """0..1-ish. YuNet faces when the model is present, else a skin-tone fallback."""
    a = _resize(frame, 640).astype(np.uint8)
    h, w = a.shape[:2]
    model = yunet
    if model is None:
        env = os.environ.get("SKY_AE")
        if env:
            cand = Path(env) / "models" / "face_detection_yunet_2023mar.onnx"
            if cand.exists():
                model = cand
        if model is None:
            # repo-relative default
            cand = Path(__file__).resolve().parent / "models" / "face_detection_yunet_2023mar.onnx"
            if cand.exists():
                model = cand
    if model and Path(model).exists():
        try:
            global _YUNET, _YUNET_WH
            if _YUNET is None:
                _YUNET = cv2.FaceDetectorYN.create(str(model), "", (w, h), 0.6, 0.3, 5000)
                _YUNET_WH = (w, h)
            det = _YUNET
            if _YUNET_WH != (w, h):
                det.setInputSize((w, h))
                _YUNET_WH = (w, h)
            _, faces = det.detect(cv2.cvtColor(a, cv2.COLOR_RGB2BGR))
            if faces is not None and len(faces):
                # area of the largest face, plus a bonus per extra face
                areas = []
                for f in faces:
                    areas.append(max(0.0, float(f[2] * f[3]) / (w * h)))
                return min(1.5, max(areas) * 6.0 + 0.25 * (len(faces) - 1))
        except Exception:
            pass
    # skin-tone fallback (works for live-action; weaker on stylised anime)
    r, g, b = a[:, :, 0], a[:, :, 1], a[:, :, 2]
    skin = ((r > 80) & (r > g) & (g > b * 0.7) & (r.astype(int) - b.astype(int) > 15)).mean()
    return float(skin)
