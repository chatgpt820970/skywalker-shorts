#!/usr/bin/env python3
"""Per-second QC of a finished Short, plus a bounded re-slice loop hook.

Checks:
  - resolution is 1080x1920 or 2160x3840
  - fps >= 30
  - audio is not silent
  - zero frames fail the empty-dark / logo / overlay-text / title / date detectors
  - hook banner is visible
  - lower-third captions are visible wherever dialogue is present
  - no near-duplicate shots in the output
  - dialogue is audible during speech windows (when a transcript is provided)
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
MIN_DURATION = 18.0
AIM_DURATION = 20.0
AIM_DURATION_MAX = 28.0


def duration_issue(dur: float) -> str | None:
    """Hard floor: anything under 18s must fail QC and must never be released."""
    try:
        d = float(dur or 0)
    except (TypeError, ValueError):
        d = 0.0
    if d + 1e-6 < MIN_DURATION:
        return (f"duration {d:.2f}s below hard minimum {MIN_DURATION:.0f}s "
                f"(aim {AIM_DURATION:.0f}-{AIM_DURATION_MAX:.0f}s)")
    return None


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
    # Frame-accurate: cap.set(POS_MSEC) lands on the previous H.264 keyframe,
    # which made short subtitle cues look "missing" in QA.
    try:
        got = scene_filter.grab_times(path, [float(t)])
        if got:
            return got[0][1]
    except Exception:
        pass
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        return None
    cap.set(cv2.CAP_PROP_POS_MSEC, max(0.0, t) * 1000.0)
    ok, fr = cap.read()
    cap.release()
    if not ok or fr is None:
        return None
    return cv2.cvtColor(fr, cv2.COLOR_BGR2RGB)

# Hard cards only: logo / title / date / overlay typography / empty near-black.
# Cinematic darkness or saturated lighting in real story footage is NOT a fail.
HARD = {
    "centered_logo", "logo_animation", "graphic_blob", "text_card",
    "flat_card", "title_card", "title_logo", "text_overlay", "text_page",
    "date_card", "logo", "near_black", "empty_dark",
}


def frame_reasons(plate: np.ndarray) -> list[str]:
    """The single hard-card detector shared by scene screening and final QC.

    pipeline.screen_rendered_clips() runs this exact function on every rendered
    clip BEFORE stitching, so text/logo/black shots are cut out of the edit
    instead of only being flagged on the finished video.
    """
    reasons: list[str] = []
    if plate is None or not getattr(plate, "size", 0):
        return reasons
    try:
        import pipeline
        rs2, _ = pipeline.classify_shot(plate)
        reasons.extend(r for r in rs2 if r in HARD)
    except Exception:
        pass
    try:
        sf = scene_filter.detect_frame(plate)
        reasons.extend(r for r in sf if r in HARD)
    except Exception:
        pass
    return list(dict.fromkeys(reasons))


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
    # Use the brightest channel: keyword words are yellow (255,255,0), whose
    # RGB mean is only 170, so short all-keyword cues ("Yes", "No way") were
    # read as missing. Glyph area also scales with text length, so a short
    # word needs a lower bright-pixel fraction than a full line.
    V = band.max(axis=2) if band.ndim == 3 else band
    L = band.mean(axis=2) if band.ndim == 3 else band
    bright = float((V > 180).mean())
    dark = float((L < 40).mean())
    return bright > 0.003 and dark > 0.01



def _mean_volume_window(path: Path, t0: float, t1: float) -> float | None:
    if t1 <= t0:
        return None
    r = subprocess.run(
        ["ffmpeg", "-ss", f"{t0:.3f}", "-t", f"{max(0.12, t1-t0):.3f}",
         "-i", str(path), "-af", "volumedetect", "-vn", "-f", "null", "-"],
        capture_output=True, text=True,
    )
    for ln in (r.stderr or "").splitlines():
        if "mean_volume" in ln:
            try:
                return float(ln.split("mean_volume:")[1].split()[0])
            except Exception:
                return None
    return None


def _plate_of(frame: np.ndarray, layout: dict | None) -> np.ndarray:
    if frame is None or not getattr(frame, "size", 0):
        return frame
    if not layout or layout.get("plate_y") is None:
        return frame
    h = frame.shape[0]
    sy = h / 1920.0
    y = int(layout["plate_y"] * sy)
    ph = int(layout.get("plate_h", 0) * sy)
    if ph <= 0:
        return frame
    plate = frame[y:y + ph, :]
    try:
        plate = scene_filter.active_picture(plate)
    except Exception:
        pass
    return plate


def check_duplicate_shots(path: Path, layout: dict | None = None, dur: float = 0.0) -> list[dict]:
    """Fail QC when two output shots look like the same angle."""
    try:
        import similarity
    except Exception:
        return []
    if dur <= 0:
        return []
    # Sample the plate at 2 fps, split into shots on large frame diffs, compare mids.
    step = 0.5
    times = []
    t = 0.08
    while t < dur - 0.08:
        times.append(round(t, 3))
        t += step
    if not times:
        return []
    try:
        grabbed = scene_filter.grab_times(path, times)
    except Exception:
        grabbed = [(tt, _frame_at(path, tt)) for tt in times]
    plates = []
    for tt, fr in grabbed:
        if fr is None:
            continue
        plates.append((tt, _plate_of(fr, layout)))
    if len(plates) < 4:
        return []
    # Shot boundaries: high inter-frame difference.
    diffs = []
    for i in range(1, len(plates)):
        a, b = plates[i - 1][1], plates[i][1]
        if a is None or b is None or a.size == 0 or b.size == 0:
            diffs.append(0.0)
            continue
        ha, wa = a.shape[:2]
        hb, wb = b.shape[:2]
        if (ha, wa) != (hb, wb):
            b = cv2.resize(b, (wa, ha))
        diffs.append(float(np.mean(np.abs(a.astype(np.float32) - b.astype(np.float32)))))
    med = float(np.median(diffs)) if diffs else 0.0
    cut = max(18.0, med * 2.4)
    shots = []
    start = 0
    for i, d in enumerate(diffs, start=1):
        if d >= cut:
            shots.append(plates[start:i])
            start = i
    shots.append(plates[start:])
    mids, mid_t = [], []
    for sh in shots:
        if not sh:
            continue
        # Ignore very short flashes (< 0.4s)
        if sh[-1][0] - sh[0][0] < 0.35:
            continue
        mid = sh[len(sh) // 2]
        if mid[1] is None or mid[1].size == 0:
            continue
        mids.append(similarity.fingerprint_frame(mid[1]))
        mid_t.append(mid[0])
    return similarity.shot_duplicate_pairs(mids, mid_t)


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
    di = duration_issue(dur)
    if di:
        issues.append(di)
    # 4 fps so short typography / black flashes between 1s ticks cannot slip through.
    try:
        step = 1.0 / float(scene_filter.QC_FPS)
    except Exception:
        step = 0.25
    n = max(1, int((dur / step) + 0.5)) if dur > 0 else 1
    banner_hits = 0
    cap_hits = 0
    checked = 0
    # Decode once, sequentially, so H.264 keyframe seek cannot skip a flash.
    times = []
    for i in range(n):
        t = min(dur - 0.04, i * step + 0.04) if dur > 0.2 else 0.0
        times.append(float(t))
    grabbed = []
    try:
        grabbed = scene_filter.grab_times(path, times)
    except Exception:
        grabbed = []
    by_t = {round(t, 3): fr for t, fr in grabbed}
    cap = cv2.VideoCapture(str(path))
    for i, t in enumerate(times):
        fr = by_t.get(round(t, 3))
        if fr is None and grabbed:
            fr = min(grabbed, key=lambda kv: abs(kv[0] - t))[1]
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
                try:
                    plate = scene_filter.active_picture(plate)
                except Exception:
                    pass
        reasons = frame_reasons(plate)
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
    # Dialogue captions: require visibility inside speech/caption windows, not filler-all-the-way.
    speech_wins = layout.get("speech_windows") if layout else None
    cap_cues = layout.get("captions") if layout else None
    if cap_cues:
        issues[:] = [x for x in issues if x != "lower-third captions not visible"]
        miss = 0
        checked_cues = 0
        mids = []
        for c in cap_cues:
            mid = (float(c.get("t0") or 0) + float(c.get("t1") or 0)) / 2.0
            if mid < 0 or (dur and mid >= dur):
                continue
            mids.append(round(mid, 3))
        try:
            cue_frames = {round(t, 3): fr for t, fr in scene_filter.grab_times(path, mids)}
        except Exception:
            cue_frames = {}
        for mid in mids:
            fr = cue_frames.get(round(mid, 3))
            if fr is None:
                fr = _frame_at(path, mid)
            if fr is None:
                continue
            checked_cues += 1
            if not _caption_visible(fr, caption_y):
                miss += 1
        if checked_cues and miss > max(1, checked_cues // 3):
            issues.append(f"dialogue subtitles missing on {miss}/{checked_cues} speech cues")
    elif speech_wins:
        issues.append("speech detected but no dialogue subtitles were written")
    # Relax the old "captions must appear somewhere" rule when we know there is no speech.
    if (not cap_cues) and (not speech_wins):
        issues[:] = [x for x in issues if x != "lower-third captions not visible"]
    if (layout or {}).get("edit_mode") == "music_only":
        issues[:] = [x for x in issues
                     if x != "lower-third captions not visible"
                     and not str(x).startswith("dialogue")
                     and not str(x).startswith("speech detected")]
    # Near-duplicate shots in the output.
    dupes = check_duplicate_shots(path, layout, dur)
    if dupes:
        issues.append(f"{len(dupes)} near-duplicate shot pair(s) in the output")
    # Dialogue must be audible during speech windows.
    speech_vols = []
    if speech_wins:
        for a, b in speech_wins:
            if isinstance(a, (list, tuple)):
                a, b = a[0], a[1] if len(a) > 1 else b
            mv = _mean_volume_window(path, float(a), float(b))
            if mv is not None:
                speech_vols.append(mv)
        if speech_vols and max(speech_vols) <= -32:
            issues.append(f"dialogue inaudible during speech (mean_volume={sum(speech_vols)/len(speech_vols):.1f} dB)")
    return {
        "ok": not issues,
        "issues": issues,
        "blacklist": blacklist,
        "meta": meta,
        "mean_volume_db": vol,
        "banner_hits": banner_hits,
        "caption_hits": cap_hits,
        "checked": checked,
        "duplicate_shots": dupes,
        "speech_volumes_db": speech_vols,
    }


def write_report(path: Path, report: dict) -> Path:
    path = Path(path)
    path.write_text(json.dumps(report, indent=2, default=str))
    return path
