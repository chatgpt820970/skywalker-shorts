#!/usr/bin/env python3
"""Skywalker auto-editor: scene-aware, beat-synced 9:16 Shorts.

Default template (caption-bar, from the "Rocket knew something" style):
- 1080x1920 black canvas.
- A TIGHT white hook bar (text-block height + ~30px pad top and bottom,
  ~195px for two lines) with a bold black italic 1-2 line hook, centered
  in the bar, fully on-screen, never cut.
- The full UNCROPPED source frame, scaled to 1080 width, sits DIRECTLY
  below the bar with no gap. Bar + frame form one block, centered
  vertically on the canvas. Never crop the source; leftover space is
  plain black (no blur).
- Moody/desaturated cinematic grade, stabilized, no Ken Burns.
- No text is burned onto the footage itself.
- Anime/movie energy (flag --energy, auto for those pillars): hard beat
  cuts, a speed ramp into the drop, and a temporary (<=0.4s) flash +
  punch-in of the whole bar+frame block. The source frame is never cropped.

Cuts snap to music beats. Transitions: hard cut, white flash, or whip.

Other templates (flag --template):
- depthtext: full-frame 9:16, teal-orange grade, slow-mo on beat drops,
  3D words composited behind the subject (u2netp / rembg / mediapipe).
- postcard: black 9:16, social-post header (logo + channel name + one-line
  hook) and the full uncropped landscape clip at full width, hard beat cuts.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import cv2
import librosa
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageFilter, ImageStat
from scipy import ndimage

REPO = Path(__file__).resolve().parent
ROOT = Path(os.environ.get("SKY_ROOT", "/workspace/skywalker"))
# CI / repo layout: models, music, assets live next to this file.
AE = Path(os.environ.get("SKY_AE", str(REPO if (REPO / "models").exists() else ROOT / "autoedit")))
WORK = Path(os.environ.get("SKY_WORK", str(AE / "work")))
CLIPS = WORK / "clips"
QA = WORK / "qa"


def _fonts_dir() -> Path:
    for p in (REPO / "fonts", AE / "fonts", WORK / "fonts"):
        if (p / "Poppins-BoldItalic.ttf").exists():
            return p
    return REPO / "fonts"


FONTS = _fonts_dir()
W, H = 1080, 1920
FPS = 24
# Caption-bar template: tight white hook bar directly above the
# uncropped 1080-wide frame. Bar + frame are one block, centered
# vertically on a black 1080x1920 canvas. Never crop the source.
BAR_PAD_Y = 30             # padding above and below the hook text
SEP_H = 2                  # 2px hairline at the bottom of the bar (no gap)
# BAR_H is computed from the text; this is only a fallback estimate
# (~195px for a 2-line hook, matching the approved layout).
BAR_H = 196
ENERGY = False             # anime/movie: beat cuts, ramp into drop, flash + punch-in
DROP_T = None              # output time of the music drop / reveal (set by apply_energy)
PILLAR_MOOD = {"anime": "anime", "movie": "movie", "celeb": "anime", "romance": "romance", "funny": "funny"}
ZOOM = 1.0                 # Ken Burns off
STABILIZE = True           # vidstabdetect + vidstabtransform
OVERLAY_TEXT = False       # never burn type onto the footage itself
TARGET_DUR = 32.0
END_CARD = 2.4
END_STILL = 52.0           # Godzilla lightning still (mezzanine time)
HOOK_TEXT = "The reason why the whole world is talking about Godzilla"
HOOK_LINES = [
    "The reason why the whole world",
    "is talking about Godzilla",
]
FONT_HOOK = FONTS / "Poppins-BoldItalic.ttf"
YUNET_MODEL = AE / "models" / "face_detection_yunet_2023mar.onnx"
# Kept so older flags still parse; the caption-bar template ignores this.
PLATE_H_TARGET = 0
FEATHER = 0

FONT_ANTON = FONTS / "Anton-Regular.ttf"
FONT_BEBAS = FONTS / "BebasNeue-Regular.ttf"
FONT_ARCH = FONTS / "ArchivoBlack-Regular.ttf"

# Godzilla Minus Zero commentary (same facts as v1, new edit)
CAPTIONS = [
    {"t0": 0.00, "t1": 2.70, "lines": ["THIS MOVIE JUST BROKE", "A 22-YEAR US RECORD"], "style": "hook"},
    {"t0": 2.70, "t1": 6.40, "lines": ["$31.6 MILLION", "NORTH AMERICAN OPENING"], "style": "stat"},
    {"t0": 6.40, "t1": 11.20, "lines": ["BIGGEST JAPANESE", "LIVE-ACTION US DEBUT"], "style": "fact"},
    {"t0": 11.20, "t1": 16.80, "lines": ["GODZILLA MINUS ZERO"], "style": "title"},
    {"t0": 16.80, "t1": 22.50, "lines": ["IT JUST MADE HISTORY"], "style": "fact"},
    {"t0": 22.50, "t1": 29.40, "lines": ["NOW IN THEATERS"], "style": "cta"},
]


def run(cmd, **kw):
    subprocess.check_call(cmd, **kw)


def ffprobe_dur(path: Path) -> float:
    out = subprocess.check_output(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration",
         "-of", "default=nw=1:nk=1", str(path)],
        text=True,
    ).strip()
    return float(out)


def even(n: int) -> int:
    return int(n) - (int(n) % 2)


# ---------------------------------------------------------------------------
# 1. Letterbox + mezzanine
# ---------------------------------------------------------------------------
def detect_letterbox(src: Path) -> tuple[int, int, int, int]:
    """Return crop w:h:x:y that strips black bars only.

    A row is letterbox only if it is black in every well-lit sample.
    Title cards, fades, and dark shots must not shrink the picture:
    we take the min top / max bottom content row, sampled across the
    whole duration (not a median of a few early timestamps).
    """
    probe = subprocess.check_output(
        ["ffprobe", "-v", "error", "-select_streams", "v:0",
         "-show_entries", "stream=width,height,duration", "-of", "csv=p=0", str(src)],
        text=True,
    ).strip()
    parts = probe.split(",")
    sw, sh = int(parts[0]), int(parts[1])
    try:
        dur = float(parts[2]) if len(parts) > 2 else 0.0
    except ValueError:
        dur = 0.0
    times = [2, 8, 16, 28, 40, 52]
    if dur and dur > 8:
        times += [max(0.5, dur * f) for f in (0.15, 0.3, 0.45, 0.6, 0.75, 0.9)]
    times = sorted({round(float(t), 2) for t in times if t < max(dur - 0.25, 3)})
    if not times:
        times = [2.0]
    ys, ye = [], []
    for t in times:
        tmp = WORK / f"_lb_{t}.png"
        subprocess.call(
            ["ffmpeg", "-y", "-ss", str(t), "-i", str(src), "-frames:v", "1",
             str(tmp)],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )
        if not tmp.exists():
            continue
        a = np.array(Image.open(tmp))
        lum = a.mean(axis=(1, 2)) if a.ndim == 3 else a.mean(axis=1)
        # Skip fades / empty frames so they cannot vote for a crop.
        if float(lum.mean()) < 8:
            tmp.unlink(missing_ok=True)
            continue
        rows = np.where(lum > 10)[0]
        if len(rows) >= 20:
            ys.append(int(rows[0]))
            ye.append(int(rows[-1]))
        tmp.unlink(missing_ok=True)
    if not ys:
        return sw, sh, 0, 0
    # Only bars present in ALL content samples are stripped.
    y1 = even(min(ys))
    y2 = int(max(ye))
    h = even(y2 - y1 + 1)
    # keep even-even for yuv420
    if y1 + h > sh:
        h = even(sh - y1)
    # Pathological case (every sample was a mid-frame title card): keep full frame.
    if h < sh * 0.5:
        return sw, sh, 0, 0
    return sw, h, 0, y1


def make_mezzanine(src: Path, dst: Path, crop) -> Path:
    # Cache is keyed to the source file, so a new --src never reuses old footage.
    key = f"{Path(src).resolve()}|{Path(src).stat().st_size}|{crop}"
    keyf = dst.with_suffix(".src.txt")
    if dst.exists() and dst.stat().st_size > 1_000_000 and keyf.exists() and keyf.read_text() == key:
        return dst
    for stale in (dst, WORK / "mezzanine_stab.mp4", WORK / "transforms.trf"):
        if stale.exists():
            stale.unlink()
    keyf.write_text(key)
    w, h, x, y = crop
    vf = f"crop={w}:{h}:{x}:{y},format=yuv420p"
    run([
        "ffmpeg", "-y", "-i", str(src), "-vf", vf,
        "-c:v", "libx264", "-preset", "fast", "-crf", "16", "-an", str(dst),
    ])
    return dst


def _has_vidstab() -> bool:
    try:
        out = subprocess.check_output(["ffmpeg", "-hide_banner", "-filters"],
                                      stderr=subprocess.STDOUT, text=True)
        return "vidstabdetect" in out and "vidstabtransform" in out
    except Exception:
        return False


def stabilize(src: Path, dst: Path, trf: Path) -> Path:
    """Two-pass libvidstab. Removes handheld jitter, keeps intentional moves."""
    if dst.exists() and dst.stat().st_size > 1_000_000 and trf.exists():
        return dst
    if not _has_vidstab():
        print("  WARNING: ffmpeg has no vidstab filters; copying mezzanine unstabilized")
        if src.resolve() != dst.resolve():
            dst.write_bytes(src.read_bytes())
        return dst
    detect = (
        f"vidstabdetect=shakiness=5:accuracy=15:stepsize=6:"
        f"mincontrast=0.3:result={trf}"
    )
    run([
        "ffmpeg", "-y", "-i", str(src),
        "-vf", detect, "-f", "null", "-",
    ])
    transform = (
        f"vidstabtransform=input={trf}:smoothing=12:optzoom=0:zoom=0:"
        f"interpol=bilinear:crop=keep,"
        f"unsharp=5:5:0.8:3:3:0.4,format=yuv420p"
    )
    run([
        "ffmpeg", "-y", "-i", str(src),
        "-vf", transform,
        "-c:v", "libx264", "-preset", "fast", "-crf", "16", "-an",
        str(dst),
    ])
    if not dst.exists() or dst.stat().st_size < 1_000_000:
        raise RuntimeError(f"stabilize failed: {dst}")
    return dst


# ---------------------------------------------------------------------------
# 2. Scene detection + scoring
# ---------------------------------------------------------------------------
def detect_scenes(mezz: Path) -> list[dict]:
    csv_dir = WORK / "scenes.csv"
    csv_dir.mkdir(parents=True, exist_ok=True)
    # PySceneDetect names the CSV after the input stem (mezzanine_stab, not always mezzanine).
    def _csv():
        exact = csv_dir / f"{mezz.stem}-Scenes.csv"
        if exact.exists():
            return exact
        hits = sorted(csv_dir.glob("*-Scenes.csv"), key=lambda p: p.stat().st_mtime, reverse=True)
        return hits[0] if hits else exact
    csv_path = _csv()
    if not csv_path.exists():
        run([
            sys.executable, "-m", "scenedetect",
            "-i", str(mezz),
            "detect-content", "--threshold", "27",
            "list-scenes", "-o", str(csv_dir),
        ])
        csv_path = _csv()
    if not csv_path.exists():
        raise FileNotFoundError(f"PySceneDetect wrote no *Scenes.csv under {csv_dir} for {mezz}")
    scenes = []
    import csv
    with open(csv_path) as f:
        # first line is timecode list, second is header
        rows = list(csv.reader(f))
    header_i = next(i for i, r in enumerate(rows) if r and r[0] == "Scene Number")
    for r in rows[header_i + 1:]:
        if not r or not r[0].isdigit():
            continue
        scenes.append({
            "id": int(r[0]),
            "start": float(r[3]),
            "end": float(r[6]),
            "dur": float(r[9]),
            "start_frame": int(r[1]),
            "end_frame": int(r[4]),
        })
    return scenes


def _sample_frames(mezz: Path, start: float, dur: float, n: int = 6) -> list[np.ndarray]:
    frames = []
    times = [start + dur * (i + 0.5) / n for i in range(n)]
    times = [min(t, start + max(dur - 1 / FPS, 0)) for t in times]
    for i, t in enumerate(times):
        p = WORK / f"_sf_{start:.3f}_{i}.jpg"
        subprocess.call(
            ["ffmpeg", "-y", "-ss", f"{t:.3f}", "-i", str(mezz),
             "-frames:v", "1", "-q:v", "3", str(p)],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )
        if p.exists():
            frames.append(np.array(Image.open(p).convert("RGB")))
            p.unlink(missing_ok=True)
    return frames


def score_scenes(mezz: Path, scenes: list[dict]) -> list[dict]:
    """Motion, luma, spectacle vs talking-head / title-card flags."""
    out = []
    for s in scenes:
        if s["dur"] < 0.45:
            s.update(score=0, skip=True, reason="too_short", motion=0,
                     luma=0, black=1, spectacle=0, skin=0, graphic=0)
            out.append(s)
            continue
        frames = _sample_frames(mezz, s["start"], s["dur"], n=5)
        if not frames:
            s.update(score=0, skip=True, reason="no_frames", motion=0,
                     luma=0, black=1, spectacle=0, skin=0, graphic=0)
            out.append(s)
            continue
        arrs = [f.astype(np.float32) for f in frames]
        # motion
        diffs = [np.mean(np.abs(arrs[i] - arrs[i - 1])) for i in range(1, len(arrs))]
        motion = float(np.mean(diffs)) if diffs else 0.0
        mid = arrs[len(arrs) // 2]
        luma = float(mid.mean())
        black = float((mid.mean(axis=2) < 18).mean())
        # skin-ish (talking heads)
        r, g, b = mid[:, :, 0], mid[:, :, 1], mid[:, :, 2]
        skin = float((((r > 80) & (r > g) & (g > b * 0.7) & (r - b > 15)).mean()))
        # typography / logo card: sharp horizontal letterforms on a smoother field
        # (tuned so live-action and blue atomic-breath shots are NOT flagged)
        L = mid.mean(axis=2)
        hL, wL = L.shape
        cL = L[int(hL * 0.12):int(hL * 0.88), int(wL * 0.10):int(wL * 0.90)]
        loc = np.abs(cL - ndimage.uniform_filter(cL, 21))
        hard = loc > 28
        horiz = ndimage.binary_opening(hard, structure=np.ones((1, 9)))
        horiz_frac = float(horiz.mean())
        quant = (mid // 32).astype(np.int32)
        keys = quant[:, :, 0] * 64 + quant[:, :, 1] * 8 + quant[:, :, 2]
        uniq = len(np.unique(keys))
        gy = np.abs(np.diff(L, axis=0)).mean()
        gx = np.abs(np.diff(L, axis=1)).mean()
        textish = float(gy / (gx + 1e-3))
        graphic = 1.0 if horiz_frac >= 0.040 else 0.0
        if black > 0.75 and luma < 20:
            graphic = 1.0
        # studio logos at the very start of a trailer
        if s["start"] < 1.20:
            graphic = 1.0
        # spectacle: motion + contrast + not skin-heavy
        contrast = float(mid.std())
        spectacle = motion * 0.55 + contrast * 0.35 + (1.0 - skin) * 12 + (1.0 - black) * 8
        # skip logo/title cards and near-black
        skip = False
        reason = ""
        if graphic >= 1.0:
            skip, reason = True, "title_or_graphic"
        elif luma < 16:
            skip, reason = True, "black"
        elif s["dur"] < 0.55:
            skip, reason = True, "too_short"
        score = spectacle
        if skip:
            score = 0
        s.update(
            score=float(score), skip=skip, reason=reason,
            motion=float(motion), luma=float(luma), black=float(black),
            spectacle=float(spectacle), skin=float(skin), graphic=float(graphic),
            uniq=int(uniq), textish=float(textish),
        )
        out.append(s)
        print(
            f"  scene {s['id']:02d} {s['start']:6.2f}-{s['end']:6.2f} "
            f"({s['dur']:.2f}s) mot={motion:5.1f} luma={luma:5.1f} "
            f"skin={skin:.2f} spec={spectacle:6.1f} "
            f"{'SKIP '+reason if skip else 'OK'}"
        )
    return out


# ---------------------------------------------------------------------------
# 3. Music analysis
# ---------------------------------------------------------------------------
def pick_music(pillar: str, seed: str = "") -> dict:
    """Pick a credited, copyright-safe track for a pillar from music/credits.json.

    anime/celeb -> phonk/hype, movie -> epic, romance -> emotional, funny -> upbeat.
    Deterministic per seed (queue item id) so re-renders keep the same track.
    """
    import hashlib
    cj = json.loads((AE / "music" / "credits.json").read_text())
    mood = PILLAR_MOOD.get((pillar or "movie").lower(), "movie")
    pool = [t for t in cj["tracks"] if t["pillar"] == mood and (AE / "music" / t["file"]).exists()]
    if not pool:
        pool = [t for t in cj["tracks"] if (AE / "music" / t["file"]).exists()]
    if not pool:
        raise SystemExit("No credited music tracks found under music/")
    h = int(hashlib.sha1((seed or pillar or "x").encode()).hexdigest(), 16)
    t = dict(pool[h % len(pool)])
    t["path"] = str(AE / "music" / t["file"])
    return t


def find_drop(y: np.ndarray, sr: int, pulse: np.ndarray, lo: float, hi: float) -> float | None:
    """Biggest sustained energy jump (the drop), snapped to a beat, in [lo, hi]."""
    hop = 512
    rms = librosa.feature.rms(y=y, hop_length=hop)[0]
    onset = librosa.onset.onset_strength(y=y, sr=sr, hop_length=hop)
    n = min(len(rms), len(onset))
    e = rms[:n] / (rms[:n].max() + 1e-9) + 0.25 * onset[:n] / (onset[:n].max() + 1e-9)
    tt = librosa.frames_to_time(np.arange(n), sr=sr, hop_length=hop)
    k = max(1, int(1.5 * sr / hop))
    best, best_t = -1e9, None
    for b in pulse:
        if b < lo or b > hi:
            continue
        i = int(np.searchsorted(tt, b))
        if i - k < 0 or i + k >= n:
            continue
        jump = float(e[i:i + k].mean() - e[i - k:i].mean())
        if jump > best:
            best, best_t = jump, float(b)
    return best_t


def analyze_music(path: Path, target: float = TARGET_DUR) -> dict:
    y, sr = librosa.load(str(path), sr=22050, mono=True)
    tempo, beats = librosa.beat.beat_track(y=y, sr=sr, units="time")
    tempo = float(tempo[0] if hasattr(tempo, "__len__") else tempo)
    # If librosa doubled the tempo (~160+), use every 2nd beat as the pulse.
    pulse = beats
    if tempo >= 140:
        pulse = beats[::2]
        tempo = tempo / 2.0
    hop = 512
    rms = librosa.feature.rms(y=y, hop_length=hop)[0]
    times = librosa.frames_to_time(np.arange(len(rms)), sr=sr, hop_length=hop)
    win = target
    # Best high-energy window, prefer later (climax) when close
    best, best_t = -1.0, 0.0
    t = 0.0
    while t + win < times[-1]:
        m = float(rms[(times >= t) & (times < t + win)].mean()) if np.any((times >= t) & (times < t + win)) else 0.0
        # slight bias toward later climax
        m_adj = m * (1.0 + 0.08 * (t / max(times[-1], 1)))
        if m_adj > best:
            best, best_t = m_adj, t
        t += 0.25
    # Snap window start to the nearest pulse at or before best_t
    earlier = pulse[pulse <= best_t + 0.05]
    start = float(earlier[-1]) if len(earlier) else float(best_t)
    # Pulses inside the window (relative to start)
    rel = pulse[(pulse >= start - 0.02) & (pulse <= start + win + 0.5)] - start
    rel = np.clip(rel, 0, None)
    rel = np.unique(np.round(rel, 4))
    if rel.size == 0 or rel[0] > 0.15:
        rel = np.insert(rel, 0, 0.0)
    else:
        rel[0] = 0.0
    drop_rel = None
    if ENERGY:
        # Align the music drop with the reveal at ~38% of the edit.
        reveal_at = max(4.0, 0.38 * target)
        dur_m = float(times[-1])
        d = find_drop(y, sr, pulse, reveal_at, max(reveal_at, dur_m - (target - reveal_at)))
        if d is not None:
            start = max(0.0, d - reveal_at)
            on = pulse[pulse <= start + 0.05]
            start = float(on[-1]) if len(on) else start
            drop_rel = d - start
            rel = pulse[(pulse >= start - 0.02) & (pulse <= start + win + 0.5)] - start
            rel = np.unique(np.round(np.clip(rel, 0, None), 4))
            if rel.size == 0 or rel[0] > 0.15:
                rel = np.insert(rel, 0, 0.0)
            else:
                rel[0] = 0.0
    return {
        "path": str(path),
        "tempo": tempo,
        "start": start,
        "drop_rel": drop_rel,
        "window": win,
        "pulses": rel.tolist(),  # seconds relative to music window / video t=0
        "abs_beats": pulse.tolist(),
    }


# ---------------------------------------------------------------------------
# 4. Edit plan
# ---------------------------------------------------------------------------
def plan_edit(scenes: list[dict], music: dict, target: float = TARGET_DUR) -> list[dict]:
    pulses = np.array(music["pulses"], dtype=float)
    ibi = float(np.median(np.diff(pulses))) if len(pulses) > 3 else 0.75
    usable = [s for s in scenes if not s.get("skip") and s["dur"] >= 0.70]
    if not usable:
        raise SystemExit("No usable scenes after filtering")
    usable_by_spec = sorted(usable, key=lambda s: s["spectacle"], reverse=True)

    body = target - END_CARD

    def dur_for(t, n):
        later = pulses[pulses >= t - 1e-3]
        if len(later) > n:
            return float(later[n] - t)
        return n * ibi

    def fits(s, vis, extra=0.12):
        return s["dur"] >= vis + extra

    used = set()
    slots = []  # list of (visible, scene)

    # Hook: highest spectacle that is not a talking-head and fits 3 pulses (else 2)
    t = 0.0
    hook_vis = max(1.2, min(dur_for(t, 3), 2.8))
    hook_cands = [s for s in usable_by_spec if s["skin"] < 0.20 and fits(s, hook_vis)]
    if not hook_cands:
        hook_vis = max(1.0, min(dur_for(t, 2), 2.2))
        hook_cands = [s for s in usable_by_spec if fits(s, hook_vis)]
    hook = hook_cands[0]
    slots.append((hook_vis, hook))
    used.add(hook["id"])
    t += hook_vis

    def pick(t, vis, prefer_spectacle, prev):
        cands = [s for s in usable if s["id"] not in used and fits(s, vis)]
        if not cands:
            cands = [s for s in usable if s["id"] not in used and s["dur"] >= vis]
        if not cands:
            return None

        def key(s):
            pen = 0.0
            if prev is not None:
                if abs(s["start"] - prev["end"]) < 0.08 or abs(s["end"] - prev["start"]) < 0.08:
                    pen += 90
                if abs(s["start"] - prev["start"]) < 2.0:
                    pen += 25
                if prev["skin"] > 0.16 and s["skin"] > 0.16:
                    pen += 30
            return s["spectacle"] - pen

        cands.sort(key=key, reverse=True)
        return cands[0]

    # Fill to body: 3 pulses normally, 2 in the climax, shrink if needed
    while t < body - 0.35:
        remaining = body - t
        climax = t >= body * 0.62
        # Prefer a pulse count that still fits the remaining time
        prefer = 2 if climax else 3
        if remaining > 3.2 and not climax:
            prefer = 4 if (len(slots) % 4 == 0) else 3
        chosen = None
        vis = 0.0
        for n in (prefer, 3, 2, 4, 1):
            vis = dur_for(t, n)
            vis = max(0.75, min(vis, remaining, 3.3))
            if vis > remaining + 0.01:
                continue
            sc = pick(t, vis, climax, slots[-1][1])
            if sc is not None:
                # shrink vis to a pulse boundary that the scene can actually hold
                if not fits(sc, vis) and sc["dur"] < vis + 0.12:
                    # step down until it fits
                    ok = False
                    for n2 in range(n, 0, -1):
                        v2 = dur_for(t, n2)
                        v2 = max(0.70, min(v2, sc["dur"] - 0.08, remaining))
                        if sc["dur"] >= v2:
                            vis = v2
                            ok = True
                            break
                    if not ok:
                        continue
                chosen = sc
                break
        if chosen is None:
            break
        slots.append((vis, chosen))
        used.add(chosen["id"])
        t += vis

    # If we still came up short, top up with leftover scenes that actually fit
    while t < body - 0.50:
        remaining = body - t
        leftovers = [s for s in usable_by_spec if s["id"] not in used and s["dur"] >= 0.70]
        if not leftovers:
            break
        placed = False
        for sc in leftovers:
            vis = min(remaining, sc["dur"] - 0.06, max(0.70, dur_for(t, 2)))
            if vis < 0.70 or sc["dur"] < vis:
                continue
            slots.append((vis, sc))
            used.add(sc["id"])
            t += vis
            placed = True
            break
        if not placed:
            break

    edl = []
    acc = 0.0
    for i, (vis, sc) in enumerate(slots):
        trans_out, td = "cut", 0.0
        if i < len(slots) - 1:
            nxt = slots[i + 1][1]
            room_a = sc["dur"] >= vis + 0.22
            room_b = nxt["dur"] >= slots[i + 1][0] + 0.22
            if room_a and room_b:
                if i == 0:
                    trans_out, td = "flash", 0.12
                elif i % 3 == 0:
                    trans_out, td = ("whip_left" if (i // 3) % 2 == 0 else "whip_right"), 0.16
        edl.append({
            "i": i,
            "scene": sc["id"],
            "src_start": round(sc["start"], 3),
            "src_end": round(sc["end"], 3),
            "src_dur": round(sc["dur"], 3),
            "visible": round(vis, 3),
            "t0": round(acc, 3),
            "t1": round(acc + vis, 3),
            "trans_out": trans_out,
            "td": td,
            "spectacle": round(sc["spectacle"], 2),
            "skin": round(sc["skin"], 3),
        })
        acc += vis
    return edl


def apply_energy(edl: list[dict], scenes: list[dict], music: dict) -> list[dict]:
    """Anime/movie energy: hard beat cuts, ramp into the drop, reveal on the drop.

    - every transition becomes a hard cut on the beat grid (plan_edit snaps to pulses)
    - the clip right before the drop gets a speed ramp (1x -> ~2.2x)
    - the strongest unused shot is swapped onto the cut nearest the music drop
    - music start is nudged so the drop lands exactly on that cut
    The flash + punch-in are added in mux_final at DROP_T (whole block, temporary, <=0.4s).
    The source frame itself is never cropped.
    """
    global DROP_T
    body = [it for it in edl if it.get("scene") != "end"]
    for it in body:
        it["trans_out"], it["td"] = "cut", 0.0
    if len(body) < 3:
        return edl
    want = music.get("drop_rel")
    if want is None:
        want = 0.38 * body[-1]["t1"]
    cand = [it for it in body[2:]]
    drop = min(cand, key=lambda it: abs(it["t0"] - want))
    # reveal = highest-spectacle scene not used as the hook that fits this slot
    by_id = {sc["id"]: sc for sc in scenes}
    hook_id = body[0]["scene"]
    best = None
    for it in body:
        if it is drop or it["scene"] == hook_id:
            continue
        sa, sb = by_id.get(it["scene"]), by_id.get(drop["scene"])
        if not sa or not sb:
            continue
        if sa["dur"] >= drop["visible"] + 0.1 and sb["dur"] >= it["visible"] + 0.1:
            if it["spectacle"] > drop["spectacle"] and (best is None or it["spectacle"] > best["spectacle"]):
                best = it
    if best is not None:
        for k in ("scene", "src_start", "src_end", "src_dur", "spectacle", "skin"):
            best[k], drop[k] = drop[k], best[k]
    drop["drop"] = True
    i = body.index(drop)
    prev = body[i - 1]
    def need(vis, k):
        return vis * k / math.log(1 + k)
    for k in (1.2, 1.0, 0.8, 0.6, 0.4, 0.25):  # speed goes 1x -> (1+k)x across the pre-drop clip
        if prev["src_dur"] >= need(prev["visible"], k) + 0.1:
            prev["ramp"] = k
            break
        # else try swapping a longer shot into the pre-drop slot
        swapped = False
        for it in body[1:]:
            if it is prev or it is drop:
                continue
            sa, sb = by_id.get(it["scene"]), by_id.get(prev["scene"])
            if sa and sb and sa["dur"] >= need(prev["visible"], k) + 0.1 and sb["dur"] >= it["visible"] + 0.1:
                for kk in ("scene", "src_start", "src_end", "src_dur", "spectacle", "skin"):
                    it[kk], prev[kk] = prev[kk], it[kk]
                prev["ramp"] = k
                swapped = True
                break
        if swapped:
            break
    # shift the music window so the drop sits exactly on the cut
    if music.get("drop_rel") is not None:
        shift = float(music["drop_rel"]) - float(drop["t0"])
        music["start"] = max(0.0, float(music["start"]) + shift)
        music["drop_rel"] = float(drop["t0"])
    DROP_T = float(drop["t0"])
    print(f"  ENERGY: drop/reveal @ {DROP_T:.2f}s (scene {drop['scene']}), ramp on clip {prev['i']}: {prev.get('ramp')}")
    return edl


def hook_lines(text: str | None = None) -> list[str]:
    """Split the hook into 1-2 lines. Prefer the canned pair for the default."""
    text = (text or HOOK_TEXT).strip()
    if text == HOOK_TEXT:
        return list(HOOK_LINES)
    words = text.split()
    if len(text) <= 36 or len(words) <= 5:
        return [text]
    # break near the midpoint, preferring a later short line like the reference
    mid = len(text) // 2
    best_i, best_d = 0, 10**9
    acc = 0
    for i, w in enumerate(words[:-1]):
        acc += len(w) + 1
        d = abs(acc - mid)
        if d < best_d:
            best_d, best_i = d, i
    return [" ".join(words[: best_i + 1]), " ".join(words[best_i + 1 :])]


def centered_block_y(block_h: int) -> int:
    """Vertical origin of a block so it sits in the middle of the 9:16 canvas."""
    y = even(max(0, (H - int(block_h)) // 2))
    if y + block_h > H:
        y = max(0, even(H - int(block_h)))
    return y


def whitebar_layout(plate_h: int, bar_h: int) -> dict:
    """Tight bar directly on top of the uncropped plate, as one centered block."""
    block_h = int(bar_h) + int(plate_h)
    y = centered_block_y(block_h)
    return {
        "bar_y": y,
        "bar_h": int(bar_h),
        "video_y": y + int(bar_h),
        "plate_h": int(plate_h),
        "block_h": block_h,
    }


def draw_caption_bar(dst: Path, lines: list[str] | None = None) -> Path:
    """Tight white bar: text height + ~30px pad top/bottom, bold black italic, centered."""
    lines = lines or hook_lines()
    probe = Image.new("RGB", (W, 400), (255, 255, 255))
    d = ImageDraw.Draw(probe)
    font_path = FONT_HOOK if FONT_HOOK.exists() else FONT_ANTON
    side = 56          # left/right margin so type is never clipped
    pad_y = BAR_PAD_Y  # approved tight padding
    size = 72
    font = None
    widths = heights = None
    while size >= 32:
        font = ImageFont.truetype(str(font_path), size)
        widths, heights = [], []
        for ln in lines:
            bbox = d.textbbox((0, 0), ln, font=font)
            widths.append(bbox[2] - bbox[0])
            heights.append(bbox[3] - bbox[1])
        if max(widths) <= W - 2 * side:
            break
        size -= 2
    # 12px gap matches the approved 2-line hook
    gap = 12 if len(lines) > 1 else 0
    total_h = sum(heights) + gap * (len(lines) - 1)
    bar_h = total_h + 2 * pad_y + SEP_H
    if bar_h % 2:
        bar_h += 1  # even height for yuv420, keep pad >= 30
    bar = Image.new("RGB", (W, bar_h), (255, 255, 255))
    d = ImageDraw.Draw(bar)
    y = pad_y
    for ln in lines:
        bbox = d.textbbox((0, 0), ln, font=font)
        tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
        x = (W - tw) // 2 - bbox[0]
        # clamp so nothing ever goes off the bar
        x = max(side - bbox[0], min(x, W - side - (bbox[2] - bbox[0]) - bbox[0]))
        d.text((x, y - bbox[1]), ln, font=font, fill=(0, 0, 0))
        y += th + gap
    if SEP_H:
        d.rectangle([0, bar_h - SEP_H, W, bar_h], fill=(0, 0, 0))
    dst.parent.mkdir(parents=True, exist_ok=True)
    bar.save(dst)
    return dst


def _grab_rgb(src: Path, t: float) -> np.ndarray | None:
    tmp = WORK / f"_crop_{t:.3f}.jpg"
    subprocess.call(
        ["ffmpeg", "-y", "-ss", f"{t:.3f}", "-i", str(src),
         "-frames:v", "1", "-q:v", "3", str(tmp)],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )
    if not tmp.exists():
        return None
    arr = np.array(Image.open(tmp).convert("RGB"))
    tmp.unlink(missing_ok=True)
    return arr


def _detect_faces(rgb: np.ndarray) -> list[tuple[float, float, float, float]]:
    """Return filtered (x0,y0,x1,y1) face boxes. Drops tiny/huge YuNet junk."""
    if not YUNET_MODEL.exists():
        return []
    h, w = rgb.shape[:2]
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    try:
        det = cv2.FaceDetectorYN_create(str(YUNET_MODEL), "", (w, h), 0.45, 0.3, 5000)
        det.setInputSize((w, h))
        _ok, faces = det.detect(bgr)
    except Exception:
        return []
    out = []
    if faces is None:
        return out
    for f in faces:
        x, y, fw, fh = [float(v) for v in f[:4]]
        if fw < 28 or fh < 28:
            continue
        if fw > 0.52 * w or fh > 0.85 * h:
            continue
        ar = fw / (fh + 1e-6)
        if ar < 0.45 or ar > 1.55:
            continue
        x0, y0 = max(0.0, x), max(0.0, y)
        x1, y1 = min(float(w), x + fw), min(float(h), y + fh)
        out.append((x0, y0, x1, y1))
    return out


def _column_heat(rgb: np.ndarray) -> np.ndarray:
    """Smoothed subject heatmap over x, used when faces don't decide the crop."""
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
    # large-scale contrast so compact subjects beat busy edges
    blur = cv2.GaussianBlur(gray, (0, 0), 18)
    residual = np.abs(gray - blur)
    mag = cv2.GaussianBlur(residual, (0, 0), 8)
    ycrcb = cv2.cvtColor(bgr, cv2.COLOR_BGR2YCrCb)
    _y, cr, cb = cv2.split(ycrcb)
    skin = ((cr > 133) & (cr < 173) & (cb > 77) & (cb < 127) & (gray > 40)).astype(np.float32)
    col = mag.mean(axis=0) + 4.0 * skin.mean(axis=0)
    # mild midtone prior (subjects are rarely pure black or blown out)
    mid = ((gray > 25) & (gray < 235)).astype(np.float32).mean(axis=0)
    col = col + 0.2 * mid * (col.mean() + 1e-6)
    col = ndimage.gaussian_filter1d(col, sigma=24)
    return col


def smart_crop_x(mezz: Path, t0: float, t1: float, mw: int, win_w: int,
                 prefer_first: bool = False) -> int:
    """Horizontal crop origin in source pixels. Keeps faces in the window."""
    win_w = even(max(2, min(win_w, mw)))
    if t1 <= t0:
        t1 = t0 + 0.08
    # sample the opening, a mid point, and just before the out-point
    times = [t0 + 0.04]
    if t1 - t0 > 0.40:
        times.append(t0 + 0.35 * (t1 - t0))
    if t1 - t0 > 0.80:
        times.append(t0 + 0.70 * (t1 - t0))
    times.append(max(t0, t1 - 0.08))
    times = sorted(set(round(t, 3) for t in times))

    heats, faces_all = [], []
    first_faces = []
    for i, t in enumerate(times):
        rgb = _grab_rgb(mezz, t)
        if rgb is None:
            continue
        heats.append(_column_heat(rgb))
        faces = _detect_faces(rgb)
        faces_all.append(faces)
        if i == 0:
            first_faces = faces

    max_x = mw - win_w
    if max_x <= 0 or not heats:
        return 0

    heat = np.mean(np.stack(heats), axis=0)
    # weight faces: larger and more central matter more
    weighted = []
    for faces in faces_all:
        for x0, y0, x1, y1 in faces:
            fw = x1 - x0
            cx = 0.5 * (x0 + x1)
            centrality = 1.0 - 0.45 * abs(cx - mw / 2) / (mw / 2)
            weighted.append((x0, x1, fw, max(0.2, centrality) * fw))
    if prefer_first and first_faces:
        weighted = []
        for x0, y0, x1, y1 in first_faces:
            fw = x1 - x0
            cx = 0.5 * (x0 + x1)
            centrality = 1.0 - 0.45 * abs(cx - mw / 2) / (mw / 2)
            weighted.append((x0, x1, fw, max(0.2, centrality) * fw * 1.4))

    def window_score(x):
        x1 = x + win_w
        s = float(heat[int(x):int(x1)].sum()) if int(x1) <= len(heat) else 0.0
        # mild center prior so no-face CUs (Godzilla's face) stay centred
        mid = x + win_w / 2
        s += 0.12 * heat.sum() * (1.0 - abs(mid - mw / 2) / (mw / 2))
        face_s = 0.0
        for fx0, fx1, fw, wt in weighted:
            overlap = max(0.0, min(x1, fx1) - max(x, fx0))
            if overlap <= 0:
                continue
            if fw <= win_w + 4:
                # face fits: heavy reward if fully inside, heavy penalty if cut
                margin = min(fx0 - x, x1 - fx1)
                if margin >= 6:
                    face_s += 12.0 * wt
                elif margin >= 0:
                    face_s += 8.0 * wt
                else:
                    cut = -margin
                    face_s -= 10.0 * wt * (cut / (fw + 1e-3))
            else:
                # face wider than the window: keep as much of it as we can, centred
                face_s += 6.0 * wt * (overlap / fw)
        return s + face_s

    best_x, best = 0, -1e18
    # coarse then refine
    for x in range(0, max_x + 1, 8):
        sc = window_score(x)
        if sc > best:
            best, best_x = sc, x
    lo, hi = max(0, best_x - 8), min(max_x, best_x + 8)
    for x in range(lo, hi + 1):
        sc = window_score(x)
        if sc > best:
            best, best_x = sc, x
    return even(best_x)


# ---------------------------------------------------------------------------
# Template helpers (whitebar default is unchanged)
# ---------------------------------------------------------------------------
TEMPLATE = "whitebar"  # whitebar | depthtext | postcard
DEPTH_WORDS = "KING RETURNS"
CHANNEL_NAME = "Skywalker"
U2NETP_MODEL = AE / "models" / "u2netp.onnx"
POSTCARD_PAD_TOP = 200
POSTCARD_HEADER_H = 196
POSTCARD_LOGO_NOTE = ""


def set_template(name: str) -> None:
    """Switch template and isolate work/clips for non-whitebar runs."""
    global TEMPLATE, CLIPS, QA
    if name not in ("whitebar", "depthtext", "postcard"):
        raise SystemExit(f"unknown template: {name}")
    TEMPLATE = name
    if name == "whitebar":
        CLIPS = WORK / "clips"
        QA = WORK / "qa"
    else:
        CLIPS = WORK / name / "clips"
        QA = WORK / name / "qa"
    CLIPS.mkdir(parents=True, exist_ok=True)
    QA.mkdir(parents=True, exist_ok=True)


def template_work() -> Path:
    if TEMPLATE == "whitebar":
        return WORK
    d = WORK / TEMPLATE
    d.mkdir(parents=True, exist_ok=True)
    return d


def cinematic_grade(template: str | None = None) -> str:
    t = template or TEMPLATE
    if t == "depthtext":
        # Teal shadows, orange highlights, slightly crushed blacks.
        return (
            "eq=saturation=0.88:contrast=1.18:brightness=-0.05:gamma=0.96,"
            "colorbalance=rs=0.08:gs=0.00:bs=-0.07:"
            "rm=0.03:gm=0.00:bm=0.03:"
            "rh=0.10:gh=0.02:bh=-0.08,"
            "curves=all='0/0 0.10/0.06 0.50/0.48 1/0.97'"
        )
    if t == "postcard":
        return "eq=saturation=1.02:contrast=1.06:brightness=0.00:gamma=1.0"
    # whitebar — exact original grade
    return (
        "eq=saturation=0.72:contrast=1.12:brightness=-0.04:gamma=0.98,"
        "curves=all='0/0 0.12/0.08 0.50/0.47 1/0.96'"
    )


def finalize_edl_for_template(edl: list[dict], music: dict, target: float) -> list[dict]:
    """Postcard: hard cuts only. Depthtext: mark beat-drop text + slow-mo."""
    if TEMPLATE == "postcard":
        for it in edl:
            it["trans_out"] = "cut"
            it["td"] = 0.0
        return edl
    if TEMPLATE != "depthtext":
        return edl
    pulses = np.array(music.get("pulses") or [], dtype=float)
    body = edl[-1]["t1"] if edl else target
    cand = []
    for i, p in enumerate(pulses):
        if 0.6 <= p <= body - 0.3:
            if i % 4 == 0 or (p >= body * 0.45 and i % 2 == 0):
                cand.append(float(p))
    chosen: list[float] = []
    for p in cand:
        if not chosen or p - chosen[-1] >= 2.2:
            chosen.append(p)
        if len(chosen) >= 3:
            break
    if not chosen and len(pulses):
        mid = pulses[min(len(pulses) - 1, len(pulses) // 2)]
        chosen = [float(mid)]
    for it in edl:
        if it.get("scene") == "end":
            continue
        t0, t1 = float(it["t0"]), float(it["t1"])
        hit = any(t0 - 0.05 <= p <= t1 + 0.05 for p in chosen)
        it["show_text"] = bool(hit)
        if hit:
            speed = 0.70
            need = (float(it["visible"]) + float(it.get("td") or 0)) * speed
            it["speed"] = speed if float(it["src_dur"]) >= need + 0.10 else 1.0
        else:
            it["speed"] = 1.0
    ntxt = sum(1 for it in edl if it.get("show_text"))
    if ntxt < 2:
        cands = [it for it in edl if it.get("scene") != "end" and float(it["t0"]) >= 1.0]
        cands.sort(key=lambda x: x.get("spectacle", 0), reverse=True)
        for it in cands:
            if it.get("show_text"):
                continue
            it["show_text"] = True
            need = (float(it["visible"]) + float(it.get("td") or 0)) * 0.70
            it["speed"] = 0.70 if float(it["src_dur"]) >= need + 0.10 else 1.0
            ntxt += 1
            if ntxt >= 2:
                break
    return edl


# ---- depthtext: subject segmentation + 3D words behind the subject ----
_ORT_SESS = None
_ORT_FAIL = False


def _seg_session():
    """Lazy CPU session: u2netp via onnxruntime, else rembg, else mediapipe."""
    global _ORT_SESS, _ORT_FAIL
    if _ORT_SESS is not None or _ORT_FAIL:
        return _ORT_SESS
    if U2NETP_MODEL.exists():
        try:
            import onnxruntime as ort
            so = ort.SessionOptions()
            so.intra_op_num_threads = 2
            so.inter_op_num_threads = 1
            _ORT_SESS = ort.InferenceSession(
                str(U2NETP_MODEL), so, providers=["CPUExecutionProvider"]
            )
            return _ORT_SESS
        except Exception as e:
            print(f"  u2netp/onnxruntime unavailable ({e}); trying fallbacks")
    # rembg (also u2netp under the hood) — optional
    try:
        from rembg import new_session
        _ORT_SESS = ("rembg", new_session("u2netp"))
        return _ORT_SESS
    except Exception:
        pass
    try:
        import mediapipe as mp
        _ORT_SESS = ("mediapipe", mp)
        return _ORT_SESS
    except Exception:
        pass
    _ORT_FAIL = True
    print("  WARNING: no segmentation backend; using saliency fallback")
    return None


def segment_subject(rgb: np.ndarray) -> np.ndarray:
    """Float mask 0..1 (subject=1), same HxW. u2netp at 320px, CPU-friendly."""
    h, w = rgb.shape[:2]
    sess = _seg_session()
    if sess is not None and not isinstance(sess, tuple):
        inp = cv2.resize(rgb, (320, 320), interpolation=cv2.INTER_AREA).astype(np.float32) / 255.0
        mean = np.array([0.485, 0.456, 0.406], np.float32)
        std = np.array([0.229, 0.224, 0.225], np.float32)
        inp = (inp - mean) / std
        chw = np.ascontiguousarray(inp.transpose(2, 0, 1)[None])
        iname = sess.get_inputs()[0].name
        pred = sess.run(None, {iname: chw})[0][0, 0]
        if pred.min() < 0 or pred.max() > 1.0:
            pred = 1.0 / (1.0 + np.exp(-pred))
        pred = cv2.resize(pred.astype(np.float32), (w, h), interpolation=cv2.INTER_CUBIC)
        pred = np.clip(pred, 0.0, 1.0)
        pred = cv2.GaussianBlur(pred, (0, 0), 2.5)
        return pred
    if isinstance(sess, tuple) and sess[0] == "rembg":
        from rembg import remove
        rgba = remove(Image.fromarray(rgb), session=sess[1])
        if not isinstance(rgba, Image.Image):
            rgba = Image.open(rgba).convert("RGBA") if not isinstance(rgba, Image.Image) else rgba
        a = np.array(rgba.convert("RGBA"))[:, :, 3].astype(np.float32) / 255.0
        if a.shape[0] != h or a.shape[1] != w:
            a = cv2.resize(a, (w, h), interpolation=cv2.INTER_CUBIC)
        return np.clip(a, 0, 1)
    if isinstance(sess, tuple) and sess[0] == "mediapipe":
        mp = sess[1]
        # Selfie segmentation (person only; weaker on non-human subjects)
        seg = mp.solutions.selfie_segmentation.SelfieSegmentation(model_selection=1)
        res = seg.process(rgb)
        mask = res.segmentation_mask.astype(np.float32)
        seg.close()
        if mask.shape[0] != h or mask.shape[1] != w:
            mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_CUBIC)
        return np.clip(mask, 0, 1)
    # Cheap saliency fallback (no model)
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY).astype(np.float32)
    blur = cv2.GaussianBlur(gray, (0, 0), 25)
    mag = np.abs(gray - blur)
    mag = cv2.GaussianBlur(mag, (0, 0), 8)
    mag /= (mag.max() + 1e-6)
    yy, xx = np.mgrid[0:h, 0:w]
    prior = np.exp(-((xx - w / 2) ** 2) / (2 * (0.28 * w) ** 2) - ((yy - h / 2) ** 2) / (2 * (0.40 * h) ** 2))
    mask = np.clip(0.6 * mag + 0.4 * prior, 0, 1).astype(np.float32)
    return mask


def draw_depth_words(size: tuple[int, int], words: str) -> Image.Image:
    """Big bold white 3D/shadowed words (2 max), stacked, upper-middle."""
    raw = [w for w in (words or "").strip().split() if w]
    if not raw:
        raw = ["KING", "RETURNS"]
    if len(raw) > 2:
        mid = (len(raw) + 1) // 2
        raw = [" ".join(raw[:mid]), " ".join(raw[mid:])]
    lines = [w.upper() for w in raw]
    W_, H_ = size
    layer = Image.new("RGBA", (W_, H_), (0, 0, 0, 0))
    font_path = FONT_ARCH if FONT_ARCH.exists() else FONT_ANTON
    probe = ImageDraw.Draw(layer)
    side = 40
    size_pt = 240
    widths = heights = [0]
    font = None
    while size_pt >= 48:
        font = ImageFont.truetype(str(font_path), size_pt)
        widths, heights = [], []
        for ln in lines:
            b = probe.textbbox((0, 0), ln, font=font)
            widths.append(b[2] - b[0])
            heights.append(b[3] - b[1])
        gap = int(size_pt * 0.06)
        if max(widths) <= W_ - 2 * side and sum(heights) + gap * (len(lines) - 1) <= int(H_ * 0.52):
            break
        size_pt -= 4
    gap = int(size_pt * 0.06)
    # measure with the chosen font
    boxes = [probe.textbbox((0, 0), ln, font=font) for ln in lines]
    widths = [b[2] - b[0] for b in boxes]
    heights = [b[3] - b[1] for b in boxes]
    total_h = sum(heights) + gap * (len(lines) - 1)
    y0 = int(H_ * 0.22)
    if y0 + total_h > int(H_ * 0.78):
        y0 = max(int(H_ * 0.12), int((H_ - total_h) * 0.35))
    positions = []
    y = y0
    for i, ln in enumerate(lines):
        b = boxes[i]
        tw, th = widths[i], heights[i]
        x = (W_ - tw) // 2 - b[0]
        positions.append((x, y, ln, b, tw, th))
        y += th + gap
    extrusion = max(10, size_pt // 12)
    # soft drop shadow
    sh = Image.new("RGBA", (W_, H_), (0, 0, 0, 0))
    ds = ImageDraw.Draw(sh)
    for x, y, ln, b, tw, th in positions:
        ds.text((x + extrusion + 8, y - b[1] + extrusion + 10), ln, font=font, fill=(0, 0, 0, 170))
    sh = sh.filter(ImageFilter.GaussianBlur(max(8, size_pt // 12)))
    layer = Image.alpha_composite(layer, sh)
    d = ImageDraw.Draw(layer)
    # 3D extrusion (dark steps down-right)
    for step in range(extrusion, 0, -1):
        shade = 18 + int(22 * (1.0 - step / extrusion))
        col = (shade, shade, shade + 6, 255)
        for x, y, ln, b, tw, th in positions:
            d.text((x + step, y - b[1] + step), ln, font=font, fill=col)
    # dark outline then white body
    for x, y, ln, b, tw, th in positions:
        for ox, oy in ((-3, 0), (3, 0), (0, -3), (0, 3), (-2, -2), (2, 2), (2, -2), (-2, 2)):
            d.text((x + ox, y - b[1] + oy), ln, font=font, fill=(16, 16, 20, 255))
        d.text((x, y - b[1]), ln, font=font, fill=(255, 255, 255, 255))
    return layer


def _text_weight(t: float, windows: list[tuple[float, float]], fade: float = 0.18) -> float:
    w = 0.0
    for a, b in windows:
        if t < a or t >= b:
            continue
        if t < a + fade:
            w = max(w, (t - a) / fade)
        elif t > b - fade:
            w = max(w, max(0.0, (b - t) / fade))
        else:
            w = 1.0
    return float(min(1.0, w))


def apply_depth_text(src: Path, edl: list[dict], words: str, dst: Path) -> Path:
    """Composite 3D words behind the subject, only on marked drop windows."""
    windows = []
    for it in edl:
        if it.get("show_text") and it.get("scene") != "end":
            windows.append((float(it["t0"]), float(it["t1"])))
    if not windows:
        print("  no depthtext drop windows; passing through")
        return src
    print(f"  depthtext windows: " + ", ".join(f"{a:.2f}-{b:.2f}s" for a, b in windows))
    # Composite on the actual uncropped plate (typically 1080 x width-fit height).
    pw, ph = _mezz_size(src)
    text_layer = draw_depth_words((pw, ph), words)
    text_rgba = np.array(text_layer)
    text_bgr = text_rgba[:, :, 2::-1].astype(np.float32)  # BGR
    text_a = text_rgba[:, :, 3].astype(np.float32) / 255.0

    cap = cv2.VideoCapture(str(src))
    if not cap.isOpened():
        raise RuntimeError(f"cannot open {src}")
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    dst.parent.mkdir(parents=True, exist_ok=True)
    log = dst.with_suffix(".ffmpeg.log")
    logf = open(log, "w")
    proc = subprocess.Popen(
        [
            "ffmpeg", "-y",
            "-f", "rawvideo", "-pix_fmt", "bgr24",
            "-s", f"{pw}x{ph}", "-r", str(FPS),
            "-i", "-",
            "-c:v", "libx264", "-preset", "veryfast", "-crf", "17",
            "-pix_fmt", "yuv420p", "-an",
            "-video_track_timescale", str(FPS * 256),
            str(dst),
        ],
        stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=logf,
    )
    last_mask = None
    last_i = -99
    i = 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            t = i / float(FPS)
            wgt = _text_weight(t, windows)
            if wgt > 0.01:
                if last_mask is None or (i - last_i) >= 2:
                    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    last_mask = segment_subject(rgb)
                    last_i = i
                mask = last_mask
                # Text lives behind the subject: visible only where the subject mask is low.
                alpha = text_a * (1.0 - mask) * wgt
                a3 = alpha[..., None]
                out = frame.astype(np.float32) * (1.0 - a3) + text_bgr * a3
                frame = np.clip(out, 0, 255).astype(np.uint8)
            if proc.stdin is not None:
                proc.stdin.write(frame.tobytes())
            i += 1
            if i % 48 == 0:
                print(f"  depthtext frames {i}/{n}")
    finally:
        cap.release()
        if proc.stdin is not None:
            proc.stdin.close()
        rc = proc.wait()
        logf.close()
    if rc != 0 or not dst.exists() or dst.stat().st_size < 1000:
        raise RuntimeError(f"depthtext composite failed rc={rc}; see {log}")
    print(f"  depthtext wrote {dst} ({i} frames)")
    return dst


# ---- postcard: social-post header + uncropped landscape clip ----
def find_channel_logo() -> tuple[Path | None, str]:
    """Search /workspace/skywalker first; then the official @Clipsjohns DP."""
    names = [
        "dp.jpg", "dp.png", "dp.jpeg", "logo.png", "logo.jpg",
        "avatar.jpg", "avatar.png", "profile.jpg", "profile.png",
        "skywalker_dp.jpg", "skywalker_dp.png", "channel_dp.jpg",
    ]
    for n in names:
        cand = ROOT / n
        if cand.is_file():
            return cand, f"found Skywalker DP at {cand}"
    official_assets = AE / "assets" / "skywalker_dp.jpg"
    workspace_dp = Path("/workspace/clipsjohns_dp.jpg")
    if official_assets.is_file() or workspace_dp.is_file():
        path = official_assets if official_assets.is_file() else workspace_dp
        note = (
            "no DP inside /workspace/skywalker; using official @Clipsjohns DP "
            f"({path})"
        )
        return path, note
    return None, "no DP found in /workspace/skywalker; using a placeholder circle"


def make_round_logo(src: Path | None, size: int) -> Image.Image:
    if src is not None and src.exists():
        pic = Image.open(src).convert("RGBA")
        w, h = pic.size
        s = min(w, h)
        pic = pic.crop(((w - s) // 2, (h - s) // 2, (w - s) // 2 + s, (h - s) // 2 + s))
        pic = pic.resize((size, size), Image.Resampling.LANCZOS)
    else:
        pic = Image.new("RGBA", (size, size), (36, 36, 42, 255))
        d = ImageDraw.Draw(pic)
        try:
            f = ImageFont.truetype(str(FONT_ARCH if FONT_ARCH.exists() else FONT_ANTON), int(size * 0.52))
        except Exception:
            f = ImageFont.load_default()
        d.text((size * 0.26, size * 0.16), "S", font=f, fill=(255, 255, 255, 255))
    mask = Image.new("L", (size, size), 0)
    ImageDraw.Draw(mask).ellipse((1, 1, size - 2, size - 2), fill=255)
    out = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    out.paste(pic, (0, 0), mask)
    return out


def postcard_layout(plate_h: int) -> dict:
    """Header sits directly above the uncropped clip; the pair is one centered block."""
    header_h = POSTCARD_HEADER_H
    group = header_h + int(plate_h)
    top = centered_block_y(group)
    return {"top": top, "header_h": header_h, "video_y": top + header_h, "plate_h": int(plate_h)}


def _caption_font(size: int):
    # Montserrat.ttf in FONTS is the Thin cut — too faint on a black header.
    candidates = [
        Path("/usr/share/fonts/truetype/sand-box/google/Poppins/Poppins-SemiBold.ttf"),
        Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"),
        Path("/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf"),
        FONT_BEBAS,
        FONT_ANTON,
    ]
    for pth in candidates:
        if pth.exists():
            try:
                return ImageFont.truetype(str(pth), size)
            except Exception:
                continue
    return ImageFont.load_default()


def render_postcard_canvas(hook: str, logo: Path | None, plate: Image.Image | None,
                           plate_h: int) -> Image.Image:
    """Black 9:16 canvas, social header, landscape plate at full width (uncropped)."""
    lay = postcard_layout(plate_h)
    canvas = Image.new("RGB", (W, H), (0, 0, 0))
    d = ImageDraw.Draw(canvas)
    logo_s = 78
    left = 48
    y = lay["top"] + 24
    logo_im = make_round_logo(logo, logo_s)
    canvas.paste(logo_im, (left, y), logo_im)
    name_font = ImageFont.truetype(str(FONT_ARCH if FONT_ARCH.exists() else FONT_ANTON), 52)
    nb = d.textbbox((0, 0), CHANNEL_NAME, font=name_font)
    nh = nb[3] - nb[1]
    nx = left + logo_s + 18
    ny = y + (logo_s - nh) // 2 - nb[1]
    d.text((nx, ny), CHANNEL_NAME, font=name_font, fill=(255, 255, 255))
    caption = " ".join((hook or HOOK_TEXT).split())
    max_w = W - left - 48
    csize = 30
    cfont = _caption_font(csize)
    while csize >= 16:
        cfont = _caption_font(csize)
        cb = d.textbbox((0, 0), caption, font=cfont)
        if cb[2] - cb[0] <= max_w:
            break
        csize -= 1
    cy = y + logo_s + 12
    d.text((left, cy), caption, font=cfont, fill=(255, 255, 255))
    if plate is not None:
        if plate.size != (W, plate_h):
            plate = plate.resize((W, plate_h), Image.Resampling.LANCZOS)
        canvas.paste(plate.convert("RGB"), (0, lay["video_y"]))
    return canvas


def render_postcard_check(dst: Path, hook: str | None = None, src: Path | None = None,
                          t: float = 5.3) -> tuple[Path, str]:
    """One 1080x1920 preview frame (no video render)."""
    global POSTCARD_LOGO_NOTE
    logo, note = find_channel_logo()
    POSTCARD_LOGO_NOTE = note
    print("Postcard logo:", note)
    src = src or (WORK / "mezzanine_stab.mp4")
    if not src.exists():
        src = WORK / "mezzanine.mp4" if (WORK / "mezzanine.mp4").exists() else ROOT / "trailer3.mp4"
    mw, mh = _mezz_size(src)
    plate_h = even(max(2, int(round(mh * W / mw))))
    tmp = template_work() / "_postcard_still.png"
    tmp.parent.mkdir(parents=True, exist_ok=True)
    run([
        "ffmpeg", "-y", "-ss", f"{t:.3f}", "-i", str(src),
        "-frames:v", "1",
        "-vf", f"scale={W}:{plate_h}:flags=lanczos,{cinematic_grade('postcard')}",
        str(tmp),
    ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    if not tmp.exists():
        raise RuntimeError("postcard still failed")
    plate = Image.open(tmp).convert("RGB")
    canvas = render_postcard_canvas(hook or HOOK_TEXT, logo, plate, plate_h)
    dst = Path(dst)
    dst.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(dst)
    print(f"Postcard check frame {dst} ({canvas.size[0]}x{canvas.size[1]})")
    return dst, note


def footage_geometry(mw: int, mh: int) -> dict:
    """Width-fit the FULL uncropped source into a 1080-wide plate. Never crop."""
    # HARD RULE: the whole original aspect ratio stays visible. Scale to the
    # 1080 width only — no zoom-to-fill, no side crop, no Ken Burns.
    scaled_w = W
    scaled_h = even(max(2, int(round(mh * W / mw))))
    return {
        "mw": mw, "mh": mh,
        "scaled_w": scaled_w, "scaled_h": scaled_h,
        "win_w_src": mw, "crop_y": 0, "no_h_crop": True,
    }


def footage_filter(geom: dict, crop_x_src: int) -> str:
    """Scale the full source to 1080 width. crop_x_src is ignored (never crop)."""
    sw, sh = geom["scaled_w"], geom["scaled_h"]
    grade = cinematic_grade()
    return (
        f"scale={sw}:{sh}:flags=lanczos,"
        f"{grade},"
        "format=yuv420p"
    )


def render_clip(mezz: Path, item: dict, geom: dict, out: Path) -> Path:
    vis = item["visible"]
    td = item["td"]
    length = vis + td  # output seconds
    speed = float(item.get("speed") or 1.0)
    if speed <= 0:
        speed = 1.0
    src_needed = length * speed if abs(speed - 1.0) > 1e-3 else length
    src_start = item["src_start"]
    max_len = max(0.05, item["src_end"] - src_start - (1 / FPS))
    if src_needed > max_len:
        if speed < 1.0:
            speed = 1.0
            src_needed = length
        if src_needed > max_len:
            src_needed = max_len
            length = max_len
    ramp = float(item.get("ramp") or 0)
    if ramp > 0:
        # src seconds needed so that integral(1/(1+k*t/L)) == length
        need = length * ramp / math.log(1 + ramp)
        if need <= max_len:
            src_needed, speed = need, 1.0
        else:
            ramp = 0.0
    crop_x = int(item.get("crop_x", 0))
    vf = footage_filter(geom, crop_x)
    if ramp > 0:
        L = src_needed
        vf = (f"{vf},setpts=PTS-STARTPTS,"
              f"setpts='({L:.4f}/{ramp:.3f})*log(1+{ramp:.3f}*T/{L:.4f})/TB',fps={FPS}")
    elif abs(speed - 1.0) > 1e-3:
        vf = f"{vf},setpts=PTS/{speed:.4f},fps={FPS}"
    preroll = 0.4
    ss = max(0.0, src_start - preroll)
    out_ss = src_start - ss
    cmd = [
        "ffmpeg", "-y",
        "-ss", f"{ss:.3f}", "-i", str(mezz),
        "-ss", f"{out_ss:.3f}",
        "-t", f"{src_needed:.3f}",
        "-vf", vf,
        "-r", str(FPS), "-vsync", "cfr",
        "-c:v", "libx264", "-preset", "veryfast", "-crf", "16",
        "-pix_fmt", "yuv420p", "-an",
        "-video_track_timescale", str(FPS * 256),
        str(out),
    ]
    run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    if not out.exists() or out.stat().st_size < 1000:
        raise RuntimeError(f"clip failed: {out}")
    return out


def _mezz_size(mezz: Path) -> tuple[int, int]:
    meas = subprocess.check_output(
        ["ffprobe", "-v", "error", "-select_streams", "v:0",
         "-show_entries", "stream=width,height", "-of", "csv=p=0", str(mezz)],
        text=True,
    ).strip()
    mw, mh = [int(x) for x in meas.split(",")[:2]]
    return mw, mh


def render_all_clips(mezz: Path, edl: list[dict], crop_h: int) -> list[Path]:
    CLIPS.mkdir(parents=True, exist_ok=True)
    mw, mh = _mezz_size(mezz)
    geom = footage_geometry(mw, mh)
    win = geom["win_w_src"]
    print(
        f"Footage plate {W}x{geom['scaled_h']} (UNCROPPED width-fit of {mw}x{mh}; "
        f"template={TEMPLATE})"
    )
    # HARD RULE: never crop. YuNet/saliency smart-crop is disabled for all templates.
    for item in edl:
        item["crop_x"] = 0

    paths = []
    with ThreadPoolExecutor(max_workers=3) as ex:
        futs = {}
        for item in edl:
            out = CLIPS / f"clip_{item['i']:02d}.mp4"
            paths.append(out)
            futs[ex.submit(render_clip, mezz, item, geom, out)] = item["i"]
        for fut in as_completed(futs):
            i = futs[fut]
            fut.result()
            print(f"  rendered clip {i:02d}")

    # End card: freeze a strong still, same uncropped width-fit (no type on it)
    still_t = END_STILL
    mezz_dur = ffprobe_dur(mezz)
    if still_t >= mezz_dur or still_t < 0:
        still_t = max(0.0, mezz_dur - 0.08)
    cx = 0
    vf_still = footage_filter(geom, cx)
    frame = WORK / "end_frame.png"
    run([
        "ffmpeg", "-y", "-ss", f"{still_t:.3f}", "-i", str(mezz),
        "-frames:v", "1", "-vf", vf_still,
        str(frame),
    ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    end = CLIPS / "clip_end.mp4"
    run([
        "ffmpeg", "-y", "-loop", "1", "-i", str(frame),
        "-t", f"{END_CARD:.2f}",
        "-vf", f"fps={FPS},format=yuv420p",
        "-c:v", "libx264", "-preset", "veryfast", "-crf", "16", "-an",
        "-pix_fmt", "yuv420p",
        str(end),
    ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    paths.append(end)
    edl.append({
        "i": len(edl), "scene": "end", "src_start": 0, "src_end": 0, "src_dur": END_CARD,
        "visible": END_CARD, "t0": edl[-1]["t1"] if edl else 0, "t1": (edl[-1]["t1"] if edl else 0) + END_CARD,
        "trans_out": "cut", "td": 0, "spectacle": 0, "skin": 0, "crop_x": cx,
    })
    return paths



# ---------------------------------------------------------------------------
# 6. Stitch + captions + music
# ---------------------------------------------------------------------------
TRANS_MAP = {
    "flash": "fadewhite",
    "whip_left": "slideleft",
    "whip_right": "slideright",
}


def stitch(paths: list[Path], edl: list[dict], out_silent: Path) -> Path:
    """Hard-cut via concat demuxer; pre-render whip/flash as short xfade clips.

    Mixing xfade and concat in one filter graph breaks CFR (rate 1/0) on
    ffmpeg 7, so each transition is a standalone clip, then everything is
    concatenated with a consistent timebase.
    """
    tmp = WORK / "stitch_parts"
    tmp.mkdir(parents=True, exist_ok=True)
    for p in tmp.glob("*"):
        p.unlink()

    enc = [
        "-c:v", "libx264", "-preset", "veryfast", "-crf", "16",
        "-pix_fmt", "yuv420p", "-r", str(FPS), "-an",
        "-video_track_timescale", str(FPS * 256),
    ]

    def emit(src, dest, ss, dur):
        run([
            "ffmpeg", "-y", "-i", str(src),
            "-ss", f"{ss:.3f}", "-t", f"{dur:.3f}",
            *enc, str(dest),
        ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    def emit_trans(a, b, typ, td, dest):
        name = TRANS_MAP[typ]
        # Pre-extract short CFR clips, then xfade. In-graph fps= after
        # a short -sseof read gives rate 1/0 on ffmpeg 7.
        pad = max(td, 5 / FPS)
        a_tmp = dest.with_name(dest.stem + "_a.mp4")
        b_tmp = dest.with_name(dest.stem + "_b.mp4")
        run([
            "ffmpeg", "-y", "-sseof", f"-{pad:.3f}", "-i", str(a),
            "-t", f"{pad:.3f}", *enc, str(a_tmp),
        ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        run([
            "ffmpeg", "-y", "-ss", "0", "-i", str(b),
            "-t", f"{pad:.3f}", *enc, str(b_tmp),
        ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        cmd = [
            "ffmpeg", "-y", "-i", str(a_tmp), "-i", str(b_tmp),
            "-filter_complex",
            f"[0:v][1:v]xfade=transition={name}:duration={td:.3f}:offset=0[v]",
            "-map", "[v]", *enc, str(dest),
        ]
        run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        a_tmp.unlink(missing_ok=True)
        b_tmp.unlink(missing_ok=True)

    parts = []
    n = len(paths)
    for i, pth in enumerate(paths):
        prev = edl[i - 1] if i > 0 else None
        cur = edl[i]
        in_td = float(prev.get("td") or 0) if (prev and prev.get("trans_out") in TRANS_MAP) else 0.0
        out_typ = cur.get("trans_out", "cut") if i < n - 1 else "cut"
        out_td = float(cur.get("td") or 0) if out_typ in TRANS_MAP else 0.0
        file_dur = ffprobe_dur(pth)
        body_d = file_dur - in_td - out_td
        if body_d < 0.08:
            # Not enough room for a transition; fall back to a hard cut
            out_td, out_typ = 0.0, "cut"
            body_d = max(0.08, file_dur - in_td)
        body = tmp / f"body_{i:02d}.mp4"
        emit(pth, body, in_td, body_d)
        parts.append(body)
        print(f"  part body {i:02d} {body_d:.2f}s")
        if out_typ in TRANS_MAP and out_td > 0:
            trans = tmp / f"trans_{i:02d}.mp4"
            emit_trans(pth, paths[i + 1], out_typ, out_td, trans)
            if trans.exists() and trans.stat().st_size > 500:
                parts.append(trans)
                print(f"  part trans {i:02d} {out_typ} {out_td:.2f}s")

    lst = tmp / "concat.txt"
    lst.write_text("".join(f"file '{p}'\n" for p in parts))
    print("Stitching", len(parts), "parts…")
    run([
        "ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", str(lst),
        "-c:v", "libx264", "-preset", "fast", "-crf", "17",
        "-pix_fmt", "yuv420p", "-r", str(FPS), "-an",
        "-video_track_timescale", str(FPS * 256),
        str(out_silent),
    ])
    return out_silent


def write_ass(edl: list[dict], dur: float, path: Path) -> Path:
    # Clamp captions to duration; last cta / title can cover end card
    def ts(t):
        t = max(0.0, t)
        h = int(t // 3600)
        m = int((t % 3600) // 60)
        s = t % 60
        return f"{h}:{m:02d}:{s:05.2f}"

    header = f"""[Script Info]
ScriptType: v4.00+
PlayResX: {W}
PlayResY: {H}
WrapStyle: 0
ScaledBorderAndShadow: yes

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Hook,Anton,78,&H00FFFFFF,&H000000FF,&H00000000,&H96000000,-1,0,0,0,100,100,0,0,1,7,3,8,50,50,110,1
Style: HookY,Anton,78,&H004AD2FF,&H000000FF,&H00000000,&H96000000,-1,0,0,0,100,100,0,0,1,7,3,8,50,50,110,1
Style: Stat,Anton,92,&H004AD2FF,&H000000FF,&H00000000,&H96000000,-1,0,0,0,100,100,0,0,1,8,4,8,50,50,110,1
Style: StatSub,Archivo Black,42,&H00FFFFFF,&H000000FF,&H00000000,&H96000000,-1,0,0,0,100,100,1,0,1,5,3,8,50,50,110,1
Style: Fact,Anton,70,&H00FFFFFF,&H000000FF,&H00000000,&H96000000,-1,0,0,0,100,100,0,0,1,7,3,8,50,50,110,1
Style: Title,Anton,86,&H00FFFFFF,&H000000FF,&H00000000,&H96000000,-1,0,0,0,100,100,0,0,1,8,4,8,50,50,110,1
Style: Cta,Anton,82,&H004AD2FF,&H000000FF,&H00000000,&H96000000,-1,0,0,0,100,100,0,0,1,8,4,8,50,50,110,1
Style: EndMain,Anton,96,&H00FFFFFF,&H000000FF,&H00000000,&H96000000,-1,0,0,0,100,100,0,0,1,8,4,8,50,50,200,1
Style: EndSub,Bebas Neue,64,&H004AD2FF,&H000000FF,&H00000000,&H96000000,-1,0,0,0,100,100,2,0,1,6,3,8,50,50,200,1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""
    # Pop animation
    pop = r"{\fad(0,180)\t(0,160,\fscx120\fscy120)\t(160,300,\fscx100\fscy100)}"
    pop2 = r"{\fad(0,180)\t(80,220,\fscx118\fscy118)\t(220,340,\fscx100\fscy100)}"
    lines_out = [header]
    # body captions (top of frame, above the plate)
    for c in CAPTIONS:
        t0, t1 = c["t0"], min(c["t1"], dur)
        if t0 >= dur:
            continue
        style = c["style"]
        if style == "hook":
            # two lines: white then gold
            if len(c["lines"]) >= 2:
                lines_out.append(
                    f"Dialogue: 0,{ts(t0)},{ts(t1)},Hook,,0,0,0,,{pop}{c['lines'][0]}"
                )
                # second line sits under first; same alignment an8, use \N in one event is cleaner
                # redo as single event with mixed colours
            # use one event with \N and colour override on line 2
            text = pop + c["lines"][0] + r"\N" + r"{\c&H4AD2FF&}" + c["lines"][1]
            # remove the first append if we do combined - rewrite last
            lines_out[-1] = f"Dialogue: 0,{ts(t0)},{ts(t1)},Hook,,0,0,0,,{text}"
        elif style == "stat":
            text = pop + c["lines"][0]
            lines_out.append(f"Dialogue: 0,{ts(t0)},{ts(t1)},Stat,,0,0,0,,{text}")
            if len(c["lines"]) > 1:
                # sub line: push lower via \N blank? Use a second event with a lower margin.
                # Archivo Black under the gold number — still top-aligned, so combine:
                text = pop + c["lines"][0] + r"\N" + r"{\fnArchivo Black\fs42\c&HFFFFFF&}" + c["lines"][1]
                lines_out[-1] = f"Dialogue: 0,{ts(t0)},{ts(t1)},Stat,,0,0,0,,{text}"
        elif style == "title":
            text = pop + c["lines"][0]
            lines_out.append(f"Dialogue: 0,{ts(t0)},{ts(t1)},Title,,0,0,0,,{text}")
        elif style == "cta":
            text = pop + c["lines"][0]
            lines_out.append(f"Dialogue: 0,{ts(t0)},{ts(t1)},Cta,,0,0,0,,{text}")
        else:
            if len(c["lines"]) >= 2:
                text = pop + c["lines"][0] + r"\N" + c["lines"][1]
            else:
                text = pop + c["lines"][0]
            lines_out.append(f"Dialogue: 0,{ts(t0)},{ts(t1)},Fact,,0,0,0,,{text}")

    # End card text (centred-upper, over the frozen plate)
    end_t0 = max(c["t0"] for c in edl if c.get("scene") == "end") if any(c.get("scene") == "end" for c in edl) else dur - END_CARD
    end_t1 = dur
    lines_out.append(
        f"Dialogue: 0,{ts(end_t0)},{ts(end_t1)},EndMain,,0,0,0,,"
        + pop + r"GODZILLA MINUS ZERO"
    )
    lines_out.append(
        f"Dialogue: 0,{ts(end_t0 + 0.15)},{ts(end_t1)},EndSub,,0,0,0,,"
        + pop2 + r"NOW IN THEATERS"
    )
    path.write_text("\n".join(lines_out))
    return path


def prepare_bed(music: dict, dur: float, dst: Path) -> Path:
    """Extract, fade, and loudnorm the music window once (no double-trim)."""
    start = music["start"]
    afilt = (
        f"atrim=start={start:.3f}:duration={dur:.3f},asetpts=PTS-STARTPTS,"
        f"afade=t=in:st=0:d=0.03,afade=t=out:st={max(0.1, dur-0.9):.3f}:d=0.9,"
        f"loudnorm=I=-11:TP=-1.5:LRA=8,aresample=48000"
    )
    run([
        "ffmpeg", "-y", "-i", str(music["path"]),
        "-af", afilt, "-c:a", "pcm_s16le", str(dst),
    ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return dst


def mux_final(silent: Path, music: dict, ass: Path, grad: Path, out: Path, dur: float,
              overlay_text: bool = OVERLAY_TEXT) -> Path:
    """Mux music. Whitebar pads footage under the hook bar; other templates differ."""
    bed = WORK / "bed.wav"
    prepare_bed(music, dur, bed)
    enc = [
        "-c:v", "libx264", "-preset", "fast", "-crf", "17",
        "-pix_fmt", "yuv420p", "-r", str(FPS),
        "-c:a", "aac", "-b:a", "192k",
        "-movflags", "+faststart",
        "-shortest",
    ]
    if TEMPLATE == "depthtext":
        # Uncropped 1080-wide plate, padded onto a black 1080x1920 canvas.
        pw, ph = _mezz_size(silent)
        y = even(max(0, (H - ph) // 2))
        if y + ph > H:
            y = 0
        vfilt = f"[0:v]pad={W}:{H}:0:{y}:color=black,format=yuv420p[v]"
        cmd = [
            "ffmpeg", "-y",
            "-i", str(silent),
            "-i", str(bed),
            "-filter_complex", vfilt,
            "-map", "[v]", "-map", "1:a",
            *enc, str(out),
        ]
        print(f"Muxing depthtext (uncropped {pw}x{ph} on black @{y}) + music…")
        run(cmd)
        return out
    if TEMPLATE == "postcard":
        logo, note = find_channel_logo()
        print("Postcard logo:", note)
        # Measure the landscape plate and build a 1080x1920 chrome still.
        pw, ph = _mezz_size(silent)
        chrome = template_work() / "postcard_chrome.png"
        canvas = render_postcard_canvas(HOOK_TEXT, logo, None, ph)
        canvas.save(chrome)
        lay = postcard_layout(ph)
        vfilt = (
            f"[0:v]fps={FPS},setpts=PTS-STARTPTS[vid];"
            f"[1:v]format=rgb24,fps={FPS}[bg];"
            f"[bg][vid]overlay=0:{lay['video_y']}:format=auto,format=yuv420p[v]"
        )
        cmd = [
            "ffmpeg", "-y",
            "-i", str(silent),
            "-loop", "1", "-i", str(chrome),
            "-i", str(bed),
            "-filter_complex", vfilt,
            "-map", "[v]", "-map", "2:a",
            "-t", f"{dur:.3f}",
            *enc, str(out),
        ]
        print("Muxing postcard chrome + uncropped landscape + music…")
        run(cmd)
        return out
    # whitebar: tight hook bar directly on the uncropped plate, one centered block
    bar = WORK / "caption_bar.png"
    if not bar.exists():
        draw_caption_bar(bar, hook_lines())
    # silent is the uncropped 1080-wide plate; bar sits flush on top, rest black.
    # The hook lives only in the bar, so xfades never fade or clip the type.
    pw, ph = _mezz_size(silent)
    bar_h = Image.open(bar).size[1]
    lay = whitebar_layout(ph, bar_h)
    layout = {
        "bar_y": lay["bar_y"], "bar_h": lay["bar_h"],
        "plate_y": lay["video_y"], "video_y": lay["video_y"],
        "plate_w": pw, "plate_h": ph, "block_h": lay["block_h"],
        "drop_t": DROP_T, "energy": ENERGY,
    }
    (WORK / "layout.json").write_text(json.dumps(layout))
    fx = ""
    if ENERGY and DROP_T is not None:
        D = float(DROP_T)
        # Temporary punch-in of the WHOLE bar+frame block (not a source crop).
        # 1.12x at the drop, decays to 1.0 in 0.35s (<= 0.4s hard cap).
        z = f"if(between(in/{FPS},{D:.3f},{D + 0.35:.3f}),1+0.12*(1-(in/{FPS}-{D:.3f})/0.35),1)"
        fx = (f",zoompan=z='{z}':x='iw/2-(iw/zoom/2)':y='ih/2-(ih/zoom/2)':d=1:s={W}x{H}:fps={FPS}"
              f",drawbox=x=0:y=0:w=iw:h=ih:color=white@1.0:t=fill:enable='between(t,{D:.3f},{D + 0.05:.3f})'"
              f",drawbox=x=0:y=0:w=iw:h=ih:color=white@0.55:t=fill:enable='between(t,{D + 0.05:.3f},{D + 0.10:.3f})'"
              f",drawbox=x=0:y=0:w=iw:h=ih:color=white@0.25:t=fill:enable='between(t,{D + 0.10:.3f},{D + 0.15:.3f})'")
    vfilt = (
        f"[0:v]pad={W}:{H}:0:{lay['video_y']}:color=black[bg];"
        f"[2:v]format=rgb24[bar];"
        f"[bg][bar]overlay=0:{lay['bar_y']}:format=auto{fx},format=yuv420p[v]"
    )
    print(f"Muxing tight whitebar (bar {bar_h}px + plate {ph}px @{lay['bar_y']}) + music…")
    cmd = [
        "ffmpeg", "-y",
        "-i", str(silent),
        "-i", str(bed),
        "-i", str(bar),
        "-filter_complex", vfilt,
        "-map", "[v]", "-map", "1:a",
        *enc, str(out),
    ]
    run(cmd)
    return out


# ---------------------------------------------------------------------------
# 7. QA
# ---------------------------------------------------------------------------
def qa(path: Path, edl: list[dict]) -> dict:
    QA.mkdir(parents=True, exist_ok=True)
    dur = ffprobe_dur(path)
    # Sample every 0.4s + around each cut
    times = list(np.arange(0.0, dur, 0.4))
    for item in edl:
        for d in (-2 / FPS, 0.0, 2 / FPS):
            t = item["t0"] + d
            if 0 <= t < dur:
                times.append(t)
        t1 = item.get("t1", None)
        if t1 is not None and 0 <= t1 < dur:
            times.append(t1 - 2 / FPS)
    times = sorted(set(round(float(t), 3) for t in times if 0 <= t < dur - 0.01))
    frames = []
    issues = []
    for i, t in enumerate(times):
        p = QA / f"f_{i:03d}_{t:.2f}s.jpg"
        subprocess.call(
            ["ffmpeg", "-y", "-ss", f"{t:.3f}", "-i", str(path),
             "-frames:v", "1", "-q:v", "3", str(p)],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )
        if not p.exists():
            issues.append(f"missing frame at {t:.2f}s")
            continue
        im = Image.open(p).convert("RGB")
        a = np.array(im)
        if a.shape[0] != H or a.shape[1] != W:
            issues.append(f"{t:.2f}s size {a.shape} != {H}x{W}")
        # Luma of the picture only — ignore a white hook bar and black letterbox.
        row_luma = a.mean(axis=(1, 2)) if a.ndim == 3 else a.mean(axis=1)
        content = (row_luma < 250) & (row_luma > 12)
        luma = float(a[content].mean()) if content.any() else float(a.mean())
        if luma < 8:
            issues.append(f"{t:.2f}s nearly black (luma={luma:.1f})")
        if luma > 245 and not (DROP_T is not None and abs(t - DROP_T) < 0.25):
            issues.append(f"{t:.2f}s nearly white (flash? luma={luma:.1f})")
        # leftover hard letterbox bars? check top 8 and a mid-plate row
        top = a[:8].mean()
        # plate region should not contain a full-width black bar
        frames.append((t, p, luma, im))

    # Contact strips: first 1s (hook), mid, cuts, overall
    def strip(items, name, cols=None):
        if not items:
            return
        ims = [it[3] for it in items]
        cols = cols or min(8, len(ims))
        tw, th = 180, 320
        rows = math.ceil(len(ims) / cols)
        sheet = Image.new("RGB", (cols * tw, rows * (th + 28)), (12, 12, 12))
        font = ImageFont.truetype(str(FONT_BEBAS), 18) if FONT_BEBAS.exists() else ImageFont.load_default()
        for i, (t, p, luma, im) in enumerate(items):
            r, c = divmod(i, cols)
            thumb = im.resize((tw, th))
            sheet.paste(thumb, (c * tw, r * (th + 28)))
            d = ImageDraw.Draw(sheet)
            d.rectangle([c * tw, r * (th + 28) + th, c * tw + tw, r * (th + 28) + th + 28], fill=(20, 20, 20))
            d.text((c * tw + 4, r * (th + 28) + th + 4), f"{t:.2f}s", fill=(255, 220, 80), font=font)
        dest = QA / name
        sheet.save(dest, quality=88)
        print("  QA sheet", dest)

    strip(frames, "qa_all.jpg", cols=10)
    hook = [f for f in frames if f[0] <= 1.05]
    strip(hook, "qa_hook.jpg", cols=len(hook) or 1)
    # around each cut
    cut_frames = []
    for item in edl[1:]:
        near = [f for f in frames if abs(f[0] - item["t0"]) <= 0.12]
        cut_frames.extend(near)
    strip(cut_frames, "qa_cuts.jpg", cols=8)

    report = {
        "path": str(path),
        "dur": dur,
        "frames_sampled": len(frames),
        "issues": issues,
        "luma_min": min((f[2] for f in frames), default=-1),
        "luma_max": max((f[2] for f in frames), default=-1),
        "luma_mean": float(np.mean([f[2] for f in frames])) if frames else 0,
    }
    (QA / "report.json").write_text(json.dumps(report, indent=2))
    print("QA issues:", issues or "none")
    return report


def _frame(path: Path, t: float) -> np.ndarray | None:
    p = QA / f"_x_{abs(hash((str(path), round(t, 3))))}.png"
    subprocess.call(["ffmpeg", "-y", "-ss", f"{t:.3f}", "-i", str(path), "-frames:v", "1", str(p)],
                    stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    if not p.exists():
        return None
    a = np.array(Image.open(p).convert("L"), dtype=np.float32)
    p.unlink(missing_ok=True)
    return a


def _corr(a: np.ndarray, b: np.ndarray) -> float:
    a = a.ravel() - a.mean()
    b = b.ravel() - b.mean()
    d = float(np.sqrt((a * a).sum() * (b * b).sum()))
    return float((a * b).sum() / d) if d > 1e-6 else 1.0


def qa_extra(out: Path, mezz: Path, src: Path, crop, edl: list[dict]) -> dict:
    """Hard QA: no-crop edge check vs source, caption visible, audio present, no shake."""
    layp = WORK / "layout.json"
    if not layp.exists():
        return {"checks": {}, "issues": ["layout.json missing"], "layout": {}}
    lay = json.loads(layp.read_text())
    checks, issues = {}, []
    # 1. geometry: plate is the FULL letterbox-free source scaled to 1080 wide
    mw, mh = _mezz_size(mezz)
    sw, sh = [int(x) for x in subprocess.check_output(
        ["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries", "stream=width,height",
         "-of", "csv=p=0", str(src)], text=True).strip().split(",")[:2]]
    ar_src, ar_plate = mw / mh, lay["plate_w"] / lay["plate_h"]
    geo_ok = (abs(ar_src - ar_plate) / ar_src < 0.01 and lay["plate_w"] == W)
    # letterbox rows removed must be black on well-lit frames, not on one fade
    lb_ok = True
    crop_w, crop_h, crop_x, crop_y = [int(x) for x in crop]
    if crop_y > 0 or crop_y + crop_h < sh:
        dur_s = ffprobe_dur(src)
        checked = 0
        for frac in (0.2, 0.35, 0.5, 0.65, 0.8, 0.9):
            t = max(0.3, min(max(dur_s - 0.3, 0.3), dur_s * frac)) if dur_s else 0.3
            f0 = _frame(src, t)
            if f0 is None or float(np.mean(f0)) < 8:
                continue
            checked += 1
            if crop_y > 0 and float(np.mean(f0[:crop_y])) >= 16:
                lb_ok = False
                break
            bot_y = crop_y + crop_h
            if bot_y < f0.shape[0] and float(np.mean(f0[bot_y:])) >= 16:
                lb_ok = False
                break
        if checked == 0:
            # Could not confirm the bars are black; do not pass a crop.
            lb_ok = False
    # 2. edge check: compare output plate vs source frame at matching times
    corrs, edges, shifts = [], [], []
    body = [it for it in edl if it.get("scene") != "end"]
    for j, it in enumerate(body):
        prev = body[j - 1] if j else None
        if it.get("ramp") or it.get("drop") or float(it.get("speed") or 1) != 1.0:
            continue
        if prev is not None and float(prev.get("td") or 0) > 0:
            continue
        dt = min(0.5, it["visible"] / 2)
        t_out = it["t0"] + dt
        if DROP_T is not None and abs(t_out - DROP_T) < 0.5:
            continue
        o = _frame(out, t_out)
        m = _frame(mezz, it["src_start"] + dt)
        if o is None or m is None:
            continue
        plate = o[lay["plate_y"]:lay["plate_y"] + lay["plate_h"], :]
        if plate.size == 0:
            continue
        mm = np.array(Image.fromarray(m.astype(np.uint8)).resize((plate.shape[1], plate.shape[0])), dtype=np.float32)
        c = _corr(plate[::4, ::4], mm[::4, ::4])
        # border ring (outer 48px on every side): a crop/zoom would misalign it
        if plate.shape[0] > 96 and plate.shape[1] > 96:
            ring = np.ones(plate.shape, bool)
            ring[48:-48, 48:-48] = False
            e = _corr(plate[ring], mm[ring])
        else:
            e = c
        (sx, sy), _ = cv2.phaseCorrelate(plate, mm)
        corrs.append(c)
        edges.append(e)
        shifts.append(float(np.hypot(sx, sy)))
        if len(corrs) >= 6:
            break
    edge_ok = (bool(corrs) and float(np.median(corrs)) > 0.75 and float(np.median(edges)) > 0.6
               and float(np.median(shifts)) < 3.0)
    checks["no_crop"] = {"ok": bool(geo_ok and lb_ok and edge_ok), "source": [sw, sh], "letterbox_free": [mw, mh],
                         "plate": [lay["plate_w"], lay["plate_h"]], "letterbox_rows_black": lb_ok,
                         "frame_corr_median": round(float(np.median(corrs)), 3) if corrs else None,
                         "edge_corr_median": round(float(np.median(edges)), 3) if edges else None,
                         "shift_px_median": round(float(np.median(shifts)), 2) if shifts else None,
                         "pairs": len(corrs)}
    # 3. caption visible: white bar with dark type directly on top of the plate
    o = _frame(out, max(0.3, (body[0]["t1"] if body else 1.0) / 2))
    cap_ok = False
    if o is not None:
        barr = o[lay["bar_y"]:lay["bar_y"] + lay["bar_h"], :]
        white = float((barr > 235).mean()) if barr.size else 0.0
        ink = float((barr < 60).mean()) if barr.size else 0.0
        cap_ok = white > 0.6 and 0.01 < ink < 0.4 and lay["plate_y"] == lay["bar_y"] + lay["bar_h"]
        checks["caption"] = {"ok": cap_ok, "white_frac": round(white, 3), "ink_frac": round(ink, 3),
                             "bar_y": lay["bar_y"], "bar_h": lay["bar_h"]}
    else:
        checks["caption"] = {"ok": False}
    # 4. audio present and audible
    r = subprocess.run(["ffmpeg", "-i", str(out), "-af", "volumedetect", "-vn", "-f", "null", "-"],
                       capture_output=True, text=True)
    mv = None
    for ln in r.stderr.splitlines():
        if "mean_volume" in ln:
            mv = float(ln.split("mean_volume:")[1].split()[0])
    checks["audio"] = {"ok": mv is not None and mv > -35, "mean_volume_db": mv}
    # 5. no shake: high-frequency jitter of global motion inside shots
    jit = []
    for it in body[:8]:
        if it["visible"] < 1.0 or it.get("drop") or it.get("ramp"):
            continue
        t0 = it["t0"] + 0.15
        tmpd = QA / "_shake"
        tmpd.mkdir(exist_ok=True)
        for f in tmpd.glob("*"):
            f.unlink()
        subprocess.call(["ffmpeg", "-y", "-ss", f"{t0:.3f}", "-i", str(out), "-t", "0.5",
                         "-vf", f"crop={W}:{lay['plate_h']}:0:{lay['plate_y']},scale=540:-2,format=gray",
                         str(tmpd / "f_%03d.png")], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        fr = [np.array(Image.open(f), dtype=np.float32) for f in sorted(tmpd.glob("*.png"))]
        if len(fr) < 6:
            continue
        sh_ = [cv2.phaseCorrelate(fr[i - 1], fr[i])[0] for i in range(1, len(fr))]
        dx = np.array([v[0] for v in sh_]); dy = np.array([v[1] for v in sh_])
        jit.append(float(np.std(np.diff(dx)) + np.std(np.diff(dy))))
    jm = float(np.median(jit)) if jit else 0.0
    checks["no_shake"] = {"ok": jm < 4.0, "jitter_px_median": round(jm, 2), "stabilized": STABILIZE, "shots": len(jit)}
    for k, v in checks.items():
        if not v.get("ok"):
            issues.append(f"QA {k} failed: {v}")
    rep_ = {"checks": checks, "issues": issues, "layout": lay}
    def _py(o):
        if isinstance(o, dict):
            return {k: _py(v) for k, v in o.items()}
        if isinstance(o, (list, tuple)):
            return [_py(v) for v in o]
        if isinstance(o, np.generic):
            return o.item()
        return o
    rep_ = _py(rep_)
    (QA / "qa_extra.json").write_text(json.dumps(rep_, indent=2))
    print("QA extra:", json.dumps(rep_["checks"]))
    return rep_


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    global PLATE_H_TARGET, STABILIZE, OVERLAY_TEXT, ZOOM, HOOK_TEXT, HOOK_LINES
    global TEMPLATE, TARGET_DUR, END_CARD, DEPTH_WORDS, CHANNEL_NAME, CLIPS, QA
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default=str(ROOT / "trailer3.mp4"))
    ap.add_argument("--music", default=str(AE / "music" / "km_Impact_Prelude.mp3"))
    ap.add_argument("--out", default=str(ROOT / "final_v4.mp4"))
    ap.add_argument("--template", choices=["whitebar", "depthtext", "postcard"], default="whitebar",
                    help="Edit template. Default: whitebar")
    ap.add_argument("--words", default=DEPTH_WORDS,
                    help="Depthtext words, max 2 (e.g. 'KING RETURNS')")
    ap.add_argument("--target-dur", type=float, default=None,
                    help="Target duration in seconds (default 32, or 15 for a depthtext sample)")
    ap.add_argument("--end-card", type=float, default=None, help="End-card length in seconds")
    ap.add_argument("--channel-name", default=CHANNEL_NAME)
    ap.add_argument("--logo", default=None, help="Optional channel logo path (postcard)")
    ap.add_argument("--check-frame", default=None,
                    help="Write a 1-frame template preview PNG and exit (used for postcard)")
    ap.add_argument("--end-still", type=float, default=None, help="Source second for the end-card still")
    ap.add_argument("--qa-only", action="store_true")
    ap.add_argument("--resume", action="store_true", help="Reuse rendered clips; stitch+mux+QA only")
    ap.add_argument("--reuse-edl", action="store_true",
                    help="Reuse edl.json + music.json but re-render plates with current defaults")
    ap.add_argument("--reuse-scenes", action="store_true",
                    help="Reuse scenes_scored.json (skip scene detect + scoring)")
    ap.add_argument("--plate-h", type=int, default=0,
                    help="Sharp plate height. Default 0 = fit-width, full frame, no side crop. "
                         "Taller values that would crop the picture are ignored.")
    ap.add_argument("--no-stabilize", action="store_true", help="Skip vidstab (on by default)")
    ap.add_argument("--captions", action="store_true",
                    help="(unused) Footage itself never gets burned-in type; the hook lives in the top bar")
    ap.add_argument("--hook", default=HOOK_TEXT,
                    help="Bold italic hook line shown in the top white bar (or postcard caption)")
    ap.add_argument("--ken-burns", type=float, default=None,
                    help="Optional Ken Burns zoom factor (off by default; e.g. 1.07)")
    ap.add_argument("--pillar", default=None, help="anime|movie|celeb|romance|funny (music + energy)")
    ap.add_argument("--item-id", default="", help="Queue item id (music picker seed)")
    ap.add_argument("--energy", action="store_true", help="Force energy edit (auto for anime/movie)")
    ap.add_argument("--qa-json", default=None, help="Write the combined QA report here")
    args = ap.parse_args()
    global END_STILL, ENERGY
    ENERGY = bool(args.energy or (args.pillar or "").lower() in ("anime", "movie"))
    if args.pillar and args.music == "auto":
        tr = pick_music(args.pillar, args.item_id)
        args.music = tr["path"]
        WORK.mkdir(parents=True, exist_ok=True)
        (WORK / "music_credit.json").write_text(json.dumps(tr, indent=2, ensure_ascii=False))
        print("Music:", tr["file"], "|", tr["credit"].splitlines()[0])
    if args.end_still is not None:
        END_STILL = float(args.end_still)

    set_template(args.template)
    DEPTH_WORDS = " ".join((args.words or DEPTH_WORDS).split()[:4])
    CHANNEL_NAME = args.channel_name or CHANNEL_NAME
    if args.target_dur is not None:
        TARGET_DUR = float(args.target_dur)
    if args.end_card is not None:
        END_CARD = float(args.end_card)
    PLATE_H_TARGET = even(args.plate_h) if args.plate_h else 0
    STABILIZE = not args.no_stabilize
    OVERLAY_TEXT = False  # never burn type onto the footage
    if args.hook:
        if args.hook.strip() != HOOK_TEXT:
            HOOK_TEXT = "\x00"  # force a fresh split for a new hook
        HOOK_LINES = hook_lines(args.hook)
        HOOK_TEXT = args.hook
    if args.ken_burns is not None:
        ZOOM = float(args.ken_burns)
    if args.logo:
        # honour a user-supplied logo by copying into the assets slot
        src_logo = Path(args.logo)
        if src_logo.exists():
            dest = AE / "assets" / "skywalker_dp.jpg"
            dest.parent.mkdir(parents=True, exist_ok=True)
            if src_logo.resolve() != dest.resolve():
                dest.write_bytes(src_logo.read_bytes())

    out_path = Path(args.out)
    # Never clobber the published whitebar render with another template.
    if (TEMPLATE != "whitebar"
            and not args.check_frame and not args.qa_only
            and out_path.resolve() == (ROOT / "final_v4.mp4").resolve()):
        raise SystemExit(
            "Refusing to overwrite final_v4.mp4 for a non-whitebar template. Pass --out."
        )

    WORK.mkdir(parents=True, exist_ok=True)
    CLIPS.mkdir(parents=True, exist_ok=True)
    QA.mkdir(parents=True, exist_ok=True)
    if TEMPLATE == "whitebar":
        draw_caption_bar(WORK / "caption_bar.png", HOOK_LINES)

    if args.check_frame:
        dst = Path(args.check_frame)
        if TEMPLATE == "postcard":
            render_postcard_check(dst, hook=HOOK_TEXT)
        elif TEMPLATE == "whitebar":
            draw_caption_bar(WORK / "caption_bar.png", HOOK_LINES)
            # one-frame composite from an existing silent or a still under the bar
            mezz = WORK / "mezzanine_stab.mp4"
            if not mezz.exists():
                mezz = WORK / "mezzanine.mp4" if (WORK / "mezzanine.mp4").exists() else Path(args.src)
            tmp = WORK / "_wb_still.png"
            geom = footage_geometry(*_mezz_size(mezz))
            cx = 0  # never crop
            run([
                "ffmpeg", "-y", "-ss", "5.300", "-i", str(mezz),
                "-frames:v", "1", "-vf", footage_filter(geom, cx), str(tmp),
            ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            foot = Image.open(tmp).convert("RGB")
            bar_im = Image.open(WORK / "caption_bar.png").convert("RGB")
            lay = whitebar_layout(foot.size[1], bar_im.size[1])
            canvas = Image.new("RGB", (W, H), (0, 0, 0))
            canvas.paste(bar_im, (0, lay["bar_y"]))
            canvas.paste(foot, (0, lay["video_y"]))
            dst.parent.mkdir(parents=True, exist_ok=True)
            canvas.save(dst)
            print("Whitebar check frame", dst)
        else:
            raise SystemExit("--check-frame for depthtext needs a rendered sample; use the main path")
        return 0

    if args.qa_only:
        edl = json.loads((WORK / "edl.json").read_text())
        qa(Path(args.out), edl)
        return

    if args.resume:
        edl = json.loads((WORK / "edl.json").read_text())
        music = json.loads((WORK / "music.json").read_text())
        paths = sorted(CLIPS.glob("clip_[0-9][0-9].mp4"))
        endp = CLIPS / "clip_end.mp4"
        if not endp.exists() or len(paths) != len(edl):
            raise SystemExit(f"resume mismatch: {len(paths)} clips vs {len(edl)} edl items")
        paths = list(paths) + [endp]
        if not any(x.get("scene") == "end" for x in edl):
            last_t1 = edl[-1]["t1"]
            edl.append({
                "i": len(edl), "scene": "end", "src_start": 0, "src_end": 0,
                "src_dur": END_CARD, "visible": END_CARD,
                "t0": last_t1, "t1": last_t1 + END_CARD,
                "trans_out": "cut", "td": 0, "spectacle": 0, "skin": 0,
            })
        silent = template_work() / "silent.mp4"
        print("=== resume: stitch ===")
        stitch(paths, edl, silent)
        silent_dur = ffprobe_dur(silent)
        print(f"  silent duration {silent_dur:.2f}s")
        if TEMPLATE == "depthtext":
            silent = apply_depth_text(silent, edl, DEPTH_WORDS, template_work() / "silent_depth.mp4")
            silent_dur = ffprobe_dur(silent)
        ass = WORK / "captions.ass"
        if OVERLAY_TEXT:
            ass = write_ass(edl, silent_dur, ass)
        mux_final(silent, music, ass, WORK / "grad.png", Path(args.out), silent_dur, OVERLAY_TEXT)
        report = qa(Path(args.out), edl)
        print("OUT", args.out, f"{ffprobe_dur(Path(args.out)):.2f}s")
        return 0 if not report["issues"] else 1

    if args.reuse_edl:
        edl = json.loads((WORK / "edl.json").read_text())
        edl = [x for x in edl if x.get("scene") != "end"]
        music = json.loads((WORK / "music.json").read_text())
        if TEMPLATE != "whitebar":
            edl = finalize_edl_for_template(edl, music, TARGET_DUR)
        src = Path(args.src)
        print("=== reuse-edl: letterbox + stabilize ===")
        crop = detect_letterbox(src)
        print("  crop w:h:x:y =", crop)
        mezz = make_mezzanine(src, WORK / "mezzanine.mp4", crop)
        if STABILIZE:
            print("=== stabilize ===")
            mezz = stabilize(mezz, WORK / "mezzanine_stab.mp4", WORK / "transforms.trf")
            print(f"  stabilized {mezz}")
        print("=== 5. Render plates ===")
        for pth in CLIPS.glob("clip_*.mp4"):
            pth.unlink()
        paths = render_all_clips(mezz, edl, crop[1])
        print("=== 6. Stitch ===")
        silent = template_work() / "silent.mp4"
        stitch(paths, edl, silent)
        silent_dur = ffprobe_dur(silent)
        print(f"  silent duration {silent_dur:.2f}s")
        if TEMPLATE == "depthtext":
            print("=== 6b. Depth text (behind subject on drops) ===")
            silent = apply_depth_text(silent, edl, DEPTH_WORDS, template_work() / "silent_depth.mp4")
            silent_dur = ffprobe_dur(silent)
        print("=== 7. Music ===")
        ass = WORK / "captions.ass"
        if OVERLAY_TEXT:
            ass = write_ass(edl, silent_dur, ass)
        mux_final(silent, music, ass, WORK / "grad.png", Path(args.out), silent_dur, OVERLAY_TEXT)
        print("=== 8. QA ===")
        report = qa(Path(args.out), edl)
        print("OUT", args.out, f"{ffprobe_dur(Path(args.out)):.2f}s", "size", Path(args.out).stat().st_size)
        print("Done.")
        return 0 if not report["issues"] else 1

    src = Path(args.src)
    music_path = Path(args.music)
    out = Path(args.out)

    print("=== 1. Letterbox detect ===")
    crop = detect_letterbox(src)
    print("  crop w:h:x:y =", crop)
    mezz = make_mezzanine(src, WORK / "mezzanine.mp4", crop)
    if STABILIZE:
        print("=== 1b. Stabilize ===")
        mezz = stabilize(mezz, WORK / "mezzanine_stab.mp4", WORK / "transforms.trf")
        print(f"  stabilized {mezz}")

    print("=== 2. Scene detect + score ===")
    scored_path = WORK / "scenes_scored.json"
    if args.reuse_scenes and scored_path.exists():
        scenes = json.loads(scored_path.read_text())
        kept = [s for s in scenes if not s.get("skip")]
        print(f"  reused {len(scenes)} scored scenes ({len(kept)} usable)")
    else:
        scenes = detect_scenes(mezz)
        print(f"  {len(scenes)} raw scenes")
        scenes = score_scenes(mezz, scenes)
        kept = [s for s in scenes if not s.get("skip")]
        print(f"  {len(kept)} usable scenes")
        scored_path.write_text(json.dumps(scenes, indent=2))

    print("=== 3. Music analysis ===")
    music = analyze_music(music_path, TARGET_DUR)
    print(f"  {Path(music['path']).name} tempo={music['tempo']:.1f} bpm "
          f"window@{music['start']:.2f}s  pulses={len(music['pulses'])}")
    (WORK / "music.json").write_text(json.dumps({k: v for k, v in music.items() if k != "abs_beats"}, indent=2))

    print("=== 4. Plan edit ===")
    edl = plan_edit(scenes, music, TARGET_DUR)
    edl = finalize_edl_for_template(edl, music, TARGET_DUR)
    if ENERGY:
        edl = apply_energy(edl, scenes, music)
        (WORK / "music.json").write_text(json.dumps({k: v for k, v in music.items() if k != "abs_beats"}, indent=2))
    for item in edl:
        extra = ""
        if item.get("show_text"):
            extra += " TEXT"
        if item.get("speed") and float(item["speed"]) != 1.0:
            extra += f" speed={item['speed']:.2f}"
        print(
            f"  [{item['t0']:5.2f}-{item['t1']:5.2f}] scene {item['scene']} "
            f"src {item['src_start']:.2f}+{item['visible']:.2f}s  "
            f"{item['trans_out']} spec={item['spectacle']}{extra}"
        )
    (WORK / "edl.json").write_text(json.dumps(edl, indent=2))
    if TEMPLATE != "whitebar":
        (template_work() / "edl.json").write_text(json.dumps(edl, indent=2))

    print("=== 5. Render plates ===")
    # clear old clips
    for p in CLIPS.glob("clip_*.mp4"):
        p.unlink()
    paths = render_all_clips(mezz, edl, crop[1])

    print("=== 6. Stitch ===")
    twork = template_work()
    silent = twork / "silent.mp4"
    stitch(paths, edl, silent)
    silent_dur = ffprobe_dur(silent)
    print(f"  silent duration {silent_dur:.2f}s")
    if TEMPLATE == "depthtext":
        print("=== 6b. Depth text (behind subject on drops) ===")
        silent = apply_depth_text(silent, edl, DEPTH_WORDS, twork / "silent_depth.mp4")
        silent_dur = ffprobe_dur(silent)

    print("=== 7. Music ===")
    ass = WORK / "captions.ass"
    if OVERLAY_TEXT:
        ass = write_ass(edl, silent_dur, ass)
    mux_final(silent, music, ass, WORK / "grad.png", out, silent_dur, OVERLAY_TEXT)

    print("=== 8. QA ===")
    # reload edl (end card appended in render)
    report = qa(out, edl)
    issues = list(report["issues"])
    if TEMPLATE == "whitebar":
        ex = qa_extra(out, mezz, src, crop, edl)
        issues += ex["issues"]
        report["checks"] = ex["checks"]
    report["issues"] = issues
    report["drop_t"] = DROP_T
    report["energy"] = ENERGY
    if args.qa_json:
        Path(args.qa_json).write_text(json.dumps(report, indent=2, default=str))
    print("OUT", out, f"{ffprobe_dur(out):.2f}s", "size", out.stat().st_size)
    print("Done.")
    return 0 if not issues else 1


if __name__ == "__main__":
    sys.exit(main() or 0)
