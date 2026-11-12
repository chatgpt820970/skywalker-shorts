#!/usr/bin/env python3
"""Original-dialogue transcription, mid-word snapping, and audio mix.

Keeps the trailer's spoken lines (never mute the source). Word-accurate
subtitles come from local faster-whisper — no paid API, no committed secrets.
When actors speak, dialogue sits at ~1.3 and music is ducked to ~0.22 with
sidechaincompress (volume-envelope fallback). Non-speech shots keep music at ~0.65.
"""
from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

import numpy as np

DIALOGUE_VOL = 1.3
MUSIC_SPEECH_VOL = 0.22
MUSIC_BED_VOL = 0.65
WHISPER_MODEL = os.environ.get("SKY_WHISPER_MODEL", "small")
_MODEL = None


def run_ffmpeg(cmd: list[str], **kw) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, capture_output=True, text=True, **kw)


def ffprobe_has_audio(path: Path) -> bool:
    try:
        raw = subprocess.check_output([
            "ffprobe", "-v", "error", "-select_streams", "a:0",
            "-show_entries", "stream=codec_type", "-of", "csv=p=0", str(path),
        ], text=True).strip()
        return bool(raw)
    except Exception:
        return False


def ffprobe_dur(path: Path) -> float:
    try:
        raw = subprocess.check_output([
            "ffprobe", "-v", "error", "-show_entries", "format=duration",
            "-of", "default=nw=1:nk=1", str(path),
        ], text=True).strip()
        return float(raw)
    except Exception:
        return 0.0


def _load_model():
    global _MODEL
    if _MODEL is not None:
        return _MODEL
    from faster_whisper import WhisperModel
    name = WHISPER_MODEL or "base"
    # CPU int8 is the GitHub Actions / local-box default. Never requires a key.
    try:
        _MODEL = WhisperModel(name, device="cpu", compute_type="int8")
    except Exception:
        _MODEL = WhisperModel("base", device="cpu", compute_type="int8")
    return _MODEL



def _load_audio_16k(path: Path) -> "np.ndarray":
    """16 kHz mono float32 via ffmpeg, then librosa/soundfile. Never uses PyAV."""
    import os, tempfile
    path = Path(path)
    fd, name = tempfile.mkstemp(prefix="sky_whisper_", suffix=".wav")
    os.close(fd)
    tmp = Path(name)
    try:
        r = run_ffmpeg([
            "ffmpeg", "-y", "-i", str(path),
            "-ac", "1", "-ar", "16000", "-c:a", "pcm_s16le", str(tmp),
        ])
        if r.returncode != 0 or not tmp.exists() or tmp.stat().st_size < 1000:
            raise RuntimeError((r.stderr or "ffmpeg audio extract failed")[-400:])
        try:
            import librosa
            y, _sr = librosa.load(str(tmp), sr=16000, mono=True)
        except Exception:
            import soundfile as sf
            y, _sr = sf.read(str(tmp), dtype="float32")
            if y.ndim > 1:
                y = y.mean(axis=1)
        return y.astype("float32")
    finally:
        tmp.unlink(missing_ok=True)


def transcribe_source(path: Path, cache: Path | None = None) -> dict:
    """Word-level transcript of the trailer audio. Cached as JSON next to the work dir."""
    path = Path(path)
    empty = {"words": [], "segments": [], "language": "", "ok": False, "error": None}
    if cache and cache.exists():
        try:
            data = json.loads(cache.read_text())
            if data.get("source") == str(path.resolve()) and data.get("words") is not None:
                return data
        except Exception:
            pass
    if not path.exists() or not ffprobe_has_audio(path):
        empty["error"] = "no audio on source"
        if cache:
            cache.write_text(json.dumps({**empty, "source": str(path)}, indent=2))
        return empty
    try:
        # Decode with ffmpeg → numpy. faster-whisper's av.open(..., metadata_errors=)
        # crashes on some PyAV/Python combos (seen on 3.13), and trailers are mp4s
        # librosa/soundfile cannot open directly.
        model = _load_model()
        audio = _load_audio_16k(path)
        # Names from the queue item (e.g. "Haymitch", "Peddi") so whisper does
        # not caption "Hey Mitch" / "Aponga Games".
        prompt = (os.environ.get("SKY_WHISPER_PROMPT") or "").strip()[:600] or None
        segments, info = model.transcribe(
            audio,
            word_timestamps=True,
            vad_filter=True,
            beam_size=5,
            initial_prompt=prompt,
            language=(os.environ.get("SKY_WHISPER_LANG") or None),
        )
        words, segs = [], []
        for seg in segments:
            text = (seg.text or "").strip()
            if text:
                segs.append({
                    "start": float(seg.start or 0),
                    "end": float(seg.end or 0),
                    "text": text,
                })
            for w in (seg.words or []):
                tok = (w.word or "").strip()
                if not tok:
                    continue
                words.append({
                    "start": float(w.start or 0),
                    "end": float(w.end or 0),
                    "word": tok,
                    "prob": float(getattr(w, "probability", 0) or 0),
                })
        data = {
            "ok": True,
            "error": None,
            "language": getattr(info, "language", "") or "",
            "duration": float(getattr(info, "duration", 0) or 0),
            "words": words,
            "segments": segs,
            "source": str(path.resolve()),
            "model": WHISPER_MODEL,
        }
    except Exception as e:
        data = {**empty, "error": str(e)[:400], "source": str(path)}
    if cache:
        cache.parent.mkdir(parents=True, exist_ok=True)
        cache.write_text(json.dumps(data, indent=2, ensure_ascii=False))
    return data


def vad_windows(path: Path, noise_db: float = -32.0, min_silence: float = 0.28) -> list[tuple[float, float]]:
    """Speech windows from ffmpeg silencedetect (no extra dependency)."""
    dur = ffprobe_dur(path)
    if dur <= 0 or not ffprobe_has_audio(path):
        return []
    r = run_ffmpeg([
        "ffmpeg", "-i", str(path),
        "-af", f"silencedetect=noise={noise_db}dB:d={min_silence}",
        "-f", "null", "-",
    ])
    silences = []
    for ln in (r.stderr or "").splitlines():
        if "silence_start:" in ln:
            try:
                silences.append([float(ln.split("silence_start:")[1].split()[0]), None])
            except Exception:
                pass
        elif "silence_end:" in ln and silences and silences[-1][1] is None:
            try:
                silences[-1][1] = float(ln.split("silence_end:")[1].split()[0])
            except Exception:
                pass
    if silences and silences[-1][1] is None:
        silences[-1][1] = dur
    speech, t = [], 0.0
    for a, b in silences:
        a = max(0.0, float(a))
        b = float(b if b is not None else dur)
        if a - t >= 0.18:
            speech.append((round(t, 3), round(a, 3)))
        t = max(t, b)
    if dur - t >= 0.18:
        speech.append((round(t, 3), round(dur, 3)))
    return speech


def speech_windows_from_transcript(tr: dict, merge_gap: float = 0.35, pad: float = 0.06) -> list[tuple[float, float]]:
    segs = list(tr.get("segments") or [])
    if not segs and tr.get("words"):
        segs = [{"start": w["start"], "end": w["end"]} for w in tr["words"]]
    if not segs:
        return []
    segs = sorted(segs, key=lambda s: float(s["start"]))
    out = []
    a, b = float(segs[0]["start"]) - pad, float(segs[0]["end"]) + pad
    for s in segs[1:]:
        s0, s1 = float(s["start"]) - pad, float(s["end"]) + pad
        if s0 <= b + merge_gap:
            b = max(b, s1)
        else:
            if b > a:
                out.append((max(0.0, a), b))
            a, b = s0, s1
    if b > a:
        out.append((max(0.0, a), b))
    return out


def words_overlapping(words: list[dict], t0: float, t1: float) -> list[dict]:
    return [w for w in words or [] if float(w["end"]) > t0 + 0.02 and float(w["start"]) < t1 - 0.02]


def complete_segments(segments: list[dict], t0: float, t1: float, slack: float = 0.12) -> list[dict]:
    """Segments whose full spoken line sits inside [t0, t1]."""
    hits = []
    for s in segments or []:
        if float(s["start"]) >= t0 - slack and float(s["end"]) <= t1 + slack:
            if (s.get("text") or "").strip():
                hits.append(s)
    return hits


def cuts_mid_word(t: float, words: list[dict], slack: float = 0.05) -> dict | None:
    for w in words or []:
        if float(w["start"]) + slack < t < float(w["end"]) - slack:
            return w
    return None


def speech_bonus(start: float, end: float, tr: dict | None) -> float:
    """Prefer shots that contain a complete spoken line; never reward mid-word leftovers."""
    if not tr:
        return 0.0
    segs = complete_segments(tr.get("segments") or [], start, end)
    words = words_overlapping(tr.get("words") or [], start, end)
    score = 0.0
    if segs:
        # Iconic = a complete line, longer is better (capped).
        best = max(len((s.get("text") or "").split()) for s in segs)
        score += 18.0 + min(24.0, best * 3.0)
        score += 6.0 * min(3, len(segs))
    elif words:
        score += min(10.0, len(words) * 0.8)
    # Penalize a window that starts or ends mid-word.
    if cuts_mid_word(start, tr.get("words") or []) or cuts_mid_word(end, tr.get("words") or []):
        score -= 20.0
    return score


def snap_window(src_start: float, visible: float, scene_start: float, scene_end: float,
                words: list[dict], min_vis: float = 0.70) -> tuple[float, float]:
    """Shift the cut so it does not start or end mid-word, staying inside the scene."""
    a = float(src_start)
    b = a + float(visible)
    S, E = float(scene_start), float(scene_end)
    a = min(max(a, S), E)
    b = min(max(b, a + min_vis), E)
    if b <= a or not words:
        return round(a, 3), round(max(min_vis, b - a), 3)

    def _snap_start(t: float) -> float:
        w = cuts_mid_word(t, words)
        if not w:
            return t
        # Prefer including the whole word when it still fits the scene.
        if float(w["start"]) >= S and (b - float(w["start"])) >= min_vis:
            return float(w["start"])
        if float(w["end"]) + min_vis <= b and float(w["end"]) < E:
            return float(w["end"])
        return t

    def _snap_end(t: float) -> float:
        w = cuts_mid_word(t, words)
        if not w:
            return t
        if float(w["end"]) <= E and (float(w["end"]) - a) >= min_vis:
            return float(w["end"])
        if float(w["start"]) - a >= min_vis and float(w["start"]) > S:
            return float(w["start"])
        return t

    a = _snap_start(a)
    b = _snap_end(b)
    a = min(max(a, S), E - min_vis)
    b = min(max(b, a + min_vis), E)
    # If a spoken line already started inside the window and finishes a little
    # later but still inside the scene, extend so we don't end mid-sentence.
    for w in words or []:
        w0, w1 = float(w["start"]), float(w["end"])
        if a < w0 < b < w1 <= E and (w1 - a) <= max(visible + 0.45, min_vis):
            b = w1
    a = min(max(a, S), E - min_vis)
    b = min(max(b, a + min_vis), E)
    return round(a, 3), round(max(min_vis, b - a), 3)



MIN_SHOT = 1.5
MAX_SHOT = 3.5


def fit_dialogue_window(src_start: float, visible: float, scene_start: float, scene_end: float,
                        transcript: dict | None, min_vis: float = MIN_SHOT, max_vis: float = MAX_SHOT,
                        hard_end: float | None = None) -> tuple[float, float]:
    """Return a 1.5–3.5s window that contains only complete spoken lines.

    Any whisper segment that overlaps the planned cut is either included in
    full (extend the window to the segment end/start) or dropped. A line is
    never left half-inside the shot. `hard_end` may reach slightly past the
    detected scene so a line that spills over by a beat can still finish.
    """
    S = float(scene_start)
    E = float(hard_end if hard_end is not None else scene_end)
    a = min(max(float(src_start), S), E)
    b = min(max(a + float(visible), a + min_vis), E)
    segs = list((transcript or {}).get("segments") or [])
    words = list((transcript or {}).get("words") or [])
    overlapping = [s for s in segs
                   if float(s["end"]) > a + 0.06 and float(s["start"]) < b - 0.06
                   and (s.get("text") or "").strip()]
    included = []
    lo, hi = a, b
    for s in sorted(overlapping, key=lambda x: float(x["start"])):
        s0, s1 = float(s["start"]), float(s["end"])
        if s0 < S - 0.10 or s1 > E + 0.10:
            continue  # cannot fit the full line in this scene
        trial_lo = min(lo if included else a, s0)
        trial_hi = max(hi if included else b, s1)
        trial_lo = max(trial_lo, S)
        trial_hi = min(trial_hi, E)
        if trial_lo > s0 + 0.08 or trial_hi < s1 - 0.08:
            continue
        if (trial_hi - trial_lo) > max_vis + 0.08:
            if not included and (s1 - s0) <= max_vis and s0 >= S and s1 <= E:
                lo, hi = s0, max(s1, s0 + min_vis)
                included = [s]
            continue
        lo, hi, included = trial_lo, trial_hi, included + [s]
    if not included:
        # Non-dialogue (or no complete line fits): mid-word snap, then shrink
        # so no partial segment remains inside the window.
        a2, vis2 = snap_window(a, max(min_vis, b - a), S, E, words, min_vis=min_vis)
        lo, hi = a2, a2 + vis2
        for s in segs:
            s0, s1 = float(s["start"]), float(s["end"])
            if s1 <= lo + 0.05 or s0 >= hi - 0.05:
                continue
            # Partial overlap: drop the line by shrinking, prefer keeping min_vis.
            if s0 < lo + 0.05 and lo < s1 < hi:
                lo = s1  # start after the leftover tail
            elif lo < s0 < hi and s1 > hi - 0.05:
                hi = s0  # end before the leftover head
            elif s0 >= lo and s1 <= hi:
                pass  # fully inside — should have been included; leave it
            else:
                # segment covers the whole window; can't keep any of it
                mid = (s0 + s1) / 2.0
                if abs(lo - s0) < abs(hi - s1):
                    lo = min(hi - min_vis, s1)
                else:
                    hi = max(lo + min_vis, s0)
        if hi - lo < min_vis - 0.05:
            # Give up on this tight window; keep the snapped one, captions will drop partials.
            lo, hi = a2, a2 + vis2
    if hi - lo < min_vis:
        # Grow into silence (not into a new partial line).
        need = min_vis - (hi - lo)
        # try later first (extend end)
        grow_hi = min(E, hi + need)
        hit = any(float(s["start"]) < grow_hi - 0.05 and float(s["end"]) > hi + 0.05 for s in segs)
        if not hit:
            hi = grow_hi
        need = min_vis - (hi - lo)
        if need > 0.02:
            grow_lo = max(S, lo - need)
            hit = any(float(s["end"]) > grow_lo + 0.05 and float(s["start"]) < lo - 0.05 for s in segs)
            if not hit:
                lo = grow_lo
    lo = min(max(lo, S), E)
    hi = min(max(hi, lo + 0.40), E)
    # Final clamp to max_vis, keeping complete included lines if possible.
    if hi - lo > max_vis + 0.08 and included:
        # keep from the first included start, cap at max_vis if all lines still fit
        lo = max(lo, float(included[0]["start"]))
        last_end = max(float(s["end"]) for s in included)
        if last_end - lo <= max_vis + 0.08:
            hi = max(last_end, lo + min_vis)
        else:
            hi = min(hi, lo + max_vis)
    elif hi - lo > max_vis + 0.08:
        hi = lo + max_vis
    return round(lo, 3), round(max(0.40, hi - lo), 3)


def segment_for_word(word: dict, segments: list[dict]) -> dict | None:
    w0, w1 = float(word.get("src_start") or word.get("start") or 0), float(word.get("src_end") or word.get("end") or 0)
    mid = 0.5 * (w0 + w1)
    for s in segments or []:
        if float(s["start"]) - 0.08 <= mid <= float(s["end"]) + 0.08:
            return s
    return None


def word_line_is_complete(word: dict, item: dict, segments: list[dict], slack: float = 0.10) -> bool:
    """True if the spoken WORD itself is fully audible inside the clip.

    The old rule required the whole whisper line to fit inside the clip.
    Trailer shots are 1-3s and lines run across cuts, so ~35 real words per
    short were dropped and QA failed with "no dialogue subtitles". Showing
    exactly the words that are heard is still word-accurate; only a word the
    cut slices in half is dropped.
    """
    a = float(item.get("src_start") or 0)
    b = a + float(item.get("visible") or 0)
    w0 = float(word.get("src_start") or word.get("start") or 0)
    w1 = float(word.get("src_end") or word.get("end") or 0)
    if not (w0 >= a - slack and w1 <= b + slack):
        return False
    seg = segment_for_word(word, segments)
    if not seg:
        return True
    s0, s1 = float(seg["start"]), float(seg["end"])
    if s0 >= a - slack and s1 <= b + slack:
        return True
    # Line runs across the cut: keep the heard words when most of the line is heard,
    # drop stray 1-word fragments of a line that is mostly in another shot.
    overlap = max(0.0, min(b, s1) - max(a, s0))
    return overlap >= 0.5 * max(0.01, s1 - s0)


def drop_incomplete_lines(mapped: list[dict], edl: list[dict], segments: list[dict]) -> list[dict]:
    """Safety net: never emit a subtitle for a line the cut slices in half."""
    by_i = {it.get("i"): it for it in (edl or []) if it.get("scene") not in (None, "end")}
    kept = []
    dropped = 0
    for w in mapped or []:
        it = by_i.get(w.get("clip"))
        if it is None:
            kept.append(w)
            continue
        if word_line_is_complete(w, it, segments):
            kept.append(w)
        else:
            dropped += 1
    if dropped:
        print(f"  dropped {dropped} word(s) from incomplete dialogue lines")
    return kept


def _out_time(item: dict, src_t: float) -> float | None:

    """Map a source timestamp onto the output timeline for one EDL item."""
    a = float(item.get("src_start") or 0)
    vis = float(item.get("visible") or 0)
    b = a + vis
    if src_t < a - 0.02 or src_t > b + 0.02:
        return None
    speed = float(item.get("speed") or 1.0) or 1.0
    # Video setpts=PTS/speed → output elapsed = (src-a)/speed. Ramp is approximated.
    ramp = float(item.get("ramp") or 0)
    dt = max(0.0, src_t - a)
    if ramp > 0 and vis > 0:
        # Inverse of setpts = (L/k)*log(1+k*T/L): T_out = (L/k)*log(1+k*dt/L_src)
        # src_needed = vis * ramp / log(1+ramp); we approximate with average speed.
        avg = (math_log_speed(ramp))
        return float(item["t0"]) + dt / max(0.25, avg)
    return float(item["t0"]) + dt / max(0.25, speed)


def math_log_speed(ramp: float) -> float:
    # mean speed across a 1→(1+ramp) ramp is ramp/log(1+ramp)
    import math
    if ramp <= 0:
        return 1.0
    return float(ramp / math.log(1.0 + ramp))


def map_words_to_output(edl: list[dict], words: list[dict]) -> list[dict]:
    """Project source words onto output time, dropping anything cut mid-window without overlap."""
    mapped = []
    body = [it for it in (edl or []) if it.get("scene") not in (None, "end")]
    for w in words or []:
        w0, w1 = float(w["start"]), float(w["end"])
        for it in body:
            a = float(it.get("src_start") or 0)
            b = a + float(it.get("visible") or 0)
            if w1 <= a + 0.02 or w0 >= b - 0.02:
                continue
            # Skip a word whose start or end was cut off (don't show a partial leftover).
            if w0 < a - 0.04 or w1 > b + 0.04:
                # still keep it if most of the word is inside
                inside = min(w1, b) - max(w0, a)
                if inside < 0.6 * max(0.01, w1 - w0):
                    continue
            t0 = _out_time(it, max(w0, a))
            t1 = _out_time(it, min(w1, b))
            if t0 is None or t1 is None or t1 <= t0:
                continue
            mapped.append({
                "t0": float(t0), "t1": float(t1),
                "word": w.get("word") or "",
                "src_start": w0, "src_end": w1,
                "clip": it.get("i"),
                "prob": float(w.get("prob", 1.0) if w.get("prob") is not None else 1.0),
            })
            break
    return mapped


MIN_WORD_PROB = float(os.environ.get("SKY_MIN_WORD_PROB", "0.2"))


CUE_MAX_WORDS = int(os.environ.get("SKY_CUE_WORDS", "3"))  # research: 2-3 words per beat read fastest on Shorts


def caption_cues(mapped_words: list[dict], max_chars: int = 42, max_dur: float = 3.6,
                 max_gap: float = 0.45, max_words: int | None = None) -> list[dict]:
    """Pack mapped words into readable lower-third cues. One cue per spoken breath."""
    # MUST sort by output time. Words arrive in source order, but the EDL
    # can place a later trailer line earlier in the short.
    mapped_words = sorted(mapped_words or [], key=lambda w: (float(w.get("t0") or 0), float(w.get("t1") or 0)))
    cues: list[dict] = []
    buf: list[dict] = []

    def flush():
        if not buf:
            return
        text = " ".join((w.get("word") or "").strip() for w in buf).strip()
        text = " ".join(text.split())
        if not text:
            buf.clear()
            return
        # Hallucinated junk (Arabic/CJK glyphs, U+FFFD) on music beds is never shown.
        if any(("\u0600" <= ch <= "\u06ff") or ("\u4e00" <= ch <= "\u9fff") or ch == "\ufffd" for ch in text):
            print(f"  dropped junk cue: {text!r}")
            buf.clear()
            return
        # Word-accurate or nothing: a cue containing a word whisper itself is
        # unsure of (likely misheard name) is dropped instead of shown wrong.
        if min(float(w.get("prob", 1.0)) for w in buf) < MIN_WORD_PROB:
            print(f"  dropped low-confidence cue: {text!r}")
            buf.clear()
            return
        t0 = float(buf[0]["t0"])
        t1 = max(float(buf[-1]["t1"]), t0 + 0.35)
        # Hold the last word a beat so the line is readable.
        t1 = min(t1 + 0.18, t0 + max_dur)
        cues.append({"t0": round(t0, 3), "t1": round(t1, 3), "text": text})
        buf.clear()

    for w in mapped_words or []:
        tok = (w.get("word") or "").strip()
        if not tok:
            continue
        if not buf:
            buf.append(w)
            continue
        gap = float(w["t0"]) - float(buf[-1]["t1"])
        trial = " ".join((x.get("word") or "").strip() for x in buf + [w])
        dur = float(w["t1"]) - float(buf[0]["t0"])
        same_clip = buf[-1].get("clip") == w.get("clip")
        mw = max_words or CUE_MAX_WORDS
        if (not same_clip) or gap > max_gap or len(trial) > max_chars or dur > max_dur or len(buf) >= mw:
            flush()
            buf.append(w)
        else:
            buf.append(w)
    flush()
    # Don't let a hold-over run into the next cue.
    for i in range(len(cues) - 1):
        cues[i]["t1"] = round(min(float(cues[i]["t1"]), float(cues[i + 1]["t0"]) - 0.04), 3)
        if cues[i]["t1"] <= cues[i]["t0"]:
            cues[i]["t1"] = round(cues[i]["t0"] + 0.20, 3)
    return cues


def map_windows_to_output(edl: list[dict], windows: list[tuple[float, float]]) -> list[tuple[float, float]]:
    out = []
    body = [it for it in (edl or []) if it.get("scene") not in (None, "end")]
    for a, b in windows or []:
        for it in body:
            s0 = float(it.get("src_start") or 0)
            s1 = s0 + float(it.get("visible") or 0)
            lo, hi = max(a, s0), min(b, s1)
            if hi - lo < 0.08:
                continue
            t0, t1 = _out_time(it, lo), _out_time(it, hi)
            if t0 is None or t1 is None or t1 - t0 < 0.08:
                continue
            out.append((round(float(t0), 3), round(float(t1), 3)))
    # merge overlaps
    if not out:
        return []
    out.sort()
    merged = [list(out[0])]
    for a, b in out[1:]:
        if a <= merged[-1][1] + 0.05:
            merged[-1][1] = max(merged[-1][1], b)
        else:
            merged.append([a, b])
    return [(a, b) for a, b in merged]


def _silence_wav(dst: Path, dur: float, sr: int = 48000) -> Path:
    run_ffmpeg([
        "ffmpeg", "-y", "-f", "lavfi", "-i", f"anullsrc=r={sr}:cl=stereo",
        "-t", f"{max(0.05, dur):.3f}", "-c:a", "pcm_s16le", str(dst),
    ])
    return dst


def extract_dialogue_wav(src: Path, edl: list[dict], dst: Path) -> bool:
    """Build a continuous dialogue bed that follows the EDL (silence on the end card)."""
    src, dst = Path(src), Path(dst)
    if not src.exists() or not ffprobe_has_audio(src):
        return False
    tmp = dst.parent / "dial_parts"
    tmp.mkdir(parents=True, exist_ok=True)
    for p in tmp.glob("*"):
        p.unlink()
    parts = []
    for i, it in enumerate(edl or []):
        vis = float(it.get("visible") or 0)
        part = tmp / f"p_{i:02d}.wav"
        if it.get("scene") == "end" or vis <= 0.04:
            _silence_wav(part, max(0.05, vis))
            parts.append(part)
            continue
        ss = float(it.get("src_start") or 0)
        speed = float(it.get("speed") or 1.0) or 1.0
        ramp = float(it.get("ramp") or 0)
        if ramp > 0:
            speed = math_log_speed(ramp)
        # Source seconds needed so atempo=speed yields `vis` output seconds.
        src_dur = vis * speed
        af = "aformat=sample_fmts=s16:sample_rates=48000:channel_layouts=stereo"
        if abs(speed - 1.0) > 0.03:
            # atempo accepts 0.5–2.0; chain if needed.
            sp = speed
            chain = []
            while sp < 0.5:
                chain.append("atempo=0.5")
                sp /= 0.5
            while sp > 2.0:
                chain.append("atempo=2.0")
                sp /= 2.0
            chain.append(f"atempo={sp:.4f}")
            af += "," + ",".join(chain)
        af += f",apad=pad_dur={vis:.3f},atrim=0:{vis:.3f},asetpts=PTS-STARTPTS"
        r = run_ffmpeg([
            "ffmpeg", "-y", "-ss", f"{ss:.3f}", "-i", str(src),
            "-t", f"{src_dur:.3f}", "-vn", "-af", af,
            "-c:a", "pcm_s16le", str(part),
        ])
        if r.returncode != 0 or not part.exists() or part.stat().st_size < 100:
            _silence_wav(part, vis)
        parts.append(part)
    if not parts:
        return False
    lst = tmp / "concat.txt"
    lst.write_text("".join(f"file '{p}'\n" for p in parts))
    r = run_ffmpeg([
        "ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", str(lst),
        "-c:a", "pcm_s16le", str(dst),
    ])
    return r.returncode == 0 and dst.exists() and dst.stat().st_size > 1000


def _speech_enable(windows: list[tuple[float, float]]) -> str:
    if not windows:
        return "0"
    parts = [f"between(t,{a:.3f},{b:.3f})" for a, b in windows]
    return "+".join(parts)


def mix_dialogue_and_music(dialogue: Path, music: Path, dst: Path, dur: float,
                           speech_out: list[tuple[float, float]] | None = None) -> Path:
    """Dialogue at 1.3; music 0.65, ducked to ~0.22 while actors speak.

    Prefers ffmpeg sidechaincompress; falls back to an explicit volume envelope
    built from speech windows if the filter is missing.
    """
    dialogue, music, dst = Path(dialogue), Path(music), Path(dst)
    enable = _speech_enable(speech_out or [])
    # Pre-duck the bed toward 0.22 during known speech, then let sidechain
    # handle the attack/release around the actual waveform.
    # 0.22/0.65 ≈ 0.338
    if enable != "0":
        music_vol = (
            f"volume={MUSIC_BED_VOL},"
            f"volume=eval=frame:volume='if({enable},{MUSIC_SPEECH_VOL/MUSIC_BED_VOL:.4f},1)'"
        )
    else:
        music_vol = f"volume={MUSIC_BED_VOL}"
    dial_af = f"highpass=f=80,volume={DIALOGUE_VOL}"
    has_sc = True
    try:
        filt = subprocess.check_output(["ffmpeg", "-hide_banner", "-filters"],
                                       stderr=subprocess.STDOUT, text=True)
        has_sc = "sidechaincompress" in filt
    except Exception:
        has_sc = False
    if has_sc:
        graph = (
            f"[0:a]{dial_af},aformat=sample_fmts=fltp:sample_rates=48000:channel_layouts=stereo[d];"
            f"[1:a]{music_vol},aformat=sample_fmts=fltp:sample_rates=48000:channel_layouts=stereo[m];"
            f"[m][d]sidechaincompress=threshold=0.03:ratio=8:attack=20:release=280:makeup=1:mix=1[md];"
            f"[md][d]amix=inputs=2:duration=first:dropout_transition=0:normalize=0,"
            f"alimiter=limit=0.95,aformat=sample_fmts=s16:sample_rates=48000:channel_layouts=stereo[a]"
        )
    else:
        graph = (
            f"[0:a]{dial_af},aformat=sample_fmts=fltp:sample_rates=48000:channel_layouts=stereo[d];"
            f"[1:a]{music_vol},aformat=sample_fmts=fltp:sample_rates=48000:channel_layouts=stereo[m];"
            f"[m][d]amix=inputs=2:duration=first:dropout_transition=0:normalize=0,"
            f"alimiter=limit=0.95,aformat=sample_fmts=s16:sample_rates=48000:channel_layouts=stereo[a]"
        )
    r = run_ffmpeg([
        "ffmpeg", "-y",
        "-i", str(dialogue), "-i", str(music),
        "-filter_complex", graph, "-map", "[a]",
        "-t", f"{dur:.3f}",
        "-c:a", "pcm_s16le", str(dst),
    ])
    if r.returncode != 0 or not dst.exists():
        # Last resort: just mix at the target volumes without ducking.
        run_ffmpeg([
            "ffmpeg", "-y", "-i", str(dialogue), "-i", str(music),
            "-filter_complex",
            f"[0:a]volume={DIALOGUE_VOL}[d];[1:a]volume={MUSIC_BED_VOL}[m];"
            f"[d][m]amix=inputs=2:duration=first:normalize=0[a]",
            "-map", "[a]", "-t", f"{dur:.3f}", "-c:a", "pcm_s16le", str(dst),
        ])
    return dst


def style_music(src: Path, dst: Path, pillar: str = "", mood: str = "", bpm: float | None = None) -> tuple[Path, str]:
    """Apply a slowed+reverb style to a copyright-safe track. Never fetches new audio.

    Romance / emotional: ~0.88x + reverb. Phonk / anime edits: slight slow (~0.92x).
    The credit still points at the original NCS / Kevin MacLeod track.
    """
    src, dst = Path(src), Path(dst)
    pillar = (pillar or "").lower()
    mood = (mood or "").lower()
    rate = None
    if pillar == "romance" or "emotional" in mood:
        rate = 0.88
    elif "phonk" in mood or pillar in ("anime", "celeb"):
        rate = 0.92
    if not rate or not src.exists():
        return src, ""
    # asetrate changes pitch + duration (classic slowed+reverb feel).
    af = (
        f"asetrate=44100*{rate:.3f},aresample=44100,"
        f"aecho=0.8:0.75:40|90|140:0.35|0.25|0.18"
    )
    r = run_ffmpeg([
        "ffmpeg", "-y", "-i", str(src), "-af", af,
        "-c:a", "pcm_s16le", str(dst),
    ])
    if r.returncode == 0 and dst.exists() and dst.stat().st_size > 1000:
        return dst, f"slowed+reverb x{rate:.2f}"
    return src, ""
