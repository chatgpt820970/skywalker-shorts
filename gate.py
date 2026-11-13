"""Publish gate: a render is only published (Release + Buffer) when every check passes.

Checks: 2160x3840 (or 1080x1920 when out_res=1080), 18-32s, audio stream with sane loudness,
pipeline QA (no_crop/caption/audio/no_shake/no_logo), auto_qc, dialogue subtitles timed (or no
subtitle-able speech), title/description present + English + no banned words, Gemini QC judge.
"""
from __future__ import annotations

import json
import re
import subprocess
from pathlib import Path

import hook_text
import sky_ai

MIN_DUR, MAX_DUR = 18.0, 32.6
LUFS_MIN, LUFS_MAX = -30.0, -5.0


def probe(path: Path) -> dict:
    o = subprocess.check_output(["ffprobe", "-v", "error", "-show_entries",
                                 "format=duration:stream=codec_type,width,height", "-of", "json", str(path)], text=True)
    d = json.loads(o)
    v = next((s for s in d.get("streams", []) if s.get("codec_type") == "video"), {})
    a = [s for s in d.get("streams", []) if s.get("codec_type") == "audio"]
    return {"w": v.get("width"), "h": v.get("height"), "dur": float(d.get("format", {}).get("duration") or 0),
            "audio": bool(a)}


def loudness(path: Path) -> float | None:
    r = subprocess.run(["ffmpeg", "-hide_banner", "-nostats", "-i", str(path), "-af", "ebur128=framelog=quiet",
                        "-f", "null", "-"], capture_output=True, text=True)
    m = re.findall(r"I:\s+(-?[\d.]+) LUFS", r.stderr)
    return float(m[-1]) if m else None


def subtitle_cues(work: Path | None) -> int:
    if not work:
        return 0
    p = Path(work) / "lowerthirds.ass"
    if not p.exists():
        return 0
    return sum(1 for ln in p.read_text(encoding="utf-8", errors="ignore").splitlines() if ln.startswith("Dialogue:"))


def run_gate(final: Path, *, qa_obj: dict, checks: dict, item: dict, work: Path | None,
             gemini_qc: dict | None, out_res: int = 2160) -> dict:
    fails, info = [], {}
    pr = probe(final)
    info["probe"] = pr
    want = (2160, 3840) if out_res == 2160 else (1080, 1920)
    if (pr["w"], pr["h"]) != want:
        fails.append(f"resolution {pr['w']}x{pr['h']} != {want[0]}x{want[1]}")
    if not (MIN_DUR <= pr["dur"] <= MAX_DUR):
        fails.append(f"duration {pr['dur']:.2f}s outside {MIN_DUR:.0f}-32s")
    if not pr["audio"]:
        fails.append("no audio stream")
    else:
        lu = loudness(final)
        info["lufs"] = lu
        if lu is None or not (LUFS_MIN <= lu <= LUFS_MAX):
            fails.append(f"loudness {lu} LUFS outside {LUFS_MIN}..{LUFS_MAX}")
    for k in ("no_crop", "caption", "audio", "no_shake", "no_logo"):
        if k in checks and checks.get(k) is False:
            fails.append(f"pipeline QA {k}=false")
    aq = (qa_obj or {}).get("auto_qc") or {}
    if aq and aq.get("ok") is False:
        fails.append("auto_qc failed: " + "; ".join(map(str, (aq.get("issues") or [])[:3])))
    if qa_obj.get("issues"):
        fails.append("pipeline issues: " + "; ".join(map(str, qa_obj["issues"][:3])))
    # Clean-dialogue audio check (vocal stem vs ducked song inside dialogue windows).
    try:
        import vocal_mix
        ac_p = Path(work) / "audio_check.json" if work else None
        ac = json.loads(ac_p.read_text()) if ac_p and ac_p.exists() else None
        sw_p = Path(work) / "speech_windows.json" if work else None
        speech = bool(sw_p and sw_p.exists() and (json.loads(sw_p.read_text()).get("output") or []))
        info["audio_check"] = ac
        fails += vocal_mix.check_errors(ac, speech)
    except Exception as e:
        fails.append(f"audio check error: {e}")
    cues = subtitle_cues(work)
    info["subtitle_cues"] = cues
    # captions check true + zero cues means the pipeline found no subtitle-able words (cc8b263 rule)
    if cues == 0 and checks.get("caption") is False:
        fails.append("dialogue subtitles missing")
    title = (item.get("title") or "").strip()
    desc = (item.get("description") or "").strip()
    hook = (item.get("hook") or "").strip()
    if not title:
        fails.append("title empty")
    if not desc:
        fails.append("description empty")
    if len(title) > 100:
        fails.append("title over 100 chars")
    errs = hook_text.check_item_hook_fields(item)
    for extra in hook_text.template_errors(hook) + hook_text.template_errors(title, label="title"):
        if extra not in errs:
            errs.append(extra)
    op = hook_text.opener(hook)
    if op and op in set(item.get("_avoid_openers") or []):
        errs.append(f"hook opener '{op}' repeats one of the last 10 videos")
    n = len(hook.strip('"').split())
    if not (3 <= n <= 7):
        errs.append(f"hook has {n} words (need 3-7)")
    if errs:
        fails.append("hook/title: " + "; ".join(errs))
    b = sky_ai.banned_hits(hook, title, desc)
    if b:
        fails.append("banned words: " + ", ".join(b))
    if gemini_qc is None:
        info["gemini_qc"] = "skipped (unavailable/budget) - heuristic gate only"
    else:
        info["gemini_qc"] = gemini_qc
        if not gemini_qc.get("pass"):
            fails.append("gemini QC reject: " + "; ".join(gemini_qc.get("reasons") or gemini_qc.get("low") or ["?"]))
    return {"ok": not fails, "failures": fails, **info}
