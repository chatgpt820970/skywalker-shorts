"""Gemini-powered helpers for one Skywalker render (scene pick, metadata, QC judge).

All functions degrade gracefully: they return None when Gemini is missing,
over budget, or misbehaves, and the caller keeps the existing heuristics.
Budget per video: meta 1 + scenes 1 + QC up to 3 (shared counter, max 5).
"""
from __future__ import annotations

import json
import re
import subprocess
from pathlib import Path

import gemini
import hook_text

BANNED = [
    "full movie", "leaked", "leak", "camrip", "free download", "download link", "fair use",
    "no copyright", "copyright free", "sub4sub", "subscribe for subscribe", "nsfw", "porn",
    "nude", "xxx", "telegram link", "watch free", "hdcam", "torrent",
]


def banned_hits(*texts: str) -> list[str]:
    blob = " ".join(t or "" for t in texts).lower()
    return [w for w in BANNED if re.search(r"(?<![a-z])" + re.escape(w) + r"(?![a-z])", blob)]


def _trend_brief(trends: dict) -> str:
    if not trends:
        return "(no trend data yet)"
    keep = {k: trends.get(k) for k in ("hook_styles", "title_patterns", "caption_styles", "hashtags", "best_length_s",
                                       "avoid", "song_picks") if trends.get(k)}
    return json.dumps(keep, ensure_ascii=False)[:1800]


# ------------------------------------------------------------------ metadata

def write_meta(item: dict, trends: dict) -> dict | None:
    """English hook (3-7 words), title, description, hashtags for one queue item."""
    film_line = (item.get("description") or "").split("\n")[0]
    prompt = f"""You write metadata for a YouTube Shorts movie/trailer edit channel (Skywalker, global + India audience).
Item: id={item.get('id')} pillar={item.get('pillar')} market={item.get('market')}
Current hook: {item.get('hook')}
Current title: {item.get('title')}
Context: {film_line}
Tags: {', '.join(item.get('tags') or [])}
Trend notes from top growing edit channels: {_trend_brief(trends)}

Rules:
- hook: PURE ENGLISH, 3-7 punchy words, curiosity or emotion, no emoji, no hashtags, no Hindi/Hinglish words.
  Film titles and character names may stay as-is. It is shown in a white bar on top of the video.
- title: "<hook> 💀 | <short scene line> | #edit" (under 95 characters, English).
- description: 2 short English lines; FIRST sentence repeats the hook exactly. Mention it is from the official trailer. Do not mention the channel name.
  No links, no music credit, no copyright claims.
- hashtags: 4-6 lowercase hashtags without '#', always include "shorts" and "edit".
Return JSON: {{"hook": str, "title": str, "description": str, "hashtags": [str]}}"""
    out = gemini.generate_json(prompt, purpose="meta", temperature=0.8)
    if not isinstance(out, dict):
        return None
    hook = " ".join(str(out.get("hook") or "").replace("\n", " ").split()).strip(" .!")
    title = " ".join(str(out.get("title") or "").split())
    desc = str(out.get("description") or "").strip()
    tags = [re.sub(r"[^a-z0-9]", "", str(t).lower().lstrip("#")) for t in (out.get("hashtags") or [])]
    tags = [t for t in tags if t][:6]
    n = len(hook.split())
    errs = []
    if not (3 <= n <= 7):
        errs.append(f"hook has {n} words")
    errs += hook_text.hook_language_errors(hook, label="hook")
    errs += hook_text.hook_language_errors(title, label="title")
    if not title or len(title) > 100:
        errs.append("bad title length")
    if not desc or not desc.lower().startswith(hook.lower()[: max(8, len(hook) // 2)]):
        desc = f"{hook}. {desc}".strip()
    if banned_hits(hook, title, desc):
        errs.append("banned words " + ",".join(banned_hits(hook, title, desc)))
    if errs:
        print("  gemini meta rejected:", "; ".join(errs))
        return None
    for must in ("shorts", "edit"):
        if must not in tags:
            tags.append(must)
    return {"hook": hook, "title": title, "description": desc, "hashtags": tags}


# ------------------------------------------------------------------ scenes

def _tile(frames: list[Path], dst: Path, cols: int = 5, w: int = 288) -> Path | None:
    try:
        from PIL import Image, ImageDraw, ImageFont
    except Exception:
        return None
    ims = []
    for f in frames:
        try:
            ims.append(Image.open(f).convert("RGB"))
        except Exception:
            ims.append(Image.new("RGB", (w, w * 9 // 16)))
    if not ims:
        return None
    h = w * 9 // 16
    rows = (len(ims) + cols - 1) // cols
    sheet = Image.new("RGB", (cols * w, rows * h), (0, 0, 0))
    dr = ImageDraw.Draw(sheet)
    try:
        font = ImageFont.truetype(str(Path(__file__).parent / "fonts" / "Anton-Regular.ttf"), 26)
    except Exception:
        font = None
    for i, im in enumerate(ims):
        im.thumbnail((w, h))
        x, y = (i % cols) * w, (i // cols) * h
        sheet.paste(im, (x, y))
        dr.rectangle([x, y, x + 44, y + 32], fill=(255, 255, 0))
        dr.text((x + 4, y + 1), str(i), fill=(0, 0, 0), font=font)
    sheet.save(dst, quality=80)
    return dst


def rank_scenes(mezz: Path, scenes: list[dict], transcript: dict, hook: str, work: Path,
                cache: Path | None = None) -> dict | None:
    """Score usable scenes 0-10 for a hooky edit and flag text/logo cards. One Gemini call."""
    kept = [s for s in scenes if not s.get("skip")]
    if len(kept) < 3:
        return None
    key = "|".join(f"{float(s['start']):.2f}-{float(s['end']):.2f}" for s in kept)
    if cache and cache.exists():
        try:
            c = json.loads(cache.read_text())
            if c.get("key") == key:
                print("  gemini[scenes]: cached")
                return c["ranks"]
        except Exception:
            pass
    kept = sorted(kept, key=lambda s: float(s.get("score") or 0), reverse=True)[:40]
    kept = sorted(kept, key=lambda s: float(s["start"]))
    tdir = work / "gem_thumbs"
    tdir.mkdir(parents=True, exist_ok=True)
    frames = []
    for i, s in enumerate(kept):
        t = (float(s["start"]) + float(s["end"])) / 2
        f = tdir / f"s{i:02d}.jpg"
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-ss", f"{t:.2f}", "-i", str(mezz), "-frames:v", "1",
                        "-vf", "scale=288:-2", str(f)], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        frames.append(f)
    sheet = _tile(frames, work / "gem_scenes.jpg")
    if not sheet:
        return None
    lines = []
    for i, s in enumerate(kept):
        words = [w.get("word", "") for w in (transcript.get("words") or [])
                 if float(s["start"]) <= float(w.get("start", -1)) < float(s["end"])]
        lines.append(f"{i}: {float(s['start']):.1f}-{float(s['end']):.1f}s speech=\"{' '.join(words)[:80]}\"")
    prompt = f"""You are the editor of a viral YouTube Shorts trailer edit. Hook on screen: "{hook}".
The contact sheet shows candidate shots, each labelled with a yellow index number (one frame per shot).
Shot list with time ranges and spoken words:
{chr(10).join(lines)}

Score every shot 0-10 for how strongly it supports the hook and keeps viewers watching (faces, emotion, action, reveals,
iconic moments, clear spoken lines). Also set card=true if the frame is mostly a studio/distributor LOGO, title card,
date/"coming soon" card, rating card, credits, or on-screen text/typography rather than story footage.
Return JSON: {{"shots": [{{"i": int, "score": number, "card": bool}}]}}"""
    out = gemini.generate_json(prompt, [sheet], purpose="scenes", temperature=0.2)
    if not isinstance(out, dict) or not isinstance(out.get("shots"), list):
        return None
    ranks = {}
    for r in out["shots"]:
        try:
            i = int(r.get("i"))
            s = kept[i]
        except Exception:
            continue
        ranks[f"{float(s['start']):.2f}-{float(s['end']):.2f}"] = {
            "score": max(0.0, min(10.0, float(r.get("score") or 0))), "card": bool(r.get("card"))}
    if cache:
        cache.write_text(json.dumps({"key": key, "ranks": ranks}))
    return ranks


def apply_scene_ranks(scenes: list[dict], ranks: dict, min_keep_s: float = 34.0) -> int:
    """Blend Gemini scores into heuristic scores; skip flagged cards only if enough footage stays."""
    if not ranks:
        return 0
    changed = 0
    kept_dur = sum(float(s["end"]) - float(s["start"]) for s in scenes if not s.get("skip"))
    for s in scenes:
        if s.get("skip"):
            continue
        r = ranks.get(f"{float(s['start']):.2f}-{float(s['end']):.2f}")
        if not r:
            continue
        d = float(s["end"]) - float(s["start"])
        if r.get("card") and kept_dur - d >= min_keep_s:
            s["skip"] = True
            s["reason"] = s["card_reason"] = "gemini_card"
            s["score"] = 0
            kept_dur -= d
            changed += 1
            continue
        g = float(r.get("score") or 5.0)
        s["gemini"] = g
        s["score"] = float(s.get("score") or 0) * (0.55 + 0.09 * g)
        changed += 1
    return changed


# ------------------------------------------------------------------ QC judge

def sample_frames(video: Path, work: Path, n: int = 8) -> Path | None:
    try:
        dur = float(subprocess.check_output(["ffprobe", "-v", "error", "-show_entries", "format=duration",
                                             "-of", "csv=p=0", str(video)], text=True).strip())
    except Exception:
        return None
    fdir = work / "gem_qc"
    fdir.mkdir(parents=True, exist_ok=True)
    frames = []
    for i in range(n):
        t = 0.6 + (dur - 1.2) * i / max(1, n - 1)
        f = fdir / f"q{i:02d}.jpg"
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-ss", f"{t:.2f}", "-i", str(video), "-frames:v", "1",
                        "-vf", "scale=270:-2", str(f)], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        frames.append(f)
    try:
        from PIL import Image
        ims = [Image.open(f).convert("RGB") for f in frames if f.exists()]
        if not ims:
            return None
        w, h = ims[0].size
        sheet = Image.new("RGB", (w * len(ims), h), (40, 40, 40))
        for i, im in enumerate(ims):
            sheet.paste(im.resize((w, h)), (i * w, 0))
        dst = work / "gem_qc_sheet.jpg"
        sheet.save(dst, quality=82)
        return dst
    except Exception:
        return None


def subtitle_text(work: Path) -> str:
    p = work / "lowerthirds.ass"
    if not p.exists():
        return ""
    out = []
    for ln in p.read_text(encoding="utf-8", errors="ignore").splitlines():
        if ln.startswith("Dialogue:"):
            txt = ln.split(",", 9)[-1]
            out.append(re.sub(r"\{[^}]*\}", "", txt).replace("\\N", " ").strip())
    return " / ".join(x for x in out if x)[:1500]


def qc_judge(video: Path, work: Path, hook: str, title: str, description: str) -> dict | None:
    sheet = sample_frames(video, work)
    if not sheet:
        return None
    subs = subtitle_text(work)
    prompt = f"""You are the final quality judge for a YouTube Short (vertical 9:16) before it is published.
The image shows 8 frames sampled evenly across the video, left to right.
Expected layout: a tight WHITE bar with black italic English hook text sitting directly on top of the full,
uncropped landscape movie frame, centered on a black canvas; only lower-third dialogue subtitles in the black
area under the picture. Nothing else should be written on the footage.
Hook in the white bar: "{hook}"
Title: "{title}"
Description: "{description[:400]}"
Burned subtitles (may be Hindi/other language if the trailer is): "{subs}"

Judge and score 0-10 each: hook_english (hook bar is readable English), layout (white bar flush on top of an uncropped
frame, picture not cut off at the sides/top), clean_footage (no studio/distributor logo intro, no title/date/rating card,
no added watermark), captions (subtitles look like real dialogue, not garbage, and not obviously mismatched to the
title topic), metadata (title/description are English, match the film, no spam).
Set reject=true ONLY for clear hard failures: a big studio logo or title/date card frame, non-English hook,
obviously cropped picture (faces cut by the frame edge because the frame was zoomed), garbage/mismatched captions,
or the title describing a different film.
Return JSON: {{"scores": {{"hook_english": n, "layout": n, "clean_footage": n, "captions": n, "metadata": n}},
"reject": bool, "reasons": [str]}}"""
    out = gemini.generate_json(prompt, [sheet], purpose="qc", temperature=0.1)
    if not isinstance(out, dict):
        return None
    sc = out.get("scores") or {}
    try:
        scores = {k: float(sc.get(k)) for k in ("hook_english", "layout", "clean_footage", "captions", "metadata")
                  if sc.get(k) is not None}
    except Exception:
        scores = {}
    low = [k for k, v in scores.items() if v < 4]
    reject = bool(out.get("reject")) or len(low) >= 2
    return {"pass": not reject, "scores": scores, "reasons": [str(r)[:200] for r in (out.get("reasons") or [])][:5],
            "low": low}
