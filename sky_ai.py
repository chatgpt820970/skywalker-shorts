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


# ------------------------------------------------------------------ hook + title from the chosen clip

import hashlib
import os

FALLBACK_HOOKS = [
    "{who} just crossed the line", "Nobody saw this coming", "Watch the last two seconds", "This look says everything",
    "{who} has one rule", "Wait for {who}'s next move", "This changes everything for {who}", "{who} is not here to talk",
    "One mistake changed everything", "{who} finally snaps", "Things get worse from here", "Was this the real plan?",
    "This scene gave me chills", "{who} saw it coming", "Nobody walks away from this", "Pay attention to the ending",
]


def film_name(item: dict) -> str:
    m = re.search(r"official (.+?) (trailer|teaser|promo)", item.get("description") or "", re.I)
    if m:
        return m.group(1).strip()
    w = (item.get("words") or "").strip()
    return w.title() if w else ""


def recent_openers(repo: Path, iid: str, n: int = 10) -> list[str]:
    rows = []
    for p in (repo / "results").glob("*.json"):
        try:
            r = json.loads(p.read_text())
        except Exception:
            continue
        if r.get("id") == iid or r.get("status") != "done":
            continue
        h = r.get("hook") or (r.get("title") or "").split(" 💀")[0]
        rows.append((r.get("slot") or "", hook_text.opener(h)))
    rows.sort()
    return [o for _, o in rows[-n:] if o]


def hook_errors(hook: str, avoid: list[str]) -> list[str]:
    errs = []
    n = len((hook or "").split())
    if not (3 <= n <= 7):
        errs.append(f"hook has {n} words (need 3-7)")
    errs += hook_text.hook_language_errors(hook, label="hook") + hook_text.template_errors(hook)
    if hook_text.opener(hook) in set(avoid or []):
        errs.append(f"hook opener '{hook_text.opener(hook)}' used in the last 10 videos")
    if banned_hits(hook):
        errs.append("banned words")
    if PROFANITY.search(hook or ""):
        errs.append("profanity in hook")
    return errs


def _clip_dialogue(edl: list, transcript: dict) -> list[str]:
    lines = []
    words = transcript.get("words") or []
    for e in edl:
        if e.get("scene") == "end" or e.get("act") == 2:  # act 2 is song only: its lines are not heard
            continue
        s0 = float(e.get("src_start") or 0)
        s1 = s0 + float(e.get("visible") or e.get("src_dur") or 0)
        ws = [w.get("word", "").strip() for w in words if s0 <= float(w.get("start", -1)) < s1]
        if ws:
            lines.append(" ".join(ws))
    return lines


PROFANITY = re.compile(r"(?<![a-z])(shit\w*|fuck\w*|bitch\w*|damn\w*|ass|asshole|bastard|crap|dick|piss\w*)(?![a-z])", re.I)


def fallback_hook(item: dict, dialogue: list[str], avoid: list[str]) -> str:
    """Content-based, varied fallback (never a template): a short spoken line from the clip, else a seeded pool.
    A quoted line must be a clean, complete sentence (capitalised start, no profanity), not a clipped fragment."""
    for ln in sorted(dialogue, key=len):
        t = " ".join(re.sub(r"[^A-Za-z0-9' ?!]", " ", ln).split())
        n = len(t.split())
        # Only a complete spoken sentence (ends with . ! ?) of 4-7 words; clipped fragments look broken on the bar.
        complete = ln.strip().endswith((".", "!", "?")) and t.split()[-1].lower() not in ("i", "a", "the", "to", "and")
        if 4 <= n <= 7 and complete and t.isascii() and t[:1].isupper() and not PROFANITY.search(t):
            cand = f'"{t.rstrip(".")}"'
            if not hook_errors(cand, avoid):
                return cand
    who = film_name(item) or "He"
    who = who.split(":")[0].strip()
    if len(who.split()) > 3:
        who = "He"
    seed = int(hashlib.sha1((item.get("id") or "x").encode()).hexdigest(), 16)
    for k in range(len(FALLBACK_HOOKS)):
        h = FALLBACK_HOOKS[(seed + k) % len(FALLBACK_HOOKS)].format(who=who)
        if not hook_errors(h, avoid):
            return h
    return "Nobody was ready for this"


def queue_hook_meta(item: dict, dialogue: list[str], avoid: list[str]) -> dict:
    """Keep a queue hook that already passes the bar when Gemini is unavailable.

    A rate-limit must not replace a specific English hook with a generic pool line.
    A banned or Hinglish queue hook still falls through to the seeded fallback.
    """
    queued = " ".join((item.get("hook") or "").split())
    if queued and not hook_errors(queued, avoid):
        title = " ".join((item.get("title") or "").split())
        desc = (item.get("description") or "").strip()
        title_bad = (
            not title
            or bool(hook_text.template_errors(title, label="title"))
            or bool(hook_text.hook_language_errors(title, label="title"))
            or bool(banned_hits(title))
        )
        tags = [t for t in (item.get("tags") or []) if t not in ("edit", "shorts", "trending")][:3]
        if title_bad or not desc:
            film = film_name(item)
            built = f"{queued.strip(chr(34))} | {film}" if film else queued
            title = build_title(built, tags or ["movieedit"])
            desc = build_description(queued, item, tags or ["movieedit"])
        return {
            "hook": queued,
            "title": title,
            "description": desc,
            "hashtags": tags,
            "source": "queue",
        }
    hook = fallback_hook(item, dialogue, avoid)
    film = film_name(item)
    tags = [t for t in (item.get("tags") or []) if t not in ("edit", "shorts", "trending")][:2] + ["movieedit"]
    title = f"{hook.strip(chr(34))} | {film}" if film else hook
    return {
        "hook": hook,
        "title": build_title(title, tags),
        "hashtags": tags,
        "description": build_description(hook, item, tags),
        "source": "fallback",
    }


def build_title(title: str, hashtags: list[str]) -> str:
    title = " ".join((title or "").split())[:70].rstrip(" |-")
    tags = [t for t in hashtags if t][:3]
    return (title + " " + " ".join("#" + t for t in tags)).strip()


def build_description(hook: str, item: dict, hashtags: list[str], extra: str = "") -> str:
    film = film_name(item)
    src = next((ln for ln in (item.get("description") or "").split("\n") if ln.startswith("Source:")), "")
    if not src and item.get("source_url"):
        src = (f"Source: official trailer ({item['source_url']}). All rights to the film/series belong to their owners; "
               f"short commentary edit.")
    first = hook.strip('"').rstrip(".!?") + "."
    body = f"{first} {extra.strip()} " if extra else f"{first} "
    body += f"From the official {film} trailer. Edit by Skywalker." if film else "Edit by Skywalker."
    tags = list(dict.fromkeys(hashtags + ["shorts", "edit"]))[:6]
    return body.strip() + "\n" + src + "\n" + " ".join("#" + t for t in tags)


def _context_line(item: dict) -> str:
    """Film/scene context without any old template hook sentence (so Gemini cannot paraphrase it)."""
    first = (item.get("description") or "").split("\n")[0]
    parts = [p for p in re.split(r"(?<=\.)\s+", first) if p and not hook_text.TEMPLATE_RE.search(p)
             and p.strip(". ").lower() != (item.get("hook") or "").strip(". ").lower()]
    return " ".join(parts)[:300]


def clip_hook_meta(mezz: Path, edl: list, transcript: dict, work: Path, iid: str) -> dict | None:
    """Gemini looks at frames of the CHOSEN shots + their dialogue + film context and writes the hook bar
    text, title, description line and hashtags. Cached per item (one call per render job). Never a template."""
    ip = os.environ.get("SKY_ITEM_JSON")
    if not ip or not Path(ip).exists():
        return None
    ctx = json.loads(Path(ip).read_text())
    item, avoid, trends = ctx.get("item") or {}, ctx.get("avoid_openers") or [], ctx.get("trends") or {}
    cache = Path(os.environ.get("SKY_ROOT") or work) / f"gemini_hook_{iid or 'x'}.json"
    if cache.exists():
        try:
            c = json.loads(cache.read_text())
            if c.get("hook") and not hook_errors(c["hook"], avoid):
                print("  gemini[hook]: cached")
                return c
        except Exception:
            pass
    dialogue = _clip_dialogue(edl, transcript)
    shots = [e for e in edl if e.get("scene") != "end"]
    frames = []
    tdir = work / "gem_hook"
    tdir.mkdir(parents=True, exist_ok=True)
    pick = shots if len(shots) <= 10 else [shots[round(i * (len(shots) - 1) / 9)] for i in range(10)]
    for i, e in enumerate(pick):
        t = float(e.get("src_start") or 0) + float(e.get("visible") or 1) / 2
        f = tdir / f"h{i:02d}.jpg"
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-ss", f"{t:.2f}", "-i", str(mezz), "-frames:v", "1",
                        "-vf", "scale=288:-2", str(f)], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        frames.append(f)
    sheet = _tile(frames, work / "gem_hook_sheet.jpg")
    film = film_name(item)
    out = None
    if sheet:
        prompt = f"""You write the on-screen hook for a viral YouTube Short edit of the official "{film}" trailer.
The contact sheet shows the shots that ARE in this edit, in order (yellow numbers). Spoken dialogue in the edit, in order:
{json.dumps(dialogue, ensure_ascii=False)[:1500]}
Context: {_context_line(item)} | tags: {', '.join((item.get('tags') or [])[:8])}
Pillar: {item.get('pillar')} | audience: {item.get('market')}
What is trending for hooks/titles right now: {_trend_brief(trends)}

Write ONE hook for the white bar on top of the video: 3-7 words, PURE ENGLISH, scroll-stopping and SPECIFIC to what
these shots and lines actually show (a character, a moment, a line, a stake). Look at the faces, action and
the dialogue first; avoid vague praise like "never backs down" or "changes everything". Pick the best style for this clip:
curiosity gap, shock, bold claim, emotional line, question, or quoting/teasing the key dialogue (translate to English).
What works on the fastest-growing edit channels right now: cold-open statements (most common), curiosity gaps,
contradictions ("He smiled. Then everyone ran."), sharp questions, a quoted/teased line of dialogue, "POV:" or
"<character> aura"/"<scene> core" style for character edits. Bar text is read in under 1 second: short, concrete words.
FORBIDDEN: "The reason why...", generic templates, emoji, hashtags, Hindi/Hinglish words.
Do NOT start with any of these openers (used recently): {avoid}.
Then write 5 DIFFERENT title candidates for this Short and rank them by click potential like the top movie/anime
edit channels (character-focused, emotional, curiosity): styles like "<Character>'s aura in this scene is unreal",
"POV: you're <character> and ...", "This scene hits different when you know ...", "Nobody noticed <detail> in this scene",
"<Character> didn't come to play", "The moment <character> ...". Each: pure English, max 62 chars, punchy, names the
character/actor or film, never "The reason why", no emoji, no hashtags, no plain announcement like "<Actor> is the new <X>".
Give each a click_score 0-100 (honest: would a scroller tap it?). Also a one-sentence English description line about
this exact moment, and 3 relevant lowercase hashtags (film, character/actor, genre; no '#').
Return JSON: {{"hook": str, "titles": [{{"title": str, "click_score": n}}, ...5], "line": str,
"hashtags": [str, str, str], "style": str}}"""
        out = gemini.generate_json(prompt, [sheet], purpose="hook", temperature=0.9)
    meta = None
    if isinstance(out, dict):
        hook = " ".join(str(out.get("hook") or "").split()).strip(" .")
        cands = []
        for c in (out.get("titles") or []):
            if isinstance(c, dict) and c.get("title"):
                try:
                    sc = float(c.get("click_score") or 0)
                except Exception:
                    sc = 0.0
                cands.append({"title": " ".join(str(c["title"]).split()).strip(' "'), "click_score": sc})
        if out.get("title"):
            cands.append({"title": " ".join(str(out["title"]).split()), "click_score": 0.0})
        cands.sort(key=lambda c: -c["click_score"])
        ok_c = [c for c in cands if len(c["title"]) <= 66 and not hook_text.template_errors(c["title"], label="title")
                and not hook_text.hook_language_errors(c["title"], label="title")
                and not banned_hits(c["title"]) and not PROFANITY.search(c["title"])]
        title = ok_c[0]["title"] if ok_c else ""
        print(f"  title candidates: {json.dumps(cands, ensure_ascii=False)}")
        tags = [re.sub(r"[^a-z0-9]", "", str(t).lower().lstrip("#")) for t in (out.get("hashtags") or [])]
        tags = [t for t in tags if t][:3]
        errs = hook_errors(hook, avoid) + hook_text.hook_language_errors(title, label="title") \
            + hook_text.template_errors(title, label="title") + (["empty title"] if not title else []) \
            + (["banned words"] if banned_hits(title, out.get("line") or "") else [])
        if errs:
            print("  gemini hook rejected:", "; ".join(errs))
        else:
            meta = {"hook": hook, "title": build_title(title, tags), "hashtags": tags,
                    "description": build_description(hook, item, tags, str(out.get("line") or "")),
                    "style": out.get("style"), "source": "gemini", "title_candidates": cands}
    if not meta:
        meta = queue_hook_meta(item, dialogue, avoid)
    if meta.get("source") == "gemini":  # a fallback is never cached: the next attempt asks Gemini again (budget allowing)
        cache.write_text(json.dumps(meta, ensure_ascii=False))
    return meta


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
                cache: Path | None = None, lines: list[dict] | None = None,
                act1_target: float = 13.0) -> dict | None:
    """Score usable scenes 0-10 for a hooky edit and flag text/logo cards. One Gemini call."""
    kept = [s for s in scenes if not s.get("skip")]
    if len(kept) < 3:
        return None
    key = "|".join(f"{float(s['start']):.2f}-{float(s['end']):.2f}" for s in kept) + f"|L{len(lines or [])}"
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
{_arrangement_brief(lines, act1_target)}
Return JSON: {{"shots": [{{"i": int, "score": number, "card": bool}}], "act1": [line ids in play order],
"act2": [shot indexes for the hype part, best first]}}"""
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
    arr = {"act1": [], "act2": []}
    ids = {int(l["id"]) for l in (lines or [])}
    for x in out.get("act1") or []:
        try:
            if int(x) in ids and int(x) not in arr["act1"]:
                arr["act1"].append(int(x))
        except Exception:
            pass
    for x in out.get("act2") or []:
        try:
            s2 = kept[int(x)]
            arr["act2"].append(f"{float(s2['start']):.2f}-{float(s2['end']):.2f}")
        except Exception:
            pass
    ranks["_arrangement"] = arr
    if cache:
        cache.write_text(json.dumps({"key": key, "ranks": ranks}))
    return ranks


def _arrangement_brief(lines: list[dict] | None, act1_target: float) -> str:
    if not lines:
        return ""
    ls = "\n".join(f'L{l["id"]}: {float(l["start"]):.1f}-{float(l["end"]):.1f}s "{l["text"][:90]}"' for l in lines[:40])
    return f"""
You also ARRANGE the Short in two acts. ACT 1 (first ~{act1_target:.0f}s): only the characters talking, no music, so
pick the spoken lines that tell a mini-story and make people curious (setup, tension, a strong last line that leads
into the hype). ACT 2: the hype song at full volume over the best action/visual shots cut on the beat, no dialogue.
Candidate spoken lines (id: source time, text):
{ls}
act1 = 2-6 line ids in the order they should play (total about {act1_target:.0f}s of speech, story order usually best).
act2 = 8-14 shot indexes from the contact sheet with the strongest action/visuals/reveals (no talking heads), best first."""


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


QC_KEYS = ("hook", "title", "speech_clarity", "shake", "crop", "length", "layout", "clean_footage", "captions",
           "post_worthy")


def _qc_proxy(video: Path, work: Path) -> Path | None:
    """Small 360x640 copy with audio (~1-2 MB) so Gemini can watch AND listen to the final cut inline."""
    dst = work / "gem_qc_proxy.mp4"
    r = subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(video), "-vf", "scale=360:-2,fps=12",
                        "-c:v", "libx264", "-preset", "veryfast", "-b:v", "350k", "-c:a", "aac", "-b:a", "64k",
                        "-ac", "1", str(dst)], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    if r.returncode != 0 or not dst.exists() or dst.stat().st_size > 15_000_000:
        return None
    return dst


def qc_judge(video: Path, work: Path, hook: str, title: str, description: str) -> dict | None:
    """Final judge: Gemini WATCHES and LISTENS to the finished cut (low-res proxy with audio) plus an 8-frame sheet,
    and returns pass/fail with scores. One Gemini call."""
    sheet = sample_frames(video, work)
    proxy = _qc_proxy(video, work)
    if not sheet and not proxy:
        return None
    subs = subtitle_text(work)
    try:
        dur = float(subprocess.check_output(["ffprobe", "-v", "error", "-show_entries", "format=duration",
                                             "-of", "csv=p=0", str(video)], text=True).strip())
    except Exception:
        dur = 0.0
    ac = {}
    try:
        ac = json.loads((work / "audio_check.json").read_text())
    except Exception:
        pass
    prompt = f"""You are the final quality judge for a YouTube Short (vertical 9:16, {dur:.1f}s) before it is published.
You get the finished video itself (watch it AND listen to it) plus an image of 8 frames sampled across it.
Expected: a tight WHITE bar with black italic English hook text directly on top of the full, uncropped landscape
movie frame, centered on black; only lower-third dialogue subtitles under the picture; nothing else on the footage.
Audio structure: ACT 1 (opening, ideally 40-55%; shorter is fine if the trailer has little clean dialogue) = only the characters talking, crystal clear, NO song; ACT 2 = the hype song
at full level over beat-cut action, no dialogue. The song must start once and never pop in and out between lines.
Hook in the white bar: "{hook}"
Title: "{title}"
Description: "{description[:300]}"
Burned subtitles: "{subs}"
Measured audio: {json.dumps(ac)[:400]}

Score 0-10 each:
hook (scroll-stopping, specific to this clip, pure English, readable in 1s),
title (curiosity/emotion, character-focused, English, would you tap it),
speech_clarity (every spoken line clear, no music under act-1 dialogue, song never toggling in/out, no muddy mix),
shake (10 = steady; low if footage is jittery/handheld-shaky in a distracting way),
crop (10 = full uncropped frame; low if faces/heads are cut by a zoom or the picture is cropped),
length (pacing fits {dur:.0f}s; no dead air, no abrupt cut-off),
layout (white bar flush on top, black canvas, nothing else written on the footage),
clean_footage (no studio logo intro, title/date/rating card, watermark, or repeated shot),
captions (subtitles real dialogue and in sync),
post_worthy (overall: would a top edit channel post this as is?).
Set reject=true for any clear hard failure (logo/title card, cropped picture, muddy/inaudible dialogue, garbage
captions, non-English hook/title, a different film) or if it is simply not post-worthy.
Return JSON: {{"scores": {{"hook": n, "title": n, "speech_clarity": n, "shake": n, "crop": n, "length": n,
"layout": n, "clean_footage": n, "captions": n, "post_worthy": n}}, "reject": bool, "reasons": [str]}}"""
    media = [p for p in (proxy, sheet) if p]
    out = gemini.generate_json(prompt, media, purpose="qc", temperature=0.1, timeout=150.0)
    if not isinstance(out, dict):
        return None
    sc = out.get("scores") or {}
    scores = {}
    for k in QC_KEYS:
        try:
            if sc.get(k) is not None:
                scores[k] = float(sc.get(k))
        except Exception:
            pass
    hard = {"speech_clarity": 5, "crop": 5, "layout": 5, "clean_footage": 5, "shake": 4, "hook": 5}
    low = [k for k, v in scores.items() if v < hard.get(k, 4)]
    if scores.get("post_worthy") is not None and scores["post_worthy"] < 6:
        low.append("post_worthy")
    reject = bool(out.get("reject")) or bool(low)
    return {"pass": not reject, "scores": scores, "reasons": [str(r)[:200] for r in (out.get("reasons") or [])][:6],
            "low": low, "watched_video": bool(proxy)}
