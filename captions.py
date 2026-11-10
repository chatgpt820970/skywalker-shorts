#!/usr/bin/env python3
"""Rule-based lower-third captions that answer the hook.

4-5 lines, synced every 4s, at Y=1420 on a 1080x1920 frame (moved down
if that would cover the footage plate). Bold text, yellow/white keywords,
thick black stroke. ASS output.

Optional LLM rewrite only if an API key is already in the environment;
keys are never committed and no paid API is required.
"""
from __future__ import annotations

import os
import re
from pathlib import Path

W, H = 1080, 1920
DEFAULT_Y = 1420
INTERVAL = 4.0
N_MIN, N_MAX = 4, 5


def _answer_templates(hook: str, words: str = "") -> list[str]:
    h = (hook or "").strip()
    # "The reason why X ..."
    m = re.match(r"(?i)^the reason why\s+(.+?)([\s.!?]*)$", h)
    subject = m.group(1).strip() if m else h
    subject = re.sub(r"\s+", " ", subject).rstrip(".!?")
    short = words.strip() if words else ""
    # keep subject readable
    if len(subject) > 42:
        subject = subject[:40].rsplit(" ", 1)[0]
    name = short or (subject.split()[0] if subject else "this")
    lines = [
        f"This is why {subject}",
        f"The moment {name} changed everything" if short else f"The moment that explains everything",
        f"Nobody walked away the same after this",
        f"Watch once — you will feel it",
        f"{name} hits different in this scene" if short else "This is the scene everyone is talking about",
    ]
    # de-dupe / tighten
    out, seen = [], set()
    for ln in lines:
        k = ln.lower()
        if k not in seen:
            out.append(ln)
            seen.add(k)
    return out[:N_MAX]


def _maybe_llm(hook: str, drafts: list[str]) -> list[str]:
    """Optional improvement. Silent no-op without a key."""
    key = os.environ.get("OPENAI_API_KEY") or os.environ.get("XAI_API_KEY") or os.environ.get("ANTHROPIC_API_KEY")
    if not key:
        return drafts
    # Keep this best-effort and short; never fail the render.
    try:
        import json, urllib.request
        if os.environ.get("OPENAI_API_KEY"):
            body = {
                "model": os.environ.get("OPENAI_MODEL", "gpt-4o-mini"),
                "messages": [
                    {"role": "system", "content": "Write 5 very short YouTube Shorts lower-thirds that ANSWER the hook. Max 7 words each. No hashtags. JSON list of strings."},
                    {"role": "user", "content": hook},
                ],
                "temperature": 0.6,
            }
            req = urllib.request.Request(
                "https://api.openai.com/v1/chat/completions",
                data=json.dumps(body).encode(),
                headers={"Authorization": f"Bearer {os.environ['OPENAI_API_KEY']}", "Content-Type": "application/json"},
            )
            with urllib.request.urlopen(req, timeout=8) as r:
                data = json.loads(r.read().decode())
            txt = data["choices"][0]["message"]["content"]
            arr = json.loads(txt) if txt.strip().startswith("[") else drafts
            if isinstance(arr, list) and len(arr) >= N_MIN:
                return [str(x)[:60] for x in arr[:N_MAX]]
    except Exception:
        return drafts
    return drafts


def generate_lines(hook: str, words: str = "", dur: float = 32.0,
                   dialogue: list[dict] | None = None) -> list[dict]:
    """Lower-thirds: word-accurate dialogue cues only.

    Generic filler ('The moment X changed everything', etc.) is never used.
    If `dialogue` is a list of {t0,t1,text} cues from the transcript, those
    are returned (clamped to duration). Otherwise no lower-thirds are emitted.
    The top white hook bar is independent and stays as-is.
    """
    caps = []
    for c in dialogue or []:
        text = " ".join(str(c.get("text") or "").split())
        if not text:
            continue
        start = max(0.0, float(c.get("t0") or 0))
        end = min(float(dur) - 0.04, float(c.get("t1") or 0))
        if end <= start + 0.12:
            continue
        caps.append({"t0": round(start, 3), "t1": round(end, 3), "text": text})
    return caps


def _keyword_ass(text: str) -> str:
    """Mark the last 1-2 content words in yellow-white; the rest stay white."""
    parts = text.split()
    if not parts:
        return text
    # last two words as keywords when the line is long enough
    n_kw = 2 if len(parts) >= 4 else 1
    head, tail = parts[:-n_kw], parts[-n_kw:]
    # ASS: &H00BBGGRR (yellow ~ 00D7FFFF-ish; use 0000FFFF for yellow)
    kw = r"{\c&H00FFFF&}" + " ".join(tail) + r"{\c&H00FFFFFF&}"
    if head:
        return " ".join(head) + " " + kw
    return kw


def caption_y(plate_bottom: int | None = None) -> int:
    """Y=1420 unless that would sit on the footage; then just below the plate."""
    y = DEFAULT_Y
    if plate_bottom is not None and plate_bottom > 0:
        # keep a 16px gap under the plate if 1420 would overlap
        if y < plate_bottom + 16:
            y = min(H - 80, plate_bottom + 24)
    return int(y)


def write_ass(path: Path, caps: list[dict], plate_bottom: int | None = None, font: str = "Poppins") -> Path:
    y = caption_y(plate_bottom)
    # Alignment 8 = top-center; MarginV is offset from top when alignment is 8? 
    # In ASS, Alignment 2 is bottom-center, MarginV is from the bottom.
    # Use Alignment 8 (top center) with MarginV = y.
    # PlayRes 1080x1920.
    header = f"""[Script Info]
ScriptType: v4.00+
PlayResX: {W}
PlayResY: {H}
WrapStyle: 2
ScaledBorderAndShadow: yes

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Lower,{font},52,&H00FFFFFF,&H000000FF,&H00000000,&H80000000,-1,0,0,0,100,100,0,0,1,5,0,8,40,40,{y},1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""
    def ts(t: float) -> str:
        t = max(0.0, t)
        h = int(t // 3600)
        m = int((t % 3600) // 60)
        s = t % 60
        return f"{h:d}:{m:02d}:{s:05.2f}"

    events = []
    for c in caps:
        ev = f"Dialogue: 0,{ts(c['t0'])},{ts(c['t1'])},Lower,,0,0,0,,{_keyword_ass(c['text'])}"
        events.append(ev)
    path = Path(path)
    path.write_text(header + "\n".join(events) + "\n", encoding="utf-8")
    return path
