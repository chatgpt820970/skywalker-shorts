#!/usr/bin/env python3
"""Rule-based lower-third captions that answer the hook.

Dialogue cues only (word-accurate). Big bold white text with a thick black
outline + soft shadow, readable on a phone. Positioned in the lower third
inside the YouTube Shorts safe zone (above the like/comment UI).
"""
from __future__ import annotations

import os
import re
from pathlib import Path

W, H = 1080, 1920
# Safe zone: keep captions above ~220px from the bottom (Shorts right-rail UI).
# Alignment 8 = top-center; MarginV is pixels from the top.
SAFE_BOTTOM_PAD = 240
DEFAULT_Y = H - SAFE_BOTTOM_PAD - 90  # ~1590 — lower third, still above UI
INTERVAL = 4.0
N_MIN, N_MAX = 4, 5
# Phone-readable: larger than the old 52px, heavy outline.
FONT_SIZE = 78
OUTLINE = 8
SHADOW = 3


def _answer_templates(hook: str, words: str = "") -> list[str]:
    h = (hook or "").strip()
    m = re.match(r"(?i)^the reason why\s+(.+?)([\s.!?]*)$", h)
    subject = m.group(1).strip() if m else h
    subject = re.sub(r"\s+", " ", subject).rstrip(".!?")
    short = words.strip() if words else ""
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

    Generic filler is never used. If `dialogue` is empty, no lower-thirds.
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
    n_kw = 2 if len(parts) >= 4 else 1
    head, tail = parts[:-n_kw], parts[-n_kw:]
    kw = r"{\c&H00FFFF&}" + " ".join(tail) + r"{\c&H00FFFFFF&}"
    if head:
        return " ".join(head) + " " + kw
    return kw


def caption_y(plate_bottom: int | None = None) -> int:
    """Lower-third Y inside the Shorts safe zone (above bottom UI).

    Prefer DEFAULT_Y (~1590). If the uncropped plate extends that low, sit
    just under the plate but never below H - SAFE_BOTTOM_PAD.
    """
    floor = H - SAFE_BOTTOM_PAD
    y = DEFAULT_Y
    if plate_bottom is not None and plate_bottom > 0:
        # Prefer just under the plate when the plate is high (letterbox 16:9).
        under = int(plate_bottom) + 28
        if under < floor:
            y = max(y, under) if under > DEFAULT_Y - 80 else under
        # Plate covers the whole canvas: keep safe-zone DEFAULT_Y on top of footage
        # only when music_only has no plate gap — still clamp to floor.
        y = min(y, floor - 40)
    return int(max(1200, min(y, floor - 40)))


def write_ass(path: Path, caps: list[dict], plate_bottom: int | None = None, font: str = "Poppins") -> Path:
    y = caption_y(plate_bottom)
    # Bold (-1), thick outline, soft shadow. Alignment 8 = top-center; MarginV from top.
    header = f"""[Script Info]
ScriptType: v4.00+
PlayResX: {W}
PlayResY: {H}
WrapStyle: 2
ScaledBorderAndShadow: yes

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Lower,{font},{FONT_SIZE},&H00FFFFFF,&H000000FF,&H00000000,&H80000000,-1,0,0,0,100,100,0,0,1,{OUTLINE},{SHADOW},8,36,36,{y},1

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
        # Force bold + outline per-event in case a player ignores the style.
        body = r"{\b1\bord%d\shad%d\fsp1}" % (OUTLINE, SHADOW) + _keyword_ass(c["text"])
        events.append(f"Dialogue: 0,{ts(c['t0'])},{ts(c['t1'])},Lower,,0,0,0,,{body}")
    path = Path(path)
    path.write_text(header + "\n".join(events) + "\n", encoding="utf-8")
    return path
