"""Trending edit-song library picker (music/library.json; mp3s from the 'music' GitHub release)."""
from __future__ import annotations

import json
import os
from pathlib import Path

LIB_PILLAR_GENRE = {
    "anime": ("phonk", "rock", "dark"), "movie": ("phonk", "rock", "dark", "mass", "slowed"),
    "celeb": ("mass", "swag", "phonk", "punjabi"), "romance": ("romance", "slowed", "bollywood", "urdu", "pop"),
    "funny": ("funk", "dance", "phonk"),
}


def pick(AE: Path, pillar: str, seed: str = "", exclude: list[str] | None = None, salt: str = "") -> dict | None:
    """Trending edit songs from music/library.json (mp3s fetched from the 'music' release into music/lib).

    Fit = pillar listed on the track + genre words for the pillar + market (IN items prefer Indian songs)
    + song picks from trends/latest.json. `exclude` carries the last 8 renders' songs and earlier attempts,
    so the same song never repeats within 8 consecutive videos. Returns None to fall back to credits.json.
    """
    import hashlib
    if os.environ.get("SKY_MUSIC_SOURCE", "library") != "library":
        return None
    lp = AE / "music" / "library.json"
    if not lp.exists():
        return None
    try:
        lib = json.loads(lp.read_text())
    except Exception:
        return None
    ldir = AE / (lib.get("dir") or "music/lib")
    tracks = [t for t in lib.get("tracks") or [] if (ldir / t["file"]).exists()]
    if not tracks:
        print("  music library: no mp3s present locally; falling back to credits.json")
        return None
    p = (pillar or "movie").lower()
    market = (os.environ.get("SKY_MARKET") or "").upper()
    excl = {x for x in (exclude or []) if x}
    picks = []
    try:
        tr = json.loads((AE / "trends" / "latest.json").read_text())
        picks = [str(x).lower() for x in (tr.get("song_picks") or [])]
    except Exception:
        pass
    indian = ("bollywood", "punjabi", "south", "hindi", "urdu")

    def score(t):
        sc = 0.0
        if p in (t.get("pillars") or []):
            sc += 30
        g = (t.get("genre") or "").lower()
        if any(w in g for w in LIB_PILLAR_GENRE.get(p, ())):
            sc += 10
        is_in = any(w in g for w in indian)
        if market == "IN" and is_in:
            sc += 12
        if market and market != "IN" and is_in:
            sc -= 12
        name = f"{t.get('title','')} {t.get('artist','')}".lower()
        if any(k and (k in name or name.split(' (')[0] in k) for k in picks):
            sc += 8
        return sc

    pool = [t for t in tracks if t["file"] not in excl] or tracks
    ranked = sorted(((score(t), t) for t in pool), key=lambda x: -x[0])
    best = ranked[0][0]
    top = [t for sc, t in ranked if sc >= best - 8]
    h = int(hashlib.sha1((f"{salt}|{seed or p}").encode()).hexdigest(), 16)
    t = dict(top[h % len(top)])
    t["path"] = str(ldir / t["file"])
    t["library"] = True
    t["source"] = "library"
    t["pillar"] = p
    t.setdefault("credit", f"Song: {t.get('artist')} - {t.get('title')}")
    t["rank"] = score(t)
    return t


