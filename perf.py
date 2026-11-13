"""Performance-weighted picks for topics, songs and hook styles.

Reads stats/latest.json (views/likes) joined with results/*.json (music, pillar,
hook) so future Shorts lean toward what worked, with exploration so the channel
does not get stuck on one formula.
"""
from __future__ import annotations

import json
import math
import re
from pathlib import Path

REPO = Path(__file__).resolve().parent


def _title_key(title: str | None) -> str:
    text = " ".join((title or "").split()).casefold()
    text = re.sub(r"\s*\|\s*", " | ", text)
    text = re.sub(r"\s+#", " #", text)
    return text


def load_perf(repo: Path | None = None) -> dict:
    """Aggregate view-weighted signals from stats + local results."""
    root = Path(repo or REPO)
    out = {
        "songs": {},       # music file or credit -> weight
        "pillars": {},     # pillar -> weight
        "tags": {},        # tag -> weight
        "hook_openers": {},  # first 2-3 words -> weight
        "total_views": 0,
        "n": 0,
    }
    stats = {}
    try:
        stats = json.loads((root / "stats" / "latest.json").read_text())
    except Exception:
        pass
    by_title = {}
    for p in (root / "results").glob("*.json"):
        try:
            r = json.loads(p.read_text())
        except Exception:
            continue
        if r.get("status") != "done":
            continue
        key = _title_key(r.get("title"))
        if key:
            by_title[key] = r
    for _vid, row in (stats.get("videos") or {}).items():
        views = float(row.get("views") or 0)
        likes = float(row.get("likes") or 0)
        # Soften with log so one viral hit does not dominate forever.
        w = math.log1p(views) + 0.35 * math.log1p(likes)
        if w <= 0:
            continue
        out["total_views"] += views
        out["n"] += 1
        result = by_title.get(_title_key(row.get("title"))) or {}
        music = result.get("music") or row.get("music") or ""
        if music:
            # strip path / keep filename
            fname = str(music).rstrip("/").split("/")[-1]
            out["songs"][fname] = out["songs"].get(fname, 0.0) + w
            credit = (result.get("music_credit") or "").lower()
            if credit:
                out["songs"][credit[:80]] = out["songs"].get(credit[:80], 0.0) + 0.5 * w
        pillar = (result.get("pillar") or "").lower()
        # pillar may live on the queue item only; try tags
        tags = [str(t).lower() for t in (result.get("tags") or [])]
        for t in tags:
            if t in ("edit", "shorts", "trending", "test"):
                continue
            out["tags"][t] = out["tags"].get(t, 0.0) + w
            if t in ("anime", "movie", "celeb", "romance", "funny"):
                out["pillars"][t] = out["pillars"].get(t, 0.0) + w
        if pillar:
            out["pillars"][pillar] = out["pillars"].get(pillar, 0.0) + w
        hook = result.get("hook") or (row.get("title") or "").split("|")[0]
        opener = " ".join((hook or "").split()[:3]).casefold()
        if opener:
            out["hook_openers"][opener] = out["hook_openers"].get(opener, 0.0) + w
    return out


def _norm_bonus(weight: float, ceiling: float = 18.0) -> float:
    if weight <= 0:
        return 0.0
    return min(ceiling, 4.0 * math.log1p(weight))


def song_bonus(perf: dict, track: dict, exploration: float = 0.25) -> float:
    """Extra score for a library track. exploration keeps non-winners in play."""
    songs = perf.get("songs") or {}
    if not songs:
        return 0.0
    fname = track.get("file") or ""
    title = f"{track.get('artist', '')} - {track.get('title', '')}".lower()
    w = songs.get(fname, 0.0)
    for k, v in songs.items():
        if k and (k in title or title[:24] in k):
            w = max(w, v)
    # Exploration: shrink the gap so mid performers still get picked.
    return _norm_bonus(w) * (1.0 - exploration) + exploration * 2.0


def topic_bonus(perf: dict, tags: list[str] | None = None, pillar: str | None = None,
                exploration: float = 0.30) -> float:
    tags = [str(t).lower() for t in (tags or [])]
    pillars = perf.get("pillars") or {}
    tag_w = perf.get("tags") or {}
    w = 0.0
    if pillar:
        w += pillars.get(pillar.lower(), 0.0)
    for t in tags:
        w += 0.6 * tag_w.get(t, 0.0)
    return _norm_bonus(w, ceiling=14.0) * (1.0 - exploration) + exploration * 1.5


def hook_opener_penalty(perf: dict, hook: str, exploration: float = 0.35) -> float:
    """Small penalty when an opener is over-used by top videos (encourage variety),
    but a bonus when a rare opener matched high views historically.
    """
    openers = perf.get("hook_openers") or {}
    if not openers:
        return 0.0
    opener = " ".join((hook or "").split()[:3]).casefold()
    if not opener:
        return 0.0
    w = openers.get(opener, 0.0)
    # Cap so we do not spam the same opener; exploration keeps trying new ones.
    if w <= 0:
        return exploration * 1.0
    return _norm_bonus(w, ceiling=8.0) * (0.5 - exploration)
