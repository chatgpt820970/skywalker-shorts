#!/usr/bin/env python3
"""Light checks for perf weighting and bigger lower-third captions."""
from __future__ import annotations

import json
from pathlib import Path

import captions
import perf


def test_caption_style_is_big_and_safe():
    assert captions.FONT_SIZE >= 70
    assert captions.OUTLINE >= 6
    assert captions.DEFAULT_Y >= 1500
    # Stay above Shorts bottom UI (~240px pad).
    assert captions.caption_y(None) <= 1920 - captions.SAFE_BOTTOM_PAD - 20
    assert captions.caption_y(1600) <= 1920 - captions.SAFE_BOTTOM_PAD


def test_write_ass_uses_bold_outline(tmp_path: Path):
    p = tmp_path / "lt.ass"
    captions.write_ass(p, [{"t0": 0.0, "t1": 1.5, "text": "Hello world test"}], plate_bottom=1100)
    text = p.read_text()
    assert f",{captions.FONT_SIZE}," in text or f",{captions.FONT_SIZE}," in text.replace(" ", "")
    assert str(captions.OUTLINE) in text
    assert r"\b1" in text or "Bold" in text or ",-1," in text


def test_song_and_topic_bonus_with_empty_stats(tmp_path: Path):
    (tmp_path / "stats").mkdir()
    (tmp_path / "stats" / "latest.json").write_text("{}")
    (tmp_path / "results").mkdir()
    data = perf.load_perf(tmp_path)
    assert data["n"] == 0
    assert perf.song_bonus(data, {"file": "x.mp3", "title": "t", "artist": "a"}) == 0.0
    assert perf.topic_bonus(data, ["movie"], "movie") >= 0.0


def test_song_bonus_prefers_winners(tmp_path: Path):
    (tmp_path / "stats").mkdir()
    (tmp_path / "results").mkdir()
    (tmp_path / "results" / "a.json").write_text(json.dumps({
        "status": "done",
        "title": "Winner Hook | Film | #edit",
        "hook": "Winner Hook",
        "music": "hit_track.mp3",
        "tags": ["movie", "edit"],
        "pillar": "movie",
    }))
    (tmp_path / "stats" / "latest.json").write_text(json.dumps({
        "videos": {
            "vid1": {"title": "Winner Hook | Film | #edit", "views": 9000, "likes": 100},
            "vid2": {"title": "Other Title | Film | #edit", "views": 10, "likes": 0},
        }
    }))
    data = perf.load_perf(tmp_path)
    hit = perf.song_bonus(data, {"file": "hit_track.mp3", "title": "Hit", "artist": "X"})
    miss = perf.song_bonus(data, {"file": "unknown.mp3", "title": "Nope", "artist": "Y"})
    assert hit > miss
    assert perf.topic_bonus(data, ["movie"], "movie") > 0
