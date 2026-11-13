"""Light tests for the publish path (no ffmpeg, no network)."""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import gemini  # noqa: E402
import music_lib  # noqa: E402
import sky_ai  # noqa: E402


def test_banned_words_ignore_safe_text():
    assert sky_ai.banned_hits("The reason why Asta never gives up", "From the official trailer.") == []
    assert "full movie" in sky_ai.banned_hits("Watch the FULL MOVIE now")


def test_gemini_disabled_without_key(monkeypatch, tmp_path):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.setenv("SKY_GEMINI_BUDGET_FILE", str(tmp_path / "b.json"))
    assert gemini.generate_json("x", purpose="t") is None
    assert sky_ai.clip_hook_meta(Path("/nonexistent"), [], {}, tmp_path, "x") is None


def test_gemini_budget_cap(monkeypatch, tmp_path):
    monkeypatch.setenv("GEMINI_API_KEY", "dummy")
    b = tmp_path / "b.json"
    b.write_text(json.dumps({"calls": 5, "http": 0}))
    monkeypatch.setenv("SKY_GEMINI_BUDGET_FILE", str(b))
    assert gemini.budget_left() == 0
    assert gemini.generate_json("x", purpose="t") is None  # no HTTP once the budget is spent


def test_scene_ranks_keep_enough_footage():
    scenes = [{"start": i * 3.0, "end": i * 3.0 + 3.0, "score": 1.0} for i in range(14)]
    ranks = {f"{s['start']:.2f}-{s['end']:.2f}": {"score": 9, "card": True} for s in scenes}
    sky_ai.apply_scene_ranks(scenes, ranks, min_keep_s=34.0)
    kept = sum(s["end"] - s["start"] for s in scenes if not s.get("skip"))
    assert kept >= 34.0


def test_library_has_tracks_and_no_repeat(tmp_path, monkeypatch):
    lib = json.loads((ROOT / "music" / "library.json").read_text())
    assert len(lib["tracks"]) >= 15
    (tmp_path / "music" / "lib").mkdir(parents=True)
    (tmp_path / "music" / "library.json").write_text(json.dumps(lib))
    for t in lib["tracks"]:
        (tmp_path / "music" / "lib" / t["file"]).write_bytes(b"x")
        assert t["credit"] and t["pillars"]
    monkeypatch.setenv("SKY_MARKET", "IN")
    used = []
    for k in range(8):
        t = music_lib.pick(tmp_path, "celeb", f"seed{k}", exclude=used)
        assert t["file"] not in used
        used.append(t["file"])


def test_publish_config():
    cfg = json.loads((ROOT / "config" / "publish.json").read_text())
    assert cfg["buffer_channel_id"] == "6af580b362b70b94b00530b0"
    assert cfg["buffer_org_id"] == "6ac41da09a0b5aca7c671500"


def test_no_reason_why_template_anywhere():
    import hook_text
    assert hook_text.template_errors("The reason why Asta never gives up")
    assert hook_text.template_errors("reason why he left")
    assert not hook_text.template_errors("Asta refuses to stay down")
    for h in sky_ai.FALLBACK_HOOKS:
        assert not hook_text.template_errors(h)


def test_fallback_hook_varied_and_valid():
    item = {"id": "20261113-0630-cliffbooth", "description": "x from the official Cliff Booth trailer. Edit."}
    h1 = sky_ai.fallback_hook(item, ["He never walks away from a fight."], [])
    assert h1.startswith('"') and not sky_ai.hook_errors(h1, [])
    # clipped fragments and profanity are never quoted on the bar
    h0 = sky_ai.fallback_hook(item, ["got Teddy's shit", "Get that shit out of here", "Doing good I", "Not maybe I"], [])
    assert not h0.startswith('"') and not sky_ai.hook_errors(h0, [])
    h2 = sky_ai.fallback_hook(item, [], [])
    assert not sky_ai.hook_errors(h2, [])
    import hook_text
    h3 = sky_ai.fallback_hook(item, [], [hook_text.opener(h2)])
    assert hook_text.opener(h3) != hook_text.opener(h2)
    t = sky_ai.build_title(h2 + " | Cliff Booth", ["cliffbooth", "bradpitt", "movieedit", "x"])
    assert t.count("#") == 3 and len(t) <= 100


def test_audio_check_gate():
    import vocal_mix
    assert vocal_mix.check_errors(None, True)
    assert not vocal_mix.check_errors({"isolated": False, "speech_segments": 3, "smr_db": 20, "smr_min_db": 18}, True)
    assert vocal_mix.check_errors({"isolated": True, "speech_segments": 3, "smr_db": 8, "smr_min_db": 5}, True)
    assert not vocal_mix.check_errors({"isolated": True, "speech_segments": 3, "smr_db": 17, "smr_min_db": 15}, True)
    assert not vocal_mix.check_errors(None, False)


def test_two_act_audio_gate():
    import vocal_mix
    good = {"mode": "two_act", "isolated": True, "act1_frac": 0.46, "song_in_act1_dbfs": -99, "voice_in_act2_dbfs": -99,
            "song_switches": 1, "act1_voice_dbfs": -17, "act2_song_dbfs": -12}
    assert not vocal_mix.check_errors(good, True)
    assert vocal_mix.check_errors(dict(good, song_switches=5), True)
    assert vocal_mix.check_errors(dict(good, song_in_act1_dbfs=-20), True)
    assert vocal_mix.check_errors(dict(good, act1_frac=0.8), True)
    assert not vocal_mix.check_errors(dict(good, isolated=False), True)  # Demucs fallback never fails a slot
