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
    for name in gemini.KEY_ENVS:
        monkeypatch.delenv(name, raising=False)
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


def test_gemini_key_rotation(monkeypatch, tmp_path):
    for name in gemini.KEY_ENVS:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("SKY_GEMINI", "1")
    monkeypatch.setenv("SKY_GEMINI_BUDGET_FILE", str(tmp_path / "b.json"))
    assert gemini.KEY_ENVS == (
        "GEMINI_API_KEY", "GEMINI_API_KEY_2", "GEMINI_API_KEY_3",
        "GEMINI_API_KEY_4", "GEMINI_API_KEY_5", "GEMINI_API_KEY_6",
    )
    assert gemini._keys() == []
    assert not gemini.available()
    # Empty / whitespace keys are skipped; slots stay in order.
    monkeypatch.setenv("GEMINI_API_KEY", "  ")
    monkeypatch.setenv("GEMINI_API_KEY_3", "secret-three")
    monkeypatch.setenv("GEMINI_API_KEY_2", "secret-two")
    monkeypatch.setenv("GEMINI_API_KEY_6", "secret-six")
    names = [n for n, _ in gemini._keys()]
    assert names == ["GEMINI_API_KEY_2", "GEMINI_API_KEY_3", "GEMINI_API_KEY_6"]
    assert gemini.available()
    # Last-good key is tried first; the budget stores the env name, never the secret.
    (tmp_path / "b.json").write_text(__import__("json").dumps({"calls": 0, "http": 0, "good_key": "GEMINI_API_KEY_6"}))
    ordered = [n for n, _ in gemini._ordered_keys()]
    assert ordered[0] == "GEMINI_API_KEY_6"
    assert ordered[1:] == ["GEMINI_API_KEY_2", "GEMINI_API_KEY_3"]
    blob = (tmp_path / "b.json").read_text()
    assert "secret-" not in blob
    assert gemini._key_slot("GEMINI_API_KEY") == "key1"
    assert gemini._key_slot("GEMINI_API_KEY_6") == "key6"
    assert gemini._quota_error(429) and gemini._quota_error(503)
    assert gemini._quota_error(403, b'{"error":{"status":"RESOURCE_EXHAUSTED"}}')
    assert not gemini._quota_error(403, b'{"error":"PERMISSION_DENIED"}')
    assert not gemini._quota_error(400)


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


def test_own_performance_joins_render_facts():
    import json
    import trends
    from pathlib import Path
    n_stats = len(json.loads(Path("stats/latest.json").read_text()).get("videos") or {})
    rows = trends.own_performance()
    assert len(rows) == n_stats and n_stats >= 12
    # Music/energy/drop/slot are joined from results; views come from live stats.
    hay = next(r for r in rows if "Haymitch" in (r.get("title") or ""))
    assert hay["views"] >= 1000
    assert "Godless" in hay["music"] and "ncs_godless.mp3" in hay["music"]
    assert hay["energy"] is True
    assert abs(float(hay["drop_t"]) - 12.562) < 0.01
    assert hay["slot"] == "06:30"
    doom = next(r for r in rows if "Doctor Doom" in (r.get("title") or ""))
    assert doom["views"] >= 100
    assert "Adrenaline" in doom["music"]
    assert doom["energy"] is True
    assert doom["slot"] == "06:30"
    maat = next(r for r in rows if "Maatrubhumi" in (r.get("title") or ""))
    assert maat["energy"] is False
    assert maat["drop_t"] is None
    assert maat["slot"] == "22:30"
    lucy = next(r for r in rows if "Lucy" in (r.get("title") or ""))
    assert lucy["energy"] is False and lucy["slot"] == "17:30"


def test_queue_hook_survives_when_gemini_is_down():
    item = {
        "id": "20261113-2230-demonslayer",
        "hook": "Tanjiro entered the Infinity Castle",
        "title": "Tanjiro entered the Infinity Castle 💀 | Into the Infinity Castle | #edit",
        "description": "Tanjiro entered the Infinity Castle. Tanjiro walks into the Infinity Castle from the official Demon Slayer: Infinity Castle trailer.",
        "tags": ["demonslayer", "tanjiro", "anime"],
    }
    meta = sky_ai.queue_hook_meta(item, ["Watch the last two seconds."], [])
    assert meta["source"] == "queue"
    assert meta["hook"] == item["hook"]
    assert meta["title"] == item["title"]
    assert meta["description"].startswith("Tanjiro entered the Infinity Castle.")
    banned = {
        "id": "x",
        "hook": "The reason why Tanjiro refuses to break",
        "description": "From the official Demon Slayer trailer.",
        "tags": ["demonslayer"],
    }
    generic = sky_ai.queue_hook_meta(banned, [], [])
    assert generic["source"] == "fallback"
    assert not sky_ai.hook_errors(generic["hook"], [])


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
    good = {"mode": "two_act", "isolated": True, "act1_frac": 0.30, "song_in_act1_dbfs": -99, "voice_in_act2_dbfs": -99,
            "song_switches": 1, "act1_voice_dbfs": -17, "act2_song_dbfs": -12}
    assert not vocal_mix.check_errors(good, True)
    assert vocal_mix.check_errors(dict(good, song_switches=5), True)
    assert vocal_mix.check_errors(dict(good, song_in_act1_dbfs=-20), True)
    assert vocal_mix.check_errors(dict(good, act1_frac=0.8), True)
    assert not vocal_mix.check_errors(dict(good, isolated=False), True)  # Demucs fallback never fails a slot


def test_edit_mode_choice():
    import edit_mode
    import vocal_mix

    # Stable 3-of-4 music_only / 1-of-4 two_act split from the item id.
    modes = [edit_mode.default_edit_mode(f"20261114-0630-item{i:03d}") for i in range(200)]
    n_two = sum(1 for m in modes if m == "two_act")
    n_mus = sum(1 for m in modes if m == "music_only")
    assert n_mus + n_two == 200
    # sha1 buckets are stable and ~1/4 two_act (allow ordinary hash jitter).
    assert 35 <= n_two <= 65 and n_mus == 200 - n_two

    mus = next(f"id{i}" for i in range(80) if edit_mode.default_edit_mode(f"id{i}") == "music_only")
    two = next(f"id{i}" for i in range(80) if edit_mode.default_edit_mode(f"id{i}") == "two_act")
    assert edit_mode.choose_edit_mode(mus) == "music_only"
    assert edit_mode.choose_edit_mode(two) == "two_act"
    # Gemini may upgrade music_only to two_act only for an iconic dialogue hook.
    assert edit_mode.choose_edit_mode(mus, iconic_hook=True) == "two_act"
    # An explicit override always wins (demo / forced music_only).
    assert edit_mode.choose_edit_mode(mus, override="music_only", iconic_hook=True) == "music_only"
    assert edit_mode.choose_edit_mode(two, override="music_only") == "music_only"
    assert edit_mode.choose_edit_mode(mus, override="two_act") == "two_act"
    # Same id always maps to the same bucket.
    assert edit_mode.item_mode_bucket("demo-superman-v4") == edit_mode.item_mode_bucket("demo-superman-v4")

    good = {"mode": "music_only", "act2_song_dbfs": -12, "speech_segments": 0}
    assert not vocal_mix.check_errors(good, True)
    assert not vocal_mix.check_errors(good, False)
    assert vocal_mix.check_errors(dict(good, act2_song_dbfs=-40), False)
    # 0.46 is now over the 0.45 two-act cap (owner: ~30%, max 38/45).
    assert vocal_mix.check_errors(
        {"mode": "two_act", "isolated": True, "act1_frac": 0.46, "song_in_act1_dbfs": -99,
         "voice_in_act2_dbfs": -99, "song_switches": 1, "act1_voice_dbfs": -17, "act2_song_dbfs": -12},
        True,
    )


def test_gemini_song_choice_respects_exclude(tmp_path, monkeypatch):
    import music_lib
    lib = json.loads((ROOT / "music" / "library.json").read_text())
    (tmp_path / "music" / "lib").mkdir(parents=True)
    (tmp_path / "music" / "library.json").write_text(json.dumps(lib))
    for tr in lib["tracks"]:
        (tmp_path / "music" / "lib" / tr["file"]).write_bytes(b"x")
    monkeypatch.setenv("SKY_MARKET", "US")
    pick = music_lib.pick(tmp_path, "movie", "seed1", prefer="sw_metamorphosis.mp3")
    assert pick["file"] == "sw_metamorphosis.mp3"
    assert pick["picked_by"] == "gemini"
    blocked = music_lib.pick(tmp_path, "movie", "seed1", prefer="sw_metamorphosis.mp3",
                             exclude=["sw_metamorphosis.mp3"])
    assert blocked["file"] != "sw_metamorphosis.mp3"
    alt = music_lib.pick(tmp_path, "movie", "seed1", prefer="missing.mp3",
                         prefer_alts=["sw_murderinmymind.mp3"])
    assert alt["file"] == "sw_murderinmymind.mp3"
    assert alt["picked_by"] == "gemini_alt"
    hashed = music_lib.pick(tmp_path, "movie", "seed1")
    assert hashed["picked_by"] == "hash"
    assert hashed["file"]


def test_ingest_license_and_slug():
    sys.path.insert(0, str(ROOT / "scripts"))
    import ingest_music
    assert ingest_music.license_ok("https://creativecommons.org/licenses/by/4.0/")
    assert ingest_music.license_ok("https://creativecommons.org/publicdomain/zero/1.0/")
    assert not ingest_music.license_ok("https://creativecommons.org/licenses/by-nc/4.0/")
    assert not ingest_music.license_ok("https://creativecommons.org/licenses/by-nd/4.0/")
    assert not ingest_music.license_ok("")
    assert ingest_music.slugify("ELA PEIDA FUNK", "x") == "sw_elapeidafunk.mp3"
    g, m, p, tags = ingest_music.tags_from_text("dark phonk drift")
    assert g == "phonk" and "anime" in p
