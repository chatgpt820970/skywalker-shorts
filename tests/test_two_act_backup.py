#!/usr/bin/env python3
"""Two-act src window, 18s duration gate, and next-topic backup fallback."""
from __future__ import annotations

import json
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import auto_qc
import gate
import pipeline
import plan


IST = timezone(timedelta(hours=5, minutes=30))


def test_duration_gates_allow_18_to_32_aim_20():
    assert pipeline.MIN_OUTPUT_DUR == 18.0
    assert gate.MIN_DUR == 18.0
    assert gate.MAX_DUR == 32.6
    assert auto_qc.MIN_DURATION == 18.0
    assert auto_qc.AIM_DURATION == 20.0
    assert auto_qc.AIM_DURATION_MAX == 28.0
    assert pipeline.MIN_UNIQUE_DUR == 20.0
    # A 26s queue target must not be pulled back up to the old 25.5s body floor.
    assert pipeline.MIN_UNIQUE_DUR + 0.5 < 26
    assert auto_qc.duration_issue(18.0) is None
    assert auto_qc.duration_issue(22.23) is None
    assert auto_qc.duration_issue(17.9)
    assert "20-28" in auto_qc.duration_issue(17.9)
    pipeline.RELAX = False
    pipeline._require_min_duration(18.0)
    pipeline._require_min_duration(20.0)
    pipeline._require_min_duration(22.23)


def test_two_act_src_end_covers_the_visible_window():
    """A line that starts in a short scene must still play its full padded window."""
    scenes = [
        {"id": 2, "start": 4.0, "end": 4.4, "dur": 0.4, "skip": False, "spectacle": 10,
         "gemini": 8, "motion": 5, "speech": 0.8, "skin": 0.05, "phash": 1, "hist": [0]*8},
        {"id": 3, "start": 4.4, "end": 7.0, "dur": 2.6, "skip": False, "spectacle": 12,
         "gemini": 8, "motion": 4, "speech": 0.7, "skin": 0.05, "phash": 2, "hist": [1]*8},
        {"id": 10, "start": 30.0, "end": 34.0, "dur": 4.0, "skip": False, "spectacle": 80,
         "gemini": 8, "motion": 20, "speech": 0.0, "skin": 0.02, "phash": 3, "hist": [2]*8},
        {"id": 11, "start": 40.0, "end": 44.0, "dur": 4.0, "skip": False, "spectacle": 70,
         "gemini": 7, "motion": 18, "speech": 0.0, "skin": 0.02, "phash": 4, "hist": [3]*8},
        {"id": 12, "start": 50.0, "end": 54.0, "dur": 4.0, "skip": False, "spectacle": 60,
         "gemini": 7, "motion": 16, "speech": 0.0, "skin": 0.02, "phash": 5, "hist": [4]*8},
        {"id": 13, "start": 60.0, "end": 64.0, "dur": 4.0, "skip": False, "spectacle": 55,
         "gemini": 6, "motion": 14, "speech": 0.0, "skin": 0.02, "phash": 6, "hist": [5]*8},
        {"id": 14, "start": 70.0, "end": 74.0, "dur": 4.0, "skip": False, "spectacle": 50,
         "gemini": 6, "motion": 12, "speech": 0.0, "skin": 0.02, "phash": 7, "hist": [6]*8},
        {"id": 15, "start": 80.0, "end": 84.0, "dur": 4.0, "skip": False, "spectacle": 45,
         "gemini": 6, "motion": 10, "speech": 0.0, "skin": 0.02, "phash": 8, "hist": [7]*8},
    ]
    # One spoken line 4.2-6.2 that spans scenes 2 and 3 (the 961605e case).
    transcript = {"words": [
        {"word": "Are", "start": 4.2, "end": 4.4, "prob": 0.9},
        {"word": "you", "start": 4.4, "end": 4.6, "prob": 0.9},
        {"word": "being", "start": 4.6, "end": 4.9, "prob": 0.9},
        {"word": "serious", "start": 4.9, "end": 5.5, "prob": 0.9},
        {"word": "right", "start": 5.5, "end": 5.8, "prob": 0.9},
        {"word": "now?", "start": 5.8, "end": 6.2, "prob": 0.9},
    ], "segments": []}
    music = {"abs_beats": [20.0 + i * 0.5 for i in range(40)], "drop_abs": 20.0, "tempo": 120,
             "pulses": [i * 0.5 for i in range(40)], "start": 0.0}
    pipeline.WORK = Path("/tmp")
    pipeline.RELAX = False
    edl = pipeline.plan_two_act(scenes, music, target=24.0, transcript=transcript, usable=scenes)
    assert edl, "two-act should plan"
    act1 = [it for it in edl if it.get("act") == 1]
    assert act1
    it = act1[0]
    # The window must be playable in full, not clipped to scene 2's 4.4s end.
    assert it["src_end"] + 1e-6 >= it["src_start"] + it["visible"] - 0.05, it
    assert it["src_end"] > 4.4, it  # past the first scene cut


def test_sync_edl_times_maps_output_to_real_source(tmp_path, monkeypatch):
    # Fake durations without ffprobe: patch ffprobe_dur
    durs = {"/c0.mp4": 0.33, "/c1.mp4": 2.00}
    monkeypatch.setattr(pipeline, "ffprobe_dur", lambda p: durs[str(p)])
    edl = [
        {"scene": 1, "src_start": 4.06, "src_end": 6.10, "visible": 2.04, "t0": 0.0, "t1": 2.04, "speed": 1.0},
        {"scene": 2, "src_start": 36.67, "src_end": 39.29, "visible": 2.62, "t0": 2.04, "t1": 4.66, "speed": 1.0},
    ]
    paths = [Path("/c0.mp4"), Path("/c1.mp4")]
    # paths don't exist — sync_edl_times breaks on missing files. Point at existing files.
    a = tmp_path / "c0.mp4"; a.write_bytes(b"x")
    b = tmp_path / "c1.mp4"; b.write_bytes(b"x")
    durs[str(a)] = 0.33
    durs[str(b)] = 2.00
    pipeline.sync_edl_times(edl, [a, b])
    assert edl[0]["t0"] == 0.0 and abs(edl[0]["t1"] - 0.33) < 1e-6
    assert abs(edl[1]["t0"] - 0.33) < 1e-6
    assert abs(edl[1]["t1"] - 2.33) < 1e-6
    # Crop-check mapping: output 0.5s is in clip 1, source = 36.67 + (0.5-0.33)
    t_out = 0.50
    it = next(x for x in edl if x["t0"] - 1e-6 <= t_out < x["t1"])
    t_src = it["src_start"] + (t_out - it["t0"]) * float(it.get("speed") or 1)
    assert abs(t_src - (36.67 + 0.17)) < 1e-6


def test_failed_backup_picks_next_topic(tmp_path, monkeypatch):
    now = datetime(2026, 11, 13, 10, 0, tzinfo=IST)
    slot = "2026-11-13T12:30:00+05:30"
    items = {"war2": {"id": "war2", "slot": slot, "status": "queued"}}
    results = {"war2": {"status": "failed"}}
    # First backup already failed for this slot
    code = "20261113-1230"
    (tmp_path / f"{code}-zootopia2.json").write_text(json.dumps({
        "status": "failed", "backup_item": "backup-zootopia2",
    }))
    pool = [
        {"id": "backup-zootopia2"},
        {"id": "backup-kalki2898ad"},
        {"id": "backup-superman"},
    ]
    monkeypatch.chdir(tmp_path)
    (tmp_path / "results").mkdir()
    (tmp_path / "results" / f"{code}-zootopia2.json").write_text(
        json.dumps({"status": "failed", "backup_item": "backup-zootopia2"})
    )
    picks, used = plan.select_slot_backups(
        items, lambda i: results.get(i), pool, set(), now, ahead=6, busy=set()
    )
    assert picks == ["backup-kalki2898ad@20261113-1230"], picks
    assert "backup-zootopia2" in used


def test_done_backup_does_not_pick_another(tmp_path, monkeypatch):
    now = datetime(2026, 11, 13, 10, 0, tzinfo=IST)
    slot = "2026-11-13T12:30:00+05:30"
    items = {"war2": {"id": "war2", "slot": slot, "status": "queued"}}
    results = {"war2": {"status": "failed"}}
    monkeypatch.chdir(tmp_path)
    (tmp_path / "results").mkdir()
    (tmp_path / "results" / "20261113-1230-zootopia2.json").write_text(
        json.dumps({"status": "done", "backup_item": "backup-zootopia2"})
    )
    pool = [{"id": "backup-zootopia2"}, {"id": "backup-kalki2898ad"}]
    picks, used = plan.select_slot_backups(
        items, lambda i: results.get(i), pool, set(), now, ahead=6, busy=set()
    )
    assert picks == []
