"""Official 8-slot IST day and queue fill (no existing post is retimed)."""
from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import slot_times


def test_official_grid_is_eight_ist_times():
    times = slot_times.official_times()
    assert times == ["06:30", "09:00", "11:30", "13:30", "16:00", "18:30", "20:30", "22:30"]
    assert slot_times.PER_DAY == 8
    assert slot_times.is_official("06:30")
    assert not slot_times.is_official("12:30")
    assert not slot_times.is_official("17:30")


def test_fill_keeps_legacy_slots_and_caps_unique_times_at_eight():
    existing = [
        "2026-11-14T06:30:00+05:30",
        "2026-11-14T12:30:00+05:30",
        "2026-11-14T17:30:00+05:30",
        "2026-11-14T22:30:00+05:30",
        "2026-11-14T06:30:00+05:30",  # replacement item, same minute
    ]
    add = slot_times.next_official_to_add(existing, "2026-11-14")
    hhmm = [a[11:16] for a in add]
    assert "12:30" not in hhmm and "17:30" not in hhmm
    taken = {s[11:16] for s in existing} | set(hhmm)
    assert len(taken) == 8
    # legacy times stay represented
    assert "12:30" in taken and "17:30" in taken


def test_empty_day_gets_all_eight_official_times():
    add = slot_times.next_official_to_add([], "2026-11-21")
    assert sorted(a[11:16] for a in add) == slot_times.official_times()
