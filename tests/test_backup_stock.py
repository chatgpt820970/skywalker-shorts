#!/usr/bin/env python3
"""Pre-rendered backup stock: refill, assign, and same-slot posting guards."""
from __future__ import annotations

import json
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import backup_stock
import plan

IST = timezone(timedelta(hours=5, minutes=30))


def _write(p: Path, obj):
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(obj, indent=2) + "\n")


def _ready_result(iid: str, **extra):
    r = {
        "id": iid,
        "status": "done",
        "asset_url": f"https://github.com/chatgpt820970/skywalker-shorts/releases/download/backup-stock/{iid}.mp4",
        "title": f"Hook for {iid}",
        "hook": "A cold open that slaps",
        "description": "From the official trailer. Edit by Skywalker.",
        "no_publish": False,
    }
    r.update(extra)
    return r


def _root(tmp_path: Path, monkeypatch, pool=None, manifest=None):
    monkeypatch.setenv("SKY_ROOT", str(tmp_path))
    monkeypatch.setattr(backup_stock, "REPO", tmp_path)
    (tmp_path / "config").mkdir()
    (tmp_path / "results").mkdir()
    (tmp_path / "queue").mkdir()
    _write(tmp_path / "config" / "backup_pool.json", {
        "items": pool or [
            {"id": "backup-zootopia2", "fallback_asset": "zootopia2_trailer_h264.mp4"},
            {"id": "backup-kalki2898ad", "fallback_asset": "kalki2898ad_trailer_h264.mp4"},
            {"id": "backup-superman", "fallback_asset": "superman_trailer_h264.mp4"},
        ]
    })
    _write(tmp_path / "config" / "backup_stock.json", manifest or {
        "target": 30, "refill_at": 5, "max_refill_per_run": 2, "release": "backup-stock", "items": []
    })
    return tmp_path


def test_refill_when_empty_picks_two_diverse_ids(tmp_path, monkeypatch):
    root = _root(tmp_path, monkeypatch)
    ids = backup_stock.next_refill_ids(root, limit=2)
    assert len(ids) == 2
    assert all(backup_stock.parse_stock_id(i) for i in ids)
    slugs = {backup_stock.parse_stock_id(i)[0] for i in ids}
    assert len(slugs) == 2  # different topics in one batch


def test_no_refill_when_above_floor(tmp_path, monkeypatch):
    root = _root(tmp_path, monkeypatch)
    items = []
    for i in range(6):
        iid = f"stock-zootopia2-{i+1:02d}"
        _write(root / "results" / f"{iid}.json", _ready_result(iid, stock_topic="backup-zootopia2"))
        items.append({"id": iid, "slug": "zootopia2", "topic_id": "backup-zootopia2",
                      "status": "ready", "asset_url": _ready_result(iid)["asset_url"]})
    _write(root / "config" / "backup_stock.json", {
        "target": 30, "refill_at": 5, "max_refill_per_run": 2, "items": items
    })
    assert backup_stock.next_refill_ids(root, limit=2) == []


def test_refill_at_five_fills_toward_target(tmp_path, monkeypatch):
    root = _root(tmp_path, monkeypatch)
    items = []
    for i in range(5):
        iid = f"stock-zootopia2-{i+1:02d}"
        _write(root / "results" / f"{iid}.json", _ready_result(iid, stock_topic="backup-zootopia2"))
        items.append({"id": iid, "slug": "zootopia2", "topic_id": "backup-zootopia2",
                      "status": "ready", "asset_url": _ready_result(iid)["asset_url"]})
    _write(root / "config" / "backup_stock.json", {
        "target": 30, "refill_at": 5, "max_refill_per_run": 2, "items": items
    })
    ids = backup_stock.next_refill_ids(root, limit=2)
    assert len(ids) == 2
    # already has stock-zootopia2-01..05, next zootopia2 would be 06; other topics start at 01
    assert "stock-zootopia2-01" not in ids


def test_assign_marks_used_and_never_reuses(tmp_path, monkeypatch):
    root = _root(tmp_path, monkeypatch)
    _write(root / "results" / "stock-kalki2898ad-01.json",
           _ready_result("stock-kalki2898ad-01", stock_topic="backup-kalki2898ad"))
    slot = "2026-11-14T16:00:00+05:30"
    now = datetime(2026, 11, 14, 15, 10, tzinfo=IST)
    got = backup_stock.assign_to_slot(root, slot, now)
    assert got is not None
    assert got["id"] == "20261114-1600-kalki2898ad"
    assert got["slot"] == slot
    assert got["assigned_from_stock"] is True
    assert got["stock_id"] == "stock-kalki2898ad-01"
    assert got.get("buffer_post_id") is None
    src = json.loads((root / "results" / "stock-kalki2898ad-01.json").read_text())
    assert src["stock_used"] is True
    assert src["assigned_slot"] == slot
    # second assign must not reuse it
    got2 = backup_stock.assign_to_slot(root, "2026-11-14T18:30:00+05:30", now)
    assert got2 is None


def test_assign_avoids_same_topic_same_day(tmp_path, monkeypatch):
    root = _root(tmp_path, monkeypatch)
    _write(root / "results" / "stock-zootopia2-01.json",
           _ready_result("stock-zootopia2-01", stock_topic="backup-zootopia2"))
    _write(root / "results" / "stock-superman-01.json",
           _ready_result("stock-superman-01", stock_topic="backup-superman"))
    # a live zootopia2 already occupies today's queue
    _write(root / "queue" / "2026-11-14.json", [
        {"id": "20261114-1230-zootopia2", "slot": "2026-11-14T12:30:00+05:30", "status": "queued"}
    ])
    now = datetime(2026, 11, 14, 15, 10, tzinfo=IST)
    got = backup_stock.assign_to_slot(root, "2026-11-14T16:00:00+05:30", now)
    assert got["stock_id"] == "stock-superman-01"
    assert "zootopia2" not in got["id"]


def test_slot_needs_stock_on_failed_or_imminent(tmp_path, monkeypatch):
    root = _root(tmp_path, monkeypatch)
    slot = "2026-11-14T16:00:00+05:30"
    _write(root / "queue" / "2026-11-14.json", [
        {"id": "20261114-1600-fantasticfour", "slot": slot, "status": "queued"}
    ])
    now = datetime(2026, 11, 14, 10, 0, tzinfo=IST)  # 6h before, not imminent
    assert backup_stock.slot_needs_stock(root, slot, now) is False
    _write(root / "results" / "20261114-1600-fantasticfour.json",
           {"id": "20261114-1600-fantasticfour", "status": "failed", "slot": slot})
    assert backup_stock.slot_needs_stock(root, slot, now) is True
    # a ready primary covers it
    _write(root / "results" / "20261114-1600-fantasticfour.json",
           _ready_result("20261114-1600-fantasticfour", slot=slot))
    assert backup_stock.slot_needs_stock(root, slot, now) is False
    # T-60 with nothing ready
    _write(root / "results" / "20261114-1600-fantasticfour.json",
           {"id": "20261114-1600-fantasticfour", "status": "retry_queued", "slot": slot})
    late = datetime(2026, 11, 14, 15, 10, tzinfo=IST)
    assert backup_stock.slot_needs_stock(root, slot, late) is True
    # no_publish must not count as coverage
    _write(root / "results" / "20261114-1600-fantasticfour.json",
           _ready_result("20261114-1600-fantasticfour", slot=slot, no_publish=True,
                         asset_url="https://github.com/x/releases/download/test-renders/x.mp4"))
    assert backup_stock.slot_needs_stock(root, slot, late) is True


def test_no_publish_does_not_cover_slot(tmp_path, monkeypatch):
    root = _root(tmp_path, monkeypatch)
    slot = "2026-11-14T16:00:00+05:30"
    _write(root / "results" / "20261114-1600-fantasticfour.json",
           _ready_result("20261114-1600-fantasticfour", slot=slot, no_publish=True,
                         asset_url="https://github.com/x/releases/download/test-renders/x.mp4"))
    assert backup_stock.slot_is_covered(root, slot) is False


def test_slot_post_conflicts_prefers_assigned_stock():
    stock = {"id": "20261114-1600-superman", "slot": "s", "assigned_from_stock": True}
    primary = {"id": "20261114-1600-fantasticfour", "slot": "s"}
    posted = {"id": "other", "slot": "s", "buffer_post_id": "bp1"}
    assert backup_stock.slot_post_conflicts(primary, [stock, primary]) is True
    assert backup_stock.slot_post_conflicts(stock, [stock, primary]) is False
    assert backup_stock.slot_post_conflicts(primary, [posted, primary]) is True


def test_plan_skips_backup_render_when_stock_covers(tmp_path, monkeypatch):
    now = datetime(2026, 11, 14, 15, 10, tzinfo=IST)
    slot = "2026-11-14T16:00:00+05:30"
    items = {"ff": {"id": "ff", "slot": slot, "status": "queued"}}
    results = {"ff": {"status": "failed"}}
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("SKY_ROOT", str(tmp_path))
    monkeypatch.setattr(backup_stock, "REPO", tmp_path)
    (tmp_path / "results").mkdir()
    (tmp_path / "config").mkdir()
    _write(tmp_path / "results" / "20261114-1600-superman.json",
           _ready_result("20261114-1600-superman", slot=slot, assigned_from_stock=True,
                         stock_id="stock-superman-01"))
    pool = [{"id": "backup-zootopia2"}]
    picks, _ = plan.select_slot_backups(
        items, lambda i: results.get(i), pool, set(), now, ahead=6, busy=set()
    )
    assert picks == []
