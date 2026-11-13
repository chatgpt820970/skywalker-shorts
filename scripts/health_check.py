#!/usr/bin/env python3
"""Verify the next 24h of IST slots have a ready video (or a renderable backup).

A slot is healthy when:
  - a results/<id>.json for that slot is status=done with an asset_url, OR
  - a queued item for that slot still has a sources-release fallback (can render), OR
  - the backup pool still has an unused pre-uploaded topic for that slot.

If short, write config/health_status.json and optionally append extra ready
backup renders into the plan (BACKUPS_ONLY / EXTRA_READY). Never deletes videos.
"""
from __future__ import annotations

import glob
import json
import os
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
import slot_times  # noqa: E402

IST = timezone(timedelta(hours=5, minutes=30))
GH = "https://github.com/chatgpt820970/skywalker-shorts/releases/download/sources"


def load_pool() -> list[dict]:
    try:
        return json.loads((REPO / "config" / "backup_pool.json").read_text()).get("items") or []
    except Exception:
        return []


def result_for(iid: str) -> dict | None:
    p = REPO / "results" / f"{iid}.json"
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text())
    except Exception:
        return None


def queue_items() -> list[dict]:
    out = []
    for f in sorted(glob.glob(str(REPO / "queue" / "2026-*.json"))):
        try:
            out.extend(json.load(open(f)))
        except Exception:
            pass
    return out


def has_preuploaded(it: dict) -> bool:
    if it.get("fallback_asset") or it.get("fallback_mp4"):
        return True
    return False


def main() -> int:
    now = datetime.now(IST)
    horizon = now + timedelta(hours=24)
    items = queue_items()
    pool = load_pool()
    used_backups = set()
    for p in (REPO / "results").glob("*.json"):
        try:
            r = json.loads(p.read_text())
        except Exception:
            continue
        if r.get("backup_item"):
            used_backups.add(r["backup_item"])

    # Group by official slot minute in the next 24h
    slots: dict[str, list[dict]] = {}
    for it in items:
        if it.get("status", "queued") not in ("queued", "done"):
            continue
        try:
            slot = datetime.fromisoformat(it["slot"])
        except Exception:
            continue
        if not (now - timedelta(hours=2) <= slot <= horizon):
            continue
        slots.setdefault(it["slot"], []).append(it)

    # Also ensure every official time in the window appears
    day0 = now.date()
    for d in range(0, 2):
        day = (day0 + timedelta(days=d)).isoformat()
        for hhmm in slot_times.official_times():
            iso = slot_times.slot_iso(day, hhmm)
            try:
                dt = datetime.fromisoformat(iso)
            except Exception:
                continue
            if now - timedelta(hours=2) <= dt <= horizon:
                slots.setdefault(iso, [])

    short = []
    ready = []
    for slot_s, group in sorted(slots.items()):
        done = False
        for it in group:
            r = result_for(it["id"])
            if r and r.get("status") == "done" and r.get("asset_url"):
                done = True
                ready.append({"slot": slot_s, "id": it["id"], "how": "done"})
                break
        if done:
            continue
        # renderable primary?
        if any(has_preuploaded(it) and it.get("status", "queued") == "queued" for it in group):
            ready.append({"slot": slot_s, "id": group[0]["id"], "how": "queued+fallback"})
            continue
        # backup pool with preuploaded source
        cand = next((b for b in pool if b["id"] not in used_backups and has_preuploaded(b)), None)
        if cand:
            ready.append({"slot": slot_s, "id": cand["id"], "how": "backup_pool"})
            used_backups.add(cand["id"])
            continue
        short.append(slot_s)

    # Extra ready target: at least 2 spare done renders beyond the next 24h live slots
    extra_needed = int(os.environ.get("EXTRA_READY_PER_DAY", "2"))
    spare_done = []
    for p in (REPO / "results").glob("*.json"):
        try:
            r = json.loads(p.read_text())
        except Exception:
            continue
        if r.get("status") == "done" and r.get("asset_url") and r.get("spare"):
            spare_done.append(r["id"])

    preuploaded_pool = [b for b in pool if has_preuploaded(b)]
    status = {
        "at": now.strftime("%Y-%m-%d %H:%M IST"),
        "slots_checked": len(slots),
        "ready": ready,
        "short": short,
        "backup_pool_size": len(pool),
        "backup_pool_preuploaded": len(preuploaded_pool),
        "spare_done": spare_done,
        "extra_ready_target": extra_needed,
        "ok": not short and len(preuploaded_pool) >= 20,
    }
    out = REPO / "config" / "health_status.json"
    out.write_text(json.dumps(status, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps({k: status[k] for k in ("slots_checked", "short", "backup_pool_preuploaded", "ok")}, indent=2))
    # Non-zero only when a live slot in the next 24h cannot be covered
    return 1 if short else 0


if __name__ == "__main__":
    raise SystemExit(main())
