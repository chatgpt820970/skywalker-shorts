"""Official Skywalker posting slots (IST). Existing Buffer posts keep their own slot."""
from __future__ import annotations

import json
from datetime import datetime, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parent
CFG = json.loads((ROOT / "config" / "slots.json").read_text())
TIMES = list(CFG["times"])
PER_DAY = int(CFG.get("per_day") or len(TIMES))
MIN_AHEAD_DAYS = int(CFG.get("min_ahead_days") or 2)


def official_times() -> list[str]:
    return list(TIMES)


def is_official(hhmm: str) -> bool:
    return (hhmm or "")[:5] in TIMES


def slot_iso(day: str, hhmm: str) -> str:
    return f"{day}T{hhmm}:00+05:30"


def parse_slot(s: str) -> datetime | None:
    try:
        return datetime.fromisoformat(s)
    except Exception:
        return None


def next_official_to_add(existing_isos: list[str], day: str, cap: int | None = None,
                        not_before=None) -> list[str]:
    """Keep every existing slot. Fill official times up to `cap` (default 8),
    always picking the official time farthest from already-taken times.
    Existing 12:30/17:30 items are never dropped or retimed.
    """
    cap = PER_DAY if cap is None else cap
    taken = []
    for s in existing_isos:
        dt = parse_slot(s)
        if dt and dt.strftime("%Y-%m-%d") == day:
            taken.append(dt)
    # 8/day is unique posting times; two items can share one slot (replacement).
    def unique_times(dts):
        return {(d.replace(second=0, microsecond=0) if d else None) for d in dts if d}
    out = []
    official = [parse_slot(slot_iso(day, t)) for t in TIMES]
    official = [d for d in official if d is not None]
    def taken_set():
        return taken + [parse_slot(x) for x in out if parse_slot(x)]
    while len(unique_times(taken_set())) < cap:
        occupied = [d for d in taken_set() if d]
        cand = []
        for d in official:
            iso = d.isoformat()
            if not_before is not None and d < not_before:
                continue
            # already have this exact official minute
            if any(o and abs((o - d).total_seconds()) < 60 for o in occupied):
                continue
            if not occupied:
                dist = 10**9
            else:
                dist = min(abs((o - d).total_seconds()) for o in occupied)
            cand.append((dist, d))
        if not cand:
            break
        cand.sort(key=lambda x: (-x[0], x[1]))
        out.append(cand[0][1].isoformat())
    return out
