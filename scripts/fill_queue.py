#!/usr/bin/env python3
"""Ensure each upcoming day has 8 official IST slots filled from topic_catalog.

Keeps existing queue items (including legacy 12:30/17:30). Never retimes or
deletes posts. Used by trends.yml and can be run manually.
"""
from __future__ import annotations

import json
import random
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
import perf  # noqa: E402
import slot_times  # noqa: E402

IST = timezone(timedelta(hours=5, minutes=30))
GH = "https://github.com/chatgpt820970/skywalker-shorts/releases/download/sources"
QUEUE = REPO / "queue"
CATALOG = REPO / "config" / "topic_catalog.json"


def _day_file(day: str) -> Path:
    return QUEUE / f"{day}.json"


def _load_day(day: str) -> list[dict]:
    p = _day_file(day)
    if not p.exists():
        return []
    try:
        return json.loads(p.read_text())
    except Exception:
        return []


def _used_slugs() -> set[str]:
    used = set()
    for p in QUEUE.glob("2026-*.json"):
        try:
            items = json.loads(p.read_text())
        except Exception:
            continue
        for it in items:
            if it.get("status") in ("replaced", "retired"):
                continue
            parts = (it.get("id") or "").split("-")
            if len(parts) >= 3 and parts[0].isdigit():
                used.add("-".join(parts[2:]).lower())
            for t in it.get("tags") or []:
                used.add(str(t).lower())
    return used


def _make_item(day: str, hhmm: str, topic: dict) -> dict:
    code = day.replace("-", "") + "-" + hhmm.replace(":", "")
    slug = topic["slug"]
    asset = topic.get("fallback_asset")
    item = {
        "id": f"{code}-{slug}",
        "slot": slot_times.slot_iso(day, hhmm),
        "pillar": topic.get("pillar") or "movie",
        "source_url": topic.get("source_url") or "",
        "template": "whitebar",
        "hook": topic["hook"],
        "words": topic.get("words") or slug.upper(),
        "title": topic["title"],
        "description": topic["description"],
        "tags": list(topic.get("tags") or [])[:10],
        "target_dur": int(topic.get("target_dur") or 26),
        "status": "queued",
        "market": topic.get("market") or "US/UK",
        "content_kind": topic.get("content_kind") or "trailer",
    }
    if asset:
        url = f"{GH}/{asset}"
        item["fallback_mp4"] = [url]
        item["fallback_release"] = "sources"
        item["fallback_asset"] = asset
    if topic.get("alt_source_urls"):
        item["alt_source_urls"] = list(topic["alt_source_urls"])
    return item


def fill_day(day: str, catalog: list[dict], used: set[str]) -> int:
    items = _load_day(day)
    existing_isos = [it.get("slot") for it in items if it.get("slot")]
    need = slot_times.next_official_to_add(existing_isos, day)
    if not need:
        return 0
    added = 0
    # rotate catalog: prefer unused slugs, weighted by stats performance
    pool = [t for t in catalog if t["slug"] not in used]
    if not pool:
        pool = list(catalog)
    try:
        perf_data = perf.load_perf(REPO)
    except Exception:
        perf_data = {}
    rng = random.Random(day)

    def topic_rank(t: dict) -> float:
        base = 0.0
        if perf_data:
            try:
                base = float(perf.topic_bonus(perf_data, t.get("tags"), t.get("pillar")))
            except Exception:
                base = 0.0
        # Exploration jitter so we do not lock onto one pillar forever.
        return base + rng.random() * 4.0

    pool = sorted(pool, key=topic_rank, reverse=True)
    pi = 0
    for iso in need:
        hhmm = iso[11:16]
        topic = pool[pi % len(pool)]
        pi += 1
        # avoid same slug twice same day
        day_slugs = {"-".join(it["id"].split("-")[2:]) for it in items if it.get("id")}
        tries = 0
        while topic["slug"] in day_slugs and tries < len(pool):
            topic = pool[pi % len(pool)]
            pi += 1
            tries += 1
        item = _make_item(day, hhmm, topic)
        items.append(item)
        used.add(topic["slug"])
        added += 1
    items.sort(key=lambda it: it.get("slot") or "")
    _day_file(day).write_text(json.dumps(items, indent=1, ensure_ascii=False) + "\n")
    return added


def days_ahead(n: int | None = None) -> list[str]:
    n = n if n is not None else max(slot_times.MIN_AHEAD_DAYS + 1, 3)
    now = datetime.now(IST).date()
    return [(now + timedelta(days=i)).isoformat() for i in range(0, n + 1)]


def main() -> int:
    cat = json.loads(CATALOG.read_text()).get("items") or []
    if not cat:
        print("empty topic catalog", file=sys.stderr)
        return 1
    used = _used_slugs()
    total = 0
    # Always cover through min_ahead_days and explicitly Nov 17-19 if still short
    targets = sorted(set(days_ahead() + ["2026-11-17", "2026-11-18", "2026-11-19"]))
    for day in targets:
        if day < "2026-11-14":
            continue
        n = fill_day(day, cat, used)
        if n:
            print(f"{day}: added {n} slot(s)")
        total += n
    print(f"fill_queue done added={total}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
