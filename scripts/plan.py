#!/usr/bin/env python3
"""Pick which queue items to render this run (only due + still queued; capped).

A push that adds 28 ideas therefore renders only what is due soon, never all at once.
"""
import glob, json, os, sys
from datetime import datetime, timedelta, timezone

IST = timezone(timedelta(hours=5, minutes=30))
now = datetime.now(IST)
ahead = float(os.environ.get("RENDER_AHEAD_HOURS", "20"))
cap = int(os.environ.get("MAX_PER_RUN", "3") or 3)
force_ids = [x.strip() for x in os.environ.get("IDS", "").split(",") if x.strip()]
retry_failed = os.environ.get("RETRY_FAILED", "false") == "true"

items = {}
for f in sorted(glob.glob("queue/*.json")):
    try:
        for it in json.load(open(f)):
            items[it["id"]] = it
    except Exception as e:  # malformed file: skip, do not crash the planner
        print(f"skip {f}: {e}", file=sys.stderr)

def result(i):
    p = f"results/{i}.json"
    return json.load(open(p)) if os.path.exists(p) else None

pick = []
if force_ids:
    pick = [i for i in force_ids if i in items]
else:
    for i, it in sorted(items.items(), key=lambda kv: kv[1].get("slot", "")):
        if it.get("status", "queued") != "queued":
            continue
        r = result(i)
        if r and (r.get("status") == "done" or (r.get("status") == "failed" and not retry_failed)):
            continue
        try:
            slot = datetime.fromisoformat(it["slot"])
        except Exception:
            continue
        if slot - now <= timedelta(hours=ahead) and now - slot <= timedelta(hours=24):
            pick.append(i)
pick = pick[:cap]
print(f"now={now.isoformat()} ahead={ahead}h cap={cap} picked={pick}", file=sys.stderr)
out = os.environ.get("GITHUB_OUTPUT")
line = f"ids={json.dumps(pick)}\ncount={len(pick)}\n"
if out:
    open(out, "a").write(line)
else:
    print(line)
