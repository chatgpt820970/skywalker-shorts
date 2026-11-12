#!/usr/bin/env python3
"""Pick which queue items to render this run (only due + still queued; capped).

A push that adds many ideas therefore renders only what is due soon, never all at once.
Cron pre-renders each slot ~2 hours before upload (ahead window is 6 hours so a skipped cron cannot miss a slot; a
75-minute render is finished about 2 hours before the slot). Done items are never
re-rendered. Items in retry_queued are left to the auto-retry workflow unless stale.
"""
import glob, json, os, sys
from datetime import datetime, timedelta, timezone

IST = timezone(timedelta(hours=5, minutes=30))
now = datetime.now(IST)
ahead = float(os.environ.get("RENDER_AHEAD_HOURS", "3"))
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

def stale_retry(r, hours=2):
    """Backup: if retry_queued was left hanging, cron may pick it up after 2h."""
    if not r or r.get("status") != "retry_queued":
        return False
    at = r.get("at") or ""
    try:
        # "YYYY-MM-DD HH:MM IST"
        dt = datetime.strptime(at.replace(" IST", ""), "%Y-%m-%d %H:%M").replace(tzinfo=IST)
        return now - dt >= timedelta(hours=hours)
    except Exception:
        return True

pick = []
if force_ids:
    pick = [i for i in force_ids if i in items]
else:
    for i, it in sorted(items.items(), key=lambda kv: kv[1].get("slot", "")):
        if it.get("status", "queued") != "queued":
            continue
        r = result(i)
        if r and r.get("status") == "done":
            continue  # never re-render a finished item
        if r and r.get("status") == "failed" and not retry_failed:
            continue
        if r and r.get("status") == "retry_queued" and not retry_failed and not stale_retry(r):
            continue  # auto-retry.yml handles these
        try:
            slot = datetime.fromisoformat(it["slot"])
        except Exception:
            continue
        if slot - now <= timedelta(hours=ahead) and now - slot <= timedelta(hours=24):
            pick.append(i)
# --- never start an id that another in-progress render run is already working on
def in_flight():
    repo = os.environ.get("GITHUB_REPOSITORY")
    if not repo or not os.environ.get("GH_TOKEN"):
        return set()
    import subprocess
    busy = set()
    try:
        runs = json.loads(subprocess.run(
            ["gh", "api", f"repos/{repo}/actions/workflows/render.yml/runs?status=in_progress&per_page=10"],
            capture_output=True, text=True, timeout=30).stdout or "{}").get("workflow_runs") or []
        for r in runs:
            if str(r.get("id")) == os.environ.get("GITHUB_RUN_ID"):
                continue
            jobs = json.loads(subprocess.run(["gh", "api", f"repos/{repo}/actions/runs/{r['id']}/jobs?per_page=50"],
                                             capture_output=True, text=True, timeout=30).stdout or "{}").get("jobs") or []
            for j in jobs:
                n = j.get("name") or ""
                if n.startswith("render (") and j.get("status") != "completed":
                    busy.add(n[len("render ("):-1])
    except Exception as e:
        print(f"in-flight check skipped: {e}", file=sys.stderr)
    return busy

busy = in_flight()
if busy:
    print(f"in flight elsewhere (skipped): {sorted(busy)}", file=sys.stderr)
pick = [i for i in pick if i not in busy]

# --- backup rule: a slot whose queued items all ended 'failed' (nothing done) gets the next unused
# item from config/backup_pool.json, rendered as <YYYYMMDD-HHMM>-<slug> for that slot. No slot is skipped.
if not force_ids:
    try:
        pool = json.load(open("config/backup_pool.json")).get("items") or []
    except Exception:
        pool = []
    used = set()
    for f in glob.glob("results/*.json"):
        try:
            r = json.load(open(f))
        except Exception:
            continue
        if r.get("backup_item"):
            used.add(r["backup_item"])
    slots = {}
    for i, it in items.items():
        if it.get("slot"):
            slots.setdefault(it["slot"], []).append(i)
    for slot_s, ids in sorted(slots.items()):
        try:
            slot = datetime.fromisoformat(slot_s)
        except Exception:
            continue
        if not (slot - now <= timedelta(hours=ahead) and now - slot <= timedelta(hours=6)):
            continue
        st = [(result(i) or {}).get("status") for i in ids]
        live = [i for i in ids if items[i].get("status", "queued") == "queued"]
        if "done" in st or not live or any((result(i) or {}).get("status") != "failed" for i in live):
            continue
        code = slot.strftime("%Y%m%d-%H%M")
        if any(os.path.exists(f"results/{code}-{b['id'][len('backup-'):]}.json") for b in pool):
            continue  # a backup already ran for this slot
        cand = [b for b in pool if b["id"] not in used]
        if not cand:
            print(f"::warning::slot {slot_s} failed and the backup pool is empty", file=sys.stderr)
            continue
        bid = f"{cand[0]['id']}@{code}"
        if bid not in busy and f"{code}-{cand[0]['id'][len('backup-'):]}" not in busy:
            print(f"backup for failed slot {slot_s}: {bid}", file=sys.stderr)
            pick.insert(0, bid)
            used.add(cand[0]["id"])

pick = pick[:cap]
print(f"now={now.isoformat()} ahead={ahead}h cap={cap} picked={pick}", file=sys.stderr)
out = os.environ.get("GITHUB_OUTPUT")
line = f"ids={json.dumps(pick)}\ncount={len(pick)}\n"
if out:
    open(out, "a").write(line)
else:
    print(line)
