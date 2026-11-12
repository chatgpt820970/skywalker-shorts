#!/usr/bin/env python3
"""Self-heal cancelled runs (manual or other-bot cancels).

render: every item whose `render (<id>)` job was cancelled is re-dispatched, unless
  - a newer render run already has a job for that id (superseded / in flight),
  - its result is already done (never re-render a finished/posted video here),
  - it was cancelled 3+ times in the window (stop looping; report instead).
publish: re-dispatched when no newer publish run exists. buffer_post.py is idempotent
  (matches existing posts by id/title), so this can never double-post.
Window: runs created in the last HEAL_HOURS (default 8).
"""
import json, os, subprocess, sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
GH_REPO = os.environ.get("GITHUB_REPOSITORY", "chatgpt820970/skywalker-shorts")
HOURS = float(os.environ.get("HEAL_HOURS", "8"))
DRY = os.environ.get("HEAL_DRY") == "1"


def gh(*args):
    return json.loads(subprocess.check_output(["gh", "api", *args], text=True))


def ts(s):
    return datetime.fromisoformat(s.replace("Z", "+00:00"))


def runs(wf):
    since = datetime.now(timezone.utc) - timedelta(hours=HOURS)
    out = gh(f"repos/{GH_REPO}/actions/workflows/{wf}/runs?per_page=50")["workflow_runs"]
    return sorted([r for r in out if ts(r["created_at"]) >= since], key=lambda r: r["created_at"])


def job_ids(run_id):
    jobs = gh(f"repos/{GH_REPO}/actions/runs/{run_id}/jobs?per_page=100")["jobs"]
    res = {}
    for j in jobs:
        n = j.get("name") or ""
        if n.startswith("render (") and n.endswith(")"):
            res[n[8:-1]] = j.get("conclusion") or j.get("status")
    return res


def result_status(iid):
    p = REPO / "results" / f"{iid}.json"
    try:
        return json.loads(p.read_text()).get("status")
    except Exception:
        return None


def queue_status(iid):
    for p in (REPO / "queue").glob("*.json"):
        try:
            for it in json.loads(p.read_text()):
                if it.get("id") == iid:
                    return it.get("status")
        except Exception:
            pass
    return None


def dispatch(wf, fields):
    cmd = ["gh", "workflow", "run", wf, "-R", GH_REPO]
    for k, v in fields.items():
        cmd += ["-f", f"{k}={v}"]
    print("dispatch:", " ".join(cmd[3:]))
    if not DRY:
        subprocess.check_call(cmd)


def heal_render():
    rs = runs("render.yml")
    jobs = {r["id"]: job_ids(r["id"]) for r in rs}
    todo, gave_up = [], []
    for i, r in enumerate(rs):
        if r.get("conclusion") != "cancelled":
            continue
        for iid, concl in jobs[r["id"]].items():
            if concl != "cancelled" or iid in todo:
                continue
            newer = [x for x in rs[i + 1:] if iid in jobs[x["id"]]]
            if newer:
                print(f"{iid}: cancelled in run {r['id']} but superseded by run {newer[-1]['id']}")
                continue
            if result_status(iid) == "done":
                print(f"{iid}: cancelled in run {r['id']} but already done; not re-rendering")
                continue
            if queue_status(iid) in ("retired", "replaced", "moved", "skip"):
                print(f"{iid}: cancelled but retired in the queue; skipping")
                continue
            n_cancel = sum(1 for x in rs if jobs[x["id"]].get(iid) == "cancelled")
            if n_cancel >= 3:
                gave_up.append(iid)
                continue
            todo.append(iid)
    if todo:
        dispatch("render.yml", {"ids": ",".join(todo), "max": str(len(todo)), "retry_failed": "true"})
    else:
        print("render: nothing to heal")
    return gave_up


def heal_publish():
    rs = runs("publish.yml")
    if rs and rs[-1].get("conclusion") == "cancelled":
        n = sum(1 for x in rs[-3:] if x.get("conclusion") == "cancelled")
        if n >= 3:
            return ["publish"]
        dispatch("publish.yml", {})
    else:
        print("publish: nothing to heal")
    return []


if __name__ == "__main__":
    gave = heal_render() + heal_publish()
    if gave:
        print("::error::cancelled 3+ times, not re-dispatching:", gave)
        sys.exit(1)
