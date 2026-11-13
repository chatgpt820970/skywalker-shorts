#!/usr/bin/env python3
"""Post passing renders to YouTube (Skywalker) through the Buffer GraphQL API.

Idempotent: a result with buffer_post_id is never posted again, and before posting we look
for an existing Buffer post on the channel with the same title (also catches manual Studio
uploads that Buffer synced). Writes buffer_* fields back into results/<id>.json.

Env: BUFFER_API_KEY (secret; never printed). DRY_RUN=1 prints what would be posted.
IDS=comma list limits to those ids. Exit 1 if a createPost failed or a scheduled post errored.
"""
import glob, json, os, sys, urllib.error, urllib.request
from datetime import datetime, timedelta, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
import hook_text  # noqa: E402
import sky_ai  # noqa: E402

API = "https://api.buffer.com"
IST = timezone(timedelta(hours=5, minutes=30))
CFG = json.loads((REPO / "config" / "publish.json").read_text())
CH, ORG = CFG["buffer_channel_id"], CFG["buffer_org_id"]
DRY = os.environ.get("DRY_RUN") == "1"
ONLY = {x.strip() for x in os.environ.get("IDS", "").split(",") if x.strip()}


def gql(query, variables=None):
    key = os.environ.get("BUFFER_API_KEY")
    if not key:
        raise SystemExit("BUFFER_API_KEY secret is not set")
    req = urllib.request.Request(API, data=json.dumps({"query": query, "variables": variables or {}}).encode(),
                                 headers={"Content-Type": "application/json", "Authorization": "Bearer " + key})
    try:
        with urllib.request.urlopen(req, timeout=60) as r:
            d = json.loads(r.read().decode())
    except urllib.error.HTTPError as e:
        raise RuntimeError(f"Buffer HTTP {e.code}: {e.read().decode()[:300]}")
    if d.get("errors"):
        raise RuntimeError("Buffer GraphQL error: " + "; ".join(x.get("message", "?") for x in d["errors"])[:400])
    return d["data"]


def channel_posts():
    q = """query P($org: OrganizationId!, $ch: [ChannelId!]) {
      posts(first: 100, input: {organizationId: $org, filter: {channelIds: $ch}}) {
        edges { node { id status dueAt sentAt externalLink error { message }
          metadata { __typename ... on YoutubePostMetadata { title } } } } } }"""
    d = gql(q, {"org": ORG, "ch": [CH]})
    return [e["node"] for e in d["posts"]["edges"]]


def norm(t):
    return " ".join((t or "").lower().split())


def create(r, due, share_now):
    title = r["title"][:100]
    text = r.get("description_final") or r.get("description") or title
    m = """mutation C($input: CreatePostInput!) { createPost(input: $input) {
      ... on PostActionSuccess { post { id status dueAt } }
      ... on MutationError { message } } }"""
    inp = {
        "text": text[:4900],
        "channelId": CH,
        "schedulingType": "automatic",
        "mode": "shareNow" if share_now else "customScheduled",
        "assets": [{"video": {"url": r["asset_url"]}}],
        "metadata": {"youtube": {"title": title, "privacy": "public", "madeForKids": False,
                                 "categoryId": str(CFG.get("youtube_category_id") or "24"),
                                 "notifySubscribers": True, "embeddable": True}},
        "source": "skywalker-shorts",
    }
    if not share_now:
        inp["dueAt"] = due.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.000Z")
    d = gql(m, {"input": inp})["createPost"]
    if not d.get("post"):
        raise RuntimeError("createPost: " + str(d.get("message")))
    return d["post"]


def edit(r, post_id, due_at):
    m = """mutation E($input: EditPostInput!) { editPost(input: $input) {
      ... on PostActionSuccess { post { id status dueAt } }
      ... on MutationError { message } } }"""
    inp = {"id": post_id, "text": (r.get("description_final") or r.get("description") or r["title"])[:4900],
           "assets": [{"video": {"url": r["asset_url"]}}],
           "metadata": {"youtube": {"title": r["title"][:100], "privacy": "public", "madeForKids": False,
                                    "categoryId": str(CFG.get("youtube_category_id") or "24"),
                                    "notifySubscribers": True, "embeddable": True}}}
    if due_at:
        inp["dueAt"] = due_at
        inp["mode"] = "customScheduled"
        inp["schedulingType"] = "automatic"
    d = gql(m, {"input": inp})["editPost"]
    if not d.get("post"):
        raise RuntimeError("editPost: " + str(d.get("message")))
    return d["post"]


def save(p, r):
    Path(p).write_text(json.dumps(r, indent=2, ensure_ascii=False))


def main():
    now = datetime.now(IST)
    start = datetime.fromisoformat(CFG["post_from_slot"])
    late_h = float(CFG.get("late_post_hours") or 6)
    posts = None
    failed, changed = [], []
    for p in sorted(glob.glob(str(REPO / "results" / "*.json"))):
        r = json.loads(Path(p).read_text())
        iid = r.get("id")
        if ONLY and iid not in ONLY:
            continue
        if r.get("status") != "done" or not r.get("asset_url") or not r.get("slot"):
            continue
        # Never post demo / dry-run / no_publish renders (test-renders release).
        if r.get("no_publish") or r.get("demo") or (r.get("asset_url") or "").find("/test-renders/") >= 0:
            continue
        if (r.get("id") or "").startswith("demo-") or (r.get("id") or "").startswith("test-"):
            continue
        slot = datetime.fromisoformat(r["slot"])
        if slot < start:
            continue
        if posts is None:
            posts = channel_posts()
        byid = {x["id"]: x for x in posts}
        # refresh status of an already-created post; a re-render with new hook/title edits the scheduled post
        if r.get("buffer_post_id"):
            x = byid.get(r["buffer_post_id"])
            bt = norm(((x or {}).get("metadata") or {}).get("title"))
            if x and x.get("status") == "scheduled" and bt and bt != norm(r["title"]) and r.get("asset_url"):
                print(f"{iid}: re-rendered with new title; editing scheduled Buffer post {x['id']} (same slot)")
                if not DRY:
                    try:
                        ed = edit(r, x["id"], x.get("dueAt"))
                        r.update(buffer_status=ed.get("status"), buffer_due_at=ed.get("dueAt"),
                                 buffer_edited_at=now.strftime("%Y-%m-%d %H:%M IST"))
                        save(p, r); changed.append(iid)
                        x["metadata"] = {"title": r["title"]}
                    except Exception as e:
                        failed.append(f"{iid}: edit failed: {e}")
                continue
            if x:
                st = x.get("status")
                upd = {"buffer_status": st, "youtube_url": x.get("externalLink") or r.get("youtube_url")}
                if x.get("sentAt"):
                    upd["buffer_posted_at"] = x["sentAt"]
                if x.get("error"):
                    upd["buffer_error"] = x["error"].get("message")
                    failed.append(f"{iid}: Buffer post {x['id']} error: {upd['buffer_error']}")
                if any(r.get(k) != v for k, v in upd.items()):
                    r.update(upd); save(p, r); changed.append(iid)
            continue
        # gate re-check on metadata (render-time gate already checked the video itself)
        g = r.get("gate") or {}
        body = (r.get("description_final") or r.get("description") or "").split("\n\nMusic:")[0]
        bad = sky_ai.banned_hits(r.get("title"), body)
        errs = hook_text.check_item_hook_fields(r) + (["banned: " + ",".join(bad)] if bad else [])
        if hook_text.template_errors(r.get("hook") or "") or hook_text.template_errors(r.get("title") or ""):
            errs.append("old 'The reason why' template; needs re-render")
        if g and g.get("ok") is False:
            errs.append("render gate failed")
        if not (r.get("title") or "").strip():
            errs.append("empty title")
        if errs:
            print(f"{iid}: NOT posting ({'; '.join(errs)})")
            continue
        # idempotency: same title already on the channel (Buffer or manual Studio upload)
        dup = next((x for x in posts if norm((x.get("metadata") or {}).get("title")) == norm(r["title"])), None)
        if dup:
            print(f"{iid}: already on Buffer as {dup['id']} ({dup.get('status')}); recording, not posting")
            r.update(buffer_post_id=dup["id"], buffer_status=dup.get("status"), buffer_due_at=dup.get("dueAt"),
                     youtube_url=dup.get("externalLink"), buffer_note="matched existing post by title")
            save(p, r); changed.append(iid)
            continue
        if slot < now - timedelta(hours=late_h):
            print(f"{iid}: slot {r['slot']} is more than {late_h}h past and not posted - needs a manual decision")
            failed.append(f"{iid}: missed slot not auto-posted (>{late_h}h late)")
            continue
        share_now = slot <= now + timedelta(minutes=3)
        print(f"{iid}: {'shareNow' if share_now else 'schedule ' + r['slot']} -> Buffer channel {CH}: {r['title']!r}")
        if DRY:
            continue
        try:
            post = create(r, slot, share_now)
        except Exception as e:
            failed.append(f"{iid}: {e}")
            r.update(buffer_status="create_failed", buffer_error=str(e)[:300]); save(p, r); changed.append(iid)
            continue
        r.update(buffer_post_id=post["id"], buffer_status=post.get("status"), buffer_due_at=post.get("dueAt"),
                 buffer_created_at=now.strftime("%Y-%m-%d %H:%M IST"))
        save(p, r); changed.append(iid)
        posts.append({"id": post["id"], "status": post.get("status"), "metadata": {"title": r["title"]}})
        print(f"{iid}: Buffer post {post['id']} status={post.get('status')} dueAt={post.get('dueAt')}")
    out = os.environ.get("GITHUB_OUTPUT")
    if out:
        open(out, "a").write(f"changed={','.join(changed)}\n")
    if failed:
        print("::error::" + " | ".join(failed)[:900])
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
