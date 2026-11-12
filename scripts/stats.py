#!/usr/bin/env python3
"""Daily per-video stats -> stats/YYYY-MM-DD.json (+ stats/latest.json).

Sources, first that works wins (all are tried; results merged by video id):
  1. YouTube Data API v3 if YOUTUBE_API_KEY is set (views, likes, comments, published time).
  2. Buffer post metrics (views, reactions, comments) for the Skywalker channel (BUFFER_API_KEY).
  3. Public channel Shorts page via yt-dlp flat metadata (often bot-blocked on runners; tolerated).
Also flags videos that crossed 1,000 views since the previous snapshot ("crossed_1000").
"""
import glob, json, os, subprocess, sys, urllib.parse, urllib.request
from datetime import datetime, timedelta, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "scripts"))
IST = timezone(timedelta(hours=5, minutes=30))
CHANNEL = "UCZQHU7LI-1EyLqU7fJz_q-g"


def _get(url, headers=None, timeout=30):
    req = urllib.request.Request(url, headers=headers or {"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode())


def from_youtube_api(key):
    base = "https://www.googleapis.com/youtube/v3/"
    ch = _get(base + "channels?" + urllib.parse.urlencode({"part": "contentDetails,statistics", "id": CHANNEL, "key": key}))
    item = ch["items"][0]
    uploads = item["contentDetails"]["relatedPlaylists"]["uploads"]
    ids, tok = [], None
    while True:
        q = {"part": "contentDetails", "playlistId": uploads, "maxResults": 50, "key": key}
        if tok:
            q["pageToken"] = tok
        pl = _get(base + "playlistItems?" + urllib.parse.urlencode(q))
        ids += [x["contentDetails"]["videoId"] for x in pl.get("items", [])]
        tok = pl.get("nextPageToken")
        if not tok or len(ids) >= 500:
            break
    vids = {}
    for i in range(0, len(ids), 50):
        v = _get(base + "videos?" + urllib.parse.urlencode({"part": "statistics,snippet,contentDetails",
                                                             "id": ",".join(ids[i:i + 50]), "key": key}))
        for x in v.get("items", []):
            st = x.get("statistics", {})
            vids[x["id"]] = {"title": x["snippet"]["title"], "published": x["snippet"]["publishedAt"],
                             "views": int(st.get("viewCount", 0)), "likes": int(st.get("likeCount", 0)),
                             "comments": int(st.get("commentCount", 0)), "duration": x["contentDetails"].get("duration")}
    return vids, {"subscribers": int(item["statistics"].get("subscriberCount", 0)),
                  "channel_views": int(item["statistics"].get("viewCount", 0))}


def from_buffer():
    from buffer_post import gql, CH, ORG
    q = """query P($org: OrganizationId!, $ch: [ChannelId!]) {
      posts(first: 100, input: {organizationId: $org, filter: {channelIds: $ch, status: [sent]}}) {
        edges { node { id sentAt externalLink metricsUpdatedAt
          metadata { ... on YoutubePostMetadata { title } } metrics { type value } } } } }"""
    d = gql(q, {"org": ORG, "ch": [CH]})
    vids = {}
    for e in d["posts"]["edges"]:
        n = e["node"]
        link = n.get("externalLink") or ""
        vid = link.rstrip("/").split("/")[-1].split("?")[0] if link else None
        if not vid:
            continue
        m = {x["type"]: x["value"] for x in n.get("metrics") or []}
        vids[vid] = {"title": (n.get("metadata") or {}).get("title"), "published": n.get("sentAt"),
                     "views": int(m.get("views") or 0), "likes": int(m.get("reactions") or 0),
                     "comments": int(m.get("comments") or 0), "buffer_post_id": n["id"],
                     "metrics_at": n.get("metricsUpdatedAt")}
    return vids


def from_scrape():
    r = subprocess.run(["yt-dlp", "--flat-playlist", "-J", "--no-warnings",
                        f"https://www.youtube.com/channel/{CHANNEL}/shorts"], capture_output=True, text=True, timeout=180)
    if r.returncode != 0:
        raise RuntimeError((r.stderr or "").strip().splitlines()[-1][:200] if r.stderr else "yt-dlp failed")
    d = json.loads(r.stdout)
    return {e["id"]: {"title": e.get("title"), "views": int(e.get("view_count") or 0)} for e in d.get("entries") or []}


def main():
    now = datetime.now(IST)
    vids, sources, errors, channel = {}, [], {}, {}
    key = os.environ.get("YOUTUBE_API_KEY")
    if key:
        try:
            v, channel = from_youtube_api(key)
            vids.update(v); sources.append("youtube_api")
        except Exception as e:
            errors["youtube_api"] = f"{type(e).__name__}: {str(e)[:200]}"
    if os.environ.get("BUFFER_API_KEY"):
        try:
            for vid, x in from_buffer().items():
                if vid in vids:
                    vids[vid].setdefault("buffer_post_id", x.get("buffer_post_id"))
                else:
                    vids[vid] = x
            sources.append("buffer_metrics")
        except Exception as e:
            errors["buffer"] = f"{type(e).__name__}: {str(e)[:200]}"
    if "youtube_api" not in sources:
        try:
            for vid, x in from_scrape().items():
                if vid in vids:
                    if x.get("views", 0) > vids[vid].get("views", 0):
                        vids[vid]["views"] = x["views"]; vids[vid]["views_source"] = "scrape"
                else:
                    vids[vid] = x
            sources.append("scrape")
        except Exception as e:
            errors["scrape"] = f"{type(e).__name__}: {str(e)[:200]}"
    # link videos to queue ids via results youtube_url
    res = {}
    for p in glob.glob(str(REPO / "results" / "*.json")):
        try:
            r = json.loads(Path(p).read_text())
        except Exception:
            continue
        link = r.get("youtube_url") or ""
        if link:
            res[link.rstrip("/").split("/")[-1]] = r
    for vid, x in vids.items():
        if vid in res:
            r = res[vid]
            x.update(queue_id=r.get("id"), slot=r.get("slot"), music=r.get("music"), hook=r.get("hook"))
    prev = {}
    lp = REPO / "stats" / "latest.json"
    if lp.exists():
        try:
            prev = json.loads(lp.read_text()).get("videos", {})
        except Exception:
            prev = {}
    crossed = [vid for vid, x in vids.items() if x.get("views", 0) >= 1000 and (prev.get(vid, {}).get("views", 0) < 1000)]
    out = {"date": now.strftime("%Y-%m-%d"), "at": now.strftime("%Y-%m-%d %H:%M IST"), "channel_id": CHANNEL,
           "sources": sources, "errors": errors, "channel": channel,
           "totals": {"videos": len(vids), "views": sum(x.get("views", 0) for x in vids.values()),
                      "likes": sum(x.get("likes", 0) for x in vids.values()),
                      "comments": sum(x.get("comments", 0) for x in vids.values())},
           "crossed_1000": crossed,
           "videos": dict(sorted(vids.items(), key=lambda kv: -kv[1].get("views", 0)))}
    (REPO / "stats").mkdir(exist_ok=True)
    (REPO / "stats" / f"{out['date']}.json").write_text(json.dumps(out, indent=1, ensure_ascii=False))
    lp.write_text(json.dumps(out, indent=1, ensure_ascii=False))
    print(json.dumps({k: out[k] for k in ("at", "sources", "errors", "totals", "crossed_1000")}, ensure_ascii=False))
    if crossed:
        print("::notice::crossed 1,000 views: " + ", ".join(f"{v} ({vids[v].get('title')})" for v in crossed))
    if not sources:
        print("::error::no stats source worked")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
