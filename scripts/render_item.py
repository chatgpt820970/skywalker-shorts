#!/usr/bin/env python3
"""Render one queue item: download -> pipeline.py (+QA) -> 4K lanczos -> Release -> results/<id>.json."""
import glob, json, os, subprocess, sys, time
from datetime import datetime, timedelta, timezone
from pathlib import Path

IST = timezone(timedelta(hours=5, minutes=30))
REPO = Path(__file__).resolve().parent.parent
iid = sys.argv[1]
OUT = REPO / "out"; OUT.mkdir(exist_ok=True)
RES = REPO / "results"; RES.mkdir(exist_ok=True)
run_url = f"{os.environ.get('GITHUB_SERVER_URL','https://github.com')}/{os.environ.get('GITHUB_REPOSITORY','')}/actions/runs/{os.environ.get('GITHUB_RUN_ID','')}"

item = None
for f in sorted(glob.glob(str(REPO / "queue" / "*.json"))):
    for it in json.load(open(f)):
        if it["id"] == iid:
            item = it
if item is None:
    sys.exit(f"item {iid} not found")

def now():
    return datetime.now(IST).strftime("%Y-%m-%d %H:%M IST")

def write(status, detail, **extra):
    r = {"id": iid, "status": status, "detail": detail, "at": now(), "run_url": run_url,
         "slot": item.get("slot"), "title": item.get("title"), **extra}
    (RES / f"{iid}.json").write_text(json.dumps(r, indent=2, ensure_ascii=False))
    print(json.dumps(r, indent=2, ensure_ascii=False))

def probe_wh(p):
    o = subprocess.check_output(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
                                 "stream=width,height", "-of", "csv=p=0", str(p)], text=True).strip()
    return [int(x) for x in o.split(",")[:2]]

def _ok_src(p):
    return p.exists() and p.stat().st_size > 500_000

def download_url(url, dest):
    """Fetch a direct mp4. GitHub release URLs use gh (token) first, then curl."""
    dest.unlink(missing_ok=True)
    if "github.com" in url and "/releases/download/" in url:
        # https://github.com/<owner>/<repo>/releases/download/<tag>/<asset>
        parts = url.split("/releases/download/")
        repo = parts[0].split("github.com/")[-1].strip("/")
        tag, asset = parts[1].split("/", 1)
        r = subprocess.run(["gh", "release", "download", tag, "-p", asset, "-R", repo, "-D", str(dest.parent), "--clobber"],
                           capture_output=True, text=True)
        cand = dest.parent / asset
        if r.returncode == 0 and _ok_src(cand):
            if cand.resolve() != dest.resolve():
                dest.write_bytes(cand.read_bytes())
            return True, f"gh:{url}"
        err_gh = (r.stderr or r.stdout or "?").strip().splitlines()[-1][:300] if (r.stderr or r.stdout) else "gh failed"
        # fall through to curl (public repo)
    r = subprocess.run(["curl", "-fsSL", "--retry", "3", "-L", "-o", str(dest), url], capture_output=True, text=True)
    if r.returncode == 0 and _ok_src(dest):
        return True, url
    return False, f"direct mp4 {url}: curl exit {r.returncode} {(r.stderr or '')[-200:]}"

# 1. download (official source; YouTube via yt-dlp, else official direct mp4s / Release fallback)
src = OUT / f"{iid}_src.mp4"
errors, used = [], None
if item.get("source_url"):
    for attempt in range(2):
        r = subprocess.run(["yt-dlp", "--no-playlist", "--remote-components", "ejs:github",
                            "-f", "bv*[height<=1080][ext=mp4]+ba[ext=m4a]/b[ext=mp4]/bv*+ba/b",
                            "--merge-output-format", "mp4", "-o", str(src), "--force-overwrites",
                            item["source_url"]], capture_output=True, text=True)
        if r.returncode == 0 and _ok_src(src):
            used = item["source_url"]; break
        errors.append("yt-dlp: " + (r.stderr.strip().splitlines() or ["?"])[-1][:300])
        time.sleep(5)
if not used:
    for u in item.get("fallback_mp4", []) or []:
        ok, info = download_url(u, src)
        if ok:
            used = info; break
        errors.append(info)
if not used and item.get("fallback_release") and item.get("fallback_asset"):
    tag, asset = item["fallback_release"], item["fallback_asset"]
    repo = os.environ.get("GITHUB_REPOSITORY", "chatgpt820970/skywalker-shorts")
    r = subprocess.run(["gh", "release", "download", tag, "-p", asset, "-R", repo, "-D", str(OUT), "--clobber"],
                       capture_output=True, text=True)
    cand = OUT / asset
    if r.returncode == 0 and _ok_src(cand):
        if cand.resolve() != src.resolve():
            src.write_bytes(cand.read_bytes())
        used = f"gh:release/{tag}/{asset}"
    else:
        errors.append("gh release " + (r.stderr.strip().splitlines() or ["?"])[-1][:300])
if not used:
    write("failed", "download failed: " + " | ".join(errors))
    print(f"::error::{iid}: download failed")
    sys.exit(1)

# 2. edit + QA
out = OUT / f"{iid}_1080.mp4"
qa_json = OUT / f"{iid}_qa.json"
cmd = [sys.executable, str(REPO / "pipeline.py"), "--src", str(src), "--out", str(out),
       "--template", item.get("template", "whitebar"), "--hook", item["hook"],
       "--pillar", item.get("pillar", "movie"), "--item-id", iid, "--music", "auto",
       "--target-dur", str(item.get("target_dur", 32)), "--qa-json", str(qa_json)]
if item.get("words"):
    cmd += ["--words", item["words"]]
env = dict(os.environ, SKY_WORK=str(OUT / f"work_{iid}"), SKY_ROOT=str(OUT))
rc = subprocess.call(cmd, env=env)
qa = json.loads(qa_json.read_text()) if qa_json.exists() else {}
credit = {}
cpath = OUT / f"work_{iid}" / "music_credit.json"
if cpath.exists():
    credit = json.loads(cpath.read_text())
checks = {k: v.get("ok") for k, v in (qa.get("checks") or {}).items()}
if rc != 0 or not out.exists():
    write("failed", f"pipeline/QA failed (exit {rc}): " + "; ".join(qa.get("issues", [])[:5])[:800],
          qa=checks, source=used)
    sys.exit(1)

# 3. 4K: upscale with lanczos when the source is under 4K
sw, sh = probe_wh(src)
final = OUT / f"render-{iid}.mp4"
if min(sw, sh) < 2160:
    subprocess.check_call(["ffmpeg", "-y", "-i", str(out), "-vf", "scale=2160:3840:flags=lanczos",
                           "-c:v", "libx264", "-preset", "medium", "-crf", "18", "-pix_fmt", "yuv420p",
                           "-c:a", "copy", "-movflags", "+faststart", str(final)])
    up = f"lanczos {sw}x{sh} source -> 2160x3840"
else:
    subprocess.check_call(["ffmpeg", "-y", "-i", str(out), "-vf", "scale=2160:3840:flags=lanczos",
                           "-c:v", "libx264", "-crf", "18", "-c:a", "copy", str(final)])
    up = "4K source, scaled to 2160x3840"
fw, fh = probe_wh(final)

# 4. Release asset render-<id>
desc = item.get("description", "")
if credit.get("credit"):
    desc = desc.rstrip() + "\n\nMusic:\n" + credit["credit"]
notes = f"{item.get('title','')}\n\n{desc}\n\nTags: {', '.join(item.get('tags', []))}\nSource: {used}\nRun: {run_url}"
tag = f"render-{iid}"
r = subprocess.run(["gh", "release", "view", tag], capture_output=True)
if r.returncode != 0:
    subprocess.check_call(["gh", "release", "create", tag, str(final), "--title", f"render {iid}", "--notes", notes])
else:
    subprocess.check_call(["gh", "release", "upload", tag, str(final), "--clobber"])
repo = os.environ.get("GITHUB_REPOSITORY", "")
write("done", f"rendered {fw}x{fh} ({up}), QA ok", release_url=f"https://github.com/{repo}/releases/tag/{tag}",
      asset_url=f"https://github.com/{repo}/releases/download/{tag}/{final.name}",
      qa=checks, drop_t=qa.get("drop_t"), energy=qa.get("energy"), source=used,
      music=credit.get("file"), music_credit=credit.get("credit"), description_final=desc)
