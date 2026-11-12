#!/usr/bin/env python3
"""Render one queue item: download -> pipeline.py (+QA) -> 4K lanczos -> Release -> results/<id>.json.

Automatic retry ladder (self-heal, no bot required):
  1) normal (yt-dlp, then official fallbacks)
  2) force the sources-release fallback mp4
  3) fallback mp4 + a different credited music track + alternate scene
     selection + relaxed NON-hard-rule thresholds
     (never relaxes no-crop, layout, or no-logo/title)

If all three attempts in this job fail and results/<id>.json has not yet used
its outer auto-retry, we write status=retry_queued and exit 0. A companion
workflow re-dispatches the item once more later. Only after that last attempt
fails do we write status=failed and exit 1 (that is what wakes BIBI).
"""
import glob, json, os, subprocess, sys, time
from datetime import datetime, timedelta, timezone
from pathlib import Path

IST = timezone(timedelta(hours=5, minutes=30))
REPO = Path(__file__).resolve().parent.parent
iid = sys.argv[1]
OUT = REPO / "out"; OUT.mkdir(exist_ok=True)
RES = REPO / "results"; RES.mkdir(exist_ok=True)
run_url = f"{os.environ.get('GITHUB_SERVER_URL','https://github.com')}/{os.environ.get('GITHUB_REPOSITORY','')}/actions/runs/{os.environ.get('GITHUB_RUN_ID','')}"
MAX_AUTO_RETRIES = 1  # outer workflow_run re-dispatch budget

item = None
for f in sorted(glob.glob(str(REPO / "queue" / "*.json"))):
    for it in json.load(open(f)):
        if it["id"] == iid:
            item = it
if item is None:
    sys.exit(f"item {iid} not found")

# Hard rule: white-bar hook and title must be pure English (no Hindi/Hinglish).
# Body subtitles may still follow the trailer audio. Fail before any download
# so a bad hook is never baked into a render; do not write results/<id>.json
# so the next cron can pick the item up after the queue text is fixed.
sys.path.insert(0, str(REPO))
import hook_text
_hook_errs = hook_text.check_item_hook_fields(item)
if _hook_errs:
    sys.exit("English-only hook/title failed: " + "; ".join(_hook_errs))

import gate
import gemini
import sky_ai

os.environ.setdefault("SKY_GEMINI_BUDGET_FILE", str(OUT / f"gemini_budget_{iid}.json"))
QUEUE_META = {k: item.get(k) for k in ("hook", "title", "description", "tags")}
GEMINI_META = None
OUT_RES = int(item.get("out_res") or os.environ.get("SKY_OUT_RES") or 2160)


def fetch_music_library():
    """Download the trending-song mp3s from the 'music' release into music/lib (CI only, cheap)."""
    lp = REPO / "music" / "library.json"
    if not lp.exists():
        return 0
    lib = json.loads(lp.read_text())
    ldir = REPO / (lib.get("dir") or "music/lib")
    ldir.mkdir(parents=True, exist_ok=True)
    have = {p.name for p in ldir.glob("*.mp3")}
    need = [t["file"] for t in lib.get("tracks") or [] if t["file"] not in have]
    if need:
        repo = os.environ.get("GITHUB_REPOSITORY", "chatgpt820970/skywalker-shorts")
        args = ["gh", "release", "download", lib.get("release") or "music", "-R", repo, "-D", str(ldir), "--clobber"]
        for f in need:
            args += ["-p", f]
        r = subprocess.run(args, capture_output=True, text=True)
        if r.returncode != 0:
            print("music library download failed (falling back to credits.json):", (r.stderr or "")[-300:])
    n = len(list(ldir.glob("*.mp3")))
    print(f"music library: {n} track(s) in {ldir}")
    return n


def recent_music(window=None):
    """Songs used by the last N renders (by slot), so a song never repeats within N videos."""
    try:
        window = window or int(json.loads((REPO / "music" / "library.json").read_text()).get("no_repeat_window") or 8)
    except Exception:
        window = window or 8
    rows = []
    for p in glob.glob(str(RES / "*.json")):
        try:
            r = json.loads(Path(p).read_text())
        except Exception:
            continue
        if r.get("id") == iid or r.get("status") != "done" or not r.get("music"):
            continue
        rows.append((r.get("slot") or r.get("at") or "", r["music"]))
    rows.sort()
    return [m for _, m in rows[-window:]]


def apply_gemini_meta():
    """Gemini writes hook/title/description/hashtags (validated); queue text stays as fallback."""
    global GEMINI_META
    if os.environ.get("SKY_GEMINI_META", "1") == "0" or not gemini.available():
        return
    meta = sky_ai.write_meta(item, gemini.load_trends(REPO))
    if not meta:
        print("gemini meta: keeping queue hook/title")
        return
    trial = dict(item, hook=meta["hook"], title=meta["title"])
    if hook_text.check_item_hook_fields(trial):
        print("gemini meta failed English check; keeping queue text")
        return
    src_line = ""
    for ln in (item.get("description") or "").split("\n"):
        if ln.startswith("Source:"):
            src_line = ln
    if not src_line and item.get("source_url"):
        src_line = (f"Source: official trailer ({item['source_url']}). All rights to the film/series belong to "
                    f"their owners; short commentary edit.")
    tags = list(dict.fromkeys(meta["hashtags"] + [t for t in (item.get("tags") or [])[:2]]))
    desc = meta["description"].rstrip() + " Edit by Skywalker.\n" + src_line + "\n" + " ".join("#" + t for t in tags[:6])
    item["hook"], item["title"], item["description"] = meta["hook"], meta["title"], desc.strip()
    item["tags"] = list(dict.fromkeys((item.get("tags") or []) + meta["hashtags"]))[:12]
    GEMINI_META = meta
    print(f"gemini meta: hook={meta['hook']!r} title={meta['title']!r}")


def now():
    return datetime.now(IST).strftime("%Y-%m-%d %H:%M IST")

def read_prev():
    p = RES / f"{iid}.json"
    if p.exists():
        try:
            return json.loads(p.read_text())
        except Exception:
            return {}
    return {}

def write(status, detail, **extra):
    r = {"id": iid, "status": status, "detail": detail, "at": now(), "run_url": run_url,
         "slot": item.get("slot"), "title": item.get("title"), "hook": item.get("hook"),
         "description": item.get("description"), "tags": item.get("tags"),
         "queue_meta": QUEUE_META if GEMINI_META else None, "gemini_meta": bool(GEMINI_META),
         "gemini_calls": gemini._budget().get("log", []), **extra}
    prev = read_prev()
    for k in ("buffer_post_id", "buffer_status", "buffer_due_at", "buffer_posted_at", "youtube_url"):
        if prev.get(k) and k not in r:
            r[k] = prev[k]  # never lose the Buffer record (idempotency)
    (RES / f"{iid}.json").write_text(json.dumps(r, indent=2, ensure_ascii=False))
    print(json.dumps(r, indent=2, ensure_ascii=False))
    return r

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
    r = subprocess.run(["curl", "-fsSL", "--retry", "3", "-L", "-o", str(dest), url], capture_output=True, text=True)
    if r.returncode == 0 and _ok_src(dest):
        return True, url
    return False, f"direct mp4 {url}: curl exit {r.returncode} {(r.stderr or '')[-200:]}"

def _src_key(info: str) -> str:
    return (info or "").rstrip("/").split("/")[-1].split("?")[0].lower()


def _is_skipped(info, skip) -> bool:
    if not info:
        return False
    skip = set(skip or [])
    if info in skip:
        return True
    key = _src_key(info)
    return any(_src_key(s) == key and key for s in skip)


def download_fallback(src, skip=None):
    errors = []
    used = None
    skip = set(skip or [])
    for u in item.get("fallback_mp4", []) or []:
        if _is_skipped(u, skip) or _is_skipped(f"gh:{u}", skip):
            continue
        ok, info = download_url(u, src)
        if ok:
            return info, []
        errors.append(info)
    if item.get("fallback_release") and item.get("fallback_asset"):
        tag, asset = item["fallback_release"], item["fallback_asset"]
        repo = os.environ.get("GITHUB_REPOSITORY", "chatgpt820970/skywalker-shorts")
        r = subprocess.run(["gh", "release", "download", tag, "-p", asset, "-R", repo, "-D", str(OUT), "--clobber"],
                           capture_output=True, text=True)
        cand = OUT / asset
        if r.returncode == 0 and _ok_src(cand):
            if cand.resolve() != src.resolve():
                src.write_bytes(cand.read_bytes())
            return f"gh:release/{tag}/{asset}", []
        errors.append("gh release " + (r.stderr.strip().splitlines() or ["?"])[-1][:300])
    # last resort: sources release, guess asset from fallback_mp4 names or id
    repo = os.environ.get("GITHUB_REPOSITORY", "chatgpt820970/skywalker-shorts")
    guesses = []
    if item.get("fallback_asset"):
        guesses.append(item["fallback_asset"])
    for u in item.get("fallback_mp4") or []:
        guesses.append(u.rstrip("/").split("/")[-1])
    for g in guesses:
        r = subprocess.run(["gh", "release", "download", "sources", "-p", g, "-R", repo, "-D", str(OUT), "--clobber"],
                           capture_output=True, text=True)
        cand = OUT / g
        if r.returncode == 0 and _ok_src(cand):
            if cand.resolve() != src.resolve():
                src.write_bytes(cand.read_bytes())
            return f"gh:release/sources/{g}", []
        errors.append("gh sources " + g + " " + (r.stderr.strip().splitlines() or ["?"])[-1][:200])
    return None, errors


def _stem_tokens(name: str) -> set[str]:
    stem = (name or "").rsplit(".", 1)[0].lower()
    for junk in ("h264", "trailer", "teaser", "official", "src", "video", "4k"):
        stem = stem.replace(junk, " ")
    parts = [x for x in stem.replace("-", "_").replace(".", "_").split("_") if len(x) >= 3]
    return set(parts)


def list_source_siblings():
    """Other official trailers/teasers already uploaded to the sources release."""
    repo = os.environ.get("GITHUB_REPOSITORY", "chatgpt820970/skywalker-shorts")
    r = subprocess.run(
        ["gh", "release", "view", "sources", "-R", repo, "--json", "assets"],
        capture_output=True, text=True,
    )
    if r.returncode != 0:
        return []
    try:
        assets = [a.get("name") for a in (json.loads(r.stdout).get("assets") or []) if a.get("name")]
    except Exception:
        return []
    seeds = []
    if item.get("fallback_asset"):
        seeds.append(item["fallback_asset"])
    for u in item.get("fallback_mp4") or []:
        seeds.append(u.rstrip("/").split("/")[-1])
    tokens = set()
    for s in seeds:
        tokens |= _stem_tokens(s)
    for part in str(item.get("id") or "").lower().split("-"):
        if len(part) >= 4 and part.isalpha():
            tokens.add(part)
    already = set(seeds)
    out = []
    for name in assets:
        if name in already or not str(name).lower().endswith(".mp4"):
            continue
        if tokens & _stem_tokens(name):
            out.append(
                f"https://github.com/{repo}/releases/download/sources/{name}"
            )
    return out


def download_source(src, force_fallback=False, skip=None):
    errors, used = [], None
    skip = set(skip or [])
    if not force_fallback and item.get("source_url") and item.get("source_url") not in skip:
        for attempt in range(2):
            ua = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
            r = subprocess.run(["yt-dlp", "--no-playlist", "--remote-components", "ejs:github",
                                "-4", "--user-agent", ua,
                                "-f", "bv*[height<=1080][ext=mp4]+ba[ext=m4a]/b[ext=mp4]/bv*+ba/b",
                                "--merge-output-format", "mp4", "-o", str(src), "--force-overwrites",
                                item["source_url"]], capture_output=True, text=True)
            if r.returncode == 0 and _ok_src(src):
                return item["source_url"], []
            errors.append("yt-dlp: " + (r.stderr.strip().splitlines() or ["?"])[-1][:300])
            time.sleep(5)
    used, ferr = download_fallback(src, skip=skip)
    errors.extend(ferr)
    if used and not _is_skipped(used, skip):
        return used, errors
    # Other official trailers / teasers listed on the item, then sibling
    # assets already sitting on the sources release.
    extras = []
    for u in item.get("alt_source_urls") or []:
        extras.append(("alt", u))
    for u in list_source_siblings():
        extras.append(("sibling", u))
    for kind, u in extras:
        if not u or _is_skipped(u, skip) or _is_skipped(f"gh:{u}", skip):
            continue
        if "youtube.com" in u or "youtu.be" in u:
            src.unlink(missing_ok=True)
            ua = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
            r = subprocess.run(["yt-dlp", "--no-playlist", "--remote-components", "ejs:github",
                                "-4", "--user-agent", ua,
                                "-f", "bv*[height<=1080][ext=mp4]+ba[ext=m4a]/b[ext=mp4]/bv*+ba/b",
                                "--merge-output-format", "mp4", "-o", str(src), "--force-overwrites",
                                u], capture_output=True, text=True)
            if r.returncode == 0 and _ok_src(src):
                return u, errors
            errors.append("yt-dlp alt: " + (r.stderr.strip().splitlines() or ["?"])[-1][:300])
            continue
        ok, info = download_url(u, src)
        if ok:
            return info, errors
        errors.append(info)
    # Optional last-resort: official/legal Pexels clip if a key is already in the env.
    # Never required; never committed.
    pex = _try_pexels(src)
    if pex:
        return pex, errors
    return None, errors


def _try_pexels(dest):
    key = os.environ.get("PEXELS_API_KEY")
    q = (item.get("pexels_query") or "").strip()
    if not key or not q:
        return None
    try:
        import json as _json, urllib.parse, urllib.request
        req = urllib.request.Request(
            "https://api.pexels.com/videos/search?" + urllib.parse.urlencode({"query": q, "per_page": 5, "orientation": "landscape"}),
            headers={"Authorization": key},
        )
        with urllib.request.urlopen(req, timeout=12) as r:
            data = _json.loads(r.read().decode())
        files = []
        for v in data.get("videos") or []:
            for f in v.get("video_files") or []:
                if (f.get("file_type") or "").endswith("mp4") and (f.get("width") or 0) >= 1280:
                    files.append(f["link"])
        for url in files[:3]:
            ok, _ = download_url(url, dest)
            if ok:
                return f"pexels:{q}"
    except Exception as e:
        print("pexels skipped:", e)
    return None

def publish(final, fw, fh, up, used, qa, credit, qa_obj):
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
    return dict(
        release_url=f"https://github.com/{repo}/releases/tag/{tag}",
        asset_url=f"https://github.com/{repo}/releases/download/{tag}/{final.name}",
        qa=qa, drop_t=qa_obj.get("drop_t"), energy=qa_obj.get("energy"), source=used,
        music=credit.get("file"), music_credit=credit.get("credit"), description_final=desc,
        detail=f"rendered {fw}x{fh} ({up}), QA ok",
    )

def run_pipeline(src, out, qa_json, work, *, relax=False, alt_scenes=False, exclude_music=None, music_salt=""):
    cmd = [sys.executable, str(REPO / "pipeline.py"), "--src", str(src), "--out", str(out),
           "--template", item.get("template", "whitebar"), "--hook", item["hook"],
           "--pillar", item.get("pillar", "movie"), "--item-id", iid, "--music", "auto",
           "--target-dur", str(item.get("target_dur", 32)), "--qa-json", str(qa_json)]
    if item.get("words"):
        cmd += ["--words", item["words"]]
    out_res = int(item.get("out_res") or os.environ.get("SKY_OUT_RES") or 2160)
    if out_res in (1080, 2160):
        cmd += ["--out-res", str(out_res)]
    if relax:
        cmd += ["--relax"]
    if alt_scenes:
        cmd += ["--alt-scenes"]
    excl_all = list(dict.fromkeys(list(exclude_music or []) + list(RECENT_MUSIC or [])))
    if excl_all:
        cmd += ["--exclude-music", ",".join(excl_all)]
    if music_salt:
        cmd += ["--music-salt", music_salt]
    env = dict(os.environ, SKY_WORK=str(work), SKY_ROOT=str(OUT), SKY_MARKET=(item.get("market") or "").upper())
    if (item.get("market") or "").upper() == "IN" or item.get("dialogue_lang") in ("hi", "te", "ta"):
        # Hindi/South dubs: whisper-small output was garbled ("पेवरेट ग्रिकेट").
        # medium + forced language is far closer to the real lines (~3 min CPU).
        env["SKY_WHISPER_MODEL"] = item.get("whisper_model") or "medium"
        env["SKY_WHISPER_LANG"] = item.get("dialogue_lang") or "hi"
        env.setdefault("SKY_WHISPER_PROMPT", "")
    if not env.get("SKY_WHISPER_PROMPT") and "SKY_WHISPER_PROMPT" not in env:
        # Character/film names help whisper spell them right in the subtitles.
        names = item.get("whisper_prompt") or (
            (item.get("description") or "").split("\n")[0] + " " +
            " ".join(t for t in (item.get("tags") or []) if t not in ("edit", "shorts", "trending")))
        env["SKY_WHISPER_PROMPT"] = names.strip()[:600]
    # fresh work dir per attempt so reused scenes/edl cannot leak
    if work.exists():
        import shutil
        shutil.rmtree(work, ignore_errors=True)
    work.mkdir(parents=True, exist_ok=True)
    rc = subprocess.call(cmd, env=env)
    qa = json.loads(qa_json.read_text()) if qa_json.exists() else {}
    credit = {}
    cpath = work / "music_credit.json"
    if cpath.exists():
        credit = json.loads(cpath.read_text())
    checks = {k: v.get("ok") if isinstance(v, dict) else v for k, v in (qa.get("checks") or {}).items()}
    return rc, qa, credit, checks

def upscale(src, out_1080, final):
    sw, sh = probe_wh(src)
    out_res = int(item.get("out_res") or os.environ.get("SKY_OUT_RES") or 2160)
    if out_res == 1080:
        # Keep the 1080x1920 master as the published file (no 4K upscale).
        import shutil
        shutil.copy2(out_1080, final)
        fw, fh = probe_wh(final)
        return fw, fh, "1080x1920 (no upscale)"
    subprocess.check_call(["ffmpeg", "-y", "-i", str(out_1080), "-vf", "scale=2160:3840:flags=lanczos",
                           "-c:v", "libx264", "-preset", "medium", "-crf", "18", "-pix_fmt", "yuv420p",
                           "-c:a", "copy", "-movflags", "+faststart", str(final)])
    up = f"lanczos {sw}x{sh} source -> 2160x3840" if min(sw, sh) < 2160 else "4K source, scaled to 2160x3840"
    fw, fh = probe_wh(final)
    return fw, fh, up

def _is_short_failure(reason, qa) -> bool:
    blob = " ".join([
        str(reason or ""),
        " ".join(str(x) for x in (qa or {}).get("issues") or []),
        str((qa or {}).get("unique_duration") or ""),
    ]).lower()
    keys = (
        "hard minimum", "unique clean shots", "aim 25-32", "below target",
        "too short", "short unique", "not publishing a short cut",
        "duration ", "need ≥20", "need >=20", "need ≥25", "need >=25",
    )
    return any(k in blob for k in keys)


def attempt(n, *, force_fallback=False, relax=False, alt_scenes=False, exclude_music=None, music_salt=""):
    src = OUT / f"{iid}_src.mp4"
    print(f"=== attempt {n}: force_fallback={force_fallback} relax={relax} alt_scenes={alt_scenes} ===", flush=True)
    tried = []
    last = None
    src_i = 0
    while True:
        used, errors = download_source(src, force_fallback=force_fallback, skip=tried)
        if not used:
            if last:
                return False, last
            reason = "download failed: " + " | ".join(errors)
            return False, {"attempt": n, "ok": False, "error": reason, "source": None, "music": None}
        if _is_skipped(used, tried):
            if last:
                return False, last
            return False, {"attempt": n, "ok": False, "error": "no unused official source left",
                           "source": used, "music": None}
        tried.append(used)
        src_i += 1
        print(f"  source {src_i}: {used}", flush=True)

        out = OUT / f"{iid}_1080.mp4"
        qa_json = OUT / f"{iid}_qa.json"
        work = OUT / f"work_{iid}"
        suffix = n if src_i == 1 else f"{n}s{src_i}"
        if n > 1 or src_i > 1:
            out = OUT / f"{iid}_1080_a{suffix}.mp4"
            qa_json = OUT / f"{iid}_qa_a{suffix}.json"
            work = OUT / f"work_{iid}_a{suffix}"
        rc, qa, credit, checks = run_pipeline(
            src, out, qa_json, work,
            relax=relax, alt_scenes=alt_scenes,
            exclude_music=exclude_music, music_salt=music_salt,
        )
        if rc != 0 or not out.exists():
            reason = f"pipeline/QA failed (exit {rc}): " + "; ".join((qa.get("issues") or [])[:5])[:800]
            if not reason.endswith(")") and not qa.get("issues"):
                reason = f"pipeline/QA failed (exit {rc})"
            last = {"attempt": n, "ok": False, "error": reason, "source": used,
                    "music": credit.get("file"), "qa": checks}
            if _is_short_failure(reason, qa):
                print(f"  source too short / under duration gate; trying next official trailer", flush=True)
                continue
            return False, last
        gem_qc = None
        if gemini.available() and gemini.budget_left() > 0:
            try:
                gem_qc = sky_ai.qc_judge(out, work, item.get("hook", ""), item.get("title", ""),
                                         item.get("description", ""))
            except Exception as e:
                print("gemini QC skipped:", e)
        if gem_qc is not None:
            print("gemini QC:", json.dumps(gem_qc, ensure_ascii=False))
            if not gem_qc.get("pass"):
                return False, {"attempt": n, "ok": False, "source": used, "music": credit.get("file"), "qa": checks,
                               "error": "gemini QC reject: " + "; ".join(gem_qc.get("reasons") or gem_qc.get("low") or ["?"])[:600]}
        try:
            final = OUT / f"render-{iid}.mp4"
            fw, fh, up = upscale(src, out, final)
            g = gate.run_gate(final, qa_obj=qa, checks=checks, item=item, work=work, gemini_qc=gem_qc, out_res=OUT_RES)
            print("publish gate:", json.dumps(g, ensure_ascii=False, default=str))
            if not g["ok"]:
                return False, {"attempt": n, "ok": False, "source": used, "music": credit.get("file"), "qa": checks,
                               "gate": g, "error": "publish gate failed: " + "; ".join(g["failures"])[:600]}
            extra = publish(final, fw, fh, up, used, checks, credit, qa)
            extra["gate"] = g
        except Exception as e:
            return False, {"attempt": n, "ok": False, "error": f"upscale/release failed: {e}",
                           "source": used, "music": credit.get("file"), "qa": checks}
        extra.update({"attempt": n, "ok": True, "error": None})
        return True, extra

RECENT_MUSIC = []
JOB_T0 = time.monotonic()
# render.yml job cap is 120 min; stop starting new attempts well before it so the
# result json is always written (no more silent cancel at the job timeout).
JOB_BUDGET_S = float(os.environ.get("SKY_JOB_BUDGET_MIN", "95")) * 60.0


def _budget_left_for(next_est_s: float) -> bool:
    used = time.monotonic() - JOB_T0
    ok = used + next_est_s * 1.1 <= JOB_BUDGET_S
    if not ok:
        print(f"::warning::time budget: {used/60:.1f} min used, next attempt ~{next_est_s/60:.1f} min "
              f"would pass the {JOB_BUDGET_S/60:.0f} min budget; skipping further attempts", flush=True)
    return ok


def main():
    prev = read_prev()
    auto_retries = int(prev.get("auto_retries") or 0)
    # This job IS the outer auto-retry if the last result asked for one.
    if prev.get("status") == "retry_queued":
        auto_retries += 1
    prior_music = []
    if prev.get("music"):
        prior_music.append(prev["music"])
    for a in prev.get("attempts") or []:
        if a.get("music"):
            prior_music.append(a["music"])

    fetch_music_library()
    recent = recent_music()
    if recent:
        print("music: skipping songs used in the last renders:", recent)
    apply_gemini_meta()
    global RECENT_MUSIC
    RECENT_MUSIC = recent

    attempts_log = []
    # Attempt 1: normal
    t_a = time.monotonic()
    ok, info = attempt(1, force_fallback=False)
    est = time.monotonic() - t_a
    attempts_log.append({k: info.get(k) for k in ("attempt", "ok", "error", "source", "music")})
    if ok:
        write("done", info["detail"], auto_retries=auto_retries, max_auto_retries=MAX_AUTO_RETRIES,
              attempts=attempts_log, last_error=None, **{k: v for k, v in info.items()
                if k in ("release_url", "asset_url", "qa", "drop_t", "energy", "source",
                         "music", "music_credit", "description_final", "gate")})
        return 0

    # Attempt 2: sources-release fallback mp4
    if not _budget_left_for(est):
        return _give_up(attempts_log, auto_retries)
    t_a = time.monotonic()
    ok, info = attempt(2, force_fallback=True)
    est = max(est, time.monotonic() - t_a)
    attempts_log.append({k: info.get(k) for k in ("attempt", "ok", "error", "source", "music")})
    if ok:
        write("done", info["detail"] + " (retry: sources fallback)", auto_retries=auto_retries,
              max_auto_retries=MAX_AUTO_RETRIES, attempts=attempts_log, last_error=None,
              **{k: v for k, v in info.items()
                 if k in ("release_url", "asset_url", "qa", "drop_t", "energy", "source",
                          "music", "music_credit", "description_final", "gate")})
        return 0

    # Attempt 3: fallback + different music + alt scenes + relaxed non-hard thresholds
    if not _budget_left_for(est):
        return _give_up(attempts_log, auto_retries)
    excl = list(dict.fromkeys([m for m in prior_music + [attempts_log[0].get("music"), attempts_log[1].get("music")] if m]))
    ok, info = attempt(3, force_fallback=True, relax=True, alt_scenes=True,
                       exclude_music=excl, music_salt=f"retry3-{auto_retries}")
    attempts_log.append({k: info.get(k) for k in ("attempt", "ok", "error", "source", "music")})
    if ok:
        write("done", info["detail"] + " (retry: alt music/scenes, non-hard relax)", auto_retries=auto_retries,
              max_auto_retries=MAX_AUTO_RETRIES, attempts=attempts_log, last_error=None,
              **{k: v for k, v in info.items()
                 if k in ("release_url", "asset_url", "qa", "drop_t", "energy", "source",
                          "music", "music_credit", "description_final", "gate")})
        return 0

    return _give_up(attempts_log, auto_retries)


def _give_up(attempts_log, auto_retries):
    last = attempts_log[-1].get("error") or "unknown failure"
    # All 3 in-job attempts failed. If we still have an outer auto-retry, queue it
    # and exit 0 so this run does not wake BIBI yet.
    if auto_retries < MAX_AUTO_RETRIES:
        write("retry_queued",
              f"all {len(attempts_log)} in-job attempt(s) failed; auto-retry {auto_retries + 1}/{MAX_AUTO_RETRIES} will be dispatched: {last}"[:800],
              auto_retries=auto_retries, max_auto_retries=MAX_AUTO_RETRIES,
              attempts=attempts_log, last_error=last)
        print(f"::warning::{iid}: queued outer auto-retry ({auto_retries + 1}/{MAX_AUTO_RETRIES})")
        return 0

    write("failed", last[:800], auto_retries=auto_retries, max_auto_retries=MAX_AUTO_RETRIES,
          attempts=attempts_log, last_error=last)
    print(f"::error::{iid}: {last[:300]}")
    return 1

if __name__ == "__main__":
    sys.exit(main())
