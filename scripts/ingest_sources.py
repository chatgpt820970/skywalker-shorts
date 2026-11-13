#!/usr/bin/env python3
"""Download official free trailers/clips and upload to the sources GitHub release.

Runs on GitHub Actions (source-ingest.yml). Prefers Archive.org and non-YouTube
hosts; YouTube only last (optional download-only cookies). No DRM / paid OTT.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
import urllib.parse
import urllib.request
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
import sourcing  # noqa: E402

UA = {"User-Agent": "SkywalkerSourceIngest/1.0"}
GH = "https://github.com/chatgpt820970/skywalker-shorts/releases/download/sources"
OUT = REPO / "out" / "sources_ingest"
OUT.mkdir(parents=True, exist_ok=True)


def ia_search(q: str, rows: int = 8) -> list[dict]:
    qs = urllib.parse.urlencode({
        "q": q + " AND mediatype:(movies)",
        "fl[]": ["identifier", "title", "downloads"],
        "rows": str(rows),
        "output": "json",
        "sort[]": "downloads desc",
    }, doseq=True)
    try:
        with urllib.request.urlopen(
            urllib.request.Request("https://archive.org/advancedsearch.php?" + qs, headers=UA),
            timeout=30,
        ) as r:
            return json.loads(r.read())["response"]["docs"]
    except Exception:
        return []


def ia_pick_mp4(ident: str) -> tuple[str, str] | None:
    try:
        with urllib.request.urlopen(
            urllib.request.Request(f"https://archive.org/metadata/{ident}", headers=UA),
            timeout=40,
        ) as r:
            meta = json.loads(r.read())
    except Exception:
        return None
    cands = []
    for f in meta.get("files") or []:
        name = f.get("name") or ""
        low = name.lower()
        if not any(low.endswith(e) for e in (".mp4", ".mkv", ".mov", ".webm")):
            continue
        if "thumb" in low or "sample" in low:
            continue
        try:
            size = int(f.get("size") or 0)
        except Exception:
            size = 0
        if size and (size < 500_000 or size > 600_000_000):
            continue
        if size > 250_000_000 and "trailer" not in low and "teaser" not in low:
            continue
        pref = 0 if ("trailer" in low or "teaser" in low) else 1
        cands.append((pref, 0 if low.endswith(".mp4") else 1, -(size or 0), name))
    if not cands:
        return None
    cands.sort()
    return ident, cands[0][3]


def download_ia(ident: str, name: str, dest: Path) -> bool:
    url = f"https://archive.org/download/{urllib.parse.quote(ident)}/{urllib.parse.quote(name)}"
    try:
        with urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=300) as r, dest.open("wb") as out:
            while True:
                chunk = r.read(1 << 20)
                if not chunk:
                    break
                out.write(chunk)
        return dest.exists() and dest.stat().st_size > 500_000
    except Exception as e:
        print("ia download failed:", type(e).__name__, e)
        return False


def to_h264(src: Path, dst: Path) -> bool:
    r = subprocess.run([
        "ffmpeg", "-y", "-hide_banner", "-loglevel", "error", "-i", str(src),
        "-vf", r"scale=min(1920\,iw):-2", "-c:v", "libx264", "-preset", "veryfast",
        "-crf", "20", "-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", "160k",
        "-movflags", "+faststart", str(dst),
    ], capture_output=True, text=True)
    return r.returncode == 0 and dst.exists() and dst.stat().st_size > 500_000


def upload(files: list[Path]) -> bool:
    if not files:
        return True
    repo = os.environ.get("GITHUB_REPOSITORY", "chatgpt820970/skywalker-shorts")
    r = subprocess.run(["gh", "release", "view", "sources", "-R", repo], capture_output=True)
    if r.returncode != 0:
        subprocess.check_call([
            "gh", "release", "create", "sources", "-R", repo,
            "--title", "sources", "--notes", "Official free trailers/clips for Skywalker Shorts",
        ])
    args = ["gh", "release", "upload", "sources", "-R", repo, "--clobber"] + [str(p) for p in files]
    r = subprocess.run(args, capture_output=True, text=True)
    if r.returncode != 0:
        print("upload failed:", (r.stderr or "")[-400:])
        return False
    return True


def release_has(asset: str) -> bool:
    repo = os.environ.get("GITHUB_REPOSITORY", "chatgpt820970/skywalker-shorts")
    r = subprocess.run(
        ["gh", "release", "view", "sources", "-R", repo, "--json", "assets"],
        capture_output=True, text=True,
    )
    if r.returncode != 0:
        return False
    try:
        names = {a.get("name") for a in json.loads(r.stdout).get("assets") or []}
    except Exception:
        return False
    return asset in names


def ingest_topic(slug: str) -> dict:
    asset = f"{slug}_trailer_h264.mp4"
    result = {"slug": slug, "asset": asset, "ok": False, "via": None}
    if release_has(asset):
        result.update(ok=True, via="already-on-release")
        return result
    cat = sourcing.load_catalog().get("topics") or {}
    row = cat.get(slug) or {}
    film = slug.replace("-", " ")
    # 1) Archive.org search
    for q in (f"{film} official trailer", f"{film} trailer"):
        for doc in ia_search(q):
            ident = doc.get("identifier") or ""
            blob = (ident + " " + (doc.get("title") or "")).lower()
            if "trailer" not in blob and "teaser" not in blob:
                continue
            picked = ia_pick_mp4(ident)
            if not picked:
                continue
            raw = OUT / f"_raw_{slug}{Path(picked[1]).suffix}"
            if download_ia(picked[0], picked[1], raw):
                dest = OUT / asset
                if to_h264(raw, dest):
                    if upload([dest]):
                        result.update(ok=True, via=f"archive:{picked[0]}")
                        return result
    # 2) Catalog / platform URLs via sourcing.download_source
    item = {
        "id": f"ingest-{slug}",
        "words": film.upper(),
        "source_url": (row.get("youtube_urls") or [None])[0],
        "alt_source_urls": list(row.get("platform_urls") or []) + list(row.get("youtube_urls") or []),
        "content_kind": row.get("content_kind") or "trailer",
    }
    dest = OUT / asset
    used, errs = sourcing.download_source(item, dest, allow_youtube=True)
    if used and dest.exists() and dest.stat().st_size > 500_000:
        # re-encode for consistency
        tmp = OUT / f"_enc_{asset}"
        if to_h264(dest, tmp):
            tmp.replace(dest)
        if upload([dest]):
            result.update(ok=True, via=used)
            return result
    result["errors"] = errs[-5:]
    return result


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--topics", default="")
    args = ap.parse_args()
    topics = [t.strip() for t in (args.topics or "").split(",") if t.strip()]
    if not topics:
        # gaps: catalog topics without a sources asset
        cat = sourcing.load_catalog().get("topics") or {}
        topics = [s for s in cat if not release_has(f"{s}_trailer_h264.mp4")]
    print("topics:", topics)
    rows = [ingest_topic(t) for t in topics]
    summary = REPO / "out" / "source_ingest_summary.json"
    summary.write_text(json.dumps({"results": rows}, indent=2) + "\n")
    print(json.dumps(rows, indent=2)[:2000])
    return 0 if any(r.get("ok") for r in rows) or not topics else 1


if __name__ == "__main__":
    raise SystemExit(main())
