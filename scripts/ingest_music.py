#!/usr/bin/env python3
"""Fetch, loudnorm, analyse, and publish CC/public-domain edit songs into the 'music' release.

Runs on GitHub Actions. Never rips commercial YouTube/Spotify/SoundCloud tracks.
Internet Archive items must carry a CC0 / CC-BY / CC-BY-SA or public-domain license
(no NC, no ND). New mp3s are uploaded to the existing 'music' release; metadata is
merged into music/library.json. Existing tracks are never deleted.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
import time
import urllib.parse
import urllib.request
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
CFG_PATH = REPO / "config" / "music_sources.json"
LIB_PATH = REPO / "music" / "library.json"
LIB_DIR = REPO / "music" / "lib"
UA = "SkywalkerShortsMusicIngest/1.0 (CC ingest; +https://github.com/chatgpt820970/skywalker-shorts)"
IA_SEARCH = "https://archive.org/advancedsearch.php"
IA_META = "https://archive.org/metadata/{id}"
IA_DL = "https://archive.org/download/{id}/{name}"

# CC licenses that allow a public YouTube edit. Skip NC (non-commercial) and ND (no derivatives).
_OK_LICENSE = (
    "creativecommons.org/publicdomain",
    "creativecommons.org/licenses/cc0",
    "creativecommons.org/licenses/by/",
    "creativecommons.org/licenses/by-sa/",
    "creativecommons.org/licenses/zero",
    "creativecommons.org/publicdomain/zero",
    "creativecommons.org/publicdomain/mark",
)
_BAD_LICENSE = ("/by-nc", "/by-nd", "/by-nc-sa", "/by-nc-nd", "/by-nd", "nc-sa", "nc-nd")


def license_ok(url: str | None) -> bool:
    u = (url or "").strip().lower()
    if not u:
        return False
    if any(b in u for b in _BAD_LICENSE):
        return False
    return any(ok in u for ok in _OK_LICENSE)


def slugify(text: str, fallback: str = "track") -> str:
    s = re.sub(r"[^a-z0-9]+", "", (text or "").lower())
    s = (s or re.sub(r"[^a-z0-9]+", "", (fallback or "").lower()) or "track")[:24]
    return f"sw_{s}.mp3"


def tags_from_text(*parts: str) -> tuple[str, str, list[str], list[str]]:
    """Heuristic genre, mood, pillars, tags from title/creator/description."""
    blob = " ".join(parts).lower()
    genre, mood = "electronic", "hype"
    pillars = ["movie", "anime"]
    tags = ["edit", "electronic"]
    rules = [
        (("phonk", "drift"), "phonk", "dark-hype", ["anime", "movie", "celeb"], ["phonk", "edit", "trending"]),
        (("funk", "baile", "brazilian"), "brazilian funk", "fun-hype", ["funny", "anime", "celeb"], ["funk", "edit"]),
        (("synthwave", "outrun", "retrowave"), "synthwave", "dark", ["movie", "anime"], ["synthwave", "edit"]),
        (("cyberpunk", "industrial", "dark electronic"), "dark electronic", "dark", ["movie", "anime"], ["dark", "edit"]),
        (("trailer", "cinematic", "epic", "orchestral"), "cinematic", "epic", ["movie"], ["cinematic", "epic", "edit"]),
        (("drum and bass", "dnb", "dnB", "liquid dnb"), "drum and bass", "hype", ["anime", "movie"], ["dnb", "edit"]),
        (("dubstep", "riddim", "brostep"), "dubstep", "hype", ["anime", "celeb"], ["dubstep", "edit"]),
        (("hardstyle", "hardcore", "gabber"), "hardstyle", "hype", ["anime", "celeb"], ["hardstyle", "edit"]),
        (("techno", "industrial techno"), "techno", "dark", ["movie", "anime"], ["techno", "edit"]),
        (("breakbeat", "break"), "breakbeat", "hype", ["anime", "movie"], ["breakbeat", "edit"]),
        (("romance", "love", "emotional", "sad"), "romance", "emotional", ["romance"], ["romance", "edit"]),
    ]
    for keys, g, m, p, t in rules:
        if any(k.lower() in blob for k in keys):
            genre, mood, pillars, tags = g, m, p, t
            break
    return genre, mood, pillars, tags


def _urlopen(url: str, timeout: int = 60):
    req = urllib.request.Request(url, headers={"User-Agent": UA})
    return urllib.request.urlopen(req, timeout=timeout)


def ia_search(query: str, rows: int = 80) -> list[dict]:
    qs = urllib.parse.urlencode({
        "q": query,
        "fl[]": ["identifier", "title", "creator", "licenseurl", "mediatype"],
        "rows": str(rows),
        "output": "json",
        "sort[]": "downloads desc",
    }, doseq=True)
    # fl[] repeated: urlencode doseq with a list works. Also add mediatype filter already in query.
    try:
        with _urlopen(f"{IA_SEARCH}?{qs}", timeout=45) as r:
            data = json.loads(r.read().decode())
        return list((data.get("response") or {}).get("docs") or [])
    except Exception as e:
        print(f"  ia search failed: {type(e).__name__}: {e}")
        return []


def ia_audio_files(ident: str) -> list[dict]:
    try:
        with _urlopen(IA_META.format(id=urllib.parse.quote(ident)), timeout=45) as r:
            meta = json.loads(r.read().decode())
    except Exception as e:
        print(f"  ia metadata {ident}: {type(e).__name__}")
        return []
    files = meta.get("files") or []
    out = []
    for f in files:
        name = f.get("name") or ""
        low = name.lower()
        if not any(low.endswith(ext) for ext in (".mp3", ".ogg", ".flac", ".wav", ".m4a")):
            continue
        # skip tiny previews / deriv thumbs
        try:
            size = int(f.get("size") or 0)
        except Exception:
            size = 0
        if size and size < 80_000:
            continue
        out.append({"name": name, "size": size, "format": f.get("format") or ""})
    # Prefer mp3, then shorter/smaller mid-size files (one or two per item)
    pref = {".mp3": 0, ".m4a": 1, ".ogg": 2, ".wav": 3, ".flac": 4}
    out.sort(key=lambda f: (pref.get(Path(f["name"]).suffix.lower(), 9), f.get("size") or 10**12))
    return out[:2]


def download(url: str, dest: Path, timeout: int = 120) -> bool:
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    try:
        with _urlopen(url, timeout=timeout) as r, tmp.open("wb") as out:
            while True:
                chunk = r.read(1 << 16)
                if not chunk:
                    break
                out.write(chunk)
        if tmp.stat().st_size < 80_000:
            tmp.unlink(missing_ok=True)
            return False
        tmp.replace(dest)
        return True
    except Exception as e:
        print(f"  download failed: {type(e).__name__}")
        tmp.unlink(missing_ok=True)
        return False


def run_ffmpeg(args: list[str]) -> bool:
    r = subprocess.run(["ffmpeg", "-y", "-hide_banner", "-loglevel", "error", *args],
                       capture_output=True, text=True)
    if r.returncode != 0:
        err = (r.stderr or r.stdout or "")[-300:]
        print(f"  ffmpeg failed: {err}")
        return False
    return True


def loudnorm_trim(src: Path, dst: Path, drop_t: float | None, max_keep: float, pre_drop: float) -> bool:
    """Loudnorm to -11 LUFS and optionally keep a window around the drop."""
    dur = _ffprobe_dur(src)
    af = "loudnorm=I=-11:TP=-1.5:LRA=8,aresample=44100"
    cmd = ["-i", str(src)]
    if dur and dur > max_keep + 2:
        start = 0.0
        if drop_t is not None:
            start = max(0.0, float(drop_t) - pre_drop)
            if start + max_keep > dur:
                start = max(0.0, dur - max_keep)
        cmd = ["-ss", f"{start:.3f}", "-t", f"{max_keep:.3f}", "-i", str(src)]
    cmd += ["-af", af, "-c:a", "libmp3lame", "-b:a", "192k", str(dst)]
    return run_ffmpeg(cmd)


def _ffprobe_dur(path: Path) -> float:
    try:
        o = subprocess.check_output(
            ["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", str(path)],
            text=True,
        ).strip()
        return float(o or 0)
    except Exception:
        return 0.0


def analyze(path: Path) -> dict | None:
    """BPM, drop, energy, duration. Librosa is imported lazily so helpers stay testable."""
    try:
        import numpy as np
        import librosa
    except Exception as e:
        print(f"  analyze skipped (librosa): {e}")
        return None
    try:
        y, sr = librosa.load(str(path), sr=22050, mono=True)
    except Exception as e:
        print(f"  librosa load failed: {e}")
        return None
    dur = float(len(y) / sr) if sr else 0.0
    if dur < 8:
        return None
    tempo, beats = librosa.beat.beat_track(y=y, sr=sr, units="time")
    tempo = float(tempo[0] if hasattr(tempo, "__len__") else tempo)
    pulse = beats
    if tempo >= 140 and len(beats) > 8:
        pulse = beats[::2]
        tempo = tempo / 2.0
    hop = 512
    rms = librosa.feature.rms(y=y, hop_length=hop)[0]
    energy = float(np.clip(np.mean(rms) / (np.percentile(rms, 90) + 1e-9), 0, 1))
    drop_t = _find_drop(y, sr, pulse) if len(pulse) else None
    if drop_t is None:
        # loudest 2s window
        tt = librosa.frames_to_time(np.arange(len(rms)), sr=sr, hop_length=hop)
        best, best_t = -1.0, 0.0
        for t0 in tt[::4]:
            m = float(rms[(tt >= t0) & (tt < t0 + 2)].mean()) if np.any((tt >= t0) & (tt < t0 + 2)) else 0
            if m > best:
                best, best_t = m, float(t0)
        drop_t = best_t
    return {
        "bpm": round(float(tempo), 1),
        "drop_t": round(float(drop_t or 0.0), 2),
        "duration_s": round(dur, 1),
        "energy": round(float(energy), 3),
    }


def _find_drop(y, sr, pulse):
    import numpy as np
    import librosa
    hop = 512
    rms = librosa.feature.rms(y=y, hop_length=hop)[0]
    onset = librosa.onset.onset_strength(y=y, sr=sr, hop_length=hop)
    n = min(len(rms), len(onset))
    e = rms[:n] / (rms[:n].max() + 1e-9) + 0.25 * onset[:n] / (onset[:n].max() + 1e-9)
    tt = librosa.frames_to_time(np.arange(n), sr=sr, hop_length=hop)
    k = max(1, int(1.5 * sr / hop))
    best, best_t = -1e9, None
    lo, hi = 1.0, max(2.0, (len(y) / sr) * 0.75)
    for b in pulse:
        if b < lo or b > hi:
            continue
        i = int(np.searchsorted(tt, b))
        if i - k < 0 or i + k >= n:
            continue
        jump = float(e[i:i + k].mean() - e[i - k:i].mean())
        if jump > best:
            best, best_t = jump, float(b)
    return best_t


def load_lib() -> dict:
    if LIB_PATH.exists():
        return json.loads(LIB_PATH.read_text())
    return {"note": "", "release": "music", "dir": "music/lib", "no_repeat_window": 8, "tracks": []}


def save_lib(lib: dict) -> None:
    LIB_PATH.write_text(json.dumps(lib, indent=2, ensure_ascii=False) + "\n")


def existing_files(lib: dict) -> set[str]:
    return {t.get("file") for t in (lib.get("tracks") or []) if t.get("file")}


def existing_idents(lib: dict) -> set[str]:
    return {t.get("source_id") for t in (lib.get("tracks") or []) if t.get("source_id")}


def unique_filename(name: str, taken: set[str]) -> str:
    if name not in taken:
        return name
    stem = Path(name).stem
    for i in range(2, 80):
        cand = f"{stem}{i}.mp3"
        if cand not in taken:
            return cand
    h = hashlib.sha1(name.encode()).hexdigest()[:6]
    return f"{stem}_{h}.mp3"


def upload_release(files: list[Path], tag: str = "music") -> bool:
    if not files:
        return True
    repo = os.environ.get("GITHUB_REPOSITORY", "chatgpt820970/skywalker-shorts")
    # ensure release exists
    r = subprocess.run(["gh", "release", "view", tag, "-R", repo], capture_output=True, text=True)
    if r.returncode != 0:
        subprocess.check_call(["gh", "release", "create", tag, "-R", repo, "--title", "music",
                               "--notes", "Skywalker Shorts edit-song library (CC / credited tracks)."])
    args = ["gh", "release", "upload", tag, "-R", repo, "--clobber"]
    args += [str(p) for p in files]
    r = subprocess.run(args, capture_output=True, text=True)
    if r.returncode != 0:
        print("  release upload failed:", (r.stderr or r.stdout or "")[-400:])
        return False
    print(f"  uploaded {len(files)} file(s) to release {tag}")
    return True


def process_candidate(item: dict, fileinfo: dict, lib: dict, cfg: dict, taken: set[str]) -> dict | None:
    ident = item.get("identifier") or ""
    title = (item.get("title") or ident or "track").strip()
    artist = (item.get("creator") or "Unknown").strip()
    if isinstance(artist, list):
        artist = ", ".join(str(a) for a in artist[:3])
    license_url = item.get("licenseurl") or ""
    if not license_ok(license_url):
        return None
    fname = unique_filename(slugify(title, ident), taken)
    raw = LIB_DIR / f"_raw_{fname}"
    url = IA_DL.format(id=urllib.parse.quote(ident), name=urllib.parse.quote(fileinfo["name"]))
    if not download(url, raw):
        return None
    dur = _ffprobe_dur(raw)
    if dur < float(cfg.get("min_duration_s") or 25):
        raw.unlink(missing_ok=True)
        return None
    # first-pass analyse on the raw file so we can trim around the drop
    pre = analyze(raw) or {}
    out = LIB_DIR / fname
    ok = loudnorm_trim(
        raw, out,
        drop_t=pre.get("drop_t"),
        max_keep=float(cfg.get("max_keep_s") or 80),
        pre_drop=float(cfg.get("pre_drop_s") or 12),
    )
    raw.unlink(missing_ok=True)
    if not ok or not out.exists() or out.stat().st_size < 80_000:
        out.unlink(missing_ok=True)
        return None
    meta = analyze(out)
    if not meta or meta["duration_s"] < float(cfg.get("min_duration_s") or 25):
        out.unlink(missing_ok=True)
        return None
    genre, mood, pillars, tags = tags_from_text(title, artist, ident)
    track = {
        "file": fname,
        "title": title[:120],
        "artist": str(artist)[:80],
        "genre": genre,
        "mood": mood,
        "pillars": pillars,
        "bpm": meta["bpm"],
        "drop_t": meta["drop_t"],
        "duration_s": meta["duration_s"],
        "energy": meta["energy"],
        "tags": tags,
        "credit": f"Song: {artist} - {title} ({license_url})",
        "license": license_url,
        "source": "archive.org",
        "source_id": ident,
        "source_file": fileinfo["name"],
    }
    taken.add(fname)
    return track


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-new", type=int, default=0, help="Cap new tracks this run (0 = config default)")
    ap.add_argument("--target", type=int, default=0)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    cfg = json.loads(CFG_PATH.read_text()) if CFG_PATH.exists() else {}
    max_new = args.max_new or int(cfg.get("max_new_per_run") or 120)
    target = args.target or int(cfg.get("target_count") or 250)
    LIB_DIR.mkdir(parents=True, exist_ok=True)
    lib = load_lib()
    lib.setdefault("release", "music")
    lib.setdefault("dir", "music/lib")
    lib.setdefault("no_repeat_window", int(cfg.get("no_repeat_window") or 8))
    tracks = list(lib.get("tracks") or [])
    taken = existing_files(lib)
    seen_ids = existing_idents(lib)
    already = len(tracks)
    print(f"library has {already} track(s); target {target}; max new this run {max_new}")
    if already >= target:
        print("target already met; nothing to do")
        return 0

    docs = []
    seen_doc = set()
    for q in cfg.get("queries") or []:
        print(f"search: {q[:80]}…")
        for d in ia_search(q, rows=80):
            ident = d.get("identifier")
            if not ident or ident in seen_doc or ident in seen_ids:
                continue
            if not license_ok(d.get("licenseurl")):
                continue
            seen_doc.add(ident)
            docs.append(d)
        time.sleep(1.0)
    print(f"  {len(docs)} licensed IA items to consider")

    added, uploaded = [], []
    for d in docs:
        if len(tracks) >= target or len(added) >= max_new:
            break
        ident = d.get("identifier")
        files = ia_audio_files(ident)
        time.sleep(0.4)
        if not files:
            continue
        for fi in files:
            if len(tracks) >= target or len(added) >= max_new:
                break
            print(f"  try {ident} / {fi['name']}")
            if args.dry_run:
                continue
            tr = process_candidate(d, fi, lib, cfg, taken)
            time.sleep(0.6)
            if not tr:
                continue
            tracks.append(tr)
            added.append(tr)
            uploaded.append(LIB_DIR / tr["file"])
            print(f"    + {tr['file']}  {tr['bpm']}bpm drop={tr['drop_t']}s  {tr['duration_s']}s  {tr['genre']}")

    lib["tracks"] = tracks
    if added and not args.dry_run:
        if not upload_release(uploaded, lib.get("release") or "music"):
            print("upload failed; library.json not updated for this run")
            return 1
        save_lib(lib)
    print(f"DONE added={len(added)} total={len(tracks)} target={target}")
    summary = REPO / "music" / "ingest_summary.json"
    summary.write_text(json.dumps({
        "added": len(added), "total": len(tracks), "target": target,
        "files": [t["file"] for t in added],
        "skipped_youtube_rips": True,
    }, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main() or 0)
