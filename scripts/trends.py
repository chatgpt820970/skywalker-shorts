#!/usr/bin/env python3
"""Every 3 days: what is working for fast-growing movie/trailer edit Shorts channels -> trends/latest.json.

Cheap: 2 Gemini calls (1 Google-Search-grounded research pass + 1 JSON synthesis with our own stats).
Inputs reachable from a runner: Gemini grounded search, our stats/latest.json, our results (hooks,
songs), plus a best-effort yt-dlp search of trending edit Shorts (often bot-blocked; tolerated).
The render step reads trends/latest.json (hook/title style + song_picks). Slot times are never
changed automatically; posting_times_ist is advisory for the owner.
"""
from __future__ import annotations

import glob, json, os, re, subprocess, sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
import gemini  # noqa: E402

IST = timezone(timedelta(hours=5, minutes=30))


def yt_search_sample():
    out = []
    for q in ("bollywood movie edit shorts", "south movie edit shorts", "hollywood movie edit shorts phonk"):
        try:
            r = subprocess.run(["yt-dlp", "--flat-playlist", "-J", "--no-warnings", f"ytsearch8:{q}"],
                               capture_output=True, text=True, timeout=45)
            if r.returncode == 0:
                for e in json.loads(r.stdout).get("entries") or []:
                    out.append({"q": q, "title": e.get("title"), "channel": e.get("channel"),
                                "views": e.get("view_count"), "duration": e.get("duration")})
        except Exception:
            pass
    return out[:24]


def _slot_hhmm(slot):
    text = str(slot or "")
    if "T" in text:
        return text.split("T", 1)[1][:5]
    return text or None


def _music_label(result: dict) -> str | None:
    credit = (result.get("music_credit") or "").splitlines()
    song = ""
    if credit and credit[0].lower().startswith("song:"):
        song = credit[0].split(":", 1)[1].strip()
    file = result.get("music") or ""
    if song and file:
        return f"{song} ({file})"
    return song or file or None


def _title_key(title: str | None) -> str:
    """Match stats titles to results even when the pipe/hashtag spacing differs."""
    text = " ".join((title or "").split()).casefold()
    text = re.sub(r"\s*\|\s*", " | ", text)
    text = re.sub(r"\s+#", " #", text)
    return text


def _results_by_title() -> dict:
    """Local results only. Prefer a finished render when titles collide."""
    by_title = {}
    for path in (REPO / "results").glob("*.json"):
        try:
            row = json.loads(path.read_text())
        except Exception:
            continue
        key = _title_key(row.get("title"))
        if not key:
            continue
        prev = by_title.get(key)
        if prev and prev.get("status") == "done" and row.get("status") != "done":
            continue
        by_title[key] = row
    return by_title


def own_performance():
    """Stats rows plus music / energy / drop / slot from results when stats omit them.

    No network and no Gemini. The next trends synthesis can see which songs and
    slots actually earned the views.
    """
    st = {}
    try:
        st = json.loads((REPO / "stats" / "latest.json").read_text())
    except Exception:
        pass
    by_title = _results_by_title()
    vids = []
    for _vid, x in (st.get("videos") or {}).items():
        row = {
            "title": x.get("title"),
            "views": x.get("views"),
            "likes": x.get("likes"),
            "music": x.get("music"),
            "published": x.get("published"),
            "duration": x.get("duration"),
            "energy": x.get("energy"),
            "drop_t": x.get("drop_t"),
            "slot": x.get("slot"),
        }
        result = by_title.get(_title_key(x.get("title")))
        measured_drop = False
        if result:
            if not row.get("music"):
                row["music"] = _music_label(result)
            if row.get("energy") is None and "energy" in result:
                row["energy"] = result.get("energy")
            if row.get("drop_t") is None and "drop_t" in result:
                row["drop_t"] = result.get("drop_t")
                measured_drop = True
            if not row.get("slot") and result.get("slot"):
                row["slot"] = _slot_hhmm(result.get("slot"))
        # Omit unknown fields. A joined render with drop_t null is a real "no drop".
        clean = {k: v for k, v in row.items() if v is not None}
        if measured_drop and result.get("drop_t") is None:
            clean["drop_t"] = None
        vids.append(clean)
    vids.sort(key=lambda r: -(r.get("views") or 0))
    return vids[:30]


def main():
    os.environ.setdefault("SKY_GEMINI_BUDGET_FILE", "/tmp/gemini_trends_budget.json")
    os.environ.setdefault("GEMINI_MAX_CALLS", "3")
    now = datetime.now(IST)
    lib = json.loads((REPO / "music" / "library.json").read_text())
    songs = [f"{t['title']} - {t['artist']}" for t in lib.get("tracks", [])]
    sample = yt_search_sample()
    ours = own_performance()
    research = gemini.generate_text(
        f"""Today is {now:%d %B %Y}. Research (use Google Search) the fastest-growing YouTube Shorts channels and
viral edits of the last 30 days in the movie/trailer edit niche: Bollywood, South Indian (Telugu/Tamil/Kannada) and
Hollywood movie edits, celebrity edits and anime edits. Report concretely: example channels, hook text styles shown on
screen, title patterns, typical lengths in seconds, edit templates (beat sync, slow-mo, captions style), the trending
edit songs/audios right now (phonk, Bollywood/Punjabi mashups, slowed+reverb, South mass songs), hashtags, and best
posting times for India + global audiences. Be specific and current; list song titles with artists.""",
        purpose="trends-research", search=True) or ""
    prompt = f"""You are the growth strategist for "Skywalker", a YouTube Shorts channel posting 4 movie/trailer edits a day
(slots 06:30 Hollywood, 12:30 Indian, 17:30 romance/funny, 22:30 Indian celeb or anime; IST). Layout is fixed (white
English hook bar on top of the uncropped frame, lower-third subtitles, aim about 20-28s, hard publish 18-32s).
Fresh research notes:
{research[:7000]}
Best-effort YouTube search sample: {json.dumps(sample, ensure_ascii=False)[:2500]}
Our own videos (views so far; music, energy, drop_t and slot are joined from local render results when stats omit them): {json.dumps(ours, ensure_ascii=False)[:6000]}
Songs currently in our music library: {json.dumps(songs, ensure_ascii=False)}

Return JSON with keys:
"hook_styles": [5-8 short English hook formulas with an example each, 3-7 words, curiosity/emotion],
"title_patterns": [3-5 patterns that fit "<hook> 💀 | <scene> | #edit"],
"best_length_s": number (prefer 20-28; about 24s with a beat drop has worked best),
"posting_times_ist": [advisory list, do not change our slots],
"caption_styles": [3-5 notes],
"song_picks": [8-12 exact song titles (title only) best for our edits right now; prefer ones in our library],
"songs_to_add": [up to 8 "Title - Artist" trending edit songs NOT in our library],
"hashtags": [8-12 lowercase without #],
"avoid": [3-6 things that hurt retention],
"top_channels": [up to 8 channel names or handles with one-line why],
"our_insights": [3-5 notes on what works/doesn't for our own videos based on views]"""
    data = gemini.generate_json(prompt, purpose="trends-json", temperature=0.4, timeout=90)
    if not isinstance(data, dict):
        print("::warning::trends synthesis failed; keeping previous trends/latest.json")
        return 0 if (REPO / "trends" / "latest.json").exists() else 1
    data["best_length_s"] = max(20, min(28, float(data.get("best_length_s") or 24)))
    data.update(generated_at=now.strftime("%Y-%m-%d %H:%M IST"), research_chars=len(research),
                yt_sample=len(sample), own_videos=len(ours))
    (REPO / "trends").mkdir(exist_ok=True)
    (REPO / "trends" / "latest.json").write_text(json.dumps(data, indent=1, ensure_ascii=False))
    (REPO / "trends" / f"{now:%Y-%m-%d}.json").write_text(json.dumps(data, indent=1, ensure_ascii=False))
    print(json.dumps({k: data.get(k) for k in ("song_picks", "songs_to_add", "hook_styles")}, ensure_ascii=False)[:1500])
    return 0


if __name__ == "__main__":
    sys.exit(main())
