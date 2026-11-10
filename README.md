# Skywalker Shorts auto-editor

Pipeline for the YouTube channel Skywalker (@Skywalker-k3i).

## Who does what
- **GitHub Actions (this repo):** downloads sources, edits (scene cuts, beat sync, stabilize, grade), renders templates, QA, 4K upscale, and publishes Release assets.
- **Helper bot:** fills `queue/` with trending topics and hooks, builds and fixes the workflows and code.
- **BIBI:** checks frames, uploads through YouTube Studio, watches copyright, and runs analytics every 3 days.

## HARD RULES
- NEVER crop the source frame. The full original aspect ratio must be visible.
- whitebar (default): a tight white caption bar (bold black italic hook, ~30px padding) sits directly on top of the full frame (1080 wide), with no gap. Bar + frame form one block centered vertically on a black 1080x1920 canvas.
- depthtext: big 3D words behind the subject. postcard: a logo + name + hook header directly above the clip. Same no-crop rule.
- Nothing is written on the footage itself. Stabilized, no shake.
- Never include studio/distributor logo intros (Netflix N, Warner, Lionsgate, Marvel, etc.), title cards, rating cards, coming-soon/date cards, credits, or near-black empty filler. Only real story/character footage. A small corner watermark that is part of the original frame may stay (do not crop it out).
- Only copyright-safe music (CC BY Kevin MacLeod / NCS) with exact credit in the description. No kids content.
- Anime/movie edits need ENERGY: hype or phonk music, hard beat cuts, drop flash, and the reveal on the drop.

## Queue item schema (queue/<date>.json, a list)
{id, slot, source_url, template, hook, words, title, description, tags, target_dur}

Usage: `python pipeline.py --template whitebar|depthtext|postcard --out out.mp4 ...` (see `--help`).

## Render workflow (`.github/workflows/render.yml`)
- Triggers: push to `queue/*.json`, `workflow_dispatch` (optional `ids`, `max`, `retry_failed`), and a 3-hourly cron.
- `scripts/plan.py` renders only items that are `status: queued`, have no done result, and whose IST `slot` is within 3h (max 3 per run), so cron can pre-render ~2 hours before the upload slot. Done items are never re-rendered. Committing the whole 7-day queue therefore never fires 28 renders.
- `scripts/render_item.py` self-heals before any bot steps in: (1) normal yt-dlp + official fallbacks, (2) force the `sources` Release fallback mp4, (3) fallback + a different credited music track + alternate scene selection + relaxed *non-hard* thresholds. Hard rules are never relaxed: no-crop, tight whitebar layout, and no logo/title/dark cards. If all three attempts fail, `results/<id>.json` is `retry_queued` (run stays green) and `auto-retry.yml` re-dispatches the item once. Only after that last automatic attempt fails is the result `failed` and the workflow run red (that wakes BIBI). The failure reason is always in `results/<id>.json`.
- `pipeline.py` skips the first ~7s and last ~10s of long trailers, and drops studio/distributor logo intros, title/rating/coming-soon/credit cards, and near-black empty filler. Only real story/character footage stays. A small corner watermark that is part of the original frame is kept (never cropped). QA samples every selected shot and fails the render if any is a logo/title/dark card.
- 4K is a lanczos upscale of the 1080x1920 edit to 2160x3840. It does not invent detail.
- Last-resort local fallback: `scripts/local_render.sh <queue-id>` (venv `/workspace/skywalker/.editvenv`).

## Dialogue, subtitles, uniqueness
- Original trailer audio is kept. When actors speak, dialogue is at ~1.3 volume and the music is ducked to ~0.22 (`sidechaincompress`, with a volume-envelope fallback). Non-speech shots keep music at ~0.65. A beat drop on the reveal is still allowed.
- Lower-thirds are word-accurate subtitles of what is actually said, transcribed with local `faster-whisper` (no paid API, no committed secrets). Generic filler captions are never used. The white top hook bar is unchanged. Subtitles sit in the bottom black and cover the whole video wherever there is dialogue.
- Every cut must be unique: a used source range `[s, e]` blocks `[s-6, e+6]`, and visually similar shots (pHash / histogram) are rejected. Auto-QC fails the render if the output still has near-duplicate shots.
- Prefer shots that contain a complete spoken line; cuts are snapped so a word is never sliced in half.

## Music
`music/credits.json` lists the 18 tracks the picker may use (NCS + Kevin MacLeod CC BY 4.0) with exact credit text, BPM, mood and license URL. The credit is appended to the Release notes / `description_final`. Romance prefers mid-tempo emotional NCS; anime/celeb prefers phonk. A slowed+reverb style may be applied to the credited file (the credit still names the original track). Copyrighted trending songs are never downloaded.
