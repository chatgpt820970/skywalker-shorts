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
- Only copyright-safe music (CC BY Kevin MacLeod / NCS) with exact credit in the description. No kids content.
- Anime/movie edits need ENERGY: hype or phonk music, hard beat cuts, drop flash, and the reveal on the drop.

## Queue item schema (queue/<date>.json, a list)
{id, slot, source_url, template, hook, words, title, description, tags, target_dur}

Usage: `python pipeline.py --template whitebar|depthtext|postcard --out out.mp4 ...` (see `--help`).
