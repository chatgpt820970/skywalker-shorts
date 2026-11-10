#!/usr/bin/env bash
# Last-resort local render for a queue item.
# Uses the box venv at /workspace/skywalker/.editvenv, runs pipeline.py,
# then lanczos-upscales to 2160x3840.
#
# Usage: scripts/local_render.sh <queue-id>
# Optional env:
#   SKY_VENV   python venv (default /workspace/skywalker/.editvenv)
#   SKY_OUT    output directory (default <repo>/out)
set -euo pipefail

REPO="$(cd "$(dirname "$0")/.." && pwd)"
ID="${1:-}"
if [[ -z "$ID" ]]; then
  echo "Usage: scripts/local_render.sh <queue-id>" >&2
  exit 2
fi

VENV="${SKY_VENV:-/workspace/skywalker/.editvenv}"
PY="$VENV/bin/python"
if [[ ! -x "$PY" ]]; then
  echo "venv python not found at $PY" >&2
  exit 1
fi

export GH_TOKEN="${GH_TOKEN:-${GITHUB_TOKEN:-}}"
cd "$REPO"
# Prefer a local-friendly root so pipeline finds music/ next to this repo.
export SKY_AE="$REPO"
export SKY_ROOT="${SKY_OUT:-$REPO/out}"
mkdir -p "$SKY_ROOT"

# ffmpeg with vidstab is required (same as CI).
if ! ffmpeg -hide_banner -filters 2>/dev/null | grep -q vidstabdetect; then
  echo "ffmpeg lacks vidstab; install a build with libvidstab (CI uses johnvansickle static)." >&2
  exit 1
fi

"$PY" "$REPO/scripts/render_item.py" "$ID"
echo "Local render finished for $ID"
# Point at the usual artifacts
ls -lh "$SKY_ROOT/render-$ID.mp4" "$REPO/results/$ID.json" 2>/dev/null || true
