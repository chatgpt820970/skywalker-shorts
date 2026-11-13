"""Pick music_only vs two_act for a queue item.

About 3 of every 4 items are a song-only Short (no character voice). The other
1 of 4 uses the two-act dialogue-then-song structure. The split is a stable
hash of the item id so a re-render of the same item stays in the same mode.
Gemini may upgrade music_only to two_act only when a spoken line is a genuinely
iconic hook. An explicit override (queue edit_mode / SKY_EDIT_MODE / --edit-mode)
always wins.
"""
from __future__ import annotations

import hashlib
import os


def item_mode_bucket(item_id: str) -> int:
    """Stable 0-3 bucket from the queue item id (sha1, not Python hash())."""
    return int(hashlib.sha1((item_id or "").encode("utf-8")).hexdigest()[:8], 16) % 4


def default_edit_mode(item_id: str) -> str:
    """3 of every 4 items are music_only; bucket 0 is two_act."""
    return "two_act" if item_mode_bucket(item_id) == 0 else "music_only"


def choose_edit_mode(item_id: str, override: str | None = None, *,
                     iconic_hook: bool = False) -> str:
    """Pick music_only or two_act. An explicit override always wins."""
    ov = (override or os.environ.get("SKY_EDIT_MODE") or "").strip().lower()
    if ov in ("music_only", "two_act"):
        return ov
    mode = default_edit_mode(item_id)
    if mode == "music_only" and iconic_hook:
        return "two_act"
    return mode
