#!/usr/bin/env python3
"""Duration gate, clean sub-shot splitting, and serif overlay detection."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import auto_qc
import pipeline
import scene_filter
import similarity


def test_duration_issue_hard_floor():
    assert auto_qc.MIN_DURATION == 20.0
    assert auto_qc.duration_issue(3.9)
    assert auto_qc.duration_issue(19.99)
    assert auto_qc.duration_issue(20.0) is None
    assert auto_qc.duration_issue(25.4) is None
    assert "hard minimum" in auto_qc.duration_issue(1.5)


def test_range_pad_is_two_seconds_not_six():
    assert similarity.RANGE_PAD == 2.0
    used = [(10.0, 12.0)]
    assert similarity.range_blocked(13.5, 15.0, used)
    assert not similarity.range_blocked(14.5, 16.0, used)
    # A 6s pad would block 16.5-18; the default must not.
    assert not similarity.range_blocked(16.5, 18.0, used)


def test_clean_intervals_keeps_only_clean_slices():
    # Dirty flash in the middle of a 6s cut -> two leftovers, not an empty shot.
    runs = scene_filter.clean_intervals(
        10.0, 16.0,
        [(10.2, False), (11.0, False), (12.0, True), (13.0, True), (14.5, False), (15.5, False)],
        min_dur=0.55,
    )
    assert runs == [(10.0, 11.5), (13.75, 16.0)]
    assert scene_filter.clean_intervals(10, 16, [(11, True), (13, True)], 0.55) == []
    assert scene_filter.clean_intervals(10, 16, [(11, False), (14, False)], 0.55) == [(10.0, 16.0)]


def test_plan_edit_refuses_short_unique_cut():
    pipeline.RELAX = False
    rng = np.random.default_rng(1)

    def scene(i, start):
        fr = rng.integers(0, 255, (48, 48, 3), dtype=np.uint8)
        fr[:, :, 0] = (i * 40) % 255
        fp = similarity.fingerprint_frame(fr)
        return {
            "id": i, "start": start, "end": start + 2.0, "dur": 2.0,
            "skip": False, "score": 20.0, "spectacle": 20.0, "skin": 0.05,
            "motion": 8.0, "speech": 0.0,
            "phash": fp["phash"], "hist": fp["hist"],
        }

    scenes = [scene(i, i * 8.0) for i in range(3)]  # 6s total unique
    music = {"pulses": list(np.linspace(0, 32, 45))}
    try:
        pipeline.plan_edit(scenes, music, target=32)
    except SystemExit as e:
        msg = str(e)
        assert "unique clean shots" in msg
        assert "20" in msg or "25" in msg
    else:
        raise AssertionError("plan_edit must fail closed on a short unique cut")


def test_require_min_duration_never_relaxes_below_20():
    pipeline.RELAX = True
    try:
        pipeline._require_min_duration(12.0)
    except SystemExit as e:
        assert "20" in str(e)
    else:
        raise AssertionError("even --relax must refuse <20s")
    pipeline.RELAX = False
    pipeline._require_min_duration(25.0)


def _serif_frame(text="In exchange for 10 million yen"):
    font_path = "/usr/share/fonts/truetype/dejavu/DejaVuSerif.ttf"
    img = Image.new("RGB", (1280, 720), (22, 26, 32))
    pix = np.array(img)
    yy = np.linspace(0, 40, 720).reshape(-1, 1)
    pix[:, :, 0] = np.clip(pix[:, :, 0] + yy, 0, 255)
    pix[:, :, 1] = np.clip(pix[:, :, 1] + yy * 0.6, 0, 255)
    img = Image.fromarray(pix.astype(np.uint8))
    d = ImageDraw.Draw(img)
    font = ImageFont.truetype(font_path, 38)
    bbox = d.textbbox((0, 0), text, font=font)
    tw = bbox[2] - bbox[0]
    d.text(((1280 - tw) // 2, 340), text, font=font, fill=(236, 230, 214))
    return np.array(img)


def test_serif_on_screen_caption_is_detected():
    arr = _serif_frame()
    reasons = scene_filter.detect_overlay_text(arr) + scene_filter.detect_frame(arr)
    assert "text_overlay" in reasons or "text_card" in reasons


def test_plain_footage_is_not_flagged_as_text():
    pix = np.zeros((720, 1280, 3), dtype=np.uint8)
    pix[:, :, 0] = 30
    pix[:, :, 1] = 40
    pix[:, :, 2] = 50
    yy = np.linspace(0, 80, 720).reshape(-1, 1)
    pix[:, :, 1] = np.clip(pix[:, :, 1] + yy, 0, 255)
    assert scene_filter.detect_overlay_text(pix) == []


def test_plan_edit_can_fill_with_short_recovered_slices():
    """Sub-shot leftovers are often 0.7–1.4s; they must still count toward 25s."""
    pipeline.RELAX = False
    rng = np.random.default_rng(7)
    scenes = []
    t = 0.0
    for i in range(32):
        fr = rng.integers(0, 255, (64, 64, 3), dtype=np.uint8)
        fr[:, :] = ((i * 17) % 255, (i * 41) % 255, (255 - i * 9) % 255)
        # unique texture so pHash does not collapse them
        fr.reshape(-1)[i::32] = 255
        fp = similarity.fingerprint_frame(fr)
        dur = 0.95
        scenes.append({
            "id": i + 1, "start": t, "end": t + dur, "dur": dur,
            "skip": False, "score": 12.0 + i, "spectacle": 12.0 + i, "skin": 0.04,
            "motion": 6.0, "speech": 0.0,
            "phash": fp["phash"], "hist": fp["hist"],
        })
        t += 4.0  # well past the 2s range pad
    music = {"pulses": list(np.linspace(0, 36, 50))}
    edl = pipeline.plan_edit(scenes, music, target=32)
    acc = float(edl[-1]["t1"]) if edl else 0.0
    assert acc + 1e-6 >= 25.0, acc
