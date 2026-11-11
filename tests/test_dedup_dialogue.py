#!/usr/bin/env python3
"""Unit tests for shot uniqueness, mid-word snapping, and subtitle mapping."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import similarity
import dialogue
import captions


def solid(r, g, b, w=64, h=64):
    fr = np.zeros((h, w, 3), dtype=np.uint8)
    fr[:] = (r, g, b)
    return fr


def test_phash_identical_and_different():
    a = solid(200, 40, 40)
    b = solid(200, 40, 40)
    c = solid(20, 180, 80)
    assert similarity.frames_similar(a, b)
    assert not similarity.frames_similar(a, c)


def test_phash_tolerates_light_noise():
    rng = np.random.default_rng(0)
    yy, xx = np.mgrid[0:64, 0:64]
    a = np.zeros((64, 64, 3), dtype=np.uint8)
    a[..., 0] = (40 + xx * 2 + (yy // 3) * 4).clip(0, 255)
    a[..., 1] = (80 + yy * 2).clip(0, 255)
    a[..., 2] = (160 + ((xx + yy) % 17) * 5).clip(0, 255)
    b = np.clip(a.astype(np.int16) + rng.integers(-6, 7, a.shape), 0, 255).astype(np.uint8)
    assert similarity.frames_similar(a, b)
    c = np.zeros((64, 64, 3), dtype=np.uint8)
    c[..., 0] = (200 - xx * 2).clip(0, 255)
    c[..., 1] = (20 + (xx // 2)).clip(0, 255)
    c[..., 2] = (20 + yy).clip(0, 255)
    assert not similarity.frames_similar(a, c)


def test_range_pad_blocks_nearby():
    used = [(10.0, 12.0)]
    # Default pad is 2s — nearby enough to collide, far enough not to.
    assert similarity.RANGE_PAD == 2.0
    assert similarity.range_blocked(13.5, 15.0, used)          # 13.5 < 12+2
    assert similarity.range_blocked(8.5, 9.5, used)            # 9.5 > 10-2
    assert not similarity.range_blocked(14.5, 16.0, used)      # 14.5 >= 14
    assert not similarity.range_blocked(0.0, 7.5, used)
    # Explicit wider pad still works.
    assert similarity.range_blocked(16.5, 18.0, used, pad=6.0)
    assert not similarity.range_blocked(19.0, 21.0, used, pad=6.0)


def test_edl_duplicate_pairs_range_and_visual():
    red = similarity.fingerprint_frame(solid(220, 20, 20))
    blu = similarity.fingerprint_frame(solid(20, 20, 220))
    scenes = {
        1: {"id": 1, **red},
        2: {"id": 2, **red},  # same look
        3: {"id": 3, **blu},
    }
    edl = [
        {"i": 0, "scene": 1, "src_start": 4.0, "visible": 2.0, **red},
        {"i": 1, "scene": 2, "src_start": 20.0, "visible": 2.0, **red},
        {"i": 2, "scene": 3, "src_start": 40.0, "visible": 2.0, **blu},
    ]
    hits = similarity.edl_duplicate_pairs(edl, scenes)
    reasons = {tuple(h["reasons"]) for h in hits}
    assert any("visual" in h["reasons"] for h in hits)
    # 4-6 vs 20-22 are far past the 2s pad, so range_pad should not fire on 1 vs 2
    close = [
        {"i": 0, "scene": 1, "src_start": 4.0, "visible": 2.0, **blu},
        {"i": 1, "scene": 3, "src_start": 7.5, "visible": 2.0, **red},  # 7.5 is inside [4-2, 6+2]
    ]
    hits2 = similarity.edl_duplicate_pairs(close, {1: {"id": 1, **blu}, 3: {"id": 3, **red}})
    assert any("range_pad" in h["reasons"] for h in hits2)


def test_snap_does_not_cut_mid_word():
    words = [
        {"start": 1.0, "end": 1.40, "word": "Hello"},
        {"start": 1.45, "end": 1.90, "word": "Lucy"},
        {"start": 3.00, "end": 3.50, "word": "wait"},
    ]
    # Window starts inside "Hello" and ends inside "wait"
    a, vis = dialogue.snap_window(1.20, 2.20, 0.5, 5.0, words, min_vis=0.7)
    assert dialogue.cuts_mid_word(a, words) is None
    assert dialogue.cuts_mid_word(a + vis, words) is None
    assert vis >= 0.7


def test_speech_bonus_prefers_complete_line():
    tr = {
        "segments": [{"start": 10.0, "end": 12.5, "text": "I don't do the math on love"}],
        "words": [
            {"start": 10.0, "end": 10.3, "word": "I"},
            {"start": 10.3, "end": 10.6, "word": "don't"},
            {"start": 10.6, "end": 10.9, "word": "do"},
            {"start": 10.9, "end": 11.1, "word": "the"},
            {"start": 11.1, "end": 11.4, "word": "math"},
            {"start": 11.4, "end": 11.6, "word": "on"},
            {"start": 11.6, "end": 12.5, "word": "love"},
        ],
    }
    complete = dialogue.speech_bonus(9.8, 12.8, tr)
    partial = dialogue.speech_bonus(11.2, 12.0, tr)
    none = dialogue.speech_bonus(20.0, 22.0, tr)
    assert complete > none
    assert complete > partial  # mid-word leftovers are penalized, not rewarded


def test_map_words_and_caption_cues():
    edl = [
        {"i": 0, "scene": 1, "src_start": 10.0, "visible": 3.0, "t0": 0.0, "t1": 3.0, "speed": 1.0},
        {"i": 1, "scene": 2, "src_start": 20.0, "visible": 2.0, "t0": 3.0, "t1": 5.0, "speed": 1.0},
    ]
    words = [
        {"start": 10.1, "end": 10.4, "word": "I"},
        {"start": 10.4, "end": 10.8, "word": "can't"},
        {"start": 10.8, "end": 11.3, "word": "love"},
        {"start": 20.2, "end": 20.7, "word": "you"},
        {"start": 30.0, "end": 30.4, "word": "SKIPPED"},
    ]
    mapped = dialogue.map_words_to_output(edl, words)
    assert [w["word"] for w in mapped] == ["I", "can't", "love", "you"]
    # 10.1 in first clip -> output 0.1
    assert abs(mapped[0]["t0"] - 0.1) < 0.05
    # 20.2 in second clip -> output 3.2
    assert abs(mapped[-1]["t0"] - 3.2) < 0.05
    cues = dialogue.caption_cues(mapped)
    texts = " ".join(c["text"] for c in cues)
    assert "I can't love" in texts
    assert "you" in texts
    assert "SKIPPED" not in texts
    # generate_lines must use dialogue, never filler templates
    caps = captions.generate_lines("The reason why Lucy couldn't do the math on love",
                                   "MATERIALISTS", 32.0, dialogue=cues)
    assert caps
    assert all("changed everything" not in c["text"].lower() for c in caps)
    assert captions.generate_lines("hook", "X", 32.0, dialogue=None) == []


def test_caption_cues_sort_by_output_time():
    """Words are produced in source order; cues must follow the output timeline."""
    mapped = [
        {"t0": 9.8, "t1": 10.2, "word": "settle", "clip": 5},
        {"t0": 10.2, "t1": 10.6, "word": "down", "clip": 5},
        {"t0": 5.5, "t1": 5.9, "word": "Coconut", "clip": 2},
        {"t0": 6.0, "t1": 6.4, "word": "beer", "clip": 2},
        {"t0": 13.0, "t1": 13.4, "word": "lonely", "clip": 6},
        {"t0": 6.5, "t1": 6.9, "word": "HAVE", "clip": 3},  # next clip, must not join beer
    ]
    cues = dialogue.caption_cues(mapped)
    texts = [c["text"] for c in cues]
    starts = [c["t0"] for c in cues]
    assert starts == sorted(starts)
    joined = " | ".join(texts)
    assert "Coconut beer" in joined
    assert "settle down" in joined
    assert not any("settle" in c and "Coconut" in c for c in texts)
    assert not any("beer" in c and "HAVE" in c for c in texts)


def test_fit_dialogue_extends_to_segment_end():
    tr = {
        "segments": [{"start": 10.0, "end": 12.4, "text": "You can do better than me."}],
        "words": [
            {"start": 10.0, "end": 10.3, "word": "You"},
            {"start": 10.3, "end": 10.5, "word": "can"},
            {"start": 10.5, "end": 10.7, "word": "do"},
            {"start": 10.7, "end": 11.0, "word": "better"},
            {"start": 11.0, "end": 11.3, "word": "than"},
            {"start": 11.3, "end": 12.4, "word": "me."},
        ],
    }
    # Planned 1.8s window would have cut "me."
    a, vis = dialogue.fit_dialogue_window(10.0, 1.8, 9.5, 13.0, tr, min_vis=1.5, max_vis=3.5)
    assert a <= 10.0 + 0.05
    assert a + vis >= 12.4 - 0.05
    assert 1.5 <= vis <= 3.5


def test_fit_dialogue_drops_line_that_cannot_fit():
    tr = {
        "segments": [{"start": 10.0, "end": 14.5, "text": "a very long line that exceeds max_vis"}],
        "words": [{"start": 10.0, "end": 14.5, "word": "long"}],
    }
    # Scene is long enough, but the line is 4.5s > max_vis 3.5, so do not slice it
    # into a subtitle. The visual shot may remain; captions must drop the line.
    a, vis = dialogue.fit_dialogue_window(10.0, 2.0, 9.0, 16.0, tr, min_vis=1.5, max_vis=3.5)
    fully = a <= 10.05 and (a + vis) >= 14.45
    outside = (a + vis) <= 10.05 or a >= 14.45
    if not (fully or outside):
        edl = [{"i": 0, "scene": 1, "src_start": a, "visible": vis}]
        # the word itself (10.0-14.5) is cut by the window, so it must not be captioned
        mapped = [{"t0": 0.0, "t1": 0.2, "word": "long", "clip": 0, "src_start": 10.0, "src_end": 14.5}]
        assert dialogue.drop_incomplete_lines(mapped, edl, tr["segments"]) == []


def test_drop_incomplete_lines_from_captions():
    # Clip hears only 0.9s of a 2.4s line: fragment is dropped.
    edl = [{"i": 0, "scene": 1, "src_start": 10.0, "visible": 0.9, "t0": 0.0, "t1": 0.9}]
    segs = [{"start": 10.0, "end": 12.4, "text": "You can do better than me."}]
    mapped = [
        {"t0": 0.0, "t1": 0.4, "word": "You", "clip": 0, "src_start": 10.0, "src_end": 10.4},
        {"t0": 0.4, "t1": 0.8, "word": "can", "clip": 0, "src_start": 10.4, "src_end": 10.8},
    ]
    kept = dialogue.drop_incomplete_lines(mapped, edl, segs)
    assert kept == []
    # Most of the line is heard (1.8/2.4s): the fully-audible words are kept.
    edl[0]["visible"] = 1.8
    kept = dialogue.drop_incomplete_lines(mapped, edl, segs)
    assert [w["word"] for w in kept] == ["You", "can"]


if __name__ == "__main__":
    tests = [v for k, v in list(globals().items()) if k.startswith("test_") and callable(v)]
    failed = 0
    for fn in tests:
        try:
            fn()
            print("OK", fn.__name__)
        except Exception as e:
            failed += 1
            print("FAIL", fn.__name__, type(e).__name__, e)
    if failed:
        raise SystemExit(failed)
    print(f"{len(tests)} passed")
