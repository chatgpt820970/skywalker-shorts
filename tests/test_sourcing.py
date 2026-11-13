"""Source-chain unit tests (no network)."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import sourcing


def test_rank_prefers_github_and_studio_over_youtube():
    gh = "https://github.com/chatgpt820970/skywalker-shorts/releases/download/sources/x.mp4"
    studio = "https://www.marvel.com/movies/foo"
    yt = "https://www.youtube.com/watch?v=abc"
    assert sourcing._rank_url(gh) < sourcing._rank_url(studio) < sourcing._rank_url(yt)


def test_build_candidate_urls_orders_fallback_before_youtube():
    item = {
        "id": "20261114-0900-runningman",
        "words": "RUNNING MAN",
        "source_url": "https://www.youtube.com/watch?v=KD18ddeFuyM",
        "fallback_mp4": [
            "https://github.com/chatgpt820970/skywalker-shorts/releases/download/sources/runningman_trailer_h264.mp4"
        ],
        "alt_source_urls": [
            "https://www.dailymotion.com/video/x9u83ec",
            "https://www.youtube.com/watch?v=other",
        ],
    }
    urls = sourcing.build_candidate_urls(item)
    assert urls[0].endswith("runningman_trailer_h264.mp4")
    yt_idxs = [i for i, u in enumerate(urls) if "youtube.com" in u or "youtu.be" in u]
    non_yt = [i for i, u in enumerate(urls) if "youtube.com" not in u and "youtu.be" not in u]
    assert non_yt and yt_idxs
    assert max(non_yt) < min(yt_idxs) or urls.index(item["fallback_mp4"][0]) == 0


def test_content_kinds_include_scenes_not_only_trailers():
    assert "best scene" in sourcing.CONTENT_KINDS
    assert "fight scene" in sourcing.CONTENT_KINDS
    assert "romance scene" in sourcing.CONTENT_KINDS
    item = {"id": "x-y-sololeveling", "words": "SOLO LEVELING", "pillar": "anime",
            "content_kind": "fight scene"}
    qs = sourcing.search_queries(item)
    assert any("fight" in q.lower() or "official clip" in q.lower() for q in qs)


def test_topic_key_from_queue_id():
    assert sourcing.topic_key({"id": "20261114-0900-runningman"}) == "runningman"
    assert sourcing.topic_key({"id": "backup-superman"}) == "superman"
