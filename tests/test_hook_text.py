#!/usr/bin/env python3
"""English-only hook/title checks, plus a scan of upcoming queue items."""
from __future__ import annotations

import json
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import hook_text

IST = timezone(timedelta(hours=5, minutes=30))
# Published Raja Saab (2026-11-12 12:30 IST) is left as historical Hinglish.
# Everything from the next slot onward must be English.
UPCOMING_FROM = datetime(2026, 11, 12, 17, 30, tzinfo=IST)

REPO = Path(__file__).resolve().parents[1]


def test_rejects_devanagari():
    errs = hook_text.hook_language_errors("Raja Saab सबको", label="hook")
    assert any("Devanagari" in e for e in errs)


def test_rejects_known_hinglish_hooks():
    samples = [
        "The reason why Raja Saab sabko hypnotise kar deta hai",
        "The reason why Peddi playground ka hero ban gaya",
        "The reason why Salman ki Maatrubhumi sabko hit karti hai",
        "The reason why SRK ka Jawan still hits different",
        "The reason why War 2 ka shadow kabhi nahi rukta",
        "The reason why Kantara ka daiva sabko chhoo gaya",
        "The reason why Housefull 5 mein chaos kabhi rukta nahi",
        "The reason why Pushpa ka rule ruka hi nahi",
        "The reason why Saiyaara sabke dil pe lag gaya",
    ]
    for text in samples:
        errs = hook_text.hook_language_errors(text)
        assert errs, f"should reject: {text}"


def test_accepts_english_hooks():
    samples = [
        "The reason why Doctor Doom terrifies every Avenger",
        "The reason why SRK's Jawan still hits different",
        "The reason why War 2's shadow never stops",
        "The reason why Kantara's forest spirit touches everyone",
        "The reason why Thalaivar's Coolie still rules the room",
        "The reason why Mahavatar Narsimha stays in everyone's heart",
        "The reason why the chaos in Housefull 5 never stops",
        "The reason why Pushpa's rule never stopped",
        "The reason why Saiyaara hits everyone right in the heart",
        "The reason why Se7en still shocks",
        "The reason why Koyuki finally smiled",
    ]
    for text in samples:
        errs = hook_text.hook_language_errors(text)
        assert not errs, f"should accept {text!r}: {errs}"


def test_item_checks_hook_and_title():
    bad = {
        "hook": "The reason why SRK ka Jawan still hits different",
        "title": "The reason why SRK ka Jawan still hits different 💀 | #edit",
    }
    errs = hook_text.check_item_hook_fields(bad)
    assert any(e.startswith("hook") for e in errs)
    assert any(e.startswith("title") for e in errs)
    hook_text.require_english_hook({
        "hook": "The reason why Doctor Doom terrifies every Avenger",
        "title": "The reason why Doctor Doom terrifies every Avenger 💀 | #edit",
    })


def test_upcoming_queue_hooks_are_english():
    problems = []
    for path in sorted((REPO / "queue").glob("*.json")):
        for item in json.loads(path.read_text()):
            try:
                slot = datetime.fromisoformat(item["slot"])
            except Exception:
                continue
            if slot < UPCOMING_FROM:
                continue
            errs = hook_text.check_item_hook_fields(item)
            if errs:
                problems.append(f"{item.get('id')}: {'; '.join(errs)}")
    assert not problems, "upcoming hooks/titles must be English:\n" + "\n".join(problems)


if __name__ == "__main__":
    test_rejects_devanagari()
    test_rejects_known_hinglish_hooks()
    test_accepts_english_hooks()
    test_item_checks_hook_and_title()
    test_upcoming_queue_hooks_are_english()
    print("ok")
