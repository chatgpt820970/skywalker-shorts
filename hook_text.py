#!/usr/bin/env python3
"""English-only checks for the white top-bar hook and the YouTube title.

Body subtitles may follow the trailer audio (including Hindi). This module
only guards hook, title, and any hook sentence copied into the description.
"""
from __future__ import annotations

import re

DEVANAGARI = re.compile(r"[\u0900-\u097F]")

# Unambiguous Hindi/Hinglish romanizations. Keep this conservative so English
# words and proper names are not blocked. Film titles and character names in
# a hook are fine. Short postpositions are matched as whole tokens only
# (so "Se7en" and "SRK's" are not flagged).
HINGLISH_WORDS = frozenset({
    "kar", "hai", "hain", "sabko", "sabke", "sabse", "kyun", "kyu", "kya",
    "bhai", "wala", "wali", "wale", "nahi", "nahin", "toh",
    "hoon", "hoga", "hogi", "deta", "deti", "dete",
    "raha", "rahi", "rahe", "gaya", "gayi", "gaye",
    "bahut", "itna", "itni", "apna", "apni", "usse", "usko", "isko",
    "yeh", "woh", "acha", "accha", "bohot", "zabardast",
    "dekho", "suno", "bolo", "yaar", "lekin", "phir", "abhi",
    "andar", "bahar", "pehle", "kaise", "kaisa", "kitna", "kitni",
    "kyunki", "isliye", "bilkul", "ekdum", "pagal", "duniya",
    "pyaar", "ishq", "zindagi", "jhooth", "dushman", "ladka", "ladki",
    "aaj", "subah", "shaam", "paani", "khana", "gaadi", "paisa",
    "logon", "unko", "unka", "unki", "hamara", "hamari", "inhe", "unhe",
    "jisne", "jisko", "arey", "arre", "kamaal", "mazak", "masti",
    "bindaas", "jhakkas", "sasta", "mehnga", "mehanga", "thoda", "zyada",
    "sirf", "mein", "kabhi", "rukta", "rukti", "ruka", "chhoo", "chhu",
    "utar", "karti", "karta", "karte", "bhi", "tha", "thi",
    # Hindi postpositions that are not English words. Matched as whole tokens.
    "ka", "ki", "ke", "ko", "pe",
})

# Whole tokens: letters, digits, and apostrophes (SRK's, War 2).
TOKEN = re.compile(r"[A-Za-z0-9']+")


def _tokens(text: str) -> list[str]:
    return TOKEN.findall(text or "")


def hook_language_errors(text: str, *, label: str = "hook") -> list[str]:
    """Return human-readable errors if text is not pure English."""
    errors: list[str] = []
    text = text or ""
    if DEVANAGARI.search(text):
        errors.append(f"{label} contains Devanagari; use English only")
    bad = sorted({tok.lower() for tok in _tokens(text) if tok.lower() in HINGLISH_WORDS})
    if bad:
        errors.append(f"{label} has Hinglish/Hindi word(s): {', '.join(bad)}")
    return errors


# Owner rule (12 Nov 2026 night): no fixed "The reason why ..." template in the hook bar or title.
TEMPLATE_RE = re.compile(r"^\W*(the\s+)?reasons?\s+why\b", re.I)


def template_errors(text: str, *, label: str = "hook") -> list[str]:
    if TEMPLATE_RE.search(text or ""):
        return [f"{label} uses the banned 'The reason why' template"]
    return []


def opener(text: str) -> str:
    """First two words, lowercased, without punctuation (used to avoid repeating hook openers)."""
    toks = [t.lower().strip("'") for t in _tokens(text or "")]
    return " ".join(toks[:2])


def check_item_hook_fields(item: dict) -> list[str]:
    """Check hook and title on a queue item. Description is not required to
    be fully English (source lines, hashtags), but the hook sentence that
    is copied into it should already match the English hook.
    """
    errors: list[str] = []
    for key in ("hook", "title"):
        errors.extend(hook_language_errors(item.get(key) or "", label=key))
    return errors


def require_english_hook(item: dict) -> None:
    errors = check_item_hook_fields(item)
    if errors:
        raise ValueError(
            "Hook/title must be pure English (no Hindi/Hinglish). " + "; ".join(errors)
        )
