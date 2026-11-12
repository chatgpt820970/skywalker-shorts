"""Tiny Gemini REST client for the Skywalker pipeline (stdlib only).

- Key comes from env GEMINI_API_KEY (GitHub secret). It is never printed or logged.
- Per-video budget: at most GEMINI_MAX_CALLS (default 5) logical calls and
  GEMINI_MAX_HTTP (default 10) HTTP requests, tracked in a small JSON file
  (SKY_GEMINI_BUDGET_FILE) so every attempt/process of one render shares it.
- Every helper returns None on any failure; callers fall back to heuristics.
"""
from __future__ import annotations

import base64
import json
import os
import time
import urllib.error
import urllib.request
from pathlib import Path

API = "https://generativelanguage.googleapis.com/v1beta/models/{m}:generateContent"
DEFAULT_MODELS = ["gemini-3.5-flash", "gemini-3.8-flash", "gemini-flash-latest", "gemini-3.1-flash-lite"]


def _models() -> list[str]:
    env = (os.environ.get("GEMINI_MODELS") or "").strip()
    return [m.strip() for m in env.split(",") if m.strip()] or list(DEFAULT_MODELS)


def available() -> bool:
    return bool(os.environ.get("GEMINI_API_KEY")) and os.environ.get("SKY_GEMINI", "1") != "0"


def _budget_path() -> Path | None:
    p = os.environ.get("SKY_GEMINI_BUDGET_FILE")
    return Path(p) if p else None


def _budget() -> dict:
    p = _budget_path()
    if p and p.exists():
        try:
            return json.loads(p.read_text())
        except Exception:
            pass
    return {"calls": 0, "http": 0, "log": []}


def _save_budget(b: dict) -> None:
    p = _budget_path()
    if p:
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(b, indent=1))


def budget_left() -> int:
    mx = int(os.environ.get("GEMINI_MAX_CALLS", "5"))
    return max(0, mx - int(_budget().get("calls", 0)))


def _img_part(path: str | Path) -> dict:
    data = Path(path).read_bytes()
    low = str(path).lower()
    mime = ("image/png" if low.endswith(".png") else "video/mp4" if low.endswith(".mp4")
            else "audio/mpeg" if low.endswith(".mp3") else "image/jpeg")
    return {"inline_data": {"mime_type": mime, "data": base64.b64encode(data).decode()}}


def generate_json(prompt: str, images: list | None = None, *, purpose: str = "",
                  temperature: float = 0.6, timeout: float = 90.0) -> dict | list | None:
    """One logical call. Returns parsed JSON or None (never raises)."""
    if not available():
        return None
    b = _budget()
    mx_calls = int(os.environ.get("GEMINI_MAX_CALLS", "5"))
    mx_http = int(os.environ.get("GEMINI_MAX_HTTP", "16"))
    if b.get("calls", 0) >= mx_calls:
        print(f"  gemini[{purpose}]: budget exhausted ({mx_calls} calls); using heuristics")
        return None
    parts = [{"text": prompt}] + [_img_part(p) for p in (images or []) if p and Path(p).exists()]
    body = json.dumps({
        "contents": [{"role": "user", "parts": parts}],
        "generationConfig": {"responseMimeType": "application/json", "temperature": temperature},
    }).encode()
    key = os.environ["GEMINI_API_KEY"]
    b["calls"] = b.get("calls", 0) + 1
    _save_budget(b)
    delay = 4.0
    deadline = time.time() + float(os.environ.get("GEMINI_DEADLINE_S", "300"))
    models = _models()
    good = _budget().get("good_model")
    if good in models:  # the model that last answered goes first (quota differs per model)
        models = [good] + [m for m in models if m != good]
    for model in models:
        for _try in range(2):
            if time.time() > deadline:
                print(f"  gemini[{purpose}]: deadline reached; using heuristics")
                return None
            b = _budget()
            if b.get("http", 0) >= mx_http:
                print(f"  gemini[{purpose}]: http budget exhausted")
                return None
            b["http"] = b.get("http", 0) + 1
            _save_budget(b)
            t0 = time.time()
            try:
                req = urllib.request.Request(API.format(m=model), data=body, method="POST",
                                             headers={"Content-Type": "application/json", "x-goog-api-key": key})
                with urllib.request.urlopen(req, timeout=timeout) as r:
                    d = json.loads(r.read().decode())
                txt = "".join(p.get("text", "") for p in d["candidates"][0]["content"]["parts"])
                txt = txt.strip()
                if txt.startswith("```"):
                    txt = txt.strip("`").split("\n", 1)[-1].rsplit("```", 1)[0]
                out = json.loads(txt)
                b = _budget()
                b["good_model"] = model
                b.setdefault("log", []).append({"purpose": purpose, "model": model, "s": round(time.time() - t0, 1), "ok": True})
                _save_budget(b)
                print(f"  gemini[{purpose}]: ok via {model} in {time.time() - t0:.1f}s")
                return out
            except urllib.error.HTTPError as e:
                code = e.code
                msg = f"HTTP {code}"
                if code in (400, 401, 403):
                    print(f"  gemini[{purpose}]: {msg} (auth/request); giving up")
                    return None
                if code == 404:
                    break  # model gone; next model
                if code == 429:
                    b = _budget()
                    b.setdefault("log", []).append({"purpose": purpose, "model": model, "err": msg})
                    _save_budget(b)
                    print(f"  gemini[{purpose}]: {model} rate-limited (429); next model")
                    time.sleep(2)
                    break  # quota is per model: move on instead of burning the HTTP budget
            except Exception as e:  # timeout, bad JSON, empty candidates
                msg = type(e).__name__
            b = _budget()
            b.setdefault("log", []).append({"purpose": purpose, "model": model, "err": msg})
            _save_budget(b)
            print(f"  gemini[{purpose}]: {model} failed ({msg}); backoff {delay:.0f}s")
            time.sleep(delay)
            delay = min(delay * 2, 30)
    return None


# ---------------------------------------------------------------- helpers

def load_trends(repo: Path) -> dict:
    p = repo / "trends" / "latest.json"
    try:
        return json.loads(p.read_text())
    except Exception:
        return {}


def generate_text(prompt: str, *, purpose: str = "", search: bool = False, timeout: float = 90.0) -> str | None:
    """Free-text call, optionally grounded with Google Search (used by the trends job). Never raises."""
    if not available():
        return None
    b = _budget()
    if b.get("calls", 0) >= int(os.environ.get("GEMINI_MAX_CALLS", "5")):
        return None
    b["calls"] = b.get("calls", 0) + 1
    _save_budget(b)
    payload = {"contents": [{"role": "user", "parts": [{"text": prompt}]}]}
    if search:
        payload["tools"] = [{"google_search": {}}]
    key = os.environ["GEMINI_API_KEY"]
    delay = 5.0
    plan = [(m, True) for m in _models()[:1]] if search else []
    plan += [(m, False) for m in _models()]
    deadline = time.time() + float(os.environ.get("GEMINI_DEADLINE_S", "300"))
    for model, with_search in plan:
        if time.time() > deadline:
            print(f"  gemini[{purpose}]: deadline reached; giving up")
            return None
        p2 = dict(payload)
        if not with_search:
            p2.pop("tools", None)
        body = json.dumps(p2).encode()
        for _try in range(1):
            try:
                req = urllib.request.Request(API.format(m=model), data=body, method="POST",
                                             headers={"Content-Type": "application/json", "x-goog-api-key": key})
                with urllib.request.urlopen(req, timeout=timeout) as r:
                    d = json.loads(r.read().decode())
                txt = "".join(p.get("text", "") for p in d["candidates"][0]["content"]["parts"]).strip()
                if txt:
                    print(f"  gemini[{purpose}]: ok via {model}")
                    return txt
            except urllib.error.HTTPError as e:
                if e.code in (400, 401, 403):
                    print(f"  gemini[{purpose}]: HTTP {e.code}; giving up")
                    return None
                if e.code == 404:
                    break
                print(f"  gemini[{purpose}]: {model} HTTP {e.code}; backoff")
            except Exception as e:
                print(f"  gemini[{purpose}]: {model} {type(e).__name__}; backoff")
            time.sleep(delay)
            delay = min(delay * 2, 30)
    return None
