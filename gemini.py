"""Tiny Gemini REST client for the Skywalker pipeline (stdlib only).

- Keys come from env GEMINI_API_KEY and GEMINI_API_KEY_2..GEMINI_API_KEY_6
  (GitHub secrets). Values are never printed, logged, or written to the budget file.
- On 429 / quota / 503 for a key, try the next model, then the next key.
  The last key (and model) that worked is tried first on the next call.
  Missing or empty keys are skipped.
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
KEY_ENVS = ("GEMINI_API_KEY",) + tuple(f"GEMINI_API_KEY_{i}" for i in range(2, 7))


def _models() -> list[str]:
    env = (os.environ.get("GEMINI_MODELS") or "").strip()
    return [m.strip() for m in env.split(",") if m.strip()] or list(DEFAULT_MODELS)


def _key_slot(name: str) -> str:
    """Log label only (key1, key2, ...); never the secret value."""
    if name == "GEMINI_API_KEY":
        return "key1"
    return "key" + name.rsplit("_", 1)[-1]


def _keys() -> list[tuple[str, str]]:
    """Non-empty keys as (env_name, value), in slot order. Callers must not log the value."""
    out = []
    for name in KEY_ENVS:
        v = (os.environ.get(name) or "").strip()
        if v:
            out.append((name, v))
    return out


def available() -> bool:
    return bool(_keys()) and os.environ.get("SKY_GEMINI", "1") != "0"


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


def _ordered_keys() -> list[tuple[str, str]]:
    keys = _keys()
    good = _budget().get("good_key")
    if good and any(n == good for n, _ in keys):
        keys = [(n, k) for n, k in keys if n == good] + [(n, k) for n, k in keys if n != good]
    return keys


def _ordered_models() -> list[str]:
    models = _models()
    good = _budget().get("good_model")
    if good in models:
        models = [good] + [m for m in models if m != good]
    return models


def _quota_error(code: int, body: bytes | str | None = None) -> bool:
    if code in (429, 503):
        return True
    if body and code in (403,):
        text = body.decode("utf-8", "ignore") if isinstance(body, (bytes, bytearray)) else str(body)
        low = text.lower()
        return "quota" in low or "resource_exhausted" in low or "rate limit" in low or "rate-limit" in low
    return False


def _img_part(path: str | Path) -> dict:
    data = Path(path).read_bytes()
    low = str(path).lower()
    mime = ("image/png" if low.endswith(".png") else "video/mp4" if low.endswith(".mp4")
            else "audio/mpeg" if low.endswith(".mp3") else "image/jpeg")
    return {"inline_data": {"mime_type": mime, "data": base64.b64encode(data).decode()}}


def _post(model: str, key: str, body: bytes, timeout: float):
    """POST generateContent. Returns ('ok', data) | ('http', code, raw) | ('err', typ)."""
    req = urllib.request.Request(
        API.format(m=model), data=body, method="POST",
        headers={"Content-Type": "application/json", "x-goog-api-key": key},
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return "ok", json.loads(r.read().decode())
    except urllib.error.HTTPError as e:
        raw = b""
        try:
            raw = e.read() or b""
        except Exception:
            pass
        return "http", e.code, raw
    except Exception as e:
        return "err", type(e).__name__


def _text_from_response(d: dict) -> str:
    return "".join(p.get("text", "") for p in d["candidates"][0]["content"]["parts"]).strip()


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
    b["calls"] = b.get("calls", 0) + 1
    _save_budget(b)
    delay = 4.0
    deadline = time.time() + float(os.environ.get("GEMINI_DEADLINE_S", "300"))
    for name, key in _ordered_keys():
        slot = _key_slot(name)
        for model in _ordered_models():
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
                result = _post(model, key, body, timeout)
                kind = result[0]
                if kind == "ok":
                    try:
                        txt = _text_from_response(result[1])
                        if txt.startswith("```"):
                            txt = txt.strip("`").split("\n", 1)[-1].rsplit("```", 1)[0]
                        out = json.loads(txt)
                    except Exception as e:
                        result = ("err", type(e).__name__)
                        kind = "err"
                    else:
                        b = _budget()
                        b["good_model"] = model
                        b["good_key"] = name  # env name only, never the secret
                        b.setdefault("log", []).append({
                            "purpose": purpose, "model": model, "key": slot,
                            "s": round(time.time() - t0, 1), "ok": True,
                        })
                        _save_budget(b)
                        print(f"  gemini[{purpose}]: ok via {slot}/{model} in {time.time() - t0:.1f}s")
                        return out
                    # parse failed: fall through to retry/backoff
                if kind == "http":
                    code, raw = result[1], result[2]
                    msg = f"HTTP {code}"
                    if code == 400:
                        print(f"  gemini[{purpose}]: {msg} (request); giving up")
                        return None
                    if code in (401, 403) and not _quota_error(code, raw):
                        print(f"  gemini[{purpose}]: {slot} {msg} (auth); next key")
                        b = _budget()
                        b.setdefault("log", []).append({"purpose": purpose, "model": model, "key": slot, "err": msg})
                        _save_budget(b)
                        model = None  # leave inner loops
                        break
                    if code == 404:
                        break  # model gone; next model, same key
                    if _quota_error(code, raw):
                        b = _budget()
                        b.setdefault("log", []).append({"purpose": purpose, "model": model, "key": slot, "err": msg})
                        _save_budget(b)
                        print(f"  gemini[{purpose}]: {slot}/{model} rate-limited ({msg}); next model/key")
                        time.sleep(2)
                        break  # next model; after models, next key
                else:
                    msg = result[1]
                b = _budget()
                b.setdefault("log", []).append({"purpose": purpose, "model": model, "key": slot, "err": msg})
                _save_budget(b)
                print(f"  gemini[{purpose}]: {slot}/{model} failed ({msg}); backoff {delay:.0f}s")
                time.sleep(delay)
                delay = min(delay * 2, 30)
            if model is None:
                break  # auth failure: next key
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
    mx_calls = int(os.environ.get("GEMINI_MAX_CALLS", "5"))
    mx_http = int(os.environ.get("GEMINI_MAX_HTTP", "16"))
    if b.get("calls", 0) >= mx_calls:
        return None
    b["calls"] = b.get("calls", 0) + 1
    _save_budget(b)
    payload = {"contents": [{"role": "user", "parts": [{"text": prompt}]}]}
    if search:
        payload["tools"] = [{"google_search": {}}]
    delay = 5.0
    plan = [(m, True) for m in _models()[:1]] if search else []
    plan += [(m, False) for m in _ordered_models()]
    deadline = time.time() + float(os.environ.get("GEMINI_DEADLINE_S", "300"))
    for name, key in _ordered_keys():
        slot = _key_slot(name)
        for model, with_search in plan:
            if time.time() > deadline:
                print(f"  gemini[{purpose}]: deadline reached; giving up")
                return None
            b = _budget()
            if b.get("http", 0) >= mx_http:
                print(f"  gemini[{purpose}]: http budget exhausted")
                return None
            b["http"] = b.get("http", 0) + 1
            _save_budget(b)
            p2 = dict(payload)
            if not with_search:
                p2.pop("tools", None)
            body = json.dumps(p2).encode()
            result = _post(model, key, body, timeout)
            kind = result[0]
            if kind == "ok":
                txt = _text_from_response(result[1])
                if txt:
                    b = _budget()
                    b["good_model"] = model
                    b["good_key"] = name
                    _save_budget(b)
                    print(f"  gemini[{purpose}]: ok via {slot}/{model}")
                    return txt
            elif kind == "http":
                code, raw = result[1], result[2]
                if code == 400:
                    print(f"  gemini[{purpose}]: HTTP {code}; giving up")
                    return None
                if code in (401, 403) and not _quota_error(code, raw):
                    print(f"  gemini[{purpose}]: {slot} HTTP {code} (auth); next key")
                    break  # next key
                if code == 404:
                    continue
                print(f"  gemini[{purpose}]: {slot}/{model} HTTP {code}; next model/key")
                if _quota_error(code, raw):
                    time.sleep(2)
                    continue
            else:
                print(f"  gemini[{purpose}]: {slot}/{model} {result[1]}; backoff")
            time.sleep(delay)
            delay = min(delay * 2, 30)
    return None
