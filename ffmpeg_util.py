#!/usr/bin/env python3
"""FFmpeg helpers with self-recovery.

On CalledProcessError, log stderr and retry with a safe filtergraph preset
(scale + format + setsar, no fancy filters) so a single bad filter does not
kill the whole render.
"""
from __future__ import annotations

import shlex
import subprocess
import sys
from pathlib import Path


SAFE_VF = "scale=1080:-2:flags=lanczos,format=yuv420p,setsar=1"


def _is_ffmpeg(cmd: list) -> bool:
    return bool(cmd) and Path(str(cmd[0])).name in {"ffmpeg", "ffprobe"}


def _with_safe_vf(cmd: list[str]) -> list[str] | None:
    """Replace a -vf / -filter_complex chain with a safe preset when possible."""
    out = list(cmd)
    for flag in ("-filter_complex", "-vf", "-filter:v"):
        if flag in out:
            i = out.index(flag)
            if i + 1 < len(out):
                out[i] = "-vf"
                out[i + 1] = SAFE_VF
                # drop any dangling label maps that a filter_complex output used
                # keep -map 0:v / 0:a if present; if -map [v] exists, swap to 0:v
                cleaned = []
                skip = False
                j = 0
                while j < len(out):
                    if out[j] == "-map" and j + 1 < len(out) and str(out[j + 1]).startswith("["):
                        cleaned.extend(["-map", "0:v:0"])
                        j += 2
                        continue
                    cleaned.append(out[j])
                    j += 1
                return cleaned
    return None


def run(cmd, **kw):
    """subprocess.check_call with ffmpeg stderr capture and one safe retry."""
    cmd = [str(c) for c in cmd]
    if not _is_ffmpeg(cmd):
        subprocess.check_call(cmd, **kw)
        return
    kw = dict(kw)
    # capture stderr so a failure is diagnosable; still inherit stdout when not set
    try:
        subprocess.check_call(cmd, **kw)
        return
    except subprocess.CalledProcessError as e:
        err = ""
        try:
            r = subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True)
            err = (r.stderr or "")[-4000:]
        except Exception:
            err = str(e)
        print(f"ffmpeg failed (exit {e.returncode}). stderr (tail):\n{err}", file=sys.stderr)
        alt = _with_safe_vf(cmd)
        if not alt or alt == cmd:
            raise
        print("Retrying with safe filtergraph preset:", SAFE_VF, file=sys.stderr)
        subprocess.check_call(alt, **kw)
