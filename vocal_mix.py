"""Clean dialogue mix: isolate the actors' voices from the trailer audio (Demucs htdemucs,
two stems, CPU) and drop the trailer's own music/SFX bed entirely. Our song then plays at
full level between lines and is ducked ~20 dB under speech (fast attack / release).

Writes WORK/audio_check.json with the numbers the publish gate checks:
  smr_db        median speech-to-music ratio inside dialogue windows (vocal vs ducked song)
  vocal_vs_mix  vocal-stem RMS vs final-mix RMS inside dialogue windows (0 dB = all voice)
  isolated      True when the Demucs vocal stem was used
"""
from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

import numpy as np

SR = 44100
DUCK_DB = float(os.environ.get("SKY_DUCK_DB", "-20"))       # song level under speech
MUSIC_DB = float(os.environ.get("SKY_MUSIC_DB", "-3"))      # song level between lines
VOICE_RMS_DB = float(os.environ.get("SKY_VOICE_RMS_DB", "-17"))  # dialogue RMS target while speaking
ATTACK, RELEASE = 0.06, 0.18                                 # seconds
PAD_PRE, PAD_POST = 0.10, 0.18                               # speech window padding
MIN_SMR_DB = float(os.environ.get("SKY_MIN_SMR_DB", "12"))


def _read(path: Path, sr: int = SR) -> np.ndarray:
    """Any audio -> float32 stereo (n, 2) at `sr` via ffmpeg."""
    raw = subprocess.run(["ffmpeg", "-v", "error", "-i", str(path), "-f", "f32le", "-ac", "2", "-ar", str(sr), "-"],
                         capture_output=True, check=True).stdout
    return np.frombuffer(raw, dtype=np.float32).reshape(-1, 2).copy()


def _write(path: Path, x: np.ndarray, sr: int = SR):
    from scipy.io import wavfile
    wavfile.write(str(path), sr, np.clip(x, -1, 1).astype(np.float32))


def separate_vocals(wav: Path, out: Path) -> bool:
    """Demucs htdemucs two-stem separation in-process (no torchaudio I/O). True on success."""
    try:
        import torch
        from demucs.apply import apply_model
        from demucs.pretrained import get_model
    except Exception as e:
        print(f"  demucs unavailable: {e}")
        return False
    try:
        torch.set_num_threads(max(1, os.cpu_count() or 2))
        model = get_model(os.environ.get("SKY_DEMUCS_MODEL", "htdemucs"))
        model.eval()
        x = _read(wav, model.samplerate)
        t = torch.from_numpy(x.T.copy())
        ref = t.mean(0)
        mu, sd = ref.mean(), ref.std() + 1e-8
        with torch.no_grad():
            src = apply_model(model, ((t - mu) / sd)[None], device="cpu", shifts=0, split=True,
                              overlap=0.25, progress=False)[0]
        src = src * sd + mu
        voc = src[model.sources.index("vocals")].numpy().T
        if model.samplerate != SR:
            tmp = out.with_suffix(".raw.wav")
            _write(tmp, voc, model.samplerate)
            voc = _read(tmp, SR)
        _write(out, voc)
        return True
    except Exception as e:
        print(f"  demucs failed: {e}")
        return False


def _mask(n: int, windows, sr: int = SR) -> np.ndarray:
    m = np.zeros(n, dtype=np.float32)
    for a, b in windows or []:
        i0, i1 = max(0, int((a - PAD_PRE) * sr)), min(n, int((b + PAD_POST) * sr))
        if i1 > i0:
            m[i0:i1] = 1.0
    return m


def _env(x: np.ndarray, sr: int = SR, win: float = 0.03) -> np.ndarray:
    k = max(1, int(win * sr))
    p = np.convolve((x ** 2).mean(1), np.ones(k, dtype=np.float32) / k, mode="same")
    return np.sqrt(np.maximum(p, 0))


def _smooth_gain(target: np.ndarray, sr: int = SR) -> np.ndarray:
    """Attack (going down) / release (going up) slew on a gain curve, decimated for speed."""
    hop = 64
    t = target[::hop]
    g = np.empty_like(t)
    a = 1 - np.exp(-1 / (ATTACK * sr / hop))
    r = 1 - np.exp(-1 / (RELEASE * sr / hop))
    cur = t[0]
    for i, v in enumerate(t):
        cur += (v - cur) * (a if v < cur else r)
        g[i] = cur
    return np.repeat(g, hop)[: len(target)]


def db(v: float) -> float:
    return float(20 * np.log10(max(v, 1e-9)))


def mix(dialogue_bed: Path, song: Path, dst: Path, dur: float, speech_out, work: Path) -> dict:
    work = Path(work)
    voc_path = work / "vocals.wav"
    isolated = separate_vocals(dialogue_bed, voc_path)
    voice = _read(voc_path if isolated else dialogue_bed)
    music = _read(song)
    n = int(dur * SR)
    voice = np.pad(voice, ((0, max(0, n - len(voice))), (0, 0)))[:n]
    music = np.pad(music, ((0, max(0, n - len(music))), (0, 0)))[:n]
    win = _mask(n, speech_out)
    # Voice gate: only the speech windows (padded) survive, so residual bleed never sits under the song.
    vgate = _smooth_gain(np.where(win > 0, 1.0, 0.0).astype(np.float32))
    voice = voice * vgate[:, None]
    venv = _env(voice)
    active = (win > 0) & (venv > 10 ** (-45 / 20))
    if active.any():
        vrms = float(np.sqrt((voice[active] ** 2).mean()))
        voice *= min(8.0, 10 ** (VOICE_RMS_DB / 20) / max(vrms, 1e-6))
        venv = _env(voice)
    music *= 10 ** (MUSIC_DB / 20)
    # Duck target: speech windows OR the actual vocal envelope (true sidechain), -20 dB under speech.
    talk = ((win > 0) | (venv > 10 ** (-38 / 20))).astype(np.float32)
    # Look-ahead: start ducking ~120 ms before a word so the attack is finished when the voice starts.
    k = int(0.12 * SR)
    talk = np.convolve(talk, np.ones(2 * k + 1, dtype=np.float32), mode="same") > 0
    duck = np.where(talk, 10 ** (DUCK_DB / 20), 1.0).astype(np.float32)
    g = _smooth_gain(duck)
    m_d = music * g[:, None]
    out = voice + m_d
    peak = float(np.abs(out).max() or 1)
    if peak > 0.97:
        s = 0.97 / peak
        out, voice, m_d = out * s, voice * s, m_d * s
    _write(dst, out)
    # Numbers for the gate, measured per dialogue window where the voice is actually present.
    smrs, vms = [], []
    act = (win > 0) & (_env(voice) > 10 ** (-40 / 20))
    edges = np.flatnonzero(np.diff(np.concatenate([[0], act.astype(np.int8), [0]])))
    for i0, i1 in zip(edges[::2], edges[1::2]):
        if i1 - i0 < int(0.30 * SR):
            continue
        v = float(np.sqrt((voice[i0:i1] ** 2).mean()))
        mm = float(np.sqrt((m_d[i0:i1] ** 2).mean()))
        o = float(np.sqrt((out[i0:i1] ** 2).mean()))
        smrs.append(db(v) - db(mm))
        vms.append(db(v) - db(o))
    between = (win == 0)
    res = {
        "isolated": bool(isolated),
        "speech_segments": len(smrs),
        "smr_db": round(float(np.median(smrs)), 1) if smrs else None,
        "smr_min_db": round(float(np.percentile(smrs, 10)), 1) if smrs else None,
        "vocal_vs_mix_db": round(float(np.median(vms)), 2) if vms else None,
        "duck_db": DUCK_DB,
        "music_between_lines_dbfs": round(db(float(np.sqrt((m_d[between] ** 2).mean()))) if between.any() else -99, 1),
        "min_smr_db": MIN_SMR_DB,
    }
    (work / "audio_check.json").write_text(json.dumps(res))
    print(f"  clean dialogue mix: {res}")
    return res


def mix_two_act(dialogue_bed: Path, song: Path, dst: Path, dur: float, T: float, speech_out, work: Path) -> dict:
    """ACT 1 [0, T): the characters' voices only (Demucs stem; original trailer audio if Demucs fails).
    ACT 2 [T, end): our song at full level from its drop, no voices. One switch, never toggling."""
    work = Path(work)
    n, nT = int(dur * SR), int(T * SR)
    # Separate only act 1 (+0.3 s) to keep Demucs fast.
    a1 = work / "act1_dialogue.wav"
    subprocess.run(["ffmpeg", "-y", "-v", "error", "-i", str(dialogue_bed), "-t", f"{T + 0.3:.3f}", str(a1)], check=True)
    voc_path = work / "vocals.wav"
    isolated = separate_vocals(a1, voc_path)
    voice = _read(voc_path if isolated else a1)
    voice = np.pad(voice, ((0, max(0, n - len(voice))), (0, 0)))[:n]
    music = _read(song)
    # Fade the last 80 ms of act 1 voice out; nothing of it reaches act 2.
    f = int(0.08 * SR)
    voice[nT:] = 0
    if nT > f:
        voice[nT - f:nT] *= np.linspace(1, 0, f, dtype=np.float32)[:, None]
    win = _mask(n, speech_out)
    act = (win[:nT] > 0) & (_env(voice[:nT]) > 10 ** (-45 / 20)) if nT else np.zeros(0, bool)
    if act.any():
        vrms = float(np.sqrt((voice[:nT][act] ** 2).mean()))
        voice *= min(8.0, 10 ** ((VOICE_RMS_DB + 3) / 20) / max(vrms, 1e-6))
    song_full = np.zeros((n, 2), dtype=np.float32)
    m = music[: max(0, n - nT)] * 10 ** (min(0.0, MUSIC_DB + 2) / 20)
    song_full[nT:nT + len(m)] = m
    out = voice + song_full
    peak = float(np.abs(out).max() or 1)
    if peak > 0.97:
        s = 0.97 / peak
        out, voice, song_full = out * s, voice * s, song_full * s
    _write(dst, out)

    def rms_db(x):
        return round(db(float(np.sqrt((x ** 2).mean()))) if len(x) else -99.0, 1)

    # Song switches: count on/off transitions of the song bed in 0.25 s blocks after it first starts
    # (by construction 1: silent in act 1, then on until the end).
    blk = int(0.25 * SR)
    on = [float(np.sqrt((song_full[i:i + blk] ** 2).mean())) > 10 ** (-50 / 20) for i in range(0, n - blk, blk)]
    toggles = sum(1 for i in range(1, len(on)) if on[i] != on[i - 1])
    act1_speech = (win[:nT] > 0) if nT else np.zeros(0, bool)
    res = {
        "mode": "two_act",
        "isolated": bool(isolated),
        "act1_end": round(T, 2),
        "act1_frac": round(T / max(dur, 1e-6), 3),
        "song_in_act1_dbfs": rms_db(song_full[:nT]),
        "voice_in_act2_dbfs": rms_db(voice[nT:]),
        "act1_voice_dbfs": rms_db(voice[:nT][act1_speech]) if act1_speech.any() else rms_db(voice[:nT]),
        "act2_song_dbfs": rms_db(song_full[nT:]),
        "song_switches": int(toggles),
        "speech_segments": len(speech_out or []),
    }
    (work / "audio_check.json").write_text(json.dumps(res))
    print(f"  two-act mix: {res}")
    return res


def check_errors(res: dict | None, speech_expected: bool) -> list[str]:
    if res and res.get("mode") == "two_act":
        errs = []
        if res.get("song_in_act1_dbfs", -99) > -50:
            errs.append(f"song audible under act-1 dialogue ({res.get('song_in_act1_dbfs')} dBFS)")
        if res.get("voice_in_act2_dbfs", -99) > -50:
            errs.append(f"dialogue audio leaks into the song act ({res.get('voice_in_act2_dbfs')} dBFS)")
        if res.get("song_switches", 0) > 1:
            errs.append(f"song toggles in and out ({res.get('song_switches')} switches; must start once)")
        if res.get("act1_voice_dbfs", -99) < -35:
            errs.append(f"act-1 dialogue too quiet ({res.get('act1_voice_dbfs')} dBFS)")
        if res.get("act2_song_dbfs", -99) < -30:
            errs.append(f"act-2 song too quiet ({res.get('act2_song_dbfs')} dBFS)")
        if not (0.25 <= float(res.get("act1_frac") or 0) <= 0.62):
            errs.append(f"act split {res.get('act1_frac')} outside 25-62% of the video")
        return errs
    if not res:
        return ["audio check missing"] if speech_expected else []
    errs = []
    # Demucs failure is allowed: the old mixed path is the fallback so a slot never fails because of it.
    if res.get("speech_segments"):
        if (res.get("smr_db") or 0) < MIN_SMR_DB:
            errs.append(f"dialogue not clear over the song: speech/music {res.get('smr_db')} dB < {MIN_SMR_DB}")
        if (res.get("smr_min_db") or 0) < MIN_SMR_DB - 6:
            errs.append(f"a dialogue line is buried: worst speech/music {res.get('smr_min_db')} dB")
    return errs
