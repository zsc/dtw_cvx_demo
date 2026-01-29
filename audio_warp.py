from __future__ import annotations

from typing import Tuple

import librosa
import numpy as np
import soundfile as sf

from silence import trim_long_silence


def load_audio_mono(
    path: str,
    sample_rate: int = 16000,
    trim_silence: bool = True,
    trim_top_db: float = 40.0,
    trim_min_silence_sec: float = 0.5,
    trim_frame_length: int = 2048,
    trim_hop_length: int = 512,
) -> Tuple[np.ndarray, float, float]:
    audio, sr = librosa.load(path, sr=sample_rate, mono=True)
    audio = audio.astype(np.float32)
    duration_raw = float(audio.shape[0]) / float(sr) if audio.size else 0.0
    if trim_silence and audio.size:
        audio, _ = trim_long_silence(
            audio,
            sample_rate=sr,
            top_db=trim_top_db,
            min_silence_sec=trim_min_silence_sec,
            frame_length=trim_frame_length,
            hop_length=trim_hop_length,
        )
    duration = float(audio.shape[0]) / float(sr) if audio.size else 0.0
    return audio, duration, duration_raw


def warp_audio_to_match(
    audio_b: np.ndarray,
    sr: int,
    u: np.ndarray,
    v: np.ndarray,
    D1: float,
    D2: float,
) -> np.ndarray:
    if audio_b.size == 0 or D1 <= 0.0 or D2 <= 0.0:
        return np.zeros((0,), dtype=np.float32)

    out_len = max(int(round(D1 * sr)), 1)
    u_query = np.linspace(0.0, 1.0, out_len, dtype=np.float32)
    v_query = np.interp(u_query, u, v)
    t2 = v_query * D2
    idx = t2 * sr

    x = np.arange(audio_b.shape[0], dtype=np.float32)
    warped = np.interp(idx, x, audio_b).astype(np.float32)
    return warped


def save_wav(path: str, audio: np.ndarray, sr: int) -> None:
    sf.write(path, audio, sr)
