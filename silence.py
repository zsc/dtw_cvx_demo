from __future__ import annotations

from typing import List, Tuple

import librosa
import numpy as np


def trim_long_silence(
    audio: np.ndarray,
    sample_rate: int,
    top_db: float = 40.0,
    min_silence_sec: float = 0.5,
    frame_length: int = 2048,
    hop_length: int = 512,
) -> Tuple[np.ndarray, List[Tuple[int, int]]]:
    """
    Remove long silent gaps while preserving short pauses.
    Returns (trimmed_audio, kept_intervals_samples).
    """
    if audio.size == 0:
        return audio, []

    intervals = librosa.effects.split(
        audio, top_db=top_db, frame_length=frame_length, hop_length=hop_length
    )
    if len(intervals) == 0:
        return audio, []

    kept: List[Tuple[int, int]] = []

    # Keep leading short silence if present.
    lead_gap = intervals[0][0] / float(sample_rate)
    current_start = 0 if lead_gap <= min_silence_sec else int(intervals[0][0])
    current_end = int(intervals[0][1])

    for start, end in intervals[1:]:
        gap_sec = (start - current_end) / float(sample_rate)
        if gap_sec <= min_silence_sec:
            current_end = int(end)
        else:
            kept.append((current_start, current_end))
            current_start = int(start)
            current_end = int(end)

    # Keep trailing short silence if present.
    tail_gap = (len(audio) - current_end) / float(sample_rate)
    if tail_gap <= min_silence_sec:
        current_end = len(audio)
    kept.append((current_start, current_end))

    if not kept:
        return audio, []

    trimmed = np.concatenate([audio[s:e] for s, e in kept]) if kept else audio
    if trimmed.size == 0:
        return audio, []
    return trimmed.astype(np.float32), kept
