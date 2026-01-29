from typing import Tuple

import numpy as np
import torch

from silence import trim_long_silence

def _load_whisper_module():
    try:
        import whisper
    except Exception as exc:  # pragma: no cover - dependency errors
        raise RuntimeError(
            "Failed to import openai-whisper. Ensure it is installed in the active environment."
        ) from exc
    return whisper


def _load_audio(
    wav_path: str,
    sample_rate: int = 16000,
    trim_silence: bool = True,
    trim_top_db: float = 40.0,
    trim_min_silence_sec: float = 0.5,
    trim_frame_length: int = 2048,
    trim_hop_length: int = 512,
) -> Tuple[np.ndarray, float, float]:
    whisper = _load_whisper_module()
    audio = whisper.load_audio(wav_path)
    if audio.size == 0:
        return audio.astype(np.float32), 0.0, 0.0
    duration_raw = float(audio.shape[0]) / float(sample_rate)
    if trim_silence:
        audio, _ = trim_long_silence(
            audio.astype(np.float32),
            sample_rate=sample_rate,
            top_db=trim_top_db,
            min_silence_sec=trim_min_silence_sec,
            frame_length=trim_frame_length,
            hop_length=trim_hop_length,
        )
    duration = float(audio.shape[0]) / float(sample_rate) if audio.size else 0.0
    return audio.astype(np.float32), duration, duration_raw


def _log_mel(audio: np.ndarray) -> np.ndarray:
    whisper = _load_whisper_module()
    mel = whisper.log_mel_spectrogram(audio)
    return mel.cpu().numpy() if hasattr(mel, "cpu") else np.asarray(mel)


def _chunk_audio(audio: np.ndarray, chunk_samples: int):
    n_samples = audio.shape[0]
    if n_samples == 0:
        return
    start = 0
    while start < n_samples:
        end = min(start + chunk_samples, n_samples)
        yield audio[start:end]
        start = end


def extract_whisper_features(
    wav_path: str,
    mode: str = "whisper_encoder",
    model_name: str = "base",
    device: str = "cpu",
    trim_silence: bool = True,
    trim_top_db: float = 40.0,
    trim_min_silence_sec: float = 0.5,
    trim_frame_length: int = 2048,
    trim_hop_length: int = 512,
) -> Tuple[np.ndarray, float, float]:
    """
    Returns (features, duration_sec, duration_raw_sec).
    Features are float32 of shape (T, d).
    """
    whisper = _load_whisper_module()

    audio, duration, duration_raw = _load_audio(
        wav_path,
        trim_silence=trim_silence,
        trim_top_db=trim_top_db,
        trim_min_silence_sec=trim_min_silence_sec,
        trim_frame_length=trim_frame_length,
        trim_hop_length=trim_hop_length,
    )
    if audio.size == 0:
        return np.zeros((0, 1), dtype=np.float32), duration, duration_raw

    if mode == "log_mel":
        mel = _log_mel(audio)
        # Transpose to (T, d)
        return mel.T.astype(np.float32), duration, duration_raw

    if mode != "whisper_encoder":
        raise ValueError(f"Unsupported feature_mode: {mode}")

    model = whisper.load_model(model_name, device=device)
    model.eval()

    audio_module = whisper.audio
    chunk_samples = getattr(audio_module, "N_SAMPLES", 30 * 16000)
    max_frames = getattr(audio_module, "N_FRAMES", 3000)
    feats = []

    for audio_chunk in _chunk_audio(audio, chunk_samples):
        mel_raw = whisper.log_mel_spectrogram(audio_chunk)
        n_frames = int(mel_raw.shape[-1])

        if n_frames < max_frames:
            padded = audio_module.pad_or_trim(audio_chunk, length=chunk_samples)
            mel = whisper.log_mel_spectrogram(padded)
        else:
            mel = mel_raw

        mel_tensor = mel.to(device) if hasattr(mel, "to") else torch.from_numpy(mel).to(device)
        with torch.no_grad():
            encoded = model.encoder(mel_tensor.unsqueeze(0))
        encoded = encoded[0].detach().cpu().numpy()

        # Whisper encoder downsamples by 2 (stride-2 conv); length = ceil(n_frames / 2)
        out_frames = (n_frames + 1) // 2
        feats.append(encoded[:out_frames].astype(np.float32))

    if not feats:
        return np.zeros((0, 1), dtype=np.float32), duration, duration_raw
    return np.concatenate(feats, axis=0), duration, duration_raw


def l2_normalize(x: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    if x.size == 0:
        return x
    norm = np.linalg.norm(x, axis=1, keepdims=True)
    return x / (norm + eps)
