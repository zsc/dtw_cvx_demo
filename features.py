from typing import Tuple

import numpy as np
import torch


def _load_whisper_module():
    try:
        import whisper
    except Exception as exc:  # pragma: no cover - dependency errors
        raise RuntimeError(
            "Failed to import openai-whisper. Ensure it is installed in the active environment."
        ) from exc
    return whisper


def _load_audio(wav_path: str, sample_rate: int = 16000) -> Tuple[np.ndarray, float]:
    whisper = _load_whisper_module()
    audio = whisper.load_audio(wav_path)
    if audio.size == 0:
        return audio.astype(np.float32), 0.0
    duration = float(audio.shape[0]) / float(sample_rate)
    return audio.astype(np.float32), duration


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
) -> Tuple[np.ndarray, float]:
    """
    Returns (features, duration_sec).
    Features are float32 of shape (T, d).
    """
    whisper = _load_whisper_module()

    audio, duration = _load_audio(wav_path)
    if audio.size == 0:
        return np.zeros((0, 1), dtype=np.float32), duration

    if mode == "log_mel":
        mel = _log_mel(audio)
        # Transpose to (T, d)
        return mel.T.astype(np.float32), duration

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
        return np.zeros((0, 1), dtype=np.float32), duration
    return np.concatenate(feats, axis=0), duration


def l2_normalize(x: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    if x.size == 0:
        return x
    norm = np.linalg.norm(x, axis=1, keepdims=True)
    return x / (norm + eps)
