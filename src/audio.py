"""Audio loading, validation, and preprocessing module.

Handles audio ingestion from file paths or memory buffers, performs validation
checks (unreadable, empty, too short), converts multi-channel audio to mono,
and resamples signals to 16 kHz.
"""

from __future__ import annotations

import io
from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO, Tuple, Union

import numpy as np
import soundfile as sf
import torch
import torchaudio


class AudioValidationError(Exception):
    """Raised when an audio recording fails validation checks."""
    pass


@dataclass
class AudioData:
    """Validated and preprocessed audio container."""
    waveform: torch.Tensor  # Shape [1, time] at target_sample_rate
    sample_rate: int        # Standardized sample rate (e.g. 16000 Hz)
    duration_sec: float     # Total duration in seconds
    original_sr: int        # Original sample rate before resampling
    num_channels: int       # Original channel count before mono conversion


def load_and_validate_audio(
    audio_source: Union[str, Path, BinaryIO, bytes],
    target_sample_rate: int = 16000,
    min_duration_sec: float = 1.0,
    max_duration_sec: float = 60.0,
) -> AudioData:
    """Loads an audio file or stream, validates its contents, and normalizes it.

    Args:
        audio_source: File path, file-like object (BytesIO), or raw bytes.
        target_sample_rate: Target sample rate for the downstream model (16 kHz).
        min_duration_sec: Minimum allowed duration in seconds.
        max_duration_sec: Maximum allowed duration in seconds (to avoid excessive memory).

    Returns:
        AudioData dataclass containing normalized torch tensor and metadata.

    Raises:
        AudioValidationError: If audio cannot be decoded, is empty, or is too short.
    """
    # Wrap bytes or binary streams
    if isinstance(audio_source, bytes):
        if len(audio_source) == 0:
            raise AudioValidationError("Audio file is completely empty (0 bytes).")
        stream = io.BytesIO(audio_source)
    elif hasattr(audio_source, "read"):
        # File-like object (e.g. Streamlit UploadedFile)
        data = audio_source.read()
        if len(data) == 0:
            raise AudioValidationError("Audio file is completely empty (0 bytes).")
        stream = io.BytesIO(data)
    else:
        path = Path(audio_source)
        if not path.exists():
            raise AudioValidationError(f"Audio file does not exist: {path}")
        if path.stat().st_size == 0:
            raise AudioValidationError(f"Audio file is empty (0 bytes): {path}")
        stream = str(path)

    # Read audio data using soundfile
    try:
        data, sample_rate = sf.read(stream, dtype="float32", always_2d=True)
    except Exception as e:
        raise AudioValidationError(
            f"Unable to decode audio format. File may be corrupted or in an unsupported format. Details: {e}"
        ) from e

    num_samples, num_channels = data.shape

    if num_samples == 0:
        raise AudioValidationError("Audio file contains zero audio samples.")

    duration_sec = num_samples / float(sample_rate)

    if duration_sec < min_duration_sec:
        raise AudioValidationError(
            f"Recording is too short ({duration_sec:.2f} seconds). "
            f"Please provide at least {min_duration_sec:.1f} second(s) of audio "
            f"for reliable speaker representation."
        )

    if duration_sec > max_duration_sec:
        raise AudioValidationError(
            f"Recording is too long ({duration_sec:.2f} seconds). "
            f"Please provide an audio sample under {max_duration_sec:.0f} seconds."
        )

    # Convert to mono by averaging channels if multi-channel
    if num_channels > 1:
        mono_data = np.mean(data, axis=1)
    else:
        mono_data = data[:, 0]

    # Convert numpy array to torch tensor: shape [1, samples]
    waveform = torch.from_numpy(mono_data).unsqueeze(0)

    # Resample if sample rate doesn't match target_sample_rate
    if sample_rate != target_sample_rate:
        try:
            resampler = torchaudio.transforms.Resample(
                orig_freq=sample_rate, new_freq=target_sample_rate
            )
            waveform = resampler(waveform)
        except Exception as e:
            raise AudioValidationError(
                f"Failed to resample audio from {sample_rate} Hz to {target_sample_rate} Hz: {e}"
            ) from e

    # Ensure waveform is 2D: [1, time]
    if waveform.dim() == 1:
        waveform = waveform.unsqueeze(0)

    return AudioData(
        waveform=waveform,
        sample_rate=target_sample_rate,
        duration_sec=duration_sec,
        original_sr=sample_rate,
        num_channels=num_channels,
    )
