"""Audio I/O utilities for loading and normalizing audio files."""

from pathlib import Path
from dataclasses import dataclass

import numpy as np
import soundfile as sf


@dataclass
class AudioData:
    """Container for loaded audio and metadata."""
    samples: np.ndarray  # Shape: (n_samples, n_channels)
    sample_rate: int
    duration: float
    channel_count: int
    file_path: str


def load_audio(file_path: str | Path) -> AudioData:
    """
    Load audio file (WAV/AIFF) to float64 stereo or mono.
    
    Args:
        file_path: Path to audio file
        
    Returns:
        AudioData with samples, sample_rate, duration, channel_count
        
    Raises:
        FileNotFoundError: If file does not exist
        ValueError: If file cannot be read
    """
    file_path = Path(file_path)
    
    if not file_path.exists():
        raise FileNotFoundError(f"Audio file not found: {file_path}")
    
    try:
        samples, sample_rate = sf.read(file_path, dtype='float64')
    except Exception as e:
        raise ValueError(f"Failed to read audio file: {e}")
    
    # Ensure 2D shape (samples, channels)
    if samples.ndim == 1:
        samples = samples.reshape(-1, 1)
    
    channel_count = samples.shape[1]
    duration = samples.shape[0] / sample_rate
    
    return AudioData(
        samples=samples,
        sample_rate=sample_rate,
        duration=duration,
        channel_count=channel_count,
        file_path=str(file_path)
    )


def validate_stereo(audio: AudioData) -> bool:
    """Check if audio is stereo or warn if mono."""
    return audio.channel_count == 2
