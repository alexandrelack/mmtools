"""Loudness and true peak analysis using pyloudnorm and oversampling."""

from dataclasses import dataclass

import numpy as np
import pyloudnorm as pyln
from scipy import signal


@dataclass
class LoudnessMetrics:
    """Loudness and dynamic range metrics."""
    lufs_integrated: float
    true_peak_dbtp: float
    peak_dbfs: float


def get_loudness_meter(sample_rate: int) -> pyln.Meter:
    """Get loudness meter for given sample rate."""
    try:
        return pyln.Meter(sample_rate)
    except Exception:
        # Fallback if pyloudnorm has issues
        return None


def analyze_loudness(audio: np.ndarray, sample_rate: int) -> LoudnessMetrics:
    """
    Analyze loudness (LUFS) and true peak.
    
    Args:
        audio: Shape (samples, 2) for stereo or (samples, 1) for mono
        sample_rate: Sample rate in Hz
    """
    # Ensure stereo for loudness analysis
    if audio.shape[1] == 1:
        audio_stereo = np.hstack([audio, audio])
    else:
        audio_stereo = audio
    
    # LUFS integrated loudness
    try:
        meter = get_loudness_meter(sample_rate)
        lufs = meter.integrated_loudness(audio_stereo)
        if np.isnan(lufs):
            lufs = -np.inf
    except Exception:
        lufs = -np.inf
    
    # True peak with 4x oversampling
    true_peak_dbtp = _estimate_true_peak(audio, sample_rate)
    
    # Peak in dBFS
    peak_linear = np.max(np.abs(audio))
    peak_dbfs = 20 * np.log10(np.maximum(peak_linear, 1e-10))
    
    return LoudnessMetrics(
        lufs_integrated=lufs,
        true_peak_dbtp=true_peak_dbtp,
        peak_dbfs=peak_dbfs
    )


def _estimate_true_peak(audio: np.ndarray, sample_rate: int, 
                        upsample_factor: int = 4) -> float:
    """
    Estimate true peak using polyphase resampling (4x).
    True peak can be higher than instantaneous peak due to intersample peaks.
    """
    # Upsample each channel
    audio_max = 0.0
    
    for ch in range(audio.shape[1]):
        resampled = signal.resample_poly(
            audio[:, ch], 
            up=upsample_factor, 
            down=1,
            window=('kaiser', 5.0)
        )
        ch_max = np.max(np.abs(resampled))
        audio_max = max(audio_max, ch_max)
    
    # Convert to dBTP (True Peak dB)
    true_peak_dbtp = 20 * np.log10(np.maximum(audio_max, 1e-10))
    
    return true_peak_dbtp


def estimate_headroom(lufs: float, target_lufs: float = -14.0) -> float:
    """
    Estimate headroom in dB.
    Positive value = room available before hitting target LUFS.
    Negative value = already above target.
    """
    if np.isinf(lufs) or np.isnan(lufs):
        return 0.0
    return target_lufs - lufs
