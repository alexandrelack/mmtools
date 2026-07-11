"""Basic audio metrics: peak, RMS, DC offset, clipping."""

from dataclasses import dataclass

import numpy as np


@dataclass
class ChannelMetrics:
    """Metrics for a single channel."""
    peak_dbfs: float
    rms_dbfs: float
    dc_offset: float
    crest_factor: float


@dataclass
class ClippingMetrics:
    """Clipping analysis."""
    clipped_samples_l: int
    clipped_samples_r: int
    clipping_ratio_l: float  # 0.0-1.0
    clipping_ratio_r: float


def dbfs(value: float, epsilon: float = 1e-10) -> float:
    """Convert linear amplitude to dBFS, safe for zero."""
    return 20 * np.log10(np.maximum(np.abs(value), epsilon))


def get_peak_dbfs(audio: np.ndarray, channel: int) -> float:
    """Peak amplitude in dBFS for a channel."""
    peak = np.max(np.abs(audio[:, channel]))
    return dbfs(peak)


def get_rms_dbfs(audio: np.ndarray, channel: int) -> float:
    """RMS amplitude in dBFS for a channel."""
    rms = np.sqrt(np.mean(audio[:, channel] ** 2))
    return dbfs(rms)


def get_dc_offset(audio: np.ndarray, channel: int) -> float:
    """DC offset (mean) as raw value."""
    return np.mean(audio[:, channel])


def get_crest_factor(audio: np.ndarray, channel: int) -> float:
    """Crest factor: peak / RMS."""
    peak = np.max(np.abs(audio[:, channel]))
    rms = np.sqrt(np.mean(audio[:, channel] ** 2))
    if rms < 1e-10:
        return 0.0
    return peak / rms


def analyze_channel_metrics(audio: np.ndarray, channel: int) -> ChannelMetrics:
    """Analyze metrics for a single channel."""
    return ChannelMetrics(
        peak_dbfs=get_peak_dbfs(audio, channel),
        rms_dbfs=get_rms_dbfs(audio, channel),
        dc_offset=get_dc_offset(audio, channel),
        crest_factor=get_crest_factor(audio, channel)
    )


def analyze_clipping(audio: np.ndarray, threshold: float = 0.9999) -> ClippingMetrics:
    """
    Count samples above clipping threshold per channel.
    
    Args:
        audio: Shape (samples, channels)
        threshold: Amplitude threshold (default near digital ceiling)
    """
    clipped_l = np.sum(np.abs(audio[:, 0]) > threshold)
    clipped_r = np.sum(np.abs(audio[:, 1]) > threshold) if audio.shape[1] > 1 else 0
    
    total_samples = audio.shape[0]
    
    return ClippingMetrics(
        clipped_samples_l=int(clipped_l),
        clipped_samples_r=int(clipped_r),
        clipping_ratio_l=float(clipped_l / total_samples),
        clipping_ratio_r=float(clipped_r / total_samples) if audio.shape[1] > 1 else 0.0
    )


def estimate_noise_floor(audio: np.ndarray, channel: int, 
                         percentile: float = 10.0) -> float:
    """
    Estimate noise floor as low percentile of abs values.
    Represents typical background noise/silence level.
    """
    abs_audio = np.abs(audio[:, channel])
    floor_linear = np.percentile(abs_audio, percentile)
    return dbfs(floor_linear)
