"""Stereo analysis: Mid/Side, correlation, delay, phase."""

from dataclasses import dataclass, field

import numpy as np
from scipy import signal


@dataclass
class StereoMetrics:
    """Stereo field metrics."""
    correlation_global: float
    delay_samples: int
    delay_ms: float
    mid_rms_dbfs: float
    side_rms_dbfs: float
    correlation_by_band: dict = field(default_factory=dict)  # band_name -> correlation
    energy_mid_by_band: dict = field(default_factory=dict)   # band_name -> dBFS
    energy_side_by_band: dict = field(default_factory=dict)  # band_name -> dBFS


# Standard frequency bands for analysis
FREQ_BANDS = [
    (20, 40, "20-40 Hz"),
    (40, 80, "40-80 Hz"),
    (80, 160, "80-160 Hz"),
    (160, 300, "160-300 Hz"),
    (300, 600, "300-600 Hz"),
    (600, 1200, "600-1.2k"),
    (1200, 2500, "1.2-2.5k"),
    (2500, 5000, "2.5-5k"),
    (5000, 10000, "5-10k"),
    (10000, 20000, "10-20k"),
]


def mid_side_encode(left: np.ndarray, right: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Convert L/R to Mid/Side."""
    mid = (left + right) / 2.0
    side = (left - right) / 2.0
    return mid, side


def mid_side_decode(mid: np.ndarray, side: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Convert Mid/Side back to L/R."""
    left = mid + side
    right = mid - side
    return left, right


def get_correlation(left: np.ndarray, right: np.ndarray) -> float:
    """
    Compute correlation coefficient between L and R channels.
    Range: -1 (opposite) to +1 (identical)
    """
    if len(left) < 2 or len(right) < 2:
        return 0.0
    
    # Normalize
    left_norm = (left - np.mean(left)) / (np.std(left) + 1e-10)
    right_norm = (right - np.mean(right)) / (np.std(right) + 1e-10)
    
    correlation = np.mean(left_norm * right_norm)
    return float(np.clip(correlation, -1.0, 1.0))


def estimate_delay(left: np.ndarray, right: np.ndarray, 
                   sample_rate: int, max_delay_ms: float = 20.0) -> tuple[int, float]:
    """
    Estimate delay between L and R using cross-correlation.
    
    Returns:
        (delay_samples, delay_ms)
        Positive delay = R is delayed relative to L
    """
    max_delay_samples = int((max_delay_ms / 1000.0) * sample_rate)
    
    # Compute cross-correlation
    correlation = signal.correlate(left, right, mode='same')
    delay_samples = signal.argrelextrema(correlation, np.greater)[0]
    
    if len(delay_samples) == 0:
        # Find maximum correlation
        max_idx = np.argmax(np.abs(correlation))
    else:
        max_idx = delay_samples[np.argmax(np.abs(correlation[delay_samples]))]
    
    # Convert from correlation center index to actual delay
    center = len(correlation) // 2
    delay = max_idx - center
    
    # Limit to reasonable range
    delay = np.clip(delay, -max_delay_samples, max_delay_samples)
    delay_ms = (delay / sample_rate) * 1000.0
    
    return int(delay), float(delay_ms)


def get_frequency_band(audio: np.ndarray, low_freq: float, high_freq: float,
                       sample_rate: int) -> np.ndarray:
    """Extract frequency band using Butterworth filter."""
    order = 4
    nyquist = sample_rate / 2
    
    # Normalize frequencies to [0, 1]
    low_norm = low_freq / nyquist
    high_norm = high_freq / nyquist
    
    # Clamp to valid range
    low_norm = np.clip(low_norm, 0.001, 0.999)
    high_norm = np.clip(high_norm, 0.001, 0.999)
    
    if low_norm >= high_norm:
        return audio.copy()
    
    try:
        b, a = signal.butter(order, [low_norm, high_norm], btype='band')
        filtered = signal.filtfilt(b, a, audio)
        return filtered
    except Exception:
        return audio.copy()


def analyze_stereo_correlation_by_band(audio: np.ndarray, 
                                      sample_rate: int) -> dict[str, float]:
    """Compute correlation for each frequency band."""
    correlations = {}
    
    for low_freq, high_freq, band_name in FREQ_BANDS:
        left_band = get_frequency_band(audio[:, 0], low_freq, high_freq, sample_rate)
        right_band = get_frequency_band(audio[:, 1], low_freq, high_freq, sample_rate)
        
        corr = get_correlation(left_band, right_band)
        correlations[band_name] = corr
    
    return correlations


def analyze_energy_by_band(audio: np.ndarray, sample_rate: int) -> dict[str, float]:
    """Compute RMS energy for each frequency band (in dBFS)."""
    energies = {}
    
    for low_freq, high_freq, band_name in FREQ_BANDS:
        for ch in range(audio.shape[1]):
            band = get_frequency_band(audio[:, ch], low_freq, high_freq, sample_rate)
            rms = np.sqrt(np.mean(band ** 2))
            energy_dbfs = 20 * np.log10(np.maximum(rms, 1e-10))
            key = f"{band_name}_Ch{ch + 1}"
            energies[key] = energy_dbfs
    
    return energies


def analyze_mid_side_by_band(audio: np.ndarray, 
                             sample_rate: int) -> tuple[dict[str, float], dict[str, float]]:
    """Analyze Mid/Side energy by band."""
    left = audio[:, 0]
    right = audio[:, 1] if audio.shape[1] > 1 else left
    
    mid, side = mid_side_encode(left, right)
    
    mid_energies = {}
    side_energies = {}
    
    for low_freq, high_freq, band_name in FREQ_BANDS:
        mid_band = get_frequency_band(mid, low_freq, high_freq, sample_rate)
        side_band = get_frequency_band(side, low_freq, high_freq, sample_rate)
        
        mid_rms = np.sqrt(np.mean(mid_band ** 2))
        side_rms = np.sqrt(np.mean(side_band ** 2))
        
        mid_energies[band_name] = 20 * np.log10(np.maximum(mid_rms, 1e-10))
        side_energies[band_name] = 20 * np.log10(np.maximum(side_rms, 1e-10))
    
    return mid_energies, side_energies


def analyze_stereo(audio: np.ndarray, sample_rate: int) -> StereoMetrics:
    """Complete stereo analysis."""
    if audio.shape[1] < 2:
        # Mono - return default metrics
        return StereoMetrics(
            correlation_global=0.0,
            delay_samples=0,
            delay_ms=0.0,
            mid_rms_dbfs=-np.inf,
            side_rms_dbfs=-np.inf
        )
    
    left = audio[:, 0]
    right = audio[:, 1]
    
    # Global correlation
    correlation_global = get_correlation(left, right)
    
    # Delay estimation
    delay_samples, delay_ms = estimate_delay(left, right, sample_rate)
    
    # Mid/Side
    mid, side = mid_side_encode(left, right)
    mid_rms = np.sqrt(np.mean(mid ** 2))
    side_rms = np.sqrt(np.mean(side ** 2))
    mid_rms_dbfs = 20 * np.log10(np.maximum(mid_rms, 1e-10))
    side_rms_dbfs = 20 * np.log10(np.maximum(side_rms, 1e-10))
    
    # By-band analysis
    corr_by_band = analyze_stereo_correlation_by_band(audio, sample_rate)
    mid_by_band, side_by_band = analyze_mid_side_by_band(audio, sample_rate)
    
    return StereoMetrics(
        correlation_global=correlation_global,
        delay_samples=delay_samples,
        delay_ms=delay_ms,
        mid_rms_dbfs=mid_rms_dbfs,
        side_rms_dbfs=side_rms_dbfs,
        correlation_by_band=corr_by_band,
        energy_mid_by_band=mid_by_band,
        energy_side_by_band=side_by_band
    )
