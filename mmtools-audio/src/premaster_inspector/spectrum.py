"""Spectrum analysis: frequency content, hum, infrasonic, ultrasonic."""

from dataclasses import dataclass

import numpy as np
from scipy import signal, fft


@dataclass
class SpectrumMetrics:
    """Frequency domain metrics."""
    infrasonic_energy_dbfs: float  # <20 Hz
    ultrasonic_energy_dbfs: float  # >Nyquist - 1 kHz
    hum_50hz_dbfs: float
    hum_60hz_dbfs: float
    hum_50hz_harmonics: list[tuple[float, float]]  # (freq, dbfs)
    hum_60hz_harmonics: list[tuple[float, float]]


def compute_spectrum(audio: np.ndarray, sample_rate: int,
                     nperseg: int = 4096) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute average power spectrum across channels.
    
    Returns:
        (frequencies, power_dbfs)
    """
    # Average across channels
    audio_mono = np.mean(audio, axis=1)
    
    # Adjust nperseg if signal is too short
    nperseg = min(nperseg, len(audio_mono))
    
    # Welch's method for smooth spectrum
    freqs, pxx = signal.welch(audio_mono, fs=sample_rate, nperseg=nperseg)
    
    # Convert to dBFS
    pxx_dbfs = 10 * np.log10(np.maximum(pxx, 1e-10))
    
    return freqs, pxx_dbfs


def get_energy_in_range(freqs: np.ndarray, power_dbfs: np.ndarray,
                        low_freq: float, high_freq: float) -> float:
    """Get average energy in frequency range."""
    mask = (freqs >= low_freq) & (freqs <= high_freq)
    if not np.any(mask):
        return -np.inf
    
    # Average energy in range
    energy = np.mean(power_dbfs[mask])
    return float(energy)


def detect_hum(freqs: np.ndarray, power_dbfs: np.ndarray, 
               fundamental: float, threshold_db: float = -40.0) -> list[tuple[float, float]]:
    """
    Detect hum and harmonics above background.
    
    Returns:
        List of (frequency, amplitude_dbfs) for detected components
    """
    results = []
    nyquist = freqs[-1]
    
    # Check up to 5 harmonics or Nyquist
    for harmonic in range(1, 6):
        target_freq = fundamental * harmonic
        if target_freq > nyquist * 0.95:
            break
        
        # Find bin closest to target
        idx = np.argmin(np.abs(freqs - target_freq))
        actual_freq = freqs[idx]
        actual_power = power_dbfs[idx]
        
        # Simple threshold test
        if actual_power > threshold_db:
            results.append((float(actual_freq), float(actual_power)))
    
    return results


def estimate_aliasing_risk(freqs: np.ndarray, power_dbfs: np.ndarray,
                           sample_rate: int, 
                           threshold_db: float = -30.0) -> bool:
    """
    Detect potential aliasing as elevated energy near Nyquist.
    This is a heuristic; true aliasing detection requires time-domain analysis.
    """
    nyquist = sample_rate / 2
    # Check upper 5% of spectrum
    high_freq_start = nyquist * 0.95
    mask = freqs >= high_freq_start
    
    if not np.any(mask):
        return False
    
    max_power = np.max(power_dbfs[mask])
    return max_power > threshold_db


def analyze_spectrum(audio: np.ndarray, sample_rate: int) -> SpectrumMetrics:
    """Complete spectrum analysis."""
    freqs, power_dbfs = compute_spectrum(audio, sample_rate)
    
    nyquist = sample_rate / 2
    
    # Infrasonic (<20 Hz)
    infra_energy = get_energy_in_range(freqs, power_dbfs, 0, 20)
    
    # Ultrasonic (upper 1 kHz)
    ultra_energy = get_energy_in_range(freqs, power_dbfs, nyquist - 1000, nyquist)
    
    # Hum detection
    hum_50hz = detect_hum(freqs, power_dbfs, 50.0)
    hum_60hz = detect_hum(freqs, power_dbfs, 60.0)
    
    return SpectrumMetrics(
        infrasonic_energy_dbfs=infra_energy,
        ultrasonic_energy_dbfs=ultra_energy,
        hum_50hz_dbfs=hum_50hz[0][1] if hum_50hz else -np.inf,
        hum_60hz_dbfs=hum_60hz[0][1] if hum_60hz else -np.inf,
        hum_50hz_harmonics=hum_50hz,
        hum_60hz_harmonics=hum_60hz
    )
