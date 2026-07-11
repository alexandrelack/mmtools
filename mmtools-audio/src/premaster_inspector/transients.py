"""Transient and onset detection."""

from dataclasses import dataclass

import numpy as np
from scipy import signal


@dataclass
class TransientMetrics:
    """Transient analysis."""
    outlier_count: int
    outlier_ratio: float  # ratio of outliers to total transients
    max_outlier_factor: float  # how many dB above mean
    pre_ring_detected: bool
    pre_ring_confidence: float


def compute_onset_strength(audio: np.ndarray, sr: int,
                           hop_length: int = 512) -> np.ndarray:
    """
    Compute onset strength curve (envelope-based).
    Higher values indicate transient activity.
    """
    # High-pass filter to emphasize transients
    sos = signal.butter(4, 200, 'high', fs=sr, output='sos')
    filtered = signal.sosfilt(sos, audio)
    
    # Compute energy envelope
    frame_length = hop_length * 2
    n_frames = len(audio) // hop_length
    onset_strength = np.zeros(n_frames)
    
    for i in range(n_frames):
        frame = filtered[i * hop_length:(i + 1) * hop_length + frame_length]
        if len(frame) > 0:
            onset_strength[i] = np.sqrt(np.mean(frame ** 2))
    
    return onset_strength


def detect_transient_outliers(audio: np.ndarray, sr: int,
                              threshold_factor: float = 2.5) -> tuple[int, float, float]:
    """
    Detect transients that are outliers in amplitude.
    
    Returns:
        (outlier_count, outlier_ratio, max_outlier_factor)
    """
    hop_length = 512
    onset_strength = compute_onset_strength(audio, sr, hop_length)
    
    if len(onset_strength) < 2:
        return 0, 0.0, 0.0
    
    # Find peaks
    peaks, properties = signal.find_peaks(onset_strength, height=0)
    
    if len(peaks) == 0:
        return 0, 0.0, 0.0
    
    heights = properties['peak_heights']
    mean_height = np.mean(heights)
    
    if mean_height < 1e-10:
        return 0, 0.0, 0.0
    
    # Find outliers
    outliers = heights > (mean_height * threshold_factor)
    outlier_count = np.sum(outliers)
    outlier_ratio = float(outlier_count / len(heights))
    
    if outlier_count > 0:
        max_factor = np.max(heights[outliers]) / mean_height
    else:
        max_factor = 0.0
    
    return int(outlier_count), float(outlier_ratio), float(max_factor)


def detect_pre_ringing(audio: np.ndarray, sr: int,
                       analysis_window_ms: float = 10.0) -> tuple[bool, float]:
    """
    Heuristic detection of probable pre-ringing.
    Looks for unusual energy before strong transients.
    
    Returns:
        (detected, confidence)
    """
    hop_length = 256
    window_samples = int((analysis_window_ms / 1000.0) * sr)
    
    # Compute onset strength
    onset_strength = compute_onset_strength(audio, sr, hop_length)
    
    if len(onset_strength) < 4:
        return False, 0.0
    
    # Find strong peaks
    mean_strength = np.mean(onset_strength)
    std_strength = np.std(onset_strength)
    
    if std_strength < 1e-10:
        return False, 0.0
    
    threshold = mean_strength + (2.0 * std_strength)
    peaks, _ = signal.find_peaks(onset_strength, height=threshold)
    
    if len(peaks) < 2:
        return False, 0.0
    
    # Check for energy before peaks
    pre_ring_count = 0
    
    for peak_idx in peaks[:20]:  # Check first 20 peaks
        if peak_idx < 2:
            continue
        
        # Energy before peak vs energy at peak
        peak_energy = onset_strength[peak_idx]
        pre_energy = np.mean(onset_strength[max(0, peak_idx - 2):peak_idx])
        
        # If significant pre-peak energy, might be pre-ringing
        if pre_energy > (peak_energy * 0.3):
            pre_ring_count += 1
    
    # Confidence based on detection frequency
    confidence = min(0.7, float(pre_ring_count / max(1, len(peaks))))
    detected = pre_ring_count >= 2
    
    return detected, confidence


def analyze_transients(audio: np.ndarray, sr: int) -> TransientMetrics:
    """Complete transient analysis."""
    outlier_count, outlier_ratio, max_factor = detect_transient_outliers(audio, sr)
    pre_ring_detected, pre_ring_conf = detect_pre_ringing(audio, sr)
    
    return TransientMetrics(
        outlier_count=outlier_count,
        outlier_ratio=outlier_ratio,
        max_outlier_factor=max_factor,
        pre_ring_detected=pre_ring_detected,
        pre_ring_confidence=pre_ring_conf
    )
