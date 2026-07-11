"""Tests for metrics module."""

import pytest
import numpy as np
from premaster_inspector import metrics


def test_peak_dbfs():
    """Test peak calculation."""
    # Full scale
    audio = np.array([[1.0], [0.5], [-0.8]], dtype=np.float64)
    peak = metrics.get_peak_dbfs(audio, 0)
    assert abs(peak - 0.0) < 0.1  # Should be ~0 dBFS
    
    # Half scale
    audio = np.array([[0.5], [0.25]], dtype=np.float64)
    peak = metrics.get_peak_dbfs(audio, 0)
    assert abs(peak - (-6.0)) < 0.1  # Should be ~-6 dBFS


def test_dc_offset():
    """Test DC offset detection."""
    # No DC offset
    audio = np.array([[0.1], [-0.1]], dtype=np.float64)
    dc = metrics.get_dc_offset(audio, 0)
    assert abs(dc) < 0.01
    
    # Positive DC offset
    audio = np.array([[0.5], [0.5]], dtype=np.float64)
    dc = metrics.get_dc_offset(audio, 0)
    assert abs(dc - 0.5) < 0.01


def test_clipping_detection():
    """Test clipping sample counting."""
    # No clipping
    audio = np.array([[0.5, 0.5], [0.3, 0.3]], dtype=np.float64)
    clipping = metrics.analyze_clipping(audio, threshold=0.9)
    assert clipping.clipped_samples_l == 0
    assert clipping.clipped_samples_r == 0
    
    # Some clipping
    audio = np.array([[0.99999, 0.99999], [0.5, 0.5]], dtype=np.float64)
    clipping = metrics.analyze_clipping(audio, threshold=0.999)
    assert clipping.clipped_samples_l > 0


def test_crest_factor():
    """Test crest factor calculation."""
    # Sine wave
    t = np.linspace(0, 1, 1000)
    sine = 0.5 * np.sin(2 * np.pi * t)
    audio = sine.reshape(-1, 1)
    
    cf = metrics.get_crest_factor(audio, 0)
    # For sine wave, CF ≈ sqrt(2) ≈ 1.414
    assert 1.3 < cf < 1.5


def test_rms_calculation():
    """Test RMS computation."""
    # DC signal
    audio = np.array([[0.5], [0.5]], dtype=np.float64)
    rms = metrics.get_rms_dbfs(audio, 0)
    expected = 20 * np.log10(0.5)
    assert abs(rms - expected) < 0.1


def test_noise_floor_estimate():
    """Test noise floor estimation."""
    # Mostly quiet with some louder content
    audio = np.ones((1000, 1), dtype=np.float64) * 0.001
    audio[100:110, 0] = 0.5  # Loud section
    
    floor = metrics.estimate_noise_floor(audio, 0, percentile=10)
    # Should be low (around the quiet parts)
    assert floor < -30
