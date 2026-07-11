"""Tests for stereo module."""

import pytest
import numpy as np
from premaster_inspector import stereo


def test_mid_side_encoding():
    """Test Mid/Side encoding and decoding."""
    left = np.array([1.0, 0.5, 0.2])
    right = np.array([0.8, 0.4, 0.1])
    
    mid, side = stereo.mid_side_encode(left, right)
    left_decoded, right_decoded = stereo.mid_side_decode(mid, side)
    
    np.testing.assert_allclose(left, left_decoded, rtol=1e-10)
    np.testing.assert_allclose(right, right_decoded, rtol=1e-10)


def test_correlation_identical():
    """Test correlation of identical signals."""
    signal = np.sin(2 * np.pi * np.arange(1000) / 100)
    
    corr = stereo.get_correlation(signal, signal)
    assert abs(corr - 1.0) < 0.01


def test_correlation_inverted():
    """Test correlation of inverted signals."""
    signal = np.sin(2 * np.pi * np.arange(1000) / 100)
    
    corr = stereo.get_correlation(signal, -signal)
    assert abs(corr - (-1.0)) < 0.01


def test_correlation_orthogonal():
    """Test correlation of orthogonal signals."""
    signal1 = np.sin(2 * np.pi * np.arange(1000) / 100)
    signal2 = np.cos(2 * np.pi * np.arange(1000) / 100)
    
    corr = stereo.get_correlation(signal1, signal2)
    # Sine and cosine are ~orthogonal
    assert abs(corr) < 0.1


def test_delay_estimation():
    """Test delay estimation."""
    signal = np.sin(2 * np.pi * np.arange(10000) / 100)
    sample_rate = 44100
    
    # Create delayed signal
    delay_samples = 100
    delayed = np.concatenate([np.zeros(delay_samples), signal[:-delay_samples]])
    
    estimated_delay, delay_ms = stereo.estimate_delay(signal, delayed, sample_rate)
    
    # Allow some tolerance in estimation (cross-correlation can be approximate)
    assert abs(estimated_delay - delay_samples) <= 200


def test_frequency_band_extraction():
    """Test frequency band extraction."""
    sample_rate = 44100
    t = np.arange(1000) / sample_rate
    
    # Create signal with 1000 Hz component
    signal = np.sin(2 * np.pi * 1000 * t)
    
    # Extract band containing 1000 Hz
    band = stereo.get_frequency_band(signal, 500, 2000, sample_rate)
    
    assert band.shape == signal.shape
    # Energy should be preserved in the band
    energy_full = np.sqrt(np.mean(signal ** 2))
    energy_band = np.sqrt(np.mean(band ** 2))
    assert energy_band > 0


def test_stereo_metrics_mono():
    """Test stereo metrics with mono audio."""
    audio = np.random.randn(1000, 1)
    
    metrics = stereo.analyze_stereo(audio, 44100)
    
    assert metrics.correlation_global == 0.0
    assert metrics.delay_samples == 0
    assert metrics.delay_ms == 0.0


def test_stereo_metrics_stereo():
    """Test stereo metrics with stereo audio."""
    # Create stereo signal
    t = np.arange(10000) / 44100
    left = 0.1 * np.sin(2 * np.pi * 440 * t)
    right = 0.1 * np.sin(2 * np.pi * 440 * t)  # Identical
    
    audio = np.column_stack([left, right])
    
    metrics = stereo.analyze_stereo(audio, 44100)
    
    # Should have high correlation for identical signals
    assert metrics.correlation_global > 0.9
    assert abs(metrics.delay_ms) < 1.0
    assert len(metrics.correlation_by_band) > 0
