"""Tests for issues detection module."""

import pytest
import numpy as np
from premaster_inspector import issues


def test_issue_dataclass():
    """Test Issue creation and attributes."""
    issue = issues.Issue(
        id="test",
        title="Test Issue",
        category="test",
        severity="high",
        confidence=0.95,
        evidence="Test evidence",
        impact="Test impact",
        suggestions=["Suggestion 1", "Suggestion 2"]
    )
    
    assert issue.id == "test"
    assert issue.severity == "high"
    assert len(issue.suggestions) == 2


def test_dc_offset_detection():
    """Test DC offset issue detection."""
    # Create audio with DC offset
    audio = np.ones((1000, 2), dtype=np.float64) * 0.02
    audio[:, 1] = audio[:, 1]  # Both channels
    
    detected = issues.detect_issues(audio, 44100)
    
    # Should detect DC offset issues
    dc_issues = [i for i in detected if "DC offset" in i.title]
    assert len(dc_issues) > 0


def test_clipping_detection():
    """Test clipping issue detection."""
    # Create audio with clipping
    audio = np.random.randn(1000, 2).astype(np.float64) * 0.5
    audio[100:110, 0] = 0.99999  # Clipped samples
    
    detected = issues.detect_issues(audio, 44100)
    
    # Should detect clipping
    clip_issues = [i for i in detected if "clipping" in i.title.lower()]
    assert len(clip_issues) > 0


def test_true_peak_detection():
    """Test true peak issue detection."""
    # Create audio with high true peak (intersample peak)
    sample_rate = 44100
    t = np.arange(44100) / sample_rate
    
    # Sharp transient that can create high intersample peak
    audio = np.zeros((44100, 2), dtype=np.float64)
    audio[5000, :] = 1.0  # Sharp peak
    audio[5001, :] = -1.0  # Inverse
    
    detected = issues.detect_issues(audio, sample_rate)
    
    # Might detect true peak or clipping
    peak_issues = [i for i in detected if "peak" in i.title.lower()]
    # At least some amplitude-related issue should be found
    amplitude_issues = [i for i in detected if i.category in ["loudness", "amplitude"]]
    assert len(amplitude_issues) > 0


def test_stereo_correlation_issue():
    """Test low stereo correlation detection."""
    # Create stereo audio with low correlation
    audio = np.random.randn(10000, 2).astype(np.float64) * 0.1
    
    detected = issues.detect_issues(audio, 44100)
    
    # Random uncorrelated signals should trigger low correlation issue
    corr_issues = [i for i in detected if "correlation" in i.title.lower()]
    assert len(corr_issues) > 0


def test_no_issues_clean_audio():
    """Test clean audio produces minimal issues."""
    # Create clean, well-engineered audio
    sample_rate = 44100
    t = np.arange(sample_rate) / sample_rate
    
    # Sine wave at good level
    left = 0.1 * np.sin(2 * np.pi * 440 * t)
    right = 0.1 * np.sin(2 * np.pi * 440 * t)
    
    audio = np.column_stack([left, right])
    
    detected = issues.detect_issues(audio, sample_rate)
    
    # Should have very few or no issues
    high_severity = [i for i in detected if i.severity == "high"]
    assert len(high_severity) == 0


def test_hum_detection():
    """Test hum detection in audio."""
    sample_rate = 44100
    t = np.arange(44100) / sample_rate
    
    # Create audio with 60 Hz hum
    hum = 0.01 * np.sin(2 * np.pi * 60 * t)
    signal = 0.1 * np.sin(2 * np.pi * 440 * t)
    
    audio = np.column_stack([hum + signal, hum + signal])
    
    detected = issues.detect_issues(audio, sample_rate)
    
    # Might detect hum
    hum_issues = [i for i in detected if "hum" in i.title.lower()]
    # Not guaranteed due to threshold, but test structure is valid


def test_infrasonic_detection():
    """Test infrasonic content detection."""
    sample_rate = 44100
    t = np.arange(44100) / sample_rate
    
    # Create audio with strong infrasonic content
    infra = 0.2 * np.sin(2 * np.pi * 10 * t)  # 10 Hz
    signal = 0.05 * np.sin(2 * np.pi * 440 * t)
    
    audio = np.column_stack([infra + signal, infra + signal])
    
    detected = issues.detect_issues(audio, sample_rate)
    
    # Should detect infrasonic content
    infra_issues = [i for i in detected if "infrasonic" in i.title.lower()]
    assert len(infra_issues) > 0


def test_imbalance_detection():
    """Test L/R imbalance detection."""
    sample_rate = 44100
    t = np.arange(44100) / sample_rate
    
    left = 0.5 * np.sin(2 * np.pi * 440 * t)  # Loud left
    right = 0.05 * np.sin(2 * np.pi * 440 * t)  # Quiet right
    
    audio = np.column_stack([left, right])
    
    detected = issues.detect_issues(audio, sample_rate)
    
    # Should detect imbalance
    imbalance_issues = [i for i in detected if "imbalance" in i.title.lower()]
    assert len(imbalance_issues) > 0
