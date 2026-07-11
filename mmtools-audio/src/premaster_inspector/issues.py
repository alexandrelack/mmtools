"""Issue detection logic and Issue dataclass."""

from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from . import metrics, loudness, stereo, spectrum, transients


@dataclass
class Issue:
    """A detected technical issue in audio."""
    id: str
    title: str
    category: str  # e.g., "loudness", "stereo", "frequency", "transients"
    severity: str  # "low", "medium", "high"
    confidence: float  # 0.0 to 1.0
    frequency_range: Optional[tuple[float, float]] = None  # (low, high) in Hz
    evidence: Optional[str] = None  # Numerical evidence
    impact: Optional[str] = None  # What this means for mastering
    suggestions: list[str] = field(default_factory=list)


def detect_issues(audio_data, sample_rate: int) -> list[Issue]:
    """
    Detect all technical issues in audio.
    
    Args:
        audio_data: Shape (samples, channels)
        sample_rate: Sample rate in Hz
        
    Returns:
        List of Issue objects
    """
    issues = []
    
    # 1. Compute all metrics
    metrics_l = metrics.analyze_channel_metrics(audio_data, 0)
    metrics_r = metrics.analyze_channel_metrics(audio_data, 1) if audio_data.shape[1] > 1 else metrics_l
    
    clipping_metrics = metrics.analyze_clipping(audio_data)
    
    loud_metrics = loudness.analyze_loudness(audio_data, sample_rate)
    
    stereo_metrics = stereo.analyze_stereo(audio_data, sample_rate)
    
    spec_metrics = spectrum.analyze_spectrum(audio_data, sample_rate)
    
    trans_metrics = transients.analyze_transients(
        np.mean(audio_data, axis=1), sample_rate
    )
    
    # 2. Generate issues
    
    # DC Offset
    dc_threshold = 0.01
    if abs(metrics_l.dc_offset) > dc_threshold:
        issues.append(Issue(
            id="dc_offset_l",
            title="DC offset detected on Left channel",
            category="amplitude",
            severity="medium",
            confidence=0.9,
            evidence=f"DC offset: {metrics_l.dc_offset:.6f}",
            impact="DC offset wastes headroom and can cause clicks on export/format changes",
            suggestions=[
                "Use a high-pass filter (linear-phase) or DC offset removal plugin",
                "Check for faulty audio interface or preamp",
                "Verify mix bus output meter for DC bias"
            ]
        ))
    
    if audio_data.shape[1] > 1 and abs(metrics_r.dc_offset) > dc_threshold:
        issues.append(Issue(
            id="dc_offset_r",
            title="DC offset detected on Right channel",
            category="amplitude",
            severity="medium",
            confidence=0.9,
            evidence=f"DC offset: {metrics_r.dc_offset:.6f}",
            impact="DC offset wastes headroom and can cause clicks on export/format changes",
            suggestions=[
                "Use a high-pass filter (linear-phase) or DC offset removal plugin",
                "Check for faulty audio interface or preamp",
                "Verify mix bus output meter for DC bias"
            ]
        ))
    
    # Clipping
    if clipping_metrics.clipping_ratio_l > 0.0:
        issues.append(Issue(
            id="clipping_l",
            title="Digital clipping detected on Left channel",
            category="amplitude",
            severity="high",
            confidence=0.99,
            evidence=f"{clipping_metrics.clipped_samples_l} clipped samples ({clipping_metrics.clipping_ratio_l*100:.3f}%)",
            impact="Clipping introduces harmonic distortion and cannot be recovered",
            suggestions=[
                "Reduce gain before master bus or mix volume",
                "Check for gain staging issues in mixer",
                "Review individual track levels and automation",
                "Render/bounce with headroom safety margin"
            ]
        ))
    
    if audio_data.shape[1] > 1 and clipping_metrics.clipping_ratio_r > 0.0:
        issues.append(Issue(
            id="clipping_r",
            title="Digital clipping detected on Right channel",
            category="amplitude",
            severity="high",
            confidence=0.99,
            evidence=f"{clipping_metrics.clipped_samples_r} clipped samples ({clipping_metrics.clipping_ratio_r*100:.3f}%)",
            impact="Clipping introduces harmonic distortion and cannot be recovered",
            suggestions=[
                "Reduce gain before master bus or mix volume",
                "Check for gain staging issues in mixer",
                "Review individual track levels and automation",
                "Render/bounce with headroom safety margin"
            ]
        ))
    
    # True Peak
    if loud_metrics.true_peak_dbtp > -1.0:
        issues.append(Issue(
            id="true_peak_high",
            title="True Peak exceeds -1 dBTP limit",
            category="loudness",
            severity="high",
            confidence=0.95,
            evidence=f"True Peak: {loud_metrics.true_peak_dbtp:.2f} dBTP",
            impact="Intersample peaks may cause issues in playback and format conversions",
            suggestions=[
                "Add a limiter (minimum headroom 1 dB true peak before export)",
                "Use linear-phase limiter to avoid pre-ringing artifacts",
                "Reduce mix level slightly and remix if needed",
                "Check for spurious peaks from oversampled plugins"
            ]
        ))
    
    # Headroom
    if not np.isinf(loud_metrics.lufs_integrated):
        headroom = loudness.estimate_headroom(loud_metrics.lufs_integrated, target_lufs=-14.0)
        if headroom < 1.0 and headroom > -np.inf:
            severity = "low" if headroom > 0.5 else "medium"
            issues.append(Issue(
                id="low_headroom",
                title="Limited headroom available for mastering",
                category="loudness",
                severity=severity,
                confidence=0.8,
                evidence=f"LUFS: {loud_metrics.lufs_integrated:.2f}, Headroom: {headroom:.2f} dB vs -14 LUFS",
                impact="Very limited room for EQ, compression, or loudness adjustment in mastering",
                suggestions=[
                    "Consider remixing with higher level ceiling (lower loudness target)",
                    "Reduce competing elements or use parallel compression more subtly",
                    "Ensure proper gain staging before master bus",
                    "Discuss loudness expectations with mastering engineer"
                ]
            ))
    
    # Stereo Issues
    if audio_data.shape[1] > 1:
        
        # Low correlation
        if stereo_metrics.correlation_global < 0.5:
            issues.append(Issue(
                id="low_correlation_global",
                title="Low global L/R correlation",
                category="stereo",
                severity="low",
                confidence=0.7,
                evidence=f"Correlation: {stereo_metrics.correlation_global:.3f}",
                impact="May indicate extreme stereo image or potential phase issues",
                suggestions=[
                    "Review stereo width and panning consistency",
                    "Check phase relationships between L and R buses",
                    "Verify no accidental invertion or phase cancellation",
                    "Ensure drum and bass are not excessively stereo-wide"
                ]
            ))
        
        # Delay between channels
        if abs(stereo_metrics.delay_ms) > 2.0:
            issues.append(Issue(
                id="channel_delay",
                title=f"Possible relative delay between channels",
                category="stereo",
                severity="medium",
                confidence=0.7,
                evidence=f"Estimated delay: {stereo_metrics.delay_ms:.2f} ms ({stereo_metrics.delay_samples} samples)",
                impact="May cause phase anomalies or comb filtering when summed to mono",
                suggestions=[
                    "Check for latency difference between plugin chains on L/R",
                    "Review automation or processing delays on different channels",
                    "Verify parallel bus latencies are aligned",
                    "Use delay compensation on mixed tracks if needed"
                ]
            ))
        
        # Excessive stereo in low frequencies
        low_freq_bands = ["20-40 Hz", "40-80 Hz"]
        excessive_stereo_low = False
        for band in low_freq_bands:
            if band in stereo_metrics.energy_side_by_band:
                side_energy = stereo_metrics.energy_side_by_band[band]
                mid_energy = stereo_metrics.energy_mid_by_band.get(band, -np.inf)
                if side_energy > mid_energy - 6:  # Side within 6dB of Mid
                    excessive_stereo_low = True
                    issues.append(Issue(
                        id=f"excessive_stereo_{band}",
                        title=f"Excessive stereo width in {band}",
                        category="stereo",
                        severity="high",
                        confidence=0.8,
                        frequency_range=(int(band.split('-')[0]), int(band.split('-')[1].split()[0])),
                        evidence=f"Mid: {mid_energy:.1f} dBFS, Side: {side_energy:.1f} dBFS",
                        impact="Wide stereo bass can cause phase issues in mono and playback problems",
                        suggestions=[
                            "Use Mid/Side EQ to reduce side content in low frequencies",
                            "Consider collapsing bass below 80 Hz to mono in mastering",
                            "Review stereo imaging on bass instruments (kicks, bass guitar)",
                            "Use a correlation meter to verify mono compatibility"
                        ]
                    ))
    
    # Frequency Domain Issues
    
    # Infrasonic
    if spec_metrics.infrasonic_energy_dbfs > -70.0:
        issues.append(Issue(
            id="infrasonic_content",
            title="Elevated infrasonic content (<20 Hz)",
            category="frequency",
            severity="medium",
            confidence=0.7,
            frequency_range=(0, 20),
            evidence=f"Infrasonic energy: {spec_metrics.infrasonic_energy_dbfs:.1f} dBFS",
            impact="Infrasonic rumble wastes loudness headroom and speaker resources",
            suggestions=[
                "Apply high-pass filter below 20 Hz (steep, linear-phase",
                "Check for rumble from air conditioning, traffic, or vibration",
                "Use subsonic filter in mastering chain",
                "Ensure microphones/preamps don't have low-freq noise"
            ]
        ))
    
    # Ultrasonic
    if spec_metrics.ultrasonic_energy_dbfs > -60.0:
        issues.append(Issue(
            id="ultrasonic_content",
            title="Elevated ultrasonic content near Nyquist",
            category="frequency",
            severity="low",
            confidence=0.6,
            frequency_range=(sample_rate // 2 - 1000, sample_rate // 2),
            evidence=f"Ultrasonic energy (Nyquist -1kHz): {spec_metrics.ultrasonic_energy_dbfs:.1f} dBFS",
            impact="May indicate aliasing artifacts or noise; wastes loudness in invisible range",
            suggestions=[
                "Use anti-aliasing low-pass filter above 18-20 kHz",
                "Check for aliasing from resampling or aggressive EQ",
                "Verify no unintended high-frequency noise or hum harmonics",
                "Consider if ultra-high-frequency content is intentional"
            ]
        ))
    
    # Hum
    if len(spec_metrics.hum_50hz_harmonics) > 0 and spec_metrics.hum_50hz_dbfs > -50.0:
        issues.append(Issue(
            id="hum_50hz",
            title="Possible 50 Hz hum and harmonics",
            category="frequency",
            severity="medium",
            confidence=0.7,
            frequency_range=(50, 500),
            evidence=f"50 Hz: {spec_metrics.hum_50hz_dbfs:.1f} dBFS, Harmonics detected: {len(spec_metrics.hum_50hz_harmonics)}",
            impact="Audible hum reduces clarity and perceived professionalism",
            suggestions=[
                "Use notch filter or parametric EQ to remove 50 Hz and harmonics",
                "Check for ground loop (XLR cables, power supplies, USB grounding)",
                "Verify recording location for AC mains interference",
                "Use hum removal plugin if necessary"
            ]
        ))
    
    if len(spec_metrics.hum_60hz_harmonics) > 0 and spec_metrics.hum_60hz_dbfs > -50.0:
        issues.append(Issue(
            id="hum_60hz",
            title="Possible 60 Hz hum and harmonics",
            category="frequency",
            severity="medium",
            confidence=0.7,
            frequency_range=(60, 600),
            evidence=f"60 Hz: {spec_metrics.hum_60hz_dbfs:.1f} dBFS, Harmonics detected: {len(spec_metrics.hum_60hz_harmonics)}",
            impact="Audible hum reduces clarity and perceived professionalism",
            suggestions=[
                "Use notch filter or parametric EQ to remove 60 Hz and harmonics",
                "Check for ground loop (XLR cables, power supplies, USB grounding)",
                "Verify recording location for AC mains interference",
                "Use hum removal plugin if necessary"
            ]
        ))
    
    # Transients
    if trans_metrics.max_outlier_factor > 6.0:
        issues.append(Issue(
            id="transient_outliers",
            title="Isolated transient peaks detected",
            category="transients",
            severity="medium",
            confidence=0.65,
            evidence=f"Outlier ratio: {trans_metrics.outlier_ratio:.1%}, Peak factor: {trans_metrics.max_outlier_factor:.1f}x",
            impact="Few very loud peaks may trigger aggressive limiting and reduce apparent loudness",
            suggestions=[
                "Use soft clipper before limiter to catch outlier peaks",
                "Consider multi-band compression for isolated peak bands",
                "Use transient editing to slightly reduce peak amplitude",
                "Verify limiter settings and lookahead time in mastering"
            ]
        ))
    
    if trans_metrics.pre_ring_detected:
        issues.append(Issue(
            id="pre_ringing",
            title="Possible pre-ringing before transients",
            category="transients",
            severity="low",
            confidence=trans_metrics.pre_ring_confidence,
            evidence=f"Pre-ringing heuristic triggered (confidence: {trans_metrics.pre_ring_confidence:.1%})",
            impact="Pre-ringing can indicate non-linear processing or potential aliasing",
            suggestions=[
                "Use linear-phase filters in mastering",
                "Check for non-linear plugins that may introduce pre-echo",
                "Verify no unusual compression settings on individual tracks",
                "Review if convolver or reverb is causing the effect"
            ]
        ))
    
    # L/R Imbalance
    if audio_data.shape[1] > 1:
        imbalance_db = abs(metrics_l.peak_dbfs - metrics_r.peak_dbfs)
        if imbalance_db > 3.0:
            issues.append(Issue(
                id="lr_imbalance",
                title="Significant L/R level imbalance",
                category="stereo",
                severity="low",
                confidence=0.8,
                evidence=f"L peak: {metrics_l.peak_dbfs:.1f} dBFS, R peak: {metrics_r.peak_dbfs:.1f} dBFS (diff: {imbalance_db:.1f} dB)",
                impact="Stereo image shifts toward louder channel; may indicate mix issue or processing asymmetry",
                suggestions=[
                    "Check panning of stereo elements",
                    "Verify no asymmetric compression or EQ on left/right",
                    "Balance fader levels on mix bus if needed",
                    "Use correlation meter to check stereo field center"
                ]
            ))
    
    return issues
