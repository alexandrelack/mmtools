"""
Audio Player Module - Interactive playback with stereo/mono switching
"""
import numpy as np
from typing import Tuple

def convert_to_mono(audio: np.ndarray) -> np.ndarray:
    """Convert stereo to mono (L+R)/2"""
    if len(audio.shape) == 1:
        return audio
    return (audio[:, 0] + audio[:, 1]) / 2 if audio.shape[1] > 1 else audio[:, 0]

def get_mono_phase_shift(audio: np.ndarray) -> float:
    """Estimate phase shift in mono collapse"""
    if len(audio.shape) == 1:
        return 0.0
    
    L = audio[:, 0]
    R = audio[:, 1] if audio.shape[1] > 1 else audio[:, 0]
    correlation = np.corrcoef(L, R)[0, 1]
    return float(correlation)

def analyze_stereo_to_mono(audio: np.ndarray) -> dict:
    """Detailed stereo-to-mono analysis"""
    if len(audio.shape) == 1:
        return {'mode': 'mono', 'energy_loss': 0.0, 'phase_safe': True}
    
    L = audio[:, 0]
    R = audio[:, 1] if audio.shape[1] > 1 else audio[:, 0]
    M = (L + R) / 2
    
    energy_l = np.sqrt(np.mean(L**2))
    energy_r = np.sqrt(np.mean(R**2))
    energy_mono = np.mean(np.abs(M))
    energy_stereo = np.sqrt(energy_l**2 + energy_r**2)
    
    energy_loss = (1 - energy_mono / (energy_stereo + 1e-10)) * 100
    correlation = np.corrcoef(L, R)[0, 1]
    phase_safe = correlation > -0.2
    
    return {
        'mode': 'stereo',
        'energy_loss': float(energy_loss),
        'phase_safe': phase_safe,
        'correlation': float(correlation)
    }

def prepare_for_playback(audio: np.ndarray, mode: str = 'stereo', normalize: bool = False) -> np.ndarray:
    """Prepare audio for playback"""
    if mode == 'mono':
        audio = convert_to_mono(audio)
    
    if normalize:
        max_val = np.max(np.abs(audio))
        if max_val > 0:
            audio = audio / max_val * 0.95
    
    return audio.astype(np.float32)
