"""
Prosodic features:
  - f0_mean               Mean fundamental frequency (Hz)
  - f0_std                F0 standard deviation
  - f0_range              F0 range (max – min, Hz)
  - f0_iqr                F0 interquartile range (robust alternative to range)
  - pitch_instability     Mean absolute frame-to-frame F0 difference
  - speech_rate_vad       Voiced frames / total duration (voiced fraction proxy)
"""

from __future__ import annotations

import numpy as np
from typing import List, Tuple, Optional

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config
from features.utils import safe_mean, safe_std

try:
    import parselmouth
    from parselmouth.praat import call
    _PARSELMOUTH = True
except ImportError:
    _PARSELMOUTH = False


def extract_f0(
    audio: np.ndarray,
    sr: int = config.SAMPLE_RATE,
    f0_min: float = config.F0_MIN_HZ,
    f0_max: float = config.F0_MAX_HZ,
) -> np.ndarray:
    """
    Extract F0 contour (voiced frames only) using Praat via parselmouth.
    Returns array of Hz values for voiced frames.
    """
    if not _PARSELMOUTH:
        raise ImportError("praat-parselmouth is required for F0 extraction.")

    snd = parselmouth.Sound(audio.astype(np.float64), sampling_frequency=sr)
    pitch = snd.to_pitch(
        time_step=config.HOP_LENGTH,
        pitch_floor=f0_min,
        pitch_ceiling=f0_max,
    )
    # Extract all voiced F0 values (unvoiced = 0)
    f0_values = pitch.selected_array["frequency"]
    return f0_values[f0_values > 0]          # drop unvoiced frames


def compute_prosodic(
    audio: np.ndarray,
    sr: int = config.SAMPLE_RATE,
) -> dict:
    """
    Compute all prosodic features from raw audio.
    Uses Praat pitch tracker; speech_rate uses energy VAD as backup.
    """
    feats: dict = {}

    # ── F0 features ────────────────────────────────────────────────────────
    f0 = extract_f0(audio, sr)

    if len(f0) < 2:
        feats.update({
            "f0_mean": float("nan"),
            "f0_std": float("nan"),
            "f0_range": float("nan"),
            "f0_iqr": float("nan"),
            "pitch_instability": float("nan"),
        })
    else:
        feats["f0_mean"]           = safe_mean(f0)
        feats["f0_std"]            = safe_std(f0)
        feats["f0_range"]          = float(np.max(f0) - np.min(f0))
        feats["f0_iqr"]            = float(np.percentile(f0, 75) - np.percentile(f0, 25))
        # Pitch instability: mean |Δf0| between consecutive voiced frames
        feats["pitch_instability"] = float(np.mean(np.abs(np.diff(f0))))

    # ── Speech rate (voiced fraction) ──────────────────────────────────────
    # Count voiced frames from Praat pitch object as proxy for voiced time
    if _PARSELMOUTH:
        snd = parselmouth.Sound(audio.astype(np.float64), sampling_frequency=sr)
        pitch = snd.to_pitch(
            time_step=config.HOP_LENGTH,
            pitch_floor=config.F0_MIN_HZ,
            pitch_ceiling=config.F0_MAX_HZ,
        )
        all_f0 = pitch.selected_array["frequency"]
        total_frames  = len(all_f0)
        voiced_frames = int(np.sum(all_f0 > 0))
        total_dur_s   = len(audio) / sr
        # voiced frames per second
        feats["speech_rate_voiced_frac"] = (
            voiced_frames / total_frames if total_frames > 0 else float("nan")
        )
        # syllables/sec proxy: voiced segment count / duration
        # (each contiguous voiced run ≈ a syllable nucleus)
        voiced_bool = all_f0 > 0
        transitions = np.diff(voiced_bool.astype(int))
        n_syllable_nuclei = int(np.sum(transitions == 1))  # rising edges
        feats["speech_rate_syllables_per_sec"] = (
            n_syllable_nuclei / total_dur_s if total_dur_s > 0 else float("nan")
        )
    else:
        feats["speech_rate_voiced_frac"]       = float("nan")
        feats["speech_rate_syllables_per_sec"] = float("nan")

    return feats
