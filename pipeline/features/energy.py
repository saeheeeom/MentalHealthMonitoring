"""
Energy / intensity features:
  - intensity_mean         Mean RMS energy (dBFS)
  - intensity_std          Std of RMS energy contour (energy variability)
  - intensity_range        Range of RMS energy contour
  - intensity_mean_linear  Mean RMS in linear scale (for ratio computations)
"""

from __future__ import annotations

import numpy as np
import librosa
from typing import Optional

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config
from features.utils import safe_mean, safe_std


def compute_energy(
    audio: np.ndarray,
    sr: int = config.SAMPLE_RATE,
    frame_len: int = config.FRAME_SAMPLES,
    hop_len:   int = config.HOP_SAMPLES,
) -> dict:
    """
    Compute frame-level RMS energy and derive intensity statistics.
    All dB values are dBFS (0 dBFS = full scale).
    """
    rms = librosa.feature.rms(
        y=audio,
        frame_length=frame_len,
        hop_length=hop_len,
    )[0]                                     # shape: (n_frames,)

    rms_db = 20.0 * np.log10(rms + 1e-9)    # convert to dBFS

    feats = {
        "intensity_mean_db":    safe_mean(rms_db),
        "intensity_std_db":     safe_std(rms_db),
        "intensity_range_db":   float(np.max(rms_db) - np.min(rms_db)),
        "intensity_mean_linear":safe_mean(rms),
    }
    return feats
