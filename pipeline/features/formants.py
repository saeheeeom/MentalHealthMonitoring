"""
Formant features via Praat (parselmouth):
  - f1_mean, f1_std      First formant (vowel height)
  - f2_mean, f2_std      Second formant (vowel backness)
  - f3_mean, f3_std      Third formant (voice quality / lip rounding)
  - f1_f2_ratio          F1/F2 ratio (vowel space indicator)

Formants are extracted only from voiced frames (where F0 > 0).
"""

from __future__ import annotations

import numpy as np
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config
from features.utils import safe_mean, safe_std

try:
    import parselmouth
    from parselmouth.praat import call
    _PARSELMOUTH = True
except ImportError:
    _PARSELMOUTH = False


def compute_formants(
    audio: np.ndarray,
    sr: int = config.SAMPLE_RATE,
) -> dict:
    if not _PARSELMOUTH:
        return {k: float("nan") for k in
                ["f1_mean","f1_std","f2_mean","f2_std",
                 "f3_mean","f3_std","f1_f2_ratio"]}

    feats: dict = {}
    snd = parselmouth.Sound(audio.astype(np.float64), sampling_frequency=sr)

    try:
        # Praat Burg algorithm — standard for formant tracking
        formant_obj = call(
            snd, "To Formant (burg)",
            config.HOP_LENGTH,          # time step
            config.N_FORMANTS,          # max number of formants
            config.MAX_FORMANT_HZ,      # max formant frequency
            0.025,                      # window length (s)
            50.0,                       # pre-emphasis from (Hz)
        )

        # Also get pitch to restrict to voiced frames
        pitch_obj = call(snd, "To Pitch",
                         config.HOP_LENGTH,
                         config.F0_MIN_HZ,
                         config.F0_MAX_HZ)

        n_frames = int(snd.duration / config.HOP_LENGTH)
        f1_vals, f2_vals, f3_vals = [], [], []

        for i in range(n_frames):
            t = i * config.HOP_LENGTH
            f0 = call(pitch_obj, "Get value at time", t, "Hertz", "Linear")
            if not (np.isfinite(f0) and f0 > 0):
                continue                # skip unvoiced frames
            for fn, lst in zip([1, 2, 3], [f1_vals, f2_vals, f3_vals]):
                v = call(formant_obj, "Get value at time", fn, t, "Hertz", "Linear")
                if np.isfinite(v) and v > 0:
                    lst.append(v)

        feats["f1_mean"] = safe_mean(np.array(f1_vals))
        feats["f1_std"]  = safe_std(np.array(f1_vals))
        feats["f2_mean"] = safe_mean(np.array(f2_vals))
        feats["f2_std"]  = safe_std(np.array(f2_vals))
        feats["f3_mean"] = safe_mean(np.array(f3_vals))
        feats["f3_std"]  = safe_std(np.array(f3_vals))
        feats["f1_f2_ratio"] = (
            feats["f1_mean"] / feats["f2_mean"]
            if feats["f2_mean"] and np.isfinite(feats["f2_mean"])
            else float("nan")
        )
    except Exception:
        feats = {k: float("nan") for k in
                 ["f1_mean","f1_std","f2_mean","f2_std",
                  "f3_mean","f3_std","f1_f2_ratio"]}

    return feats
