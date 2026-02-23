"""
Voice quality features via Praat (parselmouth):
  - jitter_local          Cycle-to-cycle F0 period variation (%)
  - jitter_local_abs      Absolute jitter (seconds)
  - jitter_rap            Relative Average Perturbation
  - shimmer_local         Cycle-to-cycle amplitude variation (%)
  - shimmer_local_db      Shimmer in dB
  - shimmer_apq3          Amplitude Perturbation Quotient (3-cycle)
  - hnr_mean              Harmonics-to-Noise Ratio (dB), mean over voiced frames
  - hnr_std               HNR standard deviation
  - breathiness           H1 – H2 amplitude difference (dB), voiced frames mean
                          (higher value → more breathy / less tense glottal closure)
  - vocal_tension         Spectral tilt in high-frequency region (2–8 kHz slope)
                          (less negative / more positive → more tense/pressed voice)
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


def _nan_dict() -> dict:
    return {k: float("nan") for k in [
        "jitter_local", "jitter_local_abs", "jitter_rap",
        "shimmer_local", "shimmer_local_db", "shimmer_apq3",
        "hnr_mean", "hnr_std",
        "breathiness", "vocal_tension",
    ]}


def compute_voice_quality(
    audio: np.ndarray,
    sr: int = config.SAMPLE_RATE,
) -> dict:
    """
    Parameters
    ----------
    audio : FULL unsegmented audio waveform (not speech-concatenated).
            Praat handles silence natively; concatenated audio breaks pulse tracking.
    sr    : Sample rate (default 16000 Hz)
    """
    if not _PARSELMOUTH:
        return _nan_dict()

    feats: dict = {}
    snd = parselmouth.Sound(audio.astype(np.float64), sampling_frequency=sr)

    # ── Jitter & Shimmer via Praat PointProcess ─────────────────────────────
    # Pitch-synchronous; Praat automatically skips unvoiced frames.
    try:
        pitch = call(snd, "To Pitch", 0.0, config.F0_MIN_HZ, config.F0_MAX_HZ)
        point_process = call([snd, pitch], "To PointProcess (cc)")

        feats["jitter_local"]     = call(point_process, "Get jitter (local)",
                                         0, 0, 0.0001, 0.02, 1.3)
        feats["jitter_local_abs"] = call(point_process, "Get jitter (local, absolute)",
                                         0, 0, 0.0001, 0.02, 1.3)
        feats["jitter_rap"]       = call(point_process, "Get jitter (rap)",
                                         0, 0, 0.0001, 0.02, 1.3)
        feats["shimmer_local"]    = call([snd, point_process], "Get shimmer (local)",
                                         0, 0, 0.0001, 0.02, 1.3, 1.6)
        feats["shimmer_local_db"] = call([snd, point_process], "Get shimmer (local_dB)",
                                         0, 0, 0.0001, 0.02, 1.3, 1.6)
        feats["shimmer_apq3"]     = call([snd, point_process], "Get shimmer (apq3)",
                                         0, 0, 0.0001, 0.02, 1.3, 1.6)
    except Exception:
        feats.update({k: float("nan") for k in
                      ["jitter_local","jitter_local_abs","jitter_rap",
                       "shimmer_local","shimmer_local_db","shimmer_apq3"]})

    # ── HNR (Harmonics-to-Noise Ratio) ─────────────────────────────────────
    try:
        harmonicity = call(snd, "To Harmonicity (cc)", config.HOP_LENGTH,
                           config.F0_MIN_HZ, 0.1, 1.0)
        n_frames = int(snd.duration / config.HOP_LENGTH)
        hnr_vals = np.array([
            call(harmonicity, "Get value at time", t * config.HOP_LENGTH, "Nearest")
            for t in range(n_frames)
        ], dtype=float)
        hnr_vals = hnr_vals[np.isfinite(hnr_vals) & (hnr_vals > -200)]
        feats["hnr_mean"] = safe_mean(hnr_vals)
        feats["hnr_std"]  = safe_std(hnr_vals)
    except Exception:
        feats["hnr_mean"] = float("nan")
        feats["hnr_std"]  = float("nan")

    # ── Breathiness: H1 – H2 via FFT (fast, vectorised) ────────────────────
    # Per-frame Praat extract is too slow (~65k frames × Extract Part).
    # Instead: use librosa STFT + Praat pitch track to find H1 and H2 bins.
    # H1-H2 > 0 dB → more breathy; H1-H2 < 0 dB → more modal/pressed voice.
    try:
        import librosa
        pitch_obj = call(snd, "To Pitch", config.HOP_LENGTH,
                         config.F0_MIN_HZ, config.F0_MAX_HZ)
        S_db = 20.0 * np.log10(
            np.abs(librosa.stft(audio.astype(np.float32),
                                n_fft=config.N_FFT,
                                hop_length=config.HOP_SAMPLES)) + 1e-9
        )   # shape: (n_freqs, n_frames)
        freqs   = librosa.fft_frequencies(sr=sr, n_fft=config.N_FFT)
        n_frames = S_db.shape[1]
        freq_res = freqs[1] - freqs[0]           # Hz per FFT bin

        h1_minus_h2 = []
        for i in range(n_frames):
            t = i * config.HOP_LENGTH
            f0 = call(pitch_obj, "Get value at time", t, "Hertz", "Linear")
            if not (np.isfinite(f0) and f0 > 0):
                continue
            # Find spectral bins nearest to F0 and 2·F0
            def _peak_db(target_hz: float, window_hz: float = 1.5 * freq_res) -> float:
                lo = max(0, int((target_hz - window_hz) / freq_res))
                hi = min(len(freqs) - 1, int((target_hz + window_hz) / freq_res) + 1)
                return float(S_db[lo:hi+1, i].max()) if lo <= hi else float("nan")

            amp_h1 = _peak_db(f0)
            amp_h2 = _peak_db(2.0 * f0)
            if np.isfinite(amp_h1) and np.isfinite(amp_h2):
                h1_minus_h2.append(amp_h1 - amp_h2)

        feats["breathiness"] = (float(np.mean(h1_minus_h2))
                                if h1_minus_h2 else float("nan"))
    except Exception:
        feats["breathiness"] = float("nan")

    # ── Vocal Tension: spectral tilt in 2–8 kHz region ─────────────────────
    # Less negative (or positive) slope = more energy in high frequencies → tenser voice
    try:
        import librosa
        S = np.abs(librosa.stft(audio.astype(np.float32),
                                n_fft=config.N_FFT,
                                hop_length=config.HOP_SAMPLES)) ** 2
        freqs = librosa.fft_frequencies(sr=sr, n_fft=config.N_FFT)
        mean_power = S.mean(axis=1)
        mask = (freqs >= 2000) & (freqs <= 8000)
        if mask.sum() > 2:
            log_p  = 10.0 * np.log10(mean_power[mask] + 1e-12)
            slope  = float(np.polyfit(freqs[mask], log_p, 1)[0])
            feats["vocal_tension"] = slope
        else:
            feats["vocal_tension"] = float("nan")
    except Exception:
        feats["vocal_tension"] = float("nan")

    return feats
