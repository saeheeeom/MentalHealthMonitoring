"""
Spectral features:
  - mfcc_1 … mfcc_13       Mean of each MFCC coefficient
  - mfcc_delta_1…13        Mean of MFCC delta (velocity) coefficients
  - spectral_centroid_mean  Mean spectral centroid (Hz)
  - spectral_centroid_std
  - spectral_slope          Slope of log-power spectrum vs frequency (dB/Hz)
  - spectral_flatness_mean  Mean spectral flatness (Wiener entropy)
  - spectral_flux_mean      Mean frame-to-frame spectral change
  - hammarberg_index        Energy(0–2 kHz peak) – Energy(2–5 kHz peak) in dB
  - alpha_ratio             10·log10(E_low/E_high), split at 1 kHz
  - band_*_energy_db        Mean energy in custom bands (config.BAND_EDGES)
"""

from __future__ import annotations

import numpy as np
import librosa
from typing import Dict, Tuple
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config
from features.utils import safe_mean, safe_std


# ─────────────────────────────────────────────────────────────────────────────
# Internal helpers
# ─────────────────────────────────────────────────────────────────────────────

def _power_spectrum(
    audio: np.ndarray,
    sr: int,
    n_fft: int = config.N_FFT,
    hop_len: int = config.HOP_SAMPLES,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute short-time power spectrum.
    Returns (freqs_hz, mean_power) where mean_power is averaged over frames.
    """
    S = np.abs(librosa.stft(audio, n_fft=n_fft, hop_length=hop_len)) ** 2
    freqs = librosa.fft_frequencies(sr=sr, n_fft=n_fft)
    mean_power = S.mean(axis=1)      # average across time frames
    return freqs, mean_power


def _band_energy_db(
    freqs: np.ndarray,
    power: np.ndarray,
    low_hz: float,
    high_hz: float,
) -> float:
    """Mean power (dB) within a frequency band."""
    mask = (freqs >= low_hz) & (freqs <= high_hz)
    if not mask.any():
        return float("nan")
    band_power = power[mask].mean()
    return float(10.0 * np.log10(band_power + 1e-12))


# ─────────────────────────────────────────────────────────────────────────────
# Main extractor
# ─────────────────────────────────────────────────────────────────────────────

def compute_spectral(
    audio: np.ndarray,
    sr: int = config.SAMPLE_RATE,
    n_fft: int = config.N_FFT,
    hop_len: int = config.HOP_SAMPLES,
    frame_len: int = config.FRAME_SAMPLES,
) -> dict:
    feats: dict = {}

    # ── STFT / power spectrum ───────────────────────────────────────────────
    S_mag  = np.abs(librosa.stft(audio, n_fft=n_fft, hop_length=hop_len))
    S_pow  = S_mag ** 2
    freqs  = librosa.fft_frequencies(sr=sr, n_fft=n_fft)
    mean_power = S_pow.mean(axis=1)   # (n_fft//2 + 1,)

    # ── MFCCs 1–13 ─────────────────────────────────────────────────────────
    mfccs = librosa.feature.mfcc(
        y=audio, sr=sr,
        n_mfcc=config.N_MFCC,
        n_fft=n_fft, hop_length=hop_len,
    )                                  # (13, n_frames)
    delta_mfccs = librosa.feature.delta(mfccs)

    for i in range(config.N_MFCC):
        feats[f"mfcc_{i+1}_mean"] = safe_mean(mfccs[i])
        feats[f"mfcc_{i+1}_std"]  = safe_std(mfccs[i])
        feats[f"mfcc_delta_{i+1}_mean"] = safe_mean(delta_mfccs[i])

    # ── Spectral centroid ───────────────────────────────────────────────────
    centroid = librosa.feature.spectral_centroid(
        y=audio, sr=sr, n_fft=n_fft, hop_length=hop_len,
    )[0]
    feats["spectral_centroid_mean"] = safe_mean(centroid)
    feats["spectral_centroid_std"]  = safe_std(centroid)

    # ── Spectral slope (linear regression of log-power vs freq) ────────────
    log_power = 10.0 * np.log10(mean_power + 1e-12)
    # Use only frequencies > 0 Hz for log scale
    valid = freqs > 0
    if valid.sum() > 1:
        slope = float(np.polyfit(freqs[valid], log_power[valid], 1)[0])
    else:
        slope = float("nan")
    feats["spectral_slope"] = slope

    # ── Spectral flatness ───────────────────────────────────────────────────
    flatness = librosa.feature.spectral_flatness(
        y=audio, n_fft=n_fft, hop_length=hop_len,
    )[0]
    feats["spectral_flatness_mean"] = safe_mean(flatness)
    feats["spectral_flatness_std"]  = safe_std(flatness)

    # ── Spectral flux ───────────────────────────────────────────────────────
    # Mean frame-to-frame L2 difference in normalised spectra
    S_norm = S_mag / (S_mag.sum(axis=0, keepdims=True) + 1e-9)
    flux = np.sqrt(np.sum(np.diff(S_norm, axis=1) ** 2, axis=0))
    feats["spectral_flux_mean"] = safe_mean(flux)
    feats["spectral_flux_std"]  = safe_std(flux)

    # ── Hammarberg Index ────────────────────────────────────────────────────
    # Max energy below 2 kHz  minus  max energy between 2–5 kHz  (dB)
    mask_lo = (freqs > 0)  & (freqs <= 2000)
    mask_hi = (freqs > 2000) & (freqs <= 5000)
    if mask_lo.any() and mask_hi.any():
        peak_lo_db = 10.0 * np.log10(mean_power[mask_lo].max() + 1e-12)
        peak_hi_db = 10.0 * np.log10(mean_power[mask_hi].max() + 1e-12)
        feats["hammarberg_index"] = float(peak_lo_db - peak_hi_db)
    else:
        feats["hammarberg_index"] = float("nan")

    # ── Alpha Ratio ─────────────────────────────────────────────────────────
    # 10·log10( E_below_1kHz / E_above_1kHz )
    mask_alpha_lo = freqs <= 1000
    mask_alpha_hi = freqs >  1000
    if mask_alpha_lo.any() and mask_alpha_hi.any():
        e_lo = mean_power[mask_alpha_lo].sum()
        e_hi = mean_power[mask_alpha_hi].sum()
        feats["alpha_ratio"] = float(10.0 * np.log10(e_lo / (e_hi + 1e-12)))
    else:
        feats["alpha_ratio"] = float("nan")

    # ── Custom band energies ────────────────────────────────────────────────
    for band_name, (lo, hi) in config.BAND_EDGES.items():
        feats[f"{band_name}_energy_db"] = _band_energy_db(freqs, mean_power, lo, hi)

    return feats
