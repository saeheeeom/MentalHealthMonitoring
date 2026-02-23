"""
test_pipeline.py
────────────────
Integration test for the audio feature extraction pipeline.
Runs against a single participant and validates every feature module.

Usage:
    python pipeline/test_pipeline.py [--participant_dir PATH]

Default participant_dir: /tmp/300_P  (extracted from 300_P.tar.gz)
If the directory doesn't exist the script will extract it automatically
from data/edaic/raw/300_P.tar.gz.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import tarfile
import tempfile
import time
from pathlib import Path

import numpy as np

# ── path setup ────────────────────────────────────────────────────────────────
PIPELINE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(PIPELINE_DIR))

import config
from features.utils         import load_audio, load_transcript, concatenate_speech, segments_from_transcript
from features.prosodic      import compute_prosodic
from features.energy        import compute_energy
from features.spectral      import compute_spectral
from features.voice_quality import compute_voice_quality
from features.formants      import compute_formants
from features.temporal      import compute_temporal
from extract_features       import extract_participant

# ── helpers ───────────────────────────────────────────────────────────────────

PASS = "\033[92m✓ PASS\033[0m"
FAIL = "\033[91m✗ FAIL\033[0m"
WARN = "\033[93m⚠ WARN\033[0m"

def _result(ok: bool, warn: bool = False) -> str:
    if warn: return WARN
    return PASS if ok else FAIL

def _check(name: str, condition: bool, detail: str = "", warn_only: bool = False):
    status = _result(condition, warn_only and not condition)
    marker = "⚠" if (warn_only and not condition) else ("✓" if condition else "✗")
    print(f"  [{marker}] {name}", f"— {detail}" if detail else "")
    return condition or warn_only


# ── individual module tests ───────────────────────────────────────────────────

def test_utils(audio: np.ndarray, sr: int, transcript) -> bool:
    print("\n── utils ──────────────────────────────────────────────────────")
    ok = True
    ok &= _check("audio loaded",       len(audio) > 0, f"{len(audio)/sr:.1f}s @ {sr}Hz")
    ok &= _check("transcript loaded",  transcript is not None and len(transcript) > 0,
                                       f"{len(transcript)} utterances")
    segs = segments_from_transcript(transcript)
    ok &= _check("speech segments",    len(segs) > 0, f"{len(segs)} segments")
    speech = concatenate_speech(audio, sr, segs)
    ok &= _check("speech audio",       len(speech) > sr, f"{len(speech)/sr:.1f}s of speech")
    return ok


def test_prosodic(audio: np.ndarray, sr: int) -> bool:
    print("\n── prosodic ───────────────────────────────────────────────────")
    t0 = time.time()
    feats = compute_prosodic(audio, sr)
    elapsed = time.time() - t0
    ok = True
    expected = ["f0_mean","f0_std","f0_range","f0_iqr","pitch_instability",
                "speech_rate_voiced_frac","speech_rate_syllables_per_sec"]
    for k in expected:
        v = feats.get(k)
        ok &= _check(k, v is not None and math.isfinite(v), f"{v:.4f}" if v else "missing")
    ok &= _check("f0_mean plausible", 50 < feats.get("f0_mean",0) < 500,
                 f"{feats.get('f0_mean',0):.1f} Hz")
    print(f"  ⏱  {elapsed:.1f}s")
    return ok


def test_energy(speech_audio: np.ndarray, sr: int) -> bool:
    print("\n── energy ─────────────────────────────────────────────────────")
    t0 = time.time()
    feats = compute_energy(speech_audio, sr)
    elapsed = time.time() - t0
    ok = True
    for k in ["intensity_mean_db","intensity_std_db","intensity_range_db","intensity_mean_linear"]:
        v = feats.get(k)
        ok &= _check(k, v is not None and math.isfinite(v), f"{v:.4f}" if v else "missing")
    ok &= _check("intensity_mean_db plausible", -80 < feats.get("intensity_mean_db",0) < 0,
                 f"{feats.get('intensity_mean_db',0):.1f} dBFS")
    print(f"  ⏱  {elapsed:.1f}s")
    return ok


def test_spectral(speech_audio: np.ndarray, sr: int) -> bool:
    print("\n── spectral ───────────────────────────────────────────────────")
    t0 = time.time()
    feats = compute_spectral(speech_audio, sr)
    elapsed = time.time() - t0
    ok = True
    # MFCCs
    for i in range(1, config.N_MFCC + 1):
        v = feats.get(f"mfcc_{i}_mean")
        ok &= _check(f"mfcc_{i}_mean", v is not None and math.isfinite(v),
                     f"{v:.2f}" if v else "missing")
    # Key spectral features
    for k in ["spectral_centroid_mean","spectral_slope","spectral_flatness_mean",
              "spectral_flux_mean","hammarberg_index","alpha_ratio"]:
        v = feats.get(k)
        ok &= _check(k, v is not None and math.isfinite(v), f"{v:.4f}" if v else "missing")
    # Band energies
    for band in config.BAND_EDGES:
        k = f"{band}_energy_db"
        v = feats.get(k)
        ok &= _check(k, v is not None and math.isfinite(v), f"{v:.2f} dB" if v else "missing")
    print(f"  ⏱  {elapsed:.1f}s")
    return ok


def test_voice_quality(audio: np.ndarray, sr: int) -> bool:
    print("\n── voice quality ──────────────────────────────────────────────")
    t0 = time.time()
    feats = compute_voice_quality(audio, sr)
    elapsed = time.time() - t0
    ok = True
    for k in ["jitter_local","jitter_local_abs","jitter_rap",
              "shimmer_local","shimmer_local_db","shimmer_apq3",
              "hnr_mean","hnr_std","breathiness","vocal_tension"]:
        v = feats.get(k)
        ok &= _check(k, v is not None and math.isfinite(v), f"{v:.5f}" if v else "missing")
    # Sanity bounds
    ok &= _check("jitter_local < 5%",  feats.get("jitter_local", 1) < 0.05,
                 f"{feats.get('jitter_local',0)*100:.2f}%", warn_only=True)
    ok &= _check("hnr_mean > -20 dB",  feats.get("hnr_mean", -999) > -20,
                 f"{feats.get('hnr_mean',0):.1f} dB", warn_only=True)
    print(f"  ⏱  {elapsed:.1f}s")
    return ok


def test_formants(audio: np.ndarray, sr: int) -> bool:
    print("\n── formants ───────────────────────────────────────────────────")
    t0 = time.time()
    feats = compute_formants(audio, sr)
    elapsed = time.time() - t0
    ok = True
    for k in ["f1_mean","f1_std","f2_mean","f2_std","f3_mean","f3_std","f1_f2_ratio"]:
        v = feats.get(k)
        ok &= _check(k, v is not None and math.isfinite(v), f"{v:.1f}" if v else "missing")
    ok &= _check("F1 plausible (200–1000 Hz)", 200 < feats.get("f1_mean",0) < 1000,
                 f"{feats.get('f1_mean',0):.0f} Hz")
    ok &= _check("F2 plausible (600–3000 Hz)", 600 < feats.get("f2_mean",0) < 3000,
                 f"{feats.get('f2_mean',0):.0f} Hz")
    ok &= _check("F3 plausible (1500–4000 Hz)", 1500 < feats.get("f3_mean",0) < 4000,
                 f"{feats.get('f3_mean',0):.0f} Hz")
    print(f"  ⏱  {elapsed:.1f}s")
    return ok


def test_temporal(transcript, total_duration_s: float) -> bool:
    print("\n── temporal ───────────────────────────────────────────────────")
    t0 = time.time()
    feats = compute_temporal(transcript, total_duration_s)
    elapsed = time.time() - t0
    ok = True
    for k in ["utterance_count","utterance_dur_mean","pause_count","pause_dur_mean",
              "proportion_silence","filled_pause_count","fragmented_speech_ratio","speech_rate_wpm"]:
        v = feats.get(k)
        ok &= _check(k, v is not None and math.isfinite(v), f"{v:.3f}" if v else "missing")
    ok &= _check("proportion_silence in [0,1]",
                 0 <= feats.get("proportion_silence", -1) <= 1,
                 f"{feats.get('proportion_silence',0):.1%}", warn_only=True)
    ok &= _check("pause_count > 0", feats.get("pause_count", 0) > 0,
                 f"{feats.get('pause_count',0)} pauses")
    print(f"  ⏱  {elapsed:.1f}s")
    return ok


def test_full_pipeline(participant_dir: Path) -> bool:
    print("\n── full extraction (end-to-end) ───────────────────────────────")
    t0 = time.time()
    feats = extract_participant(participant_dir)
    elapsed = time.time() - t0

    total = len(feats) - 1          # exclude participant_id
    nan_keys = [k for k, v in feats.items()
                if k != "participant_id" and (v is None or (isinstance(v, float) and math.isnan(v)))]
    ok = True
    ok &= _check("total features ≥ 90",  total >= 90,  f"{total} features")
    ok &= _check("zero NaN values",       len(nan_keys) == 0,
                 f"NaN in: {nan_keys}" if nan_keys else "all finite")
    print(f"  ⏱  {elapsed:.1f}s total")

    # Feature-count breakdown by group
    groups = {
        "Prosodic":   [k for k in feats if any(x in k for x in ["f0","pitch","speech_rate"])],
        "Energy":     [k for k in feats if "intensity" in k],
        "Spectral":   [k for k in feats if any(x in k for x in ["spectral","hammarberg","alpha","band_","mfcc"])],
        "VoiceQ":     [k for k in feats if any(x in k for x in ["jitter","shimmer","hnr","breathiness","vocal_tension"])],
        "Formants":   [k for k in feats if any(x in k for x in ["f1_","f2_","f3_","f1_f2"])],
        "Temporal":   [k for k in feats if any(x in k for x in ["utterance","pause","silence","fragmented","filled","wpm"])],
    }
    print("\n  Feature counts per group:")
    for grp, keys in groups.items():
        print(f"    {grp:<12} {len(keys):>3} features")

    # Save full output for inspection
    out_path = PIPELINE_DIR / "test_output_300.json"
    with open(out_path, "w") as fp:
        json.dump(feats, fp, indent=2, default=lambda x: None if (isinstance(x, float) and math.isnan(x)) else x)
    print(f"\n  Full output saved → {out_path}")
    return ok


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--participant_dir", default="/tmp/300_P",
                        help="Path to extracted participant folder (default: /tmp/300_P)")
    args = parser.parse_args()
    participant_dir = Path(args.participant_dir)

    # Auto-extract if needed
    if not participant_dir.exists():
        archive = config.DATA_RAW / "300_P.tar.gz"
        if not archive.exists():
            print(f"ERROR: {archive} not found. Make sure the download has completed.")
            sys.exit(1)
        print(f"Extracting {archive} to /tmp …")
        with tarfile.open(archive) as tf:
            tf.extractall("/tmp")
        print("Done.\n")

    print("=" * 64)
    print("  E-DAIC Feature Extraction — Pipeline Test")
    print(f"  Participant: {participant_dir.name}")
    print("=" * 64)

    # ── Load shared inputs once ───────────────────────────────────────────
    audio_path      = next(participant_dir.rglob("*AUDIO.wav"), None)
    transcript_path = next(participant_dir.rglob("*Transcript.csv"), None)

    if audio_path is None:
        print("ERROR: No audio file found.")
        sys.exit(1)

    audio, sr   = load_audio(audio_path)
    transcript  = load_transcript(transcript_path) if transcript_path else None
    segs        = segments_from_transcript(transcript) if transcript is not None else []
    speech_audio = concatenate_speech(audio, sr, segs) if segs else audio
    total_dur   = len(audio) / sr

    print(f"\n  Audio  : {total_dur:.1f}s, {sr}Hz, {len(audio):,} samples")
    print(f"  Speech : {len(speech_audio)/sr:.1f}s ({len(speech_audio)/sr/total_dur:.0%} of session)")
    if transcript is not None:
        print(f"  Transcript: {len(transcript)} utterances")

    # ── Run each module ───────────────────────────────────────────────────
    results = {
        "utils":         test_utils(audio, sr, transcript),
        "prosodic":      test_prosodic(audio, sr),
        "energy":        test_energy(speech_audio, sr),
        "spectral":      test_spectral(speech_audio, sr),
        "voice_quality": test_voice_quality(audio, sr),
        "formants":      test_formants(audio, sr),
        "temporal":      test_temporal(transcript, total_dur) if transcript is not None else True,
        "full_pipeline": test_full_pipeline(participant_dir),
    }

    # ── Summary ───────────────────────────────────────────────────────────
    print("\n" + "=" * 64)
    print("  SUMMARY")
    print("=" * 64)
    all_pass = True
    for module, passed in results.items():
        status = PASS if passed else FAIL
        print(f"  {status}  {module}")
        all_pass = all_pass and passed

    print("=" * 64)
    if all_pass:
        print("  \033[92mAll tests passed.\033[0m")
    else:
        print("  \033[91mSome tests failed — see details above.\033[0m")
    print("=" * 64)
    sys.exit(0 if all_pass else 1)


if __name__ == "__main__":
    main()
