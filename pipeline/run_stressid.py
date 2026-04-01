"""
run_stressid.py
───────────────
Batch acoustic feature extraction for the StressID dataset.

StressID layout:
    data/StressID Dataset/Audio/{subject_id}/{subject_id}_{task}.wav

Each .wav becomes one row in the output CSV, keyed by (subject_id, task).
No transcripts are available, so temporal features are derived from
energy-based VAD segments instead of transcript turns.

Usage:
    python run_stressid.py [--workers 4] [--output PATH]
"""

from __future__ import annotations

import argparse
import logging
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
import config
from features.utils import load_audio, energy_vad, concatenate_speech, safe_mean, safe_std
from features.prosodic import compute_prosodic
from features.energy import compute_energy
from features.spectral import compute_spectral
from features.voice_quality import compute_voice_quality
from features.formants import compute_formants

log = logging.getLogger(__name__)

# ── Paths ─────────────────────────────────────────────────────────────────────
STRESSID_ROOT = config.ROOT / "data" / "StressID Dataset"
STRESSID_AUDIO = STRESSID_ROOT / "Audio"
DEFAULT_OUTPUT = config.ROOT / "data" / "stressid" / "features" / "stressid_features.csv"


# ─────────────────────────────────────────────────────────────────────────────
# VAD-based temporal features (no transcript available)
# ─────────────────────────────────────────────────────────────────────────────

def compute_temporal_from_vad(
    audio: np.ndarray,
    sr: int,
    total_duration_s: float,
    min_pause_s: float = config.MIN_PAUSE_DURATION_S,
    frag_s: float = config.FRAG_DURATION_S,
) -> dict:
    """Derive temporal/pause features from energy-VAD segments."""
    segments = energy_vad(audio, sr)

    feats: dict = {}

    if not segments:
        return {k: float("nan") for k in [
            "utterance_count", "utterance_dur_mean", "utterance_dur_std",
            "utterance_dur_total", "pause_count", "pause_dur_mean",
            "pause_dur_max", "pause_dur_total", "proportion_silence",
            "fragmented_speech_count", "fragmented_speech_ratio",
        ]}

    durations = np.array([e - s for s, e in segments])

    # Utterance stats
    feats["utterance_count"] = int(len(segments))
    feats["utterance_dur_mean"] = safe_mean(durations)
    feats["utterance_dur_std"] = safe_std(durations)
    feats["utterance_dur_total"] = float(durations.sum())

    # Pause stats (gaps between consecutive VAD segments)
    gaps = []
    for i in range(1, len(segments)):
        gap = segments[i][0] - segments[i - 1][1]
        if gap >= min_pause_s:
            gaps.append(gap)

    feats["pause_count"] = int(len(gaps))
    feats["pause_dur_mean"] = safe_mean(np.array(gaps)) if gaps else 0.0
    feats["pause_dur_max"] = float(max(gaps)) if gaps else 0.0
    feats["pause_dur_total"] = float(sum(gaps)) if gaps else 0.0
    feats["proportion_silence"] = (
        feats["pause_dur_total"] / total_duration_s
        if total_duration_s > 0 else float("nan")
    )

    # Fragmented speech
    frag_count = int(np.sum(durations < frag_s))
    feats["fragmented_speech_count"] = frag_count
    feats["fragmented_speech_ratio"] = (
        frag_count / len(durations) if len(durations) > 0 else float("nan")
    )

    return feats


# ─────────────────────────────────────────────────────────────────────────────
# Single-file extraction
# ─────────────────────────────────────────────────────────────────────────────

def extract_single(wav_path: Path) -> dict:
    """Extract all acoustic features from one StressID .wav file."""
    # Parse subject_id and task from filename: {subject_id}_{task}.wav
    stem = wav_path.stem
    subject_id = wav_path.parent.name
    task = stem[len(subject_id) + 1:] if stem.startswith(subject_id) else stem

    result: dict = {"subject_id": subject_id, "task": task}

    try:
        audio, sr = load_audio(wav_path)
        total_duration_s = len(audio) / sr
        result["total_duration_s"] = total_duration_s

        # Build speech-only audio via VAD (no transcript available)
        segments = energy_vad(audio, sr)
        speech_audio = concatenate_speech(audio, sr, segments) if segments else audio

        # Acoustic features (reuse E-DAIC pipeline modules)
        result.update(compute_prosodic(audio, sr))
        result.update(compute_energy(speech_audio, sr))
        result.update(compute_spectral(speech_audio, sr))
        result.update(compute_voice_quality(audio, sr))
        result.update(compute_formants(audio, sr))
        result.update(compute_temporal_from_vad(audio, sr, total_duration_s))

    except Exception as exc:
        log.error("[%s/%s] Failed: %s", subject_id, task, traceback.format_exc())
        result["_error"] = str(exc)

    return result


def _extract_wrapper(wav_path_str: str) -> dict:
    """Picklable wrapper for multiprocessing."""
    return extract_single(Path(wav_path_str))


# ─────────────────────────────────────────────────────────────────────────────
# Batch runner
# ─────────────────────────────────────────────────────────────────────────────

def run_pipeline(
    audio_dir: Path = STRESSID_AUDIO,
    output_path: Path = DEFAULT_OUTPUT,
    workers: int = 4,
) -> pd.DataFrame:
    """Extract features from all StressID audio files."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    wav_files = sorted(
        w for w in audio_dir.rglob("*.wav") if not w.name.startswith("._")
    )
    log.info("Found %d .wav files in %s", len(wav_files), audio_dir)

    if not wav_files:
        raise FileNotFoundError(f"No .wav files in {audio_dir}")

    # Resume support: skip already-processed files
    completed: set = set()
    if output_path.exists():
        existing = pd.read_csv(output_path)
        completed = set(zip(existing["subject_id"].astype(str),
                            existing["task"].astype(str)))
        log.info("Resuming — %d rows already processed.", len(completed))

    pending = [
        w for w in wav_files
        if (w.parent.name, w.stem[len(w.parent.name) + 1:]) not in completed
    ]
    log.info("%d files to process.", len(pending))

    if not pending:
        log.info("Nothing to do — all files already processed.")
        return pd.read_csv(output_path)

    # Run extraction
    all_results: list[dict] = []

    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = {
            pool.submit(_extract_wrapper, str(w)): w for w in pending
        }
        for future in tqdm(as_completed(futures), total=len(futures),
                           desc="StressID features", unit="file"):
            try:
                result = future.result()
                all_results.append(result)
            except Exception as exc:
                src = futures[future]
                log.error("Uncaught error for %s: %s", src.name, exc)

            # Incremental save every 50 files
            if len(all_results) % 50 == 0 and all_results:
                _save(all_results, output_path)

    _save(all_results, output_path)

    df = pd.read_csv(output_path)
    log.info("Done — %d rows, %d feature columns.", len(df), df.shape[1] - 2)
    return df


def _save(new_results: list[dict], output_path: Path) -> None:
    """Merge new results with existing CSV and save."""
    if not new_results:
        return
    new_df = pd.DataFrame(new_results)

    if output_path.exists():
        existing_df = pd.read_csv(output_path)
        # Drop duplicates being re-written
        keys = set(zip(new_df["subject_id"].astype(str),
                       new_df["task"].astype(str)))
        existing_df = existing_df[
            ~existing_df.apply(
                lambda r: (str(r["subject_id"]), str(r["task"])) in keys, axis=1
            )
        ]
        combined = pd.concat([existing_df, new_df], ignore_index=True)
    else:
        combined = new_df

    # Drop error column from output
    if "_error" in combined.columns:
        errors = combined[combined["_error"].notna()]
        if len(errors):
            log.warning("%d files had errors.", len(errors))
        combined = combined.drop(columns=["_error"])

    combined.to_csv(output_path, index=False)
    log.info("Saved %d rows to %s", len(combined), output_path)


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )

    parser = argparse.ArgumentParser(
        description="Batch StressID audio feature extraction."
    )
    parser.add_argument("--audio_dir", default=str(STRESSID_AUDIO),
                        help="Directory with StressID Audio/ subfolders")
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT),
                        help="Output CSV path")
    parser.add_argument("--workers", type=int, default=4,
                        help="Number of parallel workers (default: 4)")
    args = parser.parse_args()

    run_pipeline(
        audio_dir=Path(args.audio_dir),
        output_path=Path(args.output),
        workers=args.workers,
    )
