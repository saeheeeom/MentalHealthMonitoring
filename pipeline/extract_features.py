"""
extract_features.py
────────────────────
Extract all audio features for a single E-DAIC participant.

Usage (standalone):
    python extract_features.py --participant_dir /path/to/300_P  [--output features.json]

Returns a flat dict of feature_name → value.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Optional

import numpy as np

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
import config
from features.utils       import load_audio, load_transcript, concatenate_speech, segments_from_transcript
from features.prosodic    import compute_prosodic
from features.energy      import compute_energy
from features.spectral    import compute_spectral
from features.voice_quality import compute_voice_quality
from features.formants    import compute_formants
from features.temporal    import compute_temporal

log = logging.getLogger(__name__)


def find_participant_files(participant_dir: Path) -> tuple[Optional[Path], Optional[Path]]:
    """
    Locate the AUDIO.wav and Transcript.csv inside a participant directory.
    Handles minor naming variations (e.g. 300_AUDIO.wav vs AUDIO.wav).
    """
    audio_path = transcript_path = None
    for f in participant_dir.rglob("*"):
        name = f.name.lower()
        if "audio" in name and name.endswith(".wav"):
            audio_path = f
        if "transcript" in name and name.endswith(".csv"):
            transcript_path = f
    return audio_path, transcript_path


def extract_participant(
    participant_dir: str | Path,
    participant_id:  Optional[str] = None,
) -> dict:
    """
    Extract all features for one participant.

    Parameters
    ----------
    participant_dir : Path to the extracted participant folder (e.g. data/300_P/)
    participant_id  : Override the ID (defaults to folder name)

    Returns
    -------
    dict  { "participant_id": ..., "feature_name": value, ... }
    """
    participant_dir = Path(participant_dir)
    pid = participant_id or participant_dir.name.replace("_P", "")

    result: dict = {"participant_id": pid}

    # ── Locate files ───────────────────────────────────────────────────────
    audio_path, transcript_path = find_participant_files(participant_dir)

    if audio_path is None:
        log.warning("[%s] No audio file found — skipping.", pid)
        return result

    # ── Load audio ─────────────────────────────────────────────────────────
    audio, sr = load_audio(audio_path)
    total_duration_s = len(audio) / sr
    result["total_duration_s"] = total_duration_s

    # ── Load transcript ────────────────────────────────────────────────────
    transcript = None
    if transcript_path is not None:
        try:
            transcript = load_transcript(transcript_path)
        except Exception as e:
            log.warning("[%s] Could not load transcript: %s", pid, e)

    # ── Build speech-only audio (for voice-quality features) ───────────────
    # Use transcript segments if available, otherwise use full audio
    if transcript is not None and not transcript.empty:
        speech_segments = segments_from_transcript(transcript)
        speech_audio    = concatenate_speech(audio, sr, speech_segments)
    else:
        speech_audio = audio
        log.info("[%s] No transcript — using full audio for voiced features.", pid)

    # ── Feature extraction ─────────────────────────────────────────────────
    # Praat-based features (prosodic, voice quality, formants) must use the
    # FULL audio — Praat skips silence naturally, and speech-concatenated
    # audio creates discontinuities that break pitch/pulse tracking.
    # Energy and spectral features use speech_audio to exclude silence.

    log.info("[%s] Extracting prosodic features …", pid)
    result.update(compute_prosodic(audio, sr))

    log.info("[%s] Extracting energy features …", pid)
    result.update(compute_energy(speech_audio, sr))

    log.info("[%s] Extracting spectral features …", pid)
    result.update(compute_spectral(speech_audio, sr))

    log.info("[%s] Extracting voice quality features …", pid)
    result.update(compute_voice_quality(audio, sr))   # full audio — see note above

    log.info("[%s] Extracting formant features …", pid)
    result.update(compute_formants(audio, sr))        # full audio — see note above

    log.info("[%s] Extracting temporal/pause features …", pid)
    if transcript is not None:
        result.update(compute_temporal(transcript, total_duration_s))
    else:
        log.warning("[%s] No transcript — temporal features will be NaN.", pid)

    log.info("[%s] Done. %d features extracted.", pid, len(result))
    return result


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )

    parser = argparse.ArgumentParser(description="Extract audio features for one E-DAIC participant.")
    parser.add_argument("--participant_dir", required=True,
                        help="Path to extracted participant folder (e.g. data/edaic/extracted/300_P)")
    parser.add_argument("--output", default=None,
                        help="Optional path to save features as JSON")
    args = parser.parse_args()

    features = extract_participant(args.participant_dir)

    if args.output:
        out = Path(args.output)
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w") as fp:
            json.dump(features, fp, indent=2, default=lambda x: None if np.isnan(x) else x)
        print(f"Saved to {out}")
    else:
        print(json.dumps(features, indent=2, default=lambda x: None if np.isnan(x) else x))
