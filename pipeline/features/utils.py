"""
Shared utilities: audio loading, Voice Activity Detection (VAD),
transcript parsing, and segment concatenation.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import soundfile as sf
from pathlib import Path
from typing import List, Tuple, Optional

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config


# ─────────────────────────────────────────────────────────────────────────────
# Audio I/O
# ─────────────────────────────────────────────────────────────────────────────

def load_audio(path: str | Path, sr: int = config.SAMPLE_RATE) -> Tuple[np.ndarray, int]:
    """Load a WAV file and resample if necessary.  Returns (samples, sample_rate)."""
    audio, file_sr = sf.read(str(path), dtype="float32", always_2d=False)
    if audio.ndim == 2:                 # stereo → mono
        audio = audio.mean(axis=1)
    if file_sr != sr:
        import librosa
        audio = librosa.resample(audio, orig_sr=file_sr, target_sr=sr)
    return audio, sr


# ─────────────────────────────────────────────────────────────────────────────
# Transcript parsing
# ─────────────────────────────────────────────────────────────────────────────

def load_transcript(path: str | Path) -> pd.DataFrame:
    """
    Load the E-DAIC transcript CSV.
    Expected columns: Start_Time, End_Time, Text, Confidence
    Returns a DataFrame sorted by Start_Time with only participant turns.
    """
    df = pd.read_csv(str(path))
    df.columns = [c.strip() for c in df.columns]

    # Normalise column names (handle minor variations)
    rename = {}
    for col in df.columns:
        lc = col.lower().replace(" ", "_")
        if lc in ("start_time", "starttime", "start"):
            rename[col] = "start"
        elif lc in ("end_time", "endtime", "stop_time", "stop"):
            rename[col] = "end"
        elif lc in ("text", "value", "word", "utterance"):
            rename[col] = "text"
        elif lc in ("speaker", "speaker_label"):
            rename[col] = "speaker"
    df = df.rename(columns=rename)

    # Filter to participant (Ellie is the virtual interviewer)
    if "speaker" in df.columns:
        df = df[~df["speaker"].str.upper().isin(["ELLIE", "INTERVIEWER"])]

    df = df.dropna(subset=["start", "end"]).sort_values("start").reset_index(drop=True)
    df["start"] = df["start"].astype(float)
    df["end"]   = df["end"].astype(float)
    df["text"]  = df.get("text", pd.Series([""] * len(df))).fillna("").astype(str)
    return df


# ─────────────────────────────────────────────────────────────────────────────
# Voice Activity Detection (energy-based)
# ─────────────────────────────────────────────────────────────────────────────

def energy_vad(
    audio: np.ndarray,
    sr: int = config.SAMPLE_RATE,
    frame_len: int = config.FRAME_SAMPLES,
    hop_len:   int = config.HOP_SAMPLES,
    threshold_db: float = config.VAD_ENERGY_THRESHOLD_DB,
    min_speech_s: float = config.MIN_SPEECH_DURATION_S,
    min_pause_s:  float = config.MIN_PAUSE_DURATION_S,
) -> List[Tuple[float, float]]:
    """
    Simple energy-based VAD.
    Returns a list of (start_s, end_s) speech segments.
    """
    import librosa
    rms = librosa.feature.rms(y=audio, frame_length=frame_len, hop_length=hop_len)[0]
    rms_db = 20 * np.log10(rms + 1e-9)
    is_speech = rms_db >= threshold_db

    # Convert frame labels to segments
    frame_times = librosa.frames_to_time(
        np.arange(len(is_speech)), sr=sr, hop_length=hop_len
    )

    segments: List[Tuple[float, float]] = []
    in_seg = False
    seg_start = 0.0
    for i, active in enumerate(is_speech):
        if active and not in_seg:
            in_seg = True
            seg_start = float(frame_times[i])
        elif not active and in_seg:
            in_seg = False
            seg_end = float(frame_times[i])
            if (seg_end - seg_start) >= min_speech_s:
                segments.append((seg_start, seg_end))
    if in_seg:
        seg_end = float(frame_times[-1])
        if (seg_end - seg_start) >= min_speech_s:
            segments.append((seg_start, seg_end))

    # Merge segments separated by gaps shorter than min_pause_s
    merged: List[Tuple[float, float]] = []
    for seg in segments:
        if merged and (seg[0] - merged[-1][1]) < min_pause_s:
            merged[-1] = (merged[-1][0], seg[1])
        else:
            merged.append(list(seg))
    return [tuple(s) for s in merged]


def segments_from_transcript(
    transcript: pd.DataFrame,
    min_dur: float = 0.0,
) -> List[Tuple[float, float]]:
    """Convert transcript rows to (start, end) segment list."""
    segs = []
    for _, row in transcript.iterrows():
        dur = row["end"] - row["start"]
        if dur >= min_dur:
            segs.append((float(row["start"]), float(row["end"])))
    return segs


# ─────────────────────────────────────────────────────────────────────────────
# Audio segment helpers
# ─────────────────────────────────────────────────────────────────────────────

def extract_segment(audio: np.ndarray, sr: int, start_s: float, end_s: float) -> np.ndarray:
    """Slice audio array to [start_s, end_s]."""
    s = max(0, int(start_s * sr))
    e = min(len(audio), int(end_s   * sr))
    return audio[s:e]


def concatenate_speech(
    audio: np.ndarray,
    sr: int,
    segments: List[Tuple[float, float]],
) -> np.ndarray:
    """Return a single array with only speech segments concatenated."""
    parts = [extract_segment(audio, sr, s, e) for s, e in segments]
    if not parts:
        return np.zeros(1, dtype=np.float32)
    return np.concatenate(parts)


def safe_mean(values: np.ndarray) -> float:
    v = np.asarray(values, dtype=float)
    v = v[np.isfinite(v)]
    return float(np.mean(v)) if len(v) else float("nan")


def safe_std(values: np.ndarray) -> float:
    v = np.asarray(values, dtype=float)
    v = v[np.isfinite(v)]
    return float(np.std(v)) if len(v) else float("nan")
