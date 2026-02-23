"""
Temporal / pause features (derived from transcript + VAD):
  - utterance_count            Total number of participant utterances
  - utterance_dur_mean         Mean utterance duration (s)
  - utterance_dur_std
  - utterance_dur_total        Total speaking time (s)
  - pause_count                Number of inter-utterance pauses
  - pause_dur_mean             Mean pause duration (s)
  - pause_dur_max              Longest pause (s)
  - pause_dur_total            Total pause time (s)
  - proportion_silence         pause_dur_total / total_session_duration
  - filled_pause_count         Occurrences of "um/uh/mm/hmm/er…" in transcript
  - filled_pause_rate          Filled pauses per minute of speaking time
  - fragmented_speech_count    Utterances shorter than FRAG_DURATION_S
  - fragmented_speech_ratio    fragmented_count / utterance_count
  - speech_rate_wpm            Words per minute (from transcript text)
"""

from __future__ import annotations

import re
import numpy as np
import pandas as pd
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config
from features.utils import load_transcript, safe_mean, safe_std


_FILLED_RE = re.compile(
    r'\b(' + '|'.join(re.escape(t) for t in config.FILLED_PAUSE_TOKENS) + r')\b',
    re.IGNORECASE,
)


def compute_temporal(
    transcript: pd.DataFrame,
    total_duration_s: float,
    min_pause_s: float = config.MIN_PAUSE_DURATION_S,
    frag_s: float = config.FRAG_DURATION_S,
) -> dict:
    """
    Compute temporal and pause features from the transcript DataFrame.

    Parameters
    ----------
    transcript       : DataFrame with columns [start, end, text]
    total_duration_s : Full session duration in seconds (from audio length)
    min_pause_s      : Minimum gap to count as a pause
    frag_s           : Utterances shorter than this are "fragmented"
    """
    feats: dict = {}

    if transcript.empty:
        return {k: float("nan") for k in [
            "utterance_count","utterance_dur_mean","utterance_dur_std",
            "utterance_dur_total","pause_count","pause_dur_mean","pause_dur_max",
            "pause_dur_total","proportion_silence","filled_pause_count",
            "filled_pause_rate","fragmented_speech_count","fragmented_speech_ratio",
            "speech_rate_wpm",
        ]}

    durations = (transcript["end"] - transcript["start"]).values
    texts     = transcript["text"].tolist()

    # ── Utterance stats ────────────────────────────────────────────────────
    feats["utterance_count"]      = int(len(transcript))
    feats["utterance_dur_mean"]   = safe_mean(durations)
    feats["utterance_dur_std"]    = safe_std(durations)
    feats["utterance_dur_total"]  = float(durations.sum())

    # ── Pause stats (gaps between consecutive utterances) ──────────────────
    gaps = []
    for i in range(1, len(transcript)):
        gap = transcript.iloc[i]["start"] - transcript.iloc[i - 1]["end"]
        if gap >= min_pause_s:
            gaps.append(gap)

    feats["pause_count"]     = int(len(gaps))
    feats["pause_dur_mean"]  = safe_mean(np.array(gaps)) if gaps else 0.0
    feats["pause_dur_max"]   = float(max(gaps)) if gaps else 0.0
    feats["pause_dur_total"] = float(sum(gaps)) if gaps else 0.0
    feats["proportion_silence"] = (
        feats["pause_dur_total"] / total_duration_s
        if total_duration_s > 0 else float("nan")
    )

    # ── Filled pauses ──────────────────────────────────────────────────────
    all_text = " ".join(texts)
    filled_count = len(_FILLED_RE.findall(all_text))
    feats["filled_pause_count"] = filled_count
    speaking_min = feats["utterance_dur_total"] / 60.0
    feats["filled_pause_rate"] = (
        filled_count / speaking_min if speaking_min > 0 else float("nan")
    )

    # ── Fragmented speech ──────────────────────────────────────────────────
    frag_count = int(np.sum(durations < frag_s))
    feats["fragmented_speech_count"] = frag_count
    feats["fragmented_speech_ratio"] = (
        frag_count / len(durations) if len(durations) > 0 else float("nan")
    )

    # ── Speech rate (words per minute) ────────────────────────────────────
    word_count = sum(len(t.split()) for t in texts)
    feats["speech_rate_wpm"] = (
        word_count / speaking_min if speaking_min > 0 else float("nan")
    )

    return feats
