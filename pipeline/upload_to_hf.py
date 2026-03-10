"""
upload_to_hf.py
───────────────
Upload E-DAIC data to Hugging Face Hub in two separate dataset configs:

  1. features  — features.csv (cleaned) joined with labels
  2. audio     — raw audio (.wav) + transcript (.csv) per participant,
                 streamed directly from .tar.gz archives (memory-safe for 105 GB)

Reads HF_TOKEN from .env in the repo root.

Usage:
    python pipeline/upload_to_hf.py --repo YOUR_HF_ORG/your-dataset-name
    python pipeline/upload_to_hf.py --repo YOUR_HF_ORG/your-dataset-name --features_only
    python pipeline/upload_to_hf.py --repo YOUR_HF_ORG/your-dataset-name --audio_only
"""

from __future__ import annotations

import argparse
import io
import logging
import os
import re
import sys
import tarfile
import tempfile
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv

# ── HuggingFace imports ────────────────────────────────────────────────────
from datasets import Dataset, Features, Sequence, Value
from huggingface_hub import HfApi

# ── Paths ──────────────────────────────────────────────────────────────────
ROOT       = Path(__file__).resolve().parent.parent
DATA_RAW   = ROOT / "data" / "edaic" / "raw"
LABELS_DIR = ROOT / "data" / "edaic" / "labels"
FEAT_CSV   = ROOT / "data" / "edaic" / "features" / "features.csv"

log = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _clean_pid(raw: str) -> str:
    """
    Normalise any participant_id variant to a plain integer string.
    Handles: '300', '300.tar', '300_P', '300_P.tar', etc.
    """
    return re.sub(r"[^0-9]", "", str(raw).split(".")[0].split("_")[0])


def _load_features(path: Path = FEAT_CSV) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["participant_id"] = df["participant_id"].apply(_clean_pid)
    return df


def _load_labels() -> pd.DataFrame:
    """
    Load Detailed_PHQ8_Labels.csv (basic PHQ-8 items + total).
    Rename Participant_ID → participant_id for merging.
    """
    path = LABELS_DIR / "Detailed_PHQ8_Labels.csv"
    df = pd.read_csv(path)
    # Normalise column name
    id_col = [c for c in df.columns if "participant" in c.lower() or c.lower() == "id"][0]
    df = df.rename(columns={id_col: "participant_id"})
    df["participant_id"] = df["participant_id"].apply(_clean_pid)
    return df


# ─────────────────────────────────────────────────────────────────────────────
# 1. Features + Labels
# ─────────────────────────────────────────────────────────────────────────────

def upload_features(repo_id: str, api: HfApi) -> None:
    log.info("Loading features …")
    features_df = _load_features()

    log.info("Loading labels …")
    labels_df = _load_labels()

    log.info("Merging features (%d rows) with labels (%d rows) …",
             len(features_df), len(labels_df))
    merged = features_df.merge(labels_df, on="participant_id", how="left")
    log.info("Merged shape: %s", merged.shape)

    # Convert to HuggingFace Dataset and push
    hf_ds = Dataset.from_pandas(merged, preserve_index=False)
    log.info("Pushing features+labels dataset …")
    hf_ds.push_to_hub(
        repo_id,
        config_name="features",
        split="train",
        commit_message="Add features + PHQ-8 labels",
    )
    log.info("features config uploaded.")


# ─────────────────────────────────────────────────────────────────────────────
# 2. Audio + Transcripts  (memory-safe generator)
# ─────────────────────────────────────────────────────────────────────────────

def _audio_transcript_generator(archives: list[Path]):
    """
    Yields one dict per participant, streaming archives one at a time.
    Each dict:
        participant_id : str
        audio          : {"bytes": bytes, "path": str}   ← HF Audio format
        transcript_csv : str   (raw CSV text)
    """
    for archive in archives:
        pid = re.sub(r"[^0-9]", "", archive.name.split("_")[0])
        try:
            with tarfile.open(archive, "r:gz") as tf:
                audio_bytes      = None
                transcript_text  = None

                for member in tf.getmembers():
                    name_lower = member.name.lower()
                    if "audio" in name_lower and name_lower.endswith(".wav"):
                        f = tf.extractfile(member)
                        if f:
                            audio_bytes = f.read()
                    elif "transcript" in name_lower and name_lower.endswith(".csv"):
                        f = tf.extractfile(member)
                        if f:
                            transcript_text = f.read().decode("utf-8", errors="replace")

            if audio_bytes is None:
                log.warning("[%s] No audio found in archive — skipping.", pid)
                continue

            yield {
                "participant_id": pid,
                "audio": {"bytes": audio_bytes, "path": f"{pid}_AUDIO.wav"},
                "transcript_csv": transcript_text or "",
            }

        except Exception as exc:
            log.error("[%s] Failed to read archive: %s", pid, exc)


def upload_audio(repo_id: str, api: HfApi, shard_size: str = "500MB") -> None:
    archives = sorted(DATA_RAW.glob("*_P.tar.gz"))
    if not archives:
        raise FileNotFoundError(f"No archives found in {DATA_RAW}")
    log.info("Found %d participant archives to upload.", len(archives))

    # Store audio as raw bytes dict — avoids torchcodec dependency.
    # On the HF Hub the column is tagged as audio via the path suffix (.wav).
    # Users can cast after loading: ds.cast_column("audio", Audio())
    hf_features = Features({
        "participant_id": Value("string"),
        "audio": {
            "bytes": Value("large_binary"),  # 64-bit offsets — required for large wav files
            "path":  Value("string"),
        },
        "transcript_csv": Value("string"),
    })

    log.info("Building streaming dataset …")
    hf_ds = Dataset.from_generator(
        _audio_transcript_generator,
        gen_kwargs={"archives": archives},
        features=hf_features,
        writer_batch_size=4,   # flush to disk every 4 participants (~1 GB) to avoid Arrow buffer overflow
    )

    log.info("Pushing audio+transcript dataset (%d participants) …", len(hf_ds))
    hf_ds.push_to_hub(
        repo_id,
        config_name="audio",
        split="train",
        commit_message="Add raw audio + transcripts",
        max_shard_size=shard_size,
    )
    log.info("audio config uploaded.")


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )

    # Load token from .env
    env_path = ROOT / ".env"
    if env_path.exists():
        load_dotenv(env_path)
    token = os.environ.get("HF_TOKEN")
    if not token:
        sys.exit("HF_TOKEN not set. Add it to .env or export it as an environment variable.")

    parser = argparse.ArgumentParser(description="Upload E-DAIC data to HuggingFace Hub.")
    parser.add_argument("--repo", required=True,
                        help="HuggingFace repo id, e.g. your-org/edaic")
    parser.add_argument("--features_only", action="store_true",
                        help="Upload only the features+labels config")
    parser.add_argument("--audio_only", action="store_true",
                        help="Upload only the audio+transcripts config")
    parser.add_argument("--shard_size", default="500MB",
                        help="Max shard size for audio dataset (default: 500MB)")
    args = parser.parse_args()

    api = HfApi(token=token)

    # Create the repo if it doesn't exist (dataset type)
    api.create_repo(repo_id=args.repo, repo_type="dataset", exist_ok=True)

    if args.audio_only:
        upload_audio(args.repo, api, shard_size=args.shard_size)
    elif args.features_only:
        upload_features(args.repo, api)
    else:
        upload_features(args.repo, api)
        upload_audio(args.repo, api, shard_size=args.shard_size)

    log.info("All done.")
