"""
run_pipeline.py
───────────────
Batch feature extraction over all E-DAIC participants.

Handles:
  • On-the-fly extraction from .tar.gz archives (no need to pre-extract)
  • Parallel processing via multiprocessing
  • Resumable: skips participants already present in the output CSV
  • Writes a single feature matrix CSV: data/edaic/features/features.csv

Usage:
    python run_pipeline.py [--workers 4] [--archive_dir PATH] [--output PATH]
"""

from __future__ import annotations

import argparse
import logging
import os
import tarfile
import tempfile
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Optional

import pandas as pd
from tqdm import tqdm

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
import config
from extract_features import extract_participant

log = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Helpers (module-level so multiprocessing can pickle them)
# ─────────────────────────────────────────────────────────────────────────────

def _get_pid(src: Path) -> str:
    return src.name.replace("_P.tar.gz", "").replace("_P", "")


def _process_source(args: tuple) -> dict:
    """
    Dispatch to the right worker based on mode.
    args = (src, mode)  — must be picklable, so no closures.
    """
    src, mode = args
    if mode == "archive":
        return _process_archive(src)
    else:
        pid = _get_pid(src)
        return extract_participant(src, participant_id=pid)


# ─────────────────────────────────────────────────────────────────────────────
# Worker: extract archive to temp dir, run feature extraction, return result
# ─────────────────────────────────────────────────────────────────────────────

def _process_archive(archive_path: Path) -> dict:
    """
    Worker function (runs in a subprocess).
    Extracts the archive to a temporary directory, extracts features,
    cleans up the temp dir, and returns the feature dict.
    """
    pid = _get_pid(archive_path)
    try:
        with tempfile.TemporaryDirectory(prefix=f"edaic_{pid}_") as tmpdir:
            # Extract only the files we need (audio + transcript) to save I/O
            with tarfile.open(archive_path, "r:gz") as tf:
                members = tf.getmembers()
                wanted  = [
                    m for m in members
                    if ("audio" in m.name.lower() and m.name.lower().endswith(".wav"))
                    or ("transcript" in m.name.lower() and m.name.lower().endswith(".csv"))
                ]
                if not wanted:
                    # Fallback: extract everything (shouldn't happen)
                    tf.extractall(tmpdir)
                else:
                    for m in wanted:
                        tf.extract(m, tmpdir, set_attrs=False)

            # Find participant sub-directory
            extracted_dirs = [
                d for d in Path(tmpdir).iterdir() if d.is_dir()
            ]
            participant_dir = extracted_dirs[0] if extracted_dirs else Path(tmpdir)

            return extract_participant(participant_dir, participant_id=pid)

    except Exception as exc:
        log.error("[%s] Failed: %s", pid, traceback.format_exc())
        return {"participant_id": pid, "_error": str(exc)}


# ─────────────────────────────────────────────────────────────────────────────
# Main batch runner
# ─────────────────────────────────────────────────────────────────────────────

def run_pipeline(
    archive_dir: Path = config.DATA_RAW,
    output_path: Path = config.OUTPUT_DIR / "features.csv",
    workers:     int  = 4,
    extracted_dir: Optional[Path] = None,
) -> pd.DataFrame:
    """
    Run the full pipeline.

    Parameters
    ----------
    archive_dir   : Directory with *_P.tar.gz archives (used if extracted_dir is None)
    output_path   : Where to write the final feature CSV
    workers       : Number of parallel workers
    extracted_dir : If provided, read pre-extracted participant folders from here
                    instead of extracting from archives.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # ── Collect all participant archives / directories ─────────────────────
    if extracted_dir and extracted_dir.exists():
        sources = sorted(extracted_dir.glob("*_P"))
        mode = "dir"
        log.info("Found %d extracted participant directories.", len(sources))
    else:
        sources = sorted(archive_dir.glob("*_P.tar.gz"))
        mode = "archive"
        log.info("Found %d participant archives.", len(sources))

    if not sources:
        raise FileNotFoundError(
            f"No participants found in {'extracted_dir' if mode=='dir' else 'archive_dir'}. "
            "Run the download script first."
        )

    # ── Resume: skip already-processed participants ─────────────────────────
    completed_ids: set = set()
    if output_path.exists():
        existing = pd.read_csv(output_path)
        completed_ids = set(existing["participant_id"].astype(str))
        log.info("Resuming — %d participants already processed.", len(completed_ids))

    pending = [s for s in sources if _get_pid(s) not in completed_ids]
    log.info("%d participants to process.", len(pending))

    if not pending:
        log.info("Nothing to do — all participants already processed.")
        return pd.read_csv(output_path)

    # ── Run ────────────────────────────────────────────────────────────────
    all_results = []

    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(_process_source, (src, mode)): src for src in pending}
        for future in tqdm(as_completed(futures), total=len(futures),
                           desc="Participants", unit="p"):
            try:
                result = future.result()
                all_results.append(result)
            except Exception as exc:
                src = futures[future]
                log.error("Uncaught error for %s: %s", src.name, exc)

            # Incremental save every 10 participants to protect progress
            if len(all_results) % 10 == 0:
                _save_results(all_results, output_path, completed_ids)

    # Final save
    _save_results(all_results, output_path, completed_ids)

    if not output_path.exists():
        log.warning("No results were written — all participants failed.")
        return pd.DataFrame()

    df = pd.read_csv(output_path)
    log.info("Pipeline complete — %d participants, %d features.",
             len(df), df.shape[1] - 1)
    return df


def _save_results(
    new_results: list[dict],
    output_path: Path,
    existing_ids: set,
) -> None:
    """Merge new results with existing CSV and save."""
    if not new_results:
        return
    new_df = pd.DataFrame(new_results)

    if output_path.exists():
        existing_df = pd.read_csv(output_path)
        # Remove any rows being re-written (shouldn't happen, but safety first)
        existing_df = existing_df[
            ~existing_df["participant_id"].astype(str).isin(
                new_df["participant_id"].astype(str)
            )
        ]
        combined = pd.concat([existing_df, new_df], ignore_index=True)
    else:
        combined = new_df

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

    parser = argparse.ArgumentParser(description="Batch E-DAIC audio feature extraction.")
    parser.add_argument("--archive_dir", default=str(config.DATA_RAW),
                        help="Directory with *_P.tar.gz archives (default: data/edaic/raw)")
    parser.add_argument("--extracted_dir", default=None,
                        help="Use pre-extracted participant folders instead of archives")
    parser.add_argument("--output", default=str(config.OUTPUT_DIR / "features.csv"),
                        help="Output CSV path")
    parser.add_argument("--workers", type=int, default=4,
                        help="Number of parallel workers (default: 4)")
    args = parser.parse_args()

    run_pipeline(
        archive_dir   = Path(args.archive_dir),
        output_path   = Path(args.output),
        workers       = args.workers,
        extracted_dir = Path(args.extracted_dir) if args.extracted_dir else None,
    )
