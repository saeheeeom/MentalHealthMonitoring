"""
Central configuration for the E-DAIC audio feature extraction pipeline.
Edit paths and parameters here; everything else imports from this file.
"""

from pathlib import Path

# ── Paths ──────────────────────────────────────────────────────────────────────
ROOT          = Path(__file__).resolve().parent.parent          # repo root
DATA_RAW      = ROOT / "data" / "edaic" / "raw"                 # tar.gz archives
DATA_EXTRACTED= ROOT / "data" / "edaic" / "extracted"           # extracted participants
OUTPUT_DIR    = ROOT / "data" / "edaic" / "features"            # CSV outputs

# ── Audio ──────────────────────────────────────────────────────────────────────
SAMPLE_RATE   = 16_000          # Hz — E-DAIC audio is 16 kHz mono
FRAME_LENGTH  = 0.025           # seconds (25 ms analysis window)
HOP_LENGTH    = 0.010           # seconds (10 ms hop)

# Derived in samples
FRAME_SAMPLES = int(FRAME_LENGTH * SAMPLE_RATE)   # 400
HOP_SAMPLES   = int(HOP_LENGTH   * SAMPLE_RATE)   # 160
N_FFT         = 1024             # FFT size (covers 64 ms at 16 kHz for resolution)

# ── VAD (energy-based silence detection) ──────────────────────────────────────
VAD_ENERGY_THRESHOLD_DB = -40   # frames below this RMS (dBFS) are silent
MIN_SPEECH_DURATION_S   = 0.10  # discard voiced bursts shorter than this
MIN_PAUSE_DURATION_S    = 0.20  # gaps shorter than this are not counted as pauses

# ── Prosodic ───────────────────────────────────────────────────────────────────
F0_MIN_HZ   = 50                # lower F0 bound (covers deep male voices)
F0_MAX_HZ   = 500               # upper F0 bound (covers high female/child voices)

# ── Spectral ───────────────────────────────────────────────────────────────────
N_MFCC      = 13                # number of MFCC coefficients
BAND_EDGES  = {                 # custom energy bands (Hz)
    "band_600_700" : (600,  700),
    "band_1k_2k"   : (1000, 2000),
    "band_2k_4k"   : (2000, 4000),
}

# ── Temporal / pause ───────────────────────────────────────────────────────────
FILLED_PAUSE_TOKENS = {"um", "uh", "mm", "hmm", "hm", "er", "erm"}
FRAG_DURATION_S     = 0.50      # utterances shorter than this = "fragmented"

# ── Formants ───────────────────────────────────────────────────────────────────
MAX_FORMANT_HZ  = 5500          # Praat ceiling for formant tracking
N_FORMANTS      = 5             # number of formants to track (use F1–F3)
