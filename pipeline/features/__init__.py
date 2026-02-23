"""
features/
    utils.py          – audio I/O, VAD, transcript parsing, shared helpers
    prosodic.py       – F0 mean/variance/range/instability, speech rate
    energy.py         – RMS intensity, energy variability
    spectral.py       – MFCC 1-13, spectral centroid/slope/flatness/flux,
                        Hammarberg Index, Alpha Ratio, band energy
    voice_quality.py  – Jitter, Shimmer, HNR, Breathiness (H1-H2), Vocal Tension
    formants.py       – F1, F2, F3
    temporal.py       – pause count/duration, filled pauses, fragmented speech,
                        utterance duration, proportion of silence
"""
