# Mental Health Monitoring

A multimodal mental health check-in system that combines acoustic analysis, semantic NLP, and behavioral stress modeling to compute longitudinal risk scores for individual participants.

---

## 1. Dashboard

**Live:** [mentalhealthmonitoring.streamlit.app](https://mentalhealthmonitoring.streamlit.app/)

A Streamlit app for reviewing participant check-in data. Displays per-participant stress arcs (self-reported and acoustic-predicted), risk scores, semantic summaries, and session-level notes.

**Run locally:**
```bash
pip install -r requirements.txt
streamlit run dashboard_v2.py
```

---

## 2. Check-in App

**Live:** [mental-health-check-ins.vercel.app/dashboard](https://mental-health-check-ins.vercel.app/dashboard)

A web app participants use to complete structured check-ins. Each session walks through 8 steps: self-report ratings, free narration tasks (daily events, negative events, positive story reading), and a Stroop cognitive task — all recorded as audio. Source: [`Mental-Health-Check-ins/`](Mental-Health-Check-ins/)

---

## 3. Feature Extraction (Backend)

Source: [`pipeline/`](pipeline/)

Processes raw audio and transcripts from each check-in into structured features:

- **Acoustic features** — MFCCs, pitch, energy, jitter, shimmer, HNR, speaking rate via `pipeline/extract_features.py`
- **Transcription** — Whisper STT on narration steps via `pipeline/transcriber.py`
- **Semantic features** — Sentence-transformer embeddings compared against clinical prototype sentences (cognitive distortion, rumination, social withdrawal, etc.) via `pipeline/semantic_feature_extractor.py`

To process all participant audio from S3 and update check-in JSONs:
```bash
python process_real_checkins.py
python process_real_checkins.py --participant "Participant 19" --force
```

---

## 4. Models (Backend)

Three-stage modeling pipeline:

| Notebook | Model | Task |
|---|---|---|
| `model1_semantic_acoustic_to_phq_pcl.ipynb` | Ridge + RFECV | Predict PHQ/PCL severity from acoustic + semantic features |
| `model2_acoustic_to_stress.ipynb` | RidgeCV + RFECV | Predict within-session stress from acoustic features (trained on StressID) |
| `model3_ensemble_risk_score.ipynb` | Weighted fusion | Combine semantic score + behavioral deviation into a single risk score |

**Risk score formula:**

```
risk_score = 0.5 × sigmoid(deviation × 3) + 0.5 × semantic_score

deviation   = (reactivity − μ_reactivity) − (recovery − μ_recovery)
reactivity  = S5 − S1   (stress peak − baseline)
recovery    = S5 − S8   (stress peak − post-relaxation)
```

Deviation is within-person normalized against the participant's own prior sessions. Model experiments and visualizations are in `model_acoustic_experiments.ipynb`.

Trained models are stored in `pipeline/acoustic_stress_model.pkl` and `pipeline/acoustic_stress_model_meta.pkl`.

---

## 5. Data

```
data/
├── checkins/               # Per-participant check-in JSONs (anonymized)
│   └── Participant XX/
│       └── <checkin_id>.json
├── edaic/                  # E-DAIC-WOZ dataset (PHQ/PTSD labels + features)
├── stressid/               # StressID dataset (physiological stress labels)
├── daicwoz/                # DAIC-WOZ dataset
├── participant_map.csv     # Mapping: masked ID ↔ real participant ID (not committed)
└── audio_cache/            # Downloaded audio files (not committed)
```

Check-in JSONs contain: self-report stress curve, acoustic scores per step, semantic score, reactivity/recovery, risk score, transcripts, and audio URLs.

Raw audio and the StressID dataset are excluded from version control (see `.gitignore`). The `participant_map.csv` is kept locally only.
