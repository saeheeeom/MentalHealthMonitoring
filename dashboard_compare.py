"""
dashboard_compare.py  ·  MindTrack Participant Profile Dashboard
──────────────────────────────────────────────────────────────
Therapist-facing single-participant view using real daic_features_v4.csv data.
Select any participant; all charts compare that participant against the full cohort.

Run:
    streamlit run dashboard_compare.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import streamlit.components.v1 as components
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="MindTrack · Participant Profile",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ─────────────────────────────────────────────────────────────────────────────
# Paths
# ─────────────────────────────────────────────────────────────────────────────
ROOT        = Path(__file__).resolve().parent
DATA_CSV    = ROOT.parent.parent.parent / "daic_features_v4.csv"   # repo root
LABELS_CSV  = ROOT / "data" / "edaic" / "labels" / "detailed_labels.csv"

# ─────────────────────────────────────────────────────────────────────────────
# Design tokens  (same palette as dashboard.py)
# ─────────────────────────────────────────────────────────────────────────────
C = {
    "bg":        "#F7F3EE",
    "panel":     "#FFFFFF",
    "panel_alt": "#F9F6F1",
    "border":    "#E5DFD6",
    "green":     "#5F9B6B",
    "blue":      "#4E86B5",
    "amber":     "#CC8B52",
    "red":       "#B85A50",
    "text":      "#363330",
    "text_2":    "#766E68",
    "text_3":    "#A89F98",
}

_CL = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(family="Inter, system-ui, sans-serif", color=C["text"], size=12),
    margin=dict(l=4, r=4, t=28, b=4),
    xaxis=dict(showgrid=False, zeroline=False, showline=False),
    yaxis=dict(showgrid=True, gridcolor=C["border"], zeroline=False, showline=False),
    legend=dict(orientation="h", y=-0.2, x=0, font_size=11),
    hovermode="closest",
)

def _layout(**overrides) -> dict:
    return {**_CL, **overrides}

def _rgb(h: str) -> str:
    h = h.lstrip("#")
    return f"{int(h[0:2],16)},{int(h[2:4],16)},{int(h[4:6],16)}"

# ─────────────────────────────────────────────────────────────────────────────
# CSS  (same as dashboard.py)
# ─────────────────────────────────────────────────────────────────────────────
def inject_css() -> None:
    st.markdown(f"""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    .stApp {{ background:{C["bg"]}; font-family:'Inter',system-ui,sans-serif; }}
    .main .block-container {{ padding:1.4rem 2rem 2rem 2rem; max-width:1700px; }}
    header[data-testid="stHeader"] {{ display:none; }}
    #MainMenu, footer {{ visibility:hidden; }}
    .label {{
        font-size:0.68rem; font-weight:700; letter-spacing:0.07em;
        text-transform:uppercase; color:{C["text_3"]}; margin-bottom:0.35rem;
    }}
    .divider {{ border:none; border-top:1px solid {C["border"]}; margin:0.9rem 0; }}
    .card {{
        background:{C["panel"]}; border:1px solid {C["border"]};
        border-radius:14px; padding:1.1rem 1.3rem; margin-bottom:0.85rem;
    }}
    .card-alt {{
        background:{C["panel_alt"]}; border:1px solid {C["border"]};
        border-radius:14px; padding:1.1rem 1.3rem; margin-bottom:0.85rem;
    }}
    .badge {{
        display:inline-block; padding:0.28rem 0.85rem;
        border-radius:20px; font-size:0.82rem; font-weight:600;
    }}
    .pt-name {{ font-size:1.35rem; font-weight:700; color:{C["text"]}; line-height:1.2; }}
    .pt-sub  {{ font-size:0.8rem;  color:{C["text_2"]}; margin-top:0.1rem; }}
    .lang-row {{
        display:flex; align-items:flex-start; gap:0.5rem;
        padding:0.32rem 0; border-bottom:1px solid {C["border"]};
        font-size:0.82rem; color:{C["text"]};
    }}
    .lang-row:last-child {{ border-bottom:none; }}
    .dot {{
        width:7px; height:7px; border-radius:50%;
        background:{C["green"]}; flex-shrink:0; margin-top:5px;
    }}
    .s-head {{ font-size:1.1rem; font-weight:700; color:{C["text"]}; margin-bottom:0.15rem; }}
    .s-sub  {{ font-size:0.8rem; color:{C["text_2"]}; margin-bottom:1.1rem; }}
    .stButton > button {{
        border-radius:10px; font-family:'Inter',sans-serif;
        font-size:0.81rem; font-weight:500; text-align:left; width:100%;
        padding:0.55rem 0.85rem; margin-bottom:0.3rem;
        border:1px solid {C["border"]}; background:{C["panel"]}; color:{C["text"]};
        transition:background 0.15s, border-color 0.15s;
    }}
    .stButton > button:hover {{
        background:{C["panel_alt"]} !important;
        border-color:{C["green"]} !important;
    }}
    [data-testid="stMetric"] {{
        background:{C["panel"]}; border:1px solid {C["border"]};
        border-radius:12px; padding:0.9rem 1.1rem;
    }}
    [data-testid="stMetricLabel"] {{ color:{C["text_2"]} !important; font-size:0.75rem !important; }}
    [data-testid="stMetricValue"] {{
        color:{C["text"]} !important; font-size:1.35rem !important; font-weight:600 !important;
    }}
    .snippet {{
        background:{C["panel_alt"]}; border:1px solid {C["border"]};
        border-radius:10px; padding:0.75rem 1rem;
        font-size:0.82rem; color:{C["text_2"]}; line-height:1.6; margin-bottom:0.5rem;
    }}
    .ai-box {{
        background:{C["panel_alt"]}; border-left:3px solid {C["green"]};
        border-radius:0 10px 10px 0; padding:0.75rem 1rem;
        font-size:0.83rem; line-height:1.65; color:{C["text"]};
    }}
    /* ── Info tooltip ── */
    .tip {{
        display:inline-flex; align-items:center; justify-content:center;
        width:14px; height:14px; border-radius:50%;
        background:{C["border"]}; color:{C["text_3"]};
        font-size:9px; font-weight:700; font-style:normal;
        cursor:default; position:relative; vertical-align:middle;
        margin-left:4px; flex-shrink:0; line-height:1;
    }}
    .tip::after {{
        content:attr(data-tip);
        position:absolute; bottom:130%; left:50%;
        transform:translateX(-50%);
        background:{C["text"]}; color:#fff;
        padding:0.38rem 0.6rem; border-radius:7px;
        font-size:0.72rem; font-weight:400; font-family:'Inter',sans-serif;
        width:max-content; max-width:200px; white-space:normal;
        line-height:1.45; text-transform:none; letter-spacing:normal;
        pointer-events:none; opacity:0; transition:opacity 0.15s;
        z-index:9999; box-shadow:0 2px 8px rgba(0,0,0,0.18);
    }}
    .tip:hover::after {{ opacity:1; }}
    </style>
    """, unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# Feature column definitions
# ─────────────────────────────────────────────────────────────────────────────

SEM_COLS = [
    "patient_mean_proportion_negative_sentences",
    "patient_mean_avg_cognitive_distortion_score",
    "patient_mean_avg_catastrophic_thinking_score",
    "patient_mean_avg_self_blame_score",
    "patient_mean_avg_rumination_score",
    "patient_mean_avg_low_energy_or_fatigue_score",
    "patient_mean_avg_stress_or_anxiety_score",
    "patient_mean_avg_sleep_problems_score",
    "patient_mean_avg_social_withdrawal_score",
    "patient_mean_avg_social_connectedness_score",
    "patient_mean_avg_positive_affect_score",
]
SEM_LABELS = [
    "Negative sentences",
    "Cognitive distortion",
    "Catastrophic thinking",
    "Self-blame",
    "Rumination",
    "Low energy / fatigue",
    "Stress / anxiety",
    "Sleep problems",
    "Social withdrawal",
    "Social connectedness",
    "Positive affect",
]

VOICE_COLS = [
    "speech_rate_wpm",
    "pause_dur_mean",
    "pitch_instability",
    "f0_std",
    "hnr_mean",
    "breathiness",
    "spectral_flatness_mean",
    "intensity_std_db",
]
VOICE_LABELS = [
    "Speech rate (wpm)",
    "Pause duration (s)",
    "Pitch instability",
    "Pitch variance (F0 std)",
    "HNR (voice clarity)",
    "Breathiness",
    "Spectral flatness",
    "Energy variability",
]

PCA_COLS = SEM_COLS + VOICE_COLS

# +1 = higher raw value → more risk; -1 = higher raw value → less risk
RISK_WEIGHTS: dict[str, float] = {
    "patient_mean_proportion_negative_sentences": +1.0,
    "patient_mean_avg_cognitive_distortion_score": +1.0,
    "patient_mean_avg_stress_or_anxiety_score":    +1.0,
    "patient_mean_avg_rumination_score":           +1.0,
    "patient_mean_avg_social_withdrawal_score":    +1.0,
    "patient_mean_avg_positive_affect_score":      -1.0,
    "speech_rate_wpm":                             -1.0,
    "hnr_mean":                                    -1.0,
    "pitch_instability":                           +1.0,
    "pause_dur_mean":                              +1.0,
}

# Short descriptions shown in hover tooltips throughout the dashboard
FEATURE_DOCS: dict[str, str] = {
    # ── Voice / acoustic ──────────────────────────────────────────────────────
    "speech_rate_wpm":         "Words spoken per minute. Lower values may indicate slowed thinking or low energy.",
    "pause_dur_mean":          "Average duration of silent gaps between utterances (seconds). Longer pauses can signal cognitive load or low motivation.",
    "pitch_instability":       "Short-term variability in fundamental frequency. Higher values suggest emotional arousal or vocal strain.",
    "f0_std":                  "Standard deviation of fundamental frequency (pitch). Low variance — flat pitch — can indicate flat affect or depression.",
    "hnr_mean":                "Harmonics-to-Noise Ratio (dB). Higher = cleaner, more harmonic voice. Low values indicate breathiness or dysphonia.",
    "breathiness":             "Degree of aperiodic noise in the voice signal. Higher breathiness is linked to low vocal effort or sadness.",
    "spectral_flatness_mean":  "How noise-like the frequency spectrum is (0–1). Higher = more monotone, less tonal speech.",
    "intensity_std_db":        "Standard deviation of loudness (dB). Low variability means flat, low-affect speech with little emphasis.",
    # ── Semantic / language ───────────────────────────────────────────────────
    "patient_mean_proportion_negative_sentences":         "Proportion of sentences carrying negative sentiment in the participant's speech.",
    "patient_mean_avg_cognitive_distortion_score":        "Average score across cognitive distortion patterns (e.g., black-and-white thinking, catastrophising).",
    "patient_mean_avg_catastrophic_thinking_score":       "Tendency to anticipate worst-case outcomes and overestimate their likelihood.",
    "patient_mean_avg_self_blame_score":                  "Frequency of attributing negative events or outcomes to oneself.",
    "patient_mean_avg_rumination_score":                  "Repetitive, passive focus on negative thoughts or past experiences.",
    "patient_mean_avg_low_energy_or_fatigue_score":       "Language expressing tiredness, lack of motivation, or physical depletion.",
    "patient_mean_avg_stress_or_anxiety_score":           "Language reflecting worry, tension, or sense of overwhelm.",
    "patient_mean_avg_sleep_problems_score":              "References to difficulty falling or staying asleep, or feeling unrested.",
    "patient_mean_avg_social_withdrawal_score":           "Language indicating avoidance of or disconnection from social interaction.",
    "patient_mean_avg_social_connectedness_score":        "Language reflecting positive social bonds, support, or engagement.",
    "patient_mean_avg_positive_affect_score":             "Expressions of joy, hope, gratitude, or other positive emotional states.",
    # ── Clinical scores ───────────────────────────────────────────────────────
    "risk_score":              "Composite risk score (0–10) derived from 10 acoustic and language features. Higher = more at-risk.",
    "PHQ-8 Severity":          "Patient Health Questionnaire total score (0–24). ≥10 indicates probable depression.",
    "PCL-C Severity":          "PTSD Checklist total score (0–68). Higher scores indicate greater PTSD symptom burden.",
    "Depression_severity":     "PHQ-8 total severity score (0–24). Used to classify depression diagnosis.",
    "PTSD_severity":           "PCL-C total score (0–68). Measures PTSD symptom severity.",
}

# ─────────────────────────────────────────────────────────────────────────────
# Data loading & preprocessing  (cached)
# ─────────────────────────────────────────────────────────────────────────────

@st.cache_data
def load_data() -> pd.DataFrame:
    df = pd.read_csv(DATA_CSV)
    # Drop rows with missing patient_id, then normalise to plain int string
    df = df.dropna(subset=["patient_id"])
    df["patient_id"] = df["patient_id"].apply(lambda x: str(int(float(x))))

    # Merge labels if available
    if LABELS_CSV.exists():
        labels = pd.read_csv(LABELS_CSV)
        labels = labels.dropna(subset=["Participant"])
        labels["Participant"] = labels["Participant"].apply(lambda x: str(int(float(x))))
        keep = [c for c in [
            "Participant", "Depression_severity", "Depression_label",
            "PTSD_severity", "PTSD_label", "gender", "age", "split",
            "PHQ8_1_NoInterest", "PHQ8_2_Depressed", "PHQ8_3_Sleep",
            "PHQ8_4_Tired", "PHQ8_5_Appetite", "PHQ8_6_Failure",
            "PHQ8_7_Concentration", "PHQ8_8_Psychomotor",
        ] if c in labels.columns]
        df = df.merge(
            labels[keep],
            left_on="patient_id", right_on="Participant", how="left",
        ).drop(columns=["Participant"], errors="ignore")

    # Risk score (0–10, higher = more at-risk)
    col_mins = df[list(RISK_WEIGHTS.keys())].min()
    col_maxs = df[list(RISK_WEIGHTS.keys())].max()
    risk = np.zeros(len(df))
    for col, direction in RISK_WEIGHTS.items():
        normed = (df[col] - col_mins[col]) / (col_maxs[col] - col_mins[col] + 1e-9)
        risk += direction * normed
    risk = (risk - risk.min()) / (risk.max() - risk.min()) * 10
    df["risk_score"] = np.round(risk, 2)
    df["risk_label"] = pd.cut(
        df["risk_score"],
        bins=[0, 3.33, 6.67, 10],
        labels=["Low", "Moderate", "Elevated"],
        include_lowest=True,
    ).astype(str)

    # PCA for cohort map
    scaler = StandardScaler()
    z = scaler.fit_transform(df[PCA_COLS].fillna(df[PCA_COLS].mean()))
    pca = PCA(n_components=2, random_state=42)
    coords = pca.fit_transform(z)
    df["pca_x"] = coords[:, 0]
    df["pca_y"] = coords[:, 1]
    df["pca_var"] = f"{pca.explained_variance_ratio_.sum()*100:.1f}%"

    # Per-feature min-max normalisation to 0–10 (for bar charts)
    for col in VOICE_COLS + SEM_COLS:
        lo, hi = df[col].min(), df[col].max()
        df[col + "_n"] = np.clip((df[col] - lo) / (hi - lo + 1e-9) * 10, 0, 10)

    # Cohort percentile for every feature
    for col in VOICE_COLS + SEM_COLS:
        arr = df[col].values
        df[col + "_pct"] = df[col].apply(
            lambda v: float((arr < v).mean() * 100) if pd.notna(v) else np.nan
        )

    return df


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def risk_color(label: str) -> str:
    return {"Low": C["green"], "Moderate": C["amber"], "Elevated": C["red"]}.get(label, C["text_3"])

def _tip(key: str) -> str:
    """Return a small ⓘ HTML span whose hover shows the feature description."""
    desc = FEATURE_DOCS.get(key, "")
    if not desc:
        return ""
    safe = desc.replace('"', "&quot;")
    return f'<span class="tip" data-tip="{safe}">i</span>'


def _pct_badge(pct: float, invert: bool = False) -> str:
    """Return an HTML badge string for a percentile rank."""
    if np.isnan(pct):
        return "—"
    effective = (100 - pct) if invert else pct
    color = C["red"] if effective > 66 else C["amber"] if effective > 33 else C["green"]
    return (
        f'<span style="background:rgba({_rgb(color)},0.12);color:{color};'
        f'border-radius:6px;padding:0.12rem 0.45rem;font-size:0.73rem;font-weight:700;">'
        f'{pct:.0f}th pct</span>'
    )


# ─────────────────────────────────────────────────────────────────────────────
# Left panel
# ─────────────────────────────────────────────────────────────────────────────

def render_left_panel(df: pd.DataFrame) -> str:
    """Renders left panel, returns selected patient_id (str)."""

    # Patient selector
    st.markdown(f'<div class="label">Participant</div>', unsafe_allow_html=True)
    all_ids = sorted(df["patient_id"].tolist(), key=int)
    pid = st.selectbox(
        "Participant",
        options=all_ids,
        index=0,
        label_visibility="collapsed",
        key="patient_select",
    )

    row = df[df["patient_id"] == pid].iloc[0]

    # Name / meta header
    gender = str(row.get("gender", "")).capitalize()
    age    = row.get("age", None)
    split  = str(row.get("split", "")).capitalize()
    sub    = " · ".join(filter(None, [
        f"Age {int(age)}" if pd.notna(age) else "",
        gender if gender and gender.lower() not in ("nan", "") else "",
        split if split and split.lower() not in ("nan", "") else "",
    ]))
    st.markdown(f"""
    <div class="pt-name">Participant {pid}</div>
    <div class="pt-sub">{sub if sub else "E-DAIC participant"}</div>
    """, unsafe_allow_html=True)

    st.markdown("<div style='height:0.5rem'></div>", unsafe_allow_html=True)

    # Risk badge
    rc = risk_color(row["risk_label"])
    bg = f"rgba({_rgb(rc)},0.10)"
    st.markdown(f"""
    <div class="label">Risk level</div>
    <span class="badge" style="background:{bg};color:{rc};">
      ● &nbsp;{row['risk_label']}
      &nbsp;·&nbsp; {row['risk_score']:.1f}/10
    </span>
    """, unsafe_allow_html=True)

    # Depression / PTSD badges if available
    extra_html = ""
    dep_label = row.get("Depression_label", None)
    dep_sev   = row.get("Depression_severity", None)
    ptsd_label = row.get("PTSD_label", None)
    if pd.notna(dep_label):
        dep_c = C["red"] if dep_label == 1 else C["green"]
        dep_s = f" (sev. {int(dep_sev)})" if pd.notna(dep_sev) else ""
        extra_html += f"""
        <span class="badge" style="background:rgba({_rgb(dep_c)},0.10);color:{dep_c};margin-right:0.4rem;">
          {'Dep+' if dep_label == 1 else 'Dep−'}{dep_s}
        </span>"""
    if pd.notna(ptsd_label):
        ptsd_c = C["amber"] if ptsd_label == 1 else C["green"]
        extra_html += f"""
        <span class="badge" style="background:rgba({_rgb(ptsd_c)},0.10);color:{ptsd_c};">
          {'PTSD+' if ptsd_label == 1 else 'PTSD−'}
        </span>"""
    if extra_html:
        st.markdown(f'<div style="margin-top:0.4rem;">{extra_html}</div>', unsafe_allow_html=True)

    st.markdown('<hr class="divider">', unsafe_allow_html=True)

    # Top elevated language markers for this patient
    st.markdown('<div class="label">Key language markers</div>', unsafe_allow_html=True)
    label_map = dict(zip(SEM_COLS, SEM_LABELS))
    sem_vals = {col: row[col + "_n"] for col in SEM_COLS if col + "_n" in row.index}
    top_markers = sorted(sem_vals.items(), key=lambda x: -x[1])[:5]
    rows_html = ""
    for col, val in top_markers:
        level = "High" if val > 6 else "Moderate" if val > 3.5 else "Low"
        lc    = C["red"] if level == "High" else C["amber"] if level == "Moderate" else C["text_3"]
        rows_html += f"""
        <div class="lang-row">
          <div class="dot" style="background:{lc};"></div>
          <div><strong>{label_map[col]}</strong>{_tip(col)}&nbsp;
            <span style="color:{C['text_3']};font-size:0.76rem;">({level})</span>
          </div>
        </div>"""
    st.markdown(rows_html, unsafe_allow_html=True)

    st.markdown('<hr class="divider">', unsafe_allow_html=True)

    # Quick voice stats
    st.markdown('<div class="label">Voice snapshot</div>', unsafe_allow_html=True)
    snap_items = [
        ("Speech rate",    f"{row['speech_rate_wpm']:.0f} wpm",  "speech_rate_wpm"),
        ("Pause (mean)",   f"{row['pause_dur_mean']:.2f} s",       "pause_dur_mean"),
        ("HNR",            f"{row['hnr_mean']:.1f} dB",            "hnr_mean"),
        ("Pitch instab.",  f"{row['pitch_instability']:.2f}",      "pitch_instability"),
    ]
    snap_html = ""
    for label, val, col in snap_items:
        pct = row.get(col + "_pct", float("nan"))
        snap_html += f"""
        <div style="display:flex;justify-content:space-between;align-items:center;
                    padding:0.28rem 0;border-bottom:1px solid {C['border']};font-size:0.82rem;">
          <span style="color:{C['text_2']}">{label}{_tip(col)}</span>
          <span style="font-weight:600;color:{C['text']}">{val}&nbsp;{_pct_badge(pct)}</span>
        </div>"""
    st.markdown(snap_html, unsafe_allow_html=True)

    st.markdown('<hr class="divider">', unsafe_allow_html=True)

    # Cohort context
    dep_rate = df["Depression_label"].mean() * 100 if "Depression_label" in df else None
    c1, c2 = st.columns(2)
    with c1:
        st.metric("Cohort size", len(df))
    with c2:
        st.metric("Dep. rate", f"{dep_rate:.0f}%" if dep_rate else "—")

    return pid


# ─────────────────────────────────────────────────────────────────────────────
# View 1 — Patient Overview
# ─────────────────────────────────────────────────────────────────────────────

PHQ8_COLS = [
    "PHQ8_1_NoInterest", "PHQ8_2_Depressed", "PHQ8_3_Sleep",
    "PHQ8_4_Tired", "PHQ8_5_Appetite", "PHQ8_6_Failure",
    "PHQ8_7_Concentration", "PHQ8_8_Psychomotor",
]
PHQ8_LABELS = [
    "No interest", "Feeling depressed", "Sleep problems",
    "Fatigue", "Appetite", "Failure feelings",
    "Concentration", "Psychomotor",
]

def render_overview(df: pd.DataFrame, pid: str) -> None:
    row = df[df["patient_id"] == pid].iloc[0]

    st.markdown('<div class="s-head">Participant Overview</div>', unsafe_allow_html=True)
    st.markdown('<div class="s-sub">Summary of acoustic and language indicators · cohort percentiles shown in brackets</div>',
                unsafe_allow_html=True)

    # Top metrics row
    dep_sev  = row.get("Depression_severity", None)
    ptsd_sev = row.get("PTSD_severity", None)
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        cohort_rank = int((df["risk_score"] >= row["risk_score"]).sum())
        st.metric("Risk Score", f"{row['risk_score']:.1f}/10",
                  delta=f"Rank #{cohort_rank} of {len(df)} (high = worse)",
                  delta_color="off",
                  help=FEATURE_DOCS["risk_score"])
    with c2:
        sr_mean = df["speech_rate_wpm"].mean()
        sr_delta = row["speech_rate_wpm"] - sr_mean
        st.metric("Speech Rate", f"{row['speech_rate_wpm']:.0f} wpm",
                  delta=f"{sr_delta:+.0f} vs cohort avg",
                  delta_color="normal",
                  help=FEATURE_DOCS["speech_rate_wpm"])
    with c3:
        if pd.notna(dep_sev):
            st.metric("PHQ-8 Severity", f"{int(dep_sev)}/24",
                      help=FEATURE_DOCS["PHQ-8 Severity"])
        else:
            hnr_mean = df["hnr_mean"].mean()
            st.metric("HNR (voice clarity)", f"{row['hnr_mean']:.1f} dB",
                      delta=f"{row['hnr_mean'] - hnr_mean:+.1f} vs avg",
                      help=FEATURE_DOCS["hnr_mean"])
    with c4:
        if pd.notna(ptsd_sev):
            st.metric("PCL-C Severity", f"{int(ptsd_sev)}/68",
                      help=FEATURE_DOCS["PCL-C Severity"])
        else:
            pd_mean = df["pause_dur_mean"].mean()
            st.metric("Avg Pause Duration", f"{row['pause_dur_mean']:.2f} s",
                      delta=f"{row['pause_dur_mean'] - pd_mean:+.2f} vs avg",
                      delta_color="inverse",
                      help=FEATURE_DOCS["pause_dur_mean"])

    st.markdown("<br>", unsafe_allow_html=True)

    col_a, col_b = st.columns([1.3, 1])

    with col_a:
        # Voice quality radar: patient vs cohort mean
        v_norm_cols  = [c + "_n" for c in VOICE_COLS]
        patient_vals = [float(row[c]) for c in v_norm_cols]
        cohort_vals  = [float(df[c].mean()) for c in v_norm_cols]

        fig = go.Figure()
        fig.add_trace(go.Scatterpolar(
            r=cohort_vals + [cohort_vals[0]],
            theta=VOICE_LABELS + [VOICE_LABELS[0]],
            name="Cohort mean",
            fill="toself",
            fillcolor=f"rgba({_rgb(C['blue'])},0.06)",
            line=dict(color=C["blue"], width=1.5, dash="dot"),
            hovertemplate="%{theta}: %{r:.1f}<extra>Cohort mean</extra>",
        ))
        fig.add_trace(go.Scatterpolar(
            r=patient_vals + [patient_vals[0]],
            theta=VOICE_LABELS + [VOICE_LABELS[0]],
            name=f"Participant {pid}",
            fill="toself",
            fillcolor=f"rgba({_rgb(C['green'])},0.12)",
            line=dict(color=C["green"], width=2.5),
            hovertemplate="%{theta}: %{r:.1f}<extra>Participant " + pid + "</extra>",
        ))
        fig.update_layout(**_layout(
            height=340,
            polar=dict(
                bgcolor="rgba(0,0,0,0)",
                radialaxis=dict(visible=True, range=[0, 10],
                                showticklabels=False, gridcolor=C["border"]),
                angularaxis=dict(gridcolor=C["border"]),
            ),
            legend=dict(orientation="h", y=-0.1, x=0),
            title=dict(text="Voice quality profile vs cohort mean", font_size=12,
                       x=0, xanchor="left"),
        ))
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

    with col_b:
        # PHQ-8 item bar if available; else language radar
        phq_avail = [c for c in PHQ8_COLS if c in row.index and pd.notna(row.get(c))]
        if phq_avail:
            st.markdown('<div class="label">PHQ-8 item scores</div>', unsafe_allow_html=True)
            vals   = [int(row[c]) for c in phq_avail]
            labels = [PHQ8_LABELS[PHQ8_COLS.index(c)] for c in phq_avail]
            colors = [C["red"] if v >= 2 else C["amber"] if v == 1 else C["green"] for v in vals]
            fig2 = go.Figure(go.Bar(
                x=labels, y=vals,
                marker_color=colors, marker_line_width=0,
                hovertemplate="%{x}: %{y}<extra></extra>",
            ))
            fig2.update_layout(**_layout(
                height=310, bargap=0.3,
                xaxis=dict(tickangle=-35, showgrid=False),
                yaxis=dict(title="Score (0–3)", dtick=1),
            ))
            st.plotly_chart(fig2, use_container_width=True, config={"displayModeBar": False})
        else:
            # Language radar fallback
            s_norm_cols  = [c + "_n" for c in SEM_COLS]
            pt_lang_vals = [float(row[c]) for c in s_norm_cols]
            coh_lang_vals = [float(df[c].mean()) for c in s_norm_cols]

            fig2 = go.Figure()
            fig2.add_trace(go.Scatterpolar(
                r=coh_lang_vals + [coh_lang_vals[0]],
                theta=SEM_LABELS + [SEM_LABELS[0]],
                name="Cohort mean", fill="toself",
                fillcolor=f"rgba({_rgb(C['blue'])},0.06)",
                line=dict(color=C["blue"], width=1.5, dash="dot"),
            ))
            fig2.add_trace(go.Scatterpolar(
                r=pt_lang_vals + [pt_lang_vals[0]],
                theta=SEM_LABELS + [SEM_LABELS[0]],
                name=f"Participant {pid}", fill="toself",
                fillcolor=f"rgba({_rgb(C['amber'])},0.10)",
                line=dict(color=C["amber"], width=2),
            ))
            fig2.update_layout(**_layout(
                height=310,
                polar=dict(
                    bgcolor="rgba(0,0,0,0)",
                    radialaxis=dict(visible=True, range=[0, 10],
                                    showticklabels=False, gridcolor=C["border"]),
                    angularaxis=dict(gridcolor=C["border"]),
                ),
                title=dict(text="Language profile vs cohort mean", font_size=12, x=0, xanchor="left"),
            ))
            st.plotly_chart(fig2, use_container_width=True, config={"displayModeBar": False})

    # Risk score distribution with patient marker
    st.markdown('<div class="label" style="margin-top:0.2rem;">Risk score distribution · participant position</div>',
                unsafe_allow_html=True)
    fig3 = go.Figure()
    fig3.add_trace(go.Histogram(
        x=df["risk_score"], nbinsx=30,
        marker_color=f"rgba({_rgb(C['blue'])},0.35)",
        marker_line_width=0, name="All participants",
        hovertemplate="Risk ~%{x:.1f}: %{y} participants<extra></extra>",
    ))
    fig3.add_vline(
        x=row["risk_score"], line_color=C["green"], line_width=2.5,
        annotation_text=f"  Participant {pid} ({row['risk_score']:.1f})",
        annotation_position="top",
        annotation_font=dict(size=11, color=C["green"]),
    )
    fig3.add_vline(
        x=df["risk_score"].mean(), line_color=C["text_3"],
        line_width=1.5, line_dash="dot",
        annotation_text=f"  Avg ({df['risk_score'].mean():.1f})",
        annotation_position="bottom",
        annotation_font=dict(size=10, color=C["text_3"]),
    )
    fig3.update_layout(**_layout(
        height=200, showlegend=False,
        xaxis=dict(title="Risk score (0–10)", showgrid=False),
        yaxis=dict(title="Participants", showgrid=True, gridcolor=C["border"]),
        margin=dict(l=4, r=4, t=14, b=4),
    ))
    st.plotly_chart(fig3, use_container_width=True, config={"displayModeBar": False})


# ─────────────────────────────────────────────────────────────────────────────
# View 2 — Language Patterns
# ─────────────────────────────────────────────────────────────────────────────

def render_language_patterns(df: pd.DataFrame, pid: str) -> None:
    row = df[df["patient_id"] == pid].iloc[0]

    st.markdown('<div class="s-head">Language Patterns</div>', unsafe_allow_html=True)
    st.markdown('<div class="s-sub">Semantic markers extracted from speech transcripts · participant vs cohort mean · normalised 0–10 within cohort</div>',
                unsafe_allow_html=True)

    s_norm_cols   = [c + "_n" for c in SEM_COLS]
    patient_vals  = [float(row[c]) for c in s_norm_cols]
    cohort_vals   = [float(df[c].mean()) for c in s_norm_cols]

    col_l, col_r = st.columns([1.25, 1])

    with col_l:
        # Grouped bar: patient + cohort mean
        fig = go.Figure()
        bar_colors = [
            C["red"] if pv > cv * 1.3 and pv > 5
            else C["amber"] if pv > cv * 1.15
            else C["green"]
            for pv, cv in zip(patient_vals, cohort_vals)
        ]
        fig.add_trace(go.Bar(
            name=f"Participant {pid}",
            x=SEM_LABELS, y=patient_vals,
            marker_color=bar_colors, marker_line_width=0,
            hovertemplate="%{x}: %{y:.2f}<extra>Participant " + pid + "</extra>",
        ))
        # Cohort mean as scatter line overlay
        fig.add_trace(go.Scatter(
            name="Cohort mean",
            x=SEM_LABELS, y=cohort_vals,
            mode="markers",
            marker=dict(color=C["text_3"], size=8, symbol="line-ew-open",
                        line=dict(width=2, color=C["text_3"])),
            hovertemplate="%{x}: %{y:.2f}<extra>Cohort mean</extra>",
        ))
        fig.update_layout(**_layout(
            height=320, bargap=0.3,
            xaxis=dict(tickangle=-35, showgrid=False),
            yaxis_title="Score (0–10)",
            legend=dict(orientation="h", y=1.1, x=0),
        ))
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

        # Rumination & stress trend substitute: percentile bars
        st.markdown('<div class="label">Percentile rank vs cohort <span class="tip" data-tip="How this participant ranks among all cohort participants for each marker. 50th = cohort average.">i</span></div>', unsafe_allow_html=True)
        pct_vals  = [float(row[c + "_pct"]) for c in SEM_COLS]
        pct_colors = [
            C["red"] if p > 66 else C["amber"] if p > 33 else C["green"]
            for p in pct_vals
        ]
        fig_pct = go.Figure(go.Bar(
            x=SEM_LABELS, y=pct_vals,
            marker_color=pct_colors, marker_line_width=0,
            hovertemplate="%{x}: %{y:.0f}th percentile<extra></extra>",
        ))
        fig_pct.add_hline(y=50, line_dash="dot", line_color=C["text_3"],
                          annotation_text="50th", annotation_position="right")
        fig_pct.update_layout(**_layout(
            height=230, bargap=0.3,
            xaxis=dict(tickangle=-35, showgrid=False),
            yaxis=dict(title="Percentile", range=[0, 100]),
        ))
        st.plotly_chart(fig_pct, use_container_width=True, config={"displayModeBar": False})

    with col_r:
        # Radar overlay
        fig2 = go.Figure()
        fig2.add_trace(go.Scatterpolar(
            r=cohort_vals + [cohort_vals[0]],
            theta=SEM_LABELS + [SEM_LABELS[0]],
            name="Cohort mean", fill="toself",
            fillcolor=f"rgba({_rgb(C['blue'])},0.06)",
            line=dict(color=C["blue"], width=1.5, dash="dot"),
            hovertemplate="%{theta}: %{r:.2f}<extra>Cohort mean</extra>",
        ))
        fig2.add_trace(go.Scatterpolar(
            r=patient_vals + [patient_vals[0]],
            theta=SEM_LABELS + [SEM_LABELS[0]],
            name=f"Participant {pid}", fill="toself",
            fillcolor=f"rgba({_rgb(C['amber'])},0.10)",
            line=dict(color=C["amber"], width=2.5),
            hovertemplate="%{theta}: %{r:.2f}<extra>Participant " + pid + "</extra>",
        ))
        fig2.update_layout(**_layout(
            height=340,
            polar=dict(
                bgcolor="rgba(0,0,0,0)",
                radialaxis=dict(visible=True, range=[0, 10],
                                showticklabels=False, gridcolor=C["border"]),
                angularaxis=dict(gridcolor=C["border"]),
            ),
            legend=dict(orientation="h", y=-0.12, x=0),
        ))
        st.plotly_chart(fig2, use_container_width=True, config={"displayModeBar": False})

        # Raw value table
        st.markdown('<div class="label">Raw scores</div>', unsafe_allow_html=True)
        tbl_data = {
            "Marker": SEM_LABELS,
            "Participant": [f"{row[c]:.4f}" for c in SEM_COLS],
            "Cohort avg": [f"{df[c].mean():.4f}" for c in SEM_COLS],
            "Percentile": [f"{row[c+'_pct']:.0f}th" for c in SEM_COLS],
        }
        st.dataframe(pd.DataFrame(tbl_data), use_container_width=True, hide_index=True, height=230)


# ─────────────────────────────────────────────────────────────────────────────
# View 3 — Voice Features
# ─────────────────────────────────────────────────────────────────────────────

def render_voice_features(df: pd.DataFrame, pid: str) -> None:
    row = df[df["patient_id"] == pid].iloc[0]

    st.markdown('<div class="s-head">Voice Features</div>', unsafe_allow_html=True)
    st.markdown('<div class="s-sub">Paralinguistic indicators extracted from audio · participant vs cohort distribution</div>',
                unsafe_allow_html=True)

    # Feature picker
    col_pick, _ = st.columns([1, 2])
    with col_pick:
        chosen_label = st.selectbox(
            "Feature",
            options=VOICE_LABELS,
            label_visibility="collapsed",
            key="voice_feat_select",
        )
    chosen_col = VOICE_COLS[VOICE_LABELS.index(chosen_label)]
    # Show description of the selected feature below the picker
    chosen_doc = FEATURE_DOCS.get(chosen_col, "")
    if chosen_doc:
        st.caption(chosen_doc)

    col_a, col_b = st.columns(2)

    with col_a:
        # Violin by risk level with patient marker
        fig = go.Figure()
        for rl, rc in [("Low", C["green"]), ("Moderate", C["amber"]), ("Elevated", C["red"])]:
            sub = df[df["risk_label"] == rl][chosen_col].dropna()
            if sub.empty:
                continue
            fig.add_trace(go.Violin(
                y=sub, name=rl,
                line_color=rc,
                fillcolor=f"rgba({_rgb(rc)},0.12)",
                box_visible=True, meanline_visible=True,
                hovertemplate=f"{rl}: %{{y:.3f}}<extra></extra>",
            ))
        # Patient value as scatter point
        pt_val = row[chosen_col]
        pt_rl  = str(row["risk_label"])
        if pd.notna(pt_val):
            fig.add_trace(go.Scatter(
                x=[pt_rl], y=[pt_val],
                mode="markers+text",
                showlegend=False,
                marker=dict(color=C["green"], size=14, symbol="diamond",
                            line=dict(color="white", width=1.5)),
                text=[f"  P{pid}"],
                textposition="middle right",
                textfont=dict(size=10, color=C["green"]),
                hovertemplate=f"Participant {pid}: %{{y:.3f}}<extra></extra>",
            ))
        fig.update_layout(**_layout(
            height=340, yaxis_title=chosen_label,
            xaxis=dict(title="Risk level", showgrid=False),
            violinmode="group",
        ))
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

    with col_b:
        # Histogram with patient vertical line
        vals = df[chosen_col].dropna()
        fig2 = go.Figure()
        fig2.add_trace(go.Histogram(
            x=vals, nbinsx=30,
            marker_color=f"rgba({_rgb(C['blue'])},0.4)",
            marker_line_width=0, name="All participants",
            hovertemplate="%{x:.3f} · %{y} participants<extra></extra>",
        ))
        if pd.notna(row[chosen_col]):
            fig2.add_vline(
                x=row[chosen_col], line_color=C["green"], line_width=2.5,
                annotation_text=f"  P{pid} ({row[chosen_col]:.2f})",
                annotation_position="top",
                annotation_font=dict(size=10, color=C["green"]),
            )
        fig2.add_vline(
            x=vals.mean(), line_color=C["text_3"], line_width=1.5, line_dash="dot",
            annotation_text=f"  avg ({vals.mean():.2f})",
            annotation_position="bottom",
            annotation_font=dict(size=10, color=C["text_3"]),
        )
        fig2.update_layout(**_layout(
            height=340, xaxis_title=chosen_label, yaxis_title="Count",
            showlegend=False,
        ))
        st.plotly_chart(fig2, use_container_width=True, config={"displayModeBar": False})

    # All 8 voice features grouped bar vs cohort mean
    st.markdown('<div class="label" style="margin-top:0.2rem;">All voice features · participant vs cohort mean (normalised 0–10) <span class="tip" data-tip="Each feature is scaled to 0–10 within the cohort so different units can be compared side by side.">i</span></div>',
                unsafe_allow_html=True)
    v_norm_cols  = [c + "_n" for c in VOICE_COLS]
    patient_vals = [float(row[c]) for c in v_norm_cols]
    cohort_vals  = [float(df[c].mean()) for c in v_norm_cols]

    fig3 = go.Figure()
    fig3.add_trace(go.Bar(
        name=f"Participant {pid}", x=VOICE_LABELS, y=patient_vals,
        marker_color=C["green"], marker_line_width=0, opacity=0.85,
        hovertemplate="%{x}: %{y:.2f}<extra>Participant " + pid + "</extra>",
    ))
    fig3.add_trace(go.Bar(
        name="Cohort mean", x=VOICE_LABELS, y=cohort_vals,
        marker_color=C["blue"], marker_line_width=0, opacity=0.45,
        hovertemplate="%{x}: %{y:.2f}<extra>Cohort mean</extra>",
    ))
    fig3.update_layout(**_layout(
        height=260, barmode="group", bargap=0.25,
        xaxis=dict(tickangle=-30, showgrid=False),
        yaxis_title="Score (0–10)",
        legend=dict(orientation="h", y=1.1, x=0),
    ))
    st.plotly_chart(fig3, use_container_width=True, config={"displayModeBar": False})

    # Raw value table
    st.markdown('<div class="label">Raw values</div>', unsafe_allow_html=True)
    tbl_data = {
        "Feature": VOICE_LABELS,
        "Participant": [f"{row[c]:.3f}" for c in VOICE_COLS],
        "Cohort avg": [f"{df[c].mean():.3f}" for c in VOICE_COLS],
        "Cohort SD": [f"{df[c].std():.3f}" for c in VOICE_COLS],
        "Percentile": [f"{row[c+'_pct']:.0f}th" for c in VOICE_COLS],
    }
    st.dataframe(pd.DataFrame(tbl_data), use_container_width=True, hide_index=True, height=200)


# ─────────────────────────────────────────────────────────────────────────────
# View 4 — Cohort Position
# ─────────────────────────────────────────────────────────────────────────────

def render_cohort_position(df: pd.DataFrame, pid: str) -> None:
    row = df[df["patient_id"] == pid].iloc[0]

    st.markdown('<div class="s-head">Cohort Position</div>', unsafe_allow_html=True)
    var = df["pca_var"].iloc[0]
    st.markdown(f'<div class="s-sub">Where this participant sits among all {len(df)} participants · PCA projection ({var} variance explained)</div>',
                unsafe_allow_html=True)

    col_a, col_b = st.columns([2.2, 1])

    with col_a:
        fig = go.Figure()

        # Background scatter by risk level
        for rl, rc in [("Low", C["green"]), ("Moderate", C["amber"]), ("Elevated", C["red"])]:
            sub = df[df["risk_label"] == rl]
            is_pt = sub["patient_id"] == pid
            fig.add_trace(go.Scatter(
                x=sub[~is_pt]["pca_x"], y=sub[~is_pt]["pca_y"],
                mode="markers", name=rl,
                marker=dict(
                    color=f"rgba({_rgb(rc)},0.35)",
                    size=6, line=dict(width=0),
                ),
                customdata=np.stack([
                    sub[~is_pt]["patient_id"],
                    sub[~is_pt]["risk_score"].round(1),
                ], axis=1),
                hovertemplate="Participant %{customdata[0]}<br>Risk: %{customdata[1]}/10<extra></extra>",
            ))

        # Highlight patient
        fig.add_trace(go.Scatter(
            x=[row["pca_x"]], y=[row["pca_y"]],
            mode="markers+text", showlegend=False,
            marker=dict(color=C["green"], size=18,
                        line=dict(color="white", width=2.5)),
            text=[f"  Participant {pid}"],
            textposition="middle right",
            textfont=dict(size=12, color=C["green"], family="Inter"),
            hovertemplate=f"<b>Participant {pid}</b><br>Risk: {row['risk_score']:.1f}/10<extra></extra>",
        ))

        fig.update_layout(**_layout(
            height=400,
            xaxis_title="PC 1", yaxis_title="PC 2",
            legend=dict(orientation="h", y=1.05, x=0),
        ))
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

    with col_b:
        st.markdown('<div class="label">Risk distribution</div>', unsafe_allow_html=True)
        counts = df["risk_label"].value_counts().reindex(["Elevated","Moderate","Low"]).fillna(0)
        fig2 = go.Figure(go.Bar(
            x=counts.values, y=counts.index,
            orientation="h",
            marker_color=[C["red"], C["amber"], C["green"]],
            marker_line_width=0,
            hovertemplate="%{y}: %{x} participants<extra></extra>",
        ))
        fig2.update_layout(**_layout(
            height=175,
            xaxis_title="Participants",
            yaxis=dict(showgrid=False, zeroline=False),
            margin=dict(l=0, r=0, t=10, b=0),
            showlegend=False,
        ))
        st.plotly_chart(fig2, use_container_width=True, config={"displayModeBar": False})

        # Patient risk rank
        rank = int((df["risk_score"] >= row["risk_score"]).sum())
        rc   = risk_color(row["risk_label"])
        st.markdown(f"""
        <div class="card-alt" style="margin-top:0.3rem;">
          <div class="label">Participant rank</div>
          <div style="font-size:1.1rem;font-weight:700;color:{C['text']};">
            #{rank} of {len(df)}
          </div>
          <div style="font-size:0.8rem;color:{C['text_2']};margin-top:0.2rem;">
            <span style="color:{rc};font-weight:600;">{row['risk_label']} risk</span>
            &nbsp;·&nbsp; {row['risk_score']:.1f}/10
          </div>
        </div>
        """, unsafe_allow_html=True)

    # Full percentile table
    st.markdown('<div class="label" style="margin-top:0.2rem;">Feature percentile ranks <span class="tip" data-tip="Percentile = % of cohort participants with a lower value. 50th = cohort median.">i</span></div>',
                unsafe_allow_html=True)
    all_cols    = VOICE_COLS + SEM_COLS
    all_labels  = VOICE_LABELS + SEM_LABELS
    pct_rows = []
    for col, label in zip(all_cols, all_labels):
        pct  = float(row[col + "_pct"])
        val  = float(row[col])
        mean = float(df[col].mean())
        pct_rows.append({
            "Feature": label,
            "Participant value": round(val, 4),
            "Cohort mean": round(mean, 4),
            "Percentile": f"{pct:.0f}th",
        })
    pct_df = pd.DataFrame(pct_rows)
    st.dataframe(pct_df, use_container_width=True, hide_index=True, height=340)


# ─────────────────────────────────────────────────────────────────────────────
# View 5 — Session Notes
# ─────────────────────────────────────────────────────────────────────────────

def render_session_notes(df: pd.DataFrame, pid: str) -> None:
    row = df[df["patient_id"] == pid].iloc[0]

    st.markdown('<div class="s-head">Session Notes</div>', unsafe_allow_html=True)
    st.markdown('<div class="s-sub">Notes persist within this browser session only.</div>',
                unsafe_allow_html=True)

    note_key = f"session_notes_{pid}"
    if note_key not in st.session_state:
        st.session_state[note_key] = ""

    col_note, col_ctx = st.columns([2.2, 1])

    with col_note:
        notes = st.text_area(
            "Notes",
            value=st.session_state[note_key],
            height=320,
            placeholder=(
                "Type your session observations here…\n\n"
                "E.g.: Participant showed elevated rumination markers. "
                "Speech rate below cohort average. PHQ-8 responses indicate sleep disruption. "
                "Plan to explore CBT strategies next session."
            ),
            label_visibility="collapsed",
        )
        if notes != st.session_state[note_key]:
            st.session_state[note_key] = notes

        s1, s2, _ = st.columns([1, 1, 2.5])
        with s1:
            if st.button("Save note", use_container_width=True):
                st.success("Note saved to session.")
        with s2:
            if st.button("Clear", use_container_width=True):
                st.session_state[note_key] = ""
                st.rerun()

    with col_ctx:
        dep_label = row.get("Depression_label", None)
        ptsd_label = row.get("PTSD_label", None)
        dep_sev   = row.get("Depression_severity", None)
        ptsd_sev  = row.get("PTSD_severity", None)
        gender    = str(row.get("gender", "")).capitalize()
        age       = row.get("age", None)
        split_    = str(row.get("split", "")).capitalize()

        context_rows = f"""
        <b>Participant ID:</b> {pid}<br>
        <b>Risk level:</b> {row['risk_label']} ({row['risk_score']:.1f}/10)<br>
        """
        if pd.notna(dep_label):
            context_rows += f"<b>Depression:</b> {'Positive' if dep_label == 1 else 'Negative'}"
            if pd.notna(dep_sev):
                context_rows += f" (PHQ-8 sev. {int(dep_sev)})"
            context_rows += "<br>"
        if pd.notna(ptsd_label):
            context_rows += f"<b>PTSD:</b> {'Positive' if ptsd_label == 1 else 'Negative'}"
            if pd.notna(ptsd_sev):
                context_rows += f" (PCL-C sev. {int(ptsd_sev)})"
            context_rows += "<br>"
        if gender and gender.lower() not in ("nan", ""):
            context_rows += f"<b>Gender:</b> {gender}<br>"
        if pd.notna(age):
            context_rows += f"<b>Age:</b> {int(age)}<br>"
        if split_ and split_.lower() not in ("nan", ""):
            context_rows += f"<b>Split:</b> {split_}<br>"

        st.markdown(f"""
        <div class="card-alt">
          <div class="label">Participant context</div>
          <div style="font-size:0.82rem;line-height:1.75;color:{C['text']};">
            {context_rows}
          </div>
        </div>
        """, unsafe_allow_html=True)

        # Top 3 elevated markers for quick reference
        s_norm_cols  = [c + "_n" for c in SEM_COLS]
        marker_vals  = {lab: float(row[col]) for lab, col in zip(SEM_LABELS, s_norm_cols)}
        top3 = sorted(marker_vals.items(), key=lambda x: -x[1])[:3]
        top3_html = "".join(
            f'<div style="font-size:0.8rem;padding:0.18rem 0;">'
            f'{"🔴" if v > 6 else "🟡" if v > 3.5 else "🟢"} {lab} ({v:.1f}/10)</div>'
            for lab, v in top3
        )
        st.markdown(f"""
        <div class="card-alt">
          <div class="label">Top language markers</div>
          {top3_html}
        </div>
        """, unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# Navigation
# ─────────────────────────────────────────────────────────────────────────────

VIEWS = [
    ("📋  Participant Overview",   "overview"),
    ("🗣  Language Patterns",   "language"),
    ("🎙  Voice Features",      "voice"),
    ("🗺  Cohort Position",     "cohort"),
    ("📝  Session Notes",       "notes"),
]

RENDER = {
    "overview": render_overview,
    "language": render_language_patterns,
    "voice":    render_voice_features,
    "cohort":   render_cohort_position,
    "notes":    render_session_notes,
}


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    inject_css()

    if "active_view" not in st.session_state:
        st.session_state.active_view = "overview"

    df = load_data()

    # App header
    st.markdown(f"""
    <div style="display:flex;align-items:center;gap:0.55rem;margin-bottom:1.4rem;">
      <span style="font-size:1.4rem;">🌿</span>
      <span style="font-size:1.2rem;font-weight:700;color:{C['text']};">MindTrack</span>
      <span style="font-size:0.74rem;color:{C['text_3']};margin-left:0.15rem;
                   background:{C['border']};padding:0.15rem 0.6rem;border-radius:8px;">
        Participant Profile
      </span>
    </div>
    """, unsafe_allow_html=True)

    left_col, nav_col, content_col = st.columns([1.35, 0.72, 3.6])

    with left_col:
        pid = render_left_panel(df)

    with nav_col:
        st.markdown('<div class="label">Views</div>', unsafe_allow_html=True)
        for label, key in VIEWS:
            if st.button(label, key=f"nav_{key}", use_container_width=True):
                st.session_state.active_view = key
                st.rerun()
        # Highlight the active button green via JS
        _active = st.session_state.active_view
        _kw = {"overview": "Overview", "language": "Language",
               "voice": "Voice Features", "cohort": "Cohort", "notes": "Session Notes"}
        components.html(f"""<script>
        (function() {{
            var key = "{_active}", kw = {str(_kw).replace("'", '"')}, green = "{C['green']}";
            function apply() {{
                parent.document.querySelectorAll('[data-testid="stButton"] > button').forEach(function(b) {{
                    Object.keys(kw).forEach(function(k) {{
                        if (b.innerText.indexOf(kw[k]) !== -1) {{
                            if (k === key) {{
                                b.style.setProperty('background', green, 'important');
                                b.style.setProperty('color', '#fff', 'important');
                                b.style.setProperty('border-color', green, 'important');
                            }} else {{
                                b.style.removeProperty('background');
                                b.style.removeProperty('color');
                                b.style.removeProperty('border-color');
                            }}
                        }}
                    }});
                }});
            }}
            apply(); setTimeout(apply, 80); setTimeout(apply, 300);
        }})();
        </script>""", height=0)

    with content_col:
        RENDER[st.session_state.active_view](df=df, pid=pid)


if __name__ == "__main__":
    main()
