"""
dashboard_v2.py  ·  MindTrack Therapist Dashboard
──────────────────────────────────────────────────
Three views:
  1. Overview  — risk trend, outlier alerts, participant summary (landing page)
  2. Check-in Details  — per-session stress arc, semantics, voice, transcripts
  3. Session Notes  — free-text notes for the therapist

Data source:
  data/checkins/{participant_id}/*.json  →  real pipeline output
  Falls back to synthetic mock data for demo / development.

Run:
    streamlit run dashboard_v2.py
"""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="MindTrack · Therapist Dashboard",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── Design tokens ─────────────────────────────────────────────────────────────
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
    legend=dict(orientation="h", y=-0.22, x=0, font_size=11),
    hovermode="x unified",
)


def _rgb(h: str) -> str:
    h = h.lstrip("#")
    return f"{int(h[0:2],16)},{int(h[2:4],16)},{int(h[4:6],16)}"


def _layout(**kw) -> dict:
    return {**_CL, **kw}


# ── CSS ───────────────────────────────────────────────────────────────────────
def inject_css() -> None:
    st.markdown(f"""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

    .stApp {{ background:{C["bg"]}; font-family:'Inter',system-ui,sans-serif; }}
    .main .block-container {{ padding:1.4rem 2rem 2rem 2rem; max-width:1700px; }}
    header[data-testid="stHeader"] {{ display:none; }}
    #MainMenu, footer {{ visibility:hidden; }}

    .card {{
        background:{C["panel"]}; border:1px solid {C["border"]};
        border-radius:14px; padding:1.1rem 1.3rem; margin-bottom:0.85rem;
    }}
    .card-alt {{
        background:{C["panel_alt"]}; border:1px solid {C["border"]};
        border-radius:14px; padding:1.1rem 1.3rem; margin-bottom:0.85rem;
    }}
    .label {{
        font-size:0.68rem; font-weight:700; letter-spacing:0.07em;
        text-transform:uppercase; color:{C["text_3"]}; margin-bottom:0.35rem;
    }}
    .divider {{ border:none; border-top:1px solid {C["border"]}; margin:0.9rem 0; }}
    .badge {{
        display:inline-block; padding:0.28rem 0.85rem;
        border-radius:20px; font-size:0.82rem; font-weight:600;
        letter-spacing:0.02em;
    }}
    .pt-name {{ font-size:1.35rem; font-weight:700; color:{C["text"]}; line-height:1.2; }}
    .pt-sub  {{ font-size:0.8rem; color:{C["text_2"]}; margin-top:0.1rem; }}
    .alert-box {{
        border-radius:10px; padding:0.65rem 1rem; margin-bottom:0.55rem;
        font-size:0.82rem; font-weight:500; line-height:1.5;
        border-left:4px solid;
    }}
    .alert-high     {{ background:rgba({_rgb(C["red"])},0.08);   border-color:{C["red"]};   color:{C["red"]}; }}
    .alert-moderate {{ background:rgba({_rgb(C["amber"])},0.08); border-color:{C["amber"]}; color:{C["amber"]}; }}
    .alert-low      {{ background:rgba({_rgb(C["green"])},0.08); border-color:{C["green"]}; color:{C["green"]}; }}
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
    .snippet {{
        background:{C["panel_alt"]}; border:1px solid {C["border"]};
        border-radius:10px; padding:0.75rem 1rem;
        font-size:0.82rem; font-style:italic;
        color:{C["text_2"]}; line-height:1.6; margin-bottom:0.5rem;
    }}
    .risk-number {{
        font-size:2.8rem; font-weight:700; line-height:1;
    }}
    .s-head {{ font-size:1.1rem; font-weight:700; color:{C["text"]}; margin-bottom:0.15rem; }}
    .s-sub  {{ font-size:0.8rem; color:{C["text_2"]}; margin-bottom:1.1rem; }}
    .stButton > button {{
        border-radius:10px; font-family:'Inter',sans-serif;
        font-size:0.81rem; font-weight:500; text-align:left; width:100%;
        padding:0.55rem 0.85rem; margin-bottom:0.3rem;
        border:1px solid {C["border"]};
        background:{C["panel"]}; color:{C["text"]};
        transition:background 0.15s,border-color 0.15s;
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
    [data-testid="stMetricValue"] {{ color:{C["text"]} !important; font-size:1.35rem !important; font-weight:600 !important; }}
    .timeline-item {{
        display:flex; align-items:center; gap:0.8rem;
        padding:0.5rem 0.7rem; border-radius:9px; cursor:pointer;
        transition:background 0.12s;
    }}
    .timeline-item:hover {{ background:{C["panel_alt"]}; }}
    .timeline-dot {{
        width:10px; height:10px; border-radius:50%; flex-shrink:0;
    }}
    </style>
    """, unsafe_allow_html=True)


# ── Feature label map ─────────────────────────────────────────────────────────
FEAT_LABEL = {
    "avg_cognitive_distortion_score":  "Cognitive distortion",
    "avg_catastrophic_thinking_score": "Catastrophic thinking",
    "avg_self_blame_score":            "Self-blame",
    "avg_rumination_score":            "Rumination",
    "avg_stress_or_anxiety_score":     "Stress / anxiety",
    "avg_low_energy_or_fatigue_score": "Low energy / fatigue",
    "avg_sleep_problems_score":        "Sleep problems",
    "avg_social_withdrawal_score":     "Social withdrawal",
    "avg_positive_affect_score":       "Positive affect",
    "avg_social_connectedness_score":  "Social connectedness",
    "proportion_negative_sentences":   "Negative sentences",
}


def _risk_color(tier: str | None) -> str:
    return {"High": C["red"], "Moderate": C["amber"], "Low": C["green"]}.get(tier or "", C["text_3"])


# ── Mock data ─────────────────────────────────────────────────────────────────
def _mock_checkins(n: int = 8) -> list[dict]:
    rng   = np.random.default_rng(42)
    today = datetime.now(timezone.utc).replace(hour=20, minute=0, second=0, microsecond=0)

    checkins = []
    for i in range(n):
        date   = today - timedelta(days=(n - 1 - i) * 3)
        s1_raw = int(np.clip(round(rng.normal(2.5, 0.7)), 1, 5))
        s5_raw = int(np.clip(round(s1_raw + rng.normal(1.2 - i * 0.15, 0.5)), 1, 5))
        s8_raw = int(np.clip(round(s5_raw - rng.normal(1.0 + i * 0.1, 0.4)), 1, 5))

        S1 = (s1_raw - 1) / 4
        S5 = (s5_raw - 1) / 4
        S8 = (s8_raw - 1) / 4

        react  = round(float(S5 - S1), 3)
        recov  = round(float(S5 - S8), 3)
        sem    = round(float(np.clip(0.55 - i * 0.04 + rng.normal(0, 0.06), 0.0, 1.0)), 3)

        if i >= 2:
            hist_r = [checkins[j]["reactivity"] for j in range(i)]
            hist_c = [checkins[j]["recovery"]   for j in range(i)]
            dev    = (react - np.mean(hist_r)) - (recov - np.mean(hist_c))
            ac_c   = float(1 / (1 + np.exp(-3 * dev)))
            rs     = round(0.5 * ac_c + 0.5 * sem, 4)
            tier   = "High" if rs >= 0.65 else "Moderate" if rs >= 0.45 else "Low"
        else:
            rs, tier = None, None

        top_feats = [
            ("avg_stress_or_anxiety_score",     round(float(np.clip(sem * 1.1 + rng.normal(0, 0.05), 0, 1)), 3)),
            ("avg_rumination_score",            round(float(np.clip(sem * 0.9 + rng.normal(0, 0.05), 0, 1)), 3)),
            ("avg_cognitive_distortion_score",  round(float(np.clip(sem * 0.85 + rng.normal(0, 0.05), 0, 1)), 3)),
            ("avg_self_blame_score",            round(float(np.clip(sem * 0.75 + rng.normal(0, 0.05), 0, 1)), 3)),
            ("avg_catastrophic_thinking_score", round(float(np.clip(sem * 0.70 + rng.normal(0, 0.05), 0, 1)), 3)),
        ]

        # Synthetic acoustic stress prediction (slightly offset from self-report)
        ac_step2 = round(float(np.clip(S1 + rng.normal(0.05 * (1 if i < 4 else -1), 0.1), 0, 1)), 3)
        ac_peak  = round(float(np.clip(S5 + rng.normal(-0.18 * (1 if i < 4 else 0.05), 0.08), 0, 1)), 3)
        ac_step7 = round(float(np.clip(S8 + rng.normal(0.04, 0.07), 0, 1)), 3)

        d_baseline  = round(S1 - ac_step2, 3)
        d_peak      = round(S5 - ac_peak,  3)
        d_postrelax = round(S8 - ac_step7, 3)
        d_flag      = abs(d_peak) > 0.20
        d_dir       = ("over" if d_peak > 0.20 else "under" if d_peak < -0.20 else None)

        checkins.append({
            "participant_id":    "P001",
            "checkin_id":        f"checkin_{i+1:03d}",
            "timestamp":         date.isoformat(),
            "self_reports": {
                "step1_baseline":        s1_raw,
                "step5a_after_stroop":   s5_raw,
                "step5b_after_negative": s5_raw,
                "step8_post_relax":      s8_raw,
            },
            "stress_curve": {
                "S1_baseline":   round(S1, 3),
                "S5_peak":       round(S5, 3),
                "S8_post_relax": round(S8, 3),
            },
            "acoustic_scores": {
                "step2_daily_narration": ac_step2,
                "step3_stroop":          ac_peak,
                "step4_negative_event":  ac_peak,
                "step7_positive_reading":ac_step7,
            },
            "discordance": {
                "discordance_baseline":  d_baseline,
                "discordance_peak":      d_peak,
                "discordance_postrelax": d_postrelax,
                "discordance_peak_abs":  round(abs(d_peak), 3),
                "discordance_flag":      d_flag,
                "discordance_direction": d_dir,
            },
            "reactivity":        react,
            "recovery":          recov,
            "semantic_score":    sem,
            "top_risk_features": top_feats,
            "risk_score":        rs,
            "risk_tier":         tier,
            "transcript_step2":  "I've been feeling pretty overwhelmed this week. A lot of things piled up at once." if i < 4 else "Things have been a bit better. Still busy but I feel more on top of it.",
            "transcript_step4":  "There was a conflict with my professor that really got to me. I kept replaying it in my head." if i < 4 else "Had a small disagreement with a friend but we talked it out.",
            "acoustic_summary": {
                "speech_rate_wpm": round(float(np.clip(115 + i * 4 + rng.normal(0, 8), 60, 200)), 1),
                "f0_mean":         round(float(np.clip(165 - i * 2 + rng.normal(0, 10), 80, 250)), 1),
                "f0_std":          round(float(np.clip(32 + i * 1.5 + rng.normal(0, 5), 10, 90)), 1),
                "hnr_mean":        round(float(np.clip(7.5 + i * 0.4 + rng.normal(0, 1), -5, 20)), 2),
                "pause_dur_mean":  round(float(np.clip(7.0 - i * 0.3 + rng.normal(0, 0.5), 2, 12)), 2),
            },
        })

    return checkins


# ── Data loading ──────────────────────────────────────────────────────────────
CHECKIN_DATA_DIR = Path(__file__).parent / "data" / "checkins"


def mask_id(participant_id: str) -> str:
    """Folders are already anonymized — identity function kept for call-site compatibility."""
    return participant_id


def available_participants() -> list[str]:
    """Return participant IDs (already anonymized folder names), sorted."""
    if not CHECKIN_DATA_DIR.exists():
        return []
    return sorted(
        p.name for p in CHECKIN_DATA_DIR.iterdir()
        if p.is_dir() and any(p.glob("*.json"))
    )


def load_checkins(participant_id: str = "Participant 01") -> list[dict]:
    folder = CHECKIN_DATA_DIR / participant_id
    if folder.exists():
        files = sorted(folder.glob("*.json"))
        if files:
            data = []
            for f in files:
                try:
                    data.append(json.loads(f.read_text()))
                except Exception:
                    pass
            if data:
                return sorted(data, key=lambda x: x.get("timestamp", ""))
    return _mock_checkins()


def checkins_to_df(checkins: list[dict]) -> pd.DataFrame:
    rows = []
    for c in checkins:
        sc = c.get("stress_curve", {})
        row = {
            "date":           pd.to_datetime(c["timestamp"]).tz_convert(None)
                              if pd.to_datetime(c["timestamp"]).tzinfo
                              else pd.to_datetime(c["timestamp"]),
            "checkin_id":     c["checkin_id"],
            "reactivity":     c.get("reactivity"),
            "recovery":       c.get("recovery"),
            "semantic_score": c.get("semantic_score"),
            "risk_score":     c.get("risk_score"),
            "risk_tier":      c.get("risk_tier"),
            "S1_baseline":    sc.get("S1_baseline"),
            "S5_peak":        sc.get("S5_peak"),
            "S8_post_relax":  sc.get("S8_post_relax"),
            "discordance_peak":      c.get("discordance", {}).get("discordance_peak"),
            "discordance_flag":      c.get("discordance", {}).get("discordance_flag"),
            "discordance_direction": c.get("discordance", {}).get("discordance_direction"),
        }
        for k, v in c.get("acoustic_summary", {}).items():
            row[k] = v
        rows.append(row)
    df = pd.DataFrame(rows)
    df["date"] = pd.to_datetime(df["date"]).dt.tz_localize(None)
    return df


# ── Outlier detection ─────────────────────────────────────────────────────────
def detect_flags(checkins: list[dict], threshold_sd: float = 1.5) -> dict[str, bool]:
    """Flag latest session metrics that deviate > threshold_sd from personal mean."""
    if len(checkins) < 3:
        return {}
    history = checkins[:-1]
    current = checkins[-1]
    flags   = {}
    for key in ("reactivity", "recovery", "semantic_score"):
        vals = [h[key] for h in history if h.get(key) is not None]
        cur  = current.get(key)
        if len(vals) < 2 or cur is None:
            continue
        μ, σ = float(np.mean(vals)), float(np.std(vals))
        if σ < 1e-6:
            continue
        z = (cur - μ) / σ
        # reactivity + semantic: high = risky; recovery: low = risky
        flags[key] = z > threshold_sd if key != "recovery" else z < -threshold_sd
    return flags


# ── Chart helpers ─────────────────────────────────────────────────────────────
def sparkline(dates, values, color: str, height: int = 80) -> go.Figure:
    fig = go.Figure(go.Scatter(
        x=dates, y=values, mode="lines",
        line=dict(color=color, width=2),
        fill="tozeroy", fillcolor=f"rgba({_rgb(color)},0.10)",
        hovertemplate="%{y:.3f}<extra></extra>",
    ))
    fig.update_layout(**_layout(
        height=height, margin=dict(l=0, r=0, t=2, b=0),
        xaxis_visible=False, yaxis_visible=False, showlegend=False,
    ))
    return fig


def trend_chart(df: pd.DataFrame, cols: list[str], labels: list[str],
                colors: list[str], y_range: list | None = None,
                height: int = 240, y_title: str = "") -> go.Figure:
    fig = go.Figure()
    for col, label, color in zip(cols, labels, colors):
        if col not in df.columns:
            continue
        fig.add_trace(go.Scatter(
            x=df["date"], y=df[col], name=label, mode="lines+markers",
            line=dict(color=color, width=2.2), marker=dict(size=6),
            hovertemplate=f"{label}: %{{y:.3f}}<extra></extra>",
        ))
    layout = _layout(height=height, yaxis_title=y_title)
    if y_range:
        layout["yaxis"] = {**layout.get("yaxis", {}), "range": y_range}
    fig.update_layout(**layout)
    return fig


def risk_trend_chart(df: pd.DataFrame, height: int = 260) -> go.Figure:
    valid = df.dropna(subset=["risk_score"])
    fig   = go.Figure()
    fig.add_hrect(y0=0.65, y1=1.0, fillcolor=f"rgba({_rgb(C['red'])},0.07)",   line_width=0)
    fig.add_hrect(y0=0.45, y1=0.65, fillcolor=f"rgba({_rgb(C['amber'])},0.07)", line_width=0)
    fig.add_hrect(y0=0.0,  y1=0.45, fillcolor=f"rgba({_rgb(C['green'])},0.07)", line_width=0)

    if not valid.empty:
        colors = [_risk_color(t) for t in valid["risk_tier"]]
        fig.add_trace(go.Scatter(
            x=valid["date"], y=valid["risk_score"],
            mode="lines+markers",
            line=dict(color=C["text_2"], width=2),
            marker=dict(color=colors, size=11, line=dict(color="white", width=2)),
            hovertemplate="Risk: %{y:.3f}<extra></extra>",
        ))

    fig.add_annotation(x=0.01, xref="paper", y=0.82, yref="paper",
                       text="High", showarrow=False, font=dict(color=C["red"], size=10))
    fig.add_annotation(x=0.01, xref="paper", y=0.55, yref="paper",
                       text="Moderate", showarrow=False, font=dict(color=C["amber"], size=10))
    fig.add_annotation(x=0.01, xref="paper", y=0.22, yref="paper",
                       text="Low", showarrow=False, font=dict(color=C["green"], size=10))

    fig.update_layout(**_layout(
        height=height, showlegend=False,
        yaxis=dict(range=[0, 1], title="Risk score", **_CL["yaxis"]),
    ))
    return fig


def stress_arc_chart(stress_curve: dict, height: int = 200) -> go.Figure:
    """Simple Baseline → Peak → Post-relax arc for one check-in."""
    pts = [
        ("Baseline",   stress_curve.get("S1_baseline"),  C["blue"]),
        ("Peak",       stress_curve.get("S5_peak"),       C["red"]),
        ("Post-relax", stress_curve.get("S8_post_relax"), C["green"]),
    ]
    pts = [(lbl, v, col) for lbl, v, col in pts if v is not None]
    labels = [p[0] for p in pts]
    values = [p[1] for p in pts]
    colors = [p[2] for p in pts]

    fig = go.Figure(go.Scatter(
        x=labels, y=values, mode="lines+markers",
        line=dict(color=C["text_3"], width=2.5),
        marker=dict(color=colors, size=14, line=dict(color="white", width=2)),
        hovertemplate="%{x}: %{y:.2f}<extra></extra>",
    ))
    fig.update_layout(**_layout(
        height=height, showlegend=False,
        yaxis=dict(range=[0, 1], title="Stress (0–1)", **_CL["yaxis"]),
        xaxis=dict(showgrid=False, zeroline=False, showline=False),
    ))
    return fig


def bar_chart(labels: list[str], values: list[float],
              colors: list[str] | None = None, height: int = 220) -> go.Figure:
    bar_colors = colors or [C["blue"]] * len(labels)
    fig = go.Figure(go.Bar(
        x=labels, y=values, marker_color=bar_colors,
        marker_line_width=0,
        hovertemplate="%{x}: %{y:.3f}<extra></extra>",
    ))
    fig.update_layout(**_layout(height=height, bargap=0.4,
                                yaxis=dict(range=[0, 1], **_CL["yaxis"])))
    return fig


# ── Left panel ────────────────────────────────────────────────────────────────
def render_sidebar(checkins: list[dict], df: pd.DataFrame) -> None:
    latest = checkins[-1]
    tier   = latest.get("risk_tier")
    rs     = latest.get("risk_score")
    rc     = _risk_color(tier)
    n      = len(checkins)

    # Participant header
    last_date = pd.to_datetime(latest["timestamp"]).strftime("%d %b %Y")
    st.markdown(f"""
    <div class="label">Participant</div>
    <div class="pt-name">{mask_id(latest.get("participant_id","—"))}</div>
    <div class="pt-sub">{n} check-in{"s" if n != 1 else ""} &nbsp;·&nbsp; Latest {last_date}</div>
    """, unsafe_allow_html=True)

    st.markdown('<hr class="divider">', unsafe_allow_html=True)

    # Risk score
    st.markdown('<div class="label">Risk score — latest</div>', unsafe_allow_html=True)
    if rs is not None:
        bg = f"rgba({_rgb(rc)},0.10)"
        st.markdown(f"""
        <div class="risk-number" style="color:{rc};">{rs:.2f}</div>
        <span class="badge" style="background:{bg};color:{rc};margin-top:0.3rem;display:inline-block;">
          ● {tier}
        </span>
        """, unsafe_allow_html=True)
    else:
        remaining = max(0, 3 - n)
        st.markdown(f"""
        <span style="font-size:0.83rem;color:{C['text_3']};">
          Available after {remaining} more check-in{"s" if remaining != 1 else ""}
        </span>
        """, unsafe_allow_html=True)

    st.markdown('<hr class="divider">', unsafe_allow_html=True)

    # Reactivity sparkline
    vr = df.dropna(subset=["reactivity"])
    if not vr.empty:
        st.markdown('<div class="label">Reactivity trend</div>', unsafe_allow_html=True)
        st.plotly_chart(sparkline(vr["date"], vr["reactivity"], C["red"]),
                        use_container_width=True, config={"displayModeBar": False})

    # Recovery sparkline
    vc = df.dropna(subset=["recovery"])
    if not vc.empty:
        st.markdown('<div class="label">Recovery trend</div>', unsafe_allow_html=True)
        st.plotly_chart(sparkline(vc["date"], vc["recovery"], C["green"]),
                        use_container_width=True, config={"displayModeBar": False})

    st.markdown('<hr class="divider">', unsafe_allow_html=True)

    # Self-reports (latest)
    sr = latest.get("self_reports", {})
    if sr:
        st.markdown('<div class="label">Self-reported stress — latest</div>', unsafe_allow_html=True)
        for lbl, key in [("Before tasks", "step1_baseline"),
                          ("After stressors", "step5a_after_stroop"),
                          ("After relaxation", "step8_post_relax")]:
            val = sr.get(key)
            if val is None:
                val = sr.get(key.replace("step5a_after_stroop", "step5b_after_negative"), None)
            if val is not None:
                bw  = int(val / 5 * 100)
                bc  = C["red"] if val >= 4 else C["amber"] if val >= 3 else C["green"]
                st.markdown(f"""
                <div style="margin-bottom:0.5rem;">
                  <div style="font-size:0.76rem;color:{C['text_2']};margin-bottom:2px;">
                    {lbl} &nbsp;<strong style="color:{C['text']};">{val}/5</strong>
                  </div>
                  <div style="height:5px;background:{C['border']};border-radius:4px;">
                    <div style="width:{bw}%;height:100%;background:{bc};border-radius:4px;"></div>
                  </div>
                </div>
                """, unsafe_allow_html=True)


# ── VIEW 1: Overview (landing page) ──────────────────────────────────────────
def render_overview(checkins: list[dict], df: pd.DataFrame) -> None:
    st.markdown('<div class="s-head">Overview</div>', unsafe_allow_html=True)
    st.markdown('<div class="s-sub">Risk trend and session summary across all check-ins</div>', unsafe_allow_html=True)

    flags   = detect_flags(checkins)
    latest  = checkins[-1]
    n       = len(checkins)

    # ── Alert banners ──────────────────────────────────────────────────────────
    disc      = latest.get("discordance", {})
    disc_flag = disc.get("discordance_flag", False)
    disc_dir  = disc.get("discordance_direction")
    disc_val  = disc.get("discordance_peak")

    if flags:
        flag_messages = {
            "reactivity":     ("↑ Reactivity above personal average", "high"),
            "recovery":       ("↓ Recovery below personal average",   "high"),
            "semantic_score": ("↑ Language risk signals elevated",     "moderate"),
        }
        for key, flagged in flags.items():
            if flagged:
                msg, level = flag_messages[key]
                st.markdown(f'<div class="alert-box alert-{level}">{msg} in latest check-in</div>',
                            unsafe_allow_html=True)

    if disc_flag and disc_dir:
        if disc_dir == "over":
            disc_msg = f"⚠ Discordance detected — participant reports more stress than voice reflects (Δ={disc_val:+.2f}). Possible emotional suppression or masking."
        else:
            disc_msg = f"⚠ Discordance detected — voice signals more stress than reported (Δ={disc_val:+.2f}). Possible underreporting or limited self-awareness."
        st.markdown(f'<div class="alert-box alert-moderate">{disc_msg}</div>',
                    unsafe_allow_html=True)

    elif n >= 3:
        tier = latest.get("risk_tier") or "Low"
        cls  = {"High": "high", "Moderate": "moderate", "Low": "low"}.get(tier, "low")
        rs   = latest.get("risk_score")
        msg  = f"Risk score {rs:.2f} — {tier}" if rs else "No unusual patterns detected"
        st.markdown(f'<div class="alert-box alert-{cls}">{msg} · all metrics within personal range</div>',
                    unsafe_allow_html=True)

    elif n < 3:
        st.markdown(f'<div class="alert-box alert-low">Collecting baseline — risk scoring available after 3 check-ins ({max(0,3-n)} more needed)</div>',
                    unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Summary metrics ────────────────────────────────────────────────────────
    m1, m2, m3, m4 = st.columns(4)
    react = latest.get("reactivity")
    recov = latest.get("recovery")
    sem   = latest.get("semantic_score")
    rs    = latest.get("risk_score")

    if n >= 2:
        prev = checkins[-2]
        d_react = (react or 0) - (prev.get("reactivity") or 0)
        d_recov = (recov or 0) - (prev.get("recovery")   or 0)
        d_sem   = (sem   or 0) - (prev.get("semantic_score") or 0)
    else:
        d_react = d_recov = d_sem = None

    with m1:
        st.metric("Reactivity (latest)",
                  f"{react:.3f}" if react is not None else "—",
                  delta=f"{d_react:+.3f}" if d_react is not None else None,
                  delta_color="inverse",
                  help="S5_peak − S1_baseline. Higher = bigger stress spike during tasks.")
    with m2:
        st.metric("Recovery (latest)",
                  f"{recov:.3f}" if recov is not None else "—",
                  delta=f"{d_recov:+.3f}" if d_recov is not None else None,
                  help="S5_peak − S8_post_relax. Higher = better stress reduction after relaxation.")
    with m3:
        st.metric("Semantic score (latest)",
                  f"{sem:.3f}" if sem is not None else "—",
                  delta=f"{d_sem:+.3f}" if d_sem is not None else None,
                  delta_color="inverse",
                  help="Language risk indicator from transcripts (0–1). Higher = more risk signals.")
    with m4:
        st.metric("Risk score (latest)",
                  f"{rs:.3f}" if rs is not None else "—",
                  help="Within-person composite (reactivity/recovery deviation + semantic). ≥0.65 High.")

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Main charts ────────────────────────────────────────────────────────────
    col_risk, col_react = st.columns([1.6, 1])

    with col_risk:
        st.markdown('<div class="label">Risk score over time</div>', unsafe_allow_html=True)
        valid = df.dropna(subset=["risk_score"])
        if valid.empty:
            st.info("Risk scoring starts after the 3rd check-in.")
        else:
            st.plotly_chart(risk_trend_chart(df), use_container_width=True,
                            config={"displayModeBar": False})

    with col_react:
        st.markdown('<div class="label">Reactivity vs. recovery</div>', unsafe_allow_html=True)
        fig_rr = go.Figure()
        vr = df.dropna(subset=["reactivity", "recovery"])
        fig_rr.add_trace(go.Bar(x=vr["date"], y=vr["reactivity"],
                                name="Reactivity", marker_color=C["red"],  opacity=0.85,
                                hovertemplate="Reactivity: %{y:.3f}<extra></extra>"))
        fig_rr.add_trace(go.Bar(x=vr["date"], y=vr["recovery"],
                                name="Recovery",   marker_color=C["green"], opacity=0.85,
                                hovertemplate="Recovery: %{y:.3f}<extra></extra>"))
        fig_rr.update_layout(**_layout(
            height=260, barmode="group", bargap=0.25, bargroupgap=0.06,
            yaxis=dict(range=[0, 1], title="Score (0–1)", **_CL["yaxis"]),
        ))
        st.plotly_chart(fig_rr, use_container_width=True, config={"displayModeBar": False})

    # ── Semantic score + personal averages ─────────────────────────────────────
    col_sem, col_avg = st.columns([1.6, 1])

    with col_sem:
        st.markdown('<div class="label">Semantic risk score over time</div>', unsafe_allow_html=True)
        st.plotly_chart(
            trend_chart(df, ["semantic_score"], ["Semantic risk"], [C["blue"]],
                        y_range=[0, 1], height=200),
            use_container_width=True, config={"displayModeBar": False}
        )

    with col_avg:
        st.markdown('<div class="label">Personal averages (all sessions)</div>', unsafe_allow_html=True)
        vall = df.dropna(subset=["reactivity", "recovery"])
        if not vall.empty:
            μr  = vall["reactivity"].mean()
            μc  = vall["recovery"].mean()
            μs  = df["semantic_score"].dropna().mean()
            st.markdown(f"""
            <div class="card-alt" style="font-size:0.84rem;line-height:2.1;">
              <b>μ Reactivity</b> &nbsp; {μr:.3f}<br>
              <b>μ Recovery</b> &nbsp;&nbsp;&nbsp; {μc:.3f}<br>
              <b>μ Semantic</b> &nbsp;&nbsp;&nbsp; {f"{μs:.3f}" if not np.isnan(μs) else "—"}<br>
              <b>Sessions</b> &nbsp;&nbsp;&nbsp;&nbsp;&nbsp; {len(checkins)}
            </div>
            """, unsafe_allow_html=True)


# ── VIEW 2: Check-in Details ──────────────────────────────────────────────────
def render_checkin_details(checkins: list[dict], df: pd.DataFrame) -> None:
    st.markdown('<div class="s-head">Check-in Details</div>', unsafe_allow_html=True)
    st.markdown('<div class="s-sub">Select a session to explore stress, language, and voice data</div>', unsafe_allow_html=True)

    # Session selector via timeline
    options = [
        f"{pd.to_datetime(c['timestamp']).strftime('%d %b %Y')}  ·  {c['checkin_id']}"
        for c in checkins
    ]
    sel_idx = st.selectbox(
        "Session", range(len(options)),
        format_func=lambda i: options[i],
        index=len(options) - 1,
        label_visibility="collapsed",
    )
    c  = checkins[sel_idx]
    sc = c.get("stress_curve", {})
    sr = c.get("self_reports", {})

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Stress arc + self-reports ──────────────────────────────────────────────
    ca, cb, cc, cd = st.columns(4)
    with ca:
        s1 = sr.get("step1_baseline")
        st.metric("Before tasks", f"{s1}/5" if s1 else "—")
    with cb:
        s5a = sr.get("step5a_after_stroop")
        s5b = sr.get("step5b_after_negative")
        s5  = round((s5a + s5b) / 2) if s5a and s5b else (s5a or s5b)
        st.metric("After stressors (avg)", f"{s5}/5" if s5 else "—")
    with cc:
        s8 = sr.get("step8_post_relax")
        st.metric("After relaxation", f"{s8}/5" if s8 else "—")
    with cd:
        react = c.get("reactivity")
        recov = c.get("recovery")
        st.metric("Reactivity / Recovery",
                  f"{react:.2f} / {recov:.2f}" if react is not None and recov is not None else "—")

    st.markdown("<br>", unsafe_allow_html=True)

    col_arc, col_sem = st.columns([1, 1.4])

    with col_arc:
        st.markdown('<div class="label">Stress arc — self-report (0–1)</div>', unsafe_allow_html=True)
        st.plotly_chart(stress_arc_chart(sc), use_container_width=True,
                        config={"displayModeBar": False})
        st.markdown(f"""
        <div style="font-size:0.74rem;color:{C['text_3']};margin-top:-0.4rem;">
          🔵 Baseline &nbsp;·&nbsp; 🔴 Peak &nbsp;·&nbsp; 🟢 Post-relax
        </div>
        """, unsafe_allow_html=True)

    with col_sem:
        st.markdown('<div class="label">Language risk signals</div>', unsafe_allow_html=True)
        top = c.get("top_risk_features", [])
        if top:
            feat_labels = [FEAT_LABEL.get(f, f) for f, _ in top]
            feat_vals   = [v for _, v in top]
            feat_colors = [C["red"] if v > 0.6 else C["amber"] if v > 0.35 else C["green"]
                           for v in feat_vals]
            st.plotly_chart(bar_chart(feat_labels, feat_vals, feat_colors, height=220),
                            use_container_width=True, config={"displayModeBar": False})
        else:
            st.info("No semantic features — STT not run for this session.")

    st.markdown('<hr class="divider">', unsafe_allow_html=True)

    # ── Discordance panel ──────────────────────────────────────────────────────
    disc = c.get("discordance", {})
    d_peak     = disc.get("discordance_peak")
    d_baseline = disc.get("discordance_baseline")
    d_postrelax= disc.get("discordance_postrelax")
    d_flag     = disc.get("discordance_flag", False)
    d_dir      = disc.get("discordance_direction")

    if d_peak is not None:
        st.markdown('<div class="label">Self-report vs acoustic model — discordance</div>',
                    unsafe_allow_html=True)
        st.markdown(f"""
        <div style="font-size:0.75rem;color:{C['text_3']};margin-bottom:0.7rem;">
          Discordance = self-reported stress − acoustically predicted stress at each checkpoint.
          Positive → reports more than voice reflects &nbsp;·&nbsp;
          Negative → voice suggests more than reported.
          Flagged when |peak discordance| &gt; 0.20.
        </div>
        """, unsafe_allow_html=True)

        ac_scores = c.get("acoustic_scores", {})
        ac_peak_vals = [v for k, v in ac_scores.items()
                        if k in ("step3_stroop", "step4_negative_event") and v is not None]
        ac_peak_mean = float(np.mean(ac_peak_vals)) if ac_peak_vals else None

        checkpoints = ["Baseline", "Peak (stressors)", "Post-relax"]
        sr_vals     = [sc.get("S1_baseline"), sc.get("S5_peak"), sc.get("S8_post_relax")]
        ac_vals     = [ac_scores.get("step2_daily_narration"), ac_peak_mean,
                       ac_scores.get("step7_positive_reading")]

        fig_disc = go.Figure()
        fig_disc.add_trace(go.Bar(
            name="Self-report", x=checkpoints, y=sr_vals,
            marker_color=C["blue"], opacity=0.85,
            hovertemplate="Self-report: %{y:.2f}<extra></extra>",
        ))
        fig_disc.add_trace(go.Bar(
            name="Acoustic model", x=checkpoints, y=ac_vals,
            marker_color=C["amber"], opacity=0.85,
            hovertemplate="Acoustic: %{y:.2f}<extra></extra>",
        ))
        fig_disc.update_layout(**_layout(
            height=230, barmode="group", bargap=0.25, bargroupgap=0.08,
            yaxis=dict(range=[0, 1], title="Stress (0–1)", **_CL["yaxis"]),
        ))
        st.plotly_chart(fig_disc, use_container_width=True, config={"displayModeBar": False})

        # Discordance summary
        flag_color = C["amber"] if d_flag else C["text_3"]
        flag_text  = ""
        if d_flag and d_dir == "over":
            flag_text = " — reports more stress than voice reflects (possible suppression / masking)"
        elif d_flag and d_dir == "under":
            flag_text = " — voice signals more stress than reported (possible underreporting)"

        da, db, dc = st.columns(3)
        for col, lbl, val in zip([da, db, dc],
                                  ["Baseline", "Peak", "Post-relax"],
                                  [d_baseline, d_peak, d_postrelax]):
            with col:
                is_peak = lbl == "Peak"
                color   = flag_color if (is_peak and d_flag) else C["text_2"]
                st.markdown(f"""
                <div style="text-align:center;padding:0.5rem;background:{C['panel_alt']};
                            border-radius:10px;border:1px solid {C['border']};">
                  <div style="font-size:0.68rem;font-weight:700;letter-spacing:0.06em;
                              text-transform:uppercase;color:{C['text_3']};">{lbl}</div>
                  <div style="font-size:1.4rem;font-weight:700;color:{color};">
                    {f"{val:+.2f}" if val is not None else "—"}
                  </div>
                </div>
                """, unsafe_allow_html=True)

        if d_flag:
            st.markdown(f"""
            <div class="alert-box alert-moderate" style="margin-top:0.6rem;">
              ⚠ Peak discordance flagged ({d_peak:+.2f}){flag_text}
            </div>
            """, unsafe_allow_html=True)

    st.markdown('<hr class="divider">', unsafe_allow_html=True)

    # ── Transcript summaries ───────────────────────────────────────────────────
    def _summarize(text: str, max_sentences: int = 3) -> str:
        """Return first max_sentences sentences of text, trimmed to ~120 words."""
        if not text:
            return ""
        # Split on sentence-ending punctuation
        import re
        sentences = re.split(r'(?<=[.!?])\s+', text.strip())
        summary = " ".join(sentences[:max_sentences])
        # Hard cap at 120 words
        words = summary.split()
        if len(words) > 120:
            summary = " ".join(words[:120]) + "…"
        return summary

    t2 = c.get("transcript_step2", "") or ""
    t4 = c.get("transcript_step4", "") or ""

    st.markdown('<div class="label">Transcript summaries</div>', unsafe_allow_html=True)
    tc1, tc2 = st.columns(2)
    with tc1:
        summary2 = _summarize(t2)
        if summary2:
            st.markdown(f'<div class="snippet"><b>Step 2 · Daily narration</b><br><span style="color:#888;font-size:0.76rem;">First 3 sentences</span><br><br>"{summary2}"</div>',
                        unsafe_allow_html=True)
        else:
            st.markdown('<div class="snippet" style="color:#aaa;">No transcript available.</div>',
                        unsafe_allow_html=True)
    with tc2:
        summary4 = _summarize(t4)
        if summary4:
            st.markdown(f'<div class="snippet"><b>Step 4 · Negative event</b><br><span style="color:#888;font-size:0.76rem;">First 3 sentences</span><br><br>"{summary4}"</div>',
                        unsafe_allow_html=True)
        else:
            st.markdown('<div class="snippet" style="color:#aaa;">No transcript available.</div>',
                        unsafe_allow_html=True)

    st.markdown('<hr class="divider">', unsafe_allow_html=True)

    # ── Voice features (exploratory) ───────────────────────────────────────────
    st.markdown('<div class="label">Voice features — exploratory</div>', unsafe_allow_html=True)
    st.markdown(f"""
    <div style="font-size:0.76rem;color:{C['text_3']};margin-bottom:0.7rem;">
      Acoustic indicators extracted from audio. Shown as context only — not used in risk score.
    </div>
    """, unsafe_allow_html=True)

    acoustic_cols = ["speech_rate_wpm", "f0_mean", "f0_std", "hnr_mean", "pause_dur_mean"]
    label_map = {
        "speech_rate_wpm": ("Speech rate (wpm)",       C["blue"]),
        "f0_mean":         ("Pitch mean (Hz)",          C["green"]),
        "f0_std":          ("Pitch variability (Hz)",   C["amber"]),
        "hnr_mean":        ("Voice clarity / HNR (dB)", C["green"]),
        "pause_dur_mean":  ("Mean pause duration (s)",  C["red"]),
    }
    available = [col for col in acoustic_cols if col in df.columns and df[col].notna().any()]

    if available:
        for i in range(0, len(available), 2):
            cl, cr = st.columns(2)
            for feat, col in zip(available[i:i+2], [cl, cr]):
                label, color = label_map.get(feat, (feat, C["blue"]))
                with col:
                    st.markdown(f'<div class="label">{label}</div>', unsafe_allow_html=True)
                    st.plotly_chart(
                        trend_chart(df, [feat], [label], [color], height=180),
                        use_container_width=True, config={"displayModeBar": False}
                    )

        # Current session highlight
        asum = c.get("acoustic_summary", {})
        if asum:
            rows = []
            for feat in available:
                cur_val  = asum.get(feat)
                mean_val = df[feat].dropna().mean() if feat in df.columns else None
                lbl      = label_map.get(feat, (feat,))[0]
                rows.append({
                    "Feature": lbl,
                    "This session": f"{cur_val:.2f}" if cur_val is not None else "—",
                    "Personal avg":  f"{mean_val:.2f}" if mean_val is not None else "—",
                })
            st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
    else:
        st.info("No acoustic features extracted for this participant yet.")


# ── VIEW 3: Session Notes ─────────────────────────────────────────────────────
def render_session_notes(checkins: list[dict], **_) -> None:
    st.markdown('<div class="s-head">Session Notes</div>', unsafe_allow_html=True)
    st.markdown('<div class="s-sub">Notes are stored in your browser session only.</div>', unsafe_allow_html=True)

    if "session_notes" not in st.session_state:
        st.session_state.session_notes = ""

    latest = checkins[-1]
    rs     = latest.get("risk_score")
    tier   = latest.get("risk_tier")
    rc     = _risk_color(tier)

    col_note, col_ctx = st.columns([2.2, 1])

    with col_note:
        notes = st.text_area(
            "Notes", value=st.session_state.session_notes, height=340,
            placeholder="Type session observations, themes, follow-up actions…",
            label_visibility="collapsed",
        )
        if notes != st.session_state.session_notes:
            st.session_state.session_notes = notes

        s1, s2, _ = st.columns([1, 1, 2.5])
        with s1:
            if st.button("💾  Save note", use_container_width=True):
                st.success("Saved.")
        with s2:
            if st.button("🗑  Clear", use_container_width=True):
                st.session_state.session_notes = ""
                st.rerun()

    with col_ctx:
        react = latest.get("reactivity", 0) or 0
        recov = latest.get("recovery",   0) or 0
        st.markdown(f"""
        <div class="card-alt">
          <div class="label">Latest session context</div>
          <div style="font-size:0.82rem;line-height:2;color:{C['text']};">
            <b>Participant:</b> {mask_id(latest.get("participant_id","—"))}<br>
            <b>Date:</b> {pd.to_datetime(latest["timestamp"]).strftime("%d %b %Y")}<br>
            <b>Sessions:</b> {len(checkins)}<br>
            <b>Risk score:</b> <span style="color:{rc};font-weight:600;">
              {f"{rs:.3f} [{tier}]" if rs is not None else "Insufficient data"}
            </span><br>
            <b>Reactivity:</b> {react:.3f}<br>
            <b>Recovery:</b> {recov:.3f}
          </div>
        </div>
        """, unsafe_allow_html=True)

        top = latest.get("top_risk_features", [])
        if top:
            st.markdown('<div class="label" style="margin-top:0.5rem;">Top language signals</div>',
                        unsafe_allow_html=True)
            rows_html = ""
            for feat, val in top[:4]:
                lbl   = FEAT_LABEL.get(feat, feat)
                level = "High" if val > 0.6 else "Mod" if val > 0.35 else "Low"
                color = C["red"] if level == "High" else C["amber"] if level == "Mod" else C["text_3"]
                rows_html += f"""
                <div class="lang-row">
                  <div class="dot" style="background:{color};"></div>
                  <div>{lbl} <span style="color:{C['text_3']};font-size:0.75rem;">({level} · {val:.2f})</span></div>
                </div>"""
            st.markdown(rows_html, unsafe_allow_html=True)


# ── Navigation & layout ───────────────────────────────────────────────────────
VIEWS = [
    ("📊  Overview",         "overview"),
    ("📋  Check-in Details", "details"),
    ("📝  Session Notes",    "notes"),
]

RENDER = {
    "overview": render_overview,
    "details":  render_checkin_details,
    "notes":    render_session_notes,
}


def main() -> None:
    inject_css()

    if "active_view" not in st.session_state:
        st.session_state.active_view = "overview"
    if "participant_id" not in st.session_state:
        participants = available_participants()
        st.session_state.participant_id = participants[0] if participants else "demo"

    # Header
    st.markdown(f"""
    <div style="display:flex;align-items:center;gap:0.55rem;margin-bottom:1.1rem;">
      <span style="font-size:1.4rem;">🌿</span>
      <span style="font-size:1.2rem;font-weight:700;color:{C['text']};">MindTrack</span>
      <span style="font-size:0.74rem;color:{C['text_3']};margin-left:0.15rem;
                   background:{C['border']};padding:0.15rem 0.6rem;border-radius:8px;">
        Therapist Dashboard
      </span>
    </div>
    """, unsafe_allow_html=True)

    # Participant selector
    participants = available_participants()
    if participants:
        options      = participants + ["— demo (synthetic) —"]
        current_idx  = (options.index(st.session_state.participant_id)
                        if st.session_state.participant_id in options else 0)
        sel = st.selectbox(
            "Participant", options,
            index=current_idx,
            label_visibility="collapsed",
            key="participant_select",
        )
        if sel != st.session_state.participant_id:
            st.session_state.participant_id = sel
            st.session_state.pop("checkins", None)
            st.session_state.pop("df", None)
            st.rerun()

    # Load data for current participant
    if "checkins" not in st.session_state:
        pid = st.session_state.participant_id
        checkins = load_checkins(pid) if pid != "— demo (synthetic) —" else _mock_checkins()
        st.session_state.checkins = checkins
        st.session_state.df       = checkins_to_df(checkins)

    checkins = st.session_state.checkins
    df       = st.session_state.df

    left_col, nav_col, content_col = st.columns([1.3, 0.65, 3.7])

    with left_col:
        render_sidebar(checkins, df)

    with nav_col:
        st.markdown('<div class="label">Views</div>', unsafe_allow_html=True)
        for label, key in VIEWS:
            if st.session_state.active_view == key:
                st.markdown(f"""
                <div style="background:{C['green']};color:white;border-radius:10px;
                            padding:0.55rem 0.85rem;font-size:0.81rem;font-weight:600;
                            margin-bottom:0.3rem;">{label}</div>
                """, unsafe_allow_html=True)
            else:
                if st.button(label, key=f"nav_{key}", use_container_width=True):
                    st.session_state.active_view = key
                    st.rerun()

    with content_col:
        RENDER[st.session_state.active_view](checkins=checkins, df=df)


if __name__ == "__main__":
    main()
