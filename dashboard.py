"""
dashboard.py  ·  MindTrack Therapist Dashboard
───────────────────────────────────────────────
Streamlit app displaying longitudinal patient mental health data.
All data is synthetic / mock.

Run:
    streamlit run dashboard.py
"""

from __future__ import annotations

from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

# ── Page config (must be first Streamlit call) ────────────────────────────────
st.set_page_config(
    page_title="MindTrack · Therapist Dashboard",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ─────────────────────────────────────────────────────────────────────────────
# Design tokens
# ─────────────────────────────────────────────────────────────────────────────
C = {
    "bg":            "#F7F3EE",
    "panel":         "#FFFFFF",
    "panel_alt":     "#F9F6F1",
    "border":        "#E5DFD6",
    "green":         "#5F9B6B",
    "blue":          "#4E86B5",
    "amber":         "#CC8B52",
    "red":           "#B85A50",
    "text":          "#363330",
    "text_2":        "#766E68",
    "text_3":        "#A89F98",
}

# Shared Plotly layout applied to every chart
_CL = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(family="Inter, system-ui, sans-serif", color=C["text"], size=12),
    margin=dict(l=4, r=4, t=28, b=4),
    xaxis=dict(showgrid=False, zeroline=False, showline=False),
    yaxis=dict(showgrid=True, gridcolor=C["border"], zeroline=False, showline=False),
    legend=dict(orientation="h", y=-0.18, x=0, font_size=11),
    hovermode="x unified",
)

# ─────────────────────────────────────────────────────────────────────────────
# CSS
# ─────────────────────────────────────────────────────────────────────────────

def inject_css() -> None:
    st.markdown(f"""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

    /* ── Page ── */
    .stApp {{ background: {C["bg"]}; font-family: 'Inter', system-ui, sans-serif; }}
    .main .block-container {{ padding: 1.4rem 2rem 2rem 2rem; max-width: 1700px; }}
    header[data-testid="stHeader"] {{ display: none; }}
    #MainMenu, footer {{ visibility: hidden; }}

    /* ── Cards ── */
    .card {{
        background: {C["panel"]}; border: 1px solid {C["border"]};
        border-radius: 14px; padding: 1.1rem 1.3rem; margin-bottom: 0.85rem;
    }}
    .card-alt {{
        background: {C["panel_alt"]}; border: 1px solid {C["border"]};
        border-radius: 14px; padding: 1.1rem 1.3rem; margin-bottom: 0.85rem;
    }}
    .label {{
        font-size: 0.68rem; font-weight: 700; letter-spacing: 0.07em;
        text-transform: uppercase; color: {C["text_3"]}; margin-bottom: 0.35rem;
    }}
    .divider {{ border: none; border-top: 1px solid {C["border"]}; margin: 0.9rem 0; }}

    /* ── Risk badge ── */
    .badge {{
        display: inline-block; padding: 0.28rem 0.85rem;
        border-radius: 20px; font-size: 0.82rem; font-weight: 600;
        letter-spacing: 0.02em;
    }}

    /* ── Patient header ── */
    .pt-name {{ font-size: 1.35rem; font-weight: 700; color: {C["text"]}; line-height: 1.2; }}
    .pt-sub  {{ font-size: 0.8rem;  color: {C["text_2"]}; margin-top: 0.1rem; }}

    /* ── Language bullets ── */
    .lang-row {{
        display: flex; align-items: flex-start; gap: 0.5rem;
        padding: 0.32rem 0; border-bottom: 1px solid {C["border"]};
        font-size: 0.82rem; color: {C["text"]};
    }}
    .lang-row:last-child {{ border-bottom: none; }}
    .dot {{
        width: 7px; height: 7px; border-radius: 50%;
        background: {C["green"]}; flex-shrink: 0; margin-top: 5px;
    }}

    /* ── AI summary ── */
    .ai-box {{
        background: {C["panel_alt"]}; border-left: 3px solid {C["green"]};
        border-radius: 0 10px 10px 0; padding: 0.75rem 1rem;
        font-size: 0.83rem; line-height: 1.65; color: {C["text"]};
    }}

    /* ── Nav buttons ── */
    .stButton > button {{
        border-radius: 10px; font-family: 'Inter', sans-serif;
        font-size: 0.81rem; font-weight: 500;
        text-align: left; width: 100%;
        padding: 0.55rem 0.85rem; margin-bottom: 0.3rem;
        transition: background 0.15s, border-color 0.15s;
        border: 1px solid {C["border"]};
        background: {C["panel"]}; color: {C["text"]};
    }}
    .stButton > button:hover {{
        background: {C["panel_alt"]} !important;
        border-color: {C["green"]} !important;
    }}

    /* ── Metrics ── */
    [data-testid="stMetric"] {{
        background: {C["panel"]}; border: 1px solid {C["border"]};
        border-radius: 12px; padding: 0.9rem 1.1rem;
    }}
    [data-testid="stMetricLabel"] {{ color: {C["text_2"]} !important; font-size: 0.75rem !important; }}
    [data-testid="stMetricValue"] {{ color: {C["text"]} !important; font-size: 1.35rem !important; font-weight: 600 !important; }}

    /* ── Section header in content pane ── */
    .s-head {{ font-size: 1.1rem; font-weight: 700; color: {C["text"]}; margin-bottom: 0.15rem; }}
    .s-sub  {{ font-size: 0.8rem; color: {C["text_2"]}; margin-bottom: 1.1rem; }}

    /* ── Snippet cards ── */
    .snippet {{
        background: {C["panel_alt"]}; border: 1px solid {C["border"]};
        border-radius: 10px; padding: 0.75rem 1rem;
        font-size: 0.82rem; font-style: italic;
        color: {C["text_2"]}; line-height: 1.6; margin-bottom: 0.5rem;
    }}
    .tag {{
        display: inline-block; background: #E8F2EA; color: {C["green"]};
        border-radius: 4px; padding: 0.12rem 0.45rem;
        font-size: 0.7rem; font-weight: 700; letter-spacing: 0.04em;
        font-style: normal; margin-top: 0.35rem;
    }}
    </style>
    """, unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# Static mock content
# ─────────────────────────────────────────────────────────────────────────────

SNIPPETS = [
    ("I always mess things up — it's just who I am.",       "Self-blame · Overgeneralization"),
    ("Nothing ever works out for me, and it never will.",   "Hopelessness · Absolute language"),
    ("I keep replaying that conversation over and over.",   "Rumination"),
    ("I pushed my friends away again this week.",           "Social withdrawal · Self-blame"),
    ("I don't see the point in reaching out anymore.",      "Social withdrawal · Hopelessness"),
]

AI_SUMMARY = (
    "This week's check-ins show a gradual reduction in hopelessness markers and speech-rate "
    "variability compared to last week, suggesting early stabilisation. "
    "Rumination scores remain elevated, particularly in mid-week recordings. "
    "Social withdrawal language increased on Thursday and Friday — worth exploring in session. "
    "Overall trajectory is cautiously positive; WHO-5 is trending upward for the third consecutive week."
)


# ─────────────────────────────────────────────────────────────────────────────
# Data
# ─────────────────────────────────────────────────────────────────────────────

def load_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Generate synthetic longitudinal patient data. Returns (daily_df, weekly_df)."""
    np.random.seed(42)
    n_days = 84  # 12 weeks
    today  = datetime.today().replace(hour=0, minute=0, second=0, microsecond=0)
    dates  = [today - timedelta(days=n_days - i) for i in range(n_days)]

    # Stress trend: 7.2 → 3.8 with noise (improving over time)
    trend = np.linspace(7.2, 3.8, n_days) + np.random.randn(n_days) * 0.7

    daily = pd.DataFrame({
        "date": dates,
        # Core
        "stress":             np.clip(trend, 0, 10),
        # Voice / paralinguistic
        "pitch_instability":  np.clip(trend * 0.9  + np.random.randn(n_days) * 0.5,  0, 10),
        "pitch_variance":     np.clip(10 - trend   + np.random.randn(n_days) * 0.8,  0, 10),
        "speech_rate":        np.clip(130 - trend * 3 + np.random.randn(n_days) * 8, 60, 200),
        "pause_duration":     np.clip(trend * 0.28 + np.random.randn(n_days) * 0.12, 0, 4),
        "filler_rate":        np.clip(trend * 0.7  + np.random.randn(n_days) * 0.35, 0, 8),
        "vocal_tension":      np.clip(trend * 0.85 + np.random.randn(n_days) * 0.6,  0, 10),
        "flat_affect":        np.clip(trend * 0.75 + np.random.randn(n_days) * 0.5,  0, 10),
        "breathiness":        np.clip(trend * 0.6  + np.random.randn(n_days) * 0.45, 0, 10),
        "energy_variability": np.clip(10 - trend * 0.8 + np.random.randn(n_days) * 0.7, 0, 10),
        # Language / semantic
        "negativity_bias":    np.clip(trend * 0.85 + np.random.randn(n_days) * 0.6,  0, 10),
        "hopelessness":       np.clip(trend * 0.75 + np.random.randn(n_days) * 0.55, 0, 10),
        "rumination":         np.clip(trend * 0.9  + np.random.randn(n_days) * 0.65, 0, 10),
        "self_blame":         np.clip(trend * 0.8  + np.random.randn(n_days) * 0.5,  0, 10),
        "overgeneralization": np.clip(trend * 0.65 + np.random.randn(n_days) * 0.4,  0, 10),
        "social_withdrawal":  np.clip(trend * 0.7  + np.random.randn(n_days) * 0.55, 0, 10),
        "absolute_words":     np.clip((trend * 0.8 + np.random.randn(n_days) * 0.6).astype(int), 0, 12),
    })

    n_weeks    = 12
    week_dates = [today - timedelta(weeks=n_weeks - i) for i in range(n_weeks)]
    who5_trend = np.linspace(10, 18, n_weeks) + np.random.randn(n_weeks) * 1.2

    weekly = pd.DataFrame({
        "date":                week_dates,
        "who5":                np.clip(who5_trend, 0, 25),
        "stress_interference": np.clip(8.5 - np.linspace(0, 4, n_weeks) + np.random.randn(n_weeks) * 0.8, 0, 10),
        "sleep_quality":       np.clip(4.2 + np.linspace(0, 3.5, n_weeks) + np.random.randn(n_weeks) * 0.7, 0, 10),
        "social_connectedness":np.clip(3.5 + np.linspace(0, 4, n_weeks) + np.random.randn(n_weeks) * 0.9, 0, 10),
        "desync_indicator":    np.clip(6.5 - np.linspace(0, 3, n_weeks) + np.random.randn(n_weeks) * 0.6, 0, 10),
    })

    return daily, weekly


def compute_summary_metrics(daily: pd.DataFrame, weekly: pd.DataFrame) -> dict:
    """Derive high-level summary values for the left panel and top metrics."""
    last_week = daily.tail(7)
    prev_week = daily.iloc[-14:-7]

    stress_now   = last_week["stress"].mean()
    stress_delta = stress_now - prev_week["stress"].mean()
    who5_now     = weekly.iloc[-1]["who5"]
    who5_delta   = who5_now - weekly.iloc[-2]["who5"]

    if stress_now > 6.5 or who5_now < 9:
        risk, risk_color = "Elevated", C["red"]
    elif stress_now > 4.5 or who5_now < 13:
        risk, risk_color = "Moderate", C["amber"]
    else:
        risk, risk_color = "Low", C["green"]

    lang_cols = ["negativity_bias","hopelessness","rumination","self_blame","overgeneralization","social_withdrawal"]
    top_markers = last_week[lang_cols].mean().sort_values(ascending=False).head(4)

    return {
        "stress_now":   round(stress_now, 1),
        "stress_delta": round(stress_delta, 1),
        "who5_now":     round(who5_now, 1),
        "who5_delta":   round(who5_delta, 1),
        "risk":         risk,
        "risk_color":   risk_color,
        "top_markers":  top_markers,
        "last_checkin": daily["date"].max().strftime("%A, %d %B %Y"),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Chart helpers
# ─────────────────────────────────────────────────────────────────────────────

def _roll(s: pd.Series, w: int = 7) -> pd.Series:
    return s.rolling(w, min_periods=1).mean()


def _rgb(hex_color: str) -> str:
    h = hex_color.lstrip("#")
    return f"{int(h[0:2],16)},{int(h[2:4],16)},{int(h[4:6],16)}"


def _layout(**overrides) -> dict:
    """Merge shared chart layout with per-chart overrides (overrides win)."""
    return {**_CL, **overrides}


def sparkline(dates, values, color: str, height: int = 88) -> go.Figure:
    fig = go.Figure(go.Scatter(
        x=dates, y=values, mode="lines",
        line=dict(color=color, width=2),
        fill="tozeroy", fillcolor=f"rgba({_rgb(color)},0.08)",
        hovertemplate="%{y:.1f}<extra></extra>",
    ))
    fig.update_layout(**_layout(
        height=height, margin=dict(l=0, r=0, t=2, b=0),
        xaxis_visible=False, yaxis_visible=False, showlegend=False,
    ))
    return fig


def line_chart(df: pd.DataFrame, cols: list[str], labels: list[str],
               colors: list[str], y_label: str = "",
               height: int = 270, smooth: bool = True) -> go.Figure:
    fig = go.Figure()
    for col, label, color in zip(cols, labels, colors):
        y = _roll(df[col]) if smooth else df[col]
        fig.add_trace(go.Scatter(
            x=df["date"], y=y, name=label, mode="lines",
            line=dict(color=color, width=2.2),
            hovertemplate=f"{label}: %{{y:.1f}}<extra></extra>",
        ))
    fig.update_layout(**_layout(height=height, yaxis_title=y_label))
    return fig


def bar_chart(labels: list[str], values: list[float],
              colors: list[str] | None = None, height: int = 255) -> go.Figure:
    bar_colors = colors or [C["green"]] * len(labels)
    fig = go.Figure(go.Bar(
        x=labels, y=values, marker_color=bar_colors, marker_line_width=0, bargap=0.38,
        hovertemplate="%{x}: %{y:.1f}<extra></extra>",
    ))
    fig.update_layout(**_layout(height=height))
    return fig


def radar_chart(labels: list[str], values: list[float],
                color: str, height: int = 310) -> go.Figure:
    fig = go.Figure(go.Scatterpolar(
        r=values + [values[0]], theta=labels + [labels[0]],
        fill="toself",
        fillcolor=f"rgba({_rgb(color)},0.12)",
        line=dict(color=color, width=2),
        hovertemplate="%{theta}: %{r:.1f}<extra></extra>",
    ))
    fig.update_layout(**_layout(
        height=height,
        polar=dict(
            bgcolor="rgba(0,0,0,0)",
            radialaxis=dict(visible=True, range=[0,10], showticklabels=False, gridcolor=C["border"]),
            angularaxis=dict(gridcolor=C["border"]),
        ),
    ))
    return fig


def histogram(values: pd.Series, color: str,
              x_label: str = "", height: int = 235) -> go.Figure:
    fig = go.Figure(go.Histogram(
        x=values, marker_color=color, marker_line_width=0, nbinsx=20,
        hovertemplate="Pause %{x:.2f}s · %{y} occurrences<extra></extra>",
    ))
    fig.update_layout(**_layout(height=height, xaxis_title=x_label, yaxis_title="Count"))
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# render_quick_insights  —  LEFT PANEL (always visible)
# ─────────────────────────────────────────────────────────────────────────────

def render_quick_insights(daily: pd.DataFrame, weekly: pd.DataFrame, metrics: dict) -> None:
    # Patient header
    st.markdown(f"""
    <div class="label">Current patient</div>
    <div class="pt-name">A. Thompson</div>
    <div class="pt-sub">Age 34 · 8 months in therapy</div>
    """, unsafe_allow_html=True)

    st.markdown("<div style='height:0.7rem'></div>", unsafe_allow_html=True)

    # Risk badge
    rc  = metrics["risk_color"]
    bg  = f"rgba({_rgb(rc)},0.10)"
    st.markdown(f"""
    <div class="label">Risk level</div>
    <span class="badge" style="background:{bg};color:{rc};">● &nbsp;{metrics['risk']}</span>
    """, unsafe_allow_html=True)

    st.markdown('<hr class="divider">', unsafe_allow_html=True)

    # Stress sparkline (last 4 weeks)
    st.markdown('<div class="label">Stress indicator · last 4 weeks</div>', unsafe_allow_html=True)
    last4 = daily.tail(28)
    st.plotly_chart(sparkline(last4["date"], _roll(last4["stress"], 3), C["amber"]),
                    use_container_width=True, config={"displayModeBar": False})

    # WHO-5 sparkline
    st.markdown('<div class="label">WHO-5 score · weekly</div>', unsafe_allow_html=True)
    st.plotly_chart(sparkline(weekly["date"], weekly["who5"], C["green"]),
                    use_container_width=True, config={"displayModeBar": False})

    st.markdown('<hr class="divider">', unsafe_allow_html=True)

    # Top language markers
    st.markdown('<div class="label">Key language markers this week</div>', unsafe_allow_html=True)
    label_map = {
        "rumination":         "Rumination patterns",
        "negativity_bias":    "Negativity bias",
        "hopelessness":       "Hopelessness language",
        "self_blame":         "Self-blame",
        "overgeneralization": "Overgeneralization",
        "social_withdrawal":  "Social withdrawal",
    }
    rows_html = ""
    for col, val in metrics["top_markers"].items():
        display = label_map.get(col, col.replace("_", " ").title())
        level   = "High" if val > 6 else "Moderate" if val > 3.5 else "Low"
        rows_html += f"""
        <div class="lang-row">
          <div class="dot"></div>
          <div><strong>{display}</strong>&nbsp;
            <span style="color:{C['text_3']};font-size:0.76rem;">({level})</span>
          </div>
        </div>"""
    st.markdown(rows_html, unsafe_allow_html=True)

    st.markdown('<hr class="divider">', unsafe_allow_html=True)

    # AI summary
    st.markdown('<div class="label">AI-generated summary</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="ai-box">{AI_SUMMARY}</div>', unsafe_allow_html=True)

    st.markdown('<hr class="divider">', unsafe_allow_html=True)

    # Last check-in
    st.markdown(f"""
    <div style="font-size:0.77rem;color:{C['text_3']};">
      <span style="font-weight:600;color:{C['text_2']};">Last check-in:</span>
      &nbsp;{metrics['last_checkin']}
    </div>
    """, unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# render_trends  —  View 1
# ─────────────────────────────────────────────────────────────────────────────

def render_trends(daily: pd.DataFrame, weekly: pd.DataFrame, metrics: dict) -> None:
    st.markdown('<div class="s-head">Stress & Well-being Trends</div>', unsafe_allow_html=True)
    st.markdown('<div class="s-sub">Longitudinal view of core wellbeing indicators · 7-day rolling average applied</div>', unsafe_allow_html=True)

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.metric("Avg. Stress (this week)", f"{metrics['stress_now']}/10",
                  delta=f"{metrics['stress_delta']:+.1f} vs last week",
                  delta_color="inverse")
    with c2:
        st.metric("WHO-5 Score", f"{metrics['who5_now']:.1f}/25",
                  delta=f"{metrics['who5_delta']:+.1f} vs last week")
    with c3:
        sr = round(daily.tail(7)["speech_rate"].mean())
        st.metric("Avg. Speech Rate", f"{sr} wpm")
    with c4:
        pv = round(daily.tail(7)["pitch_variance"].mean(), 1)
        st.metric("Pitch Variability", f"{pv}/10")

    st.markdown("<br>", unsafe_allow_html=True)

    ca, cb = st.columns(2)
    with ca:
        st.markdown('<div class="label">Daily Stress Indicator</div>', unsafe_allow_html=True)
        st.plotly_chart(line_chart(daily, ["stress"], ["Stress"], [C["amber"]], "Score (0–10)"),
                        use_container_width=True, config={"displayModeBar": False})
    with cb:
        st.markdown('<div class="label">WHO-5 Well-being Score (weekly)</div>', unsafe_allow_html=True)
        st.plotly_chart(line_chart(weekly, ["who5"], ["WHO-5"], [C["green"]], "Score (0–25)", smooth=False),
                        use_container_width=True, config={"displayModeBar": False})

    cc, cd = st.columns(2)
    with cc:
        st.markdown('<div class="label">Speech Rate</div>', unsafe_allow_html=True)
        st.plotly_chart(line_chart(daily, ["speech_rate"], ["Words / min"], [C["blue"]], "wpm"),
                        use_container_width=True, config={"displayModeBar": False})
    with cd:
        st.markdown('<div class="label">Pitch Variability</div>', unsafe_allow_html=True)
        st.plotly_chart(line_chart(daily, ["pitch_variance"], ["Pitch variance"], [C["green"]], "Score (0–10)"),
                        use_container_width=True, config={"displayModeBar": False})


# ─────────────────────────────────────────────────────────────────────────────
# render_language_patterns  —  View 2
# ─────────────────────────────────────────────────────────────────────────────

def render_language_patterns(daily: pd.DataFrame, **_) -> None:
    st.markdown('<div class="s-head">Language Patterns</div>', unsafe_allow_html=True)
    st.markdown('<div class="s-sub">Cognitive and semantic markers detected in check-in transcripts</div>', unsafe_allow_html=True)

    last_week   = daily.tail(7)
    lang_cols   = ["negativity_bias","hopelessness","rumination","self_blame","overgeneralization","social_withdrawal"]
    lang_labels = ["Negativity Bias","Hopelessness","Rumination","Self-Blame","Overgeneralization","Social Withdrawal"]
    lang_vals   = [round(last_week[c].mean(), 2) for c in lang_cols]
    bar_colors  = [C["red"] if v > 5.5 else C["amber"] if v > 3.5 else C["green"] for v in lang_vals]

    col_l, col_r = st.columns([1.25, 1])

    with col_l:
        st.markdown('<div class="label">Cognitive distortion markers — this week (avg. 0–10)</div>', unsafe_allow_html=True)
        st.plotly_chart(bar_chart(lang_labels, lang_vals, bar_colors),
                        use_container_width=True, config={"displayModeBar": False})

        st.markdown('<div class="label">Absolute word usage over time</div>', unsafe_allow_html=True)
        st.plotly_chart(line_chart(daily, ["absolute_words"], ["Absolute words / session"],
                                   [C["amber"]], "Count per session"),
                        use_container_width=True, config={"displayModeBar": False})

    with col_r:
        st.markdown('<div class="label">Marker profile (this week)</div>', unsafe_allow_html=True)
        st.plotly_chart(radar_chart(lang_labels, lang_vals, C["blue"]),
                        use_container_width=True, config={"displayModeBar": False})

        st.markdown('<div class="label">Rumination & self-blame trend</div>', unsafe_allow_html=True)
        st.plotly_chart(line_chart(daily, ["rumination","self_blame"],
                                   ["Rumination","Self-blame"],
                                   [C["blue"], C["amber"]], "Score (0–10)"),
                        use_container_width=True, config={"displayModeBar": False})

    st.markdown('<div class="label" style="margin-top:0.3rem;">Example transcript snippets (anonymised)</div>', unsafe_allow_html=True)
    cols_top = st.columns(3)
    cols_bot = st.columns(2)
    for i, (text, tag) in enumerate(SNIPPETS):
        target = cols_top[i] if i < 3 else cols_bot[i - 3]
        with target:
            st.markdown(f"""
            <div class="snippet">"{text}"
              <br><span class="tag">{tag}</span>
            </div>""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# render_voice_features  —  View 3
# ─────────────────────────────────────────────────────────────────────────────

def render_voice_features(daily: pd.DataFrame, **_) -> None:
    st.markdown('<div class="s-head">Voice Features</div>', unsafe_allow_html=True)
    st.markdown('<div class="s-sub">Paralinguistic indicators extracted from audio check-ins · 7-day rolling average</div>', unsafe_allow_html=True)

    col_l, col_r = st.columns(2)

    with col_l:
        st.markdown('<div class="label">Speech rate over time</div>', unsafe_allow_html=True)
        st.plotly_chart(line_chart(daily, ["speech_rate"], ["Words / min"], [C["blue"]], "wpm"),
                        use_container_width=True, config={"displayModeBar": False})

        st.markdown('<div class="label">Pitch variance & instability</div>', unsafe_allow_html=True)
        st.plotly_chart(line_chart(daily, ["pitch_variance","pitch_instability"],
                                   ["Pitch variance","Pitch instability"],
                                   [C["green"], C["amber"]], "Score (0–10)"),
                        use_container_width=True, config={"displayModeBar": False})

    with col_r:
        st.markdown('<div class="label">Pause duration distribution</div>', unsafe_allow_html=True)
        st.plotly_chart(histogram(daily["pause_duration"], C["blue"], "Pause duration (s)"),
                        use_container_width=True, config={"displayModeBar": False})

        st.markdown('<div class="label">Flat affect · energy variability · breathiness</div>', unsafe_allow_html=True)
        st.plotly_chart(line_chart(daily, ["flat_affect","energy_variability","breathiness"],
                                   ["Flat affect","Energy variability","Breathiness"],
                                   [C["amber"], C["green"], C["blue"]], "Score (0–10)"),
                        use_container_width=True, config={"displayModeBar": False})

    # Voice quality radar
    lw = daily.tail(7)
    v_labels = ["Speech Rate","Pitch Instability","Vocal Tension","Flat Affect","Breathiness","Filler Rate"]
    v_vals   = [
        round(max(0, min(10, (lw["speech_rate"].mean() - 60) / 14)), 1),
        round(lw["pitch_instability"].mean(), 1),
        round(lw["vocal_tension"].mean(), 1),
        round(lw["flat_affect"].mean(), 1),
        round(lw["breathiness"].mean(), 1),
        round(lw["filler_rate"].mean(), 1),
    ]

    st.markdown('<div class="label" style="margin-top:0.3rem;">Voice quality profile — this week</div>', unsafe_allow_html=True)
    st.plotly_chart(radar_chart(v_labels, v_vals, C["green"]),
                    use_container_width=True, config={"displayModeBar": False})


# ─────────────────────────────────────────────────────────────────────────────
# render_functioning  —  View 4
# ─────────────────────────────────────────────────────────────────────────────

def render_functioning(weekly: pd.DataFrame, **_) -> None:
    st.markdown('<div class="s-head">Functioning & Daily Impact</div>', unsafe_allow_html=True)
    st.markdown('<div class="s-sub">Self-reported weekly functioning indicators</div>', unsafe_allow_html=True)

    last, prev = weekly.iloc[-1], weekly.iloc[-2]

    c1, c2, c3, c4 = st.columns(4)
    items = [
        (c1, "Stress Interference", "stress_interference", True),
        (c2, "Sleep Quality",        "sleep_quality",       False),
        (c3, "Social Connectedness", "social_connectedness",False),
        (c4, "Desync Indicator",     "desync_indicator",    True),
    ]
    for col, label, key, inverse in items:
        delta = round(float(last[key]) - float(prev[key]), 1)
        delta_color = "inverse" if (inverse and delta > 0) or (not inverse and delta < 0) else "normal"
        with col:
            st.metric(label, f"{last[key]:.1f}/10", delta=f"{delta:+.1f}", delta_color=delta_color)

    st.markdown("<br>", unsafe_allow_html=True)

    ca, cb = st.columns(2)
    with ca:
        st.markdown('<div class="label">Sleep quality & social connectedness</div>', unsafe_allow_html=True)
        st.plotly_chart(line_chart(weekly, ["sleep_quality","social_connectedness"],
                                   ["Sleep quality","Social connectedness"],
                                   [C["blue"], C["green"]], "Score (0–10)", smooth=False),
                        use_container_width=True, config={"displayModeBar": False})
    with cb:
        st.markdown('<div class="label">Stress interference & desync indicator</div>', unsafe_allow_html=True)
        st.plotly_chart(line_chart(weekly, ["stress_interference","desync_indicator"],
                                   ["Stress interference","Desync indicator"],
                                   [C["amber"], C["red"]], "Score (0–10)", smooth=False),
                        use_container_width=True, config={"displayModeBar": False})

    st.markdown('<div class="label">All functioning indicators</div>', unsafe_allow_html=True)
    fig = go.Figure()
    for col, label, color in [
        ("stress_interference","Stress interference", C["amber"]),
        ("sleep_quality",       "Sleep quality",       C["blue"]),
        ("social_connectedness","Social connectedness",C["green"]),
        ("desync_indicator",    "Desync indicator",    C["red"]),
    ]:
        fig.add_trace(go.Scatter(
            x=weekly["date"], y=weekly[col], name=label,
            mode="lines+markers",
            line=dict(color=color, width=2),
            marker=dict(size=5),
            hovertemplate=f"{label}: %{{y:.1f}}<extra></extra>",
        ))
    fig.update_layout(**_layout(height=270))
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})


# ─────────────────────────────────────────────────────────────────────────────
# render_session_notes  —  View 5
# ─────────────────────────────────────────────────────────────────────────────

def render_session_notes(**_) -> None:
    st.markdown('<div class="s-head">Session Notes</div>', unsafe_allow_html=True)
    st.markdown('<div class="s-sub">Notes persist within this browser session only.</div>', unsafe_allow_html=True)

    if "session_notes" not in st.session_state:
        st.session_state.session_notes = ""

    col_note, col_ctx = st.columns([2.2, 1])

    with col_note:
        notes = st.text_area(
            "Notes",
            value=st.session_state.session_notes,
            height=320,
            placeholder=(
                "Type your session observations here…\n\n"
                "E.g.: Patient mentioned increased work stress. Rumination themes consistent "
                "with audio markers. Plan to explore CBT reframing of self-blame next session."
            ),
            label_visibility="collapsed",
        )
        if notes != st.session_state.session_notes:
            st.session_state.session_notes = notes

        s1, s2, _ = st.columns([1, 1, 2.5])
        with s1:
            if st.button("💾  Save note", use_container_width=True):
                st.success("Note saved to session.")
        with s2:
            if st.button("🗑  Clear", use_container_width=True):
                st.session_state.session_notes = ""
                st.rerun()

    with col_ctx:
        st.markdown(f"""
        <div class="card-alt">
          <div class="label">Context</div>
          <div style="font-size:0.82rem;line-height:1.75;color:{C['text']};">
            <b>Patient:</b> A. Thompson<br>
            <b>Session #:</b> 32<br>
            <b>Risk level:</b> Moderate<br>
            <b>Last WHO-5:</b> 17.4<br>
            <b>Last check-in:</b> Today
          </div>
        </div>
        <div class="card-alt">
          <div class="label">Previous note excerpt</div>
          <div style="font-size:0.81rem;font-style:italic;line-height:1.65;color:{C['text_2']};">
            "Patient showed signs of progress with social re-engagement.
            Rumination remains primary focus. Assigned thought-record homework."
          </div>
          <div style="font-size:0.73rem;color:{C['text_3']};margin-top:0.4rem;">Logged 7 days ago</div>
        </div>
        """, unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# Navigation config
# ─────────────────────────────────────────────────────────────────────────────

VIEWS = [
    ("📈  Stress & Well-being",      "trends"),
    ("🗣  Language Patterns",         "language"),
    ("🎙  Voice Features",            "voice"),
    ("🏠  Functioning & Daily Impact","functioning"),
    ("📝  Session Notes",             "notes"),
]

RENDER = {
    "trends":      render_trends,
    "language":    render_language_patterns,
    "voice":       render_voice_features,
    "functioning": render_functioning,
    "notes":       render_session_notes,
}


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    inject_css()

    # Session state initialisation
    if "active_view" not in st.session_state:
        st.session_state.active_view = "trends"
    if "data" not in st.session_state:
        daily, weekly = load_data()
        st.session_state.data    = (daily, weekly)
        st.session_state.metrics = compute_summary_metrics(daily, weekly)

    daily, weekly = st.session_state.data
    metrics       = st.session_state.metrics

    # App header
    st.markdown(f"""
    <div style="display:flex;align-items:center;gap:0.55rem;margin-bottom:1.4rem;">
      <span style="font-size:1.4rem;">🌿</span>
      <span style="font-size:1.2rem;font-weight:700;color:{C['text']};">MindTrack</span>
      <span style="font-size:0.74rem;color:{C['text_3']};margin-left:0.15rem;
                   background:{C['border']};padding:0.15rem 0.6rem;border-radius:8px;">
        Therapist Dashboard
      </span>
    </div>
    """, unsafe_allow_html=True)

    # Three-column layout: quick insights | nav buttons | content
    left_col, nav_col, content_col = st.columns([1.35, 0.72, 3.6])

    # ── Left panel ──────────────────────────────────────────────────────────
    with left_col:
        render_quick_insights(daily, weekly, metrics)

    # ── Nav buttons ─────────────────────────────────────────────────────────
    with nav_col:
        st.markdown(f'<div class="label">Views</div>', unsafe_allow_html=True)
        for label, key in VIEWS:
            is_active = st.session_state.active_view == key
            if is_active:
                # Active item rendered as a styled non-clickable div
                st.markdown(f"""
                <div style="background:{C['green']};color:white;border-radius:10px;
                            padding:0.55rem 0.85rem;font-size:0.81rem;font-weight:600;
                            margin-bottom:0.3rem;cursor:default;">
                  {label}
                </div>""", unsafe_allow_html=True)
            else:
                if st.button(label, key=f"nav_{key}", use_container_width=True):
                    st.session_state.active_view = key
                    st.rerun()

    # ── Content pane ─────────────────────────────────────────────────────────
    with content_col:
        RENDER[st.session_state.active_view](daily=daily, weekly=weekly, metrics=metrics)


if __name__ == "__main__":
    main()
