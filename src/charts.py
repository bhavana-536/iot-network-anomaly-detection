"""
src/charts.py
=============
All Plotly chart builder functions for the dashboard.
Each function accepts a filtered DataFrame and returns a plotly Figure.
The dashboard never manipulates go.Figure objects directly.
"""

from __future__ import annotations

import plotly.express as px
import plotly.graph_objects as go
import pandas as pd

# ── Colour palettes (used by ui_components.py too) ───────────────────────────
VERDICT_COLORS: dict[str, str] = {
    "Normal":          "#5B9BD5",   # steel-blue
    "Known Attack":    "#E05252",   # red
    "Unknown Anomaly": "#A87FD8",   # purple
}

RISK_COLORS: dict[str, str] = {
    "Low":    "#2DD4BF",   # teal
    "Medium": "#F59E0B",   # amber
    "High":   "#EF4444",   # red
}

CLASS_COLORS: dict[str, str] = {
    "Benign":               "#5B9BD5",
    "DoS_DDoS":             "#EF4444",
    "Reconnaissance":       "#F59E0B",
    "BruteForce_Password":  "#F97316",
    "Injection_Web":        "#A855F7",
    "MITM_Spoofing":        "#EC4899",
    "Botnet_Malware":       "#8B5CF6",
}

_TEMPLATE  = "plotly_dark"
_PAPER_BG  = "rgba(0,0,0,0)"
_PLOT_BG   = "rgba(0,0,0,0)"
_FONT      = dict(family="Inter, system-ui, sans-serif", size=12, color="#C8CFE0")
_MARGIN    = dict(l=8, r=8, t=36, b=8)


# ── Internal helpers ──────────────────────────────────────────────────────────
def _base(fig: go.Figure, height: int = 270) -> go.Figure:
    """Apply consistent dark styling to every figure."""
    fig.update_layout(
        template=_TEMPLATE,
        paper_bgcolor=_PAPER_BG,
        plot_bgcolor=_PLOT_BG,
        font=_FONT,
        margin=_MARGIN,
        height=height,
        legend=dict(bgcolor="rgba(0,0,0,0)", borderwidth=0, font=dict(size=11)),
    )
    return fig


def _empty(msg: str, height: int = 270) -> go.Figure:
    fig = go.Figure()
    fig.add_annotation(
        text=msg, x=0.5, y=0.5,
        xref="paper", yref="paper",
        showarrow=False,
        font=dict(size=14, color="#555"),
    )
    return _base(fig, height)


# ── Public chart functions ────────────────────────────────────────────────────

def alerts_over_time(df: pd.DataFrame, bin_min: int = 2) -> go.Figure:
    """
    Stacked area chart of alert counts, binned every `bin_min` minutes,
    coloured by verdict.
    """
    if df.empty or "timestamp" not in df.columns:
        return _empty("No data to display")

    freq = f"{bin_min}min"
    tmp  = df.copy()
    tmp["bin"] = tmp["timestamp"].dt.floor(freq)
    counts = (
        tmp.groupby(["bin", "verdict"])
        .size()
        .reset_index(name="count")
    )

    order = ["Normal", "Known Attack", "Unknown Anomaly"]
    fig = px.area(
        counts,
        x="bin", y="count", color="verdict",
        color_discrete_map=VERDICT_COLORS,
        category_orders={"verdict": order},
        labels={"bin": "", "count": "Alerts", "verdict": ""},
    )
    fig.update_traces(line=dict(width=1.5))
    _base(fig, 265)
    fig.update_layout(title_text="Alerts Over Time", title_x=0.01)
    return fig


def attack_class_bar(df: pd.DataFrame) -> go.Figure:
    """Horizontal bar chart: alert count per RF attack class."""
    if df.empty or "rf_class" not in df.columns:
        return _empty("No data to display")

    counts  = df["rf_class"].value_counts().reset_index()
    counts.columns = ["class", "count"]
    counts  = counts.sort_values("count")
    colors  = [CLASS_COLORS.get(c, "#888") for c in counts["class"]]

    fig = go.Figure(go.Bar(
        x=counts["count"],
        y=counts["class"],
        orientation="h",
        marker_color=colors,
        text=counts["count"],
        textposition="auto",
        hovertemplate="%{y}: %{x}<extra></extra>",
    ))
    _base(fig, 265)
    fig.update_layout(
        title_text="Attack Class Distribution", title_x=0.01,
        xaxis_title="Count",
        bargap=0.28,
    )
    return fig


def risk_donut(df: pd.DataFrame) -> go.Figure:
    """Donut chart of risk-level share (Low / Medium / High)."""
    if df.empty or "risk_level" not in df.columns:
        return _empty("No data to display")

    order  = ["Low", "Medium", "High"]
    counts = df["risk_level"].value_counts().reindex(order, fill_value=0)
    colors = [RISK_COLORS[r] for r in order]

    fig = go.Figure(go.Pie(
        labels=counts.index,
        values=counts.values,
        hole=0.58,
        marker_colors=colors,
        textinfo="label+percent",
        hovertemplate="%{label}: %{value}<extra></extra>",
    ))
    _base(fig, 265)
    fig.update_layout(
        title_text="Risk Distribution", title_x=0.01,
        showlegend=False,
    )
    total = int(counts.sum())
    fig.add_annotation(
        text=f"<b>{total}</b><br><span style='font-size:0.85em'>alerts</span>",
        x=0.5, y=0.5, showarrow=False,
        font=dict(size=16, color="#E8EAF0"),
    )
    return fig


def scatter_rf_if(df: pd.DataFrame) -> go.Figure:
    """
    Scatter: RF probability (x) vs IF score (y), coloured by verdict.
    Decision-boundary lines for tau and tau_high are drawn.
    """
    needed = {"rf_probability", "if_score", "verdict"}
    if df.empty or not needed.issubset(df.columns):
        return _empty("No data to display")

    sample = df.sample(min(len(df), 600), random_state=42)

    fig = px.scatter(
        sample,
        x="rf_probability",
        y="if_score",
        color="verdict",
        color_discrete_map=VERDICT_COLORS,
        hover_data={"rf_class": True, "risk_level": True,
                    "src_ip": True, "if_score": ":.4f", "rf_probability": ":.3f"},
        opacity=0.70,
        labels={"rf_probability": "RF Probability", "if_score": "IF Score (percentile)"},
        size_max=6,
    )
    fig.update_traces(marker=dict(size=6, line=dict(width=0)))

    # Threshold reference lines
    fig.add_hline(y=0.99,  line_dash="dash", line_color="#F59E0B", line_width=1,
                  annotation_text="τ = 0.99", annotation_position="top right",
                  annotation_font_color="#F59E0B")
    fig.add_hline(y=0.999, line_dash="dot",  line_color="#EF4444", line_width=1,
                  annotation_text="τ_high = 0.999", annotation_position="top right",
                  annotation_font_color="#EF4444")
    fig.add_vline(x=0.80,  line_dash="dash", line_color="#5B9BD5", line_width=1,
                  annotation_text="p = 0.80", annotation_position="top left",
                  annotation_font_color="#5B9BD5")

    _base(fig, 285)
    fig.update_layout(title_text="RF Probability vs IF Score", title_x=0.01)
    return fig


def risk_gauge(df: pd.DataFrame, minutes: int = 5) -> go.Figure:
    """
    Gauge chart showing the share of High-risk alerts in the last `minutes`
    minutes of the (unfiltered) alert stream.
    """
    if df.empty or "timestamp" not in df.columns:
        return _empty("No recent data", height=230)

    cutoff = df["timestamp"].max() - pd.Timedelta(minutes=minutes)
    recent = df[df["timestamp"] >= cutoff]

    if recent.empty:
        pct, n = 0.0, 0
    else:
        n   = len(recent)
        pct = (recent["risk_level"] == "High").sum() / n * 100

    bar_color = "#2DD4BF" if pct < 30 else "#F59E0B" if pct < 70 else "#EF4444"

    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=pct,
        number={"suffix": " %", "font": {"size": 26, "color": "#E8EAF0"}},
        title={
            "text": (
                f"High-Risk Share<br>"
                f"<span style='font-size:0.78em;color:#7A8AAD'>"
                f"last {minutes} min · {n} flows</span>"
            ),
            "font": {"size": 13, "color": "#C8CFE0"},
        },
        gauge={
            "axis": {
                "range":      [0, 100],
                "tickcolor":  "#3A4560",
                "tickfont":   {"size": 10},
            },
            "bar":       {"color": bar_color, "thickness": 0.28},
            "bgcolor":   "rgba(0,0,0,0)",
            "borderwidth": 0,
            "steps": [
                {"range": [0,  30], "color": "rgba(45,212,191,0.10)"},
                {"range": [30, 70], "color": "rgba(245,158,11,0.10)"},
                {"range": [70,100], "color": "rgba(239,68,68,0.10)"},
            ],
        },
    ))
    _base(fig, 225)
    return fig


def top_features_bar(features: list[dict]) -> go.Figure:
    """
    Horizontal bar chart of feature contributions for the detail panel.
    Positive contributions are red (attack signal), negative are blue (benign).
    """
    if not features:
        return _empty("No feature data", height=210)

    names   = [str(f.get("feature", "?")) for f in features]
    contribs = [float(f.get("contribution", 0)) for f in features]
    vals    = [float(f.get("value", 0)) for f in features]
    colors  = ["#E05252" if c > 0 else "#5B9BD5" for c in contribs]

    hover = [
        f"Value: {v:.4f}<br>Contribution: {c:+.4f}"
        for v, c in zip(vals, contribs)
    ]

    fig = go.Figure(go.Bar(
        x=contribs, y=names,
        orientation="h",
        marker_color=colors,
        text=[f"{c:+.3f}" for c in contribs],
        textposition="auto",
        hovertemplate="%{y}<br>%{customdata}<extra></extra>",
        customdata=hover,
    ))
    fig.add_vline(x=0, line_color="#3A4560", line_width=1)
    _base(fig, 215)
    fig.update_layout(
        title_text="Feature Contributions (SHAP-style)", title_x=0.01,
        xaxis_title="Contribution to verdict",
        bargap=0.32,
    )
    return fig
