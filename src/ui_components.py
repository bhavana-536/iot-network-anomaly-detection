"""
src/ui_components.py
====================
Streamlit UI helper components:
  - CSS injection (dark theme, metric cards, badges)
  - metric_card()  – KPI tile
  - show_detail_panel()  – full flow inspection panel
"""

from __future__ import annotations

import pandas as pd
import streamlit as st

from src.charts import RISK_COLORS, VERDICT_COLORS, top_features_bar
from src.data_loader import parse_top_features

# ── Custom CSS ────────────────────────────────────────────────────────────────
_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

html, body, [class*="css"]       { font-family: 'Inter', system-ui, sans-serif !important; }

/* ── Metric card ─────────────────────────────────────────────────────────── */
.kpi-card {
    background: linear-gradient(145deg, #1A2235 0%, #111827 100%);
    border: 1px solid #232F48;
    border-radius: 14px;
    padding: 18px 14px 14px;
    text-align: center;
    transition: transform .18s ease, box-shadow .18s ease;
    min-height: 106px;
}
.kpi-card:hover {
    transform: translateY(-3px);
    box-shadow: 0 10px 30px rgba(0,0,0,.45);
}
.kpi-label {
    font-size: .70rem;
    font-weight: 700;
    letter-spacing: .09em;
    text-transform: uppercase;
    color: #6B7A99;
    margin-bottom: 6px;
}
.kpi-value {
    font-size: 2.0rem;
    font-weight: 700;
    line-height: 1.1;
}
.kpi-sub {
    font-size: .70rem;
    color: #6B7A99;
    margin-top: 4px;
}

/* ── Verdict / risk colour badges ────────────────────────────────────────── */
.badge {
    display: inline-block;
    padding: 2px 11px;
    border-radius: 999px;
    font-size: .73rem;
    font-weight: 700;
    letter-spacing: .04em;
}
.v-normal   { background: rgba(91,155,213,.18);  color: #5B9BD5; }
.v-attack   { background: rgba(224,82,82,.18);   color: #E05252; }
.v-anomaly  { background: rgba(168,127,216,.18); color: #A87FD8; }
.r-low      { background: rgba(45,212,191,.15);  color: #2DD4BF; }
.r-medium   { background: rgba(245,158,11,.15);  color: #F59E0B; }
.r-high     { background: rgba(239,68,68,.15);   color: #EF4444; }

/* ── Source tag ──────────────────────────────────────────────────────────── */
.source-tag {
    display: inline-block;
    padding: 3px 13px;
    border-radius: 999px;
    font-size: .70rem;
    font-weight: 700;
    letter-spacing: .10em;
    text-transform: uppercase;
}
.src-sample { background: rgba(245,158,11,.12); color: #F59E0B;
              border: 1px solid rgba(245,158,11,.30); }
.src-live   { background: rgba(45,212,191,.12); color: #2DD4BF;
              border: 1px solid rgba(45,212,191,.30); }

/* ── Section rule ────────────────────────────────────────────────────────── */
.section-rule { border: none; border-top: 1px solid #1E2A40; margin: 18px 0; }

/* ── Alert detail panel ──────────────────────────────────────────────────── */
.detail-wrap {
    background: linear-gradient(145deg, #1A2235 0%, #111827 100%);
    border: 1px solid #232F48;
    border-radius: 14px;
    padding: 22px 22px 18px;
    margin-top: 6px;
}
.detail-hdr {
    font-size: 1.0rem;
    font-weight: 700;
    color: #E8EAF0;
    padding-bottom: 10px;
    border-bottom: 1px solid #232F48;
    margin-bottom: 16px;
}
.info-lbl {
    font-size: .68rem;
    font-weight: 700;
    letter-spacing: .07em;
    text-transform: uppercase;
    color: #6B7A99;
    margin-bottom: 2px;
}
.info-val {
    font-size: .92rem;
    color: #E8EAF0;
    margin-bottom: 12px;
}

/* Sidebar */
section[data-testid="stSidebar"] {
    background: #0C1120;
    border-right: 1px solid #1A2438;
}
section[data-testid="stSidebar"] .stSelectbox label,
section[data-testid="stSidebar"] .stMultiSelect label,
section[data-testid="stSidebar"] .stTextInput label {
    color: #8898B5 !important;
    font-size: .82rem;
}
</style>
"""


def inject_css() -> None:
    """Inject custom CSS into the Streamlit page. Call once at the top of dashboard.py."""
    st.markdown(_CSS, unsafe_allow_html=True)


# ── Metric card ───────────────────────────────────────────────────────────────
def metric_card(label: str, value: str | int, color: str = "#E8EAF0", sub: str = "") -> None:
    """Render a single KPI tile."""
    sub_html = f'<div class="kpi-sub">{sub}</div>' if sub else ""
    st.markdown(
        f'<div class="kpi-card">'
        f'  <div class="kpi-label">{label}</div>'
        f'  <div class="kpi-value" style="color:{color}">{value}</div>'
        f'  {sub_html}'
        f'</div>',
        unsafe_allow_html=True,
    )


# ── Badge helpers (used only in detail panel) ─────────────────────────────────
def _verdict_badge(verdict: str) -> str:
    css = {"Normal": "v-normal", "Known Attack": "v-attack", "Unknown Anomaly": "v-anomaly"}
    cls_name = css.get(verdict, "v-normal")
    return f'<span class="badge {cls_name}">{verdict}</span>'


def _risk_badge(risk: str) -> str:
    css = {"Low": "r-low", "Medium": "r-medium", "High": "r-high"}
    cls_name = css.get(risk, "r-medium")
    return f'<span class="badge {cls_name}">{risk}</span>'


# ── Detail panel ──────────────────────────────────────────────────────────────
def show_detail_panel(row: pd.Series) -> None:
    """
    Render a full-width detail panel for the selected alert row.
    Shows: flow identifiers, RF/IF scores, verdict badges,
           feature contribution bar-chart, explanation and recommendation.
    """
    verdict = str(row.get("verdict", "—"))
    risk    = str(row.get("risk_level", "—"))

    verdict_color = VERDICT_COLORS.get(verdict, "#888")
    risk_color    = RISK_COLORS.get(risk, "#888")
    needs_review  = row.get("needs_review", False)

    st.markdown('<div class="detail-wrap">', unsafe_allow_html=True)
    st.markdown(
        f'<div class="detail-hdr">🔍&nbsp; Flow Detail &nbsp;·&nbsp; '
        f'<code>flow_id {row.get("flow_id", "?")}</code></div>',
        unsafe_allow_html=True,
    )

    # ── Row 1: network identifiers ────────────────────────────────────────────
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown('<div class="info-lbl">Source IP</div>'
                    f'<div class="info-val"><code>{row.get("src_ip","—")}</code></div>',
                    unsafe_allow_html=True)
    with c2:
        st.markdown('<div class="info-lbl">Destination IP</div>'
                    f'<div class="info-val"><code>{row.get("dst_ip","—")}</code></div>',
                    unsafe_allow_html=True)
    with c3:
        st.markdown('<div class="info-lbl">Protocol</div>'
                    f'<div class="info-val">{row.get("protocol","—")}</div>',
                    unsafe_allow_html=True)
    with c4:
        ts = row.get("timestamp", "")
        ts_str = str(ts)[:19] if ts else "—"
        st.markdown(f'<div class="info-lbl">Timestamp</div>'
                    f'<div class="info-val">{ts_str}</div>',
                    unsafe_allow_html=True)

    st.markdown('<hr class="section-rule">', unsafe_allow_html=True)

    # ── Row 2: model scores ───────────────────────────────────────────────────
    c5, c6, c7, c8 = st.columns(4)
    with c5:
        st.markdown('<div class="info-lbl">RF Class</div>'
                    f'<div class="info-val"><b>{row.get("rf_class","—")}</b></div>',
                    unsafe_allow_html=True)
    with c6:
        prob = float(row.get("rf_probability", 0))
        bar_fill = int(prob * 100)
        bar_clr  = "#EF4444" if prob >= 0.8 else "#F59E0B" if prob >= 0.6 else "#5B9BD5"
        st.markdown(
            f'<div class="info-lbl">RF Probability</div>'
            f'<div class="info-val"><b>{prob:.2%}</b></div>'
            f'<div style="background:#1E2A40;border-radius:4px;height:6px;margin-top:-8px">'
            f'<div style="background:{bar_clr};width:{bar_fill}%;height:6px;border-radius:4px"></div></div>',
            unsafe_allow_html=True,
        )
    with c7:
        ifs = float(row.get("if_score", 0))
        ifs_clr = "#EF4444" if ifs >= 0.999 else "#F59E0B" if ifs >= 0.99 else "#2DD4BF"
        st.markdown(f'<div class="info-lbl">IF Score (percentile)</div>'
                    f'<div class="info-val" style="color:{ifs_clr}"><b>{ifs:.4f}</b></div>',
                    unsafe_allow_html=True)
    with c8:
        nr_icon = "⚠️ Yes" if needs_review else "✅ No"
        st.markdown(f'<div class="info-lbl">Needs Review</div>'
                    f'<div class="info-val">{nr_icon}</div>',
                    unsafe_allow_html=True)

    st.markdown('<hr class="section-rule">', unsafe_allow_html=True)

    # ── Row 3: verdict + risk badges ─────────────────────────────────────────
    c9, c10, _ = st.columns([1, 1, 2])
    with c9:
        st.markdown(
            f'<div class="info-lbl">Verdict</div>'
            f'<div style="margin-top:4px">{_verdict_badge(verdict)}</div>',
            unsafe_allow_html=True,
        )
    with c10:
        st.markdown(
            f'<div class="info-lbl">Risk Level</div>'
            f'<div style="margin-top:4px">{_risk_badge(risk)}</div>',
            unsafe_allow_html=True,
        )

    st.markdown('<hr class="section-rule">', unsafe_allow_html=True)

    # ── Feature contributions chart ───────────────────────────────────────────
    features = parse_top_features(str(row.get("top_features", "[]")))
    if features:
        col_chart, col_table = st.columns([3, 2])
        with col_chart:
            fig = top_features_bar(features)
            st.plotly_chart(fig, use_container_width=True,
                            config={"displayModeBar": False})
        with col_table:
            st.markdown('<div class="info-lbl" style="margin-bottom:6px">Feature values</div>',
                        unsafe_allow_html=True)
            feat_df = pd.DataFrame(features)[["feature", "value", "contribution"]]
            st.dataframe(feat_df, use_container_width=True, hide_index=True, height=215)
    else:
        st.caption("No feature data available.")

    st.markdown('<hr class="section-rule">', unsafe_allow_html=True)

    # ── Explanation & recommendation ──────────────────────────────────────────
    col_exp, col_rec = st.columns(2)
    with col_exp:
        st.markdown("**🧠 Analyst Explanation**")
        st.info(str(row.get("explanation", "No explanation available.")))
    with col_rec:
        st.markdown("**🛡️ Recommendation**")
        st.warning(str(row.get("recommendation", "No recommendation available.")))

    st.markdown("</div>", unsafe_allow_html=True)
