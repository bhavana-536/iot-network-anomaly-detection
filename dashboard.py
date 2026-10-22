"""
dashboard.py
============
AI-Based Real-Time Network Anomaly Detection – Streamlit Dashboard
(Phase 2)

Run:
    streamlit run dashboard.py

Reads alert data from the path configured in config.yaml → dashboard.alert_file,
or override the path in the sidebar at runtime.

Data source is labelled "Sample Data" or "Live Data" based on whether the
path contains "live" or "pipeline" in its name.  No model results or accuracy
numbers are displayed; the UI shows only what is in the alert file.
"""

from __future__ import annotations

import json
import os
from datetime import datetime
from pathlib import Path

import pandas as pd
import streamlit as st
import yaml

# ── Auto-refresh (streamlit-autorefresh) ──────────────────────────────────────
try:
    from streamlit_autorefresh import st_autorefresh
    _HAS_AR = True
except ImportError:
    _HAS_AR = False

# ── Project modules ───────────────────────────────────────────────────────────
from src.data_loader   import load_alerts, VALID_VALUES, REQUIRED_COLUMNS
from src.charts        import (
    alerts_over_time, attack_class_bar, risk_donut,
    scatter_rf_if, risk_gauge, VERDICT_COLORS, RISK_COLORS,
)
from src.ui_components import inject_css, metric_card, show_detail_panel
from src.pipeline_interface import start_replay   # teammate wires this

# ─────────────────────────────────────────────────────────────────────────────
#  Page configuration  (must be first Streamlit call)
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title  = "IoT Anomaly Detection",
    page_icon   = "🛡️",
    layout      = "wide",
    initial_sidebar_state = "expanded",
)


# ── Load YAML config ──────────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def _load_cfg() -> dict:
    try:
        return yaml.safe_load(Path("config.yaml").read_text(encoding="utf-8")) or {}
    except Exception:
        return {}

cfg = _load_cfg()
DEFAULT_ALERT_FILE: str = (
    cfg.get("dashboard", {}).get("alert_file", "logs/alerts.csv")
)

# ── Inject CSS ────────────────────────────────────────────────────────────────
inject_css()


# ─────────────────────────────────────────────────────────────────────────────
#  SIDEBAR
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🛡️ IoT Anomaly Detection")
    st.markdown("---")

    # ── Alert file path ───────────────────────────────────────────────────────
    alert_file: str = st.text_input(
        "Alert file path",
        value=DEFAULT_ALERT_FILE,
        help=(
            "Path to the CSV alert log.  "
            "Update `config.yaml → dashboard.alert_file` for a permanent change."
        ),
    )

    # Source label (heuristic based on filename)
    _name_lc      = Path(alert_file).stem.lower()
    _is_live      = any(k in _name_lc for k in ("live", "pipeline", "stream", "real"))
    source_label  = "Live Data" if _is_live else "Sample Data"
    source_css    = "src-live"  if _is_live else "src-sample"
    st.markdown(
        f'<span class="source-tag {source_css}">{source_label}</span>',
        unsafe_allow_html=True,
    )

    st.markdown("---")

    # ── Auto-refresh ──────────────────────────────────────────────────────────
    st.markdown("**Auto-Refresh**")
    if _HAS_AR:
        _AR_OPTS: dict[str, int] = {"Off": 0, "2 s": 2_000, "5 s": 5_000, "10 s": 10_000}
        _ar_sel   = st.selectbox("Interval", list(_AR_OPTS.keys()), index=1, label_visibility="collapsed")
        _ar_ms    = _AR_OPTS[_ar_sel]
        if _ar_ms:
            st_autorefresh(interval=_ar_ms, key="dashboard_autorefresh")
    else:
        st.caption("Install `streamlit-autorefresh` to enable auto-refresh.")
        if st.button("🔄 Refresh now", use_container_width=True):
            load_alerts.clear()
            st.rerun()

    st.markdown("---")

    # ── Filters ───────────────────────────────────────────────────────────────
    st.markdown("**Filters**")

    filt_verdict = st.multiselect(
        "Verdict",
        options=sorted(VALID_VALUES["verdict"]),
        default=sorted(VALID_VALUES["verdict"]),
    )
    filt_risk = st.multiselect(
        "Risk Level",
        options=["High", "Medium", "Low"],
        default=["High", "Medium", "Low"],
    )
    filt_class = st.multiselect(
        "Attack Class",
        options=sorted(VALID_VALUES["rf_class"]),
        default=sorted(VALID_VALUES["rf_class"]),
    )
    only_review = st.checkbox("Needs review only", value=False)

    st.markdown("**Time Window**")
    _TW_OPTS: dict[str, int] = {
        "Last 5 min":  5,
        "Last 15 min": 15,
        "Last 30 min": 30,
        "All data":    0,
    }
    _tw_sel    = st.selectbox("Window", list(_TW_OPTS.keys()), index=3, label_visibility="collapsed")
    time_window = _TW_OPTS[_tw_sel]

    st.markdown("---")

    # ── Pipeline / replay ─────────────────────────────────────────────────────
    st.markdown("**Live Pipeline**")
    if st.button("▶ Replay Demo", use_container_width=True, type="primary"):
        try:
            start_replay()    # wired by teammate via src/pipeline_interface.py
        except NotImplementedError as exc:
            st.info(
                "🔌 **Connected to the detection pipeline later.**\n\n"
                "_Your teammate wires `start_replay()` in_ "
                "`src/pipeline_interface.py` _to_ `live_pipeline.py`._"
            )

    st.markdown("---")
    st.caption(
        f"🕐 Last render: {datetime.now().strftime('%H:%M:%S')}  \n"
        "Final-Year Project · IoT Anomaly Detection"
    )


# ─────────────────────────────────────────────────────────────────────────────
#  LOAD DATA
# ─────────────────────────────────────────────────────────────────────────────
df_raw, load_warnings = load_alerts(alert_file)


# ─────────────────────────────────────────────────────────────────────────────
#  HEADER
# ─────────────────────────────────────────────────────────────────────────────
_hdr_left, _hdr_right = st.columns([5, 1])
with _hdr_left:
    st.markdown(
        "# 🛡️ IoT Network Anomaly Detection"
        "<br><span style='font-size:.85rem;color:#6B7A99'>"
        "AI-Based Real-Time Monitoring · Random Forest + Isolation Forest Hybrid"
        "</span>",
        unsafe_allow_html=True,
    )
with _hdr_right:
    st.markdown(
        f'<div style="text-align:right;padding-top:22px">'
        f'<span class="source-tag {source_css}">{source_label}</span></div>',
        unsafe_allow_html=True,
    )

# ── Warnings from loader ──────────────────────────────────────────────────────
for w in load_warnings:
    st.warning(f"⚠️ {w}")

if df_raw is None:
    st.error(
        "📂 **No alert data loaded.**  \n"
        "Run `python make_fake_alerts.py` to generate sample data, "
        "then refresh the dashboard.  \n"
        f"Configured path: `{alert_file}`"
    )
    st.stop()


# ─────────────────────────────────────────────────────────────────────────────
#  APPLY FILTERS  (df_raw is never mutated)
# ─────────────────────────────────────────────────────────────────────────────
df: pd.DataFrame = df_raw.copy()

if time_window and "timestamp" in df.columns:
    cutoff = df["timestamp"].max() - pd.Timedelta(minutes=time_window)
    df     = df[df["timestamp"] >= cutoff]

if filt_verdict and "verdict" in df.columns:
    df = df[df["verdict"].isin(filt_verdict)]

if filt_risk and "risk_level" in df.columns:
    df = df[df["risk_level"].isin(filt_risk)]

if filt_class and "rf_class" in df.columns:
    df = df[df["rf_class"].isin(filt_class)]

if only_review and "needs_review" in df.columns:
    df = df[df["needs_review"] == True]


# ─────────────────────────────────────────────────────────────────────────────
#  KPI METRIC CARDS
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("<div style='margin-top:10px'></div>", unsafe_allow_html=True)
c_t, c_ka, c_ua, c_hi, c_nr = st.columns(5)

with c_t:
    metric_card("Total Flows", f"{len(df):,}", color="#E8EAF0")
with c_ka:
    n_ka = int((df["verdict"] == "Known Attack").sum()) if "verdict" in df.columns else 0
    metric_card("Known Attacks", f"{n_ka:,}", color="#E05252")
with c_ua:
    n_ua = int((df["verdict"] == "Unknown Anomaly").sum()) if "verdict" in df.columns else 0
    metric_card("Unknown Anomalies", f"{n_ua:,}", color="#A87FD8")
with c_hi:
    n_hi = int((df["risk_level"] == "High").sum()) if "risk_level" in df.columns else 0
    metric_card("High Risk", f"{n_hi:,}", color="#EF4444")
with c_nr:
    n_nr = int((df["needs_review"] == True).sum()) if "needs_review" in df.columns else 0
    metric_card("Needs Review", f"{n_nr:,}", color="#F59E0B")

st.markdown("<div style='margin-top:8px'></div>", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
#  CHARTS – ROW 1: Time-series + Risk Gauge
# ─────────────────────────────────────────────────────────────────────────────
col_ts, col_gauge = st.columns([3, 1])

with col_ts:
    st.plotly_chart(
        alerts_over_time(df),
        use_container_width=True,
        config={"displayModeBar": False},
    )
with col_gauge:
    # Gauge always shows last 5 min of the UNFILTERED raw stream
    st.plotly_chart(
        risk_gauge(df_raw, minutes=5),
        use_container_width=True,
        config={"displayModeBar": False},
    )


# ─────────────────────────────────────────────────────────────────────────────
#  CHARTS – ROW 2: Class bar + Donut + Scatter
# ─────────────────────────────────────────────────────────────────────────────
col_bar, col_donut, col_scatter = st.columns([2, 1.5, 2.5])

with col_bar:
    st.plotly_chart(
        attack_class_bar(df),
        use_container_width=True,
        config={"displayModeBar": False},
    )
with col_donut:
    st.plotly_chart(
        risk_donut(df),
        use_container_width=True,
        config={"displayModeBar": False},
    )
with col_scatter:
    st.plotly_chart(
        scatter_rf_if(df),
        use_container_width=True,
        config={"displayModeBar": False},
    )


# ─────────────────────────────────────────────────────────────────────────────
#  ALERT TABLE
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown("### 📋 Live Alert Feed")

# ── Download + row-count row ──────────────────────────────────────────────────
_dl1, _dl2, _sp, _cnt = st.columns([1, 1, 3.5, 1.5])

with _dl1:
    csv_bytes = df.to_csv(index=False).encode("utf-8")
    st.download_button(
        "⬇ Download CSV",
        data=csv_bytes,
        file_name="alerts_filtered.csv",
        mime="text/csv",
        use_container_width=True,
    )
with _dl2:
    # Ensure top_features is parsed back to a list for JSON export
    df_export = df.copy()
    if "top_features" in df_export.columns:
        df_export["top_features"] = df_export["top_features"].apply(
            lambda v: json.loads(v) if isinstance(v, str) else v
        )
    json_bytes = df_export.to_json(
        orient="records", date_format="iso", indent=2
    ).encode("utf-8")
    st.download_button(
        "⬇ Download JSON",
        data=json_bytes,
        file_name="alerts_filtered.json",
        mime="application/json",
        use_container_width=True,
    )
with _cnt:
    st.markdown(
        f'<div style="text-align:right;padding-top:8px;'
        f'color:#6B7A99;font-size:.82rem">'
        f'Showing {len(df):,} of {len(df_raw):,} alerts</div>',
        unsafe_allow_html=True,
    )

# ── Build display table ───────────────────────────────────────────────────────
_DISP_COLS = [
    "timestamp", "src_ip", "dst_ip", "protocol",
    "rf_class", "rf_probability", "if_score",
    "verdict", "risk_level", "needs_review",
]
disp_cols   = [c for c in _DISP_COLS if c in df.columns]
display_df  = df[disp_cols].copy()

if "timestamp" in display_df.columns:
    display_df["timestamp"] = display_df["timestamp"].dt.strftime("%H:%M:%S")

# ── Column config for Streamlit table ────────────────────────────────────────
col_cfg = {}
if "rf_probability" in display_df.columns:
    col_cfg["rf_probability"] = st.column_config.ProgressColumn(
        "RF Probability", format="%.2f", min_value=0, max_value=1,
    )
if "if_score" in display_df.columns:
    col_cfg["if_score"] = st.column_config.NumberColumn(
        "IF Score", format="%.4f",
    )
if "verdict" in display_df.columns:
    col_cfg["verdict"] = st.column_config.TextColumn("Verdict", width="medium")
if "risk_level" in display_df.columns:
    col_cfg["risk_level"] = st.column_config.TextColumn("Risk", width="small")
if "needs_review" in display_df.columns:
    col_cfg["needs_review"] = st.column_config.CheckboxColumn("Review?")

# ── Render selectable table ───────────────────────────────────────────────────
_selected_rows: list[int] = []
try:
    _event = st.dataframe(
        display_df,
        use_container_width=True,
        hide_index=True,
        height=360,
        column_config=col_cfg,
        on_select="rerun",
        selection_mode="single-row",
        key="alert_table",
    )
    _selected_rows = (
        _event.selection.rows
        if hasattr(_event, "selection") and _event.selection
        else []
    )
except TypeError:
    # Older Streamlit: no on_select – fall back to index picker
    st.dataframe(display_df, use_container_width=True, hide_index=True,
                 height=360, column_config=col_cfg)
    if not df.empty:
        _sel = st.number_input(
            "Row index to inspect", min_value=0,
            max_value=max(0, len(df) - 1), value=0, step=1,
        )
        _selected_rows = [int(_sel)]

st.caption(
    "💡 Click any row to open the detail panel below.  "
    "Use the sidebar filters to narrow the view."
)


# ─────────────────────────────────────────────────────────────────────────────
#  DETAIL PANEL (shown when a row is selected)
# ─────────────────────────────────────────────────────────────────────────────
if _selected_rows:
    _idx = _selected_rows[0]
    if 0 <= _idx < len(df):
        selected_row = df.iloc[_idx]
        st.markdown("---")
        st.markdown("### 🔍 Alert Detail")
        show_detail_panel(selected_row)
