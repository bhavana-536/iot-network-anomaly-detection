"""
src/data_loader.py
==================
Robust CSV loading, schema validation, and caching for the dashboard.

Designed for concurrent access: a second process may be appending rows while
the dashboard reads the file.  We handle:
  - Missing file  →  friendly error string
  - Empty file    →  friendly error string
  - Partial last line (concurrent write)  →  strip and parse the rest
  - Missing / extra columns  →  warning, not crash
  - Bad value spellings      →  warning, not crash
  - Unreadable timestamps    →  drop those rows with a warning
"""

import io
import json
import logging
from pathlib import Path
from typing import Optional

import pandas as pd
import streamlit as st

log = logging.getLogger(__name__)

# ── Schema ────────────────────────────────────────────────────────────────────
REQUIRED_COLUMNS: list[str] = [
    "timestamp", "flow_id", "src_ip", "dst_ip", "protocol",
    "rf_class", "rf_probability", "if_score", "verdict",
    "risk_level", "needs_review", "top_features",
    "explanation", "recommendation",
]

VALID_VALUES: dict[str, set] = {
    "verdict":    {"Normal", "Known Attack", "Unknown Anomaly"},
    "risk_level": {"Low", "Medium", "High"},
    "protocol":   {"TCP", "UDP", "ICMP", "MQTT"},
    "rf_class":   {
        "Benign", "DoS_DDoS", "Reconnaissance", "BruteForce_Password",
        "Injection_Web", "MITM_Spoofing", "Botnet_Malware",
    },
}


# ── File reader (handles partial last line) ───────────────────────────────────
def _safe_read_text(path: str) -> tuple[Optional[str], Optional[str]]:
    """
    Read a CSV file defensively.
    Returns (csv_text, error_message).
    error_message is None on success.
    """
    p = Path(path)
    if not p.exists():
        return None, f"Alert file not found: **{path}**"

    try:
        raw = p.read_bytes()
    except OSError as exc:
        return None, f"Cannot read alert file: {exc}"

    if not raw.strip():
        return None, "Alert file is empty. Run `python make_fake_alerts.py` to generate data."

    text  = raw.decode("utf-8", errors="replace")
    lines = text.splitlines()

    if len(lines) < 2:
        return None, "Alert file has no data rows (header only)."

    # Strip an incomplete trailing line (partial concurrent write):
    # a complete row has the same number of commas as the header.
    header_commas = lines[0].count(",")
    if lines[-1].count(",") < header_commas:
        lines = lines[:-1]

    return "\n".join(lines), None


# ── Cached loader ─────────────────────────────────────────────────────────────
@st.cache_data(ttl=2, show_spinner=False)
def load_alerts(path: str) -> tuple[Optional[pd.DataFrame], list[str]]:
    """
    Load, validate, and return (dataframe_or_None, list_of_warning_strings).

    Results are cached for 2 seconds; auto-refresh calls clear the cache
    via st_autorefresh which forces a full script rerun.
    """
    warnings: list[str] = []

    csv_text, err = _safe_read_text(path)
    if err is not None:
        return None, [err]

    try:
        df = pd.read_csv(io.StringIO(csv_text), low_memory=False)
    except Exception as exc:
        return None, [f"Failed to parse CSV: {exc}"]

    if df.empty:
        return None, ["Alert file has no data rows."]

    # ── Column checks ─────────────────────────────────────────────────────────
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    extra   = [c for c in df.columns if c not in REQUIRED_COLUMNS]
    if missing:
        warnings.append(f"Missing required columns: `{missing}`")
    if extra:
        warnings.append(f"Unexpected extra columns (ignored in UI): `{extra}`")

    # ── Type coercions ─────────────────────────────────────────────────────────
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")

    for col in ("rf_probability", "if_score", "flow_id"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    if "needs_review" in df.columns:
        df["needs_review"] = (
            df["needs_review"]
            .astype(str)
            .str.lower()
            .map({"true": True, "false": False, "1": True, "0": False})
        )

    # ── Value-spelling checks ──────────────────────────────────────────────────
    for col, allowed in VALID_VALUES.items():
        if col not in df.columns:
            continue
        bad = sorted(set(df[col].dropna().unique()) - allowed)
        if bad:
            warnings.append(
                f"Column **{col}** contains unexpected values: `{bad}`"
            )

    # ── Drop rows with unparseable timestamps ──────────────────────────────────
    if "timestamp" in df.columns:
        n_bad = df["timestamp"].isna().sum()
        if n_bad:
            warnings.append(f"{int(n_bad)} rows dropped due to unparseable timestamps.")
        df = df.dropna(subset=["timestamp"])

    if df.empty:
        return None, warnings + ["No valid rows remain after cleaning."]

    df = df.sort_values("timestamp", ascending=False).reset_index(drop=True)
    return df, warnings


# ── Feature JSON parser ───────────────────────────────────────────────────────
def parse_top_features(raw: str) -> list[dict]:
    """Safely deserialise the top_features JSON column value."""
    try:
        data = json.loads(raw)
        if isinstance(data, list):
            return [d for d in data if isinstance(d, dict)]
    except (json.JSONDecodeError, TypeError, ValueError):
        pass
    return []
