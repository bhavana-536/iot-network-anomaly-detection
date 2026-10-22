"""
test_alerts.py
==============
Phase 3 – Validation test script.

Loads logs/alerts.csv and checks:
  - All required columns are present
  - Every enum column contains only allowed values
  - Verdict / risk / needs_review are consistent with the engine rules
  - Numeric columns are in expected ranges
  - top_features is valid JSON with the right shape

Run:
    python test_alerts.py

Exit code 0 = all checks passed.
Exit code 1 = one or more checks failed.
"""

import json
import sys
from pathlib import Path
from collections import Counter

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

import pandas as pd

# ── Schema constants (mirrors data_loader.py) ─────────────────────────────────
CSV_PATH = "logs/alerts.csv"

REQUIRED_COLUMNS = [
    "timestamp", "flow_id", "src_ip", "dst_ip", "protocol",
    "rf_class", "rf_probability", "if_score", "verdict",
    "risk_level", "needs_review", "top_features",
    "explanation", "recommendation",
]

VALID_VERDICTS    = {"Normal", "Known Attack", "Unknown Anomaly"}
VALID_RISK        = {"Low", "Medium", "High"}
VALID_PROTOCOLS   = {"TCP", "UDP", "ICMP", "MQTT"}
VALID_RF_CLASSES  = {
    "Benign", "DoS_DDoS", "Reconnaissance", "BruteForce_Password",
    "Injection_Web", "MITM_Spoofing", "Botnet_Malware",
}

IF_TAU      = 0.99
IF_TAU_HIGH = 0.999
PROB_HIGH   = 0.80
REVIEW_THR  = 0.60

PASS = "  [PASS]"
FAIL = "  [FAIL]"

failures: list[str] = []


def fail(msg: str) -> None:
    failures.append(msg)
    print(f"{FAIL}  {msg}")


def ok(msg: str) -> None:
    print(f"{PASS}  {msg}")


# ─────────────────────────────────────────────────────────────────────────────
print("=" * 60)
print(f"Testing: {CSV_PATH}")
print("=" * 60)

# ── 1. File exists ─────────────────────────────────────────────────────────────
if not Path(CSV_PATH).exists():
    print(f"{FAIL}  File not found: {CSV_PATH}")
    print("\nRun  python make_fake_alerts.py  first, then re-run this script.")
    sys.exit(1)
ok("File exists")

# ── 2. Load CSV ────────────────────────────────────────────────────────────────
try:
    df = pd.read_csv(CSV_PATH, low_memory=False)
    ok(f"CSV loaded  ({len(df)} rows, {len(df.columns)} columns)")
except Exception as exc:
    fail(f"Failed to parse CSV: {exc}")
    sys.exit(1)

# ── 3. Required columns ────────────────────────────────────────────────────────
missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
if missing:
    fail(f"Missing required columns: {missing}")
else:
    ok(f"All {len(REQUIRED_COLUMNS)} required columns present")

# ── 4. Row count ───────────────────────────────────────────────────────────────
if len(df) == 0:
    fail("No data rows")
elif len(df) != 400:
    # Non-fatal: just warn
    print(f"  ⚠️  WARN  Row count is {len(df)} (expected 400)")
else:
    ok("Row count = 400")

# ── 5. Allowed-value checks ───────────────────────────────────────────────────
for col, allowed, label in [
    ("verdict",    VALID_VERDICTS,  "verdict"),
    ("risk_level", VALID_RISK,      "risk_level"),
    ("protocol",   VALID_PROTOCOLS, "protocol"),
    ("rf_class",   VALID_RF_CLASSES,"rf_class"),
]:
    if col not in df.columns:
        continue
    bad = sorted(set(df[col].dropna().unique()) - allowed)
    if bad:
        fail(f"Column '{col}' has unexpected values: {bad}")
    else:
        ok(f"Column '{col}' values all valid")

# ── 6. Numeric range checks ───────────────────────────────────────────────────
for col, lo, hi in [
    ("rf_probability", 0.0, 1.0),
    ("if_score",       0.0, 1.0),
]:
    if col not in df.columns:
        continue
    s = pd.to_numeric(df[col], errors="coerce")
    out = ((s < lo) | (s > hi)).sum()
    if out:
        fail(f"Column '{col}': {int(out)} values outside [{lo}, {hi}]")
    else:
        ok(f"Column '{col}' in [{lo}, {hi}]")

# ── 7. Timestamp parseable ────────────────────────────────────────────────────
if "timestamp" in df.columns:
    ts   = pd.to_datetime(df["timestamp"], errors="coerce")
    bad  = ts.isna().sum()
    if bad:
        fail(f"timestamp: {int(bad)} unparseable values")
    else:
        ok("timestamp parseable for all rows")

# ── 8. flow_id unique ─────────────────────────────────────────────────────────
if "flow_id" in df.columns:
    dupes = df["flow_id"].duplicated().sum()
    if dupes:
        fail(f"flow_id has {int(dupes)} duplicate values")
    else:
        ok("flow_id: all unique")

# ── 9. Rule consistency ───────────────────────────────────────────────────────
print("\n  Checking hybrid-engine rule consistency ...")

if all(c in df.columns for c in ("rf_class", "rf_probability", "if_score",
                                  "verdict", "risk_level", "needs_review")):
    rf_prob = pd.to_numeric(df["rf_probability"], errors="coerce")
    if_sc   = pd.to_numeric(df["if_score"],       errors="coerce")

    n_verdict_err  = 0
    n_risk_err     = 0
    n_review_err   = 0

    for i, row in df.iterrows():
        rc  = str(row["rf_class"])
        p   = float(rf_prob.iloc[i]) if not pd.isna(rf_prob.iloc[i]) else 0
        s   = float(if_sc.iloc[i])   if not pd.isna(if_sc.iloc[i])   else 0
        v   = str(row["verdict"])
        rl  = str(row["risk_level"])
        nr  = str(row["needs_review"]).lower() in ("true", "1")

        # Compute expected labels
        if rc != "Benign":
            exp_v  = "Known Attack"
            exp_rl = "High" if p >= PROB_HIGH else "Medium"
        elif s >= IF_TAU:
            exp_v  = "Unknown Anomaly"
            exp_rl = "High" if s >= IF_TAU_HIGH else "Medium"
        else:
            exp_v  = "Normal"
            exp_rl = "Low"

        exp_nr = p < REVIEW_THR

        if v != exp_v:
            n_verdict_err += 1
        if rl != exp_rl:
            n_risk_err += 1
        if nr != exp_nr:
            n_review_err += 1

    for n, label in [(n_verdict_err, "verdict"), (n_risk_err, "risk_level"),
                     (n_review_err, "needs_review")]:
        if n:
            fail(f"Rule violation: {n} rows have wrong '{label}'")
        else:
            ok(f"'{label}' consistent with hybrid-engine rules (0 violations)")

# ── 10. top_features JSON shape ───────────────────────────────────────────────
if "top_features" in df.columns:
    bad_tf = 0
    for v in df["top_features"]:
        try:
            data = json.loads(v)
            if not (isinstance(data, list) and len(data) == 5):
                bad_tf += 1
            else:
                for item in data:
                    if not all(k in item for k in ("feature", "value", "contribution")):
                        bad_tf += 1
                        break
        except Exception:
            bad_tf += 1
    if bad_tf:
        fail(f"top_features: {bad_tf} rows with invalid JSON or wrong shape")
    else:
        ok("top_features: valid JSON list of 5 objects in every row")

# ── 11. Distribution stats ────────────────────────────────────────────────────
print("\n  Distribution (informational) ----------------------")
if "verdict" in df.columns:
    for k, v in df["verdict"].value_counts().items():
        print(f"    {k:<22}  {v:>3} ({v/len(df)*100:4.1f} %)")
if "risk_level" in df.columns:
    for k in ["Low", "Medium", "High"]:
        v = (df["risk_level"] == k).sum()
        print(f"    Risk {k:<17}  {v:>3} ({v/len(df)*100:4.1f} %)")


# ── Summary ────────────────────────────────────────────────────────────────────
print()
print("=" * 60)
if failures:
    print(f"RESULT: {len(failures)} CHECK(S) FAILED")
    for f in failures:
        print(f"  - {f}")
    sys.exit(1)
else:
    print("RESULT: ALL CHECKS PASSED [OK]")
    sys.exit(0)
