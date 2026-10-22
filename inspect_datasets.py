"""
inspect_datasets.py
====================
Phase 1, Step 1 – Dataset inspection.

For every dataset configured in config.yaml this script:
  1. Reads up to INSPECT_ROWS rows (across all CSV files in the path).
  2. Prints column names, dtypes, row count.
  3. For EVERY candidate label column it prints unique values + counts.
  4. Writes a compact JSON report to  reports/inspect_report.json

Run:
    python inspect_datasets.py

No changes are made to any data.
"""

import os
import glob
import json
import re
import sys

import numpy as np
import pandas as pd
import yaml

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
CONFIG_PATH = "config.yaml"
INSPECT_ROWS = 200_000          # max rows to read per dataset for inspection
CHUNK_SIZE   = 50_000
REPORT_DIR   = "reports"

# Columns that MAY be label columns; we print value-counts for ALL of them.
# This is wider than what the current prepare_datasets.py guesses.
LABEL_HINTS = [
    "label", "attack", "type", "category", "subcategory",
    "attack_cat", "class", "attack_type", "detailed-label",
    "detailed_label",
]


def load_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def normalize_name(name: str) -> str:
    """Lower-case, strip, collapse non-alphanumeric to underscore."""
    name = str(name).strip().lower()
    name = re.sub(r"[^a-z0-9]+", "_", name)
    return name.strip("_")


def get_csv_files(path: str) -> list[str]:
    """Return list of CSV files for a given path (file or directory)."""
    if not path:
        return []
    if os.path.isfile(path) and path.lower().endswith(".csv"):
        return [path]
    if os.path.isdir(path):
        return sorted(
            glob.glob(os.path.join(path, "**", "*.csv"), recursive=True)
        )
    return []


def read_sample(csv_files: list[str], max_rows: int) -> pd.DataFrame | None:
    """Read up to max_rows rows across csv_files using chunked reading."""
    frames = []
    rows_read = 0
    for fpath in csv_files:
        if rows_read >= max_rows:
            break
        remaining = max_rows - rows_read
        try:
            for chunk in pd.read_csv(
                fpath,
                sep=None,
                engine="python",
                chunksize=CHUNK_SIZE,
                on_bad_lines="skip",
                low_memory=False,
            ):
                # Normalise column names
                chunk.columns = [normalize_name(c) for c in chunk.columns]
                chunk = chunk.loc[:, ~chunk.columns.duplicated()]
                frames.append(chunk.head(remaining))
                rows_read += len(chunk)
                if rows_read >= max_rows:
                    break
        except Exception as exc:
            print(f"    [WARN] Could not read {fpath}: {exc}")
    if not frames:
        return None
    return pd.concat(frames, ignore_index=True)


def inspect_dataset(dataset_name: str, path: str) -> dict:
    """Inspect one dataset; return a dict with findings."""
    print(f"\n{'='*70}")
    print(f"  DATASET: {dataset_name}")
    print(f"  PATH   : {path}")
    print(f"{'='*70}")

    result = {
        "dataset": dataset_name,
        "path": path,
        "status": "ok",
        "csv_files": [],
        "rows_sampled": 0,
        "columns": [],
        "label_candidates": {},
    }

    csv_files = get_csv_files(path)
    if not csv_files:
        msg = "PATH NOT FOUND or no CSV files"
        print(f"  [SKIP] {msg}")
        result["status"] = msg
        return result

    result["csv_files"] = csv_files
    print(f"  CSV files found: {len(csv_files)}")
    for f in csv_files[:6]:
        print(f"    {f}")
    if len(csv_files) > 6:
        print(f"    ... and {len(csv_files)-6} more")

    df = read_sample(csv_files, INSPECT_ROWS)
    if df is None or df.empty:
        msg = "Could not read any rows"
        print(f"  [SKIP] {msg}")
        result["status"] = msg
        return result

    result["rows_sampled"] = len(df)
    print(f"\n  Rows sampled : {len(df):,}")
    print(f"  Columns ({len(df.columns)}): ")

    col_info = []
    for col in df.columns:
        dtype = str(df[col].dtype)
        n_unique = df[col].nunique(dropna=False)
        col_info.append({"col": col, "dtype": dtype, "n_unique": n_unique})
        print(f"    {col:<40}  dtype={dtype:<10}  unique={n_unique}")
    result["columns"] = col_info

    # ---- Label candidate inspection ----------------------------------------
    print("\n  --- Label / target column candidates ---")
    for hint in LABEL_HINTS:
        if hint not in df.columns:
            continue
        vc = df[hint].value_counts(dropna=False)
        result["label_candidates"][hint] = vc.to_dict()
        print(f"\n  Column: '{hint}'  ({df[hint].dtype})  {len(vc)} unique values")
        for val, cnt in vc.head(30).items():
            pct = cnt / len(df) * 100
            print(f"    {str(val):<45}  {cnt:>8,}  ({pct:5.1f}%)")
        if len(vc) > 30:
            print(f"    ... and {len(vc)-30} more values")

    if not result["label_candidates"]:
        print("  [WARN] No label-hint columns found.  Full column list above.")

    return result


def main():
    os.makedirs(REPORT_DIR, exist_ok=True)

    if not os.path.exists(CONFIG_PATH):
        sys.exit(f"ERROR: {CONFIG_PATH} not found. Run from project root.")

    cfg = load_config(CONFIG_PATH)
    dataset_paths: dict = cfg.get("dataset_paths", {})

    all_results = []
    for name, path in dataset_paths.items():
        if not path:
            print(f"\n[SKIP] {name}: no path configured.")
            continue
        result = inspect_dataset(name, path)
        all_results.append(result)

    # Save JSON report (convert int64 keys to str for JSON serialisation)
    report_path = os.path.join(REPORT_DIR, "inspect_report.json")

    def default_serialiser(obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        return str(obj)

    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, default=default_serialiser)

    print(f"\n{'='*70}")
    print(f"Inspection complete.  Report saved to: {report_path}")
    print("NEXT STEP: Review the label columns above, then run prepare_datasets.py")


if __name__ == "__main__":
    main()
