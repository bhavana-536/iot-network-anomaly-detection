"""
prepare_datasets.py
===================
Phase 1, Step 2 – Clean, deduplicate, encode, cap, and split all datasets.

Key fixes over the original version
-------------------------------------
1.  Config-driven paths  (config.yaml, no hard-coded user paths).
2.  Per-dataset label column dictionary – no "first-match" guessing.
3.  Categorical columns (proto, service, conn_state …) are label-encoded
    instead of dropped; mappings saved to artifacts/label_encodings.json.
4.  Identifier columns (IP, MAC, timestamps, flow IDs) are dropped.
5.  Deduplication via MD5 row-hashes tracked across chunks.
6.  Row cap: keep ALL benign rows (up to cap) + class-balanced attack sample.
7.  NaNs are NOT imputed here; they are left for train-time imputation.
8.  70/15/15 stratified split done AFTER full dataset is loaded (not per chunk).
9.  Adds unified_label column (filled in Phase 2).
10. Writes reports/dataset_statistics.csv.

Run:
    python prepare_datasets.py
"""

import glob
import hashlib
import json
import logging
import os
import re
import sys

import numpy as np
import pandas as pd
import yaml
from sklearn.model_selection import train_test_split

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Load configuration
# ---------------------------------------------------------------------------
CONFIG_PATH = "config.yaml"


def load_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


cfg = load_config(CONFIG_PATH)

RANDOM_SEED: int            = int(cfg["random_seed"])
CHUNK_SIZE: int             = int(cfg["chunk_size"])
MAX_ROWS: int               = int(cfg.get("max_rows_per_dataset", 500_000))
SPLIT_RATIOS: dict          = cfg["split_ratios"]
DATASET_PATHS: dict         = cfg["dataset_paths"]
DATASET_LABELS: dict        = cfg["dataset_labels"]
IDENTIFIER_COLS: set        = set(cfg.get("identifier_cols", []))
CATEGORICAL_COLS: list      = cfg.get("categorical_cols", [])
OUTPUT_DIR: str             = cfg["output_dir"]
ARTIFACTS_DIR: str          = cfg["artifacts_dir"]
REPORTS_DIR: str            = cfg["reports_dir"]

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(ARTIFACTS_DIR, exist_ok=True)
os.makedirs(REPORTS_DIR, exist_ok=True)

np.random.seed(RANDOM_SEED)

# Global encoding map  {dataset -> {col -> {raw_value -> int_code}}}
LABEL_ENCODINGS: dict = {}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def normalize_name(name: str) -> str:
    """Lower-case and replace non-alphanumeric runs with underscore."""
    name = str(name).strip().lower()
    name = re.sub(r"[^a-z0-9]+", "_", name)
    return name.strip("_")


def get_csv_files(path: str) -> list[str]:
    if not path:
        return []
    if os.path.isfile(path) and path.lower().endswith(".csv"):
        return [path]
    if os.path.isdir(path):
        return sorted(glob.glob(os.path.join(path, "**", "*.csv"), recursive=True))
    return []


def row_hash(df: pd.DataFrame, feature_cols: list[str]) -> pd.Series:
    """Return an MD5 hex-string per row (used for cross-chunk deduplication)."""
    # Stringify only the feature columns to keep it fast
    return df[feature_cols].astype(str).apply(
        lambda r: hashlib.md5("|".join(r).encode()).hexdigest(), axis=1
    )


def encode_categoricals(
    df: pd.DataFrame,
    cat_cols: list[str],
    dataset_name: str,
    existing_map: dict,
) -> tuple[pd.DataFrame, dict]:
    """
    Label-encode categorical columns.
    existing_map is updated IN PLACE and returned; new codes extend old ones
    so that code = 0 is always <unknown/unseen>.
    """
    enc_map = existing_map.get(dataset_name, {})
    for col in cat_cols:
        if col not in df.columns:
            continue
        col_map = enc_map.get(col, {})
        # Assign codes for values not yet seen
        unique_vals = df[col].dropna().unique().tolist()
        for v in unique_vals:
            key = str(v)
            if key not in col_map:
                col_map[key] = len(col_map) + 1   # 0 reserved for NaN/unknown
        enc_map[col] = col_map
        df[col] = df[col].map(lambda v, m=col_map: m.get(str(v), 0))
    existing_map[dataset_name] = enc_map
    return df, existing_map


def balanced_sample(
    df: pd.DataFrame,
    label_col: str,
    max_rows: int,
    benign_label: str,
    rng: np.random.Generator,
) -> pd.DataFrame:
    """
    From df, keep:
      - ALL benign rows (up to max_rows)
      - A class-balanced sample of attack rows filling the remaining budget.
    """
    benign_mask = df[label_col].str.lower().str.strip().isin(
        {"benign", "normal", "0", "background"}
    )
    benign_df = df[benign_mask].copy()
    attack_df = df[~benign_mask].copy()

    # Cap benign
    n_benign = min(len(benign_df), max_rows)
    if n_benign < len(benign_df):
        idx = rng.choice(len(benign_df), n_benign, replace=False)
        benign_df = benign_df.iloc[sorted(idx)]

    budget_attack = max(0, max_rows - n_benign)

    if budget_attack == 0 or attack_df.empty:
        return benign_df

    # Per-class equal quota
    classes = attack_df[label_col].unique()
    per_class = max(1, budget_attack // len(classes))
    parts = []
    for cls in classes:
        sub = attack_df[attack_df[label_col] == cls]
        n_take = min(len(sub), per_class)
        idx = rng.choice(len(sub), n_take, replace=False)
        parts.append(sub.iloc[sorted(idx)])

    # If total < budget, top up proportionally
    attack_sample = pd.concat(parts, ignore_index=True)
    shortfall = budget_attack - len(attack_sample)
    if shortfall > 0 and len(attack_df) > len(attack_sample):
        remaining = attack_df.drop(index=attack_sample.index, errors="ignore")
        extra_n = min(shortfall, len(remaining))
        idx = rng.choice(len(remaining), extra_n, replace=False)
        attack_sample = pd.concat(
            [attack_sample, remaining.iloc[sorted(idx)]], ignore_index=True
        )

    return pd.concat([benign_df, attack_sample], ignore_index=True)


# ---------------------------------------------------------------------------
# Per-dataset processing
# ---------------------------------------------------------------------------

def process_dataset(dataset_name: str) -> dict | None:
    """
    Load, clean, deduplicate, encode, cap and split one dataset.
    Returns a statistics dict or None on failure.
    """
    path = DATASET_PATHS.get(dataset_name, "")
    if not path:
        log.warning("%s: no path in config.yaml – skipped", dataset_name)
        return None

    csv_files = get_csv_files(path)
    if not csv_files:
        log.warning("%s: no CSV files found at %s – skipped", dataset_name, path)
        return None

    label_cfg = DATASET_LABELS.get(dataset_name, {})
    attack_col_raw: str | None = label_cfg.get("attack_col")
    binary_col_raw: str | None = label_cfg.get("binary_col")

    if not attack_col_raw:
        log.warning("%s: no attack_col in dataset_labels config – skipped", dataset_name)
        return None

    # Normalise the configured column names to match our normalise_name()
    attack_col = normalize_name(attack_col_raw)
    binary_col = normalize_name(binary_col_raw) if binary_col_raw else None

    log.info("="*60)
    log.info("Dataset: %s  (%d file(s))", dataset_name, len(csv_files))
    log.info("  attack_col=%s  binary_col=%s", attack_col, binary_col)

    rng = np.random.default_rng(RANDOM_SEED)
    seen_hashes: set[str] = set()
    all_chunks: list[pd.DataFrame] = []
    total_rows_read = 0
    feature_cols: list[str] | None = None
    cat_cols_present: list[str] = []
    identifier_cols_found: list[str] = []

    for fpath in csv_files:
        log.info("  Reading: %s", os.path.basename(fpath))
        try:
            reader = pd.read_csv(
                fpath,
                sep=None,
                engine="python",
                chunksize=CHUNK_SIZE,
                on_bad_lines="skip",
                low_memory=False,
            )
            for chunk in reader:
                chunk.columns = [normalize_name(c) for c in chunk.columns]
                chunk = chunk.loc[:, ~chunk.columns.duplicated()]
                total_rows_read += len(chunk)

                # ----- Validate label column existence ----------------------
                if attack_col not in chunk.columns:
                    # Try to find a fuzzy match
                    candidates = [c for c in chunk.columns if attack_col in c]
                    if candidates:
                        log.warning(
                            "  attack_col '%s' not found; using '%s'",
                            attack_col, candidates[0],
                        )
                        attack_col = candidates[0]
                    else:
                        log.error(
                            "  attack_col '%s' not in columns: %s",
                            attack_col, list(chunk.columns),
                        )
                        return None

                # ----- First chunk: detect column roles --------------------
                if feature_cols is None:
                    # Columns to drop: identifiers
                    id_norm = {normalize_name(c) for c in IDENTIFIER_COLS}
                    identifier_cols_found = [
                        c for c in chunk.columns
                        if c in id_norm
                        or c.endswith("_ip")
                        or c.endswith("_mac")
                    ]
                    # binary_col is dropped (not a feature)
                    extra_drop = []
                    if binary_col and binary_col in chunk.columns:
                        extra_drop.append(binary_col)

                    drop_set = set(identifier_cols_found) | set(extra_drop) | {attack_col}

                    # Categorical columns present in this dataset
                    cat_norm = {normalize_name(c) for c in CATEGORICAL_COLS}
                    cat_cols_present = [
                        c for c in chunk.columns
                        if c in cat_norm and c not in drop_set
                    ]

                    feature_cols = [
                        c for c in chunk.columns if c not in drop_set
                    ]

                    log.info(
                        "  Detected %d feature columns; %d categorical; "
                        "%d identifier cols dropped",
                        len(feature_cols), len(cat_cols_present),
                        len(identifier_cols_found),
                    )

                # ----- Align chunk to known schema --------------------------
                chunk = chunk.drop(
                    columns=[
                        c for c in (identifier_cols_found + ([binary_col] if binary_col else []))
                        if c in chunk.columns
                    ],
                    errors="ignore",
                )
                for fc in feature_cols:
                    if fc not in chunk.columns:
                        chunk[fc] = np.nan

                working_cols = feature_cols + [attack_col]
                chunk = chunk[[c for c in working_cols if c in chunk.columns]].copy()

                # ----- Drop rows missing the label --------------------------
                chunk = chunk.dropna(subset=[attack_col])
                if chunk.empty:
                    continue

                # ----- Cross-chunk deduplication (hash on feature cols) -----
                hash_cols = [c for c in feature_cols if c in chunk.columns][:20]
                if hash_cols:
                    hashes = row_hash(chunk, hash_cols)
                    before = len(chunk)
                    chunk = chunk[~hashes.isin(seen_hashes)]
                    seen_hashes.update(hashes[~hashes.isin(seen_hashes)].tolist())
                    dupes = before - len(chunk)
                    if dupes:
                        log.debug("  Chunk: dropped %d duplicates", dupes)

                if chunk.empty:
                    continue

                # ----- Encode categorical columns ---------------------------
                chunk, _ = encode_categoricals(
                    chunk, cat_cols_present, dataset_name, LABEL_ENCODINGS
                )

                # ----- Convert feature cols to numeric ----------------------
                for col in feature_cols:
                    if col in chunk.columns and col not in cat_cols_present:
                        chunk[col] = pd.to_numeric(chunk[col], errors="coerce")

                # ----- Replace inf ------------------------------------------
                num_cols = [c for c in feature_cols if c in chunk.columns]
                chunk[num_cols] = chunk[num_cols].replace(
                    [np.inf, -np.inf], np.nan
                )

                all_chunks.append(chunk)

                # Early exit if we already have >> MAX_ROWS raw rows
                rows_so_far = sum(len(c) for c in all_chunks)
                if MAX_ROWS and rows_so_far > MAX_ROWS * 3:
                    log.info("  Early-stop reading: already have %d raw rows", rows_so_far)
                    break

        except Exception as exc:
            log.error("  ERROR reading %s: %s", fpath, exc)

    if not all_chunks:
        log.error("%s: no usable data after reading – skipped", dataset_name)
        return None

    df = pd.concat(all_chunks, ignore_index=True)
    rows_after_dedup = len(df)
    log.info("  Total rows read: %d | After dedup: %d", total_rows_read, rows_after_dedup)

    # ----- Row cap with class-balanced sampling ----------------------------
    if MAX_ROWS and len(df) > MAX_ROWS:
        df = balanced_sample(df, attack_col, MAX_ROWS, "benign", rng)
        log.info("  After capping to %d: %d rows", MAX_ROWS, len(df))

    # ----- Add unified_label placeholder (filled in Phase 2) ---------------
    df["unified_label"] = "__unmapped__"

    # ----- Stratified 70/15/15 split ----------------------------------------
    train_ratio = SPLIT_RATIOS["train"]
    val_ratio   = SPLIT_RATIOS["validation"]
    # test_ratio  = SPLIT_RATIOS["test"]          # implicitly = remaining

    try:
        train_df, temp_df = train_test_split(
            df,
            test_size=(1.0 - train_ratio),
            random_state=RANDOM_SEED,
            stratify=df[attack_col],
        )
        relative_val = val_ratio / (1.0 - train_ratio)
        val_df, test_df = train_test_split(
            temp_df,
            test_size=(1.0 - relative_val),
            random_state=RANDOM_SEED,
            stratify=temp_df[attack_col],
        )
    except ValueError as exc:
        log.warning("  Stratified split failed (%s); using random split", exc)
        n = len(df)
        idx = rng.permutation(n)
        t1 = int(n * train_ratio)
        t2 = t1 + int(n * val_ratio)
        train_df = df.iloc[idx[:t1]]
        val_df   = df.iloc[idx[t1:t2]]
        test_df  = df.iloc[idx[t2:]]

    log.info(
        "  Split – train: %d | val: %d | test: %d",
        len(train_df), len(val_df), len(test_df),
    )

    # ----- Save splits -------------------------------------------------------
    out_dir = os.path.join(OUTPUT_DIR, dataset_name)
    os.makedirs(out_dir, exist_ok=True)
    for split_name, split_df in [
        ("train", train_df), ("validation", val_df), ("test", test_df)
    ]:
        out_path = os.path.join(out_dir, f"{split_name}.csv")
        split_df.to_csv(out_path, index=False)
        log.info("  Saved %s -> %s", split_name, out_path)

    # ----- Class counts per split --------------------------------------------
    class_counts: dict = {}
    for sname, sdf in [("train", train_df), ("validation", val_df), ("test", test_df)]:
        class_counts[sname] = sdf[attack_col].value_counts().to_dict()

    return {
        "dataset": dataset_name,
        "csv_files": len(csv_files),
        "rows_read": total_rows_read,
        "rows_after_dedup": rows_after_dedup,
        "rows_kept": len(df),
        "rows_skipped": total_rows_read - rows_after_dedup,
        "features": len(feature_cols) if feature_cols else 0,
        "attack_col": attack_col,
        "train_rows": len(train_df),
        "val_rows": len(val_df),
        "test_rows": len(test_df),
        "class_counts": class_counts,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    statistics: list[dict] = []

    for dataset_name in DATASET_PATHS:
        stats = process_dataset(dataset_name)
        if stats:
            statistics.append(stats)

    # ----- Save label encodings ----------------------------------------------
    enc_path = os.path.join(ARTIFACTS_DIR, "label_encodings.json")
    with open(enc_path, "w", encoding="utf-8") as f:
        json.dump(LABEL_ENCODINGS, f, indent=2)
    log.info("Label encodings saved to %s", enc_path)

    # ----- Save statistics CSV -----------------------------------------------
    if statistics:
        rows = []
        for s in statistics:
            base = {k: v for k, v in s.items() if k != "class_counts"}
            for split, counts in s.get("class_counts", {}).items():
                for cls, cnt in counts.items():
                    rows.append({**base, "split": split, "class": cls, "count": cnt})
        stats_df = pd.DataFrame(rows)
        csv_path = os.path.join(REPORTS_DIR, "dataset_statistics.csv")
        stats_df.to_csv(csv_path, index=False)
        log.info("Statistics saved to %s", csv_path)

        # Console summary
        print("\n" + "="*60)
        print("DATASET PROCESSING SUMMARY")
        print("="*60)
        summary_cols = [
            "dataset", "rows_read", "rows_after_dedup",
            "rows_kept", "rows_skipped", "features",
            "train_rows", "val_rows", "test_rows",
        ]
        summary_df = pd.DataFrame(statistics)[summary_cols]
        print(summary_df.to_string(index=False))
        print(f"\nFull statistics: {csv_path}")
    else:
        log.error("No datasets were processed. Check config.yaml paths.")

    log.info("prepare_datasets.py complete.")


if __name__ == "__main__":
    main()
