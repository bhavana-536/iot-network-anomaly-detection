
import os
import re
import glob
import numpy as np
import pandas as pd

# Original dataset locations
DATASETS = {
    "TON_IoT": r"C:\Users\bhava\Downloads\ton_iot\Processed_Network_dataset",
    "CICIoT2023": r"C:\Users\bhava\Downloads\CICIoT2023.csv",
    "BoT-IoT": r"C:\Users\bhava\Downloads\BoT-IoT.csv",
    "Edge-IIoTset": r"C:\Users\bhava\Downloads\Edge-IIoTset",
    "MQTTset": r"C:\Users\bhava\Downloads\MQTTset.csv",
}

OUTPUT_DIR = r"C:\Iot_project\processed_datasets"
CHUNK_SIZE = 50_000
RANDOM_SEED = 42

# Identifiers that should not be used as predictive features
IDENTIFIER_NAMES = {
    "src_ip", "dst_ip", "saddr", "daddr",
    "source_ip", "destination_ip",
    "src_mac", "dst_mac", "smac", "dmac",
    "timestamp", "ts", "stime", "ltime",
    "flow_id", "id", "pkseqid"
}

LABEL_CANDIDATES = [
    "label", "attack", "type", "category", "subcategory"
]

os.makedirs(OUTPUT_DIR, exist_ok=True)


def get_csv_files(path):
    if os.path.isfile(path) and path.lower().endswith(".csv"):
        return [path]

    if os.path.isdir(path):
        return sorted(
            glob.glob(os.path.join(path, "**", "*.csv"),
                      recursive=True)
        )

    return []


def normalize_name(name):
    name = str(name).strip().lower()
    name = re.sub(r"[^a-z0-9]+", "_", name)
    return name.strip("_")


def read_chunks(path):
    # Detect comma, semicolon, tab, or similar delimiters
    return pd.read_csv(
        path,
        sep=None,
        engine="python",
        chunksize=CHUNK_SIZE,
        on_bad_lines="error"
    )


def process_file(dataset, path):
    filename = os.path.splitext(os.path.basename(path))[0]
    output_folder = os.path.join(OUTPUT_DIR, dataset, filename)
    os.makedirs(output_folder, exist_ok=True)

    output_paths = {
        split: os.path.join(output_folder, f"{split}.csv")
        for split in ("train", "validation", "test")
    }

    # Start fresh if this file is processed again
    for output_path in output_paths.values():
        if os.path.exists(output_path):
            os.remove(output_path)

    seen_header = {split: False for split in output_paths}
    total_rows = 0
    kept_rows = 0
    target = None
    feature_columns = None

    print(f"\nProcessing: {dataset}/{os.path.basename(path)}")

    try:
        for chunk in read_chunks(path):
            chunk.columns = [
                normalize_name(col) for col in chunk.columns
            ]

            # Remove duplicate column names
            chunk = chunk.loc[:, ~chunk.columns.duplicated()]
            total_rows += len(chunk)

            if target is None:
                target = next(
                    (col for col in LABEL_CANDIDATES
                     if col in chunk.columns),
                    None
                )

                if target is None:
                    print("  SKIPPED: no recognized label column.")
                    return

                # Remove identifiers and non-predictive columns
                drop_columns = [
                    col for col in chunk.columns
                    if (
                        col in IDENTIFIER_NAMES
                        or col.endswith("_ip")
                        or col.endswith("_mac")
                        or col in {"attack_cat", "subcategory"}
                        and col != target
                    )
                ]

                chunk = chunk.drop(
                    columns=drop_columns, errors="ignore"
                )

                # Keep numeric features and the original target
                feature_columns = [
                    col for col in chunk.columns
                    if col != target
                    and pd.api.types.is_numeric_dtype(chunk[col])
                ]

                # Some numeric-looking columns may be stored as text.
                for col in chunk.columns:
                    if col == target or col in feature_columns:
                        continue

                    converted = pd.to_numeric(
                        chunk[col], errors="coerce"
                    )

                    if converted.notna().sum() > 0:
                        chunk[col] = converted
                        feature_columns.append(col)

                feature_columns = list(dict.fromkeys(feature_columns))

                if not feature_columns:
                    print("  SKIPPED: no usable numeric features.")
                    return

                print(f"  Target label: {target}")
                print(f"  Numeric features: {len(feature_columns)}")

            # Ensure every chunk has the same feature schema
            if target not in chunk.columns:
                print("  SKIPPED: label column missing in a chunk.")
                return

            for col in feature_columns:
                if col not in chunk.columns:
                    chunk[col] = np.nan

            data = chunk[feature_columns + [target]].copy()

            # Convert feature values to numeric
            for col in feature_columns:
                data[col] = pd.to_numeric(
                    data[col], errors="coerce"
                )

            data[feature_columns] = data[
                feature_columns
            ].replace([np.inf, -np.inf], np.nan)

            # Remove rows without a target or any usable features
            data = data.dropna(subset=[target])
            data = data.dropna(
                subset=feature_columns, how="all"
            )

            if data.empty:
                continue

            # Fill missing numeric values using this chunk's medians.
            # This is a preliminary cleaning step, not final
            # train-only imputation.
            medians = data[feature_columns].median()
            data[feature_columns] = data[
                feature_columns
            ].fillna(medians).fillna(0)

            # Deterministic random row split: 70/15/15
            rng = np.random.default_rng(
                RANDOM_SEED + total_rows
            )
            random_values = rng.random(len(data))

            masks = {
                "train": random_values < 0.70,
                "validation": (
                    (random_values >= 0.70)
                    & (random_values < 0.85)
                ),
                "test": random_values >= 0.85,
            }

            for split, mask in masks.items():
                part = data.loc[mask]

                if part.empty:
                    continue

                part.to_csv(
                    output_paths[split],
                    mode="a",
                    header=not seen_header[split],
                    index=False
                )
                seen_header[split] = True

            kept_rows += len(data)

        print(f"  Rows read: {total_rows}")
        print(f"  Rows retained: {kept_rows}")
        print(f"  Output: {output_folder}")

    except Exception as error:
        print(f"  ERROR: {error}")


for dataset, path in DATASETS.items():
    files = get_csv_files(path)

    if not files:
        print(f"\n{dataset}: PATH NOT FOUND - {path}")
        continue

    print(f"\n{dataset}: {len(files)} CSV file(s) found")

    for file_path in files:
        process_file(dataset, file_path)

print("\nPreliminary preprocessing finished.")
print("Output directory:", OUTPUT_DIR)
print("Review skipped files and errors before model training.")
