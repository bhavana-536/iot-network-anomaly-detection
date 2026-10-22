# AI-Based Real-Time Network Anomaly Detection for IoT Datasets

## Project overview
Hybrid ML system combining a **Random Forest** (supervised, multi-class) and an **Isolation Forest** (unsupervised, trained on benign flows only) to detect both known and unknown network attacks in IoT traffic.

## Requirements
- Python 3.11
- 16 GB RAM (no GPU required)
- Dataset files on disk (see Configuration below)

```
pip install -r requirements.txt
```

---

## Configuration
Edit **`config.yaml`** before running anything:

```yaml
dataset_paths:
  CICIoT2023:   "C:\\path\\to\\CICIoT2023.csv"
  TON_IoT:      "C:\\path\\to\\ton_iot\\Processed_Network_dataset"
  Edge-IIoTset: "C:\\path\\to\\Edge-IIoTset"
  BoT-IoT:      "C:\\path\\to\\BoT-IoT.csv"
  MQTTset:      "C:\\path\\to\\MQTTset.csv"
  IoT-23:       "C:\\path\\to\\IoT-23"
```

Set any path to `null` to skip that dataset.

---

## Run order

| Step | Script | Description |
|------|--------|-------------|
| **Phase 1a** | `python inspect_datasets.py` | Inspect raw data; shows label columns & value counts |
| **Phase 1b** | `python prepare_datasets.py` | Clean, deduplicate, encode, split; saves `data/processed/` |
| **Phase 2**  | `python harmonise_datasets.py` | Map to unified labels; build core feature set |
| **Phase 3a** | `python train_hybrid.py` | Train RF + IF on CICIoT2023; save artifacts |
| **Phase 3b** | `python evaluate.py` | Evaluate on test split; save reports + figures |
| **Phase 4**  | `python experiments.py` | E1–E6 experiments; save results_summary.md |

---

## Project structure

```
iot-network-anomaly-detection/
├── config.yaml                  # All paths and hyperparams
├── requirements.txt
├── README.md
├── inspect_datasets.py          # Phase 1a
├── prepare_datasets.py          # Phase 1b
├── harmonise_datasets.py        # Phase 2  (created in Phase 2)
├── train_hybrid.py              # Phase 3a (created in Phase 3)
├── evaluate.py                  # Phase 3b (created in Phase 3)
├── experiments.py               # Phase 4  (created in Phase 4)
├── src/
│   └── hybrid_engine.py         # Hybrid predict() logic
├── artifacts/
│   ├── rf_model.joblib
│   ├── if_model.joblib
│   ├── scaler.joblib
│   ├── imputer.joblib
│   ├── label_encoder.joblib
│   ├── thresholds.json
│   ├── feature_list.json
│   ├── label_encodings.json
│   └── core_features.json
├── data/
│   └── processed/
│       ├── CICIoT2023/
│       │   ├── train.csv
│       │   ├── validation.csv
│       │   └── test.csv
│       └── ...
└── reports/
    ├── inspect_report.json
    ├── dataset_statistics.csv
    ├── evaluation_metrics.csv
    ├── results_summary.md
    └── figures/
```

---

## Decision engine (hybrid rules)
```
RF predicts attack class
  → verdict = "Known Attack"
  → risk = High if p >= 0.8 or IF also flags
  → risk = Medium if p < 0.8 and IF does not flag
Else IF anomaly score s >= tau (99th pct of benign val scores)
  → verdict = "Unknown Anomaly"
  → risk = High if s >= tau_high (99.9th pct)
  → risk = Medium if tau <= s < tau_high
Else
  → verdict = "Normal"  |  risk = Low

Any row where top-class RF probability p < 0.6
  → flag needs_review = True
```

---

## Reproducibility
All scripts use `random_seed: 42` from `config.yaml`. Pass `--seed N` to override for ablation runs.
