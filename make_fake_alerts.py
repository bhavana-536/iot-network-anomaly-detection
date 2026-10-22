"""
make_fake_alerts.py
===================
Phase 1 – Fake alert data generator.

Writes 400 rows to:
    logs/alerts.csv   (primary format the dashboard reads)
    logs/alerts.jsonl (one JSON object per line)

All verdict / risk_level / needs_review values are derived from the same
rules used by the hybrid engine – the data is always internally consistent.

Distribution target: ~70 % Normal, ~20 % Known Attack, ~10 % Unknown Anomaly.

Run:
    python make_fake_alerts.py
"""

import csv
import json
import os
import random
from datetime import datetime, timedelta

import numpy as np

# ── Seed ──────────────────────────────────────────────────────────────────────
SEED = 42
random.seed(SEED)
rng  = np.random.default_rng(SEED)

# ── Configuration ─────────────────────────────────────────────────────────────
N_ROWS    = 400
N_NORMAL  = int(0.70 * N_ROWS)            # 280
N_ATTACK  = int(0.20 * N_ROWS)            # 80
N_ANOMALY = N_ROWS - N_NORMAL - N_ATTACK  # 40

ATTACK_CLASSES = [
    "DoS_DDoS", "Reconnaissance", "BruteForce_Password",
    "Injection_Web", "MITM_Spoofing", "Botnet_Malware",
]
PROTOCOLS        = ["TCP", "UDP", "ICMP", "MQTT"]
PROTOCOL_WEIGHTS = [0.45, 0.30, 0.10, 0.15]

SRC_PREFIXES = ["192.168.1", "192.168.2", "10.0.0", "10.0.1", "172.16.0"]
DST_PREFIXES = ["192.168.1", "10.0.0",   "10.10.0", "172.16.1"]

# ── Hybrid engine rules (single source of truth) ──────────────────────────────
IF_TAU       = 0.99    # 99th percentile of benign val scores
IF_TAU_HIGH  = 0.999   # 99.9th percentile
PROB_HIGH    = 0.80    # RF probability boundary for High vs Medium
REVIEW_THR   = 0.60    # needs_review when RF top-class prob < this


def compute_labels(rf_class: str, rf_prob: float, if_score: float):
    """Return (verdict, risk_level, needs_review) from the engine rules."""
    if rf_class != "Benign":
        verdict    = "Known Attack"
        risk_level = "High" if rf_prob >= PROB_HIGH else "Medium"
    elif if_score >= IF_TAU:
        verdict    = "Unknown Anomaly"
        risk_level = "High" if if_score >= IF_TAU_HIGH else "Medium"
    else:
        verdict    = "Normal"
        risk_level = "Low"

    needs_review = bool(rf_prob < REVIEW_THR)
    return verdict, risk_level, needs_review


# ── Feature value generators (returns list of 5 dicts) ───────────────────────
def _f(name: str, value: float, contrib: float) -> dict:
    return {
        "feature":      name,
        "value":        round(float(value), 4),
        "contribution": round(float(contrib), 4),
    }


FEATURE_TEMPLATES: dict = {
    "Benign": lambda: [
        _f("flow_duration",    rng.uniform(0.5, 60),    rng.uniform(-0.25, -0.04)),
        _f("packet_rate",      rng.uniform(1, 50),      rng.uniform(-0.18, -0.02)),
        _f("bytes_per_second", rng.uniform(100, 5000),  rng.uniform(-0.18, -0.01)),
        _f("syn_count",        rng.integers(1, 5),      rng.uniform(-0.10, -0.01)),
        _f("iat_mean",         rng.uniform(0.01, 0.5),  rng.uniform(-0.12, -0.01)),
    ],
    "DoS_DDoS": lambda: [
        _f("packet_rate",      rng.uniform(1000, 5000), rng.uniform(0.30, 0.50)),
        _f("syn_count",        rng.integers(500, 2000), rng.uniform(0.20, 0.40)),
        _f("bytes_per_second", rng.uniform(1e5, 5e5),   rng.uniform(0.15, 0.30)),
        _f("iat_mean",         rng.uniform(1e-4, 1e-3), rng.uniform(0.10, 0.25)),
        _f("flow_duration",    rng.uniform(0.1, 3),     rng.uniform(0.05, 0.18)),
    ],
    "Reconnaissance": lambda: [
        _f("flow_duration",    rng.uniform(30, 300),    rng.uniform(0.20, 0.35)),
        _f("syn_count",        rng.integers(50, 500),   rng.uniform(0.20, 0.40)),
        _f("packet_rate",      rng.uniform(1, 20),      rng.uniform(0.10, 0.25)),
        _f("bytes_per_second", rng.uniform(100, 2000),  rng.uniform(0.08, 0.20)),
        _f("rst_count",        rng.integers(20, 200),   rng.uniform(0.10, 0.25)),
    ],
    "BruteForce_Password": lambda: [
        _f("syn_count",           rng.integers(200, 1000), rng.uniform(0.25, 0.45)),
        _f("rst_count",           rng.integers(150, 900),  rng.uniform(0.20, 0.40)),
        _f("payload_size_mean",   rng.uniform(20, 80),     rng.uniform(0.15, 0.30)),
        _f("packet_rate",         rng.uniform(50, 300),    rng.uniform(0.10, 0.25)),
        _f("flow_duration",       rng.uniform(5, 60),      rng.uniform(0.05, 0.15)),
    ],
    "Injection_Web": lambda: [
        _f("payload_size_mean", rng.uniform(500, 5000),  rng.uniform(0.25, 0.45)),
        _f("bytes_per_second",  rng.uniform(2000, 2e4),  rng.uniform(0.20, 0.35)),
        _f("psh_count",         rng.integers(10, 100),   rng.uniform(0.15, 0.30)),
        _f("ack_count",         rng.integers(50, 500),   rng.uniform(0.10, 0.25)),
        _f("flow_duration",     rng.uniform(1, 30),      rng.uniform(0.08, 0.20)),
    ],
    "MITM_Spoofing": lambda: [
        _f("iat_std",          rng.uniform(0.001, 0.01), rng.uniform(0.30, 0.50)),
        _f("flow_duration",    rng.uniform(60, 600),     rng.uniform(0.15, 0.30)),
        _f("fin_count",        rng.integers(10, 100),    rng.uniform(0.12, 0.25)),
        _f("bytes_per_second", rng.uniform(500, 5000),   rng.uniform(0.08, 0.18)),
        _f("packet_rate",      rng.uniform(5, 50),       rng.uniform(0.06, 0.15)),
    ],
    "Botnet_Malware": lambda: [
        _f("flow_duration",    rng.uniform(300, 3600),   rng.uniform(0.20, 0.35)),
        _f("iat_mean",         rng.uniform(0.001, 0.01), rng.uniform(0.15, 0.30)),
        _f("bytes_per_second", rng.uniform(100, 1000),   rng.uniform(0.15, 0.25)),
        _f("packet_rate",      rng.uniform(1, 10),       rng.uniform(0.10, 0.20)),
        _f("entropy",          rng.uniform(7.5, 8.0),    rng.uniform(0.10, 0.20)),
    ],
}


# ── Explanation templates ─────────────────────────────────────────────────────
EXPLANATIONS = {
    "DoS_DDoS": (
        "Extremely high packet rate ({pkt_rate:.0f} pkt/s) and SYN flood count strongly "
        "indicate a Denial-of-Service or DDoS attack originating from {src_ip}. "
        "The IF anomaly score of {if_score:.4f} confirms severe deviation from the benign baseline."
    ),
    "Reconnaissance": (
        "Low packet rate sustained over a long duration with elevated SYN and RST counts "
        "is consistent with port-scanning or network-topology mapping from {src_ip}. "
        "RF confidence: {rf_prob:.0%}; the host appears to be probing multiple services."
    ),
    "BruteForce_Password": (
        "Repeated SYN/RST bursts with very small payload sizes indicate automated "
        "credential-stuffing or brute-force login attempts from {src_ip}. "
        "RF model is {rf_prob:.0%} confident in this BruteForce_Password classification."
    ),
    "Injection_Web": (
        "Abnormally large payload sizes and elevated PSH counts suggest web-layer injection "
        "attempts (SQL, XSS, or command injection) originating from {src_ip}. "
        "This traffic pattern deviates significantly from normal HTTP/S application flows."
    ),
    "MITM_Spoofing": (
        "Very low inter-arrival-time standard deviation and prolonged flow duration are "
        "hallmarks of ARP spoofing or a man-in-the-middle attack from {src_ip}. "
        "Immediate isolation is recommended to prevent credential interception."
    ),
    "Botnet_Malware": (
        "Highly regular inter-arrival timing and sustained long-duration flows match known "
        "C&C beacon patterns from {src_ip}. "
        "Elevated entropy suggests encrypted, scheduled communication with a botnet controller."
    ),
    "Benign_Anomaly": (
        "The Random Forest classified this flow as Benign ({rf_prob:.0%} confidence), "
        "but the Isolation Forest assigned an anomaly percentile of {if_score:.4f}, well above "
        "the 99th-percentile threshold — this may represent a novel attack pattern from {src_ip}."
    ),
    "Benign_Normal": (
        "All traffic indicators from {src_ip} are within the expected range for "
        "legitimate IoT device communication. "
        "RF confidence {rf_prob:.0%} and IF score {if_score:.4f} are both consistent with benign baseline."
    ),
}

RECOMMENDATIONS = {
    "DoS_DDoS":            "Block source IP {src_ip} at the perimeter firewall and enable rate-limiting on the affected subnet immediately.",
    "Reconnaissance":      "Add {src_ip} to the threat watch-list and enable full packet capture on all open ports for 30 minutes.",
    "BruteForce_Password": "Block {src_ip} and enforce account lockout after 5 failed attempts on the targeted authentication service.",
    "Injection_Web":       "Block {src_ip}, inspect server-side logs for successful injections, and apply WAF rules to filter malicious payloads.",
    "MITM_Spoofing":       "Isolate {src_ip} from the network segment, flush ARP caches on all local devices, and rotate credentials immediately.",
    "Botnet_Malware":      "Quarantine the device at {src_ip}, perform a full malware scan, and block known C&C domains at the DNS resolver.",
    "Benign_Anomaly":      "Manually investigate {src_ip}; capture a 5-minute packet trace and compare against known-benign baselines.",
    "Benign_Normal":       "No action required; continue standard baseline monitoring of {src_ip}.",
}


def _explanation_key(rf_class: str, verdict: str) -> str:
    if verdict == "Known Attack":
        return rf_class
    return "Benign_Anomaly" if verdict == "Unknown Anomaly" else "Benign_Normal"


def make_explanation(rf_class, rf_prob, if_score, src_ip, verdict) -> str:
    key      = _explanation_key(rf_class, verdict)
    template = EXPLANATIONS.get(key, EXPLANATIONS["Benign_Normal"])
    return template.format(
        src_ip=src_ip, rf_prob=rf_prob, if_score=if_score,
        pkt_rate=float(rng.uniform(800, 4500)),
    )


def make_recommendation(rf_class, verdict, src_ip) -> str:
    key      = _explanation_key(rf_class, verdict)
    template = RECOMMENDATIONS.get(key, RECOMMENDATIONS["Benign_Normal"])
    return template.format(src_ip=src_ip)


# ── IP helpers ────────────────────────────────────────────────────────────────
def random_ip(prefixes: list[str]) -> str:
    return f"{random.choice(prefixes)}.{random.randint(1, 254)}"


# ── Row generator ─────────────────────────────────────────────────────────────
def generate_row(category: str, timestamp: datetime, flow_id: int) -> dict:
    src_ip   = random_ip(SRC_PREFIXES)
    dst_ip   = random_ip(DST_PREFIXES)
    protocol = random.choices(PROTOCOLS, weights=PROTOCOL_WEIGHTS)[0]

    if category == "Normal":
        rf_class      = "Benign"
        rf_probability = float(rng.uniform(0.70, 0.98))
        if_score       = float(rng.uniform(0.0, 0.988))

    elif category == "Known Attack":
        rf_class = random.choice(ATTACK_CLASSES)
        # ~60 % High-risk (prob ≥ 0.8), ~40 % Medium (0.45 – 0.79)
        if rng.random() < 0.60:
            rf_probability = float(rng.uniform(0.80, 0.99))
        else:
            rf_probability = float(rng.uniform(0.45, 0.79))
        if_score = float(rng.uniform(0.60, 0.998))

    else:  # Unknown Anomaly
        rf_class       = "Benign"
        rf_probability = float(rng.uniform(0.50, 0.95))   # some will be < 0.6 → needs_review
        if_score       = float(rng.uniform(0.990, 1.0))   # must be ≥ tau

    verdict, risk_level, needs_review = compute_labels(rf_class, rf_probability, if_score)

    # Feature contributions
    fn           = FEATURE_TEMPLATES.get(rf_class, FEATURE_TEMPLATES["Benign"])
    top_features = fn()

    explanation    = make_explanation(rf_class, rf_probability, if_score, src_ip, verdict)
    recommendation = make_recommendation(rf_class, verdict, src_ip)

    return {
        "timestamp":      timestamp.isoformat(timespec="seconds"),
        "flow_id":        flow_id,
        "src_ip":         src_ip,
        "dst_ip":         dst_ip,
        "protocol":       protocol,
        "rf_class":       rf_class,
        "rf_probability": round(rf_probability, 4),
        "if_score":       round(if_score, 4),
        "verdict":        verdict,
        "risk_level":     risk_level,
        "needs_review":   needs_review,
        "top_features":   json.dumps(top_features),
        "explanation":    explanation,
        "recommendation": recommendation,
    }


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    os.makedirs("logs", exist_ok=True)

    now   = datetime.now()
    start = now - timedelta(minutes=30)

    # Build category list and shuffle with fixed seed
    categories = (
        ["Normal"]          * N_NORMAL  +
        ["Known Attack"]    * N_ATTACK  +
        ["Unknown Anomaly"] * N_ANOMALY
    )
    random.shuffle(categories)   # seed already set at top

    # Timestamps: uniformly spread over 30-min window, sorted ascending
    raw_times = sorted(
        start + timedelta(seconds=random.uniform(0, 1800))
        for _ in range(N_ROWS)
    )

    rows = []
    for i, (cat, ts) in enumerate(zip(categories, raw_times)):
        row = generate_row(cat, ts, flow_id=1000 + i)
        rows.append(row)

    fieldnames = list(rows[0].keys())

    # ── Write CSV ─────────────────────────────────────────────────────────────
    csv_path = os.path.join("logs", "alerts.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"[OK]  Wrote {len(rows)} rows  ->  {csv_path}")

    # ── Write JSONL ───────────────────────────────────────────────────────────
    jsonl_path = os.path.join("logs", "alerts.jsonl")
    with open(jsonl_path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")
    print(f"[OK]  Wrote {len(rows)} rows  ->  {jsonl_path}")

    # ── Quick stats ───────────────────────────────────────────────────────────
    from collections import Counter
    verdict_cnt = Counter(r["verdict"]    for r in rows)
    risk_cnt    = Counter(r["risk_level"] for r in rows)
    review_cnt  = sum(1 for r in rows if r["needs_review"])

    print("\n-- Verdict distribution --")
    for k in ["Normal", "Known Attack", "Unknown Anomaly"]:
        v = verdict_cnt.get(k, 0)
        print(f"  {k:<20}  {v:>3}  ({v/N_ROWS*100:4.1f} %)")

    print("\n-- Risk distribution --")
    for k in ["Low", "Medium", "High"]:
        v = risk_cnt.get(k, 0)
        print(f"  {k:<20}  {v:>3}  ({v/N_ROWS*100:4.1f} %)")

    print(f"\n-- needs_review = True:  {review_cnt}  ({review_cnt/N_ROWS*100:.1f} %)")
    print("\n[DONE] Data generation complete.")


if __name__ == "__main__":
    main()
