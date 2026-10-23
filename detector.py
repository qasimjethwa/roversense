#!/usr/bin/env python3
"""
RoverSense ground-station scorer.

Batch-scores a CSV of rover telemetry with the trained IsolationForest
plus the deterministic guardrail layers (same logic as train.py and the
browser demo in docs/).

CSV format: header row with the six channel names
    battery_v,motor_current,chassis_temp,ph,soil_moisture,gas_mq
one row per second. Missing samples may be empty (NaN).

Usage:
    python detector.py telemetry.csv [-o scored.csv]
    python detector.py --demo        # score 60 s of simulated telemetry
"""

import argparse
import csv
import json
import sys

import joblib
import numpy as np

from train import (CHANNELS, anomaly_scores, detect_stream, featurize,
                   guardrail_stats, simulate_nominal)


def load_bundle():
    model = joblib.load("model/model.joblib")
    metrics = json.load(open("model/metrics.json"))
    return model, metrics


def score_array(X, model, metrics):
    med = np.array(metrics["export_stats"]["base_median"])
    mad_x = np.array(metrics["export_stats"]["base_mad_x"])
    mad_d1 = np.array(metrics["export_stats"]["base_mad_d1"])
    thr = metrics["threshold"]
    F = featurize(X, med)
    s = anomaly_scores(model, F)
    active, layer = detect_stream(s, X, F, med, mad_x, mad_d1, thr)
    return s, active, layer


def main():
    ap = argparse.ArgumentParser(description="Score rover telemetry.")
    ap.add_argument("csv", nargs="?", help="telemetry CSV to score")
    ap.add_argument("-o", "--out", default=None, help="output CSV path")
    ap.add_argument("--demo", action="store_true",
                    help="score 60 s of simulated nominal telemetry")
    args = ap.parse_args()

    model, metrics = load_bundle()
    names = [c for c, _ in CHANNELS]

    if args.demo:
        X = simulate_nominal(60, np.random.default_rng(123))
        header = names
    elif args.csv:
        with open(args.csv) as f:
            rows = list(csv.reader(f))
        header = [h.strip() for h in rows[0]]
        if header != names:
            sys.exit(f"expected header {names}, got {header}")
        X = np.array([[float(v) if v.strip() else np.nan for v in r]
                      for r in rows[1:]])
    else:
        ap.error("give a CSV path or --demo")

    s, active, layer = score_array(X, model, metrics)
    n_inc = int(active.sum())
    print(f"scored {len(X)} samples | threshold={metrics['threshold']:.4f} | "
          f"incident samples: {n_inc}")
    # print incident spans
    in_inc, start = False, 0
    for t in range(len(X)):
        if active[t] and not in_inc:
            in_inc, start = True, t
        elif not active[t] and in_inc:
            in_inc = False
            print(f"  incident t={start}..{t - 1} "
                  f"({t - start}s) peak={s[start:t].max():.3f}")
    if in_inc:
        print(f"  incident t={start}..{len(X) - 1} (open at end)")

    if args.out:
        with open(args.out, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(header + ["anomaly_score", "incident", "layers"])
            for i in range(len(X)):
                w.writerow([f"{v:.4f}" if v == v else "" for v in X[i]] +
                           [f"{s[i]:.4f}", int(active[i]), layer[i]])
        print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
