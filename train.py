#!/usr/bin/env python3
"""
RoverSense — train the anomaly detector.

Simulates nominal rover telemetry (6 sensor channels), trains a
scikit-learn IsolationForest on raw values + first-differences (12 features),
evaluates it on a labeled test stream with 5 injected fault/anomaly episodes,
then exports the ENTIRE forest (every tree, split and leaf) to docs/model.json
so the live demo can run the exact same model in the browser — no backend.

Usage:
    python train.py
Outputs:
    model/model.joblib     fitted sklearn model (ground-station / batch use)
    model/metrics.json     evaluation metrics (also consumed by the demo UI)
    docs/model.json        full forest export for the in-browser scorer
    docs/metrics.json      copy for the demo UI
"""

import json
import math
import os
from datetime import datetime, timezone

import joblib
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.metrics import precision_recall_fscore_support, roc_auc_score

RNG = np.random.default_rng(42)

CHANNELS = [
    ("battery_v", "V"),
    ("motor_current", "A"),
    ("chassis_temp", "C"),
    ("ph", "pH"),
    ("soil_moisture", "frac"),
    ("gas_mq", "frac"),
]
N_CH = len(CHANNELS)
DELTA_NAMES = [c + "_d" for c, _ in CHANNELS]
FEATURE_NAMES = [c for c, _ in CHANNELS] + DELTA_NAMES

# ----------------------------------------------------------------------------
# Telemetry synthesis — a plausible HERC-style rover testbed run at 1 Hz.
# ----------------------------------------------------------------------------

def simulate_nominal(n, rng):
    """Return (n, 6) nominal telemetry."""
    t = np.arange(n, dtype=float)

    # Drive plan: traverse segments with rests in between.
    driving = np.zeros(n)
    i = 0
    while i < n:
        rest = int(rng.integers(120, 420))
        seg = int(rng.integers(90, 400))
        driving[i + rest:i + rest + seg] = 1.0
        # ramp up/down over ~8 s
        i += rest + seg
    drive = np.convolve(driving, np.ones(9) / 9, mode="same").clip(0, 1)

    battery = 12.6 - 0.00022 * t + rng.normal(0, 0.015, n) - 0.25 * drive
    current = 0.4 + 3.1 * drive + rng.normal(0, 0.12, n)
    current += 0.6 * np.sin(2 * np.pi * t / 47.0) * drive  # terrain ripple
    k = min(120, n)  # thermal kernel must fit short demo runs
    temp = 28.0 + 0.4 * np.sin(2 * np.pi * t / 1800.0) + 2.2 * np.convolve(
        drive, np.ones(k) / k, mode="same")[:n] + rng.normal(0, 0.08, n)

    # pH probe in a soil testbed tray: slow random walk around 6.9.
    # NOTE: noise is added AFTER the soft bounds so the channel is never
    # exactly constant in nominal data (a hard clip would fake "freeze"
    # segments and teach the model that stuck sensors are normal).
    ph_walk = 6.9 + np.cumsum(rng.normal(0, 0.0016, n))
    ph_walk = np.clip(ph_walk, 6.3, 7.5)
    ph = ph_walk + rng.normal(0, 0.02, n)

    # Soil moisture: slow dry-down, disturbed (dips) while driving.
    moist_walk = (0.34 - 0.00002 * t - 0.05 * np.convolve(
        drive, np.ones(60) / 60, mode="same"))
    moist_walk = np.clip(moist_walk, 0.06, 0.6)
    moist = moist_walk + rng.normal(0, 0.004, n)

    # Gas sensor baseline with faint noise; benign micro-puffs occasionally.
    gas = 0.040 + rng.normal(0, 0.006, n)
    if n > 160:
        for _ in range(max(1, n // 2400)):
            at = int(rng.integers(0, n - 60))
            gas[at:at + 40] += 0.02 * np.exp(-np.arange(40) / 12.0)
    gas = np.clip(gas, 0.0, None)

    return np.column_stack([battery, current, temp, ph, moist, gas])


def inject_faults(X, rng):
    """Return (X_faulty, labels, episodes) with 5 labeled episodes."""
    Xf = X.copy()
    n = len(Xf)
    labels = np.zeros(n, dtype=int)
    episodes = []  # (name, start, end, channel, kind)

    def add(name, s, e, ch, kind):
        labels[s:e] = 1
        episodes.append({"name": name, "start": int(s), "end": int(e),
                         "channel": CHANNELS[ch][0], "kind": kind})

    # 1. freeze — pH probe fouled, stuck at last value (120 s)
    s = 250
    Xf[s:s + 120, 3] = Xf[s - 1, 3]
    add("pH probe freeze", s, s + 120, 3, "freeze")

    # 2. drift — gas sensor aging, slow upward bias (150 s)
    s = 520
    drift = 0.004 * np.arange(150)
    Xf[s:s + 150, 5] += drift
    add("gas sensor drift", s, s + 150, 5, "drift")

    # 3. spike — EMI bursts on the motor-current line (200 s window)
    s = 820
    for _ in range(9):
        at = s + int(rng.integers(0, 190))
        dur = int(rng.integers(1, 4))
        Xf[at:at + dur, 1] += rng.choice([-1, 1]) * rng.uniform(1.5, 2.8)
    add("current-line EMI spikes", s, s + 200, 1, "spike")

    # 4. dropout — soil-moisture connector loose, NaN gap (90 s)
    s = 1150
    Xf[s:s + 90, 4] = np.nan
    add("moisture sensor dropout", s, s + 90, 4, "dropout")

    # 5. gas_event — a REAL gas leak: fast rise, hold, decay. Not a fault,
    #    but the true anomaly the system exists to catch.
    s = 1420
    rise = np.linspace(0, 0.42, 30)
    hold = np.full(90, 0.42)
    decay = 0.42 * np.exp(-np.arange(60) / 22.0)
    prof = np.concatenate([rise, hold, decay])
    Xf[s:s + len(prof), 5] += prof
    add("gas leak event", s, s + len(prof), 5, "gas_event")

    return Xf, labels, episodes


ROLL = 12  # trailing window (s) for the stillness feature


def rolling_std(col):
    """Causal trailing-window std."""
    n = len(col)
    rstd = np.zeros(n)
    csum = np.cumsum(np.insert(col, 0, 0.0))
    csum2 = np.cumsum(np.insert(col ** 2, 0, 0.0))
    for t in range(n):
        s = max(0, t - ROLL + 1)
        k = t - s + 1
        mean = (csum[t + 1] - csum[s]) / k
        var = max(0.0, (csum2[t + 1] - csum2[s]) / k - mean ** 2)
        rstd[t] = math.sqrt(var)
    return rstd


def impute(X):
    """Forward-fill NaN per channel (dropouts hold last known value)."""
    Xf = X.copy()
    for c in range(Xf.shape[1]):
        col = Xf[:, c]
        mask = np.isnan(col)
        if mask.any():
            idx = np.where(~mask, np.arange(len(col)), 0)
            np.maximum.accumulate(idx, out=idx)
            col[mask] = col[idx[mask]]
    return Xf


def featurize(X, base_median):
    """Build 18 context features (3 per channel).

    Per channel: |x - training median| (level shift), stillness
    -log10(trailing-12s std) (stuck sensor -> large), |1-step delta|
    (transient spike -> large). Point-wise forests cannot see "stuck"
    or "drifting" sensors; these give each sample its recent context.
    """
    Xf = impute(X)
    feats = []
    for c in range(Xf.shape[1]):
        col = Xf[:, c]
        d1 = np.zeros_like(col)
        d1[1:] = np.abs(col[1:] - col[:-1])
        still = -np.log10(rolling_std(col) + 1e-6)
        dbase = np.abs(col - base_median[c])
        feats += [dbase, still, d1]
    return np.column_stack(feats)


CHANNEL_FEATURES = ["dev_base", "stillness_12s", "abs_d1"]
FEATURE_NAMES = [f"{c}_{sfx}" for c, _ in CHANNELS for sfx in CHANNEL_FEATURES]


def anomaly_scores(model, F):
    """Paper anomaly score s in (0, 1]; higher = more anomalous."""
    return -model.score_samples(F)


def guardrail_stats(X):
    """Robust per-channel median/MAD for x and for |d1|."""
    med = np.median(X, axis=0)
    mad_x = np.median(np.abs(X - med), axis=0)
    d1 = np.abs(np.diff(X, axis=0))
    mad_d1 = np.median(d1, axis=0)
    return med, mad_x, mad_d1


STILL_THR = 4.5    # -log10(std); above => sensor effectively frozen
STUCK_DWELL = 12   # seconds of stillness before "stuck" fires
SEV_Z = 8.0        # robust-z guardrail
SPIKE_Z = 8.0      # |d1| guardrail, in units of MAD(d1)


def detect_stream(s_raw, X, F, med, mad_x, mad_d1, thr, alpha=0.3,
                  min_dwell=3):
    """Shipped detection logic — identical math ships in docs/forest.js.

    Layers (evaluated together, reported separately):
      ML       EWMA(alpha)-smoothed IF score > thr for min_dwell samples
      Z        any channel |x-med|/mad_x > SEV_Z            (level excursion)
      STUCK    any channel stillness > STILL_THR for STUCK_DWELL samples
      SPIKE    any channel |d1|/mad_d1 > SPIKE_Z            (transient)
      NAN      any channel NaN                             (dropout)
    Returns (active, layer): boolean incident array + per-sample layer tags.
    """
    n = len(s_raw)
    ewma = np.zeros(n)
    e = s_raw[0]
    for t in range(n):
        e = alpha * s_raw[t] + (1 - alpha) * e
        ewma[t] = e
    ml_over = ewma > thr
    cnt = np.zeros(n, dtype=int)
    for t in range(n):
        cnt[t] = cnt[t - 1] + 1 if ml_over[t] else 0
    trig_ml = cnt >= min_dwell

    with np.errstate(divide="ignore", invalid="ignore"):
        z = np.abs(X - med) / np.maximum(mad_x, 1e-12)
    d1f = F[:, 2::3]
    with np.errstate(divide="ignore", invalid="ignore"):
        z1 = d1f / np.maximum(mad_d1, 1e-12)
    still = F[:, 1::3]

    trig_z = (z > SEV_Z).any(axis=1)
    trig_spike = (z1 > SPIKE_Z).any(axis=1)
    trig_nan = np.isnan(X).any(axis=1)
    still_over = still > STILL_THR
    trig_stuck = np.zeros(n, dtype=bool)
    run = np.zeros(X.shape[1], dtype=int)
    for t in range(n):
        run = np.where(still_over[t], run + 1, 0)
        trig_stuck[t] = bool((run >= STUCK_DWELL).any())

    # warmup: rolling features are unreliable before the window fills
    ready = np.arange(n) >= ROLL
    trig = (trig_ml | trig_z | trig_spike | trig_nan | trig_stuck) & ready
    active = np.zeros(n, dtype=bool)
    in_inc, quiet = False, 0
    for t in range(n):
        if not in_inc and trig[t]:
            in_inc, quiet = True, 0
        elif in_inc:
            if trig[t]:
                quiet = 0
            else:
                quiet += 1
                if quiet >= min_dwell:
                    in_inc = False
        active[t] = in_inc

    layer = np.full(n, "", dtype=object)
    for t in range(n):
        ls = []
        if trig_ml[t]:
            ls.append("ML")
        if trig_z[t]:
            ls.append("Z")
        if trig_stuck[t]:
            ls.append("STUCK")
        if trig_spike[t]:
            ls.append("SPIKE")
        if trig_nan[t]:
            ls.append("NAN")
        layer[t] = "+".join(ls)
    return active, layer
    return active


# ----------------------------------------------------------------------------
# Train / evaluate / export
# ----------------------------------------------------------------------------

def main():
    os.makedirs("model", exist_ok=True)
    os.makedirs("docs", exist_ok=True)

    print("[1/5] simulating nominal telemetry…")
    X_train = simulate_nominal(3600, RNG)
    X_val = simulate_nominal(1200, np.random.default_rng(7))
    base_median = np.median(X_train, axis=0)  # fixed reference per channel

    print("[2/5] training IsolationForest…")
    model = IsolationForest(
        n_estimators=64, max_samples=1024, max_features=1.0,
        contamination="auto", random_state=42, n_jobs=-1,
    )
    model.fit(featurize(X_train, base_median))
    med, mad_x, mad_d1 = guardrail_stats(X_train)

    print("[3/5] calibrating threshold on held-out nominal data…")
    val_scores = anomaly_scores(model, featurize(X_val, base_median))
    threshold = float(np.quantile(val_scores, 0.99))  # ~1% false-positive rate
    val_active, _ = detect_stream(val_scores, X_val, featurize(X_val, base_median),
                              med, mad_x, mad_d1, threshold)
    print(f"      threshold={threshold:.4f} "
          f"(val FPR={(val_scores > threshold).mean() * 100:.2f}%, "
          f"val incidents (all layers): {val_active.sum()})")

    print("[4/5] evaluating on labeled fault stream…")
    X_test_raw, y_test, episodes = inject_faults(
        simulate_nominal(1800, np.random.default_rng(99)),
        np.random.default_rng(99))
    F_test = featurize(X_test_raw, base_median)
    s_test = anomaly_scores(model, F_test)
    active, layer = detect_stream(s_test, X_test_raw, F_test,
                              med, mad_x, mad_d1, threshold)
    y_pred = active.astype(int)

    prec, rec, f1, _ = precision_recall_fscore_support(
        y_test, y_pred, average="binary", zero_division=0)
    auc = roc_auc_score(y_test, s_test)

    per_fault = {}
    for ep in episodes:
        seg = slice(ep["start"], ep["end"])
        hits = np.where(y_pred[seg])[0]
        per_fault[ep["name"]] = {
            "kind": ep["kind"],
            "channel": ep["channel"],
            "recall": float(y_pred[seg].mean()),
            "peak_score": float(s_test[seg].max()),
            "duration_s": ep["end"] - ep["start"],
            "detect_delay_s": int(hits[0]) if len(hits) else None,
            "first_layer": layer[seg][hits[0]] if len(hits) else None,
        }

    metrics = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "model": {"algorithm": "IsolationForest", "n_estimators": 64,
                  "max_samples": 1024, "n_features": 18,
                  "feature_names": FEATURE_NAMES,
                  "train_samples": len(X_train)},
        "detection": {"if_threshold": threshold, "ewma_alpha": 0.3,
                      "min_dwell_s": 3, "guardrail_z": 8.0,
                      "layers": {"ML": "EWMA(0.3) IF score > thr, 3s dwell",
                        "Z": "|x-med|/MAD > 8",
                        "STUCK": "stillness > 4.5 for 12s",
                        "SPIKE": "|d1|/MAD(d1) > 8",
                        "NAN": "any channel NaN"},
                      "note": "ML + deterministic guardrails from the same "
                              "training stats; identical logic in docs/"},
        "threshold": threshold,
        "export_stats": {
            "base_median": [float(v) for v in med],
            "base_mad_x": [float(v) for v in mad_x],
            "base_mad_d1": [float(v) for v in mad_d1],
        },
        "overall": {"precision": float(prec), "recall": float(rec),
                    "f1": float(f1), "roc_auc": float(auc)},
        "per_fault": per_fault,
        "episodes": episodes,
    }
    print(f"      precision={prec:.3f} recall={rec:.3f} "
          f"f1={f1:.3f} roc_auc={auc:.3f}")
    for name, m in per_fault.items():
        print(f"      {name:28s} recall={m['recall']:.3f} "
              f"delay={m['detect_delay_s']}s peak={m['peak_score']:.3f} "
              f"via={m['first_layer']}")

    print("[5/5] exporting forest to JSON for the browser scorer…")
    export = {
        "meta": {
            "n_features": 18,
            "feature_names": FEATURE_NAMES,
            "n_trees": len(model.estimators_),
            "max_samples": int(model.max_samples_),
            "threshold": threshold,
            "base_median": [float(v) for v in base_median],
            "base_mad_x": [float(v) for v in mad_x],
            "base_mad_d1": [float(v) for v in mad_d1],
            "detection": metrics["detection"],
            "export_tool": "train.py",
        },
        "trees": [],
    }
    for est in model.estimators_:
        tr = est.tree_
        export["trees"].append({
            "left": tr.children_left.tolist(),
            "right": tr.children_right.tolist(),
            "feature": tr.feature.tolist(),
            "threshold": [float(v) for v in tr.threshold],
            "n_node_samples": tr.n_node_samples.tolist(),
            "depth": tr.compute_node_depths().tolist(),
        })
    with open("docs/model.json", "w") as f:
        json.dump(export, f, separators=(",", ":"))
    size_kb = os.path.getsize("docs/model.json") / 1024
    print(f"      docs/model.json — {size_kb:.0f} KB, "
          f"{len(export['trees'])} trees")

    joblib.dump(model, "model/model.joblib")
    with open("model/metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    with open("docs/metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    # Save a labeled test slice for the JS parity check.
    np.savez("model/parity.npz",
             F=F_test[:400].astype(np.float64),
             s_sklearn=s_test[:400])
    print("done.")


if __name__ == "__main__":
    main()
