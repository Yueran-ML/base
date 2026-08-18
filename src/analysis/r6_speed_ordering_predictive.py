#!/usr/bin/env python3
"""
R6 - Predictive test of the speed-dependent ordering hypothesis.

Question
--------
The paper's Section 7 reports a correlative finding: across Grokking
cells, the sign of Delta_tau = tau_F - tau_gen is correlated with
generalization speed (Spearman r ~ -0.62 against tau_gen).
Reviewer R6 asks whether this is predictive: given (lr, wd) and tau_gen
on a held-out cell, can we predict the sign of Delta_tau better than the
class prior?

Method
------
- Source: results/stage6_2d/results_stage6_2d.csv (147 cells).
- Restrict to cells with both tau_gen and tau_F detected (Grokking +
  coincident with both events), so Delta_tau is well defined.
- Target: y = 1 if Delta_tau > 500 (G<F), 0 if Delta_tau < -500 (F<G);
  drop coincident |Delta_tau| <= 500.
- Features: log10(lr), log10(wd), log1p(tau_gen).
- Model: scikit-learn LogisticRegression, leave-one-out CV.
- Report: ROC-AUC, accuracy, Brier score, baseline majority-class
  accuracy; 1000-iter bootstrap CI on AUC.

Output
------
results/posthoc/r6_speed_predictive.csv     (per-row predictions)
results/posthoc/r6_speed_predictive.md      (summary metrics)
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np


def _load_grokking_rows(csv_path: Path):
    rows = []
    for r in csv.DictReader(open(csv_path, encoding="utf-8")):
        if not r["tau_gen"] or not r["tau_F"]:
            continue
        try:
            tau_gen = float(r["tau_gen"])
            tau_F = float(r["tau_F"])
            lr = float(r["lr"])
            wd = float(r["wd"])
        except ValueError:
            continue
        delta = tau_F - tau_gen
        if abs(delta) <= 500:
            continue  # coincident; not a sign target
        rows.append({
            "lr": lr, "wd": wd, "tau_gen": tau_gen, "tau_F": tau_F,
            "delta": delta, "y": 1 if delta > 0 else 0,
            "seed": int(r["seed"]),
        })
    return rows


def _design_matrix(rows):
    X = np.array([[np.log10(r["lr"]), np.log10(r["wd"]),
                   np.log1p(r["tau_gen"])] for r in rows], dtype=float)
    y = np.array([r["y"] for r in rows], dtype=int)
    return X, y


def _logistic(X_train, y_train, X_test, n_iter=400, lr=0.05, l2=1.0):
    """Plain logistic regression with L2, gradient descent. No sklearn."""
    n, d = X_train.shape
    mu = X_train.mean(axis=0)
    sd = X_train.std(axis=0) + 1e-9
    Xn = (X_train - mu) / sd
    Xtn = (X_test - mu) / sd
    w = np.zeros(d)
    b = 0.0
    for _ in range(n_iter):
        z = Xn @ w + b
        p = 1.0 / (1.0 + np.exp(-z))
        gw = Xn.T @ (p - y_train) / n + l2 * w / n
        gb = float(np.mean(p - y_train))
        w -= lr * gw
        b -= lr * gb
    z_test = Xtn @ w + b
    return 1.0 / (1.0 + np.exp(-z_test))


def _auc(y_true, scores):
    order = np.argsort(-scores)
    y_sorted = y_true[order]
    pos = np.sum(y_true == 1)
    neg = len(y_true) - pos
    if pos == 0 or neg == 0:
        return float("nan")
    tp = np.cumsum(y_sorted == 1)
    fp = np.cumsum(y_sorted == 0)
    tpr = tp / pos
    fpr = fp / neg
    auc = float(np.trapezoid(tpr, fpr))
    return float(auc)


def _brier(y_true, p):
    return float(np.mean((p - y_true) ** 2))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src",
                    default="results/stage6_2d/results_stage6_2d.csv")
    ap.add_argument("--outdir", default="results/posthoc")
    args = ap.parse_args()

    rows = _load_grokking_rows(Path(args.src))
    n = len(rows)
    if n == 0:
        print("No Grokking rows with both tau_gen and tau_F.")
        return
    X, y = _design_matrix(rows)
    print(f"n_grokking_rows = {n};  P(G<F) = {np.mean(y):.3f}")

    # Leave-one-out CV
    p_loo = np.zeros(n)
    for i in range(n):
        idx = np.ones(n, dtype=bool); idx[i] = False
        p_loo[i] = _logistic(X[idx], y[idx], X[i:i + 1])[0]
    pred = (p_loo >= 0.5).astype(int)
    acc = float(np.mean(pred == y))
    auc = _auc(y, p_loo)
    brier = _brier(y, p_loo)
    base_acc = max(np.mean(y), 1 - np.mean(y))

    # Lr-/wd-only ablations to see whether the lift comes from speed
    X_geom = X[:, :2]  # (lr, wd) only
    p_loo_geom = np.zeros(n)
    for i in range(n):
        idx = np.ones(n, dtype=bool); idx[i] = False
        p_loo_geom[i] = _logistic(X_geom[idx], y[idx], X_geom[i:i + 1])[0]
    auc_geom = _auc(y, p_loo_geom)
    acc_geom = float(np.mean((p_loo_geom >= 0.5).astype(int) == y))

    X_speed = X[:, 2:3]
    p_loo_speed = np.zeros(n)
    for i in range(n):
        idx = np.ones(n, dtype=bool); idx[i] = False
        p_loo_speed[i] = _logistic(X_speed[idx], y[idx], X_speed[i:i + 1])[0]
    auc_speed = _auc(y, p_loo_speed)

    # Bootstrap CI on full-feature AUC
    rng = np.random.default_rng(20240425)
    aucs_b = []
    for _ in range(1000):
        b = rng.integers(0, n, size=n)
        a = _auc(y[b], p_loo[b])
        if not np.isnan(a):
            aucs_b.append(a)
    auc_lo, auc_hi = np.percentile(aucs_b, [2.5, 97.5])

    outdir = Path(args.outdir); outdir.mkdir(parents=True, exist_ok=True)
    pred_csv = outdir / "r6_speed_predictive.csv"
    with open(pred_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["lr", "wd", "seed", "tau_gen", "tau_F", "delta",
                    "y", "p_full", "p_geom", "p_speed"])
        for r, pf, pg, ps in zip(rows, p_loo, p_loo_geom, p_loo_speed):
            w.writerow([r["lr"], r["wd"], r["seed"],
                        f"{r['tau_gen']:.0f}", f"{r['tau_F']:.0f}",
                        f"{r['delta']:.0f}", r["y"],
                        f"{pf:.4f}", f"{pg:.4f}", f"{ps:.4f}"])

    md = outdir / "r6_speed_predictive.md"
    md.write_text(
        "# R6 - Predictive test of the speed-dependent ordering hypothesis\n\n"
        "Source: `results/stage6_2d/results_stage6_2d.csv` "
        f"({n} cells with both tau_gen and tau_F detected and "
        "|delta_tau| > 500).\n\n"
        f"Class prior P(G<F) = {np.mean(y):.3f}; baseline majority-class "
        f"accuracy = {base_acc:.3f}.\n\n"
        "Leave-one-out logistic regression on standardised features.\n\n"
        "| feature set | LOO AUC | 95% CI | LOO acc |\n"
        "| --- | --- | --- | --- |\n"
        f"| log10(lr), log10(wd), log1p(tau_gen) | {auc:.3f} | "
        f"[{auc_lo:.3f}, {auc_hi:.3f}] | {acc:.3f} |\n"
        f"| log1p(tau_gen) only | {auc_speed:.3f} | -- | -- |\n"
        f"| log10(lr), log10(wd) only | {auc_geom:.3f} | -- | "
        f"{acc_geom:.3f} |\n\n"
        f"Brier score (full model) = {brier:.3f}.\n\n"
        "Interpretation:\n"
        "- AUC > baseline implies (lr, wd, tau_gen) carry leave-one-out\n"
        "  predictive signal about whether a held-out grokking cell is\n"
        "  G<F or F<G; this elevates Section 7's correlation into a\n"
        "  predictive statement.\n"
        "- The tau_gen-only ablation isolates the speed contribution; if\n"
        "  it accounts for most of the AUC, the speed-dependent\n"
        "  hypothesis is the dominant driver.\n",
        encoding="utf-8",
    )
    print(f"AUC full = {auc:.3f}  CI=[{auc_lo:.3f},{auc_hi:.3f}]  "
          f"acc={acc:.3f}  brier={brier:.3f}")
    print(f"AUC speed-only = {auc_speed:.3f};  "
          f"AUC geom-only = {auc_geom:.3f}")
    print(f"-> {pred_csv}")
    print(f"-> {md}")


if __name__ == "__main__":
    main()
