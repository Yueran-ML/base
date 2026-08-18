#!/usr/bin/env python3
"""
Q2 — Null-correction sensitivity (percentile × permutation count).

For each trajectory .npz produced by src/sweeps/posthoc_trajectory_dump.py,
we recompute the permutation null at every log step under each combination
  percentile   ∈ {90, 95, 99}
  n_perms      ∈ {50, 100, 200}
and re-detect τ_F via the same BIC + EMA pipeline used in the paper
(canonical α=0.15, slope_thresh=0.01).

Output:
  results/posthoc/q2_null_sensitivity.csv
    cell_label, percentile, n_perms, tau_F, delta_tau_F_vs_canonical
  results/posthoc/q2_null_sensitivity_table.md   (ready-to-paste Markdown)

Canonical reference: (p95, n=100).  delta_tau_F is in training steps.

Usage:
  python src/analysis/q2_null_sensitivity.py
  python src/analysis/q2_null_sensitivity.py --indir results/posthoc
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from grok_metrics import (
    compute_fourier_alignment,
    estimate_changepoint,
    find_tau_sustained,
)

PERCENTILES = [90, 95, 99]
N_PERMS_LIST = [50, 100, 200]
CANONICAL = (95, 100)
EMA_ALPHA = 0.15
SLOPE_THRESH = 0.01


def _null_percentile(
    emb: np.ndarray, prime: int, n_perms: int, percentile: float,
    rng: np.random.Generator,
) -> float:
    idx = np.arange(prime)
    scores = []
    for _ in range(n_perms):
        rng.shuffle(idx)
        s, _ = compute_fourier_alignment(emb[idx], prime)
        scores.append(s)
    return float(np.percentile(scores, percentile))


def _ema(values: np.ndarray, alpha: float) -> np.ndarray:
    if alpha >= 1.0:
        return values.copy()
    out = np.zeros_like(values)
    out[0] = values[0]
    for i in range(1, len(values)):
        out[i] = alpha * values[i] + (1 - alpha) * out[i - 1]
    return out


def analyse_trajectory(npz_path: Path, seed: int) -> dict:
    """Return {(percentile, n_perms) -> tau_F} for one trajectory file."""
    data = np.load(npz_path, allow_pickle=True)
    steps = data["steps"].astype(int)
    emb_traj = data["emb"]          # (T, p, d)
    f_raw = data["f_raw"].astype(float)
    test_acc = data["test_acc"].astype(float)
    prime = emb_traj.shape[1]

    tau_gen = find_tau_sustained(steps.tolist(), test_acc.tolist(),
                                 threshold=0.9, n_sustained=3)

    out: dict[tuple[int, int], float | None] = {}
    for p in PERCENTILES:
        for n in N_PERMS_LIST:
            rng = np.random.default_rng(seed ^ 0xDEAD ^ (p * 1000 + n))
            f_null = np.array([
                _null_percentile(emb_traj[t], prime, n, p, rng)
                for t in range(len(steps))
            ])
            f_corr = np.maximum(0.0, f_raw - f_null)
            f_smooth = _ema(f_corr, EMA_ALPHA)
            tau_F = estimate_changepoint(
                steps.tolist(), f_smooth.tolist(),
                slope_rel_threshold=SLOPE_THRESH,
            )
            out[(p, n)] = tau_F
    return {"tau_gen": tau_gen, "by_cfg": out}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--indir", default="results/posthoc")
    ap.add_argument("--manifest", default=None,
                    help="CSV with cell → file mapping; default <indir>/manifest.csv")
    args = ap.parse_args()

    indir = Path(args.indir)
    manifest_path = Path(args.manifest) if args.manifest \
        else indir / "manifest.csv"

    if not manifest_path.exists():
        print(f"ERROR: manifest not found at {manifest_path}")
        return

    with open(manifest_path, newline="", encoding="utf-8") as f:
        cells = list(csv.DictReader(f))

    csv_out = indir / "q2_null_sensitivity.csv"
    with open(csv_out, "w", newline="", encoding="utf-8") as fcsv:
        w = csv.writer(fcsv)
        w.writerow(["cell_label", "category", "tau_gen",
                    "percentile", "n_perms", "tau_F",
                    "delta_tau_F_vs_canonical"])

        cell_tables: dict[str, dict] = {}

        for row in cells:
            label = row["label"]
            category = row["category"]
            seed = int(row["seed"])
            npz_path = indir / row["file"]
            if not npz_path.exists():
                print(f"  SKIP {label}: {npz_path} missing")
                continue

            print(f"Analysing {label} ({category}) ...")
            res = analyse_trajectory(npz_path, seed)
            canonical_tau = res["by_cfg"].get(CANONICAL)
            cell_tables[label] = {
                "category": category,
                "tau_gen": res["tau_gen"],
                "canonical_tau_F": canonical_tau,
                "by_cfg": res["by_cfg"],
            }

            for (p, n), tau_F in res["by_cfg"].items():
                if tau_F is None or canonical_tau is None:
                    delta = ""
                else:
                    delta = f"{tau_F - canonical_tau:.0f}"
                w.writerow([
                    label, category, res["tau_gen"],
                    p, n,
                    f"{tau_F:.0f}" if tau_F is not None else "",
                    delta,
                ])

    print(f"\nCSV → {csv_out}")

    # ---- Markdown summary table ---------------------------------------
    md = indir / "q2_null_sensitivity_table.md"
    lines = [
        "# Q2 — Null sensitivity (τ_F shift vs canonical p95, n=100)",
        "",
        "Each cell reports Δτ_F in training steps "
        "(negative = earlier detection).",
        "",
    ]
    header = ["cell (cat)"] + [f"p{p}/n{n}"
                               for p in PERCENTILES for n in N_PERMS_LIST]
    lines.append("| " + " | ".join(header) + " |")
    lines.append("|" + "|".join([" --- "] * len(header)) + "|")
    for label, info in cell_tables.items():
        canon = info["canonical_tau_F"]
        cells_row = [f"{label} ({info['category']})"]
        for p in PERCENTILES:
            for n in N_PERMS_LIST:
                tau = info["by_cfg"].get((p, n))
                if tau is None or canon is None:
                    cells_row.append("–")
                else:
                    cells_row.append(f"{tau - canon:+.0f}")
        lines.append("| " + " | ".join(cells_row) + " |")
    md.write_text("\n".join(lines), encoding="utf-8")
    print(f"MD  → {md}")


if __name__ == "__main__":
    main()
