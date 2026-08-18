#!/usr/bin/env python3
"""
Q3 — Multi-harmonic Fourier variant.

The paper-canonical F_raw uses a single-harmonic maximum:
    F_raw(t) = max_k R²(E_t | span{cos(2πk/p), sin(2πk/p)})

We extend it to a top-K multi-harmonic subspace:
    F_K(t)   = R²(E_t | span{top-K energy harmonics of E_t})
where the K harmonics with highest per-harmonic R² at each timestep are
selected (data-driven; consistent with "projection onto low-dim DFT
subspace").

K ∈ {1, 2, 3, 5}  (K=1 reproduces the single-harmonic metric, so serves
as internal sanity check).

For each cell we:
  1. recompute F_K over the trajectory,
  2. estimate a permutation null at the same K via 100 shuffles of token
     indices (p95),
  3. compute F_K^corr = max(0, F_K - null_p95^K),
  4. detect τ_F^K via the same EMA + BIC pipeline,
  5. report τ_F^K and compare G<F ordering at each K.

Output:
  results/posthoc/q3_multiharmonic.csv
  results/posthoc/q3_multiharmonic_table.md

Usage:
  python src/analysis/q3_multiharmonic.py
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from grok_metrics import (
    estimate_changepoint,
    find_tau_sustained,
)

K_VALUES = [1, 2, 3, 5]
N_NULL_PERMS = 100
NULL_PCT = 95
EMA_ALPHA = 0.15
SLOPE_THRESH = 0.01


def _harmonic_r2_each(E: np.ndarray, prime: int) -> np.ndarray:
    """Per-harmonic R² against an embedding matrix, for k=1..p//2."""
    E_c = E - E.mean(0)
    total_var = float(np.sum(E_c ** 2))
    if total_var < 1e-12:
        return np.zeros(prime // 2)
    thetas = 2.0 * np.pi * np.arange(prime) / prime
    r2s = np.zeros(prime // 2)
    for k in range(1, prime // 2 + 1):
        B = np.stack([np.cos(k * thetas), np.sin(k * thetas)], axis=1)
        Q, _ = np.linalg.qr(B)
        proj = Q @ (Q.T @ E_c)
        res = float(np.sum((E_c - proj) ** 2))
        r2s[k - 1] = 1.0 - res / total_var
    return r2s


def _multiharmonic_r2(E: np.ndarray, prime: int, K: int,
                      k_idx: np.ndarray | None = None
                      ) -> tuple[float, np.ndarray]:
    """R² onto the K harmonics with the highest individual R² at this step.

    If k_idx is given (list of harmonic indices 1..p//2), uses those directly;
    otherwise picks top-K by per-harmonic energy.
    """
    E_c = E - E.mean(0)
    total_var = float(np.sum(E_c ** 2))
    if total_var < 1e-12:
        return 0.0, np.array([])
    if k_idx is None:
        r2s = _harmonic_r2_each(E_c, prime)
        k_idx = np.argsort(r2s)[::-1][:K] + 1  # top-K, 1-indexed
    thetas = 2.0 * np.pi * np.arange(prime) / prime
    cols = []
    for k in k_idx:
        cols.append(np.cos(k * thetas))
        cols.append(np.sin(k * thetas))
    B = np.stack(cols, axis=1)
    Q, _ = np.linalg.qr(B)
    proj = Q @ (Q.T @ E_c)
    res = float(np.sum((E_c - proj) ** 2))
    return 1.0 - res / total_var, np.array(k_idx)


def _null_p95_multi(E: np.ndarray, prime: int, K: int,
                    n_perms: int, rng: np.random.Generator) -> float:
    idx = np.arange(prime)
    scores = []
    for _ in range(n_perms):
        rng.shuffle(idx)
        r2, _ = _multiharmonic_r2(E[idx], prime, K, k_idx=None)
        scores.append(r2)
    return float(np.percentile(scores, NULL_PCT))


def _ema(values: np.ndarray, alpha: float) -> np.ndarray:
    if alpha >= 1.0:
        return values.copy()
    out = np.zeros_like(values)
    out[0] = values[0]
    for i in range(1, len(values)):
        out[i] = alpha * values[i] + (1 - alpha) * out[i - 1]
    return out


def analyse_one(npz_path: Path, seed: int) -> dict:
    data = np.load(npz_path, allow_pickle=True)
    steps = data["steps"].astype(int)
    emb_traj = data["emb"]
    test_acc = data["test_acc"].astype(float)
    prime = emb_traj.shape[1]
    T = len(steps)

    tau_gen = find_tau_sustained(steps.tolist(), test_acc.tolist(),
                                 threshold=0.9, n_sustained=3)

    out: dict[int, float | None] = {}
    for K in K_VALUES:
        f_raw = np.zeros(T)
        f_null = np.zeros(T)
        for t in range(T):
            r2, _ = _multiharmonic_r2(emb_traj[t], prime, K, k_idx=None)
            f_raw[t] = r2
            rng = np.random.default_rng(seed ^ 0xDEAD ^ (K * 10_007 + t))
            f_null[t] = _null_p95_multi(emb_traj[t], prime, K,
                                        N_NULL_PERMS, rng)
        f_corr = np.maximum(0.0, f_raw - f_null)
        f_smooth = _ema(f_corr, EMA_ALPHA)
        tau_F = estimate_changepoint(
            steps.tolist(), f_smooth.tolist(),
            slope_rel_threshold=SLOPE_THRESH,
        )
        out[K] = tau_F

    return {"tau_gen": tau_gen, "by_K": out}


def _ordering(tau_gen, tau_F, coincident_thresh: int = 500) -> str:
    if tau_gen is None and tau_F is None:
        return "none"
    if tau_gen is None:
        return "F_only"
    if tau_F is None:
        return "G_only"
    delta = tau_F - tau_gen
    if abs(delta) <= coincident_thresh:
        return "tie"
    return "G<F" if delta > 0 else "F<G"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--indir", default="results/posthoc")
    ap.add_argument("--manifest", default=None)
    args = ap.parse_args()

    indir = Path(args.indir)
    manifest_path = Path(args.manifest) if args.manifest \
        else indir / "manifest.csv"

    if not manifest_path.exists():
        print(f"ERROR: manifest not found at {manifest_path}")
        return

    with open(manifest_path, newline="", encoding="utf-8") as f:
        cells = list(csv.DictReader(f))

    rows: list[dict] = []
    for row in cells:
        label = row["label"]
        seed = int(row["seed"])
        npz_path = indir / row["file"]
        if not npz_path.exists():
            print(f"  SKIP {label}: {npz_path} missing")
            continue
        print(f"Analysing {label} ...")
        res = analyse_one(npz_path, seed)
        rows.append({
            "label": label,
            "category": row["category"],
            "tau_gen": res["tau_gen"],
            "by_K": res["by_K"],
        })

    csv_out = indir / "q3_multiharmonic.csv"
    with open(csv_out, "w", newline="", encoding="utf-8") as fc:
        w = csv.writer(fc)
        w.writerow(["cell_label", "category", "tau_gen", "K",
                    "tau_F_K", "ordering_K"])
        for r in rows:
            for K, tau_F in r["by_K"].items():
                ord_K = _ordering(r["tau_gen"], tau_F)
                w.writerow([
                    r["label"], r["category"], r["tau_gen"], K,
                    f"{tau_F:.0f}" if tau_F is not None else "",
                    ord_K,
                ])
    print(f"\nCSV → {csv_out}")

    # Markdown table
    md = indir / "q3_multiharmonic_table.md"
    lines = [
        "# Q3 — Multi-harmonic variant (τ_F and ordering at K ∈ {1,2,3,5})",
        "",
        "K=1 reproduces canonical single-harmonic detection. "
        "A consistent G<F at every K ≥ 2 indicates the ordering is not an "
        "artefact of single-harmonic restriction.",
        "",
    ]
    header = ["cell (cat)", "tau_gen"] + \
             [f"K={K}: tau_F / ord." for K in K_VALUES]
    lines.append("| " + " | ".join(header) + " |")
    lines.append("|" + "|".join([" --- "] * len(header)) + "|")
    for r in rows:
        cells_row = [f"{r['label']} ({r['category']})",
                     str(r["tau_gen"]) if r["tau_gen"] is not None else "–"]
        for K in K_VALUES:
            tau = r["by_K"].get(K)
            ord_K = _ordering(r["tau_gen"], tau)
            tau_s = f"{tau:.0f}" if tau is not None else "–"
            cells_row.append(f"{tau_s} / {ord_K}")
        lines.append("| " + " | ".join(cells_row) + " |")
    md.write_text("\n".join(lines), encoding="utf-8")
    print(f"MD  → {md}")


if __name__ == "__main__":
    main()
