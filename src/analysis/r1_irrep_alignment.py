#!/usr/bin/env python3
"""
R1/R9 - Cyclic-group irreducible-representation alignment.

For Z/p (p prime), the non-trivial irreducible representations are the
1-dimensional characters chi_k(j) = exp(2 pi i k j / p), k = 1..(p-1)/2,
paired with their conjugates. As real subspaces these are exactly the
2D Fourier planes used by F_raw.

The canonical detector defines
    F_raw(t) = max_k  variance_fraction_in_Q_k(E_t),
selecting a single "best" irrep at each step.
A reviewer-style alternative, motivated by Chughtai et al. 2023, would
instead use the *average* fraction explained across all non-trivial
irreps:
    F_irrep(t) = mean_k variance_fraction_in_Q_k(E_t).

This script computes both trajectories on the 8 posthoc cells, applies
the same null-permutation correction (p95) and BIC changepoint detector,
and reports tau_F^max vs tau_F^irrep head-to-head.

Output
------
results/posthoc/r1_irrep_alignment.csv
results/posthoc/r1_irrep_alignment.md

Usage
-----
  python src/analysis/r1_irrep_alignment.py
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from grok_metrics import estimate_changepoint  # noqa: E402

P = 53
N_PERMS = 100
SEED = 20240425


def _build_qbases(p: int):
    """Return list of (p, 2) orthonormal bases for non-trivial irreps."""
    bases = []
    half = (p - 1) // 2
    j = np.arange(p, dtype=float)
    for k in range(1, half + 1):
        c = np.cos(2 * np.pi * k * j / p)
        s = np.sin(2 * np.pi * k * j / p)
        Q = np.column_stack([c, s])
        # Gram-Schmidt: cos / sin already orthogonal for k != 0.
        Q[:, 0] /= np.linalg.norm(Q[:, 0])
        Q[:, 1] /= np.linalg.norm(Q[:, 1])
        bases.append(Q.astype(np.float64))
    return bases


def _frac_per_k(E_centered: np.ndarray, Qs):
    """Return fraction of total Frobenius energy in each Q_k Q_k^T projection."""
    denom = float(np.sum(E_centered * E_centered)) + 1e-15
    fracs = np.empty(len(Qs))
    for i, Q in enumerate(Qs):
        proj = Q @ (Q.T @ E_centered)
        fracs[i] = float(np.sum(proj * proj)) / denom
    return fracs


def _alignment_trajectory(emb_traj: np.ndarray, Qs, top_k: int = 3):
    """emb_traj: (T, p, d).

    Returns (T,) max-frac and (T,) top-K-mean-frac arrays.

    The unrestricted mean over all (p-1)/2 non-trivial irreps is
    mathematically degenerate after centering (sum of variance fractions
    across the full irrep set equals 1 by Parseval, so the mean is the
    constant 2/(p-1)). We therefore use the average of the top-K largest
    irrep fractions as the irrep-based alternative scoring; K=3 contrasts
    with the canonical K=1 (max) without collapsing to the constant.
    """
    T = emb_traj.shape[0]
    f_max = np.zeros(T)
    f_topk = np.zeros(T)
    for t in range(T):
        E = emb_traj[t].astype(np.float64)
        E = E - E.mean(axis=0, keepdims=True)
        fracs = _frac_per_k(E, Qs)
        f_max[t] = float(np.max(fracs))
        f_topk[t] = float(np.mean(np.sort(fracs)[-top_k:]))
    return f_max, f_topk


def _null_p95(emb_traj: np.ndarray, Qs, top_k: int = 3,
              refresh_every=10, n_perms=N_PERMS, rng=None):
    """Compute LOCF p95 null trajectories for max- and mean-fracs.

    refresh_every: refresh the null at every k-th log-time step (the posthoc
    trajectories already use 500-step log spacing; refresh_every=10 means
    refresh every 5,000 training steps to match the canonical detector).
    """
    if rng is None:
        rng = np.random.default_rng(SEED)
    T, p, d = emb_traj.shape
    null_max = np.zeros(T)
    null_mean = np.zeros(T)
    last_max = 0.0
    last_mean = 0.0
    for t in range(T):
        if t % refresh_every == 0:
            E = emb_traj[t].astype(np.float64)
            E = E - E.mean(axis=0, keepdims=True)
            maxs = np.empty(n_perms)
            means = np.empty(n_perms)
            for n in range(n_perms):
                perm = rng.permutation(p)
                E_shuf = E[perm]
                fracs = _frac_per_k(E_shuf, Qs)
                maxs[n] = np.max(fracs)
                means[n] = np.mean(np.sort(fracs)[-top_k:])
            last_max = float(np.percentile(maxs, 95))
            last_mean = float(np.percentile(means, 95))
        null_max[t] = last_max
        null_mean[t] = last_mean
    return null_max, null_mean


def analyse(npz_path: Path, Qs):
    d = np.load(npz_path, allow_pickle=True)
    steps = d["steps"].astype(int)
    emb = d["emb"]
    f_max, f_topk = _alignment_trajectory(emb, Qs)
    null_max, null_topk = _null_p95(emb, Qs)
    fcorr_max = np.maximum(0.0, f_max - null_max)
    fcorr_topk = np.maximum(0.0, f_topk - null_topk)
    tau_max = estimate_changepoint(steps.tolist(), fcorr_max.tolist(),
                                   slope_rel_threshold=0.01)
    tau_topk = estimate_changepoint(steps.tolist(), fcorr_topk.tolist(),
                                    slope_rel_threshold=0.01)
    delta = (tau_topk - tau_max
             if tau_topk is not None and tau_max is not None else None)
    return {"tau_max": tau_max, "tau_top3_irrep": tau_topk, "delta": delta}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--indir", default="results/posthoc")
    args = ap.parse_args()
    indir = Path(args.indir)
    manifest = indir / "manifest.csv"
    rows = list(csv.DictReader(open(manifest, encoding="utf-8-sig")))

    Qs = _build_qbases(P)
    out_csv = indir / "r1_irrep_alignment.csv"
    deltas = []
    n_agree = 0
    n_total = 0
    fields = ["label", "category", "tau_F_max_harmonic",
              "tau_F_top3_irrep", "delta_steps", "agree_within_500"]
    with open(out_csv, "w", newline="", encoding="utf-8") as fcsv:
        w = csv.DictWriter(fcsv, fieldnames=fields)
        w.writeheader()
        for row in rows:
            p = indir / row["file"]
            if not p.exists():
                print(f"  SKIP {row['label']}")
                continue
            print(f"Analysing {row['label']} ({row['category']}) ...")
            res = analyse(p, Qs)
            agree = (res["delta"] is not None
                     and abs(res["delta"]) <= 500)
            n_total += 1
            if agree:
                n_agree += 1
            if res["delta"] is not None:
                deltas.append(res["delta"])
            w.writerow({
                "label": row["label"],
                "category": row["category"],
                "tau_F_max_harmonic":
                    f"{res['tau_max']:.0f}" if res['tau_max'] is not None
                    else "",
                "tau_F_top3_irrep":
                    f"{res['tau_top3_irrep']:.0f}"
                    if res['tau_top3_irrep'] is not None else "",
                "delta_steps":
                    f"{res['delta']:.0f}" if res['delta'] is not None
                    else "",
                "agree_within_500": "Y" if agree else "N",
            })
    print(f"CSV -> {out_csv}")

    out_md = indir / "r1_irrep_alignment.md"
    arr = np.array(deltas) if deltas else np.array([])
    lines = [
        "# R1/R9 - Top-K irrep-mean vs max-harmonic Fourier alignment",
        "",
        "For Z/p the non-trivial irreps are 1-D characters; as real "
        "subspaces these are the 2D Fourier planes already used by F_raw.",
        "The unrestricted mean over all (p-1)/2 irreps is degenerate after "
        "centering: by Parseval the variance fractions sum to 1, so the "
        "mean is the constant 2/(p-1) and carries no information. ",
        "We therefore contrast the canonical max-harmonic scoring (top-1) "
        "against the average of the three most-aligned irrep fractions "
        "(top-3 mean).",
        "",
        f"Cells analysed: {n_total}; tau_F agreement within 500 steps: "
        f"{n_agree}/{n_total}.",
        "",
    ]
    if arr.size:
        lines += [
            f"|delta| (top-3 - max-harmonic) summary across "
            f"{arr.size} cells:",
            f"- median = {float(np.median(np.abs(arr))):.0f} steps",
            f"- max    = {int(np.max(np.abs(arr)))} steps",
            "",
        ]
    lines += [
        "Interpretation:",
        "- If top-1 and top-3 mean agree to within the 500-step "
        "measurement bin for the bulk of cells, the choice of single-vs- "
        "few-harmonic scoring is not load-bearing for tau_F and the G<F "
        "ordering survives within the cyclic-group irrep family.",
        "- A systematic offset (e.g. top-3 consistently later) would "
        "indicate that the canonical max-harmonic detector locks onto "
        "early dominant-harmonic structure that the broader irrep average "
        "does not yet register.",
    ]
    out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"MD  -> {out_md}")


if __name__ == "__main__":
    main()
