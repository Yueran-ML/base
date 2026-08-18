#!/usr/bin/env python3
"""
Q8 — Does τ_F track embedding-layer geometry consolidation?

For each cell trajectory we compute four geometry metrics at every log step:

  1. Frobenius norm             ||E_t||_F
  2. Participation ratio (PR)   PR = (Σσ_i²)² / Σσ_i⁴          where σ = SVs of E_t-E̅_t
  3. Effective rank (e^S)       e^S with S = -Σ p_i log p_i, p_i = σ_i²/Σσ_j²  (Roy & Vetterli 2007)
  4. Mean cosine anisotropy     A = mean_{i<j} |cos(E_i, E_j)|  (smaller ⇒ more isotropic)

We additionally detect a geometry changepoint τ_geom for each metric
(BIC + EMA, same as τ_F) so we can compare τ_geom vs τ_F run-by-run.

Output:
  results/posthoc/q8_geometry.csv
  results/posthoc/q8_geometry_trajectories.png   (4 panels per cell)
  results/posthoc/q8_geometry_timing.md           (τ_geom alignment table)

Usage:
  python src/analysis/q8_embedding_geometry.py
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

EMA_ALPHA = 0.15
SLOPE_THRESH = 0.01


def _frob_norm(E: np.ndarray) -> float:
    return float(np.linalg.norm(E))


def _participation_ratio(E: np.ndarray) -> float:
    Ec = E - E.mean(0)
    _, s, _ = np.linalg.svd(Ec, full_matrices=False)
    s2 = s ** 2
    denom = float(np.sum(s2 ** 2))
    if denom < 1e-20:
        return 0.0
    return float((float(np.sum(s2)) ** 2) / denom)


def _effective_rank(E: np.ndarray) -> float:
    """e^S with S the Shannon entropy of the normalised squared SV spectrum."""
    Ec = E - E.mean(0)
    _, s, _ = np.linalg.svd(Ec, full_matrices=False)
    s2 = s ** 2
    total = float(np.sum(s2))
    if total < 1e-20:
        return 0.0
    p = s2 / total
    p = p[p > 1e-20]
    S = -float(np.sum(p * np.log(p)))
    return float(np.exp(S))


def _anisotropy(E: np.ndarray) -> float:
    """Mean |cos| between distinct row pairs after centering."""
    Ec = E - E.mean(0)
    norms = np.linalg.norm(Ec, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-12)
    Un = Ec / norms
    G = Un @ Un.T
    n = G.shape[0]
    mask = ~np.eye(n, dtype=bool)
    return float(np.mean(np.abs(G[mask])))


METRICS = [
    ("frob_norm",          _frob_norm),
    ("participation_ratio", _participation_ratio),
    ("eff_rank",           _effective_rank),
    ("anisotropy",         _anisotropy),
]


def _ema(values: np.ndarray, alpha: float) -> np.ndarray:
    if alpha >= 1.0:
        return values.copy()
    out = np.zeros_like(values)
    out[0] = values[0]
    for i in range(1, len(values)):
        out[i] = alpha * values[i] + (1 - alpha) * out[i - 1]
    return out


def _changepoint_signed(steps: np.ndarray, values: np.ndarray) -> float | None:
    """
    estimate_changepoint in grok_metrics accepts the raw signal.  Here our
    signals may decrease over training (e.g. anisotropy), so we run the
    detector on both sign-variants and take whichever yields a valid break.
    """
    vs = _ema(values, EMA_ALPHA).tolist()
    tau_up = estimate_changepoint(steps.tolist(), vs,
                                  slope_rel_threshold=SLOPE_THRESH)
    if tau_up is not None:
        return tau_up
    vs_neg = (-np.asarray(vs)).tolist()
    return estimate_changepoint(steps.tolist(), vs_neg,
                                slope_rel_threshold=SLOPE_THRESH)


def analyse_one(npz_path: Path) -> dict:
    data = np.load(npz_path, allow_pickle=True)
    steps = data["steps"].astype(int)
    emb_traj = data["emb"]            # (T, p, d)
    test_acc = data["test_acc"].astype(float)
    f_raw = data["f_raw"].astype(float)
    f_null = data["f_null_p95"].astype(float)
    T = len(steps)

    tau_gen = find_tau_sustained(steps.tolist(), test_acc.tolist(),
                                 threshold=0.9, n_sustained=3)
    f_corr = np.maximum(0.0, f_raw - f_null)
    tau_F = estimate_changepoint(steps.tolist(),
                                 _ema(f_corr, EMA_ALPHA).tolist(),
                                 slope_rel_threshold=SLOPE_THRESH)

    metric_values: dict[str, np.ndarray] = {
        name: np.zeros(T) for name, _ in METRICS
    }
    for t in range(T):
        E = emb_traj[t]
        for name, fn in METRICS:
            metric_values[name][t] = fn(E)

    tau_geom: dict[str, float | None] = {}
    for name in metric_values:
        tau_geom[name] = _changepoint_signed(steps, metric_values[name])

    return {
        "steps": steps,
        "tau_gen": tau_gen,
        "tau_F": tau_F,
        "f_corr": f_corr,
        "metrics": metric_values,
        "tau_geom": tau_geom,
    }


def plot_cells(cells_info: dict[str, dict], outpath: Path) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("  [plot] matplotlib not available, skipping")
        return

    n_cells = len(cells_info)
    fig, axes = plt.subplots(n_cells, 4, figsize=(16, 2.6 * n_cells),
                             sharex=False)
    if n_cells == 1:
        axes = axes[None, :]

    metric_names = [m[0] for m in METRICS]
    for i, (label, info) in enumerate(cells_info.items()):
        steps = info["steps"]
        for j, name in enumerate(metric_names):
            ax = axes[i][j]
            ax.plot(steps, info["metrics"][name], color="#4477AA",
                    linewidth=1.3)
            if info["tau_gen"] is not None:
                ax.axvline(info["tau_gen"], color="green", linestyle="--",
                           alpha=0.7, linewidth=1, label=r"$\tau_{gen}$")
            if info["tau_F"] is not None:
                ax.axvline(info["tau_F"], color="red", linestyle="--",
                           alpha=0.7, linewidth=1, label=r"$\tau_F$")
            tg = info["tau_geom"].get(name)
            if tg is not None:
                ax.axvline(tg, color="orange", linestyle=":",
                           alpha=0.8, linewidth=1.3,
                           label=r"$\tau_{geom}$")
            ax.set_xscale("log")
            if j == 0:
                ax.set_ylabel(label, fontsize=9)
            if i == 0:
                ax.set_title(name, fontsize=10)
            ax.grid(True, alpha=0.3)
            if i == 0 and j == 0:
                ax.legend(fontsize=7, loc="best")

    fig.suptitle("Q8 — Embedding-layer geometry trajectories (log-step axis)",
                 fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(outpath, dpi=130, bbox_inches="tight")
    import matplotlib.pyplot as plt
    plt.close(fig)
    print(f"  Figure → {outpath}")


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

    cells_info: dict[str, dict] = {}
    for row in cells:
        label = row["label"]
        npz_path = indir / row["file"]
        if not npz_path.exists():
            print(f"  SKIP {label}: {npz_path} missing")
            continue
        print(f"Analysing {label} ...")
        info = analyse_one(npz_path)
        info["category"] = row["category"]
        cells_info[label] = info

    csv_out = indir / "q8_geometry.csv"
    with open(csv_out, "w", newline="", encoding="utf-8") as fc:
        w = csv.writer(fc)
        w.writerow(["cell_label", "category", "tau_gen", "tau_F",
                    "metric", "tau_geom",
                    "delta_geom_vs_F", "delta_geom_vs_gen"])
        for label, info in cells_info.items():
            for metric_name, _ in METRICS:
                tg = info["tau_geom"].get(metric_name)
                d_F = (tg - info["tau_F"]) if (tg is not None and info["tau_F"]) else ""
                d_g = (tg - info["tau_gen"]) if (tg is not None and info["tau_gen"]) else ""
                w.writerow([
                    label, info["category"],
                    info["tau_gen"], info["tau_F"], metric_name,
                    tg if tg is not None else "",
                    f"{d_F:.0f}" if d_F != "" else "",
                    f"{d_g:.0f}" if d_g != "" else "",
                ])
    print(f"\nCSV → {csv_out}")

    # Markdown table: τ_geom alignment
    md = indir / "q8_geometry_timing.md"
    lines = [
        "# Q8 — Geometry changepoint alignment",
        "",
        "τ_geom: BIC changepoint on each geometry metric (EMA α=0.15, "
        "slope ≥ 1%). Δ vs τ_F and τ_gen in training steps.",
        "",
    ]
    header = ["cell (cat)", "τ_gen", "τ_F"] + \
             [f"{m[0]}: τ_geom (Δ vs τ_F / τ_gen)" for m in METRICS]
    lines.append("| " + " | ".join(header) + " |")
    lines.append("|" + "|".join([" --- "] * len(header)) + "|")
    for label, info in cells_info.items():
        row = [f"{label} ({info['category']})",
               str(info["tau_gen"]) if info["tau_gen"] is not None else "–",
               str(info["tau_F"])   if info["tau_F"]   is not None else "–"]
        for name, _ in METRICS:
            tg = info["tau_geom"].get(name)
            if tg is None:
                row.append("–")
            else:
                dF = (f"{tg - info['tau_F']:+.0f}"
                      if info["tau_F"] is not None else "–")
                dg = (f"{tg - info['tau_gen']:+.0f}"
                      if info["tau_gen"] is not None else "–")
                row.append(f"{tg:.0f} ({dF} / {dg})")
        lines.append("| " + " | ".join(row) + " |")
    md.write_text("\n".join(lines), encoding="utf-8")
    print(f"MD  → {md}")

    plot_cells(cells_info, indir / "q8_geometry_trajectories.png")


if __name__ == "__main__":
    main()
