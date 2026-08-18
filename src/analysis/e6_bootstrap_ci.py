#!/usr/bin/env python3
"""
e6_bootstrap_ci.py — E6 post-hoc bootstrap confidence intervals
================================================================
Pure post-hoc CPU analysis — reads existing per-run CSVs from the main
sweeps and computes non-parametric bootstrap 95% CIs for the two
headline summaries used throughout the paper:

    (a) median Δτ = τ_F − τ_gen          (Grokking-phase rows only)
    (b) P(ordering == "G<F")             (all phases; structural-order rate)

Per source we report:
    n_total, n_grokking, n_with_delta
    median Δτ [lo95, hi95]
    mean   Δτ [lo95, hi95]
    P(G<F)    [lo95, hi95]
    P(F<G)    [lo95, hi95]
    P(F_only) [lo95, hi95]   (memorization with Fourier but no τ_gen)

Sources (auto-detected; missing sources are skipped):
    results/stage2_wd/results.csv             — wd sweep @ lr=1.6e-3
    results/stage3_lr/results.csv             — lr sweep @ wd=2.5
    results/stage6_2d/results_stage6_2d.csv   — 2D (lr,wd) grid
    results/step2_circuit/results_step2_circuit.csv
                                              — three-way ordering (τ_circuit)
    results/e4/results.csv                    — decoupled weight-decay ablation

Outputs
-------
    results/posthoc/e6_bootstrap_ci.csv       — machine-readable table
    results/posthoc/e6_bootstrap_ci.md        — paper-ready markdown table

Usage
-----
    python src/analysis/e6_bootstrap_ci.py
    python src/analysis/e6_bootstrap_ci.py --n-boot 2000 --seed 0
    python src/analysis/e6_bootstrap_ci.py --sources stage2_wd stage3_lr
"""

from __future__ import annotations

import argparse
import csv
import math
import sys
from pathlib import Path
from typing import Iterable

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
RESULTS_DIR = REPO_ROOT / "results"
OUT_DIR = RESULTS_DIR / "posthoc"

# ---------------------------------------------------------------------------
# Source registry
# ---------------------------------------------------------------------------
SOURCES: dict[str, dict] = {
    "stage2_wd": {
        "path": RESULTS_DIR / "stage2_wd" / "results.csv",
        "label": "Stage 2 — wd sweep (lr=1.6e-3)",
    },
    "stage3_lr": {
        "path": RESULTS_DIR / "stage3_lr" / "results.csv",
        "label": "Stage 3 — lr sweep (wd=2.5)",
    },
    "stage6_2d": {
        "path": RESULTS_DIR / "stage6_2d" / "results_stage6_2d.csv",
        "label": "Stage 6 — 2D (lr, wd) grid",
    },
    "step2_circuit": {
        "path": RESULTS_DIR / "step2_circuit" / "results_step2_circuit.csv",
        "label": "Step 2 — circuit-formation three-way",
    },
    "e4": {
        "path": RESULTS_DIR / "e4" / "results.csv",
        "label": "E4 — decoupled weight-decay ablation",
    },
}


# ---------------------------------------------------------------------------
# IO helpers
# ---------------------------------------------------------------------------
def _read_rows(path: Path) -> list[dict]:
    with path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        return [r for r in reader if any((v or "").strip() for v in r.values())]


def _maybe_float(x: str | None) -> float | None:
    if x is None:
        return None
    s = x.strip()
    if not s:
        return None
    try:
        v = float(s)
    except ValueError:
        return None
    if math.isnan(v):
        return None
    return v


# ---------------------------------------------------------------------------
# Bootstrap primitives
# ---------------------------------------------------------------------------
def bootstrap_stat(
    values: np.ndarray,
    stat_fn,
    n_boot: int,
    rng: np.random.Generator,
    ci: float = 0.95,
) -> tuple[float, float, float]:
    """Return (point_estimate, lo, hi) for the given statistic."""
    n = values.size
    if n == 0:
        return (float("nan"), float("nan"), float("nan"))
    point = float(stat_fn(values))
    if n == 1:
        return (point, point, point)
    idx = rng.integers(0, n, size=(n_boot, n))
    resamples = values[idx]
    stats = np.apply_along_axis(stat_fn, 1, resamples)
    alpha = (1.0 - ci) / 2.0
    lo = float(np.quantile(stats, alpha))
    hi = float(np.quantile(stats, 1.0 - alpha))
    return (point, lo, hi)


def bootstrap_proportion(
    indicator: np.ndarray,
    n_boot: int,
    rng: np.random.Generator,
    ci: float = 0.95,
) -> tuple[float, float, float]:
    """Bootstrap CI for a Bernoulli proportion (indicator ∈ {0,1})."""
    n = indicator.size
    if n == 0:
        return (float("nan"), float("nan"), float("nan"))
    point = float(indicator.mean())
    if n == 1:
        return (point, point, point)
    idx = rng.integers(0, n, size=(n_boot, n))
    means = indicator[idx].mean(axis=1)
    alpha = (1.0 - ci) / 2.0
    return (point, float(np.quantile(means, alpha)), float(np.quantile(means, 1.0 - alpha)))


# ---------------------------------------------------------------------------
# Per-source summarization
# ---------------------------------------------------------------------------
def summarize_source(
    key: str,
    rows: list[dict],
    n_boot: int,
    rng: np.random.Generator,
) -> dict:
    phases = [(r.get("observed_phase") or "").strip() for r in rows]
    # Some sources (e.g. step2_circuit) name the τ_gen-vs-τ_F ordering
    # column `ordering_gf` because they also record a three-way ordering.
    def _ordering(r: dict) -> str:
        for k in ("ordering", "ordering_gf"):
            v = (r.get(k) or "").strip()
            if v:
                return v
        return ""

    orderings = [_ordering(r) for r in rows]

    n_total = len(rows)
    n_grokking = sum(1 for p in phases if p == "Grokking")
    n_memorization = sum(1 for p in phases if p == "Memorization")

    # Δτ only defined for rows with both τ_gen and τ_F
    deltas: list[float] = []
    tau_gens: list[float] = []
    tau_Fs: list[float] = []
    for r in rows:
        tg = _maybe_float(r.get("tau_gen"))
        tf = _maybe_float(r.get("tau_F"))
        # Prefer pre-computed 'delta' if present (stage2/stage3/stage6/e4),
        # fall back to tau_F - tau_gen otherwise.
        d = _maybe_float(r.get("delta"))
        if d is None and r.get("delta_gf") is not None:
            d = _maybe_float(r.get("delta_gf"))
        if tg is not None and tf is not None:
            if d is None:
                d = tf - tg
            deltas.append(d)
            tau_gens.append(tg)
            tau_Fs.append(tf)

    delta_arr = np.array(deltas, dtype=float)
    tg_arr = np.array(tau_gens, dtype=float)
    tf_arr = np.array(tau_Fs, dtype=float)

    # Ordering indicators over all rows (reflects paper's "P(G<F) across cells")
    order_arr = np.array(orderings)
    ind_gf = (order_arr == "G<F").astype(float)
    ind_fg = (order_arr == "F<G").astype(float)
    ind_fonly = (order_arr == "F_only").astype(float)
    ind_gonly = (order_arr == "G_only").astype(float)
    ind_none = (order_arr == "none").astype(float)

    median_delta = bootstrap_stat(delta_arr, np.median, n_boot, rng)
    mean_delta = bootstrap_stat(delta_arr, np.mean, n_boot, rng)
    median_tau_gen = bootstrap_stat(tg_arr, np.median, n_boot, rng)
    median_tau_F = bootstrap_stat(tf_arr, np.median, n_boot, rng)

    p_gf = bootstrap_proportion(ind_gf, n_boot, rng)
    p_fg = bootstrap_proportion(ind_fg, n_boot, rng)
    p_fonly = bootstrap_proportion(ind_fonly, n_boot, rng)
    p_gonly = bootstrap_proportion(ind_gonly, n_boot, rng)
    p_none = bootstrap_proportion(ind_none, n_boot, rng)

    return {
        "source": key,
        "label": SOURCES[key]["label"],
        "n_total": n_total,
        "n_grokking": n_grokking,
        "n_memorization": n_memorization,
        "n_with_delta": int(delta_arr.size),
        "median_delta": median_delta,
        "mean_delta": mean_delta,
        "median_tau_gen": median_tau_gen,
        "median_tau_F": median_tau_F,
        "p_gf": p_gf,
        "p_fg": p_fg,
        "p_fonly": p_fonly,
        "p_gonly": p_gonly,
        "p_none": p_none,
    }


# ---------------------------------------------------------------------------
# Output writers
# ---------------------------------------------------------------------------
def _fmt_ci(tpl: tuple[float, float, float], decimals: int = 0) -> str:
    pt, lo, hi = tpl
    if not np.isfinite(pt):
        return "n/a"
    if decimals <= 0:
        return f"{pt:.0f} [{lo:.0f}, {hi:.0f}]"
    return f"{pt:.{decimals}f} [{lo:.{decimals}f}, {hi:.{decimals}f}]"


def _fmt_prop(tpl: tuple[float, float, float]) -> str:
    pt, lo, hi = tpl
    if not np.isfinite(pt):
        return "n/a"
    return f"{pt:.2f} [{lo:.2f}, {hi:.2f}]"


def write_csv(summaries: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "source",
        "label",
        "n_total",
        "n_grokking",
        "n_memorization",
        "n_with_delta",
        "median_delta", "median_delta_lo95", "median_delta_hi95",
        "mean_delta", "mean_delta_lo95", "mean_delta_hi95",
        "median_tau_gen", "median_tau_gen_lo95", "median_tau_gen_hi95",
        "median_tau_F", "median_tau_F_lo95", "median_tau_F_hi95",
        "p_gf", "p_gf_lo95", "p_gf_hi95",
        "p_fg", "p_fg_lo95", "p_fg_hi95",
        "p_fonly", "p_fonly_lo95", "p_fonly_hi95",
        "p_gonly", "p_gonly_lo95", "p_gonly_hi95",
        "p_none", "p_none_lo95", "p_none_hi95",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for s in summaries:
            row = {
                "source": s["source"],
                "label": s["label"],
                "n_total": s["n_total"],
                "n_grokking": s["n_grokking"],
                "n_memorization": s["n_memorization"],
                "n_with_delta": s["n_with_delta"],
            }
            for key in ("median_delta", "mean_delta", "median_tau_gen",
                        "median_tau_F", "p_gf", "p_fg", "p_fonly",
                        "p_gonly", "p_none"):
                pt, lo, hi = s[key]
                row[key] = f"{pt:.4f}"
                row[f"{key}_lo95"] = f"{lo:.4f}"
                row[f"{key}_hi95"] = f"{hi:.4f}"
            w.writerow(row)


def write_markdown(summaries: list[dict], path: Path, n_boot: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines: list[str] = []
    lines.append(f"# E6 — Bootstrap 95% CIs (n_boot = {n_boot})")
    lines.append("")
    lines.append(
        "Percentile bootstrap over per-run rows. Δτ = τ_F − τ_gen "
        "computed only for rows where both changepoints are detected; "
        "proportions are computed over *all* rows in the source."
    )
    lines.append("")

    # Main table — Δτ and P(G<F)
    lines.append("## Δτ and ordering rates")
    lines.append("")
    lines.append(
        "| source | n | n_grok | n_Δ | median Δτ (steps) | mean Δτ (steps) | P(G<F) | P(F<G) | P(F_only) |"
    )
    lines.append("| --- | ---: | ---: | ---: | --- | --- | --- | --- | --- |")
    for s in summaries:
        lines.append(
            "| {label} | {n} | {ng} | {nd} | {md} | {mu} | {pgf} | {pfg} | {pfo} |".format(
                label=s["label"],
                n=s["n_total"],
                ng=s["n_grokking"],
                nd=s["n_with_delta"],
                md=_fmt_ci(s["median_delta"]),
                mu=_fmt_ci(s["mean_delta"]),
                pgf=_fmt_prop(s["p_gf"]),
                pfg=_fmt_prop(s["p_fg"]),
                pfo=_fmt_prop(s["p_fonly"]),
            )
        )
    lines.append("")

    # Secondary table — τ locations
    lines.append("## Timing medians")
    lines.append("")
    lines.append("| source | median τ_gen (steps) | median τ_F (steps) |")
    lines.append("| --- | --- | --- |")
    for s in summaries:
        lines.append(
            "| {label} | {tg} | {tf} |".format(
                label=s["label"],
                tg=_fmt_ci(s["median_tau_gen"]),
                tf=_fmt_ci(s["median_tau_F"]),
            )
        )
    lines.append("")

    path.write_text("\n".join(lines), encoding="utf-8")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def main(argv: Iterable[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n-boot", type=int, default=1000,
                    help="number of bootstrap resamples (default 1000)")
    ap.add_argument("--seed", type=int, default=0,
                    help="RNG seed for reproducibility")
    ap.add_argument("--sources", nargs="+", default=None,
                    choices=list(SOURCES.keys()),
                    help="subset of sources to summarise (default: all available)")
    ap.add_argument("--out-csv", type=Path, default=OUT_DIR / "e6_bootstrap_ci.csv")
    ap.add_argument("--out-md", type=Path, default=OUT_DIR / "e6_bootstrap_ci.md")
    args = ap.parse_args(list(argv) if argv is not None else None)

    rng = np.random.default_rng(args.seed)
    requested = args.sources or list(SOURCES.keys())

    summaries: list[dict] = []
    for key in requested:
        info = SOURCES[key]
        path = info["path"]
        if not path.exists():
            print(f"[skip] {key}: {path} not found")
            continue
        rows = _read_rows(path)
        if not rows:
            print(f"[skip] {key}: CSV is empty")
            continue
        summary = summarize_source(key, rows, args.n_boot, rng)
        summaries.append(summary)
        print(
            f"[ok]   {key}: n={summary['n_total']} "
            f"n_grok={summary['n_grokking']} n_Δ={summary['n_with_delta']} "
            f"median Δτ={_fmt_ci(summary['median_delta'])} "
            f"P(G<F)={_fmt_prop(summary['p_gf'])}"
        )

    if not summaries:
        print("No sources found — nothing to write.", file=sys.stderr)
        return 1

    write_csv(summaries, args.out_csv)
    write_markdown(summaries, args.out_md, args.n_boot)
    print(f"[wrote] {args.out_csv}")
    print(f"[wrote] {args.out_md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
