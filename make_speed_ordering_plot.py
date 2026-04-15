#!/usr/bin/env python3
"""
Generate tau_gen vs delta_tau scatter to test speed-dependent ordering hypothesis.

Plots all 55 Grokking runs (Stage 2 + Stage 3) with:
  x-axis: tau_gen (generalization step)
  y-axis: delta_tau = tau_F - tau_gen
  color/marker: ordering (G<F = blue, F<G = red, coincident = grey)

Also adds subtraction pilot as a distinct marker.
"""

import csv
import sys
from pathlib import Path

import numpy as np

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except ImportError:
    print("matplotlib not available")
    sys.exit(1)


def load_csv(path):
    if not Path(path).exists():
        return []
    with open(path) as f:
        return list(csv.DictReader(f))


def main():
    base = Path(__file__).parent

    # Load Stage 2 and Stage 3
    s2  = load_csv(base / "runs/stage2_wd/results.csv")
    s3  = load_csv(base / "runs/stage3_lr/results.csv")
    sub = load_csv(base / "runs/stage4_sub/results_stage4_sub.csv")
    mul_dlog = load_csv(base / "runs/stage5_mul_dlog/results_stage5_mul_dlog.csv")
    p97 = load_csv(base / "runs/stage5_p97/results_stage5_p97.csv")

    # Filter Grokking runs with both tau_gen and tau_F detected
    def parse_row(r, stage_label):
        try:
            phase = r.get("observed_phase", r.get("phase", ""))
            if phase != "Grokking":
                return None
            tau_gen = float(r["tau_gen"]) if r["tau_gen"] not in ("None", "", None) else None
            tau_F = float(r["tau_F"]) if r["tau_F"] not in ("None", "", None) else None
            ordering = r.get("ordering", "none")
            delta = float(r["delta"]) if r.get("delta") not in ("None", "", None) else None
            return {
                "stage": stage_label,
                "tau_gen": tau_gen,
                "tau_F": tau_F,
                "delta": delta,
                "ordering": ordering,
            }
        except (ValueError, KeyError):
            return None

    s2_rows      = [x for r in s2       if (x := parse_row(r, "S2-add"))      is not None]
    s3_rows      = [x for r in s3       if (x := parse_row(r, "S3-add"))      is not None]
    sub_rows     = [x for r in sub      if (x := parse_row(r, "S4-sub"))      is not None]
    mul_dlog_rows= [x for r in mul_dlog if (x := parse_row(r, "S5-mul-dlog")) is not None]
    p97_rows     = [x for r in p97      if (x := parse_row(r, "S5-p97"))      is not None]

    all_add = s2_rows + s3_rows

    print(f"S2 add Grokking: {len(s2_rows)}, S3 add Grokking: {len(s3_rows)}")
    print(f"Sub Grokking: {len(sub_rows)}")
    print(f"Mul-dlog Grokking: {len(mul_dlog_rows)}, p97 Grokking: {len(p97_rows)}")

    # Separate by ordering
    def split_ordering(rows):
        gf = [(r["tau_gen"], r["delta"]) for r in rows
              if r["ordering"] == "G<F" and r["tau_gen"] and r["delta"]]
        fg = [(r["tau_gen"], r["delta"]) for r in rows
              if r["ordering"] == "F<G" and r["tau_gen"] and r["delta"]]
        co = [(r["tau_gen"], r["delta"]) for r in rows
              if r["ordering"] == "coincident" and r["tau_gen"] and r["delta"] is not None]
        return gf, fg, co

    gf_add, fg_add, co_add = split_ordering(all_add)
    gf_sub, fg_sub, _      = split_ordering(sub_rows)
    gf_mul, fg_mul, _      = split_ordering(mul_dlog_rows)
    gf_p97, fg_p97, _      = split_ordering(p97_rows)

    fig, ax = plt.subplots(figsize=(8, 5.5))

    # Addition G<F
    if gf_add:
        xs, ys = zip(*gf_add)
        ax.scatter([x/1000 for x in xs], [y/1000 for y in ys],
                   c="steelblue", marker="o", s=50, alpha=0.7,
                   label=f"Addition G<F (n={len(gf_add)})", zorder=3)

    # Addition F<G
    if fg_add:
        xs, ys = zip(*fg_add)
        ax.scatter([x/1000 for x in xs], [y/1000 for y in ys],
                   c="crimson", marker="*", s=200, alpha=0.9,
                   label=f"Addition F<G (n={len(fg_add)})", zorder=4)

    # Addition coincident
    if co_add:
        xs, ys = zip(*co_add)
        ax.scatter([x/1000 for x in xs], [y/1000 for y in ys],
                   c="grey", marker="D", s=60, alpha=0.7,
                   label=f"Addition coincident (n={len(co_add)})", zorder=3)

    # Subtraction F<G
    if fg_sub:
        xs, ys = zip(*fg_sub)
        ax.scatter([x/1000 for x in xs], [y/1000 for y in ys],
                   c="darkorange", marker="^", s=150, alpha=0.9,
                   label=f"Subtraction F<G (n={len(fg_sub)}, pilot)", zorder=4)

    # Subtraction G<F (if any)
    if gf_sub:
        xs, ys = zip(*gf_sub)
        ax.scatter([x/1000 for x in xs], [y/1000 for y in ys],
                   c="seagreen", marker="^", s=150, alpha=0.9,
                   label=f"Subtraction G<F (n={len(gf_sub)})", zorder=4)

    # Mul-dlog G<F
    if gf_mul:
        xs, ys = zip(*gf_mul)
        ax.scatter([x/1000 for x in xs], [y/1000 for y in ys],
                   c="purple", marker="s", s=80, alpha=0.7,
                   label=f"Mul (dlog) G<F (n={len(gf_mul)})", zorder=3)

    # Mul-dlog F<G
    if fg_mul:
        xs, ys = zip(*fg_mul)
        ax.scatter([x/1000 for x in xs], [y/1000 for y in ys],
                   c="purple", marker="X", s=150, alpha=0.9,
                   label=f"Mul (dlog) F<G (n={len(fg_mul)})", zorder=4)

    # p97 G<F
    if gf_p97:
        xs, ys = zip(*gf_p97)
        ax.scatter([x/1000 for x in xs], [y/1000 for y in ys],
                   c="teal", marker="D", s=60, alpha=0.7,
                   label=f"Add p=97 G<F (n={len(gf_p97)})", zorder=3)

    # p97 F<G
    if fg_p97:
        xs, ys = zip(*fg_p97)
        ax.scatter([x/1000 for x in xs], [y/1000 for y in ys],
                   c="teal", marker="X", s=150, alpha=0.9,
                   label=f"Add p=97 F<G (n={len(fg_p97)})", zorder=4)

    # Reference line: delta=0
    ax.axhline(0, color="black", lw=1.0, ls="--", alpha=0.5, label="Δτ=0 (coincident)")

    # Shade regions
    xlim = ax.get_xlim() if ax.get_xlim() != (0, 1) else (0, 80)
    ax.axhspan(-100, 0, alpha=0.04, color="crimson")  # F<G region
    ax.axhspan(0, 100, alpha=0.04, color="steelblue")  # G<F region

    ax.set_xlabel(r"$\tau_\mathrm{gen}$ (k steps)", fontsize=12)
    ax.set_ylabel(r"$\Delta\tau = \tau_F - \tau_\mathrm{gen}$ (k steps)", fontsize=12)
    ax.set_title("Speed-Dependent Ordering Hypothesis\n"
                 r"Fast grokking ($\tau_\mathrm{gen}$ small) → G<F; "
                 r"Slow grokking → F<G", fontsize=11)
    ax.legend(fontsize=9, loc="upper right")
    ax.set_xlim(left=0)

    # Annotate G<F region label
    ymax = ax.get_ylim()[1]
    ymin = ax.get_ylim()[0]
    ax.text(0.02, 0.97, "G<F regime", transform=ax.transAxes,
            color="steelblue", fontsize=10, va="top", alpha=0.7)
    ax.text(0.02, 0.03, "F<G regime", transform=ax.transAxes,
            color="crimson", fontsize=10, va="bottom", alpha=0.7)

    plt.tight_layout()
    out = base / "paper" / "speed_ordering_scatter.png"
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"Saved: {out}")

    # Also print summary stats for the hypothesis
    all_with_both = [(r["tau_gen"], r["delta"], r["ordering"])
                     for r in all_add + sub_rows + mul_dlog_rows + p97_rows
                     if r["tau_gen"] and r["delta"] is not None]
    if all_with_both:
        tau_gens = np.array([x[0] for x in all_with_both])
        deltas = np.array([x[1] for x in all_with_both])
        corr = np.corrcoef(tau_gens, deltas)[0, 1]
        print(f"\nCorrelation τ_gen vs Δτ: r={corr:.3f} (n={len(all_with_both)})")
        print(f"  G<F runs: tau_gen range {min(r[0] for r in all_with_both if r[2]=='G<F')/1000:.1f}k–"
              f"{max(r[0] for r in all_with_both if r[2]=='G<F')/1000:.1f}k")
        fg_runs = [r for r in all_with_both if r[2]=='F<G']
        if fg_runs:
            print(f"  F<G runs: tau_gen range {min(r[0] for r in fg_runs)/1000:.1f}k–"
                  f"{max(r[0] for r in fg_runs)/1000:.1f}k")


if __name__ == "__main__":
    main()
