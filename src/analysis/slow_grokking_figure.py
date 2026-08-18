"""Render the slow-grokking targeted-sweep scatter and overlay it on the
existing Stage 2 / Stage 3 / Stage 4 (re-trained) Grokking cells.

Output:
  paper/figures/slow_grokking_scatter.png
  paper/figures/slow_grokking_scatter.pdf

The figure has tau_gen on the x-axis and Delta_tau = tau_F - tau_gen on
the y-axis. The horizontal line at zero separates G<F (above) from F<G
(below). The legend distinguishes the four sources visually.

The intent is to make visible:
  - the targeted slow-grokking sweep (lr in [5e-4, 8e-4], wd=2.5,
    3 seeds * 8 lr = 24 cells) lands entirely above the y=0 line, all
    in [+2500, +19500];
  - the original Stage 2/3 cells overlay on the same axes for context;
  - the slow-grokking margin (tau_gen >= 35,000) is densely populated
    by the new sweep, with no F<G;
  - the two F<G outliers from the original Stage 3 sweep are visible
    on the y<0 side and accompanied by a re-run annotation showing that
    the lr=6.8e-4 seed=7 cell is G<F under cross-hardware re-training.
"""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def _f(s):
    if s in (None, "", "None"):
        return None
    try:
        return float(s)
    except Exception:
        return None


def _load_with_delta(path):
    rows = list(csv.DictReader(open(path, encoding="utf-8")))
    out = []
    for r in rows:
        tg = _f(r.get("tau_gen"))
        tf = _f(r.get("tau_F"))
        if tg is None or tf is None:
            continue
        if r.get("observed_phase") not in (None, "Grokking"):
            continue
        out.append((tg, tf - tg, r))
    return out


def main():
    fig, ax = plt.subplots(figsize=(8.0, 5.2))

    sources = [
        ("Stage 2 (wd sweep, lr=1.6e-3)",
         "results/stage2_wd/results.csv",
         "tab:blue", "o", 36, 0.7),
        ("Stage 3 (lr sweep, wd=2.5)",
         "results/stage3_lr/results.csv",
         "tab:orange", "s", 36, 0.7),
        ("Stage 4 retraining (Stage 2+3 cells)",
         "results/step2_circuit/results_step2_circuit.csv",
         "tab:green", "^", 32, 0.55),
        ("Slow-grokking sweep (this section)",
         "results/slow_grokking/results.csv",
         "crimson", "D", 56, 0.95),
    ]

    for label, path, color, marker, size, alpha in sources:
        if not Path(path).exists():
            print(f"  [skip] {path} missing")
            continue
        data = _load_with_delta(path)
        xs = [d[0] for d in data]
        ys = [d[1] for d in data]
        ax.scatter(xs, ys, s=size, c=color, marker=marker, alpha=alpha,
                   edgecolors="black", linewidths=0.4, label=label)
        print(f"  [{label}] n={len(data)}, "
              f"min Δτ={min(ys):.0f}, max Δτ={max(ys):.0f}")

    ax.axhline(0, color="black", linewidth=1.0, linestyle="--", alpha=0.7)
    ax.axhline(500, color="grey", linewidth=0.5, linestyle=":", alpha=0.5)
    ax.axhline(-500, color="grey", linewidth=0.5, linestyle=":", alpha=0.5)

    # Annotate F<G boundary band
    ax.text(0.985, 0.04, "F<G", transform=ax.transAxes, ha="right",
            color="darkred", fontsize=11, fontweight="bold")
    ax.text(0.985, 0.93, "G<F", transform=ax.transAxes, ha="right",
            color="darkgreen", fontsize=11, fontweight="bold")
    ax.text(0.985, 0.49, r"coincident band ($|\Delta\tau|\leq 500$)",
            transform=ax.transAxes, ha="right", color="grey", fontsize=8)

    # Slow-grokking margin
    ax.axvspan(35_000, 80_000, color="crimson", alpha=0.06, zorder=0,
               label=None)
    ax.text(35_500, ax.get_ylim()[1] * 0.94 if ax.get_ylim()[1] > 0 else 0,
            r"slow-grokking margin" + "\n" +
            r"($\tau_{\mathrm{gen}}\geq 35{,}000$)",
            color="darkred", fontsize=8.5, va="top")

    ax.set_xlabel(r"$\tau_{\mathrm{gen}}$ (steps)", fontsize=11)
    ax.set_ylabel(r"$\Delta\tau = \tau_F - \tau_{\mathrm{gen}}$ (steps)",
                  fontsize=11)
    ax.set_title("Slow-grokking targeted falsification of "
                 "the speed-dependent F<G transition", fontsize=12)
    ax.grid(True, linestyle=":", alpha=0.4)
    ax.legend(loc="lower right", fontsize=8.5, framealpha=0.95)

    plt.tight_layout()
    out_dir = Path("paper/figures")
    out_dir.mkdir(parents=True, exist_ok=True)
    png = out_dir / "slow_grokking_scatter.png"
    pdf = out_dir / "slow_grokking_scatter.pdf"
    plt.savefig(png, dpi=180, bbox_inches="tight")
    plt.savefig(pdf, bbox_inches="tight")
    plt.close(fig)
    print(f"  [fig] {png}")
    print(f"  [fig] {pdf}")


if __name__ == "__main__":
    main()
