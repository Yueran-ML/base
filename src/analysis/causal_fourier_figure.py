"""Render publication-style figures for the causal Fourier-intervention analysis.

Produces two figures:

  results/causal_probe/causal_intervention_grid.png
      5-row x 4-bar-group panel. Each row is one cell; the four bar groups
      correspond to (pre_all, between_first, between_second, post_all).
      Within each group the bars are:
        base          — uninterveed test accuracy
        I1 keep       — keep top-K Fourier subspace only
        I2 remove     — remove top-K Fourier subspace
        I3 random     — matched-energy random subspace control (mean +/- std)
        I4 embed_rm   — remove top-K Fourier from embedding rows

  results/causal_probe/causal_specificity_heatmap.png
      Cell x regime heatmap of the Fourier-specificity ratio
        (drop_I2) / (drop_I3 + eps)
      annotated with the absolute drop_I2 numbers in white text.

Inputs : results/causal_probe/intervention_results.csv
Output : results/causal_probe/*.png

Usage: .venv/Scripts/python.exe src/analysis/causal_fourier_figure.py
"""

from __future__ import annotations

import csv
import math
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # non-interactive backend
import matplotlib.pyplot as plt
import numpy as np


CSV_PATH = Path("results/causal_probe/intervention_results.csv")
OUTDIR = Path("results/causal_probe")


# Bar styling
BAR_KEYS = [
    ("base_acc", "base", "#666666"),
    ("acc_I1_keep_fourier", "I1 keep top-3 Fourier", "#2E7D32"),
    ("acc_I2_remove_fourier", "I2 remove top-3 Fourier", "#C62828"),
    ("acc_I3_random_ctrl_mean", "I3 random ctrl (matched E)", "#9E9E9E"),
    ("acc_I4_remove_emb_fourier", "I4 remove embed Fourier", "#EF6C00"),
]

# Cell row ordering (top to bottom in figure)
CELL_ORDER = ["strong_CGF", "canonical_GF", "coincident",
              "G_CF_boundary", "FG_slow"]


def _f(s):
    if s in (None, "", "None"):
        return float("nan")
    try:
        return float(s)
    except Exception:
        return float("nan")


def _short_regime(regime: str) -> str:
    """Pretty-print regime labels for x-tick."""
    if regime == "pre_all":
        return "pre-all"
    if regime == "post_all":
        return "post-all"
    # 'between_circ_gen' -> 'circ -> gen'
    s = regime.replace("between_", "")
    parts = s.split("_")
    if len(parts) == 2:
        return f"{parts[0]} -> {parts[1]}"
    return regime


def _load_rows():
    if not CSV_PATH.exists():
        raise FileNotFoundError(f"{CSV_PATH} not found. "
                                "Run causal_fourier_intervention.py first.")
    return list(csv.DictReader(open(CSV_PATH, encoding="utf-8")))


def _by_cell(rows):
    out = defaultdict(list)
    for r in rows:
        out[r["label"]].append(r)
    for k in out:
        out[k].sort(key=lambda r: int(r["step"]))
    return out


def make_grid_figure(by_cell):
    cells = [c for c in CELL_ORDER if c in by_cell]
    n_cells = len(cells)
    fig, axes = plt.subplots(
        n_cells, 1, figsize=(11.5, 2.5 * n_cells + 0.6), sharey=True,
    )
    if n_cells == 1:
        axes = [axes]

    for ax, cell in zip(axes, cells):
        cell_rows = by_cell[cell]
        n_regimes = len(cell_rows)
        n_bars = len(BAR_KEYS)
        x = np.arange(n_regimes)
        bar_w = 0.85 / n_bars

        ordering = cell_rows[0]["ordering"]
        tau_c = _f(cell_rows[0]["tau_circ"])
        tau_g = _f(cell_rows[0]["tau_gen"])
        tau_f = _f(cell_rows[0]["tau_F"])

        for bi, (key, lab, color) in enumerate(BAR_KEYS):
            vals = [_f(r[key]) for r in cell_rows]
            yerr = None
            if key == "acc_I3_random_ctrl_mean":
                yerr = [_f(r["acc_I3_random_ctrl_std"]) for r in cell_rows]
            offset = (bi - (n_bars - 1) / 2) * bar_w
            ax.bar(
                x + offset, vals, bar_w,
                color=color,
                edgecolor="white", linewidth=0.5,
                yerr=yerr,
                error_kw=dict(elinewidth=0.8, ecolor="black", capsize=2),
                label=lab,
            )

        # x-tick labels: regime + step
        xtick_labels = [
            f"{_short_regime(r['regime'])}\nstep {int(r['step'])}"
            for r in cell_rows
        ]
        ax.set_xticks(x)
        ax.set_xticklabels(xtick_labels, fontsize=8)
        ax.set_ylim(0, 1.05)
        ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
        ax.grid(axis="y", linestyle=":", linewidth=0.5, alpha=0.6)

        title = (f"{cell}  ({ordering})    "
                 f"$\\tau_{{\\mathrm{{circ}}}}{{=}}{int(tau_c)}$  "
                 f"$\\tau_{{\\mathrm{{gen}}}}{{=}}{int(tau_g)}$  "
                 f"$\\tau_F{{=}}{int(tau_f)}$")
        ax.set_title(title, fontsize=10, loc="left")
        ax.set_ylabel("test acc", fontsize=9)

    # Shared legend at top
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles, labels, ncol=len(BAR_KEYS), loc="upper center",
        bbox_to_anchor=(0.5, 1.005), frameon=False, fontsize=9,
    )
    fig.suptitle(
        "Causal Fourier-subspace intervention across regime checkpoints",
        fontsize=12, y=1.02, x=0.5,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.99))

    out = OUTDIR / "causal_intervention_grid.png"
    fig.savefig(out, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"  [fig] {out}")


def make_specificity_heatmap(by_cell):
    cells = [c for c in CELL_ORDER if c in by_cell]
    # Build a regime-ordering canonical x-axis: pre_all, between (sorted by
    # step within cell), post_all
    cell_to_regimes: dict[str, list[str]] = {}
    all_regime_short = []
    for c in cells:
        regs = [r["regime"] for r in by_cell[c]]
        cell_to_regimes[c] = regs
        for r in regs:
            short = _short_regime(r)
            if short not in all_regime_short:
                all_regime_short.append(short)

    # Use the natural order seen in the first cell that has them
    canonical_x = ["pre-all"]
    seen = set(canonical_x)
    for c in cells:
        for r in cell_to_regimes[c]:
            s = _short_regime(r)
            if s not in seen and s not in ("pre-all", "post-all"):
                canonical_x.append(s); seen.add(s)
    canonical_x.append("post-all")

    spec = np.full((len(cells), len(canonical_x)), np.nan)
    drop_I2 = np.full((len(cells), len(canonical_x)), np.nan)

    for i, c in enumerate(cells):
        for r in by_cell[c]:
            short = _short_regime(r["regime"])
            if short not in canonical_x:
                continue
            j = canonical_x.index(short)
            base = _f(r["base_acc"])
            d2 = base - _f(r["acc_I2_remove_fourier"])
            d3 = base - _f(r["acc_I3_random_ctrl_mean"])
            spec[i, j] = d2 / max(d3, 1e-3) if d2 > 0 else 0.0
            drop_I2[i, j] = d2

    fig, ax = plt.subplots(figsize=(9.5, 0.85 * len(cells) + 1.4))
    # Use a log-scaled color map for ratio: 1 = matched, >>1 = Fourier-specific
    spec_to_plot = np.where(np.isnan(spec), np.nan, np.log10(np.maximum(spec, 0.1)))
    im = ax.imshow(spec_to_plot, aspect="auto", cmap="RdBu_r",
                   vmin=-1, vmax=2.5)

    ax.set_xticks(np.arange(len(canonical_x)))
    ax.set_xticklabels(canonical_x, fontsize=9, rotation=20, ha="right")
    ax.set_yticks(np.arange(len(cells)))
    ax.set_yticklabels(cells, fontsize=9)
    ax.set_title(
        r"Fourier-specificity ratio   $\log_{10}(\Delta_{I2}/\Delta_{I3})$"
        r"                  (annotated cell text = absolute $\Delta_{I2}$)",
        fontsize=10, loc="left",
    )

    # Annotate cells with absolute drop_I2 (red text on light, white on dark)
    for i in range(len(cells)):
        for j in range(len(canonical_x)):
            if np.isnan(drop_I2[i, j]):
                continue
            v = drop_I2[i, j]
            color = "white" if abs(spec_to_plot[i, j]) > 1.4 else "black"
            ax.text(j, i, f"{v:+.2f}", ha="center", va="center",
                    fontsize=9, color=color)

    cb = fig.colorbar(im, ax=ax, fraction=0.035, pad=0.02)
    cb.set_label("log10(spec ratio)", fontsize=9)
    cb.ax.tick_params(labelsize=8)

    fig.tight_layout()
    out = OUTDIR / "causal_specificity_heatmap.png"
    fig.savefig(out, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"  [fig] {out}")


def main():
    OUTDIR.mkdir(parents=True, exist_ok=True)
    rows = _load_rows()
    by_cell = _by_cell(rows)
    print(f"Loaded {sum(len(v) for v in by_cell.values())} rows "
          f"across {len(by_cell)} cells.")
    make_grid_figure(by_cell)
    make_specificity_heatmap(by_cell)


if __name__ == "__main__":
    main()
