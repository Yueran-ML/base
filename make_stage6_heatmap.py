#!/usr/bin/env python3
"""
Generate stage6_2d_heatmap.pdf/.png (Figure 8).

Left panel : G<F rate per (lr, wd) cell, averaged over 3 seeds.
Right panel: majority phase per cell (Grokking / Memorization / Comprehension).
"""
from __future__ import annotations
import csv
import math
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.patches import Patch

BASE  = Path(__file__).parent
PAPER = BASE / "paper"
CSV   = BASE / "runs/stage6_2d/results_stage6_2d.csv"

# ── load ─────────────────────────────────────────────────────────────────────

with open(CSV, encoding="utf-8") as f:
    rows = list(csv.DictReader(f))

lrs = sorted(set(float(r["lr"]) for r in rows))   # 7 values, ascending
wds = sorted(set(float(r["wd"]) for r in rows))   # 7 values, ascending

# ── build matrices ───────────────────────────────────────────────────────────

# For each (lr, wd) cell:
#   gf_rate  = #G<F / #Grokking  (nan if no Grokking)
#   phase    = majority phase
PHASE_ORDER = {"Grokking": 0, "Comprehension": 1,
               "Memorization": 2, "Confusion": 3}
PHASE_COLOR = {
    "Grokking":      "#2196F3",
    "Comprehension": "#4CAF50",
    "Memorization":  "#FF9800",
    "Confusion":     "#9E9E9E",
}

gf_rate  = np.full((len(wds), len(lrs)), float("nan"))
phase_mat = np.zeros((len(wds), len(lrs)), dtype=int)  # 0..3

lr_idx = {v: i for i, v in enumerate(lrs)}
wd_idx = {v: i for i, v in enumerate(wds)}

# group by (lr, wd)
cells: dict = {}
for r in rows:
    key = (float(r["lr"]), float(r["wd"]))
    cells.setdefault(key, []).append(r)

for (lr, wd), cell_rows in cells.items():
    ci = lr_idx[lr]
    ri = wd_idx[wd]

    grok_rows = [r for r in cell_rows if r["observed_phase"] == "Grokking"]
    gf_rows   = [r for r in grok_rows  if r["ordering"] == "G<F"]

    if grok_rows:
        gf_rate[ri, ci] = len(gf_rows) / len(grok_rows)

    # majority phase
    from collections import Counter
    phase_counts = Counter(r["observed_phase"] for r in cell_rows)
    majority = phase_counts.most_common(1)[0][0]
    phase_mat[ri, ci] = PHASE_ORDER.get(majority, 3)

# ── plot ──────────────────────────────────────────────────────────────────────

fig, (axL, axR) = plt.subplots(1, 2, figsize=(12.5, 5.0))

# ----- Left: G<F rate heatmap ------------------------------------------------

cmap_gf = matplotlib.colormaps.get_cmap("RdYlGn").copy()
cmap_gf.set_bad(color="#dddddd")

im = axL.imshow(gf_rate, origin="lower",
                cmap=cmap_gf, vmin=0, vmax=1,
                aspect="auto")

# annotate each cell
for ri in range(len(wds)):
    for ci in range(len(lrs)):
        v = gf_rate[ri, ci]
        if not math.isnan(v):
            txt = f"{v:.0%}"
            col = "white" if v < 0.3 or v > 0.85 else "black"
            axL.text(ci, ri, txt, ha="center", va="center",
                     fontsize=8, color=col, fontweight="bold")
        else:
            axL.text(ci, ri, "—", ha="center", va="center",
                     fontsize=9, color="#999")

# ticks — formatted for readability (avoid raw scientific notation)
def fmt_lr(v):
    return f"{v*1000:.2f}"   # e.g. 0.001024 → "1.02"

def fmt_wd(v):
    return f"{v:.2f}"

axL.set_xticks(range(len(lrs)))
axL.set_xticklabels([fmt_lr(v) for v in lrs], fontsize=8.5)
axL.set_yticks(range(len(wds)))
axL.set_yticklabels([fmt_wd(v) for v in wds], fontsize=8.5)
axL.set_xlabel(r"Learning rate $\times 10^{-3}$", fontsize=11)
axL.set_ylabel("Weight decay", fontsize=11)
axL.set_title("G<F rate per cell\n(avg over 3 seeds)", fontsize=11)

cb = fig.colorbar(im, ax=axL, fraction=0.046, pad=0.04)
cb.set_label("G<F rate", fontsize=10)
cb.ax.tick_params(labelsize=8)

# ----- Right: majority phase heatmap ----------------------------------------

phase_cmap = mcolors.ListedColormap(
    [PHASE_COLOR["Grokking"],
     PHASE_COLOR["Comprehension"],
     PHASE_COLOR["Memorization"],
     PHASE_COLOR["Confusion"]]
)
bounds = [-0.5, 0.5, 1.5, 2.5, 3.5]
norm_phase = mcolors.BoundaryNorm(bounds, phase_cmap.N)

axR.imshow(phase_mat, origin="lower",
           cmap=phase_cmap, norm=norm_phase,
           aspect="auto")

# label phase abbreviations
phase_abbrev = {0: "Grok", 1: "Comp", 2: "Mem", 3: "Conf"}
for ri in range(len(wds)):
    for ci in range(len(lrs)):
        p = phase_mat[ri, ci]
        axR.text(ci, ri, phase_abbrev[p],
                 ha="center", va="center", fontsize=7.5,
                 color="white" if p in (1, 2, 3) else "white",
                 fontweight="bold")

axR.set_xticks(range(len(lrs)))
axR.set_xticklabels([fmt_lr(v) for v in lrs], fontsize=8.5)
axR.set_yticks(range(len(wds)))
axR.set_yticklabels([fmt_wd(v) for v in wds], fontsize=8.5)
axR.set_xlabel(r"Learning rate $\times 10^{-3}$", fontsize=11)
axR.set_ylabel("Weight decay", fontsize=11)
axR.set_title("Majority phase per cell", fontsize=11)

legend_elements = [Patch(facecolor=PHASE_COLOR[p], label=p)
                   for p in ("Grokking", "Comprehension",
                              "Memorization", "Confusion")]
axR.legend(handles=legend_elements, fontsize=8.5,
           loc="lower right", framealpha=0.88)

# ----- save ------------------------------------------------------------------

fig.suptitle("Stage 6: 2D phase diagram  "
             r"(lr $\times$ wd grid, $7{\times}7$, 3 seeds each)",
             fontsize=12, y=1.01)
fig.tight_layout(w_pad=3)

out_pdf = PAPER / "stage6_2d_heatmap.pdf"
out_png = PAPER / "stage6_2d_heatmap.png"
fig.savefig(out_pdf, dpi=200, bbox_inches="tight")
fig.savefig(out_png, dpi=150, bbox_inches="tight")
print("  stage6_2d_heatmap.pdf / .png saved")
plt.close(fig)
