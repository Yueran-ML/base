#!/usr/bin/env python3
"""
Generate two paper figures:
  1. timeline_example.pdf  — 3-panel example showing test_acc + F_corr
     with tau_gen / tau_F marked (goes into Results intro)
  2. tau_scatter.pdf       — scatter plot tau_gen vs tau_F for all 55 runs
"""
from __future__ import annotations
import csv, json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

TRAJ_DIR  = Path("runs/sensitivity/trajectories")
STAGE2_CSV = Path("runs/stage2_wd/results.csv")
STAGE3_CSV = Path("runs/stage3_lr/results.csv")
OUT_DIR   = Path("paper")

# ── Figure 1: timeline example ──────────────────────────────────────────────
# Three panels: strong G<F, weak G<F, F<G outlier
# Load trajectories from sensitivity run (50k steps, full f_raw+f_null)

cells = [
    ("wd2.76_s42",  "Strong G<F\n(wd=2.76, seed=42)", "#1a6fb0"),
    ("wd2.45_s7",   "Weak G<F\n(wd=2.45, seed=7)",    "#2ca02c"),
    ("lr6.8e4_s7",  "F<G outlier\n(lr=6.8e-4, seed=7)", "#d62728"),
]

# get tau_gen, tau_F from sensitivity results for canonical config
sens_csv = Path("runs/sensitivity/results.csv")
sens_rows = list(csv.DictReader(sens_csv.read_text(encoding="utf-8").splitlines()))
def get_taus(cell_label):
    rows = [r for r in sens_rows if r["cell_label"] == cell_label
            and r["null_interval"] == "500"
            and r["ema_alpha"]     == "1.0"
            and r["slope_thresh"]  == "0.01"]
    if not rows: return None, None
    r = rows[0]
    tg = float(r["tau_gen"]) if r["tau_gen"] else None
    tf = float(r["tau_F"])   if r["tau_F"]   else None
    return tg, tf

from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset

fig = plt.figure(figsize=(13, 5.5))
gs  = GridSpec(1, 3, figure=fig, wspace=0.42)

all_axes = []
for col, (cell_id, title, color) in enumerate(cells):
    traj_path = TRAJ_DIR / f"{cell_id}.json"
    traj = json.loads(traj_path.read_text(encoding="utf-8"))

    steps     = np.array(traj["steps"])
    test_acc  = np.array(traj["test_acc"])
    train_acc = np.array(traj["train_acc"])
    f_raw     = np.array(traj["f_raw"])
    f_null    = np.array(traj.get("f_null", traj.get("f_null_p95",
                          [0.0]*len(f_raw))))
    f_corr    = np.maximum(0, f_raw - f_null)

    tau_gen, tau_F = get_taus(cell_id)

    ax_acc  = fig.add_subplot(gs[0, col])
    ax_four = ax_acc.twinx()
    all_axes.append((ax_acc, ax_four, steps, f_corr, f_raw, f_null,
                     tau_gen, tau_F, cell_id))

    # accuracy
    ax_acc.plot(steps/1000, test_acc,  color="#e74c3c", lw=2.0,
                label="test acc", zorder=3)
    ax_acc.plot(steps/1000, train_acc, color="#e74c3c", lw=1.0,
                ls="--", alpha=0.4, label="train acc", zorder=2)

    # fourier
    ax_four.plot(steps/1000, f_corr, color="#2980b9", lw=2.0,
                 label=r"$F_\mathrm{corr}$", zorder=3)
    ax_four.plot(steps/1000, f_raw,  color="#2980b9", lw=0.9,
                 ls=":", alpha=0.45, label=r"$F_\mathrm{raw}$", zorder=2)
    ax_four.plot(steps/1000, f_null, color="#95a5a6", lw=0.9,
                 ls="--", alpha=0.6, label="null p95", zorder=1)

    ymax_four = max(f_raw.max(), f_null.max()) * 1.15
    ax_four.set_ylim(0, max(ymax_four, 0.12))

    # vertical markers
    if tau_gen is not None:
        ax_acc.axvline(tau_gen/1000, color="#e74c3c", lw=1.8,
                       ls="--", alpha=0.85, zorder=4)
        ax_acc.text(tau_gen/1000, 0.08, r"$\tau_\mathrm{gen}$",
                    color="#e74c3c", fontsize=9.5, ha="center",
                    va="bottom",
                    bbox=dict(fc="white", ec="none", alpha=0.7, pad=1))
    if tau_F is not None:
        ax_four.axvline(tau_F/1000, color="#2980b9", lw=1.8,
                        ls="--", alpha=0.85, zorder=4)
        ax_four.text(tau_F/1000, ax_four.get_ylim()[1]*0.88,
                     r"$\tau_F$", color="#2980b9", fontsize=9.5,
                     ha="center", va="top",
                     bbox=dict(fc="white", ec="none", alpha=0.7, pad=1))

    ax_acc.set_xlabel("Training steps (×10³)", fontsize=10)
    ax_acc.set_ylabel("Accuracy", color="#e74c3c", fontsize=10)
    ax_four.set_ylabel(r"Fourier alignment ($F_\mathrm{corr}$)",
                       color="#2980b9", fontsize=10)
    ax_acc.tick_params(axis="y", colors="#e74c3c")
    ax_four.tick_params(axis="y", colors="#2980b9")
    ax_acc.set_ylim(-0.02, 1.08)
    ax_acc.set_title(title, fontsize=10.5, pad=4)

    # shade the gap region (tau_gen → tau_F for G<F; tau_F → tau_gen for F<G)
    if tau_gen is not None and tau_F is not None and tau_F > tau_gen:
        ax_acc.axvspan(tau_gen/1000, tau_F/1000, alpha=0.08,
                       color="#27ae60", zorder=0, label=r"$\Delta\tau$ gap")
    elif tau_gen is not None and tau_F is not None and tau_F < tau_gen:
        ax_acc.axvspan(tau_F/1000, tau_gen/1000, alpha=0.08,
                       color="#d62728", zorder=0, label=r"$\Delta\tau$ gap (F$<$G)")

    # legend only on first panel
    if col == 0:
        lines_a, labs_a = ax_acc.get_legend_handles_labels()
        lines_f, labs_f = ax_four.get_legend_handles_labels()
        ax_acc.legend(lines_a + lines_f, labs_a + labs_f,
                      fontsize=7.5, loc="center left",
                      framealpha=0.85)

    # For F<G panel (col==2): add inset zoom showing late-stage F_corr rise
    if col == 2 and tau_F is not None:
        zoom_start = max(0, (tau_F - 8000) / 1000)
        zoom_end   = min(steps[-1], tau_F + 5000) / 1000
        axins = inset_axes(ax_four, width="42%", height="38%",
                           loc="upper left",
                           bbox_to_anchor=ax_four.bbox,
                           bbox_transform=ax_four.transAxes,
                           borderpad=0)
        axins.plot(steps/1000, f_corr, color="#2980b9", lw=1.5)
        axins.axvline(tau_F/1000, color="#2980b9", lw=1.2, ls="--", alpha=0.85)
        if tau_gen is not None:
            axins.axvline(tau_gen/1000, color="#e74c3c", lw=1.2, ls="--", alpha=0.85)
        axins.set_xlim(zoom_start, zoom_end)
        fc_zoom = f_corr[(steps/1000 >= zoom_start) & (steps/1000 <= zoom_end)]
        yhi = max(fc_zoom.max() * 1.3, 0.025)
        axins.set_ylim(-0.002, yhi)
        axins.set_xlabel("steps (×10³)", fontsize=6.5)
        axins.set_ylabel(r"$F_\mathrm{corr}$", fontsize=6.5)
        axins.tick_params(labelsize=6)
        axins.set_title("zoom: late rise", fontsize=6.5, pad=2)
        axins.patch.set_alpha(0.9)

fig.suptitle(
    r"Representative training trajectories: test accuracy and $F_\mathrm{corr}$ "
    r"(dashed verticals = $\tau_\mathrm{gen}$, $\tau_F$; shading = $\Delta\tau$ gap)",
    fontsize=10, y=1.01)

fig.subplots_adjust(wspace=0.42, top=0.92, bottom=0.12)
out = OUT_DIR / "timeline_example.pdf"
fig.savefig(out, dpi=200)
fig.savefig(str(out).replace(".pdf", ".png"), dpi=150)
print(f"Saved {out}")
plt.close(fig)

# ── Figure 2: tau_gen vs tau_F scatter for all 55 Grokking runs ─────────────
# Load from stage2 + stage3 results
def load_grokking(csv_path, stage_label):
    rows = list(csv.DictReader(Path(csv_path).read_text(encoding="utf-8").splitlines()))
    out = []
    for r in rows:
        if r["observed_phase"] == "Grokking" and r["tau_gen"] and r["tau_F"]:
            out.append(dict(
                tg=float(r["tau_gen"]),
                tf=float(r["tau_F"]),
                delta=float(r["delta"]) if r["delta"] else 0,
                stage=stage_label,
            ))
    return out

s2 = load_grokking(STAGE2_CSV, "stage2")
s3 = load_grokking(STAGE3_CSV, "stage3")
all_runs = s2 + s3

tg_arr  = np.array([r["tg"]/1000 for r in all_runs])
tf_arr  = np.array([r["tf"]/1000 for r in all_runs])
col_arr = ["#1a6fb0" if r["stage"]=="stage2" else "#e07b39"
           for r in all_runs]
gf_mask  = np.array([r["delta"] > 500  for r in all_runs])
fg_mask  = np.array([r["delta"] < -500 for r in all_runs])

fig2, ax = plt.subplots(figsize=(5.5, 5.0))

# diagonal (tau_gen == tau_F)
mn, mx = 0, max(tg_arr.max(), tf_arr.max()) * 1.05
ax.plot([mn, mx], [mn, mx], "k--", lw=0.9, alpha=0.4, zorder=0,
        label=r"$\tau_F = \tau_\mathrm{gen}$ (coincident)")

# G<F region label
ax.text(mx*0.55, mx*0.15, r"G$<$F ($\tau_\mathrm{gen} < \tau_F$)",
        fontsize=9, color="#27ae60", alpha=0.8)
ax.text(mx*0.05, mx*0.82, r"F$<$G ($\tau_F < \tau_\mathrm{gen}$)",
        fontsize=9, color="#d62728", alpha=0.8)

# scatter — G<F and coincident runs
gf_or_co = ~fg_mask
sc = ax.scatter(tg_arr[gf_or_co], tf_arr[gf_or_co],
                c=[c for c, m in zip(col_arr, gf_or_co) if m],
                s=45, alpha=0.78, edgecolors="white",
                linewidths=0.4, zorder=3)

# F<G outliers as bright red stars — clearly visible
ax.scatter(tg_arr[fg_mask], tf_arr[fg_mask],
           marker="*", s=260, color="#d62728",
           edgecolors="#8b0000", linewidths=0.8, zorder=5,
           label="F<G runs (★)")

# annotate F<G points
for tg_v, tf_v in zip(tg_arr[fg_mask], tf_arr[fg_mask]):
    ax.annotate("F<G", (tg_v, tf_v),
                textcoords="offset points", xytext=(7, 4),
                fontsize=8, color="#d62728", fontweight="bold")

# legend
from matplotlib.lines import Line2D
legend_elements = [
    Line2D([0],[0], marker="o", color="w", markerfacecolor="#1a6fb0",
           markersize=8, label="Stage 2 (wd sweep)"),
    Line2D([0],[0], marker="o", color="w", markerfacecolor="#e07b39",
           markersize=8, label="Stage 3 (lr sweep)"),
    Line2D([0],[0], marker="*", color="w", markerfacecolor="#d62728",
           markeredgecolor="#8b0000", markeredgewidth=0.8,
           markersize=14, label=r"F$<$G runs ($\bigstar$)"),
    Line2D([0],[0], ls="--", color="k", alpha=0.4,
           label=r"$\tau_F=\tau_\mathrm{gen}$"),
]
ax.legend(handles=legend_elements, fontsize=8.5, loc="lower right")

ax.set_xlabel(r"Generalization onset $\tau_\mathrm{gen}$ (×10³ steps)", fontsize=11)
ax.set_ylabel(r"Fourier geometry onset $\tau_F$ (×10³ steps)", fontsize=11)
ax.set_title(r"All 55 Grokking runs: $\tau_\mathrm{gen}$ vs $\tau_F$", fontsize=11)
ax.set_xlim(mn, mx)
ax.set_ylim(mn, mx)
ax.set_aspect("equal")

# annotate counts
n_total = len(all_runs)
n_gf = gf_mask.sum(); n_fg = fg_mask.sum(); n_co = n_total - n_gf - n_fg
ax.text(0.02, 0.98,
        f"G<F: {n_gf}/{n_total}  |  F<G: {n_fg}/{n_total}  |  coincident: {n_co}/{n_total}",
        transform=ax.transAxes, fontsize=8.5, va="top",
        bbox=dict(fc="white", ec="#cccccc", alpha=0.85, pad=3))

fig2.tight_layout()
out2 = OUT_DIR / "tau_scatter.pdf"
fig2.savefig(out2, dpi=200, bbox_inches="tight")
fig2.savefig(str(out2).replace(".pdf", ".png"), dpi=200, bbox_inches="tight")
print(f"Saved {out2}")
plt.close(fig2)

print("Done.")
