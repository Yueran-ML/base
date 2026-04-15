#!/usr/bin/env python3
"""
Generate Stage-4 / cross-experiment paper figures with clean matplotlib mathtext.

Outputs:
  paper/step2_three_stage_scatter.pdf/.png   — Figure 5
  paper/step2_interval_boxplot.pdf/.png      — Figure 6
  paper/delta_histogram.pdf/.png
  paper/cross_experiment_summary.pdf/.png    — Figure 7 (two panels)
  paper/speed_ordering_scatter.png           — Figure 11 (add+sub only)
"""
from __future__ import annotations
import csv, math
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

BASE  = Path(__file__).parent
PAPER = BASE / "paper"

# ── helpers ─────────────────────────────────────────────────────────────────

def load_csv(rel):
    p = BASE / rel
    if not p.exists():
        print(f"  WARNING: {rel} not found")
        return []
    with open(p, encoding="utf-8") as f:
        return list(csv.DictReader(f))

def fval(r, key):
    v = r.get(key, "")
    try:
        return float(v) if v not in ("", "None", None) else None
    except ValueError:
        return None

def save_fig(fig, stem):
    pdf = PAPER / f"{stem}.pdf"
    png = PAPER / f"{stem}.png"
    fig.savefig(pdf, dpi=200, bbox_inches="tight")
    fig.savefig(png, dpi=150, bbox_inches="tight")
    print(f"  {stem}.pdf / .png saved")
    plt.close(fig)


# ── load Stage-4 circuit data ────────────────────────────────────────────────

rows4 = load_csv("runs/step2_circuit/results_step2_circuit.csv")
grok4 = [r for r in rows4 if r["observed_phase"] == "Grokking"]   # 54 runs

COLOR3 = {
    "C<G<F": "#1a6fb0",
    "C<F<G": "#e07b39",
    "G<C<F": "#d62728",
}
LABEL3 = {
    "C<G<F": r"$C{<}G{<}F$",
    "C<F<G": r"$C{<}F{<}G$",
    "G<C<F": r"$G{<}C{<}F$",
}

# ── Figure 5: Three-stage timing scatter ────────────────────────────────────

fig5a, (axL, axR) = plt.subplots(1, 2, figsize=(10.5, 4.8))

def _scatter_panel(ax, xcol, ycol, xlabel, ylabel):
    """Draw one panel of the three-stage scatter."""
    pts = {}   # ordering → list of (x, y)
    for r in grok4:
        x = fval(r, xcol)
        y = fval(r, ycol)
        if x is None or y is None:
            continue
        order = r.get("ordering_3stage", "C<G<F")
        pts.setdefault(order, ([], []))
        pts[order][0].append(x / 1000)
        pts[order][1].append(y / 1000)

    all_vals = [v for xs, ys in pts.values() for v in xs + ys]
    mn, mx = 0, max(all_vals) * 1.08

    ax.plot([mn, mx], [mn, mx], "k--", lw=0.9, alpha=0.35, zorder=0)

    for order in ("C<G<F", "C<F<G", "G<C<F"):
        if order not in pts:
            continue
        xs, ys = pts[order]
        n = len(xs)
        ax.scatter(xs, ys,
                   c=COLOR3.get(order, "#888"),
                   s=52, alpha=0.78,
                   edgecolors="white", linewidths=0.5,
                   label=f"{LABEL3.get(order, order)}  (n={n})",
                   zorder=3)

    ax.set_xlim(mn, mx)
    ax.set_ylim(mn, mx)
    ax.set_aspect("equal")
    ax.set_xlabel(xlabel, fontsize=11)
    ax.set_ylabel(ylabel, fontsize=11)
    ax.legend(fontsize=8.5, loc="upper left")
    return mn, mx

mn_L, mx_L = _scatter_panel(
    axL, "tau_circuit", "tau_gen",
    r"$\tau_\mathrm{circuit}$  (k steps)",
    r"$\tau_\mathrm{gen}$  (k steps)"
)
mn_R, mx_R = _scatter_panel(
    axR, "tau_gen", "tau_F",
    r"$\tau_\mathrm{gen}$  (k steps)",
    r"$\tau_F$  (k steps)"
)

# stat box — use 3-stage label for C<G (C<G<F + C<F<G = 50/54)
# This includes the 1 tie (tau_circuit=tau_gen), consistent with paper text
n_cg = sum(1 for r in grok4
           if r.get("ordering_3stage") in ("C<G<F", "C<F<G"))
n_gf4 = sum(1 for r in grok4
            if fval(r, "tau_gen") is not None
            and fval(r, "tau_F") is not None
            and fval(r, "tau_gen") < fval(r, "tau_F"))
n = len(grok4)

axL.text(0.03, 0.97,
         f"C<G: {n_cg}/{n} ({100*n_cg/n:.1f}%)",
         transform=axL.transAxes, fontsize=9, va="top",
         bbox=dict(fc="white", ec="#aaaaaa", alpha=0.88, pad=3))

axR.text(0.03, 0.97,
         f"G<F: {n_gf4}/{n} ({100*n_gf4/n:.1f}%)",
         transform=axR.transAxes, fontsize=9, va="top",
         bbox=dict(fc="white", ec="#aaaaaa", alpha=0.88, pad=3))

fig5a.suptitle(
    r"Stage 4 — three-stage timing scatter  ($n{=}54$ Grokking runs)",
    fontsize=12, y=1.01)
fig5a.tight_layout(w_pad=2.5)
save_fig(fig5a, "step2_three_stage_scatter")


# ── Figure 6: Interval boxplot ───────────────────────────────────────────────

dcg  = np.array([fval(r,"delta_cg")/1000 for r in grok4 if fval(r,"delta_cg") is not None])
dgf4 = np.array([(fval(r,"tau_F")-fval(r,"tau_gen"))/1000
                 for r in grok4
                 if fval(r,"tau_F") is not None and fval(r,"tau_gen") is not None])
dcf  = np.array([fval(r,"delta_cf")/1000 for r in grok4 if fval(r,"delta_cf") is not None])

fig6, ax6 = plt.subplots(figsize=(6.2, 4.6))
data6  = [dcg, dgf4, dcf]
labels6 = [
    r"$\tau_\mathrm{gen}{-}\tau_\mathrm{circuit}$" + "\n(C→G)",
    r"$\tau_F{-}\tau_\mathrm{gen}$" + "\n(G→F)",
    r"$\tau_F{-}\tau_\mathrm{circuit}$" + "\n(C→F total)",
]
colors6 = ["#4db8ff", "#ff9955", "#aaaaaa"]

bp = ax6.boxplot(data6, tick_labels=labels6, patch_artist=True,
                 medianprops=dict(color="black", lw=2.0),
                 whiskerprops=dict(lw=1.2),
                 capprops=dict(lw=1.2),
                 flierprops=dict(marker="o", ms=4, alpha=0.5))

for patch, col in zip(bp["boxes"], colors6):
    patch.set_facecolor(col)
    patch.set_alpha(0.65)

ax6.axhline(0, color="black", lw=0.9, ls="--", alpha=0.4)
ax6.set_ylabel("Time interval (k steps)", fontsize=11)
ax6.set_title(
    r"Stage 4 — interval distributions  ($n{=}54$ Grokking runs)",
    fontsize=11)

for i, d in enumerate(data6, 1):
    med = np.median(d)
    ax6.text(i, med + 0.4, f"med={med:.1f}k",
             ha="center", va="bottom", fontsize=8.5)

fig6.tight_layout()
save_fig(fig6, "step2_interval_boxplot")


# ── Delta-tau histogram (Stage 2+3 addition) ─────────────────────────────────

s2 = load_csv("runs/stage2_wd/results.csv")
s3 = load_csv("runs/stage3_lr/results.csv")

deltas_add = np.array([
    fval(r, "delta") / 1000
    for r in s2 + s3
    if r.get("observed_phase") == "Grokking" and fval(r, "delta") is not None
])

figH, axH = plt.subplots(figsize=(6.0, 4.2))
bw = 2.0
bins = np.arange(math.floor(deltas_add.min()) - 1,
                 math.ceil(deltas_add.max()) + 2, bw)

axH.hist(deltas_add[deltas_add >= 0], bins=bins,
         color="#1a6fb0", alpha=0.72, label="G<F")
axH.hist(deltas_add[deltas_add < 0],  bins=bins,
         color="#d62728", alpha=0.72, label="F<G")
axH.axvline(0, color="black", lw=1.0, ls="--", alpha=0.45)
med_add = np.median(deltas_add)
axH.axvline(med_add, color="#27ae60", lw=1.5, ls="-",
            label=f"median = {med_add:.1f}k")

n_gf_a  = int((deltas_add >  0.5).sum())
n_fg_a  = int((deltas_add < -0.5).sum())
n_co_a  = len(deltas_add) - n_gf_a - n_fg_a
axH.text(0.97, 0.97,
         f"G<F: {n_gf_a}  |  F<G: {n_fg_a}  |  coincident: {n_co_a}",
         transform=axH.transAxes, fontsize=9, va="top", ha="right",
         bbox=dict(fc="white", ec="#aaaaaa", alpha=0.88, pad=3))

axH.set_xlabel(r"$\Delta\tau = \tau_F - \tau_\mathrm{gen}$  (k steps)", fontsize=11)
axH.set_ylabel("Count", fontsize=11)
axH.set_title(
    r"$\Delta\tau$ distribution — addition $p{=}53$"
    r"  (Stage 2+3, $n{=}55$ Grokking runs)",
    fontsize=11)
axH.legend(fontsize=9)
figH.tight_layout()
save_fig(figH, "delta_histogram")


# ── Figure 7: Cross-experiment summary — TWO PANELS ─────────────────────────
# Left:  G<F rate per stage
# Right: median delta-tau per stage

experiments = [
    ("Stage 2\n(wd sweep)",    "runs/stage2_wd/results.csv"),
    ("Stage 3\n(lr sweep)",    "runs/stage3_lr/results.csv"),
    ("Stage 5A\n(mul, dlog)",  "runs/stage5_mul_dlog/results_stage5_mul_dlog.csv"),
    ("Stage 5B\n(add p=97)",   "runs/stage5_p97/results_stage5_p97.csv"),
    ("Stage 6\n(2D grid)",     "runs/stage6_2d/results_stage6_2d.csv"),
    # Stage 4-sub omitted: only 2 Grokking runs — too few for a reliable rate
]

exp_names, rates, med_lags = [], [], []
for name, path in experiments:
    rows_e = load_csv(path)
    if not rows_e:
        continue
    grok_e = [r for r in rows_e if r.get("observed_phase") == "Grokking"]
    gf_e   = [r for r in grok_e if r.get("ordering") == "G<F"]
    if len(grok_e) == 0:
        continue
    deltas_e = [fval(r,"delta")/1000 for r in gf_e
                if fval(r,"delta") is not None]
    exp_names.append(name)
    rates.append(100 * len(gf_e) / len(grok_e))
    med_lags.append(np.median(deltas_e) if deltas_e else float("nan"))
    print(f"  {name.split(chr(10))[0]}: {len(gf_e)}/{len(grok_e)} G<F,  "
          f"med Δτ={med_lags[-1]:.1f}k")

x = np.arange(len(exp_names))
fig7, (axA, axB) = plt.subplots(1, 2, figsize=(12, 4.8))

# Left — rate
barsA = axA.bar(x, rates, color="#1a6fb0", alpha=0.75,
                edgecolor="white", linewidth=0.5)
for bar, rate in zip(barsA, rates):
    axA.text(bar.get_x() + bar.get_width()/2, rate + 1.5,
             f"{rate:.0f}%", ha="center", va="bottom", fontsize=9)
axA.axhline(100, color="#27ae60", lw=1.2, ls="--", alpha=0.6)
axA.axhline(85,  color="#e07b39", lw=1.0, ls=":",  alpha=0.5)
axA.set_xticks(x); axA.set_xticklabels(exp_names, fontsize=9.5)
axA.set_ylim(0, 115)
axA.set_ylabel("G<F rate among Grokking runs (%)", fontsize=11)
axA.set_title("G<F ordering rate per stage", fontsize=11)

# Right — median lag
bar_colors = ["#1a6fb0" if not math.isnan(v) else "#cccccc" for v in med_lags]
barsB = axB.bar(x, [v if not math.isnan(v) else 0 for v in med_lags],
                color=bar_colors, alpha=0.75,
                edgecolor="white", linewidth=0.5)
for bar, v in zip(barsB, med_lags):
    if not math.isnan(v):
        axB.text(bar.get_x() + bar.get_width()/2, v + 0.5,
                 f"{v:.0f}k", ha="center", va="bottom", fontsize=9)
axB.set_xticks(x); axB.set_xticklabels(exp_names, fontsize=9.5)
axB.set_ylabel(r"Median $\Delta\tau = \tau_F - \tau_\mathrm{gen}$  (k steps)",
               fontsize=11)
axB.set_title(r"Median $\Delta\tau$ per stage (G<F runs only)", fontsize=11)

fig7.suptitle("Cross-experiment summary", fontsize=12, y=1.01)
fig7.tight_layout(w_pad=3)
save_fig(fig7, "cross_experiment_summary")


# ── Figure 11: Speed-ordering scatter — addition (55) + subtraction (2) ──────
# Use delta-threshold classification (|delta| > 500) so the coincident point
# (S2 wd=3.107 seed=2025, delta=0, CSV shows ordering=F<G) is shown correctly.
# Correlation computed over all n=57 runs → r ≈ -0.62 (matches limitation text).

sub_raw = load_csv("runs/stage4_sub/results_stage4_sub.csv")

def classify_delta(delta):
    """Return 'G<F', 'F<G', or 'coincident' based on delta value."""
    if delta is None:
        return None
    if delta > 500:
        return "G<F"
    if delta < -500:
        return "F<G"
    return "coincident"

def parse_row_speed(r):
    if r.get("observed_phase") != "Grokking":
        return None
    tg  = fval(r, "tau_gen")
    dt  = fval(r, "delta")
    if tg is None or dt is None:
        return None
    # classify using raw-step delta (threshold 500 steps)
    return tg / 1000, dt / 1000, classify_delta(dt)

# parse with corrected classification
add_pts = [parse_row_speed(r) for r in s2 + s3]
add_pts = [p for p in add_pts if p is not None]
sub_pts = [parse_row_speed(r) for r in sub_raw]
sub_pts = [p for p in sub_pts if p is not None]

fig11, ax11 = plt.subplots(figsize=(7.5, 5.2))

add_gf = [(tg, dt) for tg, dt, o in add_pts if o == "G<F"]
add_fg = [(tg, dt) for tg, dt, o in add_pts if o == "F<G"]
add_co = [(tg, dt) for tg, dt, o in add_pts if o == "coincident"]
sub_fg = [(tg, dt) for tg, dt, o in sub_pts if o == "F<G"]
sub_gf = [(tg, dt) for tg, dt, o in sub_pts if o == "G<F"]

def sc(ax, pts, c, m, s, lab, zo=3):
    if pts:
        xs, ys = zip(*pts)
        ax.scatter(xs, ys, c=c, marker=m, s=s, alpha=0.80,
                   label=lab, zorder=zo)

sc(ax11, add_gf, "steelblue", "o", 55,
   f"Addition G<F  (n={len(add_gf)})")
sc(ax11, add_fg, "crimson",   "*", 220,
   f"Addition F<G  (n={len(add_fg)})", zo=4)
sc(ax11, add_co, "grey",      "D", 65,
   f"Addition coincident  (n={len(add_co)})")
sc(ax11, sub_fg, "darkorange", "^", 160,
   f"Sub F<G  (n={len(sub_fg)}, pilot)", zo=4)
sc(ax11, sub_gf, "seagreen",   "^", 160,
   f"Sub G<F  (n={len(sub_gf)}, pilot)", zo=4)

ax11.axhline(0, color="black", lw=1.0, ls="--", alpha=0.45,
             label=r"$\Delta\tau{=}0$")
ax11.axhspan(-300, 0, alpha=0.04, color="crimson")
ax11.axhspan(0, 300, alpha=0.04, color="steelblue")

ax11.set_xlabel(r"$\tau_\mathrm{gen}$  (k steps)", fontsize=12)
ax11.set_ylabel(r"$\Delta\tau = \tau_F - \tau_\mathrm{gen}$  (k steps)",
                fontsize=12)
ax11.set_title(
    "Speed-dependent ordering hypothesis\n"
    r"Fast grokking ($\tau_\mathrm{gen}$ small) $\to$ G<F; "
    r"Slow grokking $\to$ F<G",
    fontsize=11)
ax11.legend(fontsize=9, loc="upper right")
ax11.set_xlim(left=0)

ax11.text(0.02, 0.97, "G<F regime", transform=ax11.transAxes,
          color="steelblue", fontsize=10, va="top", alpha=0.75)
ax11.text(0.02, 0.03, "F<G regime", transform=ax11.transAxes,
          color="crimson", fontsize=10, va="bottom", alpha=0.75)

# correlation over ALL 57 runs (55 addition + 2 sub pilots)
all57 = add_pts + sub_pts
all_tg = np.array([p[0] for p in all57])
all_dt = np.array([p[1] for p in all57])
r_all  = np.corrcoef(all_tg, all_dt)[0, 1]
n_all  = len(all57)
print(f"  All-57 correlation: r={r_all:.3f}, n={n_all}")
ax11.text(0.98, 0.03, f"r = {r_all:.2f}  (n={n_all})",
          transform=ax11.transAxes, fontsize=9, va="bottom", ha="right",
          bbox=dict(fc="white", ec="#aaaaaa", alpha=0.88, pad=2))

fig11.tight_layout()
out11 = PAPER / "speed_ordering_scatter.png"
fig11.savefig(out11, dpi=150, bbox_inches="tight")
print("  speed_ordering_scatter.png saved")
plt.close(fig11)

print("\nAll figures done.")
