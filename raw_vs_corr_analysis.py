#!/usr/bin/env python3
"""
Raw vs Null-Corrected Fourier: Ordering Comparison
====================================================
Issue 6: 展示 F_raw 是否会将 G<F 反转为 F<G。
使用 sensitivity 保存的轨迹，对每个 cell 分别用：
  - f_corr = max(0, f_raw - f_null)  (canonical, 论文用)
  - f_raw                             (uncorrected)
计算 tau_F 和 ordering，生成对比表 + 对比图。
"""
from __future__ import annotations
import csv, json, sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, ".")
from grok_metrics import estimate_changepoint

TRAJ_DIR = Path("runs/sensitivity/trajectories")
SENS_CSV = Path("runs/sensitivity/results.csv")
OUT_DIR  = Path("paper")

# ── canonical tau_gen from sensitivity results (null_interval=500, alpha=1.0, slope=0.01)
sens_rows = list(csv.DictReader(SENS_CSV.read_text(encoding="utf-8").splitlines()))
def get_tau_gen(cell):
    rows = [r for r in sens_rows if r["cell_label"]==cell
            and r["null_interval"]=="500" and r["ema_alpha"]=="1.0"
            and r["slope_thresh"]=="0.01"]
    if rows and rows[0]["tau_gen"]: return float(rows[0]["tau_gen"])
    return None

CELLS = [
    ("wd1.71_s7",    "strong G<F"),
    ("wd2.76_s42",   "strong G<F"),
    ("wd1.52_s42",   "medium G<F"),
    ("wd2.18_s42",   "medium G<F"),
    ("wd2.45_s7",    "weak G<F"),
    ("wd3.50_s7",    "weak G<F"),
    ("wd3.11_s2025", "coincident"),
    ("lr6.8e4_s7",   "F<G"),
]

print(f"{'Cell':<20} {'Cat':<12} {'tau_gen':>8} {'tauF_corr':>10} "
      f"{'order_corr':<12} {'tauF_raw':>9} {'order_raw':<12} {'flipped?'}")
print("-" * 100)

rows_out = []
for cell, cat in CELLS:
    traj = json.loads((TRAJ_DIR / f"{cell}.json").read_text(encoding="utf-8"))
    steps  = np.array(traj["steps"])
    f_raw  = np.array(traj["f_raw"])
    f_null = np.array(traj.get("f_null", [0.0]*len(f_raw)))
    f_corr = np.maximum(0, f_raw - f_null)

    tau_gen = get_tau_gen(cell)
    tau_F_corr = estimate_changepoint(steps, f_corr)
    tau_F_raw  = estimate_changepoint(steps, f_raw)

    def ordering(tg, tf):
        if tg is None or tf is None: return "None"
        d = tf - tg
        if abs(d) <= 500: return "coincident"
        return "G<F" if d > 0 else "F<G"

    ord_corr = ordering(tau_gen, tau_F_corr)
    ord_raw  = ordering(tau_gen, tau_F_raw)
    flipped  = (ord_corr != ord_raw) and ord_corr != "None" and ord_raw != "None"

    tgk  = f"{tau_gen/1000:.1f}k"   if tau_gen    else "None"
    tfc  = f"{tau_F_corr/1000:.1f}k" if tau_F_corr else "None"
    tfr  = f"{tau_F_raw/1000:.1f}k"  if tau_F_raw  else "None"

    marker = "<<< FLIP" if flipped else ""
    print(f"{cell:<20} {cat:<12} {tgk:>8} {tfc:>10} {ord_corr:<12} "
          f"{tfr:>9} {ord_raw:<12} {marker}")

    rows_out.append(dict(
        cell=cell, cat=cat,
        tau_gen=tau_gen, tau_F_corr=tau_F_corr, ord_corr=ord_corr,
        tau_F_raw=tau_F_raw, ord_raw=ord_raw, flipped=flipped,
    ))

# ── Summary ─────────────────────────────────────────────────────────────────
n_flip   = sum(r["flipped"] for r in rows_out)
n_gf_corr = sum(r["ord_corr"]=="G<F" for r in rows_out)
n_gf_raw  = sum(r["ord_raw"]=="G<F"  for r in rows_out)
print(f"\n  G<F under corr: {n_gf_corr}/8")
print(f"  G<F under raw:  {n_gf_raw}/8")
print(f"  Flips (corr≠raw): {n_flip}/8")

# ── Figure: side-by-side tau_F comparison ───────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(11, 5))

for ax_idx, (metric, key_tf, label, color) in enumerate([
    ("Null-corrected $F_\\mathrm{corr}$", "tau_F_corr", "Canonical (paper)", "#1a6fb0"),
    ("Raw $F_\\mathrm{raw}$",             "tau_F_raw",  "Uncorrected",       "#d62728"),
]):
    ax = axes[ax_idx]
    for r in rows_out:
        tg = (r["tau_gen"] or 0) / 1000
        tf = (r[key_tf]   or 0) / 1000
        if r["tau_gen"] and r[key_tf]:
            c = "#1a6fb0" if r[key_tf] > r["tau_gen"] else "#d62728"
            ax.scatter(tg, tf, s=70, color=c, zorder=3,
                       edgecolors="white", linewidths=0.5)
            ax.annotate(r["cell"].replace("_", "\n"),
                        (tg, tf), textcoords="offset points",
                        xytext=(5, 3), fontsize=6.5, alpha=0.8)

    # diagonal
    mn, mx = 0, max(
        max([(r["tau_gen"] or 0) for r in rows_out])/1000,
        max([(r[key_tf]   or 0) for r in rows_out])/1000,
    ) * 1.1
    ax.plot([mn, mx], [mn, mx], "k--", lw=0.8, alpha=0.4)
    ax.set_xlim(mn, mx); ax.set_ylim(mn, mx)
    ax.set_xlabel(r"$\tau_\mathrm{gen}$ (×10³ steps)", fontsize=11)
    ax.set_ylabel(rf"$\tau_F$ under {metric} (×10³ steps)", fontsize=10)
    ax.set_title(f"{label}\n"
                 f"G<F: {sum(1 for r in rows_out if r[key_tf] and r['tau_gen'] and r[key_tf]>r['tau_gen'])}/8",
                 fontsize=11)
    ax.set_aspect("equal")

    ax.text(0.03, 0.97,
            "Points above diagonal = G<F\nPoints below diagonal = F<G",
            transform=ax.transAxes, fontsize=8, va="top",
            bbox=dict(fc="white", ec="#cccccc", alpha=0.85))

fig.suptitle("Effect of null correction on ordering:\n"
             "raw Fourier (right) fires earlier, can reverse G<F → F<G",
             fontsize=11)
fig.tight_layout()
out = OUT_DIR / "raw_vs_corr_scatter.pdf"
fig.savefig(out, dpi=200, bbox_inches="tight")
fig.savefig(str(out).replace(".pdf", ".png"), dpi=150, bbox_inches="tight")
print(f"\n  Saved {out}")
plt.close(fig)
