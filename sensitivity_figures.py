#!/usr/bin/env python3
"""
Sensitivity Analysis — Paper-Quality Figures + Outlier Analysis
================================================================
生成三类输出：
  1. 稳健性热力图（paper 用）：slope_thresh × ema_alpha，颜色=P(G<F)
     - 子图 A：全部 G<F cells 平均（即论文主图，展示"稳健性高原"）
     - 子图 B：按 cell 分层的 8 格面板（附录/补充）
  2. Robustness Score 统计汇总（打印 + 写入 CSV）
  3. F<G 离群点（lr=6.8e-4 seed=7）轨迹分析：
     - 对比 tau_gen、tau_F、dec_norm 轨迹
     - 检验是否是"低频谐波"或"早期权重压缩"假说
"""

from __future__ import annotations
import csv, json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.colors import LinearSegmentedColormap

# ── paths ──────────────────────────────────────────────────────────────────
SENS_DIR  = Path("runs/sensitivity")
TRAJ_DIR  = SENS_DIR / "trajectories"
RES_CSV   = SENS_DIR / "results.csv"
OUT_DIR   = Path("paper")
OUT_DIR.mkdir(exist_ok=True)

# ── 1. load results ────────────────────────────────────────────────────────
rows = list(csv.DictReader(RES_CSV.read_text(encoding="utf-8").splitlines()))

CELLS_ORDER = [
    "wd1.71_s7", "wd2.76_s42",   # strong
    "wd1.52_s42", "wd2.18_s42",  # medium
    "wd2.45_s7", "wd3.50_s7",    # weak
    "wd3.11_s2025",              # coincident
    "lr6.8e4_s7",                # F<G
]
CELL_LABELS = {
    "wd1.71_s7":    "wd=1.71\n(strong)",
    "wd2.76_s42":   "wd=2.76\n(strong)",
    "wd1.52_s42":   "wd=1.52\n(medium)",
    "wd2.18_s42":   "wd=2.18\n(medium)",
    "wd2.45_s7":    "wd=2.45\n(weak)",
    "wd3.50_s7":    "wd=3.50\n(weak)",
    "wd3.11_s2025": "wd=3.11\n(coin.)",
    "lr6.8e4_s7":   "lr=6.8e-4\n(F<G)",
}
NULL_INTERVALS = [500, 2500, 5000]
EMA_ALPHAS     = [1.0, 0.30, 0.15, 0.05]   # y-axis (top=no smooth)
SLOPE_THRESHS  = [0.005, 0.01, 0.02]       # x-axis

def p_gf(subset):
    """Fraction G<F in subset of rows."""
    if not subset: return float("nan")
    return sum(1 for r in subset if r["ordering"] == "G<F") / len(subset)

# ── 2. Robustness Score ────────────────────────────────────────────────────
print("=" * 60)
print("ROBUSTNESS SCORES")
print("=" * 60)

RS_rows = []
for cell in CELLS_ORDER:
    cell_rows = [r for r in rows if r["cell_label"] == cell]
    n_total = len(cell_rows)
    n_gf    = sum(1 for r in cell_rows if r["ordering"] == "G<F")
    n_fg    = sum(1 for r in cell_rows if r["ordering"] == "F<G")
    n_coin  = sum(1 for r in cell_rows if r["ordering"] == "coincident")
    rs      = n_gf / n_total if n_total else float("nan")
    print(f"  {cell:<20}  RS={rs:.3f}  G<F={n_gf}  F<G={n_fg}  coin={n_coin}  total={n_total}")
    RS_rows.append(dict(cell=cell, rs=rs, n_gf=n_gf, n_fg=n_fg, n_coin=n_coin, n_total=n_total))

# overall (excluding the known F<G cell)
gf_cells = [r for r in RS_rows if "FG" not in r["cell"] and r["cell"] != "lr6.8e4_s7"]
overall_rs = np.mean([r["rs"] for r in gf_cells])
print(f"\n  Mean RS (7 non-F<G cells): {overall_rs:.3f}")

gf_only = [r for r in RS_rows if r["cell"] != "lr6.8e4_s7"]
mean_all_rs = np.nanmean([r["rs"] for r in gf_only])
print(f"  Mean RS (all 7 G<F/coin cells): {mean_all_rs:.3f}")

# write RS summary
rs_path = SENS_DIR / "robustness_scores.csv"
with open(rs_path, "w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=["cell","rs","n_gf","n_fg","n_coin","n_total"])
    w.writeheader(); w.writerows(RS_rows)
print(f"\n  Saved {rs_path}")

# ── 3. Heatmap: per-cell P(G<F) as fn of (slope_thresh, ema_alpha) ─────────
# marginalized over null_interval (average across 3 values)

def cell_heatmap_data(cell_rows):
    """Returns (4×3) array: rows=ema_alpha (high→low), cols=slope_thresh."""
    mat = np.full((len(EMA_ALPHAS), len(SLOPE_THRESHS)), float("nan"))
    for i, ea in enumerate(EMA_ALPHAS):
        for j, st in enumerate(SLOPE_THRESHS):
            subset = [r for r in cell_rows
                      if float(r["ema_alpha"]) == ea and float(r["slope_thresh"]) == st]
            mat[i, j] = p_gf(subset)
    return mat

# ── Figure A: single aggregate heatmap (G<F cells only) ────────────────────
fig_a, ax = plt.subplots(figsize=(4.5, 3.5))

GF_CELLS = [c for c in CELLS_ORDER if c != "lr6.8e4_s7"]
agg_mat = np.zeros((len(EMA_ALPHAS), len(SLOPE_THRESHS)))
for cell in GF_CELLS:
    cell_rows = [r for r in rows if r["cell_label"] == cell]
    agg_mat += cell_heatmap_data(cell_rows)
agg_mat /= len(GF_CELLS)

# custom colormap: white→blue
cmap = LinearSegmentedColormap.from_list("wb", ["#ffffff", "#1a5fa8"])
im = ax.imshow(agg_mat, cmap=cmap, vmin=0, vmax=1, aspect="auto")

ax.set_xticks(range(len(SLOPE_THRESHS)))
ax.set_xticklabels([str(s) for s in SLOPE_THRESHS], fontsize=10)
ax.set_xlabel("BIC slope threshold", fontsize=11)

ax.set_yticks(range(len(EMA_ALPHAS)))
ax.set_yticklabels([f"α={a}" for a in EMA_ALPHAS], fontsize=10)
ax.set_ylabel("EMA smoothing (α)", fontsize=11)

# annotate cells
for i in range(len(EMA_ALPHAS)):
    for j in range(len(SLOPE_THRESHS)):
        v = agg_mat[i, j]
        ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                fontsize=9, color="white" if v > 0.6 else "black", fontweight="bold")

plt.colorbar(im, ax=ax, label="P(G<F)", fraction=0.046, pad=0.04)
ax.set_title("Robustness plateau: avg P(G<F)\nacross 7 Grokking cells, marginalized over null interval",
             fontsize=10)
fig_a.tight_layout()
path_a = OUT_DIR / "sensitivity_robustness_heatmap.pdf"
fig_a.savefig(path_a, dpi=200, bbox_inches="tight")
fig_a.savefig(str(path_a).replace(".pdf", ".png"), dpi=150, bbox_inches="tight")
print(f"\n  Saved {path_a}")
plt.close(fig_a)

# ── Figure B: 8-panel per-cell heatmaps ────────────────────────────────────
fig_b, axes = plt.subplots(2, 4, figsize=(14, 7))
axes = axes.flatten()

for idx, cell in enumerate(CELLS_ORDER):
    ax = axes[idx]
    cell_rows = [r for r in rows if r["cell_label"] == cell]
    mat = cell_heatmap_data(cell_rows)

    if cell == "lr6.8e4_s7":
        cmap_c = LinearSegmentedColormap.from_list("wr", ["#ffffff", "#c0392b"])
    else:
        cmap_c = cmap

    im = ax.imshow(mat, cmap=cmap_c, vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(range(len(SLOPE_THRESHS)))
    ax.set_xticklabels([str(s) for s in SLOPE_THRESHS], fontsize=8)
    ax.set_yticks(range(len(EMA_ALPHAS)))
    ax.set_yticklabels([f"α={a}" for a in EMA_ALPHAS], fontsize=8)
    ax.set_title(CELL_LABELS[cell], fontsize=9)

    rs = sum(1 for r in cell_rows if r["ordering"]=="G<F") / len(cell_rows)
    ax.set_xlabel(f"RS={rs:.2f}", fontsize=8)

    for i in range(len(EMA_ALPHAS)):
        for j in range(len(SLOPE_THRESHS)):
            v = mat[i, j]
            if not np.isnan(v):
                ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                        fontsize=7.5, color="white" if v > 0.55 else "black")

plt.suptitle("Per-cell P(G<F) across detector configurations (marginalized over null_interval)",
             fontsize=11, y=1.01)
fig_b.tight_layout()
path_b = OUT_DIR / "sensitivity_percell_heatmap.pdf"
fig_b.savefig(path_b, dpi=200, bbox_inches="tight")
fig_b.savefig(str(path_b).replace(".pdf", ".png"), dpi=150, bbox_inches="tight")
print(f"  Saved {path_b}")
plt.close(fig_b)

# ── 4. F<G outlier analysis ────────────────────────────────────────────────
print("\n" + "=" * 60)
print("F<G OUTLIER ANALYSIS: lr6.8e4_s7")
print("=" * 60)

traj_path = TRAJ_DIR / "lr6.8e4_s7.json"
fg_traj   = json.loads(traj_path.read_text(encoding="utf-8"))

# also load a strong G<F cell for comparison
ref_path  = TRAJ_DIR / "wd1.71_s7.json"
ref_traj  = json.loads(ref_path.read_text(encoding="utf-8"))

steps_fg  = np.array(fg_traj["steps"])
f_raw_fg  = np.array(fg_traj["f_raw"])
f_null_fg = np.array(fg_traj.get("f_null_p95", fg_traj.get("f_null", [0]*len(f_raw_fg))))
f_corr_fg = np.maximum(0, f_raw_fg - f_null_fg)
test_acc_fg  = np.array(fg_traj["test_acc"])
train_acc_fg = np.array(fg_traj["train_acc"])

steps_ref  = np.array(ref_traj["steps"])
f_raw_ref  = np.array(ref_traj["f_raw"])
f_null_ref = np.array(ref_traj.get("f_null_p95", ref_traj.get("f_null", [0]*len(f_raw_ref))))
f_corr_ref = np.maximum(0, f_raw_ref - f_null_ref)
test_acc_ref  = np.array(ref_traj["test_acc"])
train_acc_ref = np.array(ref_traj["train_acc"])

tau_gen_fg = float(fg_traj.get("tau_gen") or "nan")
tau_F_fg   = float(fg_traj.get("tau_F",   "nan") or "nan")
# tau_F not stored in trajectory — read from results.csv
fg_results = [r for r in rows if r["cell_label"] == "lr6.8e4_s7" and r["null_interval"] == "500" and r["ema_alpha"] == "1.0" and r["slope_thresh"] == "0.01"]
if fg_results and fg_results[0]["tau_F"]:
    tau_F_fg = float(fg_results[0]["tau_F"])

print(f"  tau_gen = {tau_gen_fg}")
print(f"  tau_F   = {tau_F_fg}")
print(f"  Δ       = {tau_F_fg - tau_gen_fg if not (np.isnan(tau_gen_fg) or np.isnan(tau_F_fg)) else 'N/A'}")

# Key diagnostic: f_raw at tau_gen
idx_tg = np.searchsorted(steps_fg, tau_gen_fg) if not np.isnan(tau_gen_fg) else None
if idx_tg is not None and idx_tg < len(f_raw_fg):
    print(f"\n  f_raw  at tau_gen: {f_raw_fg[idx_tg]:.4f}")
    print(f"  f_null at tau_gen: {f_null_fg[idx_tg]:.4f}")
    print(f"  f_corr at tau_gen: {f_corr_fg[idx_tg]:.4f}")

# Check: is f_raw already high very early? (weight compression / fast structure formation)
early_idx = steps_fg < 5000
print(f"\n  Max f_raw in first 5k steps: {f_raw_fg[early_idx].max():.4f}")
print(f"  Max f_null in first 5k steps: {f_null_fg[early_idx].max():.4f}")

# Find step where f_corr first exceeds 0.01
thresh = 0.01
above = np.where(f_corr_fg > thresh)[0]
if len(above):
    print(f"\n  f_corr first > {thresh} at step {steps_fg[above[0]]}")
else:
    print(f"\n  f_corr never > {thresh}")

# Check dominant harmonic at tau_gen vs reference
# (We only have aggregate F stats in trajectory, not per-harmonic breakdown)
# So instead: compare f_raw trajectory shape
print(f"\n  f_raw at step 500  (FG):  {f_raw_fg[steps_fg==500][0] if 500 in steps_fg else 'N/A'}")
print(f"  f_raw at step 1000 (FG):  {f_raw_fg[steps_fg==1000][0] if 1000 in steps_fg else 'N/A'}")
print(f"  f_raw at step 2000 (FG):  {f_raw_fg[steps_fg==2000][0] if 2000 in steps_fg else 'N/A'}")
print(f"  f_raw at step 5000 (FG):  {f_raw_fg[steps_fg==5000][0] if 5000 in steps_fg else 'N/A'}")

# ── Figure C: F<G outlier trajectory ─────────────────────────────────────
fig_c, axes_c = plt.subplots(1, 2, figsize=(11, 4))

# Left: F<G outlier
ax = axes_c[0]
ax.plot(steps_fg/1000, test_acc_fg,  color="#e74c3c", lw=1.8, label="test acc")
ax.plot(steps_fg/1000, train_acc_fg, color="#e74c3c", lw=1.0, ls="--", alpha=0.5, label="train acc")
ax2 = ax.twinx()
ax2.plot(steps_fg/1000, f_corr_fg,  color="#2980b9", lw=1.8, label="$F_\\mathrm{corr}$")
ax2.plot(steps_fg/1000, f_raw_fg,   color="#2980b9", lw=1.0, ls=":", alpha=0.5, label="$F_\\mathrm{raw}$")
ax2.plot(steps_fg/1000, f_null_fg,  color="#95a5a6", lw=1.0, ls="--", alpha=0.7, label="null p95")

if not np.isnan(tau_gen_fg):
    ax.axvline(tau_gen_fg/1000, color="#e74c3c", lw=1.5, ls="--", alpha=0.8)
    ax.text(tau_gen_fg/1000+0.3, 0.55, r"$\tau_\mathrm{gen}$", color="#e74c3c", fontsize=10)
if not np.isnan(tau_F_fg):
    ax2.axvline(tau_F_fg/1000, color="#2980b9", lw=1.5, ls="--", alpha=0.8)
    ax2.text(tau_F_fg/1000+0.3, ax2.get_ylim()[1]*0.8, r"$\tau_F$", color="#2980b9", fontsize=10)

ax.set_xlabel("Training steps (×10³)", fontsize=11)
ax.set_ylabel("Accuracy", fontsize=11, color="#e74c3c")
ax2.set_ylabel("Fourier alignment", fontsize=11, color="#2980b9")
ax.set_title("F<G outlier: lr=6.8e-4, seed=7\n(F<G: τ_F before τ_gen)", fontsize=10)
lines1, labels1 = ax.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax.legend(lines1+lines2, labels1+labels2, fontsize=8, loc="center left")
ax.tick_params(axis="y", colors="#e74c3c")
ax2.tick_params(axis="y", colors="#2980b9")

# Right: reference G<F cell
ax = axes_c[1]
ax.plot(steps_ref/1000, test_acc_ref,  color="#e74c3c", lw=1.8, label="test acc")
ax.plot(steps_ref/1000, train_acc_ref, color="#e74c3c", lw=1.0, ls="--", alpha=0.5, label="train acc")
ax2 = ax.twinx()
ax2.plot(steps_ref/1000, f_corr_ref, color="#2980b9", lw=1.8, label="$F_\\mathrm{corr}$")
ax2.plot(steps_ref/1000, f_raw_ref,  color="#2980b9", lw=1.0, ls=":", alpha=0.5, label="$F_\\mathrm{raw}$")
ax2.plot(steps_ref/1000, f_null_ref, color="#95a5a6", lw=1.0, ls="--", alpha=0.7, label="null p95")

tau_gen_ref = float(ref_traj.get("tau_gen") or "nan")
tau_F_ref   = float(ref_traj.get("tau_F", "nan") or "nan")
ref_results = [r for r in rows if r["cell_label"] == "wd1.71_s7" and r["null_interval"] == "500" and r["ema_alpha"] == "1.0" and r["slope_thresh"] == "0.01"]
if ref_results and ref_results[0]["tau_F"]:
    tau_F_ref = float(ref_results[0]["tau_F"])
if not np.isnan(tau_gen_ref):
    ax.axvline(tau_gen_ref/1000, color="#e74c3c", lw=1.5, ls="--", alpha=0.8)
    ax.text(tau_gen_ref/1000+0.3, 0.55, r"$\tau_\mathrm{gen}$", color="#e74c3c", fontsize=10)
if not np.isnan(tau_F_ref):
    ax2.axvline(tau_F_ref/1000, color="#2980b9", lw=1.5, ls="--", alpha=0.8)
    ax2.text(tau_F_ref/1000+0.3, ax2.get_ylim()[1]*0.8, r"$\tau_F$", color="#2980b9", fontsize=10)

ax.set_xlabel("Training steps (×10³)", fontsize=11)
ax.set_ylabel("Accuracy", fontsize=11, color="#e74c3c")
ax2.set_ylabel("Fourier alignment", fontsize=11, color="#2980b9")
ax.set_title("G<F reference: wd=1.71, seed=7\n(G<F: τ_gen before τ_F)", fontsize=10)
lines1, labels1 = ax.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax.legend(lines1+lines2, labels1+labels2, fontsize=8, loc="center left")
ax.tick_params(axis="y", colors="#e74c3c")
ax2.tick_params(axis="y", colors="#2980b9")

fig_c.suptitle("F<G outlier vs G<F reference: trajectory comparison", fontsize=11)
fig_c.tight_layout()
path_c = OUT_DIR / "sensitivity_outlier_trajectory.pdf"
fig_c.savefig(path_c, dpi=200, bbox_inches="tight")
fig_c.savefig(str(path_c).replace(".pdf", ".png"), dpi=150, bbox_inches="tight")
print(f"\n  Saved {path_c}")
plt.close(fig_c)

print("\nAll done.")
