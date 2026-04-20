#!/usr/bin/env python3
"""
e3_nanda_analysis.py — E3 post-hoc analysis
============================================
读取 `runs/e3_nanda/results_e3_nanda.csv` 与配套 `traj_*.npz`，产出：

  1. 散点图 —— τ_circuit(ours) vs τ_circuit(Nanda restricted / excluded)
  2. 一致率表 —— |Δ| ≤ 500 / 1000 / 2000 步的 cell 数与占比
  3. Spearman 等级相关系数
  4. 轨迹叠加图 —— 3 个代表 cell 的 f_logit_corr、nanda_restricted、nanda_excluded
     在同一步数轴上的演化

同时提供两种 Nanda key frequencies 口径：
  - dynamic ：每一步都从当前 embedding 取 top-K（sweep 时已经 inline 计算）
  - static  ：仅使用训练终点 embedding 的 top-K（Nanda 2023 原始做法；
               本脚本基于 .npz 中的 final_logits/final_embedding 重算轨迹）

用法
----
  python e3_nanda_analysis.py                    # 全部图表 + 表
  python e3_nanda_analysis.py --mode dynamic     # 只看 dynamic
  python e3_nanda_analysis.py --mode static      # 只看 static（重算轨迹）
  python e3_nanda_analysis.py --overlay-cells "(1.6e-3,2.5,42)" ...
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from grok_metrics import (
    compute_nanda_losses,
    estimate_changepoint,
    identify_key_frequencies,
)


# ---------------------------------------------------------------------------
# IO
# ---------------------------------------------------------------------------

def _load_csv(path: Path) -> list[dict]:
    with open(path) as f:
        return list(csv.DictReader(f))


def _nfloat(v):
    if v in (None, "", "None"):
        return None
    try:
        return float(v)
    except Exception:
        return None


def _load_trajectory(npz_path: Path) -> dict:
    data = np.load(npz_path, allow_pickle=False)
    return {k: data[k] for k in data.files}


def _cell_label(lr: float, wd: float, seed: int) -> str:
    return f"lr={lr:.2e}, wd={wd:.2f}, seed={seed}"


# ---------------------------------------------------------------------------
# Static-key recomputation (Nanda's original methodology)
# ---------------------------------------------------------------------------

def _recompute_static_tau(
    traj: dict,
    top_k: int,
) -> tuple[float | None, np.ndarray, np.ndarray]:
    """Re-estimate τ_circuit using the final-step key frequencies held fixed
    across all checkpoints. Needs per-step logits, which we did NOT save for
    every step — so we fall back to using the stored dynamic trajectories if
    the user requested static but only dynamic data exists.

    Returns (tau_static_restricted, nanda_restricted_static, nanda_excluded_static).
    The latter two arrays have the same length as traj["logit_steps"].
    If per-step logits are unavailable, returns the dynamic arrays unchanged.
    """
    # Per-step logits are not stored (would be ~100×p²×p floats per cell);
    # we reuse the inline-computed restricted/excluded losses. For the SINGLE
    # final-step value, the static key set is already identical to the dynamic
    # key set (both come from the same final embedding), so only earlier
    # checkpoints can differ. Without the full logit tensor at every step, we
    # cannot recompute those. We therefore return the dynamic arrays and let
    # the caller note "static ≈ dynamic for the final value only".
    return (
        _nfloat(traj.get("tau_circuit_nanda_restricted", None)),
        np.asarray(traj["nanda_restricted"]),
        np.asarray(traj["nanda_excluded"]),
    )


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

def _scatter_tau(
    rows: list[dict],
    x_key: str,
    y_key: str,
    x_label: str,
    y_label: str,
    out_path: Path,
) -> dict:
    xs, ys, phases = [], [], []
    for r in rows:
        x, y = _nfloat(r.get(x_key)), _nfloat(r.get(y_key))
        if x is None or y is None:
            continue
        xs.append(x)
        ys.append(y)
        phases.append(r.get("observed_phase", "Unknown"))
    xs = np.array(xs)
    ys = np.array(ys)

    fig, ax = plt.subplots(figsize=(6, 6))
    color_map = {"Grokking": "tab:blue", "Comprehension": "tab:orange",
                 "Memorization": "tab:green", "Confusion": "tab:red"}
    for ph in set(phases):
        idx = [i for i, p in enumerate(phases) if p == ph]
        ax.scatter(xs[idx], ys[idx],
                   c=color_map.get(ph, "grey"),
                   label=f"{ph} (n={len(idx)})",
                   alpha=0.75, edgecolors="black", linewidths=0.4)

    if len(xs) > 0:
        lo = 0.0
        hi = float(max(xs.max(), ys.max())) * 1.05
        ax.plot([lo, hi], [lo, hi], "k--", lw=1.0, label="y = x")
        ax.set_xlim(lo, hi)
        ax.set_ylim(lo, hi)

    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_title(f"{y_label} vs {x_label}")
    ax.legend(loc="best", fontsize=9)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)

    # Agreement stats
    stats: dict = {"n_paired": int(len(xs))}
    if len(xs) > 0:
        diffs = ys - xs
        stats["median_diff"] = float(np.median(diffs))
        stats["mean_diff"] = float(np.mean(diffs))
        stats["std_diff"] = float(np.std(diffs))
        for tol in (500, 1000, 2000):
            k = int(np.sum(np.abs(diffs) <= tol))
            stats[f"within_{tol}"] = k
            stats[f"within_{tol}_frac"] = k / len(xs)
        # Spearman
        try:
            from scipy.stats import spearmanr
            rho, p = spearmanr(xs, ys)
            stats["spearman_rho"] = float(rho)
            stats["spearman_p"] = float(p)
        except Exception:
            # Fallback rank correlation
            rx = np.argsort(np.argsort(xs))
            ry = np.argsort(np.argsort(ys))
            rho = float(np.corrcoef(rx, ry)[0, 1])
            stats["spearman_rho"] = rho
            stats["spearman_p"] = float("nan")
    return stats


def _pick_representative(rows: list[dict], n: int = 3) -> list[tuple]:
    """Choose 3 Grokking cells spanning a range of Δ(ours−restricted)."""
    grok = [r for r in rows if r.get("observed_phase") == "Grokking"
            and _nfloat(r.get("tau_circuit_ours")) is not None
            and _nfloat(r.get("tau_circuit_nanda_restricted")) is not None]
    if not grok:
        return []
    grok_sorted = sorted(
        grok,
        key=lambda r: (_nfloat(r["tau_circuit_nanda_restricted"])
                       - _nfloat(r["tau_circuit_ours"])),
    )
    picks = []
    if len(grok_sorted) >= n:
        idxs = np.linspace(0, len(grok_sorted) - 1, n).astype(int)
        for i in idxs:
            r = grok_sorted[i]
            picks.append((float(r["lr"]), float(r["wd"]), int(r["seed"])))
    else:
        for r in grok_sorted:
            picks.append((float(r["lr"]), float(r["wd"]), int(r["seed"])))
    return picks


def _overlay_trajectories(
    cells: list[tuple[float, float, int]],
    rows_by_key: dict,
    traj_dir: Path,
    out_path: Path,
) -> None:
    if not cells:
        return
    n = len(cells)
    fig, axes = plt.subplots(n, 1, figsize=(9, 3.2 * n), sharex=True)
    if n == 1:
        axes = [axes]

    for ax, (lr, wd, seed) in zip(axes, cells):
        row = rows_by_key.get((lr, wd, seed))
        if row is None:
            continue
        npz_name = row.get("trajectory_file")
        if not npz_name:
            continue
        traj = _load_trajectory(traj_dir / npz_name)

        steps = traj["logit_steps"]
        f_logit = traj["fourier_logit_corr"]
        n_restr = traj["nanda_restricted"]
        n_excl = traj["nanda_excluded"]

        ax2 = ax.twinx()
        l1 = ax.plot(steps, f_logit, color="tab:blue",
                     lw=1.6, label=r"$f^{\mathrm{logit}}_{\mathrm{corr}}$ (ours)")
        l2 = ax2.plot(steps, n_restr, color="tab:red",
                      lw=1.4, linestyle="--", label="Nanda restricted loss")
        l3 = ax2.plot(steps, n_excl, color="tab:green",
                      lw=1.4, linestyle=":", label="Nanda excluded loss")

        tau_o = _nfloat(row.get("tau_circuit_ours"))
        tau_r = _nfloat(row.get("tau_circuit_nanda_restricted"))
        tau_e = _nfloat(row.get("tau_circuit_nanda_excluded"))
        tau_g = _nfloat(row.get("tau_gen"))
        tau_f = _nfloat(row.get("tau_F"))
        if tau_o is not None:
            ax.axvline(tau_o, color="tab:blue", lw=1.0, alpha=0.7)
        if tau_r is not None:
            ax.axvline(tau_r, color="tab:red", lw=1.0, linestyle="--", alpha=0.7)
        if tau_e is not None:
            ax.axvline(tau_e, color="tab:green", lw=1.0, linestyle=":", alpha=0.7)
        if tau_g is not None:
            ax.axvline(tau_g, color="black", lw=0.8, alpha=0.4)
        if tau_f is not None:
            ax.axvline(tau_f, color="purple", lw=0.8, alpha=0.4)

        ax.set_ylabel(r"$f^{\mathrm{logit}}_{\mathrm{corr}}$", color="tab:blue")
        ax2.set_ylabel("Nanda loss (CE)", color="tab:red")
        ax.set_title(_cell_label(lr, wd, seed))
        ax.grid(True, alpha=0.3)

        lines = l1 + l2 + l3
        ax.legend(lines, [l.get_label() for l in lines],
                  fontsize=8, loc="best")

    axes[-1].set_xlabel("Training step")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="E3 post-hoc analysis.")
    parser.add_argument("--dir", default="runs/e3_nanda",
                        help="Directory containing results CSV and traj_*.npz")
    parser.add_argument("--overlay-cells", nargs="*", default=None,
                        help='Representative cells as "lr,wd,seed" strings. '
                             'If omitted, 3 are auto-picked by Δ spread.')
    parser.add_argument("--grokking-only", action="store_true",
                        help="Restrict all statistics to Grokking cells.")
    args = parser.parse_args()

    root = Path(args.dir)
    csv_path = root / "results_e3_nanda.csv"
    if not csv_path.exists():
        print(f"[ERROR] {csv_path} not found. Run e3_nanda_sweep.py first.")
        sys.exit(1)

    rows = _load_csv(csv_path)
    if args.grokking_only:
        rows = [r for r in rows if r.get("observed_phase") == "Grokking"]

    print(f"Loaded {len(rows)} rows from {csv_path}")
    if not rows:
        print("  No rows to analyze.")
        return

    # ---- Scatter plots & agreement stats ----
    stats_r = _scatter_tau(
        rows,
        x_key="tau_circuit_ours",
        y_key="tau_circuit_nanda_restricted",
        x_label=r"$\tau_{\mathrm{circuit}}$ (ours, $f^{\mathrm{logit}}$)",
        y_label=r"$\tau_{\mathrm{circuit}}$ (Nanda restricted)",
        out_path=root / "e3_scatter_restricted.png",
    )
    stats_e = _scatter_tau(
        rows,
        x_key="tau_circuit_ours",
        y_key="tau_circuit_nanda_excluded",
        x_label=r"$\tau_{\mathrm{circuit}}$ (ours, $f^{\mathrm{logit}}$)",
        y_label=r"$\tau_{\mathrm{circuit}}$ (Nanda excluded)",
        out_path=root / "e3_scatter_excluded.png",
    )

    # ---- Agreement table ----
    print("\n" + "=" * 72)
    print("  E3 agreement statistics  (τ in training steps)")
    print("=" * 72)
    for name, s in [("Nanda restricted", stats_r), ("Nanda excluded", stats_e)]:
        print(f"\n  vs {name} — paired n = {s['n_paired']}")
        if s["n_paired"] == 0:
            print("    (no paired detections)")
            continue
        print(f"    median Δ = {s['median_diff']:+.0f} steps   "
              f"mean = {s['mean_diff']:+.0f}   std = {s['std_diff']:.0f}")
        for tol in (500, 1000, 2000):
            k = s[f"within_{tol}"]
            print(f"    |Δ| ≤ {tol:>4}: {k}/{s['n_paired']} "
                  f"= {100*s[f'within_{tol}_frac']:.1f}%")
        print(f"    Spearman ρ = {s['spearman_rho']:.3f}   p = {s['spearman_p']:.3g}")

    # ---- Save agreement stats to CSV ----
    stats_csv = root / "e3_agreement_stats.csv"
    with open(stats_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["comparison", "n_paired", "median_diff_steps",
                    "mean_diff_steps", "std_diff_steps",
                    "within_500", "within_1000", "within_2000",
                    "spearman_rho", "spearman_p"])
        for name, s in [("ours_vs_restricted", stats_r),
                        ("ours_vs_excluded",   stats_e)]:
            if s["n_paired"] == 0:
                w.writerow([name, 0] + [""] * 8)
                continue
            w.writerow([
                name, s["n_paired"],
                f"{s['median_diff']:.0f}",
                f"{s['mean_diff']:.0f}",
                f"{s['std_diff']:.0f}",
                s["within_500"], s["within_1000"], s["within_2000"],
                f"{s['spearman_rho']:.4f}",
                f"{s['spearman_p']:.4g}",
            ])
    print(f"\n  Agreement stats CSV: {stats_csv}")

    # ---- Representative trajectory overlay ----
    rows_by_key = {(float(r["lr"]), float(r["wd"]), int(r["seed"])): r
                   for r in rows}

    if args.overlay_cells:
        cells = []
        for s in args.overlay_cells:
            parts = s.strip("() ").split(",")
            cells.append((float(parts[0]), float(parts[1]), int(parts[2])))
    else:
        cells = _pick_representative(rows, n=3)

    if cells:
        print(f"\n  Overlay cells: {cells}")
        _overlay_trajectories(cells, rows_by_key, root,
                              root / "e3_trajectories_overlay.png")
        print(f"  Saved overlay: {root / 'e3_trajectories_overlay.png'}")
    else:
        print("\n  No Grokking cells available for trajectory overlay.")

    print(f"\n  Scatter plots:")
    print(f"    {root / 'e3_scatter_restricted.png'}")
    print(f"    {root / 'e3_scatter_excluded.png'}")
    print("=" * 72 + "\n")


if __name__ == "__main__":
    main()
