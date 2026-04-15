#!/usr/bin/env python3
"""
Detector Sensitivity Analysis
==============================
验证 G<F 结论是否依赖于 detector 参数选择。

策略：
  - 选取 8 个代表性 cells（按 |Δτ| 分层：强/中/弱/coincident/F<G）
  - 训练时每 500 步保存完整轨迹（f_raw, f_null, train_acc, test_acc）
  - null 在每个 log step 都计算（最密粒度），事后 subsampling 模拟更粗频率
  - 事后对同一轨迹应用 36 种 detector 配置（null_interval × ema_alpha × slope_thresh）
  - 输出：每个 (cell, 配置) → τ_F, τ_gen, Δτ, ordering，汇总热力图

Cell 分层选取（来自 Stage 2 & Stage 3 已知结果）：
  强  G<F  (Δ>10k): wd=1.71 seed=7,    wd=2.76 seed=42
  中  G<F  (Δ5-10k): wd=1.52 seed=42,  wd=2.18 seed=42
  弱  G<F  (Δ1-5k): wd=2.45 seed=7,   wd=3.50 seed=7
  Coincident (|Δ|≤500): wd=3.11 seed=2025
  F<G  (Δ<0): lr=6.8e-4 seed=7  (Stage 3 outlier)

Detector 配置空间 (3×4×3 = 36):
  null_interval  ∈ {500, 2500, 5000}          (模拟更新频率)
  ema_alpha      ∈ {1.0, 0.30, 0.15, 0.05}   (1.0 = 无平滑)
  slope_thresh   ∈ {0.005, 0.01, 0.02}        (BIC 后继斜率阈值)

Usage:
  python sensitivity_analysis.py
  python sensitivity_analysis.py --max-steps 500 --log-interval 200  # smoketest
"""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import sys
import time
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, str(Path(__file__).parent))
from grokking_baseline import (
    GrokkingTransformer,
    build_optimizer,
    build_scheduler,
    get_token_embeddings,
    make_dataset,
    set_seed,
    split_dataset,
)
from grok_metrics import (
    classify_phase,
    compute_fourier_alignment,
    compute_fourier_null_p95,
    find_tau_sustained,
)

# ---------------------------------------------------------------------------
# Cell definitions (stratified by Δτ category)
# ---------------------------------------------------------------------------

CELLS = [
    # (label,              lr,       wd,   seed,  category)
    ("wd1.71_s7",    1.6e-3,   1.7145, 7,    "strong_GF"),   # Δ≈+6500
    ("wd2.76_s42",   1.6e-3,   2.7591, 42,   "strong_GF"),   # Δ≈+17500
    ("wd1.52_s42",   1.6e-3,   1.5223, 42,   "medium_GF"),   # Δ≈+7500
    ("wd2.18_s42",   1.6e-3,   2.175,  42,   "medium_GF"),   # Δ≈+8500
    ("wd2.45_s7",    1.6e-3,   2.4497, 7,    "weak_GF"),     # Δ≈+4000
    ("wd3.50_s7",    1.6e-3,   3.5,    7,    "weak_GF"),     # Δ≈+2500
    ("wd3.11_s2025", 1.6e-3,   3.1075, 2025, "coincident"),  # Δ=0
    ("lr6.8e4_s7",   6.804e-4, 2.5,    7,    "FG"),          # Δ≈-1500
]

# Detector configuration space
NULL_INTERVALS   = [500, 2500, 5000]
EMA_ALPHAS       = [1.0, 0.30, 0.15, 0.05]  # 1.0 = no smoothing
SLOPE_THRESHOLDS = [0.005, 0.01, 0.02]

COINCIDENT_THRESH = 500  # |Δτ| ≤ 500 → coincident

# ---------------------------------------------------------------------------
# Training: save full trajectory
# ---------------------------------------------------------------------------

def run_cell_save_traj(
    lr: float,
    wd: float,
    seed: int,
    max_steps: int = 50_000,
    log_interval: int = 500,
    n_null_perms: int = 200,   # more perms for accuracy
    prime: int = 53,
    train_fraction: float = 0.3,
    embed_lr: float = 1e-3,
    d_model: int = 256,
    n_heads: int = 4,
    n_layers: int = 2,
    d_ff: int = 1024,
) -> dict:
    """Train one cell, compute null at EVERY log step, return full trajectory."""
    t0 = time.time()
    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    x, y = make_dataset(prime, "add", seed)
    train_x, train_y, test_x, test_y = split_dataset(x, y, train_fraction)

    model = GrokkingTransformer(
        prime=prime, d_model=d_model, n_heads=n_heads,
        n_layers=n_layers, d_ff=d_ff, dropout=0.0,
    ).to(device)

    loader = DataLoader(
        TensorDataset(train_x, train_y),
        batch_size=len(train_x), shuffle=True,
        generator=torch.Generator().manual_seed(seed),
    )
    criterion = nn.CrossEntropyLoss()
    optimizer = build_optimizer(model, decoder_lr=lr, embed_lr=embed_lr,
                                decoder_weight_decay=wd)
    scheduler = build_scheduler(optimizer, warmup_steps=10, lr_schedule="constant",
                                lr_min_ratio=0.05, max_steps=max_steps)

    # Trajectory storage
    steps_log:      list[int]   = []
    train_acc_log:  list[float] = []
    test_acc_log:   list[float] = []
    f_raw_log:      list[float] = []
    f_null_log:     list[float] = []   # null p95 computed at EVERY log step

    null_rng = np.random.default_rng(seed ^ 0xDEAD)

    step = 0
    while step < max_steps:
        for xb, yb in loader:
            model.train()
            logits = model(xb.to(device))
            loss = criterion(logits, yb.to(device))
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            step += 1

            if step % log_interval == 0 or step == 1 or step >= max_steps:
                model.eval()
                with torch.no_grad():
                    tr_acc = (model(train_x.to(device)).argmax(1).cpu()
                              == train_y).float().mean().item()
                    te_acc = (model(test_x.to(device)).argmax(1).cpu()
                              == test_y).float().mean().item()

                emb = get_token_embeddings(model, prime)
                f_raw, _ = compute_fourier_alignment(emb, prime)
                # Compute null at EVERY log step (most expensive, but gives full data)
                f_null = compute_fourier_null_p95(emb, prime,
                                                  n_perms=n_null_perms, rng=null_rng)

                steps_log.append(step)
                train_acc_log.append(tr_acc)
                test_acc_log.append(te_acc)
                f_raw_log.append(f_raw)
                f_null_log.append(f_null)

            if step >= max_steps:
                break

    tau_gen = find_tau_sustained(steps_log, test_acc_log, threshold=0.9, n_sustained=3)
    observed_phase = classify_phase(steps_log, train_acc_log, test_acc_log)

    return {
        "steps":        steps_log,
        "train_acc":    train_acc_log,
        "test_acc":     test_acc_log,
        "f_raw":        f_raw_log,
        "f_null":       f_null_log,   # full null trajectory
        "tau_gen":      tau_gen,
        "observed_phase": observed_phase,
        "runtime_sec":  time.time() - t0,
    }


# ---------------------------------------------------------------------------
# Post-hoc detector: apply one config to a saved trajectory
# ---------------------------------------------------------------------------

def apply_detector(
    steps: list[int],
    f_raw: list[float],
    f_null_dense: list[float],
    tau_gen: Optional[float],
    log_interval: int,
    null_interval: int,
    ema_alpha: float,
    slope_thresh: float,
) -> dict:
    """Apply one detector config to a saved trajectory. Returns τ_F and ordering."""
    steps_arr  = np.array(steps)
    f_raw_arr  = np.array(f_raw)
    f_null_arr = np.array(f_null_dense)

    # 1. Simulate null_interval by piecewise-constant subsampling (LOCF)
    #    null_interval/log_interval gives the stride
    stride = max(1, null_interval // log_interval)
    f_null_sim = np.zeros_like(f_null_arr)
    last_null = f_null_arr[0]
    for i in range(len(f_null_arr)):
        if i % stride == 0:
            last_null = f_null_arr[i]
        f_null_sim[i] = last_null

    # 2. Compute f_corr
    f_corr = np.maximum(0.0, f_raw_arr - f_null_sim)

    # 3. EMA smoothing (alpha=1.0 means no smoothing: output = input)
    if ema_alpha >= 1.0:
        smoothed = f_corr.copy()
    else:
        smoothed = np.zeros_like(f_corr)
        smoothed[0] = f_corr[0]
        for i in range(1, len(f_corr)):
            smoothed[i] = ema_alpha * f_corr[i] + (1 - ema_alpha) * smoothed[i - 1]

    # 4. BIC changepoint on log-time scale
    tau_F = _estimate_changepoint_bic(steps_arr, smoothed, slope_thresh=slope_thresh)

    # 5. Ordering
    delta = None
    if tau_gen is not None and tau_F is not None:
        delta = tau_F - tau_gen
        if abs(delta) <= COINCIDENT_THRESH:
            ordering = "coincident"
        elif delta > 0:
            ordering = "G<F"
        else:
            ordering = "F<G"
    elif tau_F is not None:
        ordering = "F_only"
    elif tau_gen is not None:
        ordering = "G_only"
    else:
        ordering = "none"

    return {"tau_F": tau_F, "delta": delta, "ordering": ordering}


def _estimate_changepoint_bic(
    steps: np.ndarray,
    values: np.ndarray,
    slope_thresh: float = 0.01,
    interior_frac: float = 1/6,
) -> Optional[float]:
    """BIC 1-breakpoint segmented regression on log-step scale."""
    if len(steps) < 10:
        return None

    log_t = np.log1p(steps).astype(float)
    v = values.astype(float)
    n = len(v)
    val_range = float(v.max() - v.min())

    # No-break baseline BIC
    rss_nobreak = _ols_rss(log_t, v)
    bic_nobreak = n * np.log(max(rss_nobreak, 1e-30) / n) + 2 * np.log(n)

    skip = max(2, int(n * interior_frac))
    best_bic   = bic_nobreak
    best_step  = None

    for bp in range(skip, n - skip):
        rss_l = _ols_rss(log_t[:bp], v[:bp])
        rss_r = _ols_rss(log_t[bp:], v[bp:])
        rss   = rss_l + rss_r
        bic   = n * np.log(max(rss, 1e-30) / n) + 4 * np.log(n)
        if bic < best_bic:
            # Persistence: post-break slope must be significant
            if len(log_t[bp:]) >= 3:
                post_slope = np.polyfit(log_t[bp:], v[bp:], 1)[0]
                if abs(post_slope) > slope_thresh * val_range:
                    best_bic  = bic
                    best_step = int(steps[bp])

    return float(best_step) if best_step is not None else None


def _ols_rss(x: np.ndarray, y: np.ndarray) -> float:
    if len(x) < 2:
        return float(np.sum(y**2))
    coeffs = np.polyfit(x, y, 1)
    resid  = y - np.polyval(coeffs, x)
    return float(np.sum(resid**2))


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_heatmaps(all_results: list[dict], outdir: Path) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import matplotlib.colors as mcolors
    except ImportError:
        print("  matplotlib not available, skipping plots")
        return

    # For each cell: show ordering across (null_interval, ema_alpha) configs
    # (collapse over slope_thresh by majority vote)
    from collections import defaultdict

    cells = sorted({r["cell_label"] for r in all_results})
    ni_vals  = sorted(set(r["null_interval"]  for r in all_results))
    ema_vals = sorted(set(r["ema_alpha"]      for r in all_results), reverse=True)

    order_to_num = {"G<F": 1, "coincident": 0, "F<G": -1, "F_only": -2,
                    "G_only": 2, "none": -3}

    n_cells = len(cells)
    fig, axes = plt.subplots(2, max(1, (n_cells + 1) // 2),
                             figsize=(4 * max(1, (n_cells + 1)//2), 7),
                             squeeze=False)
    axes_flat = axes.flatten()

    for idx, cell in enumerate(cells):
        ax = axes_flat[idx]
        # Build matrix: rows=ema_alpha, cols=null_interval, value=fraction G<F
        mat = np.zeros((len(ema_vals), len(ni_vals)))
        mat_label = np.empty((len(ema_vals), len(ni_vals)), dtype=object)
        for i, ea in enumerate(ema_vals):
            for j, ni in enumerate(ni_vals):
                cell_rows = [r for r in all_results
                             if r["cell_label"] == cell
                             and r["ema_alpha"] == ea
                             and r["null_interval"] == ni]
                if not cell_rows:
                    mat[i, j] = float("nan")
                    mat_label[i, j] = "—"
                else:
                    n_gf = sum(1 for r in cell_rows if r["ordering"] == "G<F")
                    mat[i, j] = n_gf / len(cell_rows)
                    # Δτ summary
                    deltas = [r["delta"] for r in cell_rows if r["delta"] is not None]
                    med_d  = np.median(deltas) / 1000 if deltas else float("nan")
                    mat_label[i, j] = f"{mat[i,j]:.0%}\n{med_d:+.1f}k"

        im = ax.imshow(mat, vmin=0, vmax=1, cmap="RdYlGn", aspect="auto")
        ax.set_xticks(range(len(ni_vals)))
        ax.set_xticklabels([str(ni) for ni in ni_vals], fontsize=8)
        ax.set_yticks(range(len(ema_vals)))
        ax.set_yticklabels([f"α={ea}" for ea in ema_vals], fontsize=8)
        ax.set_xlabel("null_interval (steps)", fontsize=8)
        ax.set_ylabel("EMA α", fontsize=8)
        cell_cat = next((c[4] for c in CELLS if c[0] == cell), "?")
        ax.set_title(f"{cell}\n[{cell_cat}]", fontsize=8)

        for i in range(len(ema_vals)):
            for j in range(len(ni_vals)):
                ax.text(j, i, mat_label[i, j], ha="center", va="center",
                        fontsize=6.5, color="black")

        plt.colorbar(im, ax=ax, fraction=0.046, label="P(G<F)")

    # Hide unused axes
    for idx in range(len(cells), len(axes_flat)):
        axes_flat[idx].set_visible(False)

    fig.suptitle("Detector Sensitivity: P(G<F) and median Δτ\n"
                 "Rows=EMA α, Cols=null_interval. Green=G<F stable.",
                 fontsize=10)
    fig.tight_layout()
    out_path = outdir / "sensitivity_heatmap.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {out_path}")

    # Summary: fraction of configs where G<F holds, per cell
    fig2, ax2 = plt.subplots(figsize=(8, 4))
    cell_gf_frac = []
    for cell in cells:
        rows = [r for r in all_results if r["cell_label"] == cell]
        frac = sum(1 for r in rows if r["ordering"] == "G<F") / max(len(rows), 1)
        cell_gf_frac.append(frac)
        cat = next((c[4] for c in CELLS if c[0] == cell), "?")

    colors = {"strong_GF": "#4477AA", "medium_GF": "#66BBEE",
              "weak_GF": "#AADDCC", "coincident": "#DDCC77", "FG": "#CC6677"}
    bar_colors = [colors.get(next((c[4] for c in CELLS if c[0] == cell), "?"), "grey")
                  for cell in cells]

    ax2.bar(cells, cell_gf_frac, color=bar_colors)
    ax2.axhline(0.9, color="black", ls="--", lw=1, alpha=0.6, label="90%")
    ax2.set_ylabel("Fraction of 36 configs → G<F", fontsize=10)
    ax2.set_xlabel("Cell", fontsize=10)
    ax2.set_ylim(0, 1.05)
    ax2.set_title("G<F robustness across 36 detector configurations per cell", fontsize=10)
    ax2.tick_params(axis='x', rotation=30)
    ax2.legend(fontsize=9)

    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor=v, label=k) for k, v in colors.items()]
    ax2.legend(handles=legend_elements, fontsize=8, loc="lower left")

    ax2.grid(True, axis='y', alpha=0.3)
    fig2.tight_layout()
    out_path2 = outdir / "sensitivity_summary.png"
    fig2.savefig(out_path2, dpi=150, bbox_inches="tight")
    plt.close(fig2)
    print(f"  Saved {out_path2}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--outdir",       default="runs/sensitivity")
    p.add_argument("--max-steps",    type=int, default=50_000)
    p.add_argument("--log-interval", type=int, default=500)
    p.add_argument("--n-null-perms", type=int, default=200)
    return p.parse_args()


def main():
    args = parse_args()
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    configs = list(itertools.product(NULL_INTERVALS, EMA_ALPHAS, SLOPE_THRESHOLDS))
    print(f"Sensitivity Analysis")
    print(f"  {len(CELLS)} cells × {len(configs)} detector configs = "
          f"{len(CELLS) * len(configs)} analysis runs")
    print(f"  Training: {len(CELLS)} runs × {args.max_steps:,} steps\n")

    all_results: list[dict] = []
    csv_path = outdir / "results.csv"
    fieldnames = [
        "cell_label", "category", "lr", "wd", "seed",
        "observed_phase", "tau_gen",
        "null_interval", "ema_alpha", "slope_thresh",
        "tau_F", "delta", "ordering",
    ]

    # Resume: skip already-trained cells
    trained_cells: set[str] = set()
    traj_cache: dict[str, dict] = {}
    traj_cache_dir = outdir / "trajectories"
    traj_cache_dir.mkdir(exist_ok=True)

    for cell in CELLS:
        label = cell[0]
        traj_file = traj_cache_dir / f"{label}.json"
        if traj_file.exists():
            print(f"  [cache] Loading trajectory for {label}")
            traj_cache[label] = json.loads(traj_file.read_text(encoding="utf-8"))
            trained_cells.add(label)

    # Resume: load already-computed analysis results
    done_keys: set[tuple] = set()
    if csv_path.exists():
        with open(csv_path, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                all_results.append(row)
                done_keys.add((row["cell_label"], float(row["null_interval"]),
                               float(row["ema_alpha"]), float(row["slope_thresh"])))
        print(f"  Resuming: {len(done_keys)} analysis results already done\n")

    csv_file = open(csv_path, "a", newline="", encoding="utf-8")
    writer   = csv.DictWriter(csv_file, fieldnames=fieldnames)
    if not done_keys:
        writer.writeheader()

    for c_idx, (label, lr, wd, seed, category) in enumerate(CELLS):
        # Step 1: Train (or load from cache)
        if label not in traj_cache:
            print(f"[{c_idx+1}/{len(CELLS)}] Training {label} "
                  f"(lr={lr:.2e} wd={wd} seed={seed}) ...", flush=True)
            traj = run_cell_save_traj(
                lr=lr, wd=wd, seed=seed,
                max_steps=args.max_steps,
                log_interval=args.log_interval,
                n_null_perms=args.n_null_perms,
            )
            # Save trajectory
            traj_file = traj_cache_dir / f"{label}.json"
            traj_file.write_text(json.dumps(traj), encoding="utf-8")
            traj_cache[label] = traj
            print(f"  → phase={traj['observed_phase']}  tau_gen={traj['tau_gen']}  "
                  f"t={traj['runtime_sec']:.0f}s", flush=True)
        else:
            traj = traj_cache[label]
            print(f"[{c_idx+1}/{len(CELLS)}] {label} loaded from cache "
                  f"(phase={traj['observed_phase']}  tau_gen={traj['tau_gen']})")

        # Step 2: Apply all detector configs post-hoc
        n_configs = len(configs)
        for cfg_idx, (null_interval, ema_alpha, slope_thresh) in enumerate(configs):
            key = (label, float(null_interval), float(ema_alpha), float(slope_thresh))
            if key in done_keys:
                continue

            result = apply_detector(
                steps          = traj["steps"],
                f_raw          = traj["f_raw"],
                f_null_dense   = traj["f_null"],
                tau_gen        = traj["tau_gen"],
                log_interval   = args.log_interval,
                null_interval  = null_interval,
                ema_alpha      = ema_alpha,
                slope_thresh   = slope_thresh,
            )

            row = {
                "cell_label":     label,
                "category":       category,
                "lr":             lr,
                "wd":             wd,
                "seed":           seed,
                "observed_phase": traj["observed_phase"],
                "tau_gen":        traj["tau_gen"] if traj["tau_gen"] is not None else "",
                "null_interval":  null_interval,
                "ema_alpha":      ema_alpha,
                "slope_thresh":   slope_thresh,
                "tau_F":          result["tau_F"]   if result["tau_F"]   is not None else "",
                "delta":          result["delta"]   if result["delta"]   is not None else "",
                "ordering":       result["ordering"],
            }
            writer.writerow(row)
            csv_file.flush()
            all_results.append({k: str(v) for k, v in row.items()})
            done_keys.add(key)

        # Per-cell summary
        cell_rows = [r for r in all_results if r["cell_label"] == label]
        n_gf  = sum(1 for r in cell_rows if r["ordering"] == "G<F")
        n_fg  = sum(1 for r in cell_rows if r["ordering"] == "F<G")
        n_coin= sum(1 for r in cell_rows if r["ordering"] == "coincident")
        print(f"  Config results ({len(cell_rows)}/{n_configs}): "
              f"G<F={n_gf} ({n_gf/max(len(cell_rows),1):.0%})  "
              f"F<G={n_fg}  coincident={n_coin}\n")

    csv_file.close()

    # Summary table
    print(f"\n{'Cell':<20} {'Cat':<12} {'G<F':>6} {'F<G':>6} {'coin':>6} "
          f"{'τ_F=None':>9} {'med|Δτ|':>9}")
    print("-" * 70)
    cells_seen = []
    for label, lr, wd, seed, category in CELLS:
        cell_rows = [r for r in all_results if r["cell_label"] == label]
        if not cell_rows:
            continue
        cells_seen.append(label)
        n_total = len(cell_rows)
        n_gf    = sum(1 for r in cell_rows if r["ordering"] == "G<F")
        n_fg    = sum(1 for r in cell_rows if r["ordering"] == "F<G")
        n_coin  = sum(1 for r in cell_rows if r["ordering"] == "coincident")
        n_none  = sum(1 for r in cell_rows if r["tau_F"] in ("", "None", None))
        deltas  = [abs(float(r["delta"])) for r in cell_rows
                   if r["delta"] not in ("", "None", None)]
        med_d   = np.median(deltas) / 1000 if deltas else float("nan")
        print(f"{label:<20} {category:<12} {n_gf/n_total:>6.0%} "
              f"{n_fg/n_total:>6.0%} {n_coin/n_total:>6.0%} "
              f"{n_none/n_total:>9.0%} {med_d:>9.1f}k")

    print(f"\nGenerating plots...")
    plot_rows_typed = []
    for r in all_results:
        pr = dict(r)
        pr["null_interval"] = int(float(r["null_interval"])) if r["null_interval"] else 500
        pr["ema_alpha"]     = float(r["ema_alpha"]) if r["ema_alpha"] else 0.15
        pr["slope_thresh"]  = float(r["slope_thresh"]) if r["slope_thresh"] else 0.01
        pr["delta"]         = float(r["delta"]) if r["delta"] not in ("", "None", None) else None
        plot_rows_typed.append(pr)

    plot_heatmaps(plot_rows_typed, outdir)
    print(f"\nAll done. Results in {outdir}/")
    print(f"  results.csv          — per-(cell, config) ordering")
    print(f"  sensitivity_heatmap.png  — P(G<F) vs null_interval × EMA α per cell")
    print(f"  sensitivity_summary.png  — fraction of configs → G<F per cell")


if __name__ == "__main__":
    main()
