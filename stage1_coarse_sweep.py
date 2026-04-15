#!/usr/bin/env python3
"""
Stage 1: 5×5 Coarse Phase-Diagram Sweep
=========================================
Maps tau_gen, tau_F, and the G<F timing gap (Delta = tau_F - tau_gen)
across a 25-cell grid spanning the Grokking and Memorization regions.

Grid:
  lr  in [1.0e-3, 1.6e-3, 2.5e-3, 4.0e-3, 6.3e-3]   (5 values, log-spaced)
  wd  in [0.04, 0.16, 0.63, 2.5, 10]                  (5 values, log-spaced)
  seeds = [42, 7, 2025]
  max_steps = 50,000

Optimisations vs Stage 0:
  - Circle Score skipped (never reaches 0.8 threshold in Stage 0; saves ~30% runtime)
  - log_interval = 500 (100 checkpoints per run vs 500; sufficient for BIC changepoint)

Outputs (--outdir):
  results.csv         — per-(lr, wd, seed) row: observed_phase, tau_gen, tau_F, Delta, ordering
  phase_heatmap.png   — 5×5 most-common observed phase
  delta_heatmap.png   — 5×5 median(Delta) where both tau_gen and tau_F present
  timing_heatmap.png  — 5×5 median(tau_F) and median(tau_gen) side-by-side

Usage:
  python stage1_coarse_sweep.py
  python stage1_coarse_sweep.py --max-steps 50000 --outdir runs/stage1_coarse
  python stage1_coarse_sweep.py --seeds 42 --lr-idx 0 1 2  # quick partial test
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from collections import Counter
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
    compute_effective_dim,
    get_token_embeddings,
    make_dataset,
    set_seed,
    split_dataset,
)
from grok_metrics import (
    compute_fourier_alignment,
    compute_fourier_null_p95,
    estimate_changepoint,
    find_tau_sustained as _find_tau_sustained,
    classify_phase as _classify_phase,
    compute_ordering,
)

# ---------------------------------------------------------------------------
# Grid definition
# ---------------------------------------------------------------------------

LR_VALUES = [1.0e-3, 1.6e-3, 2.5e-3, 4.0e-3, 6.3e-3]
WD_VALUES = [0.04, 0.16, 0.63, 2.5, 10.0]
DEFAULT_SEEDS = [42, 7, 2025]

# ---------------------------------------------------------------------------
# Metric functions — all imported from grok_metrics.py
# compute_fourier_alignment, compute_fourier_null_p95, estimate_changepoint,
# _find_tau_sustained (as find_tau_sustained), _classify_phase (as classify_phase),
# compute_ordering
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Core run function (no Circle Score)
# ---------------------------------------------------------------------------

def run_one(
    lr: float, wd: float, seed: int,
    prime: int = 53, train_fraction: float = 0.3,
    max_steps: int = 50_000, log_interval: int = 500,
    null_interval: int = 5000, n_null_perms: int = 100,
    embed_lr: float = 1e-3, d_model: int = 256,
    n_heads: int = 4, n_layers: int = 2, d_ff: int = 1024,
) -> dict:
    """Train one (lr, wd, seed) cell; return a dict of scalar results."""
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
    optimizer = build_optimizer(
        model, decoder_lr=lr, embed_lr=embed_lr, decoder_weight_decay=wd,
    )
    scheduler = build_scheduler(
        optimizer, warmup_steps=10, lr_schedule="constant",
        lr_min_ratio=0.05, max_steps=max_steps,
    )

    steps_log: list[int] = []
    train_acc_log: list[float] = []
    test_acc_log: list[float] = []
    fourier_corr_log: list[float] = []

    null_rng = np.random.default_rng(seed ^ 0xDEAD)
    current_null95 = 0.0

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
                    tr_acc = (model(train_x.to(device)).argmax(1).cpu() == train_y).float().mean().item()
                    te_acc = (model(test_x.to(device)).argmax(1).cpu() == test_y).float().mean().item()

                emb = get_token_embeddings(model, prime)

                if step % null_interval == 0 or step == 1:
                    current_null95 = compute_fourier_null_p95(
                        emb, prime, n_perms=n_null_perms, rng=null_rng,
                    )

                f_raw, _ = compute_fourier_alignment(emb, prime)
                f_corr = max(0.0, f_raw - current_null95)

                steps_log.append(step)
                train_acc_log.append(tr_acc)
                test_acc_log.append(te_acc)
                fourier_corr_log.append(f_corr)

            if step >= max_steps:
                break

    # Post-training analysis
    observed_phase = _classify_phase(steps_log, train_acc_log, test_acc_log)
    tau_gen = _find_tau_sustained(steps_log, test_acc_log, threshold=0.9, n_sustained=3)
    tau_F = estimate_changepoint(steps_log, fourier_corr_log)

    # Delta = tau_F - tau_gen (positive = F after G, i.e. G<F ordering)
    delta = None
    if tau_gen is not None and tau_F is not None:
        delta = tau_F - tau_gen

    ordering = compute_ordering(tau_gen, tau_F)

    runtime = time.time() - t0
    return {
        "lr": lr, "wd": wd, "seed": seed,
        "observed_phase": observed_phase,
        "tau_gen": tau_gen,
        "tau_F": tau_F,
        "delta": delta,
        "ordering": ordering,
        "max_train_acc": max(train_acc_log),
        "max_test_acc": max(test_acc_log),
        "right_censored": tau_gen is None and observed_phase == "Grokking",
        "runtime_sec": runtime,
    }


# ---------------------------------------------------------------------------
# Heatmap generation
# ---------------------------------------------------------------------------

PHASE_COLORS = {
    "Grokking": "#4477AA",
    "Memorization": "#EE6677",
    "Comprehension": "#228833",
    "Confusion": "#CCBB44",
}


def make_heatmaps(rows: list[dict], lr_vals: list[float], wd_vals: list[float],
                  outdir: Path) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import matplotlib.colors as mcolors
    except ImportError:
        print("  [heatmap] matplotlib not available, skipping plots")
        return

    # Aggregate per (lr, wd): mode phase, median tau_gen, median tau_F, median delta
    from collections import defaultdict
    cell_data: dict[tuple, list[dict]] = defaultdict(list)
    for r in rows:
        cell_data[(r["lr"], r["wd"])].append(r)

    n_lr = len(lr_vals)
    n_wd = len(wd_vals)

    phase_grid = [[None] * n_wd for _ in range(n_lr)]
    tauF_grid  = [[float("nan")] * n_wd for _ in range(n_lr)]
    tauG_grid  = [[float("nan")] * n_wd for _ in range(n_lr)]
    delta_grid = [[float("nan")] * n_wd for _ in range(n_lr)]

    for i, lr in enumerate(lr_vals):
        for j, wd in enumerate(wd_vals):
            cell_rows = cell_data.get((lr, wd), [])
            if not cell_rows:
                continue
            phases = [r["observed_phase"] for r in cell_rows]
            phase_grid[i][j] = Counter(phases).most_common(1)[0][0]

            taus_F = [r["tau_F"] for r in cell_rows if r["tau_F"] is not None]
            taus_G = [r["tau_gen"] for r in cell_rows if r["tau_gen"] is not None]
            deltas = [r["delta"] for r in cell_rows if r["delta"] is not None]

            if taus_F:
                tauF_grid[i][j] = float(np.median(taus_F))
            if taus_G:
                tauG_grid[i][j] = float(np.median(taus_G))
            if deltas:
                delta_grid[i][j] = float(np.median(deltas))

    # --- Phase heatmap ---
    fig, ax = plt.subplots(figsize=(8, 6))
    unique_phases = list(PHASE_COLORS.keys())
    phase_to_int = {p: i for i, p in enumerate(unique_phases)}
    cmap = mcolors.ListedColormap([PHASE_COLORS[p] for p in unique_phases])

    img_data = np.full((n_lr, n_wd), np.nan)
    for i in range(n_lr):
        for j in range(n_wd):
            p = phase_grid[i][j]
            if p:
                img_data[i, j] = phase_to_int.get(p, 0)

    ax.imshow(img_data, cmap=cmap, vmin=-0.5, vmax=len(unique_phases) - 0.5,
              aspect="auto", origin="lower")
    ax.set_xticks(range(n_wd))
    ax.set_xticklabels([f"{w:.2g}" for w in wd_vals])
    ax.set_yticks(range(n_lr))
    ax.set_yticklabels([f"{l:.1e}" for l in lr_vals])
    ax.set_xlabel("Weight Decay (wd)")
    ax.set_ylabel("Learning Rate (lr)")
    ax.set_title("Stage 1 Coarse Grid — Observed Phase (mode across 3 seeds)")

    # Phase labels in cells
    for i in range(n_lr):
        for j in range(n_wd):
            p = phase_grid[i][j]
            if p:
                ax.text(j, i, p[:4], ha="center", va="center", fontsize=7,
                        color="white", fontweight="bold")

    # Legend
    from matplotlib.patches import Patch
    legend_patches = [Patch(color=PHASE_COLORS[p], label=p) for p in unique_phases]
    ax.legend(handles=legend_patches, loc="upper right", fontsize=8,
              bbox_to_anchor=(1.35, 1.0))
    fig.tight_layout()
    fig.savefig(outdir / "phase_heatmap.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved phase_heatmap.png")

    # --- Delta heatmap (tau_F - tau_gen) ---
    delta_arr = np.array(delta_grid, dtype=float)
    valid_deltas = delta_arr[~np.isnan(delta_arr)]

    if len(valid_deltas) > 0:
        vmax = float(np.nanpercentile(np.abs(delta_arr), 95))
        fig, ax = plt.subplots(figsize=(8, 6))
        im = ax.imshow(delta_arr, cmap="RdBu_r", vmin=-vmax, vmax=vmax,
                       aspect="auto", origin="lower")
        plt.colorbar(im, ax=ax, label="Δ = tau_F − tau_gen (steps)\nPositive = G<F ordering")
        ax.set_xticks(range(n_wd))
        ax.set_xticklabels([f"{w:.2g}" for w in wd_vals])
        ax.set_yticks(range(n_lr))
        ax.set_yticklabels([f"{l:.1e}" for l in lr_vals])
        ax.set_xlabel("Weight Decay (wd)")
        ax.set_ylabel("Learning Rate (lr)")
        ax.set_title("Stage 1 — Timing Gap Δ = tau_F − tau_gen (median across 3 seeds)\nRed=G<F (generalization first), Blue=F<G (structure first)")

        for i in range(n_lr):
            for j in range(n_wd):
                v = delta_grid[i][j]
                if not np.isnan(v):
                    ax.text(j, i, f"{v/1000:.1f}k", ha="center", va="center",
                            fontsize=7, color="white" if abs(v) > vmax * 0.4 else "black")

        fig.tight_layout()
        fig.savefig(outdir / "delta_heatmap.png", dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved delta_heatmap.png")

    # --- Timing heatmap: tau_gen and tau_F side-by-side ---
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for ax, grid, title, cmap_name in [
        (axes[0], tauG_grid, "tau_gen (median, steps)", "YlOrRd"),
        (axes[1], tauF_grid, "tau_F (median, steps)", "YlGn"),
    ]:
        arr = np.array(grid, dtype=float)
        vmax = float(np.nanmax(arr)) if not np.all(np.isnan(arr)) else 50000
        im = ax.imshow(arr, cmap=cmap_name, vmin=0, vmax=vmax,
                       aspect="auto", origin="lower")
        plt.colorbar(im, ax=ax, label="steps")
        ax.set_xticks(range(n_wd))
        ax.set_xticklabels([f"{w:.2g}" for w in wd_vals])
        ax.set_yticks(range(n_lr))
        ax.set_yticklabels([f"{l:.1e}" for l in lr_vals])
        ax.set_xlabel("Weight Decay (wd)")
        ax.set_ylabel("Learning Rate (lr)")
        ax.set_title(title)
        for i in range(n_lr):
            for j in range(n_wd):
                v = grid[i][j]
                if not np.isnan(v):
                    ax.text(j, i, f"{v/1000:.0f}k", ha="center", va="center",
                            fontsize=7, color="white" if v > vmax * 0.6 else "black")

    fig.suptitle("Stage 1 — Onset Timing across 5×5 Grid", fontsize=12)
    fig.tight_layout()
    fig.savefig(outdir / "timing_heatmap.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved timing_heatmap.png")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--outdir",     default="runs/stage1_coarse")
    p.add_argument("--max-steps",  type=int,   default=50_000)
    p.add_argument("--seeds",      type=int,   nargs="+", default=DEFAULT_SEEDS)
    p.add_argument("--lr-idx",     type=int,   nargs="+", default=list(range(len(LR_VALUES))),
                   help="Indices into LR_VALUES (0-4); subset for partial runs")
    p.add_argument("--wd-idx",     type=int,   nargs="+", default=list(range(len(WD_VALUES))),
                   help="Indices into WD_VALUES (0-4); subset for partial runs")
    return p.parse_args()


def main():
    args = parse_args()
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    lr_vals = [LR_VALUES[i] for i in args.lr_idx]
    wd_vals = [WD_VALUES[i] for i in args.wd_idx]
    seeds = args.seeds

    cells = [(lr, wd) for lr in lr_vals for wd in wd_vals]
    total_runs = len(cells) * len(seeds)

    print(f"Stage 1 Coarse Sweep: {len(lr_vals)}×{len(wd_vals)} grid × {len(seeds)} seeds = {total_runs} runs")
    print(f"Max steps: {args.max_steps:,} | Log interval: 500 | No Circle Score")
    print(f"Output: {outdir}\n")

    csv_path = outdir / "results.csv"
    fieldnames = [
        "lr", "wd", "seed", "observed_phase",
        "tau_gen", "tau_F", "delta", "ordering",
        "max_train_acc", "max_test_acc", "right_censored", "runtime_sec",
    ]

    # Resume: collect already-done (lr, wd, seed) combos
    done = set()
    all_rows: list[dict] = []
    if csv_path.exists():
        with open(csv_path, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                all_rows.append(row)
                done.add((float(row["lr"]), float(row["wd"]), int(row["seed"])))
        print(f"  Resuming: {len(done)} runs already done, {total_runs - len(done)} remaining\n")

    csv_file = open(csv_path, "a", newline="", encoding="utf-8")
    writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
    if not done:
        writer.writeheader()

    run_idx = 0
    for lr, wd in cells:
        for seed in seeds:
            run_idx += 1
            if (lr, wd, seed) in done:
                print(f"  [{run_idx}/{total_runs}] lr={lr:.1e} wd={wd} seed={seed} — SKIPPED (already done)")
                continue
            print(f"[{run_idx}/{total_runs}] lr={lr:.1e} wd={wd} seed={seed} ...", end="", flush=True)
            result = run_one(lr=lr, wd=wd, seed=seed, max_steps=args.max_steps, log_interval=500)
            tau_gen_str = f"{result['tau_gen']:.0f}" if result["tau_gen"] is not None else ""
            tau_F_str   = f"{result['tau_F']:.0f}"   if result["tau_F"]   is not None else ""
            delta_str   = f"{result['delta']:.0f}"   if result["delta"]   is not None else ""
            print(f"  phase={result['observed_phase']} | tau_gen={tau_gen_str or '—'} | "
                  f"tau_F={tau_F_str or '—'} | Δ={delta_str or '—'} | "
                  f"ordering={result['ordering']} | {result['runtime_sec']:.0f}s")

            row = {
                "lr": lr, "wd": wd, "seed": seed,
                "observed_phase": result["observed_phase"],
                "tau_gen": result["tau_gen"] if result["tau_gen"] is not None else "",
                "tau_F":   result["tau_F"]   if result["tau_F"]   is not None else "",
                "delta":   result["delta"]   if result["delta"]   is not None else "",
                "ordering": result["ordering"],
                "max_train_acc": f"{result['max_train_acc']:.4f}",
                "max_test_acc":  f"{result['max_test_acc']:.4f}",
                "right_censored": result["right_censored"],
                "runtime_sec": f"{result['runtime_sec']:.1f}",
            }
            writer.writerow(row)
            csv_file.flush()
            all_rows.append({k: str(v) for k, v in row.items()})

    csv_file.close()

    # Convert all_rows back to typed dicts for heatmap
    typed_rows = []
    for r in all_rows:
        typed_rows.append({
            "lr": float(r["lr"]),
            "wd": float(r["wd"]),
            "seed": int(r["seed"]),
            "observed_phase": r["observed_phase"],
            "tau_gen": float(r["tau_gen"]) if r["tau_gen"] else None,
            "tau_F":   float(r["tau_F"])   if r["tau_F"]   else None,
            "delta":   float(r["delta"])   if r["delta"]   else None,
            "ordering": r["ordering"],
        })

    print(f"\nAll {total_runs} runs complete. Generating heatmaps ...")
    make_heatmaps(typed_rows, lr_vals, wd_vals, outdir)

    # Print summary table
    print("\n=== Summary ===")
    print(f"{'lr':>10} {'wd':>6} | {'Phases':30} | {'G<F frac':>10} | {'median Δ':>10}")
    print("-" * 70)
    from collections import defaultdict
    by_cell: dict[tuple, list] = defaultdict(list)
    for r in typed_rows:
        by_cell[(r["lr"], r["wd"])].append(r)
    for lr in lr_vals:
        for wd in wd_vals:
            cell_rows = by_cell.get((lr, wd), [])
            if not cell_rows:
                continue
            phase_str = "/".join(Counter(r["observed_phase"] for r in cell_rows).keys())
            n_gf = sum(1 for r in cell_rows if r["ordering"] == "G<F")
            n_total = len(cell_rows)
            deltas = [r["delta"] for r in cell_rows if r["delta"] is not None]
            med_delta = f"{np.median(deltas)/1000:.1f}k" if deltas else "—"
            print(f"{lr:>10.1e} {wd:>6.2g} | {phase_str:30} | {n_gf}/{n_total:>8} | {med_delta:>10}")

    print(f"\nResults saved to {csv_path}")


if __name__ == "__main__":
    main()
