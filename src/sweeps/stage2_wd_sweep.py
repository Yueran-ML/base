#!/usr/bin/env python3
"""
Stage 2: Fine-Grained wd Sweep at lr = 1.6e-3
================================================
Maps the G<F ordering boundary as a function of weight decay.

In Stage 1, all runs at wd=2.5 showed G<F ordering (generalization precedes
Fourier structure formation), while wd=0.04–0.63 showed only Memorization
and wd=10 showed Confusion. Stage 2 resolves the transition region by sweeping
wd densely in [1.2, 3.5] at fixed lr=1.6e-3.

Primary outputs (three curves vs wd):
  P(G<F)     — fraction of seeds where ordering == "G<F"
  median Δ   — median(tau_F − tau_gen) for seeds with both tau detected
  P(F_only)  — fraction of seeds where ordering == "F_only"

Grid:
  lr  = 1.6e-3  (fixed)
  wd  = 10 log-spaced values in [1.2, 3.5]
  seeds = [42, 7, 2025]
  max_steps = 50,000

Usage:
  python stage2_wd_sweep.py
  python stage2_wd_sweep.py --outdir runs/stage2_wd --seeds 42 7 2025
  python stage2_wd_sweep.py --wd-idx 0 1 2 3  # partial run
"""

from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
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
    compute_ordering,
    estimate_changepoint,
    find_tau_sustained,
)

# ---------------------------------------------------------------------------
# Grid definition
# ---------------------------------------------------------------------------

FIXED_LR: float = 1.6e-3
WD_VALUES: list[float] = list(
    np.round(np.logspace(np.log10(1.2), np.log10(3.5), 10), 4).tolist()
)
DEFAULT_SEEDS: list[int] = [42, 7, 2025]

# ---------------------------------------------------------------------------
# Core run function
# ---------------------------------------------------------------------------


def run_one(
    lr: float,
    wd: float,
    seed: int,
    prime: int = 53,
    train_fraction: float = 0.3,
    max_steps: int = 50_000,
    log_interval: int = 500,
    null_interval: int = 5000,
    n_null_perms: int = 100,
    embed_lr: float = 1e-3,
    d_model: int = 256,
    n_heads: int = 4,
    n_layers: int = 2,
    d_ff: int = 1024,
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

    # Post-training analysis (uses unified grok_metrics pipeline)
    observed_phase = classify_phase(steps_log, train_acc_log, test_acc_log)
    tau_gen = find_tau_sustained(steps_log, test_acc_log, threshold=0.9, n_sustained=3)
    tau_F = estimate_changepoint(steps_log, fourier_corr_log)

    delta = (tau_F - tau_gen) if (tau_gen is not None and tau_F is not None) else None
    ordering = compute_ordering(tau_gen, tau_F)

    return {
        "lr": lr, "wd": wd, "seed": seed,
        "observed_phase": observed_phase,
        "tau_gen": tau_gen,
        "tau_F": tau_F,
        "delta": delta,
        "ordering": ordering,
        "max_train_acc": max(train_acc_log),
        "max_test_acc": max(test_acc_log),
        "runtime_sec": time.time() - t0,
    }


# ---------------------------------------------------------------------------
# Curve plotting (three panels: P(G<F), median Δ, P(F_only) vs wd)
# ---------------------------------------------------------------------------


def make_curves(rows: list[dict], wd_vals: list[float], outdir: Path) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("  [plot] matplotlib not available, skipping")
        return

    from collections import defaultdict
    wd_rows: dict[float, list[dict]] = defaultdict(list)
    for r in rows:
        wd_rows[float(r["wd"])].append(r)

    p_gf    = []  # P(G<F) per wd
    med_delta = []  # median Δ per wd (NaN if no runs with both taus)
    p_fonly = []  # P(F_only) per wd
    n_seeds_list = []

    for wd in wd_vals:
        cell = wd_rows.get(wd, [])
        n = len(cell)
        n_seeds_list.append(n)
        if n == 0:
            p_gf.append(float("nan"))
            med_delta.append(float("nan"))
            p_fonly.append(float("nan"))
            continue

        p_gf.append(sum(1 for r in cell if r["ordering"] == "G<F") / n)
        p_fonly.append(sum(1 for r in cell if r["ordering"] == "F_only") / n)

        deltas = [float(r["delta"]) for r in cell
                  if r["delta"] not in (None, "", "None")]
        med_delta.append(float(np.median(deltas)) if deltas else float("nan"))

    wd_arr = np.array(wd_vals)
    p_gf_arr = np.array(p_gf)
    med_delta_arr = np.array(med_delta)
    p_fonly_arr = np.array(p_fonly)

    fig, axes = plt.subplots(3, 1, figsize=(8, 11), sharex=True)

    # Panel 1: P(G<F)
    ax = axes[0]
    ax.plot(wd_arr, p_gf_arr, "o-", color="#4477AA", linewidth=2, markersize=7)
    ax.axhline(0.5, color="grey", linestyle="--", linewidth=0.8, alpha=0.7)
    ax.set_ylabel("P(G<F)\nfraction of seeds", fontsize=10)
    ax.set_ylim(-0.05, 1.05)
    ax.set_title(
        f"Stage 2: wd sweep at lr=1.6e-3  (3 seeds/cell, 50k steps)\n"
        "Canonical classifier: grok_gap=2000, n_sustained=3",
        fontsize=10,
    )
    ax.grid(True, alpha=0.3)

    # Panel 2: median Δ = tau_F - tau_gen
    ax = axes[1]
    ax.plot(wd_arr, med_delta_arr / 1000, "s-", color="#EE6677", linewidth=2, markersize=7)
    ax.axhline(0, color="grey", linestyle="--", linewidth=0.8, alpha=0.7)
    ax.set_ylabel("median Δ (k steps)\ntau_F − tau_gen", fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.annotate("positive = G<F (gen. first)", xy=(0.02, 0.88),
                xycoords="axes fraction", fontsize=8, color="#EE6677")

    # Panel 3: P(F_only)
    ax = axes[2]
    ax.plot(wd_arr, p_fonly_arr, "^-", color="#228833", linewidth=2, markersize=7)
    ax.set_ylabel("P(F_only)\nfraction of seeds", fontsize=10)
    ax.set_xlabel("Weight Decay (wd)", fontsize=11)
    ax.set_ylim(-0.05, 1.05)
    ax.grid(True, alpha=0.3)

    # Label each wd tick with seed count
    axes[2].set_xticks(wd_arr)
    axes[2].set_xticklabels([f"{w:.3g}\n(n={n_seeds_list[i]})" for i, w in enumerate(wd_vals)],
                             fontsize=7, rotation=30)

    fig.tight_layout()
    out_path = outdir / "wd_sweep_curves.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {out_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--outdir",    default="runs/stage2_wd")
    p.add_argument("--max-steps", type=int, default=50_000)
    p.add_argument("--seeds",     type=int, nargs="+", default=DEFAULT_SEEDS)
    p.add_argument("--wd-idx",    type=int, nargs="+",
                   default=list(range(len(WD_VALUES))),
                   help="Indices into WD_VALUES (0-9); subset for partial runs")
    return p.parse_args()


def main():
    args = parse_args()
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    wd_vals = [WD_VALUES[i] for i in args.wd_idx]
    seeds = args.seeds
    total_runs = len(wd_vals) * len(seeds)

    print(f"Stage 2 wd Sweep: lr={FIXED_LR} | {len(wd_vals)} wd values × {len(seeds)} seeds = {total_runs} runs")
    print(f"wd values: {[f'{w:.3g}' for w in wd_vals]}")
    print(f"Max steps: {args.max_steps:,} | Log interval: 500 | Unified grok_metrics classifier")
    print(f"Output: {outdir}\n")

    csv_path = outdir / "results.csv"
    fieldnames = [
        "lr", "wd", "seed", "observed_phase",
        "tau_gen", "tau_F", "delta", "ordering",
        "max_train_acc", "max_test_acc", "runtime_sec",
    ]

    # Resume: collect already-done (wd, seed) combos
    done: set[tuple] = set()
    all_rows: list[dict] = []
    if csv_path.exists():
        with open(csv_path, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                all_rows.append(row)
                done.add((float(row["wd"]), int(row["seed"])))
        print(f"  Resuming: {len(done)} runs already done, {total_runs - len(done)} remaining\n")

    csv_file = open(csv_path, "a", newline="", encoding="utf-8")
    writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
    if not done:
        writer.writeheader()

    run_idx = 0
    for wd in wd_vals:
        for seed in seeds:
            run_idx += 1
            if (wd, seed) in done:
                print(f"  [{run_idx}/{total_runs}] wd={wd:.3g} seed={seed} — SKIPPED")
                continue

            print(f"[{run_idx}/{total_runs}] lr={FIXED_LR:.1e} wd={wd:.4g} seed={seed} ...",
                  end="", flush=True)
            result = run_one(lr=FIXED_LR, wd=wd, seed=seed, max_steps=args.max_steps)

            tau_gen_str = f"{result['tau_gen']:.0f}" if result["tau_gen"] is not None else ""
            tau_F_str   = f"{result['tau_F']:.0f}"   if result["tau_F"]   is not None else ""
            delta_str   = f"{result['delta']:.0f}"   if result["delta"]   is not None else ""
            print(
                f"  phase={result['observed_phase']} | "
                f"tau_gen={tau_gen_str or '—'} | tau_F={tau_F_str or '—'} | "
                f"Δ={delta_str or '—'} | ordering={result['ordering']} | "
                f"{result['runtime_sec']:.0f}s"
            )

            row = {
                "lr": FIXED_LR,
                "wd": wd,
                "seed": seed,
                "observed_phase": result["observed_phase"],
                "tau_gen":  result["tau_gen"]  if result["tau_gen"]  is not None else "",
                "tau_F":    result["tau_F"]    if result["tau_F"]    is not None else "",
                "delta":    result["delta"]    if result["delta"]    is not None else "",
                "ordering": result["ordering"],
                "max_train_acc": f"{result['max_train_acc']:.4f}",
                "max_test_acc":  f"{result['max_test_acc']:.4f}",
                "runtime_sec":   f"{result['runtime_sec']:.1f}",
            }
            writer.writerow(row)
            csv_file.flush()
            all_rows.append({k: str(v) for k, v in row.items()})

    csv_file.close()

    print(f"\nAll runs complete. Generating plots...")
    # Convert ordering field back for plotting
    plot_rows = []
    for r in all_rows:
        pr = dict(r)
        pr["delta"] = r["delta"] if r["delta"] != "" else None
        pr["ordering"] = r["ordering"]
        plot_rows.append(pr)

    make_curves(plot_rows, wd_vals, outdir)

    # Print summary table
    print(f"\n{'wd':>8}  {'P(G<F)':>8}  {'medΔ(k)':>9}  {'P(F_only)':>10}  n_seeds")
    print("-" * 55)
    from collections import defaultdict
    wd_rows: dict[float, list] = defaultdict(list)
    for r in plot_rows:
        wd_rows[float(r["wd"])].append(r)

    for wd in wd_vals:
        cell = wd_rows.get(wd, [])
        n = len(cell)
        if n == 0:
            print(f"{wd:>8.3g}  {'—':>8}  {'—':>9}  {'—':>10}  {n}")
            continue
        p_gf = sum(1 for r in cell if r["ordering"] == "G<F") / n
        p_fo = sum(1 for r in cell if r["ordering"] == "F_only") / n
        deltas = [float(r["delta"]) for r in cell if r["delta"] not in (None, "", "None")]
        med_d = np.median(deltas) / 1000 if deltas else float("nan")
        print(f"{wd:>8.3g}  {p_gf:>8.2f}  {med_d:>9.1f}  {p_fo:>10.2f}  {n}")

    print(f"\nResults saved to {csv_path}")


if __name__ == "__main__":
    main()
