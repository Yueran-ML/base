#!/usr/bin/env python3
"""
Stage 3: Fine-Grained lr Sweep at wd = 2.5
============================================
Tests robustness of the G<F ordering along the lr axis.

Stage 2 established that G<F holds across wd ∈ [1.2, 3.5] at fixed lr=1.6e-3.
Stage 3 fixes wd=2.5 (confirmed Grokking region) and sweeps lr ∈ [5e-4, 8e-3]
to test whether G<F is lr-invariant within the Grokking phase.

Key design decisions (from Codex review):
  - NO early stopping after tau_gen: tau_F may appear 2.5k–25.5k steps later
  - Extension rule: if train_acc > 0.9 but tau_gen=None at base_steps → extend to 80k
  - coincident category: |Δ| ≤ 500 steps (below log_interval resolution)
  - Canonical grok_metrics.py throughout (null-corrected Fourier)
  - lr=1.6e-3 is included in the sweep grid for direct Stage 2 comparison

Primary outputs (three curves vs lr):
  P(G<F)       — fraction of seeds where ordering == "G<F"
  median Δ     — median(tau_F − tau_gen) for seeds with both tau detected
  P(F_only)    — fraction of seeds where ordering == "F_only"
  phase_counts — fraction per phase (Grokking / Memorization / Comprehension)

Grid:
  wd   = 2.5   (fixed)
  lr   = 10 log-spaced values in [5e-4, 8e-3]
  seeds = [42, 7, 2025]
  base_steps = 50,000 | extension to 80k if needed

Usage:
  python stage3_lr_sweep.py
  python stage3_lr_sweep.py --outdir runs/stage3_lr --seeds 42 7 2025
  python stage3_lr_sweep.py --lr-idx 0 1 2 3      # partial run
  python stage3_lr_sweep.py --base-steps 500 --log-interval 200 --seeds 42  # smoketest
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
    estimate_changepoint,
    find_tau_sustained,
)

# ---------------------------------------------------------------------------
# Grid definition
# ---------------------------------------------------------------------------

FIXED_WD: float = 2.5
LR_VALUES: list[float] = list(
    np.round(np.logspace(np.log10(5e-4), np.log10(8e-3), 10), 7).tolist()
)
DEFAULT_SEEDS: list[int] = [42, 7, 2025]

# |Δ| <= COINCIDENT_THRESH → label "coincident" (below log_interval resolution)
COINCIDENT_THRESH: int = 500

# ---------------------------------------------------------------------------
# Ordering with coincident category
# ---------------------------------------------------------------------------

def compute_ordering_stage3(
    tau_gen: Optional[float],
    tau_F: Optional[float],
) -> str:
    """Like grok_metrics.compute_ordering but adds 'coincident' for |Δ|<=500."""
    if tau_gen is not None and tau_F is not None:
        delta = tau_F - tau_gen
        if abs(delta) <= COINCIDENT_THRESH:
            return "coincident"
        return "G<F" if delta > 0 else "F<G"
    if tau_F is not None:
        return "F_only"
    if tau_gen is not None:
        return "G_only"
    return "none"


# ---------------------------------------------------------------------------
# Core run function
# ---------------------------------------------------------------------------

def run_one(
    lr: float,
    wd: float,
    seed: int,
    prime: int = 53,
    train_fraction: float = 0.3,
    base_steps: int = 50_000,
    extend_steps: int = 80_000,
    log_interval: int = 500,
    null_interval: int = 5000,
    n_null_perms: int = 100,
    embed_lr: float = 1e-3,
    d_model: int = 256,
    n_heads: int = 4,
    n_layers: int = 2,
    d_ff: int = 1024,
) -> dict:
    """Train one (lr, wd, seed) cell; return a dict of scalar results.

    Extension rule: if train_acc > 0.9 but tau_gen still None at base_steps,
    extend to extend_steps (right-censoring prevention for slow-grokking cells).
    No early stopping on tau_gen detection (tau_F may appear much later).
    """
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
    # Scheduler uses extend_steps as max; constant LR after warmup regardless of extension
    scheduler = build_scheduler(
        optimizer, warmup_steps=10, lr_schedule="constant",
        lr_min_ratio=0.05, max_steps=extend_steps,
    )

    steps_log: list[int] = []
    train_acc_log: list[float] = []
    test_acc_log: list[float] = []
    fourier_corr_log: list[float] = []

    null_rng = np.random.default_rng(seed ^ 0xDEAD)
    current_null95 = 0.0

    step = 0
    max_steps = base_steps
    extended = False

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

                # Extension check: at base_steps, if memorizing but not yet grokking
                if step == base_steps and not extended:
                    tau_gen_so_far = find_tau_sustained(steps_log, test_acc_log, 0.9)
                    if tau_gen_so_far is None and max(train_acc_log[-5:], default=0) > 0.9:
                        max_steps = extend_steps
                        extended = True
                        print(f"    [extend→{extend_steps}k] train_acc>{0.9:.0%}, tau_gen=None", flush=True)

            if step >= max_steps:
                break

    # Post-training analysis (canonical grok_metrics pipeline)
    observed_phase = classify_phase(steps_log, train_acc_log, test_acc_log)
    tau_gen = find_tau_sustained(steps_log, test_acc_log, threshold=0.9, n_sustained=3)
    tau_F = estimate_changepoint(steps_log, fourier_corr_log)

    delta = (tau_F - tau_gen) if (tau_gen is not None and tau_F is not None) else None
    ordering = compute_ordering_stage3(tau_gen, tau_F)

    return {
        "lr": lr, "wd": wd, "seed": seed,
        "observed_phase": observed_phase,
        "tau_gen": tau_gen,
        "tau_F": tau_F,
        "delta": delta,
        "ordering": ordering,
        "max_train_acc": max(train_acc_log),
        "max_test_acc": max(test_acc_log),
        "extended": extended,
        "steps_used": step,
        "runtime_sec": time.time() - t0,
    }


# ---------------------------------------------------------------------------
# Curve plotting (four panels: P(G<F), median Δ, P(F_only), phase_frac vs lr)
# ---------------------------------------------------------------------------

def make_curves(rows: list[dict], lr_vals: list[float], outdir: Path) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("  [plot] matplotlib not available, skipping")
        return

    from collections import defaultdict
    lr_rows: dict[float, list[dict]] = defaultdict(list)
    for r in rows:
        lr_rows[float(r["lr"])].append(r)

    p_gf      = []
    med_delta = []
    p_fonly   = []
    p_grokking = []
    n_seeds_list = []

    for lr in lr_vals:
        cell = lr_rows.get(lr, [])
        n = len(cell)
        n_seeds_list.append(n)
        if n == 0:
            for lst in [p_gf, med_delta, p_fonly, p_grokking]:
                lst.append(float("nan"))
            continue

        p_gf.append(sum(1 for r in cell if r["ordering"] == "G<F") / n)
        p_fonly.append(sum(1 for r in cell if r["ordering"] == "F_only") / n)
        p_grokking.append(sum(1 for r in cell if r["observed_phase"] == "Grokking") / n)

        deltas = [float(r["delta"]) for r in cell
                  if r["delta"] not in (None, "", "None")]
        med_delta.append(float(np.median(deltas)) if deltas else float("nan"))

    lr_arr        = np.array(lr_vals)
    p_gf_arr      = np.array(p_gf)
    med_delta_arr = np.array(med_delta)
    p_fonly_arr   = np.array(p_fonly)
    p_grok_arr    = np.array(p_grokking)

    fig, axes = plt.subplots(4, 1, figsize=(8, 13), sharex=True)

    # Panel 1: P(G<F)
    ax = axes[0]
    ax.semilogx(lr_arr, p_gf_arr, "o-", color="#4477AA", linewidth=2, markersize=7)
    ax.axhline(0.5, color="grey", linestyle="--", linewidth=0.8, alpha=0.7)
    ax.set_ylabel("P(G<F)\nfraction of seeds", fontsize=10)
    ax.set_ylim(-0.05, 1.05)
    ax.set_title(
        f"Stage 3: lr sweep at wd=2.5  (3 seeds/cell, 50k/80k steps)\n"
        "Canonical classifier: grok_gap=2000, n_sustained=3  |  null-corrected Fourier",
        fontsize=10,
    )
    ax.grid(True, alpha=0.3, which="both")
    # Mark Stage 2 fixed lr for reference
    ax.axvline(1.6e-3, color="#AA3344", linestyle=":", linewidth=1.2, alpha=0.7,
               label="Stage 2 lr (1.6e-3)")
    ax.legend(fontsize=8, loc="lower left")

    # Panel 2: median Δ = tau_F - tau_gen
    ax = axes[1]
    ax.semilogx(lr_arr, med_delta_arr / 1000, "s-", color="#EE6677", linewidth=2, markersize=7)
    ax.axhline(0, color="grey", linestyle="--", linewidth=0.8, alpha=0.7)
    ax.axvline(1.6e-3, color="#AA3344", linestyle=":", linewidth=1.2, alpha=0.7)
    ax.set_ylabel("median Δ (k steps)\ntau_F − tau_gen", fontsize=10)
    ax.grid(True, alpha=0.3, which="both")
    ax.annotate("positive = G<F (gen. first)", xy=(0.02, 0.88),
                xycoords="axes fraction", fontsize=8, color="#EE6677")

    # Panel 3: P(F_only)
    ax = axes[2]
    ax.semilogx(lr_arr, p_fonly_arr, "^-", color="#228833", linewidth=2, markersize=7)
    ax.axvline(1.6e-3, color="#AA3344", linestyle=":", linewidth=1.2, alpha=0.7)
    ax.set_ylabel("P(F_only)\nfraction of seeds", fontsize=10)
    ax.set_ylim(-0.05, 1.05)
    ax.grid(True, alpha=0.3, which="both")

    # Panel 4: P(Grokking phase)
    ax = axes[3]
    ax.semilogx(lr_arr, p_grok_arr, "D-", color="#BBAA00", linewidth=2, markersize=7)
    ax.axvline(1.6e-3, color="#AA3344", linestyle=":", linewidth=1.2, alpha=0.7)
    ax.set_ylabel("P(Grokking phase)\nfraction of seeds", fontsize=10)
    ax.set_xlabel("Learning Rate (lr)", fontsize=11)
    ax.set_ylim(-0.05, 1.05)
    ax.grid(True, alpha=0.3, which="both")

    # lr tick labels with seed count
    axes[3].set_xticks(lr_arr)
    axes[3].set_xticklabels(
        [f"{lr:.2e}\n(n={n_seeds_list[i]})" for i, lr in enumerate(lr_vals)],
        fontsize=7, rotation=30,
    )

    fig.tight_layout()
    out_path = outdir / "lr_sweep_curves.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {out_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--outdir",       default="runs/stage3_lr")
    p.add_argument("--base-steps",   type=int, default=50_000)
    p.add_argument("--extend-steps", type=int, default=80_000)
    p.add_argument("--log-interval", type=int, default=500)
    p.add_argument("--seeds",        type=int, nargs="+", default=DEFAULT_SEEDS)
    p.add_argument("--lr-idx",       type=int, nargs="+",
                   default=list(range(len(LR_VALUES))),
                   help="Indices into LR_VALUES (0-9); subset for partial runs")
    return p.parse_args()


def main():
    args = parse_args()
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    lr_vals = [LR_VALUES[i] for i in args.lr_idx]
    seeds = args.seeds
    total_runs = len(lr_vals) * len(seeds)

    print(f"Stage 3 lr Sweep: wd={FIXED_WD} | {len(lr_vals)} lr values × {len(seeds)} seeds = {total_runs} runs")
    print(f"lr values: {[f'{lr:.2e}' for lr in lr_vals]}")
    print(f"Base steps: {args.base_steps:,} | Extend to: {args.extend_steps:,} | Log interval: {args.log_interval}")
    print(f"Coincident threshold: |Δ| ≤ {COINCIDENT_THRESH} steps")
    print(f"Output: {outdir}\n")

    csv_path = outdir / "results.csv"
    fieldnames = [
        "lr", "wd", "seed", "observed_phase",
        "tau_gen", "tau_F", "delta", "ordering",
        "max_train_acc", "max_test_acc",
        "extended", "steps_used", "runtime_sec",
    ]

    # Resume: collect already-done (lr, seed) combos
    done: set[tuple] = set()
    all_rows: list[dict] = []
    if csv_path.exists():
        with open(csv_path, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                all_rows.append(row)
                done.add((float(row["lr"]), int(row["seed"])))
        print(f"  Resuming: {len(done)} runs already done, {total_runs - len(done)} remaining\n")

    csv_file = open(csv_path, "a", newline="", encoding="utf-8")
    writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
    if not done:
        writer.writeheader()

    run_idx = 0
    for lr in lr_vals:
        for seed in seeds:
            run_idx += 1
            if (lr, seed) in done:
                print(f"  [{run_idx}/{total_runs}] lr={lr:.2e} seed={seed} — SKIPPED")
                continue

            print(f"[{run_idx}/{total_runs}] lr={lr:.2e} wd={FIXED_WD} seed={seed} ...",
                  end="", flush=True)
            result = run_one(
                lr=lr, wd=FIXED_WD, seed=seed,
                base_steps=args.base_steps,
                extend_steps=args.extend_steps,
                log_interval=args.log_interval,
            )

            tau_gen_str = f"{result['tau_gen']:.0f}" if result["tau_gen"] is not None else ""
            tau_F_str   = f"{result['tau_F']:.0f}"   if result["tau_F"]   is not None else ""
            delta_str   = f"{result['delta']:.0f}"   if result["delta"]   is not None else ""
            ext_flag    = " [ext]" if result["extended"] else ""
            print(
                f"  phase={result['observed_phase']} | "
                f"tau_gen={tau_gen_str or '—'} | tau_F={tau_F_str or '—'} | "
                f"Δ={delta_str or '—'} | ordering={result['ordering']} | "
                f"{result['runtime_sec']:.0f}s{ext_flag}"
            )

            row = {
                "lr":            result["lr"],
                "wd":            FIXED_WD,
                "seed":          seed,
                "observed_phase": result["observed_phase"],
                "tau_gen":       result["tau_gen"]  if result["tau_gen"]  is not None else "",
                "tau_F":         result["tau_F"]    if result["tau_F"]    is not None else "",
                "delta":         result["delta"]    if result["delta"]    is not None else "",
                "ordering":      result["ordering"],
                "max_train_acc": f"{result['max_train_acc']:.4f}",
                "max_test_acc":  f"{result['max_test_acc']:.4f}",
                "extended":      result["extended"],
                "steps_used":    result["steps_used"],
                "runtime_sec":   f"{result['runtime_sec']:.1f}",
            }
            writer.writerow(row)
            csv_file.flush()
            all_rows.append({k: str(v) for k, v in row.items()})

    csv_file.close()

    print(f"\nAll runs complete. Generating plots...")
    plot_rows = []
    for r in all_rows:
        pr = dict(r)
        pr["delta"] = r["delta"] if r["delta"] != "" else None
        plot_rows.append(pr)

    make_curves(plot_rows, lr_vals, outdir)

    # Summary table
    from collections import defaultdict
    lr_rows_summary: dict[float, list] = defaultdict(list)
    for r in plot_rows:
        lr_rows_summary[float(r["lr"])].append(r)

    print(f"\n{'lr':>10}  {'phase_G%':>8}  {'P(G<F)':>7}  {'medΔ(k)':>9}  {'P(F_only)':>10}  {'P(coin)':>8}  n")
    print("-" * 68)
    for lr in lr_vals:
        cell = lr_rows_summary.get(lr, [])
        n = len(cell)
        if n == 0:
            print(f"{lr:>10.2e}  {'—':>8}  {'—':>7}  {'—':>9}  {'—':>10}  {'—':>8}  {n}")
            continue
        p_grok = sum(1 for r in cell if r["observed_phase"] == "Grokking") / n
        p_gf   = sum(1 for r in cell if r["ordering"] == "G<F") / n
        p_fo   = sum(1 for r in cell if r["ordering"] == "F_only") / n
        p_coin = sum(1 for r in cell if r["ordering"] == "coincident") / n
        deltas = [float(r["delta"]) for r in cell if r["delta"] not in (None, "", "None")]
        med_d  = np.median(deltas) / 1000 if deltas else float("nan")
        print(f"{lr:>10.2e}  {p_grok:>8.2f}  {p_gf:>7.2f}  {med_d:>9.1f}  {p_fo:>10.2f}  {p_coin:>8.2f}  {n}")

    print(f"\nResults saved to {csv_path}")


if __name__ == "__main__":
    main()
