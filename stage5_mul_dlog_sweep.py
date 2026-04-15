#!/usr/bin/env python3
"""
Stage 5A: (a·b) mod 53 — Multiplication with Discrete-Log Fourier Metric
=========================================================================
Replicates the Stage-2 wd sweep on the *multiplication* task, using the
discrete-log-reindexed Fourier alignment metric so that F_corr is
correctly calibrated for the multiplicative cyclic group structure.

Grid (matches Stage 2):
  operation = "mul", prime = 53
  lr = 1.6e-3 (fixed)
  wd = 10 log-spaced values in [1.2, 3.5]
  seeds = [42, 7, 2025]
  max_steps = 50,000 (auto-extend to 80k if still grokking)

Usage:
  python stage5_mul_dlog_sweep.py
  python stage5_mul_dlog_sweep.py --wd-idx 0 1 2 --seeds 42
"""

from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path

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
    compute_fourier_alignment_mul,
    compute_fourier_null_p95_mul,
    compute_ordering,
    estimate_changepoint,
    find_tau_sustained,
)

# ---------------------------------------------------------------------------
# Grid
# ---------------------------------------------------------------------------

OPERATION = "mul"
PRIME = 53
FIXED_LR = 1.6e-3
WD_VALUES = list(np.round(np.logspace(np.log10(1.2), np.log10(3.5), 10), 4).tolist())
DEFAULT_SEEDS = [42, 7, 2025]


# ---------------------------------------------------------------------------
# Run one cell
# ---------------------------------------------------------------------------

def run_one(
    lr: float, wd: float, seed: int,
    prime: int = PRIME,
    train_fraction: float = 0.3,
    max_steps: int = 50_000,
    log_interval: int = 500,
    null_interval: int = 5_000,
    n_null_perms: int = 100,
    embed_lr: float = 1e-3,
    d_model: int = 256, n_heads: int = 4, n_layers: int = 2, d_ff: int = 1024,
) -> dict:
    t0 = time.time()
    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    x, y = make_dataset(prime, OPERATION, seed)
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
    optimizer = build_optimizer(model, decoder_lr=lr, embed_lr=embed_lr, decoder_weight_decay=wd)
    scheduler = build_scheduler(optimizer, warmup_steps=10, lr_schedule="constant",
                                lr_min_ratio=0.05, max_steps=max_steps)

    steps_log, train_acc_log, test_acc_log, fourier_corr_log = [], [], [], []
    null_rng = np.random.default_rng(seed ^ 0xDEAD)
    current_null95 = 0.0

    step = 0
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
                    current_null95 = compute_fourier_null_p95_mul(
                        emb, prime, n_perms=n_null_perms, rng=null_rng,
                    )

                f_raw, _ = compute_fourier_alignment_mul(emb, prime)
                f_corr = max(0.0, f_raw - current_null95)

                steps_log.append(step)
                train_acc_log.append(tr_acc)
                test_acc_log.append(te_acc)
                fourier_corr_log.append(f_corr)

            if step >= max_steps:
                break

        # Auto-extend
        if step >= max_steps and not extended:
            tau_gen_check = find_tau_sustained(steps_log, test_acc_log, threshold=0.9, n_sustained=3)
            if train_acc_log and train_acc_log[-1] > 0.9 and tau_gen_check is None and max_steps < 80_000:
                print(f"    [extend] train_acc={train_acc_log[-1]:.3f}, extending to 80k")
                max_steps = 80_000
                extended = True
            else:
                break
        elif step >= max_steps:
            break

    observed_phase = classify_phase(steps_log, train_acc_log, test_acc_log)
    tau_gen = find_tau_sustained(steps_log, test_acc_log, threshold=0.9, n_sustained=3)
    tau_F = estimate_changepoint(steps_log, fourier_corr_log)
    delta = (tau_F - tau_gen) if (tau_gen is not None and tau_F is not None) else None
    ordering = compute_ordering(tau_gen, tau_F)

    return {
        "operation": OPERATION, "prime": prime,
        "lr": lr, "wd": wd, "seed": seed,
        "observed_phase": observed_phase,
        "tau_gen": tau_gen, "tau_F": tau_F,
        "delta": delta, "ordering": ordering,
        "max_train_acc": max(train_acc_log) if train_acc_log else 0.0,
        "max_test_acc": max(test_acc_log) if test_acc_log else 0.0,
        "runtime_sec": time.time() - t0,
    }


# ---------------------------------------------------------------------------
# CSV helpers
# ---------------------------------------------------------------------------

def _write_csv(rows, path):
    if not rows:
        return
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    print(f"  [csv] {path}")

def _load_csv(path):
    if not Path(path).exists():
        return []
    with open(path) as f:
        return list(csv.DictReader(f))


# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------

def make_curves(rows, wd_vals, outdir):
    try:
        import matplotlib; matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return
    from collections import defaultdict
    wd_rows = defaultdict(list)
    for r in rows:
        wd_rows[float(r["wd"])].append(r)

    p_gf, med_delta, p_fonly = [], [], []
    for wd in wd_vals:
        cell = wd_rows.get(wd, [])
        n = len(cell)
        if n == 0:
            p_gf.append(float("nan")); med_delta.append(float("nan")); p_fonly.append(float("nan"))
            continue
        p_gf.append(sum(1 for r in cell if r["ordering"] == "G<F") / n)
        p_fonly.append(sum(1 for r in cell if r["ordering"] == "F_only") / n)
        deltas = [float(r["delta"]) for r in cell if r.get("delta") not in (None, "", "None")]
        med_delta.append(float(np.median(deltas)) if deltas else float("nan"))

    fig, axes = plt.subplots(3, 1, figsize=(7, 8), sharex=True)
    fig.suptitle(f"Stage 5A: G<F Ordering — mul mod {PRIME} (dlog metric)\nlr={FIXED_LR:.1e}", fontsize=11)
    axes[0].plot(wd_vals, p_gf, "o-", color="steelblue"); axes[0].set_ylabel("P(G<F)"); axes[0].set_ylim(-0.05, 1.1)
    axes[1].bar(range(len(wd_vals)), [d/1000 if not np.isnan(d) else 0 for d in med_delta], color="steelblue", alpha=0.7)
    axes[1].set_xticks(range(len(wd_vals))); axes[1].set_xticklabels([f"{w:.2f}" for w in wd_vals], rotation=45, fontsize=8)
    axes[1].set_ylabel("Median Δτ (k steps)"); axes[1].axhline(0, color="k", lw=0.8)
    axes[2].plot(wd_vals, p_fonly, "s--", color="darkorange"); axes[2].set_ylabel("P(F_only)"); axes[2].set_ylim(-0.05, 1.1)
    axes[2].set_xlabel("Weight decay")
    plt.tight_layout()
    out = outdir / "mul_dlog_curves.png"
    plt.savefig(out, dpi=150); plt.close()
    print(f"  [plot] {out}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--outdir", default="runs/stage5_mul_dlog")
    parser.add_argument("--seeds", nargs="+", type=int, default=DEFAULT_SEEDS)
    parser.add_argument("--wd-idx", nargs="+", type=int, default=None)
    parser.add_argument("--max-steps", type=int, default=50_000)
    args = parser.parse_args()

    outdir = Path(args.outdir); outdir.mkdir(parents=True, exist_ok=True)
    csv_path = outdir / "results_stage5_mul_dlog.csv"

    wd_vals = [WD_VALUES[i] for i in args.wd_idx] if args.wd_idx else WD_VALUES
    existing = _load_csv(csv_path)
    done_keys = {(float(r["wd"]), int(r["seed"])) for r in existing}
    rows = list(existing)

    total = len(wd_vals) * len(args.seeds)
    done = 0
    for wd in wd_vals:
        for seed in args.seeds:
            done += 1
            if (wd, seed) in done_keys:
                print(f"  [skip] wd={wd:.4f} seed={seed}"); continue
            print(f"\n[run {done}/{total}] op=mul p={PRIME} lr={FIXED_LR:.1e} wd={wd:.4f} seed={seed}")
            result = run_one(lr=FIXED_LR, wd=wd, seed=seed, prime=PRIME, max_steps=args.max_steps)
            rows.append(result); _write_csv(rows, csv_path)
            print(f"  phase={result['observed_phase']}  ordering={result['ordering']}  "
                  f"tau_gen={result['tau_gen']}  tau_F={result['tau_F']}  delta={result['delta']}  "
                  f"max_test={result['max_test_acc']:.3f}  t={result['runtime_sec']:.0f}s")

    grok = [r for r in rows if r.get("observed_phase") == "Grokking"]
    if grok:
        n_gf = sum(1 for r in grok if r["ordering"] == "G<F")
        deltas = [float(r["delta"]) for r in grok if r.get("delta") not in (None,"","None")]
        print(f"\n=== Stage 5A Summary: mul mod {PRIME} (dlog metric) ===")
        print(f"  Grokking: {len(grok)}/{len(rows)}")
        print(f"  G<F: {n_gf}/{len(grok)} ({100*n_gf/len(grok):.1f}%)")
        print(f"  Median Δτ: {int(np.median(deltas)) if deltas else 'N/A'} steps")
    else:
        from collections import Counter
        print(f"\n=== Stage 5A: No Grokking in {len(rows)} runs ===")
        print(dict(Counter(r.get("observed_phase","?") for r in rows)))

    make_curves(rows, WD_VALUES, outdir)
    print(f"\nDone. Results: {csv_path}")

if __name__ == "__main__":
    main()
