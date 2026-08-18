#!/usr/bin/env python3
"""
Stage 5A — F_p* recompute sweep
================================

Re-runs the same Stage 5A grid (lr=1.6e-3, wd in [1.2, 3.5] x10, seeds {42, 7, 2025};
30 cells total) on (a*b) mod 53, but evaluates test accuracy on TWO disjoint
splits in parallel:

  * full_test  : standard 70/30 random split over all p*p input pairs
                 (same as the original Stage 5A pipeline; matches the paper's
                 reported tau_gen)
  * fpstar_test: only the test pairs with both a != 0 and b != 0 (i.e.,
                 the F_p^* subgrid that F_corr and F_logit are computed on)

For each cell we record both
  tau_gen_full     := first checkpoint where full_test_acc   >= 0.9 sustained 3x
  tau_gen_fpstar   := first checkpoint where fpstar_test_acc >= 0.9 sustained 3x
and re-derive Delta_tau on each split:
  delta_full   = tau_F - tau_gen_full
  delta_fpstar = tau_F - tau_gen_fpstar

Output: results/stage5a_fpstar/results.csv with both numbers per cell.

Runtime estimate: ~30 cells * 20-45 min = 12-22 GPU-hours on RTX 4080.

Usage:
  .venv/Scripts/python.exe src/sweeps/stage5a_fpstar_recompute.py
  .venv/Scripts/python.exe src/sweeps/stage5a_fpstar_recompute.py --wd-idx 0 1 --seeds 42
  .venv/Scripts/python.exe src/sweeps/stage5a_fpstar_recompute.py --smoke
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
    compute_fourier_alignment_mul,
    compute_fourier_null_p95_mul,
    compute_ordering,
    estimate_changepoint,
    find_tau_sustained,
)

OPERATION = "mul"
PRIME = 53
FIXED_LR = 1.6e-3
WD_VALUES = list(np.round(np.logspace(np.log10(1.2), np.log10(3.5), 10), 4).tolist())
DEFAULT_SEEDS = [42, 7, 2025]


def _split_test_into_full_and_fpstar(test_x: torch.Tensor, test_y: torch.Tensor):
    """Return (full_x, full_y, fpstar_x, fpstar_y) where fpstar excludes a=0 or b=0."""
    a = test_x[:, 0]
    b = test_x[:, 1]
    nonzero = (a != 0) & (b != 0)
    return test_x, test_y, test_x[nonzero], test_y[nonzero]


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
    full_x, full_y, fps_x, fps_y = _split_test_into_full_and_fpstar(test_x, test_y)
    fps_count = int(fps_x.shape[0])
    full_count = int(full_x.shape[0])

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

    steps_log = []
    train_acc_log = []
    full_test_acc_log = []
    fps_test_acc_log = []
    fourier_corr_log = []
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
                    tr_logits = model(train_x.to(device)).argmax(1).cpu()
                    tr_acc = float((tr_logits == train_y).float().mean().item())
                    full_logits = model(full_x.to(device)).argmax(1).cpu()
                    full_acc = float((full_logits == full_y).float().mean().item())
                    if fps_count > 0:
                        fps_logits = model(fps_x.to(device)).argmax(1).cpu()
                        fps_acc = float((fps_logits == fps_y).float().mean().item())
                    else:
                        fps_acc = float("nan")

                emb = get_token_embeddings(model, prime)
                if step % null_interval == 0 or step == 1:
                    current_null95 = compute_fourier_null_p95_mul(
                        emb, prime, n_perms=n_null_perms, rng=null_rng,
                    )
                f_raw, _ = compute_fourier_alignment_mul(emb, prime)
                f_corr = max(0.0, f_raw - current_null95)

                steps_log.append(step)
                train_acc_log.append(tr_acc)
                full_test_acc_log.append(full_acc)
                fps_test_acc_log.append(fps_acc)
                fourier_corr_log.append(f_corr)

            if step >= max_steps:
                break

        if step >= max_steps and not extended:
            tau_gen_check = find_tau_sustained(steps_log, full_test_acc_log,
                                               threshold=0.9, n_sustained=3)
            if (train_acc_log and train_acc_log[-1] > 0.9
                    and tau_gen_check is None and max_steps < 80_000):
                print(f"    [extend] train_acc={train_acc_log[-1]:.3f}, extending to 80k")
                max_steps = 80_000
                extended = True
            else:
                break
        elif step >= max_steps:
            break

    observed_phase = classify_phase(steps_log, train_acc_log, full_test_acc_log)
    tau_gen_full = find_tau_sustained(steps_log, full_test_acc_log,
                                      threshold=0.9, n_sustained=3)
    tau_gen_fpstar = find_tau_sustained(steps_log, fps_test_acc_log,
                                        threshold=0.9, n_sustained=3)
    tau_F = estimate_changepoint(steps_log, fourier_corr_log)

    delta_full = ((tau_F - tau_gen_full)
                  if (tau_gen_full is not None and tau_F is not None) else None)
    delta_fpstar = ((tau_F - tau_gen_fpstar)
                    if (tau_gen_fpstar is not None and tau_F is not None) else None)
    ordering_full = compute_ordering(tau_gen_full, tau_F)
    ordering_fpstar = compute_ordering(tau_gen_fpstar, tau_F)

    return {
        "operation": OPERATION, "prime": prime,
        "lr": lr, "wd": wd, "seed": seed,
        "observed_phase": observed_phase,
        "n_test_full": full_count,
        "n_test_fpstar": fps_count,
        "tau_gen_full": tau_gen_full,
        "tau_gen_fpstar": tau_gen_fpstar,
        "tau_F": tau_F,
        "delta_full": delta_full,
        "delta_fpstar": delta_fpstar,
        "ordering_full": ordering_full,
        "ordering_fpstar": ordering_fpstar,
        "max_train_acc": max(train_acc_log) if train_acc_log else 0.0,
        "max_test_acc_full": max(full_test_acc_log) if full_test_acc_log else 0.0,
        "max_test_acc_fpstar":
            max(v for v in fps_test_acc_log if not np.isnan(v))
            if any(not np.isnan(v) for v in fps_test_acc_log) else 0.0,
        "runtime_sec": time.time() - t0,
    }


def _write_csv(rows, path):
    if not rows:
        return
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"  [csv] {path}")


def _load_csv(path):
    if not Path(path).exists():
        return []
    with open(path, encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _key(r):
    return (round(float(r["lr"]), 6), round(float(r["wd"]), 6), int(r["seed"]))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--outdir", default="results/stage5a_fpstar")
    parser.add_argument("--seeds", nargs="+", type=int, default=DEFAULT_SEEDS)
    parser.add_argument("--wd-idx", nargs="+", type=int, default=None)
    parser.add_argument("--max-steps", type=int, default=50_000)
    parser.add_argument("--smoke", action="store_true",
                        help="Run only 1 cell x 1 seed at 5k steps for sanity check.")
    args = parser.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    csv_path = outdir / "results.csv"

    wd_vals = ([WD_VALUES[i] for i in args.wd_idx]
               if args.wd_idx else WD_VALUES)
    seeds = list(args.seeds)
    max_steps = args.max_steps
    if args.smoke:
        wd_vals = wd_vals[:1]
        seeds = seeds[:1]
        max_steps = 5000

    existing = _load_csv(csv_path)
    done = {_key(r) for r in existing}
    rows = list(existing)

    n_total = len(wd_vals) * len(seeds)
    n_done = sum(1 for k in done
                 if any(k == (round(FIXED_LR, 6), round(wd, 6), s)
                        for wd in wd_vals for s in seeds))
    print(f"Total cells: {n_total}; resuming with {n_done} already done.")

    for wd in wd_vals:
        for seed in seeds:
            k = (round(FIXED_LR, 6), round(wd, 6), seed)
            if k in done:
                print(f"  [skip] wd={wd}, seed={seed} (already done)")
                continue
            print(f"[run] lr={FIXED_LR:.2e} wd={wd:.4f} seed={seed} ...",
                  flush=True)
            res = run_one(FIXED_LR, wd, seed, max_steps=max_steps)
            print(f"      tau_gen_full={res['tau_gen_full']} "
                  f"tau_gen_fpstar={res['tau_gen_fpstar']} "
                  f"tau_F={res['tau_F']} "
                  f"order_full={res['ordering_full']} "
                  f"order_fpstar={res['ordering_fpstar']} "
                  f"runtime={res['runtime_sec']:.0f}s",
                  flush=True)
            rows.append(res)
            _write_csv(rows, csv_path)

    if not rows:
        print("No rows written.")
        return

    n_diff_order = sum(
        1 for r in rows
        if r.get("ordering_full") != r.get("ordering_fpstar")
        and r.get("ordering_full") and r.get("ordering_fpstar")
    )
    print(f"\nSummary:  {len(rows)} cells; "
          f"ordering disagrees between full and F_p* in {n_diff_order} cell(s).")
    print(f"CSV -> {csv_path}")


if __name__ == "__main__":
    main()
