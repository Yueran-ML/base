#!/usr/bin/env python3
"""
Stage 6: Full 2D (lr × wd) Phase Diagram
==========================================
Sweeps a 7×7 grid of (decoder_lr, decoder_wd) within the confirmed Grokking
region, to answer: "Is G<F ordering a broad property of the Grokking interior,
or is it concentrated near the Stage 2/3 operating points?"

Grid:
  operation = "add", prime = 53
  lr  = 7 log-spaced values in [5e-4, 4.3e-3]  (Stage 3 range)
  wd  = 7 log-spaced values in [1.2, 3.5]       (Stage 2 range)
  seeds = [42, 7, 2025]
  max_steps = 50,000 (auto-extend to 80k if train_acc > 0.9 but tau_gen=None)
  Total: 7 × 7 × 3 = 147 cells

Parallelisation (A800 recommended):
  --parallel N  runs N cells simultaneously using ProcessPoolExecutor
  A800 (80 GB) can safely handle N=8~12 at once (each cell ~400 MB VRAM)

Checkpoint resume:
  Each cell writes to CSV immediately on completion.
  Restart resumes from where it left off (done_keys check).

Usage:
  python stage6_2d_sweep.py                        # single-process
  python stage6_2d_sweep.py --parallel 8           # 8 workers (A800)
  python stage6_2d_sweep.py --parallel 10 --seeds 42  # smoke-test subset
"""

from __future__ import annotations

import argparse
import csv
import fcntl
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

# ---------------------------------------------------------------------------
# Sys-path for local imports
# ---------------------------------------------------------------------------
import sys
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
# Grid
# ---------------------------------------------------------------------------

LR_VALUES: list[float] = list(
    np.round(np.logspace(np.log10(5e-4), np.log10(4.3e-3), 7), 6).tolist()
)
WD_VALUES: list[float] = list(
    np.round(np.logspace(np.log10(1.2), np.log10(3.5), 7), 4).tolist()
)
DEFAULT_SEEDS: list[int] = [42, 7, 2025]
PRIME: int = 53
OPERATION: str = "add"
BASE_STEPS: int = 50_000
EXTEND_STEPS: int = 80_000


# ---------------------------------------------------------------------------
# Single-cell training
# ---------------------------------------------------------------------------

def run_one(
    lr: float,
    wd: float,
    seed: int,
    prime: int = PRIME,
    train_fraction: float = 0.3,
    max_steps: int = BASE_STEPS,
    log_interval: int = 500,
    null_interval: int = 5_000,
    n_null_perms: int = 100,
    embed_lr: float = 1e-3,
    d_model: int = 256,
    n_heads: int = 4,
    n_layers: int = 2,
    d_ff: int = 1024,
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
    optimizer = build_optimizer(model, decoder_lr=lr, embed_lr=embed_lr,
                                decoder_weight_decay=wd)
    scheduler = build_scheduler(optimizer, warmup_steps=10,
                                lr_schedule="constant", lr_min_ratio=0.05,
                                max_steps=max_steps)

    steps_log, train_acc_log, test_acc_log, fourier_corr_log = [], [], [], []
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
                    tr_acc = (model(train_x.to(device)).argmax(1).cpu()
                              == train_y).float().mean().item()
                    te_acc = (model(test_x.to(device)).argmax(1).cpu()
                              == test_y).float().mean().item()

                emb = get_token_embeddings(model, prime)
                if step % null_interval == 0 or step == 1:
                    current_null95 = compute_fourier_null_p95(
                        emb, prime, n_perms=n_null_perms, rng=null_rng)
                f_raw, _ = compute_fourier_alignment(emb, prime)
                f_corr = max(0.0, f_raw - current_null95)

                steps_log.append(step)
                train_acc_log.append(tr_acc)
                test_acc_log.append(te_acc)
                fourier_corr_log.append(f_corr)

            if step >= max_steps:
                break

    # Auto-extend: if train converged but tau_gen not yet found, run to 80k
    max_train = max(train_acc_log) if train_acc_log else 0.0
    tau_gen = find_tau_sustained(steps_log, test_acc_log, threshold=0.9, n_sustained=3)
    if max_train > 0.9 and tau_gen is None and max_steps < EXTEND_STEPS:
        return run_one(lr=lr, wd=wd, seed=seed, prime=prime,
                       train_fraction=train_fraction, max_steps=EXTEND_STEPS,
                       log_interval=log_interval, null_interval=null_interval,
                       n_null_perms=n_null_perms, embed_lr=embed_lr,
                       d_model=d_model, n_heads=n_heads, n_layers=n_layers,
                       d_ff=d_ff)

    observed_phase = classify_phase(steps_log, train_acc_log, test_acc_log)
    tau_F = estimate_changepoint(steps_log, fourier_corr_log)
    delta = (tau_F - tau_gen) if (tau_gen is not None and tau_F is not None) else None
    ordering = compute_ordering(tau_gen, tau_F)

    return {
        "operation": OPERATION, "prime": prime,
        "lr": lr, "wd": wd, "seed": seed,
        "observed_phase": observed_phase,
        "tau_gen": tau_gen, "tau_F": tau_F,
        "delta": delta, "ordering": ordering,
        "max_train_acc": max_train,
        "max_test_acc": max(test_acc_log) if test_acc_log else 0.0,
        "max_steps_used": max_steps if not (max_train > 0.9 and tau_gen is None) else EXTEND_STEPS,
        "runtime_sec": time.time() - t0,
    }


# ---------------------------------------------------------------------------
# CSV helpers (file-lock safe for parallel writes)
# ---------------------------------------------------------------------------

FIELDNAMES = [
    "operation", "prime", "lr", "wd", "seed",
    "observed_phase", "tau_gen", "tau_F", "delta", "ordering",
    "max_train_acc", "max_test_acc", "max_steps_used", "runtime_sec",
]


def _append_row(path: Path, row: dict) -> None:
    """Append one result row to CSV with file lock (parallel-safe)."""
    write_header = not path.exists() or path.stat().st_size == 0
    with open(path, "a", newline="") as f:
        # File lock (Unix); on Windows this is a no-op but writes are atomic enough
        try:
            fcntl.flock(f, fcntl.LOCK_EX)
        except Exception:
            pass
        w = csv.DictWriter(f, fieldnames=FIELDNAMES)
        if write_header:
            w.writeheader()
        w.writerow(row)
        try:
            fcntl.flock(f, fcntl.LOCK_UN)
        except Exception:
            pass


def _load_done_keys(path: Path) -> set[tuple]:
    if not path.exists():
        return set()
    with open(path) as f:
        rows = list(csv.DictReader(f))
    return {(float(r["lr"]), float(r["wd"]), int(r["seed"])) for r in rows}


# ---------------------------------------------------------------------------
# Worker (called in subprocess for parallel mode)
# ---------------------------------------------------------------------------

def _worker(args: tuple) -> dict:
    lr, wd, seed, csv_path_str = args
    result = run_one(lr=lr, wd=wd, seed=seed)
    _append_row(Path(csv_path_str), result)
    label = (f"lr={lr:.2e} wd={wd:.4f} seed={seed} → "
             f"phase={result['observed_phase']} ordering={result['ordering']} "
             f"tau_gen={result['tau_gen']} tau_F={result['tau_F']} "
             f"delta={result['delta']} t={result['runtime_sec']:.0f}s")
    print(label, flush=True)
    return result


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Stage 6: 2D (lr × wd) sweep")
    parser.add_argument("--outdir", default="runs/stage6_2d",
                        help="Output directory")
    parser.add_argument("--seeds", nargs="+", type=int, default=DEFAULT_SEEDS)
    parser.add_argument("--lr-idx", nargs="+", type=int, default=None,
                        help="Subset of lr indices to run (default: all)")
    parser.add_argument("--wd-idx", nargs="+", type=int, default=None,
                        help="Subset of wd indices to run (default: all)")
    parser.add_argument("--parallel", type=int, default=1,
                        help="Number of parallel workers (default: 1). "
                             "Recommended: 8-10 on A800.")
    parser.add_argument("--max-steps", type=int, default=BASE_STEPS)
    args = parser.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    csv_path = outdir / "results_stage6_2d.csv"

    lr_vals = [LR_VALUES[i] for i in args.lr_idx] if args.lr_idx else LR_VALUES
    wd_vals = [WD_VALUES[i] for i in args.wd_idx] if args.wd_idx else WD_VALUES

    done_keys = _load_done_keys(csv_path)
    total = len(lr_vals) * len(wd_vals) * len(args.seeds)
    todo = [
        (lr, wd, seed, str(csv_path))
        for lr in lr_vals
        for wd in wd_vals
        for seed in args.seeds
        if (lr, wd, seed) not in done_keys
    ]

    print(f"\n{'─'*60}")
    print(f"  Stage 6 — 2D (lr × wd) sweep, add mod {PRIME}")
    print(f"  Grid: {len(lr_vals)} lr × {len(wd_vals)} wd × {len(args.seeds)} seeds")
    print(f"  Progress: {total - len(todo)}/{total} done, {len(todo)} remaining")
    print(f"  Workers: {args.parallel}")
    print(f"  Output:  {csv_path}")
    print(f"{'─'*60}\n")

    if not todo:
        print("All cells complete.")
        _print_summary(csv_path)
        return

    if args.parallel <= 1:
        # Sequential
        for i, task in enumerate(todo, 1):
            lr, wd, seed, _ = task
            print(f"[{i}/{len(todo)}] lr={lr:.2e} wd={wd:.4f} seed={seed} ...",
                  end=" ", flush=True)
            _worker(task)
    else:
        # Parallel
        with ProcessPoolExecutor(max_workers=args.parallel) as exe:
            futures = {exe.submit(_worker, t): t for t in todo}
            done = 0
            for fut in as_completed(futures):
                done += 1
                try:
                    fut.result()
                except Exception as e:
                    t = futures[fut]
                    print(f"  ERROR lr={t[0]:.2e} wd={t[1]:.4f} seed={t[2]}: {e}")
                if done % 10 == 0 or done == len(todo):
                    print(f"  Progress: {done}/{len(todo)} completed", flush=True)

    _print_summary(csv_path)


def _print_summary(csv_path: Path) -> None:
    if not csv_path.exists():
        return
    with open(csv_path) as f:
        rows = list(csv.DictReader(f))

    grok = [r for r in rows if r.get("observed_phase") == "Grokking"]
    if not grok:
        from collections import Counter
        phases = Counter(r.get("observed_phase", "?") for r in rows)
        print(f"\n=== Stage 6 Summary: {len(rows)} runs, no Grokking ===")
        print(dict(phases))
        return

    n_gf = sum(1 for r in grok if r.get("ordering") == "G<F")
    n_fg = sum(1 for r in grok if r.get("ordering") == "F<G")
    n_co = sum(1 for r in grok if r.get("ordering") == "coincident")
    deltas = [float(r["delta"]) for r in grok
              if r.get("delta") not in (None, "", "None")]
    med = int(np.median(deltas)) if deltas else None

    print(f"\n{'='*60}")
    print(f"  Stage 6 Summary — add mod {PRIME}, 2D grid")
    print(f"  Total runs: {len(rows)}  Grokking: {len(grok)}")
    print(f"  G<F: {n_gf}  F<G: {n_fg}  coincident: {n_co}")
    if grok:
        print(f"  G<F rate: {n_gf}/{len(grok)} = {100*n_gf/len(grok):.1f}%")
    if med is not None:
        print(f"  Median Δτ: {med:,} steps")
    print(f"{'='*60}\n")
    print(f"Results saved to: {csv_path}")


if __name__ == "__main__":
    main()
