#!/usr/bin/env python3
"""
Stage 5B: (a+b) mod 97 — Different Prime
=========================================
Replicates Stage-2 wd sweep with prime p=97 (instead of 53) to test
whether the G<F ordering generalises across moduli.

Grid:
  operation = "add", prime = 97
  lr = 1.6e-3 (fixed)
  wd = 10 log-spaced values in [1.2, 3.5]  (same as Stage 2)
  seeds = [42, 7, 2025]
  max_steps = 80,000  (p=97 groks slower; generous budget)

Usage:
  python stage5_p97_sweep.py
  python stage5_p97_sweep.py --wd-idx 0 1 2 --seeds 42
"""

from __future__ import annotations

import argparse
import csv
import fcntl
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
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
    compute_fourier_alignment,
    compute_fourier_null_p95,
    compute_ordering,
    estimate_changepoint,
    find_tau_sustained,
)

# ---------------------------------------------------------------------------
# Grid
# ---------------------------------------------------------------------------

OPERATION = "add"
PRIME = 97
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
    max_steps: int = 80_000,
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
# Helpers
# ---------------------------------------------------------------------------

FIELDNAMES = [
    "operation", "prime", "lr", "wd", "seed",
    "observed_phase", "tau_gen", "tau_F", "delta", "ordering",
    "max_train_acc", "max_test_acc", "runtime_sec",
]

def _append_row(path: Path, row: dict) -> None:
    """Append one row to CSV with file lock (parallel-safe)."""
    write_header = not path.exists() or path.stat().st_size == 0
    with open(path, "a", newline="") as f:
        try:
            fcntl.flock(f, fcntl.LOCK_EX)
        except Exception:
            pass
        w = csv.DictWriter(f, fieldnames=FIELDNAMES)
        if write_header:
            w.writeheader()
        w.writerow({k: row.get(k) for k in FIELDNAMES})
        try:
            fcntl.flock(f, fcntl.LOCK_UN)
        except Exception:
            pass

def _write_csv(rows, path):
    if not rows:
        return
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDNAMES)
        w.writeheader(); w.writerows({k: r.get(k) for k in FIELDNAMES} for r in rows)
    print(f"  [csv] {path}")

def _load_csv(path):
    if not Path(path).exists():
        return []
    with open(path) as f:
        return list(csv.DictReader(f))

def _load_done_keys(path: Path) -> set[tuple]:
    rows = _load_csv(path)
    return {(float(r["wd"]), int(r["seed"])) for r in rows}

def _worker(args: tuple) -> dict:
    wd, seed, max_steps, csv_path_str = args
    result = run_one(lr=FIXED_LR, wd=wd, seed=seed, prime=PRIME, max_steps=max_steps)
    _append_row(Path(csv_path_str), result)
    print(f"wd={wd:.4f} seed={seed} → phase={result['observed_phase']} "
          f"ordering={result['ordering']} tau_gen={result['tau_gen']} "
          f"tau_F={result['tau_F']} delta={result['delta']} "
          f"t={result['runtime_sec']:.0f}s", flush=True)
    return result

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
        deltas = [float(r["delta"]) for r in cell if r.get("delta") not in (None,"","None")]
        med_delta.append(float(np.median(deltas)) if deltas else float("nan"))

    fig, axes = plt.subplots(3, 1, figsize=(7, 8), sharex=True)
    fig.suptitle(f"Stage 5B: G<F Ordering — add mod {PRIME}\nlr={FIXED_LR:.1e}", fontsize=11)
    axes[0].plot(wd_vals, p_gf, "o-", color="steelblue"); axes[0].set_ylabel("P(G<F)"); axes[0].set_ylim(-0.05, 1.1)
    axes[1].bar(range(len(wd_vals)), [d/1000 if not np.isnan(d) else 0 for d in med_delta], color="steelblue", alpha=0.7)
    axes[1].set_xticks(range(len(wd_vals))); axes[1].set_xticklabels([f"{w:.2f}" for w in wd_vals], rotation=45, fontsize=8)
    axes[1].set_ylabel("Median Δτ (k steps)"); axes[1].axhline(0, color="k", lw=0.8)
    axes[2].plot(wd_vals, p_fonly, "s--", color="darkorange"); axes[2].set_ylabel("P(F_only)"); axes[2].set_ylim(-0.05, 1.1)
    axes[2].set_xlabel("Weight decay")
    plt.tight_layout()
    out = outdir / "p97_curves.png"
    plt.savefig(out, dpi=150); plt.close()
    print(f"  [plot] {out}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--outdir", default="runs/stage5_p97")
    parser.add_argument("--seeds", nargs="+", type=int, default=DEFAULT_SEEDS)
    parser.add_argument("--wd-idx", nargs="+", type=int, default=None)
    parser.add_argument("--max-steps", type=int, default=80_000)
    parser.add_argument("--parallel", type=int, default=1,
                        help="Parallel workers (default: 1, A800 recommend 4)")
    args = parser.parse_args()

    outdir = Path(args.outdir); outdir.mkdir(parents=True, exist_ok=True)
    csv_path = outdir / "results_stage5_p97.csv"

    wd_vals = [WD_VALUES[i] for i in args.wd_idx] if args.wd_idx else WD_VALUES
    done_keys = _load_done_keys(csv_path)

    todo = [
        (wd, seed, args.max_steps, str(csv_path))
        for wd in wd_vals
        for seed in args.seeds
        if (wd, seed) not in done_keys
    ]
    total = len(wd_vals) * len(args.seeds)

    print(f"\n{'─'*55}")
    print(f"  Stage 5B — add mod {PRIME}, lr={FIXED_LR:.1e}")
    print(f"  Progress: {total - len(todo)}/{total} done, {len(todo)} remaining")
    print(f"  Workers: {args.parallel}")
    print(f"  Output:  {csv_path}")
    print(f"{'─'*55}\n")

    if not todo:
        print("All cells complete.")
    elif args.parallel <= 1:
        for i, (wd, seed, max_steps, csv_str) in enumerate(todo, 1):
            print(f"[{i}/{len(todo)}] wd={wd:.4f} seed={seed} ...", end=" ", flush=True)
            _worker((wd, seed, max_steps, csv_str))
    else:
        with ProcessPoolExecutor(max_workers=args.parallel) as exe:
            futures = {exe.submit(_worker, t): t for t in todo}
            done = 0
            for fut in as_completed(futures):
                done += 1
                try:
                    fut.result()
                except Exception as e:
                    t = futures[fut]
                    print(f"  ERROR wd={t[0]:.4f} seed={t[1]}: {e}")
                if done % 5 == 0 or done == len(todo):
                    print(f"  Progress: {done}/{len(todo)} completed", flush=True)

    rows = _load_csv(csv_path)
    grok = [r for r in rows if r.get("observed_phase") == "Grokking"]
    if grok:
        n_gf = sum(1 for r in grok if r["ordering"] == "G<F")
        deltas = [float(r["delta"]) for r in grok if r.get("delta") not in (None,"","None")]
        print(f"\n=== Stage 5B Summary: add mod {PRIME} ===")
        print(f"  Grokking: {len(grok)}/{len(rows)}")
        print(f"  G<F: {n_gf}/{len(grok)} ({100*n_gf/len(grok):.1f}%)")
        print(f"  Median Δτ: {int(np.median(deltas)) if deltas else 'N/A'} steps")
    else:
        from collections import Counter
        print(f"\n=== Stage 5B: No Grokking in {len(rows)} runs ===")
        print(dict(Counter(r.get("observed_phase","?") for r in rows)))

    make_curves(rows, WD_VALUES, outdir)
    print(f"\nDone. Results: {csv_path}")

if __name__ == "__main__":
    main()
