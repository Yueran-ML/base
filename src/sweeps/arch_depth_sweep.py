#!/usr/bin/env python3
"""
Architecture-Depth Generality Check
====================================
Minimal extrapolation experiment: confirm that G<F (tau_gen < tau_F)
is not specific to the canonical 2-layer transformer used throughout
Stages 2-6 of the paper.

We re-train the canonical operating point (lr=1.6e-3, wd=2.5) under
two alternative depths --- n_layers in {1, 3} --- with the same
3 seeds. Everything else is held at the canonical setting:

    p = 53, operation = add, train_fraction = 0.3,
    d_model = 256, n_heads = 4, d_ff = 1024,
    embed_lr = 1e-3, embed_wd = 0,
    max_steps = 50,000, log_interval = 500,
    null_interval = 5,000, n_null_perms = 100.

Goal: produce a small per-cell results CSV that can be folded into a
post-hoc "architecture generality" subsection of the paper. The
prediction under our G<F finding is:

    Across n_layers in {1, 2, 3} at the same (lr, wd, seeds), G<F
    holds (tau_gen < tau_F) whenever Grokking occurs.

Anti-prediction: G<F is specific to n_layers=2 and reverses or
disappears at n_layers in {1, 3}.

Total cost: 2 archs x 3 seeds = 6 runs. At 20-45 min/run on an
RTX 4080 at 50k steps, expect ~3 GPU-hours.

Usage:
    python src/sweeps/arch_depth_sweep.py
    python src/sweeps/arch_depth_sweep.py --outdir results/arch_depth
    python src/sweeps/arch_depth_sweep.py --n-layers 1 --seeds 42 7 2025
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
    compute_fourier_alignment,
    compute_fourier_null_p95,
    compute_ordering,
    estimate_changepoint,
    find_tau_sustained,
)

# ---------------------------------------------------------------------------
# Grid definition (canonical operating point of Stage 2 / Stage 3)
# ---------------------------------------------------------------------------
FIXED_LR: float = 1.6e-3
FIXED_WD: float = 2.5
DEFAULT_N_LAYERS: list[int] = [1, 3]   # canonical 2 already in Stage 2/3
DEFAULT_SEEDS: list[int] = [42, 7, 2025]


def run_one(
    n_layers: int,
    seed: int,
    lr: float = FIXED_LR,
    wd: float = FIXED_WD,
    prime: int = 53,
    train_fraction: float = 0.3,
    max_steps: int = 50_000,
    log_interval: int = 500,
    null_interval: int = 5000,
    n_null_perms: int = 100,
    embed_lr: float = 1e-3,
    d_model: int = 256,
    n_heads: int = 4,
    d_ff: int = 1024,
) -> dict:
    """Train one (n_layers, seed) cell at canonical (lr, wd)."""
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

    observed_phase = classify_phase(steps_log, train_acc_log, test_acc_log)
    tau_gen = find_tau_sustained(steps_log, test_acc_log, threshold=0.9, n_sustained=3)
    tau_F = estimate_changepoint(steps_log, fourier_corr_log)

    delta = (tau_F - tau_gen) if (tau_gen is not None and tau_F is not None) else None
    ordering = compute_ordering(tau_gen, tau_F)

    return {
        "n_layers": n_layers,
        "lr": lr,
        "wd": wd,
        "seed": seed,
        "observed_phase": observed_phase,
        "tau_gen": tau_gen,
        "tau_F": tau_F,
        "delta": delta,
        "ordering": ordering,
        "max_train_acc": max(train_acc_log),
        "max_test_acc": max(test_acc_log),
        "runtime_sec": time.time() - t0,
    }


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--outdir", default="results/arch_depth")
    p.add_argument("--max-steps", type=int, default=50_000)
    p.add_argument("--n-layers", type=int, nargs="+", default=DEFAULT_N_LAYERS)
    p.add_argument("--seeds", type=int, nargs="+", default=DEFAULT_SEEDS)
    p.add_argument("--lr", type=float, default=FIXED_LR)
    p.add_argument("--wd", type=float, default=FIXED_WD)
    return p.parse_args()


def main():
    args = parse_args()
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    n_layers_list = args.n_layers
    seeds = args.seeds
    total_runs = len(n_layers_list) * len(seeds)

    print(
        f"Architecture-Depth Sweep: lr={args.lr} wd={args.wd} | "
        f"{len(n_layers_list)} archs x {len(seeds)} seeds = {total_runs} runs"
    )
    print(f"  n_layers: {n_layers_list}")
    print(f"  seeds:    {seeds}")
    print(f"  max_steps: {args.max_steps:,} | log_interval: 500")
    print(f"  output:   {outdir}\n")

    csv_path = outdir / "results.csv"
    fieldnames = [
        "n_layers", "lr", "wd", "seed", "observed_phase",
        "tau_gen", "tau_F", "delta", "ordering",
        "max_train_acc", "max_test_acc", "runtime_sec",
    ]

    # Resume support
    done: set[tuple] = set()
    if csv_path.exists():
        with open(csv_path, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                done.add((int(row["n_layers"]), int(row["seed"])))
        print(f"  Resuming: {len(done)} runs already done, "
              f"{total_runs - len(done)} remaining\n")

    csv_file = open(csv_path, "a", newline="", encoding="utf-8")
    writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
    if not done:
        writer.writeheader()

    run_idx = 0
    for n_layers in n_layers_list:
        for seed in seeds:
            run_idx += 1
            if (n_layers, seed) in done:
                print(f"  [{run_idx}/{total_runs}] n_layers={n_layers} seed={seed} -- SKIPPED")
                continue

            print(
                f"[{run_idx}/{total_runs}] n_layers={n_layers} seed={seed} "
                f"lr={args.lr:.1e} wd={args.wd:.3g} ...",
                end="", flush=True,
            )
            result = run_one(
                n_layers=n_layers, seed=seed,
                lr=args.lr, wd=args.wd, max_steps=args.max_steps,
            )

            tau_gen_str = f"{result['tau_gen']:.0f}" if result["tau_gen"] is not None else ""
            tau_F_str   = f"{result['tau_F']:.0f}"   if result["tau_F"]   is not None else ""
            delta_str   = f"{result['delta']:.0f}"   if result["delta"]   is not None else ""
            print(
                f"  phase={result['observed_phase']} | "
                f"tau_gen={tau_gen_str or '-'} | tau_F={tau_F_str or '-'} | "
                f"D={delta_str or '-'} | ordering={result['ordering']} | "
                f"{result['runtime_sec']:.0f}s"
            )

            row = {
                "n_layers": n_layers,
                "lr": args.lr,
                "wd": args.wd,
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

    csv_file.close()
    print(f"\nAll runs complete. CSV: {csv_path}")
    summarize(csv_path)


def summarize(csv_path: Path) -> None:
    """Print a short, paper-ready summary table grouped by n_layers."""
    rows = []
    with open(csv_path, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            rows.append(r)

    # Pull canonical 2-layer rows from Stage 2 (lr=1.6e-3, wd=2.5)
    # and Stage 3 (lr=1.6e-3, wd=2.5) if available, so we can include
    # the n_layers=2 reference row in the paper-ready summary.
    canonical_rows = []
    for stage_csv in [Path("results/stage2_wd/results.csv"),
                      Path("results/stage3_lr/results.csv")]:
        if not stage_csv.exists():
            continue
        with open(stage_csv, newline="", encoding="utf-8") as f:
            for r in csv.DictReader(f):
                lr = float(r.get("lr", FIXED_LR))
                wd = float(r.get("wd", FIXED_WD))
                if abs(lr - FIXED_LR) < 1e-9 and abs(wd - FIXED_WD) < 1e-9:
                    r["n_layers"] = 2
                    canonical_rows.append(r)

    by_n: dict[int, list[dict]] = {}
    for r in rows:
        by_n.setdefault(int(r["n_layers"]), []).append(r)
    if canonical_rows:
        by_n.setdefault(2, []).extend(canonical_rows)

    print()
    print("=" * 60)
    print("Architecture-Depth Generality Summary")
    print("=" * 60)
    print(f"{'n_layers':>9} | {'n_runs':>6} | {'n_grok':>6} | "
          f"{'G<F':>6} | {'med D (k)':>10} | {'med tau_gen':>11}")
    print("-" * 60)
    for n in sorted(by_n.keys()):
        cell = by_n[n]
        n_grok = sum(1 for r in cell
                     if r.get("observed_phase") == "Grokking")
        n_gf = sum(1 for r in cell if r.get("ordering") == "G<F")
        deltas = [float(r["delta"]) for r in cell
                  if r.get("delta") not in ("", None, "None")]
        taugens = [float(r["tau_gen"]) for r in cell
                   if r.get("tau_gen") not in ("", None, "None")]
        med_d = np.median(deltas) / 1000 if deltas else float("nan")
        med_tg = np.median(taugens) if taugens else float("nan")
        print(f"{n:>9} | {len(cell):>6} | {n_grok:>6} | "
              f"{n_gf:>6} | {med_d:>10.2f} | {med_tg:>11.0f}")
    print()
    print("Prediction: G<F holds across n_layers in {1, 2, 3}.")


if __name__ == "__main__":
    main()
