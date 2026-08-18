#!/usr/bin/env python3
"""
Depth Scan: how the three-event ordering varies with transformer depth
======================================================================
The paper's headline result (G<F: embedding-Fourier consolidation lags
generalization) is established at n_layers=2. A 3-seed pilot at the
canonical operating point found the ordering reversed at n_layers in
{1, 3} (results/arch_depth/results.csv: 1-layer 3/3 F<G with delta
-7,500 to -10,000; 3-layer 2/3 F<G). Those magnitudes are comparable to
or larger than the +6,000 median lag at 2 layers, so the reversal is
not a resolution artefact and the single operating point is too thin a
basis to characterise it.

This script runs the systematic version: a depth x cell grid so the
transition can be described rather than merely flagged.

    n_layers    in {1, 2, 3, 4}
    wd          in {1.71, 2.5, 3.11}    (three Grokking cells, lr fixed)
    seeds       in {42, 7, 2025}
    => 36 runs

n_layers=2 is included deliberately rather than reused from Stage 2/3:
the paper documents ~9pp cross-hardware variance in ordering labels, so
the depth comparison must be run on one machine to be controlled.

All three events are recorded per run (tau_circuit, tau_gen, tau_F) so
the scan reports both the G<F ordering and the full three-way label,
using the same detectors and null correction as the main experiments.

Cost: 36 runs. On an RTX 4080 at 50k steps a 2-layer run takes 20-45
min; 3- and 4-layer runs are proportionally slower. Budget roughly
15-30 GPU-hours for the full grid. The script checkpoints after every
run and resumes on (n_layers, wd, seed), so it can be stopped and
restarted freely.

Usage:
    python src/sweeps/depth_scan_sweep.py
    python src/sweeps/depth_scan_sweep.py --n-layers 1 2 --seeds 42
    python src/sweeps/depth_scan_sweep.py --outdir results/depth_scan
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
from grokking_baseline import (  # noqa: E402
    GrokkingTransformer,
    build_optimizer,
    build_scheduler,
    get_token_embeddings,
    make_dataset,
    set_seed,
    split_dataset,
)
from grok_metrics import (  # noqa: E402
    classify_phase,
    compute_fourier_alignment,
    compute_fourier_logit_alignment,
    compute_fourier_logit_null_p95,
    compute_fourier_null_p95,
    compute_ordering,
    estimate_changepoint,
    find_tau_sustained,
)

PRIME: int = 53
OPERATION: str = "add"
FIXED_LR: float = 1.6e-3

DEFAULT_N_LAYERS: list[int] = [1, 2, 3, 4]
DEFAULT_WDS: list[float] = [1.71, 2.5, 3.11]
DEFAULT_SEEDS: list[int] = [42, 7, 2025]

BASE_STEPS: int = 50_000
EXTEND_STEPS: int = 80_000
LOG_INTERVAL: int = 500
LOGIT_INTERVAL: int = 500
NULL_INTERVAL: int = 5_000


def _get_all_logits(model, prime: int, device) -> np.ndarray:
    """Forward pass on all (a,b) pairs. Returns (prime*prime, prime)."""
    aa, bb = torch.meshgrid(torch.arange(prime), torch.arange(prime),
                            indexing="ij")
    all_pairs = torch.stack([aa.flatten(), bb.flatten()], dim=1).to(device)
    with torch.no_grad():
        return model(all_pairs).cpu().numpy()


def run_one(
    n_layers: int,
    wd: float,
    seed: int,
    lr: float = FIXED_LR,
    prime: int = PRIME,
    train_fraction: float = 0.3,
    max_steps: int = BASE_STEPS,
    log_interval: int = LOG_INTERVAL,
    logit_interval: int = LOGIT_INTERVAL,
    null_interval: int = NULL_INTERVAL,
    n_null_perms: int = 100,
    embed_lr: float = 1e-3,
    d_model: int = 256,
    n_heads: int = 4,
    d_ff: int = 1024,
) -> dict:
    """Train one (n_layers, wd, seed) cell and extract all three events."""
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

    steps_log: list[int] = []
    train_acc_log: list[float] = []
    test_acc_log: list[float] = []
    fourier_emb_log: list[float] = []
    logit_steps_log: list[int] = []
    fourier_logit_log: list[float] = []

    null_rng_emb = np.random.default_rng(seed ^ 0xDEAD)
    null_rng_logit = np.random.default_rng(seed ^ 0xBEEF)
    current_null95_emb = 0.0
    current_null95_logit = 0.0

    step = 0
    while step < max_steps:
        for xb, yb in loader:
            model.train()
            loss = criterion(model(xb.to(device)), yb.to(device))
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            step += 1

            log_now = (step % log_interval == 0 or step == 1
                       or step >= max_steps)
            logit_now = (step % logit_interval == 0 or step == 1
                         or step >= max_steps)
            null_now = (step % null_interval == 0 or step == 1)

            if log_now or logit_now:
                model.eval()
                with torch.no_grad():
                    tr_acc = (model(train_x.to(device)).argmax(1).cpu()
                              == train_y).float().mean().item()
                    te_acc = (model(test_x.to(device)).argmax(1).cpu()
                              == test_y).float().mean().item()

                emb = get_token_embeddings(model, prime)
                if null_now:
                    current_null95_emb = compute_fourier_null_p95(
                        emb, prime, n_perms=n_null_perms, rng=null_rng_emb)
                f_emb_raw, _ = compute_fourier_alignment(emb, prime)
                f_emb_corr = max(0.0, f_emb_raw - current_null95_emb)

                if log_now:
                    steps_log.append(step)
                    train_acc_log.append(tr_acc)
                    test_acc_log.append(te_acc)
                    fourier_emb_log.append(f_emb_corr)

                if logit_now:
                    all_logits = _get_all_logits(model, prime, device)
                    if null_now:
                        current_null95_logit = compute_fourier_logit_null_p95(
                            all_logits, prime, operation=OPERATION,
                            n_perms=n_null_perms, rng=null_rng_logit)
                    f_logit_raw, _ = compute_fourier_logit_alignment(
                        all_logits, prime, operation=OPERATION)
                    logit_steps_log.append(step)
                    fourier_logit_log.append(
                        max(0.0, f_logit_raw - current_null95_logit))

            if step >= max_steps:
                break

    # Auto-extend when training converged but tau_gen has not fired yet.
    max_train = max(train_acc_log) if train_acc_log else 0.0
    tau_gen = find_tau_sustained(steps_log, test_acc_log,
                                 threshold=0.9, n_sustained=3)
    if max_train > 0.9 and tau_gen is None and max_steps < EXTEND_STEPS:
        return run_one(
            n_layers=n_layers, wd=wd, seed=seed, lr=lr, prime=prime,
            train_fraction=train_fraction, max_steps=EXTEND_STEPS,
            log_interval=log_interval, logit_interval=logit_interval,
            null_interval=null_interval, n_null_perms=n_null_perms,
            embed_lr=embed_lr, d_model=d_model, n_heads=n_heads, d_ff=d_ff,
        )

    observed_phase = classify_phase(steps_log, train_acc_log, test_acc_log)
    tau_train = find_tau_sustained(steps_log, train_acc_log,
                                   threshold=0.9, n_sustained=3)
    tau_F = estimate_changepoint(steps_log, fourier_emb_log)
    tau_circuit = estimate_changepoint(logit_steps_log, fourier_logit_log)

    delta = (tau_F - tau_gen) if (tau_gen is not None
                                  and tau_F is not None) else None
    ordering = compute_ordering(tau_gen, tau_F)

    # Three-way label, using the same 500-step coincidence band as the paper.
    if None in (tau_circuit, tau_gen, tau_F):
        ordering3 = "incomplete"
    else:
        events = sorted([(tau_circuit, "C"), (tau_gen, "G"), (tau_F, "F")])
        ordering3 = "<".join(tag for _, tag in events)

    return {
        "n_layers": n_layers, "lr": lr, "wd": wd, "seed": seed,
        "observed_phase": observed_phase,
        "tau_train": tau_train, "tau_circuit": tau_circuit,
        "tau_gen": tau_gen, "tau_F": tau_F,
        "delta": delta, "ordering": ordering, "ordering3": ordering3,
        "max_train_acc": max(train_acc_log),
        "max_test_acc": max(test_acc_log),
        "max_steps_used": max_steps,
        "runtime_sec": time.time() - t0,
    }


FIELDNAMES = [
    "n_layers", "lr", "wd", "seed", "observed_phase",
    "tau_train", "tau_circuit", "tau_gen", "tau_F",
    "delta", "ordering", "ordering3",
    "max_train_acc", "max_test_acc", "max_steps_used", "runtime_sec",
]


def _fmt(v) -> str:
    return "" if v is None else (f"{v:.0f}" if isinstance(v, float) else str(v))


def summarize(csv_path: Path) -> None:
    with open(csv_path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        return

    by_n: dict[int, list[dict]] = {}
    for r in rows:
        by_n.setdefault(int(r["n_layers"]), []).append(r)

    print()
    print("=" * 74)
    print("Depth Scan Summary (lr fixed, wd in {1.71, 2.5, 3.11})")
    print("=" * 74)
    print(f"{'n_layers':>8} | {'runs':>4} | {'grok':>4} | {'G<F':>4} | "
          f"{'F<G':>4} | {'med delta':>10} | {'med tau_gen':>11}")
    print("-" * 74)
    for n in sorted(by_n):
        cell = by_n[n]
        grok = sum(1 for r in cell if r["observed_phase"] == "Grokking")
        gf = sum(1 for r in cell if r["ordering"] == "G<F")
        fg = sum(1 for r in cell if r["ordering"] == "F<G")
        ds = [float(r["delta"]) for r in cell
              if r["delta"] not in ("", None, "None")]
        tg = [float(r["tau_gen"]) for r in cell
              if r["tau_gen"] not in ("", None, "None")]
        med_d = f"{np.median(ds):,.0f}" if ds else "-"
        med_g = f"{np.median(tg):,.0f}" if tg else "-"
        print(f"{n:>8} | {len(cell):>4} | {grok:>4} | {gf:>4} | {fg:>4} | "
              f"{med_d:>10} | {med_g:>11}")

    print()
    print("Per-depth three-way ordering labels:")
    for n in sorted(by_n):
        counts: dict[str, int] = {}
        for r in by_n[n]:
            counts[r["ordering3"]] = counts.get(r["ordering3"], 0) + 1
        summary = ", ".join(f"{k}:{v}" for k, v in sorted(counts.items()))
        print(f"  n_layers={n}: {summary}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", default="results/depth_scan")
    ap.add_argument("--n-layers", type=int, nargs="+",
                    default=DEFAULT_N_LAYERS)
    ap.add_argument("--wds", type=float, nargs="+", default=DEFAULT_WDS)
    ap.add_argument("--seeds", type=int, nargs="+", default=DEFAULT_SEEDS)
    ap.add_argument("--lr", type=float, default=FIXED_LR)
    ap.add_argument("--max-steps", type=int, default=BASE_STEPS)
    args = ap.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    csv_path = outdir / "results.csv"

    grid = [(n, wd, s) for n in args.n_layers
            for wd in args.wds for s in args.seeds]
    total = len(grid)

    print(f"Depth Scan: lr={args.lr:.1e} | {len(args.n_layers)} depths x "
          f"{len(args.wds)} wd x {len(args.seeds)} seeds = {total} runs")
    print(f"  n_layers: {args.n_layers}")
    print(f"  wd:       {args.wds}")
    print(f"  seeds:    {args.seeds}")
    print(f"  output:   {csv_path}\n")

    done: set[tuple] = set()
    if csv_path.exists():
        with open(csv_path, newline="", encoding="utf-8") as f:
            for r in csv.DictReader(f):
                done.add((int(r["n_layers"]), round(float(r["wd"]), 4),
                          int(r["seed"])))
        print(f"  Resuming: {len(done)} done, {total - len(done)} remaining\n")

    with open(csv_path, "a", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=FIELDNAMES)
        if not done:
            writer.writeheader()

        for idx, (n_layers, wd, seed) in enumerate(grid, start=1):
            key = (n_layers, round(wd, 4), seed)
            if key in done:
                print(f"  [{idx}/{total}] L={n_layers} wd={wd} seed={seed} "
                      f"-- SKIPPED")
                continue

            print(f"[{idx}/{total}] L={n_layers} wd={wd} seed={seed} ...",
                  end="", flush=True)
            r = run_one(n_layers=n_layers, wd=wd, seed=seed, lr=args.lr,
                        max_steps=args.max_steps)
            print(f"  phase={r['observed_phase']} | "
                  f"C={_fmt(r['tau_circuit']) or '-'} "
                  f"G={_fmt(r['tau_gen']) or '-'} "
                  f"F={_fmt(r['tau_F']) or '-'} | "
                  f"D={_fmt(r['delta']) or '-'} | {r['ordering']} "
                  f"({r['ordering3']}) | {r['runtime_sec']:.0f}s")

            writer.writerow({
                k: (_fmt(r[k]) if k in ("tau_train", "tau_circuit",
                                        "tau_gen", "tau_F", "delta")
                    else r[k])
                for k in FIELDNAMES
            })
            csv_file.flush()

    print(f"\nAll runs complete. CSV: {csv_path}")
    summarize(csv_path)


if __name__ == "__main__":
    main()
