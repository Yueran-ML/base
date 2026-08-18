#!/usr/bin/env python3
"""
Slow-grokking targeted sweep
============================

Fixed wd=2.5, fixed seeds={42,7,2025}, lr swept in the slow-grokking
margin lr in [5.0, 8.0]e-4 to test the strong form of the
speed-dependent ordering hypothesis (Section 7 of the paper).

Strong form prediction: as tau_gen pushes into the slow-grokking
margin (>=35,000 steps at our base wd setting), the F<G ordering
(tau_F < tau_gen) should appear systematically rather than
anecdotally.

Grid:
  lr  in {5.0, 5.5, 6.0, 6.5, 6.8, 7.2, 7.6, 8.0} x 10^-4   (8 values)
  wd  = 2.5                                                  (fixed)
  seeds = {42, 7, 2025}                                      (3 seeds)
  -> 24 cells total

Training:
  max_steps = 100,000 (auto-extend to 120,000 if train_acc>0.9 but
  tau_gen not yet detected)

The detector pipeline matches step2_circuit_sweep.py exactly so the
results can be merged into the existing C<G<F analysis. We track
tau_F (embedding Fourier), tau_gen, AND tau_circuit (logit Fourier)
for every cell.

Output:
  results/slow_grokking/results.csv

Usage:
  .venv/Scripts/python.exe src/sweeps/slow_grokking_sweep.py
  .venv/Scripts/python.exe src/sweeps/slow_grokking_sweep.py --lr-idx 0 1 2 --seeds 42
  .venv/Scripts/python.exe src/sweeps/slow_grokking_sweep.py --smoke
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


PRIME = 53
OPERATION = "add"
FIXED_WD = 2.5
LR_VALUES = [
    5.0e-4, 5.5e-4, 6.0e-4, 6.5e-4,
    6.8e-4, 7.2e-4, 7.6e-4, 8.0e-4,
]
DEFAULT_SEEDS = [42, 7, 2025]
EMBED_LR = 1e-3

BASE_STEPS = 100_000
EXTEND_STEPS = 120_000
LOG_INTERVAL = 500
LOGIT_INTERVAL = 500
NULL_INTERVAL = 5_000


def _get_all_logits(model: GrokkingTransformer, prime: int,
                    device: torch.device) -> np.ndarray:
    a_vals = torch.arange(prime)
    b_vals = torch.arange(prime)
    aa, bb = torch.meshgrid(a_vals, b_vals, indexing="ij")
    all_pairs = torch.stack([aa.flatten(), bb.flatten()], dim=1).to(device)
    with torch.no_grad():
        logits = model(all_pairs).cpu().numpy()
    return logits


def run_one(
    lr: float, wd: float, seed: int,
    prime: int = PRIME,
    train_fraction: float = 0.3,
    max_steps: int = BASE_STEPS,
    log_interval: int = LOG_INTERVAL,
    logit_interval: int = LOGIT_INTERVAL,
    null_interval: int = NULL_INTERVAL,
    n_null_perms: int = 100,
    embed_lr: float = EMBED_LR,
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
    optimizer = build_optimizer(model, decoder_lr=lr, embed_lr=embed_lr,
                                decoder_weight_decay=wd)
    scheduler = build_scheduler(optimizer, warmup_steps=10,
                                lr_schedule="constant", lr_min_ratio=0.05,
                                max_steps=max_steps)

    steps_log = []
    train_acc_log = []
    test_acc_log = []
    f_emb_corr_log = []

    logit_steps_log = []
    f_logit_corr_log = []

    null_rng_emb = np.random.default_rng(seed ^ 0xDEAD)
    null_rng_logit = np.random.default_rng(seed ^ 0xBEEF)
    cur_null_emb = 0.0
    cur_null_logit = 0.0

    step = 0
    extended = False
    while step < max_steps:
        for xb, yb in loader:
            model.train()
            logits_train = model(xb.to(device))
            loss = criterion(logits_train, yb.to(device))
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

            if log_now:
                model.eval()
                with torch.no_grad():
                    tr_acc = (model(train_x.to(device)).argmax(1).cpu()
                              == train_y).float().mean().item()
                    te_acc = (model(test_x.to(device)).argmax(1).cpu()
                              == test_y).float().mean().item()
                emb = get_token_embeddings(model, prime)
                if null_now:
                    cur_null_emb = compute_fourier_null_p95(
                        emb, prime, n_perms=n_null_perms, rng=null_rng_emb,
                    )
                f_emb_raw, _ = compute_fourier_alignment(emb, prime)
                f_emb_corr = max(0.0, f_emb_raw - cur_null_emb)

                steps_log.append(step)
                train_acc_log.append(tr_acc)
                test_acc_log.append(te_acc)
                f_emb_corr_log.append(f_emb_corr)

            if logit_now:
                logits_all = _get_all_logits(model, prime, device)
                if null_now:
                    cur_null_logit = compute_fourier_logit_null_p95(
                        logits_all, prime, n_perms=n_null_perms,
                        rng=null_rng_logit,
                    )
                f_logit_raw, _ = compute_fourier_logit_alignment(
                    logits_all, prime)
                f_logit_corr = max(0.0, f_logit_raw - cur_null_logit)
                logit_steps_log.append(step)
                f_logit_corr_log.append(f_logit_corr)

            if step >= max_steps:
                break

        if step >= max_steps and not extended:
            tg_check = find_tau_sustained(steps_log, test_acc_log,
                                          threshold=0.9, n_sustained=3)
            if (train_acc_log and train_acc_log[-1] > 0.9
                    and tg_check is None and max_steps < EXTEND_STEPS):
                print(f"    [extend] train_acc={train_acc_log[-1]:.3f}, "
                      f"extending to {EXTEND_STEPS}")
                max_steps = EXTEND_STEPS
                extended = True
            else:
                break
        elif step >= max_steps:
            break

    observed_phase = classify_phase(steps_log, train_acc_log, test_acc_log)
    tau_gen = find_tau_sustained(steps_log, test_acc_log, threshold=0.9,
                                 n_sustained=3)
    tau_F = estimate_changepoint(steps_log, f_emb_corr_log)
    tau_circuit = estimate_changepoint(logit_steps_log, f_logit_corr_log)

    delta_gf = (tau_F - tau_gen
                if (tau_gen is not None and tau_F is not None) else None)
    ordering_gf = compute_ordering(tau_gen, tau_F)

    # 3-way ordering string
    if (tau_circuit is not None and tau_gen is not None
            and tau_F is not None):
        events = sorted([("C", tau_circuit), ("G", tau_gen), ("F", tau_F)],
                        key=lambda e: e[1])
        ordering_3stage = "<".join(e[0] for e in events)
    else:
        ordering_3stage = "incomplete"

    return {
        "operation": OPERATION, "prime": prime,
        "lr": lr, "wd": wd, "seed": seed,
        "observed_phase": observed_phase,
        "tau_gen": tau_gen, "tau_F": tau_F,
        "tau_circuit": tau_circuit,
        "delta_gf": delta_gf,
        "ordering_gf": ordering_gf,
        "ordering_3stage": ordering_3stage,
        "max_train_acc": max(train_acc_log) if train_acc_log else 0.0,
        "max_test_acc": max(test_acc_log) if test_acc_log else 0.0,
        "max_steps_used": max_steps,
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
    parser.add_argument("--outdir", default="results/slow_grokking")
    parser.add_argument("--seeds", nargs="+", type=int, default=DEFAULT_SEEDS)
    parser.add_argument("--lr-idx", nargs="+", type=int, default=None,
                        help="Indices into LR_VALUES; default = all 8.")
    parser.add_argument("--max-steps", type=int, default=BASE_STEPS)
    parser.add_argument("--smoke", action="store_true",
                        help="Run only lr_idx 0 with 1 seed at 5k steps.")
    args = parser.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    csv_path = outdir / "results.csv"

    lr_vals = ([LR_VALUES[i] for i in args.lr_idx]
               if args.lr_idx else LR_VALUES)
    seeds = list(args.seeds)
    max_steps = args.max_steps
    if args.smoke:
        lr_vals = lr_vals[:1]
        seeds = seeds[:1]
        max_steps = 5_000

    existing = _load_csv(csv_path)
    done = {_key(r) for r in existing}
    rows = list(existing)

    n_total = len(lr_vals) * len(seeds)
    print(f"Total cells: {n_total}; "
          f"resuming from {sum(1 for k in done if k[1] == round(FIXED_WD, 6))} done.")

    for lr in lr_vals:
        for seed in seeds:
            k = (round(lr, 6), round(FIXED_WD, 6), seed)
            if k in done:
                print(f"  [skip] lr={lr:.2e}, seed={seed} (already done)")
                continue
            print(f"[run] lr={lr:.2e} wd={FIXED_WD} seed={seed} "
                  f"max_steps={max_steps} ...", flush=True)
            res = run_one(lr, FIXED_WD, seed, max_steps=max_steps)
            print(f"      tau_gen={res['tau_gen']} "
                  f"tau_F={res['tau_F']} "
                  f"tau_circ={res['tau_circuit']} "
                  f"delta_gf={res['delta_gf']} "
                  f"order={res['ordering_3stage']} "
                  f"phase={res['observed_phase']} "
                  f"runtime={res['runtime_sec']:.0f}s",
                  flush=True)
            rows.append(res)
            _write_csv(rows, csv_path)

    # Summary
    if not rows:
        print("No rows written.")
        return

    grok = [r for r in rows if r.get("observed_phase") == "Grokking"]
    fg = [r for r in grok
          if r.get("ordering_gf") == "F<G"]
    gf = [r for r in grok
          if r.get("ordering_gf") == "G<F"]
    coinc = [r for r in grok
             if r.get("ordering_gf") == "coincident"]

    print(f"\nSummary:  {len(rows)} total, {len(grok)} Grokking.")
    print(f"  G<F: {len(gf)}/{len(grok)}    "
          f"F<G: {len(fg)}/{len(grok)}    "
          f"coincident: {len(coinc)}/{len(grok)}")
    if grok:
        tau_gens = [float(r["tau_gen"]) for r in grok if r["tau_gen"]]
        if tau_gens:
            print(f"  tau_gen range: {int(min(tau_gens))} -- "
                  f"{int(max(tau_gens))} steps")
            print(f"  median tau_gen: {int(np.median(tau_gens))} steps")
    print(f"CSV -> {csv_path}")


if __name__ == "__main__":
    main()
