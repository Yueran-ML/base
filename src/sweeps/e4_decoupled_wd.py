#!/usr/bin/env python3
"""
E4 — Decoupled weight-decay ablation
=====================================
Disentangles the role of weight decay on embeddings vs. the decoder in the
G<F ordering finding.

Two experiments sharing the same training harness:

A) **2×2 embed_wd × decoder_wd ablation** (fixed throughout training):
     decoder_wd ∈ {0, wd*},   embed_wd ∈ {0, wd*},     wd* = 2.5
   → 4 cells × 3 seeds = 12 runs

   Logic:
     (0,0)         — no wd anywhere; expect no Fourier consolidation
     (0,wd*)       — embed_wd only; isolates direct pressure on embeddings
     (wd*,0)       — paper-canonical; decoder wd only (indirect → embeddings)
     (wd*,wd*)     — both

B) **Decoder-wd schedule shutoff** at step s*:
     decoder_wd = wd*  for step < s*, then  0
     s* ∈ {0.5·τ_gen_ref, τ_gen_ref, 2·τ_gen_ref}
     where τ_gen_ref ≈ 17,000 (median τ_gen at wd=1.7, lr=1.6e-3)
   → 3 shutoff steps × 3 seeds = 9 runs

Together: 21 runs × ~25 min on a 4080 = ~9 GPU-hours.

Outputs:
  results/e4/results.csv        — per-run τ_gen, τ_F, ordering, etc.
  results/e4/traj_*.npz         — dense trajectories (if --save-traj)

Usage:
  python src/sweeps/e4_decoupled_wd.py                    # both experiments
  python src/sweeps/e4_decoupled_wd.py --mode 2x2         # only ablation
  python src/sweeps/e4_decoupled_wd.py --mode schedule    # only shutoff
  python src/sweeps/e4_decoupled_wd.py --max-steps 2000   # smoketest
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
    set_decoder_weight_decay,
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
# Canonical reference point (centre of Stage 2 G<F plateau)
# ---------------------------------------------------------------------------
LR = 1.6e-3
WD_STAR = 2.5
EMBED_LR = 1e-3
PRIME = 53
D_MODEL = 256
N_HEADS = 4
N_LAYERS = 2
D_FF = 1024
TRAIN_FRACTION = 0.3
SEEDS = [42, 7, 2025]
LOG_INTERVAL = 500
NULL_INTERVAL = 5000
N_NULL_PERMS = 100
TAU_GEN_REF = 17_000  # median τ_gen at wd=1.71, lr=1.6e-3


def _run(
    decoder_wd: float,
    embed_wd: float,
    seed: int,
    shutoff_step: int | None,
    max_steps: int,
    save_traj_path: Path | None,
) -> dict:
    """Run one training; optionally turn decoder wd off at shutoff_step."""
    t0 = time.time()
    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    x, y = make_dataset(PRIME, "add", seed)
    train_x, train_y, test_x, test_y = split_dataset(x, y, TRAIN_FRACTION)

    model = GrokkingTransformer(
        prime=PRIME, d_model=D_MODEL, n_heads=N_HEADS,
        n_layers=N_LAYERS, d_ff=D_FF, dropout=0.0,
    ).to(device)

    loader = DataLoader(
        TensorDataset(train_x, train_y),
        batch_size=len(train_x), shuffle=True,
        generator=torch.Generator().manual_seed(seed),
    )
    criterion = nn.CrossEntropyLoss()
    optimizer = build_optimizer(
        model,
        decoder_lr=LR,
        embed_lr=EMBED_LR,
        decoder_weight_decay=decoder_wd,
        embed_weight_decay=embed_wd,
    )
    scheduler = build_scheduler(
        optimizer, warmup_steps=10, lr_schedule="constant",
        lr_min_ratio=0.05, max_steps=max_steps,
    )

    null_rng = np.random.default_rng(seed ^ 0xDEAD)
    current_null95 = 0.0

    steps_log:     list[int]   = []
    train_acc_log: list[float] = []
    test_acc_log:  list[float] = []
    f_raw_log:     list[float] = []
    f_null_log:    list[float] = []
    f_corr_log:    list[float] = []
    emb_snaps:     list[np.ndarray] = []
    shutoff_triggered = False

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

            if (shutoff_step is not None and not shutoff_triggered
                    and step >= shutoff_step):
                set_decoder_weight_decay(optimizer, 0.0)
                shutoff_triggered = True

            if step % LOG_INTERVAL == 0 or step == 1 or step >= max_steps:
                model.eval()
                with torch.no_grad():
                    tr_acc = (model(train_x.to(device)).argmax(1).cpu()
                              == train_y).float().mean().item()
                    te_acc = (model(test_x.to(device)).argmax(1).cpu()
                              == test_y).float().mean().item()

                emb = get_token_embeddings(model, PRIME)
                if step % NULL_INTERVAL == 0 or step == 1:
                    current_null95 = compute_fourier_null_p95(
                        emb, PRIME, n_perms=N_NULL_PERMS, rng=null_rng,
                    )
                f_raw, _ = compute_fourier_alignment(emb, PRIME)
                f_corr = max(0.0, f_raw - current_null95)

                steps_log.append(step)
                train_acc_log.append(tr_acc)
                test_acc_log.append(te_acc)
                f_raw_log.append(float(f_raw))
                f_null_log.append(float(current_null95))
                f_corr_log.append(float(f_corr))
                if save_traj_path is not None:
                    emb_snaps.append(emb.astype(np.float32))

            if step >= max_steps:
                break

    observed_phase = classify_phase(steps_log, train_acc_log, test_acc_log)
    tau_gen = find_tau_sustained(steps_log, test_acc_log,
                                 threshold=0.9, n_sustained=3)
    tau_F = estimate_changepoint(steps_log, f_corr_log)
    delta = (tau_F - tau_gen) if (tau_gen is not None and tau_F is not None) else None
    ordering = compute_ordering(tau_gen, tau_F)

    if save_traj_path is not None and emb_snaps:
        np.savez_compressed(
            save_traj_path,
            steps=np.array(steps_log, dtype=np.int32),
            emb=np.stack(emb_snaps, axis=0),
            train_acc=np.array(train_acc_log, dtype=np.float32),
            test_acc=np.array(test_acc_log, dtype=np.float32),
            f_raw=np.array(f_raw_log, dtype=np.float32),
            f_null_p95=np.array(f_null_log, dtype=np.float32),
            f_corr=np.array(f_corr_log, dtype=np.float32),
        )

    return {
        "decoder_wd": decoder_wd,
        "embed_wd": embed_wd,
        "shutoff_step": shutoff_step if shutoff_step is not None else "",
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


def enumerate_2x2():
    """4 cells × 3 seeds = 12 runs."""
    for dec in (0.0, WD_STAR):
        for emb in (0.0, WD_STAR):
            for seed in SEEDS:
                yield {"decoder_wd": dec, "embed_wd": emb,
                       "shutoff_step": None, "seed": seed,
                       "cfg_tag": f"dec{dec:g}_emb{emb:g}"}


def enumerate_schedule():
    """3 shutoff steps × 3 seeds = 9 runs (always decoder_wd=wd*, embed_wd=0)."""
    for s in (int(0.5 * TAU_GEN_REF), TAU_GEN_REF, 2 * TAU_GEN_REF):
        for seed in SEEDS:
            yield {"decoder_wd": WD_STAR, "embed_wd": 0.0,
                   "shutoff_step": s, "seed": seed,
                   "cfg_tag": f"shutoff{s}"}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--outdir", default="results/e4")
    p.add_argument("--max-steps", type=int, default=50_000)
    p.add_argument("--mode", choices=["2x2", "schedule", "both"],
                   default="both")
    p.add_argument("--save-traj", action="store_true",
                   help="Also save full embedding trajectories (~5 MB/run)")
    return p.parse_args()


def main():
    args = parse_args()
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    if args.save_traj:
        (outdir / "trajectories").mkdir(exist_ok=True)

    plan = []
    if args.mode in ("2x2", "both"):
        plan.extend(enumerate_2x2())
    if args.mode in ("schedule", "both"):
        plan.extend(enumerate_schedule())

    csv_path = outdir / "results.csv"
    fieldnames = [
        "cfg_tag", "decoder_wd", "embed_wd", "shutoff_step", "seed",
        "observed_phase", "tau_gen", "tau_F", "delta", "ordering",
        "max_train_acc", "max_test_acc", "runtime_sec",
    ]

    done = set()
    if csv_path.exists():
        with open(csv_path, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                done.add((row["cfg_tag"], int(row["seed"])))

    mode = "a" if csv_path.exists() else "w"
    cf = open(csv_path, mode, newline="", encoding="utf-8")
    writer = csv.DictWriter(cf, fieldnames=fieldnames)
    if mode == "w":
        writer.writeheader()

    print(f"E4 decoupled-wd sweep: {len(plan)} runs, mode={args.mode}, "
          f"max_steps={args.max_steps:,}")
    print(f"Output: {outdir}\n")

    for i, cfg in enumerate(plan, 1):
        key = (cfg["cfg_tag"], cfg["seed"])
        if key in done:
            print(f"[{i}/{len(plan)}] {cfg['cfg_tag']} seed={cfg['seed']} — SKIPPED")
            continue

        traj_path = None
        if args.save_traj:
            traj_path = outdir / "trajectories" / \
                f"traj_{cfg['cfg_tag']}_s{cfg['seed']}.npz"

        print(f"[{i}/{len(plan)}] {cfg['cfg_tag']} seed={cfg['seed']} "
              f"(dec_wd={cfg['decoder_wd']}, emb_wd={cfg['embed_wd']}, "
              f"shutoff={cfg['shutoff_step']}) ...", flush=True)

        r = _run(
            decoder_wd=cfg["decoder_wd"],
            embed_wd=cfg["embed_wd"],
            seed=cfg["seed"],
            shutoff_step=cfg["shutoff_step"],
            max_steps=args.max_steps,
            save_traj_path=traj_path,
        )

        row = {
            "cfg_tag": cfg["cfg_tag"],
            "decoder_wd": cfg["decoder_wd"],
            "embed_wd": cfg["embed_wd"],
            "shutoff_step": r["shutoff_step"],
            "seed": cfg["seed"],
            "observed_phase": r["observed_phase"],
            "tau_gen": r["tau_gen"] if r["tau_gen"] is not None else "",
            "tau_F": r["tau_F"] if r["tau_F"] is not None else "",
            "delta": r["delta"] if r["delta"] is not None else "",
            "ordering": r["ordering"],
            "max_train_acc": f"{r['max_train_acc']:.4f}",
            "max_test_acc": f"{r['max_test_acc']:.4f}",
            "runtime_sec": f"{r['runtime_sec']:.1f}",
        }
        writer.writerow(row)
        cf.flush()

        print(f"    phase={r['observed_phase']} | tau_gen={r['tau_gen']} | "
              f"tau_F={r['tau_F']} | ordering={r['ordering']} | "
              f"{r['runtime_sec']:.0f}s")

    cf.close()
    print(f"\nAll runs complete → {csv_path}")


if __name__ == "__main__":
    main()
