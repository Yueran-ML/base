#!/usr/bin/env python3
"""
step2_circuit_sweep.py — τ_circuit 三阶段时序测量
=================================================
重跑 Stage 2（lr=1.6e-3, wd sweep）和 Stage 3（wd=2.5, lr sweep）的全部
超参数格点，在训练循环中额外计算 Logit Fourier Alignment，用 BIC 变点检测
得到 τ_circuit。

三阶段假说：τ_circuit < τ_gen < τ_F

网格
----
  Stage 2: lr=1.6e-3, wd ∈ [1.2, 3.5] × 10 log-spaced, seeds=[42,7,2025]
  Stage 3: wd=2.5,   lr ∈ [5e-4,8e-3] × 10 log-spaced, seeds=[42,7,2025]
  合计: 20 × 3 = 60 cells（其中 lr=1.6e-3 / wd=2.5 在两组各出现一次）

断点续连
--------
  每个 cell 完成后立即写入 CSV；重启自动跳过已完成 (lr,wd,seed)。

用法
----
  python step2_circuit_sweep.py                    # 全部，顺序
  python step2_circuit_sweep.py --parallel 8      # 8 并行（A800）
  python step2_circuit_sweep.py --stage 2         # 只跑 Stage 2
  python step2_circuit_sweep.py --stage 3         # 只跑 Stage 3
  python step2_circuit_sweep.py --smoke           # 每组仅跑 1 cell × 1 seed
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

# ---------------------------------------------------------------------------
# Local imports
# ---------------------------------------------------------------------------
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
    compute_fourier_logit_alignment,
    compute_fourier_logit_null_p95,
    compute_fourier_null_p95,
    compute_ordering,
    estimate_changepoint,
    find_tau_sustained,
)

# ---------------------------------------------------------------------------
# Grid definitions
# ---------------------------------------------------------------------------

PRIME: int = 53
OPERATION: str = "add"
DEFAULT_SEEDS: list[int] = [42, 7, 2025]

# Stage 2: fixed lr, sweep wd
STAGE2_LR: float = 1.6e-3
STAGE2_WD_VALUES: list[float] = list(
    np.round(np.logspace(np.log10(1.2), np.log10(3.5), 10), 4).tolist()
)

# Stage 3: fixed wd, sweep lr
STAGE3_WD: float = 2.5
STAGE3_LR_VALUES: list[float] = list(
    np.round(np.logspace(np.log10(5e-4), np.log10(8e-3), 10), 6).tolist()
)

BASE_STEPS: int = 50_000
EXTEND_STEPS: int = 80_000

# Logging intervals
LOG_INTERVAL: int = 500        # accuracy + embedding Fourier
LOGIT_INTERVAL: int = 500      # logit Fourier (same cadence for fine resolution)
NULL_INTERVAL: int = 5_000     # permutation-null re-estimation

# ---------------------------------------------------------------------------
# Helper: get all logits from the model
# ---------------------------------------------------------------------------

def _get_all_logits(model: GrokkingTransformer, prime: int, device: torch.device) -> np.ndarray:
    """Forward pass on all (a,b) pairs. Returns shape (prime*prime, prime)."""
    a_vals = torch.arange(prime)
    b_vals = torch.arange(prime)
    aa, bb = torch.meshgrid(a_vals, b_vals, indexing="ij")
    all_pairs = torch.stack([aa.flatten(), bb.flatten()], dim=1).to(device)  # (p², 2)
    with torch.no_grad():
        logits = model(all_pairs).cpu().numpy()  # (p², p)
    return logits


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
    log_interval: int = LOG_INTERVAL,
    logit_interval: int = LOGIT_INTERVAL,
    null_interval: int = NULL_INTERVAL,
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

    # Logging containers
    steps_log: list[int] = []
    train_acc_log: list[float] = []
    test_acc_log: list[float] = []
    fourier_emb_log: list[float] = []   # embedding Fourier R² (corrected)
    fourier_logit_log: list[float] = [] # logit Fourier R² (corrected)
    logit_steps_log: list[int] = []

    null_rng_emb   = np.random.default_rng(seed ^ 0xDEAD)
    null_rng_logit = np.random.default_rng(seed ^ 0xBEEF)
    current_null95_emb   = 0.0
    current_null95_logit = 0.0

    step = 0
    while step < max_steps:
        for xb, yb in loader:
            model.train()
            logits_batch = model(xb.to(device))
            loss = criterion(logits_batch, yb.to(device))
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            step += 1

            log_now   = (step % log_interval == 0 or step == 1 or step >= max_steps)
            logit_now = (step % logit_interval == 0 or step == 1 or step >= max_steps)
            null_now  = (step % null_interval == 0 or step == 1)

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
                    f_logit_corr = max(0.0, f_logit_raw - current_null95_logit)

                    logit_steps_log.append(step)
                    fourier_logit_log.append(f_logit_corr)

            if step >= max_steps:
                break

    # Auto-extend: train converged but τ_gen not yet found
    max_train = max(train_acc_log) if train_acc_log else 0.0
    tau_gen = find_tau_sustained(steps_log, test_acc_log, threshold=0.9, n_sustained=3)
    if max_train > 0.9 and tau_gen is None and max_steps < EXTEND_STEPS:
        return run_one(
            lr=lr, wd=wd, seed=seed, prime=prime,
            train_fraction=train_fraction, max_steps=EXTEND_STEPS,
            log_interval=log_interval, logit_interval=logit_interval,
            null_interval=null_interval, n_null_perms=n_null_perms,
            embed_lr=embed_lr, d_model=d_model, n_heads=n_heads,
            n_layers=n_layers, d_ff=d_ff,
        )

    # Compute τ values
    observed_phase  = classify_phase(steps_log, train_acc_log, test_acc_log)
    tau_train       = find_tau_sustained(steps_log, train_acc_log, threshold=0.9, n_sustained=3)
    tau_F           = estimate_changepoint(steps_log, fourier_emb_log)
    tau_circuit     = estimate_changepoint(logit_steps_log, fourier_logit_log)

    delta_gf  = (tau_F - tau_gen)        if (tau_gen  is not None and tau_F      is not None) else None
    delta_cg  = (tau_gen - tau_circuit)  if (tau_circuit is not None and tau_gen is not None) else None
    delta_cf  = (tau_F - tau_circuit)    if (tau_circuit is not None and tau_F   is not None) else None

    # Three-stage ordering
    def _three_stage_ordering(tc, tg, tf):
        """Returns one of: C<G<F, C<F<G, G<C<F, G<F<C, F<C<G, F<G<C,
                            C<G=F, G=C<F, ..., or partial variants."""
        present = [(tc, "C"), (tg, "G"), (tf, "F")]
        present = [(t, lbl) for t, lbl in present if t is not None]
        if len(present) == 0:
            return "none"
        present.sort(key=lambda x: x[0])
        return "<".join(lbl for _, lbl in present)

    ordering_3stage = _three_stage_ordering(tau_circuit, tau_gen, tau_F)
    ordering_gf     = compute_ordering(tau_gen, tau_F)

    return {
        "operation": OPERATION, "prime": prime,
        "lr": lr, "wd": wd, "seed": seed,
        "observed_phase": observed_phase,
        "tau_train": tau_train,
        "tau_gen": tau_gen,
        "tau_circuit": tau_circuit,
        "tau_F": tau_F,
        "delta_gf": delta_gf,
        "delta_cg": delta_cg,
        "delta_cf": delta_cf,
        "ordering_gf": ordering_gf,
        "ordering_3stage": ordering_3stage,
        "max_train_acc": max_train,
        "max_test_acc": max(test_acc_log) if test_acc_log else 0.0,
        "runtime_sec": time.time() - t0,
    }


# ---------------------------------------------------------------------------
# CSV helpers (file-lock safe)
# ---------------------------------------------------------------------------

FIELDNAMES = [
    "operation", "prime", "lr", "wd", "seed",
    "observed_phase",
    "tau_train", "tau_gen", "tau_circuit", "tau_F",
    "delta_gf", "delta_cg", "delta_cf",
    "ordering_gf", "ordering_3stage",
    "max_train_acc", "max_test_acc", "runtime_sec",
]


def _append_row(path: Path, row: dict) -> None:
    write_header = not path.exists() or path.stat().st_size == 0
    with open(path, "a", newline="") as f:
        try:
            import fcntl
            fcntl.flock(f, fcntl.LOCK_EX)
        except Exception:
            pass
        w = csv.DictWriter(f, fieldnames=FIELDNAMES)
        if write_header:
            w.writeheader()
        w.writerow(row)
        try:
            import fcntl
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
# Worker
# ---------------------------------------------------------------------------

def _worker(args: tuple) -> dict:
    lr, wd, seed, csv_path_str = args
    result = run_one(lr=lr, wd=wd, seed=seed)
    _append_row(Path(csv_path_str), result)
    label = (
        f"lr={lr:.2e} wd={wd:.4f} seed={seed} → "
        f"phase={result['observed_phase']} "
        f"3stage={result['ordering_3stage']} "
        f"τ_circuit={result['tau_circuit']} "
        f"τ_gen={result['tau_gen']} "
        f"τ_F={result['tau_F']} "
        f"Δ(G-C)={result['delta_cg']} "
        f"t={result['runtime_sec']:.0f}s"
    )
    print(label, flush=True)
    return result


# ---------------------------------------------------------------------------
# Summary printer
# ---------------------------------------------------------------------------

def _print_summary(csv_path: Path, label: str = "Step 2") -> None:
    if not csv_path.exists():
        return
    with open(csv_path) as f:
        rows = list(csv.DictReader(f))

    grok = [r for r in rows if r.get("observed_phase") == "Grokking"]
    print(f"\n{'='*65}")
    print(f"  {label} — {csv_path.name}")
    print(f"  Total runs: {len(rows)}   Grokking: {len(grok)}")

    if not grok:
        from collections import Counter
        print(f"  Phase distribution: {dict(Counter(r['observed_phase'] for r in rows))}")
        print(f"{'='*65}\n")
        return

    # τ_circuit detection rate
    n_circuit = sum(1 for r in grok if r.get("tau_circuit") not in (None, "", "None"))
    print(f"  τ_circuit detected: {n_circuit}/{len(grok)}")

    # Three-stage ordering
    from collections import Counter
    orderings = Counter(r["ordering_3stage"] for r in grok)
    print(f"  3-stage ordering distribution: {dict(orderings)}")

    # τ_circuit < τ_gen rate
    n_cg = sum(1 for r in grok
               if r.get("ordering_3stage", "").startswith("C"))
    print(f"  C<G (τ_circuit before τ_gen): {n_cg}/{len(grok)} = {100*n_cg/len(grok):.1f}%")

    # Δ(G-C) = τ_gen - τ_circuit
    deltas_cg = [float(r["delta_cg"]) for r in grok
                 if r.get("delta_cg") not in (None, "", "None")]
    if deltas_cg:
        print(f"  Δ(τ_gen - τ_circuit): median={int(np.median(deltas_cg)):,} "
              f"  min={int(min(deltas_cg)):,}  max={int(max(deltas_cg)):,}")

    # Δ(F-G) = τ_F - τ_gen
    deltas_gf = [float(r["delta_gf"]) for r in grok
                 if r.get("delta_gf") not in (None, "", "None")]
    if deltas_gf:
        print(f"  Δ(τ_F - τ_gen):       median={int(np.median(deltas_gf)):,} "
              f"  min={int(min(deltas_gf)):,}  max={int(max(deltas_gf)):,}")

    print(f"{'='*65}\n")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Step 2: τ_circuit sweep (Stage 2+3 retrain)")
    parser.add_argument("--outdir", default="runs/step2_circuit",
                        help="Output directory (default: runs/step2_circuit)")
    parser.add_argument("--seeds", nargs="+", type=int, default=DEFAULT_SEEDS)
    parser.add_argument("--stage", choices=["2", "3", "all"], default="all",
                        help="Which stage grid to run (default: all)")
    parser.add_argument("--parallel", type=int, default=1,
                        help="Parallel workers (default: 1, A800 recommend 6-8)")
    parser.add_argument("--max-steps", type=int, default=BASE_STEPS)
    parser.add_argument("--smoke", action="store_true",
                        help="Smoke-test: 1 cell × 1 seed per stage, 5k steps")
    args = parser.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    csv_path = outdir / "results_step2_circuit.csv"

    seeds = args.seeds if not args.smoke else [42]
    max_steps = 5_000 if args.smoke else args.max_steps

    # Build task list
    tasks_all: list[tuple[float, float, int, str]] = []

    if args.stage in ("2", "all"):
        lr_list = [STAGE2_LR]
        wd_list  = STAGE2_WD_VALUES[:1] if args.smoke else STAGE2_WD_VALUES
        for wd in wd_list:
            for seed in seeds:
                tasks_all.append((STAGE2_LR, wd, seed, str(csv_path)))

    if args.stage in ("3", "all"):
        lr_list = STAGE3_LR_VALUES[:1] if args.smoke else STAGE3_LR_VALUES
        for lr in lr_list:
            for seed in seeds:
                # Skip duplicates that already appear in Stage 2
                if args.stage == "all" and abs(lr - STAGE2_LR) < 1e-9 and abs(STAGE3_WD - STAGE2_WD_VALUES[0]) < 1e-9:
                    continue
                tasks_all.append((lr, STAGE3_WD, seed, str(csv_path)))

    # Remove duplicates preserving order
    seen: set[tuple] = set()
    tasks_dedup: list[tuple] = []
    for t in tasks_all:
        key = (t[0], t[1], t[2])
        if key not in seen:
            seen.add(key)
            tasks_dedup.append(t)

    # Resume: skip done cells
    done_keys = _load_done_keys(csv_path)
    todo = [t for t in tasks_dedup if (t[0], t[1], t[2]) not in done_keys]

    total = len(tasks_dedup)
    print(f"\n{'─'*65}")
    print(f"  Step 2 — τ_circuit sweep  (p={PRIME}, {OPERATION} mod p)")
    print(f"  Stage 2: lr={STAGE2_LR:.1e}, wd∈[1.2,3.5]×10, seeds={seeds}")
    print(f"  Stage 3: wd={STAGE3_WD},   lr∈[5e-4,8e-3]×10, seeds={seeds}")
    print(f"  Progress: {total - len(todo)}/{total} done, {len(todo)} remaining")
    print(f"  Workers: {args.parallel}")
    print(f"  Output:  {csv_path}")
    if args.smoke:
        print(f"  [SMOKE TEST: 1 cell/stage, {max_steps} steps]")
    print(f"{'─'*65}\n")

    if not todo:
        print("All cells complete.")
        _print_summary(csv_path)
        return

    if args.parallel <= 1:
        for i, task in enumerate(todo, 1):
            lr, wd, seed, _ = task
            print(f"[{i}/{len(todo)}] lr={lr:.2e} wd={wd:.4f} seed={seed} ...",
                  end=" ", flush=True)
            t0 = time.time()
            # Inject max_steps override if smoke / custom
            result = run_one(lr=lr, wd=wd, seed=seed, max_steps=max_steps)
            _append_row(csv_path, result)
            elapsed = int(time.time() - t0)
            print(f"done [{elapsed}s] → {result['observed_phase']} "
                  f"3stage={result['ordering_3stage']} "
                  f"τ_C={result['tau_circuit']} τ_G={result['tau_gen']} τ_F={result['tau_F']}")
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
                    print(f"  ERROR lr={t[0]:.2e} wd={t[1]:.4f} seed={t[2]}: {e}")
                if done % 5 == 0 or done == len(todo):
                    print(f"  Progress: {done}/{len(todo)} completed", flush=True)

    _print_summary(csv_path)


if __name__ == "__main__":
    main()
