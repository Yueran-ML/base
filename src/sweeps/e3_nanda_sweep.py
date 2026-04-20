#!/usr/bin/env python3
"""
e3_nanda_sweep.py — E3: Nanda 2023 progress measures vs. our τ_circuit
======================================================================
对 Stage 4（Stage 2 + Stage 3 Grokking 格点）的 60 个 cell 重新训练，在每个
logit 采样点同时计算：

  1. f_logit_corr           —— 本文 τ_circuit 的 BIC 变点信号（保持不变）
  2. nanda_restricted_loss  —— Nanda 2023 的 restricted logit loss
  3. nanda_excluded_loss    —— Nanda 2023 的 excluded logit loss

Nanda 关键频率使用 *当前* 步的 token embedding FFT 取 top-K（默认 K=5），这是
inline 动态版本；为了允许事后按 Nanda 原始方法（使用训练结束的 static key
frequencies）重算，每个 cell 的最终 embedding 与最终 logits 一并存入 .npz。

产物
----
  runs/e3_nanda/results_e3_nanda.csv
      每行一个 cell，包含三个 τ_circuit 估计与成对差值
  runs/e3_nanda/traj_<lr>_<wd>_<seed>.npz
      每个 cell 的完整轨迹 + final embedding + final logits

用法
----
  python e3_nanda_sweep.py                    # 全部 cells 顺序跑
  python e3_nanda_sweep.py --parallel 6      # 6 并行
  python e3_nanda_sweep.py --stage 2         # 只跑 Stage 2
  python e3_nanda_sweep.py --grokking-only   # 跳过 Comprehension / Confusion 预期 cell
  python e3_nanda_sweep.py --top-k 4         # 改用 top-4 key frequencies
  python e3_nanda_sweep.py --smoke           # 每个 stage 1 cell × 1 seed × 5k 步
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
    compute_fourier_logit_alignment,
    compute_fourier_logit_null_p95,
    compute_fourier_null_p95,
    compute_nanda_losses,
    compute_ordering,
    estimate_changepoint,
    find_tau_sustained,
    identify_key_frequencies,
)

# ---------------------------------------------------------------------------
# Grid definitions (mirrors step2_circuit_sweep.py)
# ---------------------------------------------------------------------------

PRIME: int = 53
OPERATION: str = "add"
DEFAULT_SEEDS: list[int] = [42, 7, 2025]

STAGE2_LR: float = 1.6e-3
STAGE2_WD_VALUES: list[float] = list(
    np.round(np.logspace(np.log10(1.2), np.log10(3.5), 10), 4).tolist()
)

STAGE3_WD: float = 2.5
STAGE3_LR_VALUES: list[float] = list(
    np.round(np.logspace(np.log10(5e-4), np.log10(8e-3), 10), 6).tolist()
)

BASE_STEPS: int = 50_000
EXTEND_STEPS: int = 80_000

LOG_INTERVAL: int = 500
LOGIT_INTERVAL: int = 500
NULL_INTERVAL: int = 5_000

DEFAULT_TOP_K: int = 5


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _get_all_logits(model: GrokkingTransformer, prime: int, device: torch.device) -> np.ndarray:
    """Forward on every (a,b) pair. Returns (prime*prime, prime)."""
    a_vals = torch.arange(prime)
    b_vals = torch.arange(prime)
    aa, bb = torch.meshgrid(a_vals, b_vals, indexing="ij")
    all_pairs = torch.stack([aa.flatten(), bb.flatten()], dim=1).to(device)
    with torch.no_grad():
        logits = model(all_pairs).cpu().numpy()
    return logits


def _traj_filename(lr: float, wd: float, seed: int) -> str:
    """Filename-safe trajectory key."""
    return f"traj_lr{lr:.6e}_wd{wd:.4f}_seed{seed}.npz"


# ---------------------------------------------------------------------------
# Single-cell training
# ---------------------------------------------------------------------------

def run_one(
    lr: float,
    wd: float,
    seed: int,
    outdir: Path,
    prime: int = PRIME,
    train_fraction: float = 0.3,
    max_steps: int = BASE_STEPS,
    log_interval: int = LOG_INTERVAL,
    logit_interval: int = LOGIT_INTERVAL,
    null_interval: int = NULL_INTERVAL,
    n_null_perms: int = 100,
    top_k: int = DEFAULT_TOP_K,
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
    fourier_emb_log: list[float] = []

    logit_steps_log: list[int] = []
    fourier_logit_log: list[float] = []     # our τ_circuit signal
    nanda_restricted_log: list[float] = []  # Nanda restricted loss
    nanda_excluded_log: list[float] = []    # Nanda excluded loss
    nanda_full_log: list[float] = []        # full CE on unmodified logits (sanity)
    nanda_keys_log: list[list[int]] = []    # dynamic key freqs per checkpoint

    null_rng_emb   = np.random.default_rng(seed ^ 0xDEAD)
    null_rng_logit = np.random.default_rng(seed ^ 0xBEEF)
    current_null95_emb   = 0.0
    current_null95_logit = 0.0

    final_embedding: np.ndarray | None = None
    final_logits: np.ndarray | None = None

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

                    # Nanda metrics with dynamic (current-step) key freqs
                    key_freqs_now = identify_key_frequencies(
                        emb, prime, top_k=top_k, operation=OPERATION)
                    nanda = compute_nanda_losses(
                        all_logits, prime, key_freqs=key_freqs_now,
                        operation=OPERATION)

                    logit_steps_log.append(step)
                    fourier_logit_log.append(f_logit_corr)
                    nanda_restricted_log.append(nanda["restricted_loss"])
                    nanda_excluded_log.append(nanda["excluded_loss"])
                    nanda_full_log.append(nanda["full_loss"])
                    nanda_keys_log.append(key_freqs_now)

                    # Cache final state for .npz
                    final_embedding = emb
                    final_logits = all_logits

            if step >= max_steps:
                break

    # Auto-extend: train converged but τ_gen not yet found
    max_train = max(train_acc_log) if train_acc_log else 0.0
    tau_gen = find_tau_sustained(steps_log, test_acc_log, threshold=0.9, n_sustained=3)
    if max_train > 0.9 and tau_gen is None and max_steps < EXTEND_STEPS:
        return run_one(
            lr=lr, wd=wd, seed=seed, outdir=outdir, prime=prime,
            train_fraction=train_fraction, max_steps=EXTEND_STEPS,
            log_interval=log_interval, logit_interval=logit_interval,
            null_interval=null_interval, n_null_perms=n_null_perms,
            top_k=top_k, embed_lr=embed_lr,
            d_model=d_model, n_heads=n_heads, n_layers=n_layers, d_ff=d_ff,
        )

    # Compute τ values
    observed_phase  = classify_phase(steps_log, train_acc_log, test_acc_log)
    tau_train       = find_tau_sustained(steps_log, train_acc_log, threshold=0.9, n_sustained=3)
    tau_F           = estimate_changepoint(steps_log, fourier_emb_log)

    tau_circuit_ours        = estimate_changepoint(logit_steps_log, fourier_logit_log)
    # Nanda restricted_loss decreases during circuit formation → invert sign so
    # estimate_changepoint (which looks for slope magnitude change) fires symmetrically.
    tau_circuit_restricted  = estimate_changepoint(
        logit_steps_log, [-x for x in nanda_restricted_log])
    tau_circuit_excluded    = estimate_changepoint(logit_steps_log, nanda_excluded_log)

    # Final static-key freqs (Nanda's original methodology)
    static_keys: list[int] = []
    static_nanda: dict | None = None
    if final_embedding is not None and final_logits is not None:
        static_keys = identify_key_frequencies(
            final_embedding, prime, top_k=top_k, operation=OPERATION)
        static_nanda = compute_nanda_losses(
            final_logits, prime, key_freqs=static_keys, operation=OPERATION)

    # Save trajectories (.npz per cell)
    traj_path = outdir / _traj_filename(lr, wd, seed)
    np.savez_compressed(
        traj_path,
        operation=np.array(OPERATION),
        prime=np.array(prime),
        lr=np.array(lr, dtype=np.float64),
        wd=np.array(wd, dtype=np.float64),
        seed=np.array(seed, dtype=np.int64),
        top_k=np.array(top_k, dtype=np.int64),
        steps=np.array(steps_log, dtype=np.int64),
        train_acc=np.array(train_acc_log, dtype=np.float64),
        test_acc=np.array(test_acc_log, dtype=np.float64),
        fourier_emb_corr=np.array(fourier_emb_log, dtype=np.float64),
        logit_steps=np.array(logit_steps_log, dtype=np.int64),
        fourier_logit_corr=np.array(fourier_logit_log, dtype=np.float64),
        nanda_restricted=np.array(nanda_restricted_log, dtype=np.float64),
        nanda_excluded=np.array(nanda_excluded_log, dtype=np.float64),
        nanda_full=np.array(nanda_full_log, dtype=np.float64),
        # Pad variable-length keys with -1 to fixed width for np.save
        nanda_keys_dynamic=np.array(
            [k + [-1] * (top_k - len(k)) for k in nanda_keys_log],
            dtype=np.int64,
        ) if nanda_keys_log else np.zeros((0, top_k), dtype=np.int64),
        final_embedding=np.array(final_embedding, dtype=np.float32)
            if final_embedding is not None else np.zeros((0,), dtype=np.float32),
        final_logits=np.array(final_logits, dtype=np.float32)
            if final_logits is not None else np.zeros((0,), dtype=np.float32),
        static_key_frequencies=np.array(static_keys, dtype=np.int64),
    )

    # Summary row
    def _sub(a, b):
        if a is None or b is None:
            return None
        return a - b

    delta_restricted_vs_ours = _sub(tau_circuit_restricted, tau_circuit_ours)
    delta_excluded_vs_ours   = _sub(tau_circuit_excluded, tau_circuit_ours)
    delta_gc_ours            = _sub(tau_gen, tau_circuit_ours)
    delta_gc_restricted      = _sub(tau_gen, tau_circuit_restricted)

    return {
        "operation": OPERATION, "prime": prime,
        "lr": lr, "wd": wd, "seed": seed,
        "top_k": top_k,
        "observed_phase": observed_phase,
        "tau_train": tau_train,
        "tau_gen": tau_gen,
        "tau_F": tau_F,
        "tau_circuit_ours": tau_circuit_ours,
        "tau_circuit_nanda_restricted": tau_circuit_restricted,
        "tau_circuit_nanda_excluded": tau_circuit_excluded,
        "delta_restricted_vs_ours": delta_restricted_vs_ours,
        "delta_excluded_vs_ours": delta_excluded_vs_ours,
        "delta_gc_ours": delta_gc_ours,
        "delta_gc_nanda_restricted": delta_gc_restricted,
        "nanda_keys_static": ",".join(str(k) for k in static_keys),
        "nanda_restricted_loss_final": (
            static_nanda["restricted_loss"] if static_nanda else None),
        "nanda_excluded_loss_final": (
            static_nanda["excluded_loss"] if static_nanda else None),
        "nanda_full_loss_final": (
            static_nanda["full_loss"] if static_nanda else None),
        "max_train_acc": max_train,
        "max_test_acc": max(test_acc_log) if test_acc_log else 0.0,
        "trajectory_file": traj_path.name,
        "runtime_sec": time.time() - t0,
    }


# ---------------------------------------------------------------------------
# CSV I/O
# ---------------------------------------------------------------------------

FIELDNAMES = [
    "operation", "prime", "lr", "wd", "seed", "top_k",
    "observed_phase",
    "tau_train", "tau_gen", "tau_F",
    "tau_circuit_ours",
    "tau_circuit_nanda_restricted",
    "tau_circuit_nanda_excluded",
    "delta_restricted_vs_ours",
    "delta_excluded_vs_ours",
    "delta_gc_ours",
    "delta_gc_nanda_restricted",
    "nanda_keys_static",
    "nanda_restricted_loss_final",
    "nanda_excluded_loss_final",
    "nanda_full_loss_final",
    "max_train_acc", "max_test_acc",
    "trajectory_file", "runtime_sec",
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


def _load_grokking_keys(step2_csv: Path) -> set[tuple]:
    """Read Stage 4 result CSV and return the (lr,wd,seed) tuples labelled Grokking."""
    if not step2_csv.exists():
        return set()
    with open(step2_csv) as f:
        rows = list(csv.DictReader(f))
    keys = set()
    for r in rows:
        if r.get("observed_phase") == "Grokking":
            keys.add((float(r["lr"]), float(r["wd"]), int(r["seed"])))
    return keys


# ---------------------------------------------------------------------------
# Worker
# ---------------------------------------------------------------------------

def _worker(args: tuple) -> dict:
    lr, wd, seed, outdir_str, csv_path_str, top_k, max_steps = args
    outdir = Path(outdir_str)
    result = run_one(lr=lr, wd=wd, seed=seed, outdir=outdir,
                     top_k=top_k, max_steps=max_steps)
    _append_row(Path(csv_path_str), result)
    print(
        f"lr={lr:.2e} wd={wd:.4f} seed={seed} → "
        f"phase={result['observed_phase']} "
        f"τ_C(ours)={result['tau_circuit_ours']} "
        f"τ_C(restr)={result['tau_circuit_nanda_restricted']} "
        f"τ_C(excl)={result['tau_circuit_nanda_excluded']} "
        f"τ_G={result['tau_gen']} τ_F={result['tau_F']} "
        f"t={result['runtime_sec']:.0f}s",
        flush=True,
    )
    return result


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

def _print_summary(csv_path: Path) -> None:
    if not csv_path.exists():
        return
    with open(csv_path) as f:
        rows = list(csv.DictReader(f))

    grok = [r for r in rows if r.get("observed_phase") == "Grokking"]
    print(f"\n{'='*72}")
    print(f"  E3 — {csv_path.name}")
    print(f"  Total: {len(rows)}   Grokking: {len(grok)}")

    def _nfloat(val):
        if val in (None, "", "None"):
            return None
        try:
            return float(val)
        except Exception:
            return None

    # Detection rates
    det_ours   = sum(1 for r in grok if _nfloat(r.get("tau_circuit_ours")) is not None)
    det_restr  = sum(1 for r in grok if _nfloat(r.get("tau_circuit_nanda_restricted")) is not None)
    det_excl   = sum(1 for r in grok if _nfloat(r.get("tau_circuit_nanda_excluded")) is not None)
    print(f"  Detection:  ours={det_ours}/{len(grok)}  restricted={det_restr}/{len(grok)}  "
          f"excluded={det_excl}/{len(grok)}")

    # Pairwise agreement (our vs restricted)
    pairs_r = []
    for r in grok:
        a, b = _nfloat(r.get("tau_circuit_ours")), _nfloat(r.get("tau_circuit_nanda_restricted"))
        if a is not None and b is not None:
            pairs_r.append((a, b))
    if pairs_r:
        diffs = np.array([b - a for a, b in pairs_r])
        print(f"  τ_C(restr) − τ_C(ours): median={np.median(diffs):+.0f}  "
              f"mean={np.mean(diffs):+.0f}  "
              f"|Δ|≤500: {int(np.sum(np.abs(diffs)<=500))}/{len(pairs_r)}  "
              f"|Δ|≤2000: {int(np.sum(np.abs(diffs)<=2000))}/{len(pairs_r)}")
    print(f"{'='*72}\n")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="E3: Nanda restricted/excluded progress measures vs ours.")
    parser.add_argument("--outdir", default="runs/e3_nanda",
                        help="Output directory (default: runs/e3_nanda)")
    parser.add_argument("--seeds", nargs="+", type=int, default=DEFAULT_SEEDS)
    parser.add_argument("--stage", choices=["2", "3", "all"], default="all",
                        help="Which stage grid to run (default: all)")
    parser.add_argument("--parallel", type=int, default=1)
    parser.add_argument("--max-steps", type=int, default=BASE_STEPS)
    parser.add_argument("--top-k", type=int, default=DEFAULT_TOP_K,
                        help="Number of key frequencies (default: 5)")
    parser.add_argument("--grokking-only", action="store_true",
                        help="Only re-run cells labelled Grokking in "
                             "runs/step2_circuit/results_step2_circuit.csv")
    parser.add_argument("--step2-csv",
                        default="runs/step2_circuit/results_step2_circuit.csv",
                        help="Source CSV for --grokking-only filtering.")
    parser.add_argument("--smoke", action="store_true",
                        help="Smoke-test: 1 cell × 1 seed per stage, 5k steps")
    args = parser.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    csv_path = outdir / "results_e3_nanda.csv"

    seeds = args.seeds if not args.smoke else [42]
    max_steps = 5_000 if args.smoke else args.max_steps
    top_k = args.top_k

    # Build task list (same dedup as step2_circuit_sweep)
    tasks_all: list[tuple[float, float, int, str, str, int, int]] = []

    if args.stage in ("2", "all"):
        wd_list = STAGE2_WD_VALUES[:1] if args.smoke else STAGE2_WD_VALUES
        for wd in wd_list:
            for seed in seeds:
                tasks_all.append((STAGE2_LR, wd, seed,
                                  str(outdir), str(csv_path), top_k, max_steps))

    if args.stage in ("3", "all"):
        lr_list = STAGE3_LR_VALUES[:1] if args.smoke else STAGE3_LR_VALUES
        for lr in lr_list:
            for seed in seeds:
                tasks_all.append((lr, STAGE3_WD, seed,
                                  str(outdir), str(csv_path), top_k, max_steps))

    # Dedup preserving order
    seen: set[tuple] = set()
    tasks_dedup: list[tuple] = []
    for t in tasks_all:
        key = (t[0], t[1], t[2])
        if key not in seen:
            seen.add(key)
            tasks_dedup.append(t)

    # Optional filter: only Grokking cells from Stage 4
    if args.grokking_only:
        grok_keys = _load_grokking_keys(Path(args.step2_csv))
        before = len(tasks_dedup)
        tasks_dedup = [t for t in tasks_dedup if (t[0], t[1], t[2]) in grok_keys]
        print(f"  [--grokking-only] kept {len(tasks_dedup)}/{before} cells "
              f"from {args.step2_csv}")

    # Resume: skip done cells
    done_keys = _load_done_keys(csv_path)
    todo = [t for t in tasks_dedup if (t[0], t[1], t[2]) not in done_keys]

    total = len(tasks_dedup)
    print(f"\n{'─'*72}")
    print(f"  E3 — Nanda progress measures sweep  (p={PRIME}, {OPERATION} mod p)")
    print(f"  top_k = {top_k}")
    print(f"  Stage 2: lr={STAGE2_LR:.1e}, wd∈[1.2,3.5]×10, seeds={seeds}")
    print(f"  Stage 3: wd={STAGE3_WD},   lr∈[5e-4,8e-3]×10, seeds={seeds}")
    print(f"  Progress: {total - len(todo)}/{total} done, {len(todo)} remaining")
    print(f"  Workers: {args.parallel}    max_steps={max_steps}")
    print(f"  Output:  {csv_path}")
    print(f"  Trajectories: {outdir}/traj_*.npz")
    if args.smoke:
        print(f"  [SMOKE TEST]")
    print(f"{'─'*72}\n")

    if not todo:
        print("All cells complete.")
        _print_summary(csv_path)
        return

    if args.parallel <= 1:
        for i, task in enumerate(todo, 1):
            lr, wd, seed, _, _, _, _ = task
            print(f"[{i}/{len(todo)}] lr={lr:.2e} wd={wd:.4f} seed={seed} ...",
                  end=" ", flush=True)
            t0 = time.time()
            result = run_one(lr=lr, wd=wd, seed=seed, outdir=outdir,
                             top_k=top_k, max_steps=max_steps)
            _append_row(csv_path, result)
            elapsed = int(time.time() - t0)
            print(
                f"done [{elapsed}s] → {result['observed_phase']} "
                f"τ_C(ours)={result['tau_circuit_ours']} "
                f"τ_C(restr)={result['tau_circuit_nanda_restricted']} "
                f"τ_C(excl)={result['tau_circuit_nanda_excluded']}"
            )
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
