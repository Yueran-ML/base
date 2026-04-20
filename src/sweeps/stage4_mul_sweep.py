#!/usr/bin/env python3
"""
Stage 4: Second-Task Replication — (a·b) mod 53
================================================
Replicates the Stage-2 wd sweep but on the *multiplication* task to test
whether G<F ordering is specific to modular addition or holds more broadly.

Grid (matches Stage 2 to allow direct comparison):
  operation = "mul"
  prime     = 53
  lr        = 1.6e-3  (fixed, same as Stage 2)
  wd        = 10 log-spaced values in [1.2, 3.5]
  seeds     = [42, 7, 2025]
  max_steps = 50,000

Usage:
  python stage4_sub_sweep.py
  python stage4_sub_sweep.py --outdir runs/stage4_mul --seeds 42 7 2025
  python stage4_sub_sweep.py --wd-idx 0 1 2  # partial run
"""

from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path
from typing import Optional

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
# Grid definition
# ---------------------------------------------------------------------------

OPERATION: str = "sub"
PRIME: int = 53
FIXED_LR: float = 1.6e-3
WD_VALUES: list[float] = list(
    np.round(np.logspace(np.log10(1.2), np.log10(3.5), 10), 4).tolist()
)
DEFAULT_SEEDS: list[int] = [42, 7, 2025]


# ---------------------------------------------------------------------------
# Core run function
# ---------------------------------------------------------------------------


def run_one(
    lr: float,
    wd: float,
    seed: int,
    prime: int = PRIME,
    operation: str = OPERATION,
    train_fraction: float = 0.3,
    max_steps: int = 50_000,
    log_interval: int = 500,
    null_interval: int = 5000,
    n_null_perms: int = 100,
    embed_lr: float = 1e-3,
    d_model: int = 256,
    n_heads: int = 4,
    n_layers: int = 2,
    d_ff: int = 1024,
) -> dict:
    """Train one (lr, wd, seed) cell on the multiplication task; return scalar results."""
    t0 = time.time()
    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    x, y = make_dataset(prime, operation, seed)
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

        # Auto-extend to 80k if still grokking
        if step >= max_steps and not extended:
            last_train = train_acc_log[-1] if train_acc_log else 0.0
            last_test = test_acc_log[-1] if test_acc_log else 0.0
            tau_gen_check = find_tau_sustained(steps_log, test_acc_log, threshold=0.9, n_sustained=3)
            if last_train > 0.9 and tau_gen_check is None and max_steps < 80_000:
                print(f"    [extend] train_acc={last_train:.3f} but no tau_gen yet; extending to 80k")
                max_steps = 80_000
                extended = True
            else:
                break
        elif step >= max_steps:
            break

    # Post-training analysis
    observed_phase = classify_phase(steps_log, train_acc_log, test_acc_log)
    tau_gen = find_tau_sustained(steps_log, test_acc_log, threshold=0.9, n_sustained=3)
    tau_F = estimate_changepoint(steps_log, fourier_corr_log)

    delta = (tau_F - tau_gen) if (tau_gen is not None and tau_F is not None) else None
    ordering = compute_ordering(tau_gen, tau_F)

    return {
        "operation": operation,
        "prime": prime,
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
        "max_steps_used": max_steps,
        "runtime_sec": time.time() - t0,
    }


# ---------------------------------------------------------------------------
# Summary plot
# ---------------------------------------------------------------------------


def make_curves(rows: list[dict], wd_vals: list[float], outdir: Path) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("  [plot] matplotlib not available, skipping")
        return

    from collections import defaultdict
    wd_rows: dict[float, list[dict]] = defaultdict(list)
    for r in rows:
        wd_rows[float(r["wd"])].append(r)

    p_gf, med_delta, p_fonly, phases = [], [], [], []
    for wd in wd_vals:
        cell = wd_rows.get(wd, [])
        n = len(cell)
        if n == 0:
            p_gf.append(float("nan"))
            med_delta.append(float("nan"))
            p_fonly.append(float("nan"))
            phases.append("?")
            continue
        p_gf.append(sum(1 for r in cell if r["ordering"] == "G<F") / n)
        p_fonly.append(sum(1 for r in cell if r["ordering"] == "F_only") / n)
        deltas = [float(r["delta"]) for r in cell if r["delta"] not in (None, "", "None")]
        med_delta.append(float(np.median(deltas)) if deltas else float("nan"))
        from collections import Counter
        phase_counts = Counter(r["observed_phase"] for r in cell)
        phases.append(phase_counts.most_common(1)[0][0] if phase_counts else "?")

    wd_arr = np.array(wd_vals)
    fig, axes = plt.subplots(3, 1, figsize=(7, 8), sharex=True)
    fig.suptitle(f"Stage 4: G<F Ordering — ({OPERATION}) mod {PRIME}\n"
                 f"lr={FIXED_LR:.1e}, wd∈[{wd_vals[0]:.2f},{wd_vals[-1]:.2f}]", fontsize=12)

    ax0, ax1, ax2 = axes
    ax0.plot(wd_arr, p_gf, "o-", color="steelblue", label="P(G<F)")
    ax0.axhline(0.95, ls="--", color="grey", lw=0.8)
    ax0.set_ylabel("P(G<F)")
    ax0.set_ylim(-0.05, 1.1)
    ax0.legend(fontsize=9)

    med_delta_k = [d / 1000 if not np.isnan(d) else float("nan") for d in med_delta]
    ax1.bar(range(len(wd_vals)), med_delta_k, color="steelblue", alpha=0.7)
    ax1.set_xticks(range(len(wd_vals)))
    ax1.set_xticklabels([f"{w:.2f}" for w in wd_vals], rotation=45, fontsize=8)
    ax1.set_ylabel("Median Δτ (k steps)")
    ax1.axhline(0, color="black", lw=0.8)

    ax2.plot(wd_arr, p_fonly, "s--", color="darkorange", label="P(F_only)")
    ax2.set_ylabel("P(F_only)")
    ax2.set_ylim(-0.05, 1.1)
    ax2.set_xlabel("Weight decay")
    ax2.legend(fontsize=9)

    plt.tight_layout()
    out = outdir / "mul_sweep_curves.png"
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"  [plot] saved {out}")


# ---------------------------------------------------------------------------
# CSV helpers
# ---------------------------------------------------------------------------


def _write_csv(rows: list[dict], path: Path) -> None:
    if not rows:
        return
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"  [csv] saved {path}")


def _load_csv(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with open(path) as f:
        return list(csv.DictReader(f))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description="Stage 4: mul sweep")
    parser.add_argument("--outdir", default="runs/stage4_mul")
    parser.add_argument("--seeds", nargs="+", type=int, default=DEFAULT_SEEDS)
    parser.add_argument("--wd-idx", nargs="+", type=int, default=None,
                        help="Run only these wd indices (0-based)")
    parser.add_argument("--max-steps", type=int, default=50_000)
    args = parser.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    csv_path = outdir / "results_stage4_mul.csv"

    wd_vals = WD_VALUES
    if args.wd_idx is not None:
        wd_vals = [WD_VALUES[i] for i in args.wd_idx]

    existing = _load_csv(csv_path)
    done_keys = {(float(r["wd"]), int(r["seed"])) for r in existing}
    rows = list(existing)

    total = len(wd_vals) * len(args.seeds)
    done = 0
    for wd in wd_vals:
        for seed in args.seeds:
            done += 1
            key = (wd, seed)
            if key in done_keys:
                print(f"  [skip] wd={wd:.4f} seed={seed} already done")
                continue

            tag = f"wd={wd:.4f} seed={seed} ({done}/{total})"
            print(f"\n[run] {tag}  op={OPERATION} p={PRIME} lr={FIXED_LR:.1e}")
            result = run_one(
                lr=FIXED_LR, wd=wd, seed=seed,
                prime=PRIME, operation=OPERATION,
                max_steps=args.max_steps,
            )
            rows.append(result)
            _write_csv(rows, csv_path)

            phase_str = result["observed_phase"]
            ord_str = result["ordering"]
            tau_g = result["tau_gen"]
            tau_f = result["tau_F"]
            delta = result["delta"]
            print(f"  phase={phase_str}  ordering={ord_str}  "
                  f"tau_gen={tau_g}  tau_F={tau_f}  delta={delta}  "
                  f"max_test={result['max_test_acc']:.3f}  "
                  f"t={result['runtime_sec']:.0f}s")

    # Summary
    grokking_rows = [r for r in rows if r.get("observed_phase") == "Grokking"]
    if grokking_rows:
        n_gf = sum(1 for r in grokking_rows if r["ordering"] == "G<F")
        n_fg = sum(1 for r in grokking_rows if r["ordering"] == "F<G")
        n_co = sum(1 for r in grokking_rows if r["ordering"] == "coincident")
        n_fo = sum(1 for r in grokking_rows if r["ordering"] == "F_only")
        deltas = [float(r["delta"]) for r in grokking_rows if r["delta"] not in (None, "", "None")]
        med_d = int(np.median(deltas)) if deltas else None
        print(f"\n=== Stage 4 Summary ({OPERATION} mod {PRIME}) ===")
        print(f"  Grokking runs: {len(grokking_rows)}/{len(rows)}")
        print(f"  G<F: {n_gf}  F<G: {n_fg}  coincident: {n_co}  F_only: {n_fo}")
        print(f"  G<F rate: {n_gf/len(grokking_rows)*100:.1f}%")
        print(f"  Median Δτ: {med_d} steps")
    else:
        print(f"\n=== Stage 4 Summary ===")
        print(f"  No Grokking runs found in {len(rows)} total runs.")
        print(f"  Phase distribution: "
              + str({r["observed_phase"]: 0 for r in rows}))
        from collections import Counter
        phase_counts = Counter(r.get("observed_phase", "?") for r in rows)
        print(f"  {dict(phase_counts)}")

    make_curves(rows, WD_VALUES, outdir)
    print(f"\nDone. Results: {csv_path}")


if __name__ == "__main__":
    main()
