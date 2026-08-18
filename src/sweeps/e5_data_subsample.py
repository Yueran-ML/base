#!/usr/bin/env python3
"""
E5 — Data-subsampling speed control.

Holds (lr, wd) fixed at the canonical Stage~2 operating point and varies the
training-set fraction to independently manipulate tau_gen while keeping the
weight-decay clock identical.  This directly tests the speed-dependent
ordering hypothesis (Section "A Speed-Dependent Ordering Hypothesis"):
if ordering tracks tau_gen rather than the weight-decay schedule, then
reducing the train fraction should delay tau_gen far enough that F<G becomes
reachable without changing (lr, wd, wd_embed).

Grid (12 runs)
--------------
  lr              = 1.6e-3     (canonical)
  wd_decoder      = 2.5        (canonical)
  wd_embed        = 0.0        (canonical, decoder-only decay)
  train_fraction  ∈ {0.2, 0.3, 0.4, 0.5}
  seeds           = [42, 7, 2025]
  max_steps       = 50_000
  prime           = 53, operation = "add"

Extended grid (optional, --extended): add fractions {0.15, 0.25, 0.35, 0.45}
and seeds [11, 17] for denser coverage (24 extra runs).

Output
------
  results/e5/results.csv
      per-run metrics (frac, seed, phase, tau_gen, tau_F, delta, ordering,
      max_train_acc, max_test_acc, runtime_sec)
  results/e5/e5_curves.png     (optional, if matplotlib available)

Usage
-----
  python src/sweeps/e5_data_subsample.py
  python src/sweeps/e5_data_subsample.py --extended
  python src/sweeps/e5_data_subsample.py --fractions 0.2 0.3 0.4
  python src/sweeps/e5_data_subsample.py --smoke        # 1 frac × 1 seed × 3k steps

Estimated compute
-----------------
  12 runs × ~13 min/run ≈ 2.5 h on RTX 4080 Laptop (50k steps each).
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
# Grid
# ---------------------------------------------------------------------------

PRIME: int = 53
OPERATION: str = "add"

FIXED_LR: float = 1.6e-3
FIXED_WD_DEC: float = 2.5
FIXED_WD_EMB: float = 0.0

DEFAULT_FRACTIONS: list[float] = [0.2, 0.3, 0.4, 0.5]
EXTENDED_FRACTIONS: list[float] = [0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5]

DEFAULT_SEEDS: list[int] = [42, 7, 2025]
EXTENDED_SEEDS: list[int] = [42, 7, 2025, 11, 17]

BASE_STEPS: int = 50_000
LOG_INTERVAL: int = 500
NULL_INTERVAL: int = 5_000
N_NULL_PERMS: int = 100


# ---------------------------------------------------------------------------
# Single-cell training
# ---------------------------------------------------------------------------

def run_one(
    train_fraction: float,
    seed: int,
    lr: float = FIXED_LR,
    wd_decoder: float = FIXED_WD_DEC,
    wd_embed: float = FIXED_WD_EMB,
    prime: int = PRIME,
    max_steps: int = BASE_STEPS,
    log_interval: int = LOG_INTERVAL,
    null_interval: int = NULL_INTERVAL,
    n_null_perms: int = N_NULL_PERMS,
    embed_lr: float = 1e-3,
    d_model: int = 256,
    n_heads: int = 4,
    n_layers: int = 2,
    d_ff: int = 1024,
) -> dict:
    """Train one (train_fraction, seed) cell; return scalar metrics dict."""
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
    optimizer = build_optimizer(
        model,
        decoder_lr=lr,
        embed_lr=embed_lr,
        decoder_weight_decay=wd_decoder,
        embed_weight_decay=wd_embed,
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
                    tr_acc = (model(train_x.to(device)).argmax(1).cpu()
                              == train_y).float().mean().item()
                    te_acc = (model(test_x.to(device)).argmax(1).cpu()
                              == test_y).float().mean().item()

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
    tau_gen = find_tau_sustained(steps_log, test_acc_log,
                                 threshold=0.9, n_sustained=3)
    tau_F = estimate_changepoint(steps_log, fourier_corr_log)

    delta = (tau_F - tau_gen) if (tau_gen is not None and tau_F is not None) else None
    ordering = compute_ordering(tau_gen, tau_F)

    return {
        "train_fraction": train_fraction,
        "seed": seed,
        "lr": lr,
        "wd_decoder": wd_decoder,
        "wd_embed": wd_embed,
        "observed_phase": observed_phase,
        "tau_gen": tau_gen,
        "tau_F": tau_F,
        "delta": delta,
        "ordering": ordering,
        "max_train_acc": max(train_acc_log),
        "max_test_acc": max(test_acc_log),
        "runtime_sec": time.time() - t0,
    }


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def make_curves(rows: list[dict], fractions: list[float], outdir: Path) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("  [plot] matplotlib not available, skipping")
        return

    from collections import defaultdict
    cell_rows: dict[float, list[dict]] = defaultdict(list)
    for r in rows:
        cell_rows[float(r["train_fraction"])].append(r)

    med_gen = []
    med_F = []
    med_delta = []
    p_gf = []
    p_fg = []
    ns = []
    for f in fractions:
        rr = cell_rows.get(f, [])
        ns.append(len(rr))
        if not rr:
            med_gen.append(np.nan); med_F.append(np.nan)
            med_delta.append(np.nan); p_gf.append(np.nan); p_fg.append(np.nan)
            continue
        gens = [float(r["tau_gen"]) for r in rr if r["tau_gen"] not in (None, "", "None")]
        fs   = [float(r["tau_F"])   for r in rr if r["tau_F"]   not in (None, "", "None")]
        dels = [float(r["delta"])   for r in rr if r["delta"]   not in (None, "", "None")]
        med_gen.append(np.median(gens) if gens else np.nan)
        med_F.append(np.median(fs) if fs else np.nan)
        med_delta.append(np.median(dels) if dels else np.nan)
        p_gf.append(sum(1 for r in rr if r["ordering"] == "G<F") / len(rr))
        p_fg.append(sum(1 for r in rr if r["ordering"] == "F<G") / len(rr))

    f_arr = np.array(fractions)
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.2), sharex=True)

    ax = axes[0]
    ax.plot(f_arr, np.array(med_gen) / 1000, "o-", color="#228833",
            label=r"median $\tau_{gen}$", linewidth=2, markersize=6)
    ax.plot(f_arr, np.array(med_F) / 1000, "s-", color="#EE6677",
            label=r"median $\tau_F$", linewidth=2, markersize=6)
    ax.set_xlabel("train_fraction"); ax.set_ylabel("step (k)")
    ax.legend(fontsize=9); ax.grid(True, alpha=0.3)
    ax.set_title("Timing vs train_fraction")

    ax = axes[1]
    ax.plot(f_arr, np.array(med_delta) / 1000, "^-", color="#4477AA",
            linewidth=2, markersize=6)
    ax.axhline(0, color="grey", linestyle="--", linewidth=0.8)
    ax.set_xlabel("train_fraction"); ax.set_ylabel(r"median $\Delta\tau$ (k)")
    ax.set_title("Ordering gap Δτ = τ_F − τ_gen")
    ax.grid(True, alpha=0.3)

    ax = axes[2]
    ax.plot(f_arr, p_gf, "o-", color="#228833", label="P(G<F)",
            linewidth=2, markersize=6)
    ax.plot(f_arr, p_fg, "v-", color="#CC3311", label="P(F<G)",
            linewidth=2, markersize=6)
    ax.set_xlabel("train_fraction"); ax.set_ylabel("fraction of seeds")
    ax.set_ylim(-0.05, 1.05); ax.legend(fontsize=9)
    ax.set_title("Ordering rate"); ax.grid(True, alpha=0.3)

    fig.suptitle(
        "E5 — Data-subsampling speed control "
        f"(lr={FIXED_LR}, wd_dec={FIXED_WD_DEC}, wd_emb={FIXED_WD_EMB}, 50k steps)",
        fontsize=11,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    out_path = outdir / "e5_curves.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {out_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--outdir", default="results/e5")
    p.add_argument("--max-steps", type=int, default=BASE_STEPS)
    p.add_argument("--fractions", type=float, nargs="+", default=None,
                   help="Train-set fractions to sweep (default: 0.2 0.3 0.4 0.5)")
    p.add_argument("--seeds", type=int, nargs="+", default=None)
    p.add_argument("--extended", action="store_true",
                   help="Use extended fractions+seeds grid (~40 runs, ~8.5h)")
    p.add_argument("--smoke", action="store_true",
                   help="1 frac × 1 seed × 3k steps — pipeline smoketest")
    return p.parse_args()


def main():
    args = parse_args()

    if args.smoke:
        fractions = [0.3]
        seeds = [42]
        max_steps = 3000
    else:
        if args.extended:
            fractions = args.fractions or EXTENDED_FRACTIONS
            seeds     = args.seeds or EXTENDED_SEEDS
        else:
            fractions = args.fractions or DEFAULT_FRACTIONS
            seeds     = args.seeds or DEFAULT_SEEDS
        max_steps = args.max_steps

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    csv_path = outdir / "results.csv"

    fieldnames = [
        "train_fraction", "seed",
        "lr", "wd_decoder", "wd_embed",
        "observed_phase", "tau_gen", "tau_F", "delta", "ordering",
        "max_train_acc", "max_test_acc", "runtime_sec",
    ]

    # Resume — treat rows whose runtime looks like a smoke test as invalid
    # so that a prior `--smoke` run does not poison the full sweep.
    # A real 50k-step cell on this machine is ~380s; 60s is a safe floor.
    SMOKE_RUNTIME_S = 60.0
    done: set[tuple[float, int]] = set()
    all_rows: list[dict] = []
    if csv_path.exists():
        # `utf-8-sig` tolerates a UTF-8 BOM, which Windows PowerShell's
        # `Set-Content -Encoding utf8` writes by default and which would
        # otherwise corrupt the first header cell ("\ufefftrain_fraction").
        with open(csv_path, newline="", encoding="utf-8-sig") as f:
            for row in csv.DictReader(f):
                try:
                    rt = float(row.get("runtime_sec", "") or "nan")
                except ValueError:
                    rt = float("nan")
                # Skip (do not mark as done) rows that are shorter than any
                # plausible full run — these are smoke-test residue.
                if not (rt >= SMOKE_RUNTIME_S):
                    print(f"[resume] dropping stale smoke row "
                          f"frac={row['train_fraction']} seed={row['seed']} "
                          f"runtime={rt:.1f}s")
                    continue
                all_rows.append(row)
                done.add((float(row["train_fraction"]), int(row["seed"])))

    total_runs = len(fractions) * len(seeds)
    print(f"E5 data subsample: {len(fractions)} fractions × {len(seeds)} seeds = "
          f"{total_runs} runs | max_steps={max_steps:,}")
    print(f"fractions: {fractions}")
    print(f"seeds:     {seeds}")
    print(f"lr={FIXED_LR}, wd_dec={FIXED_WD_DEC}, wd_emb={FIXED_WD_EMB}")
    print(f"Resuming: {len(done)} done, {total_runs - len(done)} remaining\n")

    csv_file = open(csv_path, "a", newline="", encoding="utf-8")
    writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
    if not done:
        writer.writeheader()

    run_idx = 0
    for frac in fractions:
        for seed in seeds:
            run_idx += 1
            if (frac, seed) in done:
                print(f"  [{run_idx}/{total_runs}] frac={frac:.2f} seed={seed} — SKIPPED")
                continue

            print(f"[{run_idx}/{total_runs}] frac={frac:.2f} seed={seed} ...",
                  end=" ", flush=True)
            res = run_one(train_fraction=frac, seed=seed, max_steps=max_steps)

            tg = f"{res['tau_gen']:.0f}" if res["tau_gen"] is not None else "—"
            tf = f"{res['tau_F']:.0f}"   if res["tau_F"]   is not None else "—"
            dl = f"{res['delta']:.0f}"   if res["delta"]   is not None else "—"
            print(f"phase={res['observed_phase']} | τg={tg} | τF={tf} | "
                  f"Δ={dl} | ord={res['ordering']} | {res['runtime_sec']:.0f}s")

            row = {
                "train_fraction": frac, "seed": seed,
                "lr": FIXED_LR, "wd_decoder": FIXED_WD_DEC,
                "wd_embed": FIXED_WD_EMB,
                "observed_phase": res["observed_phase"],
                "tau_gen": res["tau_gen"] if res["tau_gen"] is not None else "",
                "tau_F":   res["tau_F"]   if res["tau_F"]   is not None else "",
                "delta":   res["delta"]   if res["delta"]   is not None else "",
                "ordering": res["ordering"],
                "max_train_acc": f"{res['max_train_acc']:.4f}",
                "max_test_acc":  f"{res['max_test_acc']:.4f}",
                "runtime_sec":   f"{res['runtime_sec']:.1f}",
            }
            writer.writerow(row)
            csv_file.flush()
            all_rows.append({k: str(v) for k, v in row.items()})

    csv_file.close()
    print(f"\nAll runs complete. CSV → {csv_path}")

    # Plot + summary table
    plot_rows = []
    for r in all_rows:
        pr = dict(r)
        pr["delta"] = r["delta"] if r["delta"] != "" else None
        plot_rows.append(pr)
    make_curves(plot_rows, fractions, outdir)

    # Summary
    print(f"\n{'frac':>6}  {'n':>3}  {'med_τg(k)':>10}  {'med_τF(k)':>10}  "
          f"{'med_Δ(k)':>9}  {'P(G<F)':>7}  {'P(F<G)':>7}")
    print("-" * 65)
    from collections import defaultdict
    cell_rows: dict[float, list] = defaultdict(list)
    for r in plot_rows:
        cell_rows[float(r["train_fraction"])].append(r)
    for f in fractions:
        rr = cell_rows.get(f, [])
        n = len(rr)
        if n == 0:
            print(f"{f:>6.2f}  {n:>3}  {'—':>10}  {'—':>10}  {'—':>9}  {'—':>7}  {'—':>7}")
            continue
        gens = [float(r["tau_gen"]) for r in rr if r["tau_gen"] not in (None, "", "None")]
        fs   = [float(r["tau_F"])   for r in rr if r["tau_F"]   not in (None, "", "None")]
        dels = [float(r["delta"])   for r in rr if r["delta"]   not in (None, "", "None")]
        mg = np.median(gens) / 1000 if gens else float("nan")
        mf = np.median(fs) / 1000   if fs   else float("nan")
        md = np.median(dels) / 1000 if dels else float("nan")
        pgf = sum(1 for r in rr if r["ordering"] == "G<F") / n
        pfg = sum(1 for r in rr if r["ordering"] == "F<G") / n
        print(f"{f:>6.2f}  {n:>3}  {mg:>10.2f}  {mf:>10.2f}  "
              f"{md:>9.2f}  {pgf:>7.2f}  {pfg:>7.2f}")


if __name__ == "__main__":
    main()
