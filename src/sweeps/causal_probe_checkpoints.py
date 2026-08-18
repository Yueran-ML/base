#!/usr/bin/env python3
"""
Causal probe — re-train 5 representative cells and save model state_dicts at
4 strategic regime-boundary checkpoints per cell, for a Fourier-subspace
intervention analysis.

For each cell, we read pre-detected (tau_circuit, tau_gen, tau_F) from
results/step2_circuit/results_step2_circuit.csv, sort the three events along
the time axis, and pick four checkpoints that fall inside the four resulting
regimes (with one extra margin past the final event):

  regime_A: t < min event                   (pre-everything)
  regime_B: between the two earliest events
  regime_C: between the two latest events
  regime_D: t > max event                   (post-everything)

Special handling:
  - Coincident events (delta < 500 steps): the regime between them is empty
    and we skip that checkpoint, so a coincident cell may yield only 3 saved
    checkpoints instead of 4.
  - Steps are snapped to the nearest 500-step log point.

Output:
  runs/causal_probe/{label}/{lr}_{wd}_{seed}/
       checkpoints/step_{step:06d}.pt    -- model state_dict + step + acc
       train_log.csv                      -- per-step train/test acc, F_corr
       meta.json                          -- cell metadata + checkpoint list

Usage:
  python src/sweeps/causal_probe_checkpoints.py
  python src/sweeps/causal_probe_checkpoints.py --cells strong_CGF G_CF_boundary
  python src/sweeps/causal_probe_checkpoints.py --smoke
"""

from __future__ import annotations

import argparse
import csv
import json
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
    compute_fourier_alignment,
    compute_fourier_null_p95,
    compute_fourier_logit_alignment,
    compute_fourier_logit_null_p95,
)


# 5 cells with their (lr, wd, seed) and pre-detected (tau_circ, tau_gen, tau_F)
# from results/step2_circuit/results_step2_circuit.csv.
CELLS = {
    "strong_CGF": dict(
        lr=1.6e-3, wd=2.7591, seed=7,
        tau_circuit=10000.0, tau_gen=12500.0, tau_F=18500.0,
        ordering="C<G<F",
    ),
    "G_CF_boundary": dict(
        lr=2.333e-3, wd=2.5, seed=7,
        tau_circuit=8000.0, tau_gen=6500.0, tau_F=12000.0,
        ordering="G<C<F",
    ),
    "FG_slow": dict(
        lr=5.0e-4, wd=2.5, seed=42,
        tau_circuit=31500.0, tau_gen=44500.0, tau_F=42000.0,
        ordering="C<F<G",
    ),
    "canonical_GF": dict(
        lr=1.6e-3, wd=1.7145, seed=7,
        tau_circuit=9500.0, tau_gen=11000.0, tau_F=26000.0,
        ordering="C<G<F",
    ),
    "coincident": dict(
        lr=1.6e-3, wd=3.1075, seed=2025,
        tau_circuit=8000.0, tau_gen=9500.0, tau_F=19500.0,
        ordering="C<G<F",
    ),
}

LOG_INTERVAL = 500
PRIME = 53
EMBED_LR = 1e-3


def _snap(step: float) -> int:
    """Snap to nearest 500-step log point, in [500, T_max]."""
    return max(LOG_INTERVAL, int(round(step / LOG_INTERVAL)) * LOG_INTERVAL)


def _pick_checkpoints(cell: dict, max_steps: int) -> dict:
    """Return {regime_label: target_step} with up to 4 entries.

    Regimes are defined by sorting the three events along the time axis.
    """
    events = sorted([
        ("circ", cell["tau_circuit"]),
        ("gen", cell["tau_gen"]),
        ("F", cell["tau_F"]),
    ], key=lambda e: e[1])
    e1, e2, e3 = events[0][1], events[1][1], events[2][1]

    cps = {}
    cps["pre_all"] = _snap(e1 * 0.5)
    if e2 - e1 >= 1000:
        cps[f"between_{events[0][0]}_{events[1][0]}"] = _snap((e1 + e2) / 2)
    if e3 - e2 >= 1000:
        cps[f"between_{events[1][0]}_{events[2][0]}"] = _snap((e2 + e3) / 2)
    cps["post_all"] = _snap(min(max_steps, e3 + (e3 - e1) * 0.25))

    # Dedupe: if two checkpoints land at the same step, keep one
    seen = set()
    unique = {}
    for k, v in cps.items():
        if v not in seen:
            unique[k] = v
            seen.add(v)
    return unique


def run_one(label: str, cell: dict, outdir: Path, max_steps: int) -> dict:
    target_cps = _pick_checkpoints(cell, max_steps)
    print(f"[{label}] cell={cell}; target checkpoints: {target_cps}")

    set_seed(cell["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    x, y = make_dataset(PRIME, "add", cell["seed"])
    train_x, train_y, test_x, test_y = split_dataset(x, y, train_fraction=0.3)

    model = GrokkingTransformer(
        prime=PRIME, d_model=256, n_heads=4, n_layers=2, d_ff=1024,
        dropout=0.0,
    ).to(device)

    loader = DataLoader(
        TensorDataset(train_x, train_y),
        batch_size=len(train_x), shuffle=True,
        generator=torch.Generator().manual_seed(cell["seed"]),
    )
    criterion = nn.CrossEntropyLoss()
    optimizer = build_optimizer(model, decoder_lr=cell["lr"],
                                embed_lr=EMBED_LR,
                                decoder_weight_decay=cell["wd"])
    scheduler = build_scheduler(optimizer, warmup_steps=10,
                                lr_schedule="constant", lr_min_ratio=0.05,
                                max_steps=max_steps)

    cell_outdir = outdir / label
    cp_outdir = cell_outdir / "checkpoints"
    cp_outdir.mkdir(parents=True, exist_ok=True)

    target_steps_set = set(target_cps.values())
    saved_cps: dict[str, dict] = {}

    train_log = []
    null_rng_emb = np.random.default_rng(cell["seed"] ^ 0xDEAD)
    null_rng_logit = np.random.default_rng(cell["seed"] ^ 0xBEEF)
    cur_null_emb = 0.0
    cur_null_logit = 0.0

    step = 0
    t0 = time.time()
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

            log_now = (step % LOG_INTERVAL == 0 or step == 1
                       or step >= max_steps)

            if log_now or step in target_steps_set:
                model.eval()
                with torch.no_grad():
                    tr_acc = (model(train_x.to(device)).argmax(1).cpu()
                              == train_y).float().mean().item()
                    te_acc = (model(test_x.to(device)).argmax(1).cpu()
                              == test_y).float().mean().item()
                emb = get_token_embeddings(model, PRIME)
                if step % 5000 == 0 or step == 1:
                    cur_null_emb = compute_fourier_null_p95(
                        emb, PRIME, n_perms=100, rng=null_rng_emb,
                    )
                f_emb_raw, _ = compute_fourier_alignment(emb, PRIME)
                f_corr = max(0.0, f_emb_raw - cur_null_emb)
                train_log.append({
                    "step": step, "train_acc": tr_acc, "test_acc": te_acc,
                    "f_emb_corr": f_corr,
                })

            if step in target_steps_set:
                # Find which regime label this step belongs to
                regime_for_step = [k for k, v in target_cps.items()
                                   if v == step]
                for regime in regime_for_step:
                    cp_path = cp_outdir / f"step_{step:06d}_{regime}.pt"
                    torch.save({
                        "model_state": model.state_dict(),
                        "step": step,
                        "regime": regime,
                        "train_acc": tr_acc,
                        "test_acc": te_acc,
                    }, cp_path)
                    saved_cps[regime] = {
                        "step": step,
                        "path": str(cp_path.relative_to(outdir)),
                        "train_acc": tr_acc,
                        "test_acc": te_acc,
                    }
                    print(f"  [save] {regime} @ step {step}: "
                          f"train={tr_acc:.3f} test={te_acc:.3f}")

            if step >= max_steps:
                break
        if step >= max_steps:
            break

    # Persist train log + meta
    with open(cell_outdir / "train_log.csv", "w", newline="") as f:
        if train_log:
            w = csv.DictWriter(f, fieldnames=list(train_log[0].keys()))
            w.writeheader(); w.writerows(train_log)

    meta = {
        "label": label,
        "cell": cell,
        "max_steps": max_steps,
        "target_checkpoints": target_cps,
        "saved_checkpoints": saved_cps,
        "runtime_sec": time.time() - t0,
    }
    with open(cell_outdir / "meta.json", "w") as f:
        json.dump(meta, f, indent=2)
    print(f"[{label}] done, {len(saved_cps)} checkpoints saved, "
          f"{meta['runtime_sec']:.0f}s")
    return meta


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--outdir", default="runs/causal_probe")
    parser.add_argument("--cells", nargs="+", default=list(CELLS.keys()))
    parser.add_argument("--max-steps", type=int, default=80_000)
    parser.add_argument("--smoke", action="store_true",
                        help="Run only the strong_CGF cell at 5k steps.")
    args = parser.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    cells_to_run = args.cells
    max_steps = args.max_steps
    if args.smoke:
        cells_to_run = ["strong_CGF"]
        max_steps = 5_000

    metas = []
    for label in cells_to_run:
        if label not in CELLS:
            print(f"  [skip] unknown cell label: {label}")
            continue
        cell_meta_path = outdir / label / "meta.json"
        if cell_meta_path.exists():
            print(f"  [skip] {label} already has meta.json; "
                  f"delete to re-run.")
            continue
        meta = run_one(label, CELLS[label], outdir, max_steps)
        metas.append(meta)

    summary = {
        "n_cells_run": len(metas),
        "cell_labels": [m["label"] for m in metas],
        "total_runtime_sec": sum(m["runtime_sec"] for m in metas),
    }
    with open(outdir / "run_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nAll done: {summary['n_cells_run']} cells in "
          f"{summary['total_runtime_sec']:.0f}s.")


if __name__ == "__main__":
    main()
