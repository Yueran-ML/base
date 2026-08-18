#!/usr/bin/env python3
"""
Dense-checkpoint causal probe.

Sister to ``causal_probe_checkpoints.py``. Same 5 cells, same training
loop, but saves model state_dicts at a DENSE grid of checkpoints
(every ``--dense-stride`` steps, default 1000) covering from before
the earliest pre-detected event through after the latest one, with
margin.

Purpose
-------
The headline E7 result uses 4 checkpoints per cell (one per regime
boundary), which gives a discrete 4-point story. To turn it into a
smooth ``I2 drop vs step`` curve we need more checkpoints. This
script produces ~10-20 checkpoints per cell so the existing
``causal_fourier_intervention.py`` (or its top-K spectrum sibling)
can be run on the dense grid to localize *when* the load-bearing
Fourier subspace becomes causally necessary.

This is a defensive experiment: it does NOT change any pre-existing
claim. It produces a finer-resolution version of E7 that strengthens
the time-resolved causal story.

Each saved checkpoint is auto-tagged with a regime label
``pre_all`` / ``between_<a>_<b>`` / ``post_all`` based on which
interval the step falls in (using the cell's pre-detected
tau_circuit / tau_gen / tau_F).

Output
------
  runs/causal_probe_dense/<cell-label>/checkpoints/step_<NNNNNN>.pt
  runs/causal_probe_dense/<cell-label>/meta.json
  runs/causal_probe_dense/<cell-label>/train_log.csv

After running this, run any of:
  python src/analysis/causal_fourier_intervention.py \
         --probe-root runs/causal_probe_dense \
         --outdir results/causal_probe_dense
  python src/analysis/causal_fourier_topk_spectrum.py \
         --probe-root runs/causal_probe_dense \
         --outdir results/causal_probe_dense

Usage
-----
  python src/sweeps/causal_probe_dense_checkpoints.py
  python src/sweeps/causal_probe_dense_checkpoints.py --dense-stride 500
  python src/sweeps/causal_probe_dense_checkpoints.py --cells strong_CGF G_CF_boundary

Compute estimate
----------------
5 cells x 50k-80k steps on RTX 4080 ~ 20-45 min per cell, so the
full sweep is ~ 2-4 GPU-hours. With more cells / shorter stride the
compute and disk footprint grow linearly.
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

# Reuse the SAME 5 cells as causal_probe_checkpoints.py so the dense
# probe is directly comparable to the headline E7 result.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from causal_probe_checkpoints import (  # noqa: E402
    CELLS,
    LOG_INTERVAL,
    PRIME,
    EMBED_LR,
    _snap,
)


def _dense_target_steps(cell: dict, stride: int, max_steps: int,
                        margin_frac: float = 0.25) -> list[int]:
    """Pick a dense grid of target checkpoints covering the relevant
    region: from (0.5 * earliest event) through (latest event +
    margin), every `stride` steps, snapped to LOG_INTERVAL."""
    events = sorted([cell["tau_circuit"], cell["tau_gen"], cell["tau_F"]])
    e_first, e_last = events[0], events[-1]
    span = e_last - e_first
    start = max(LOG_INTERVAL, _snap(e_first * 0.5))
    end = min(max_steps, _snap(e_last + span * margin_frac))
    if end <= start:
        end = min(max_steps, start + stride)
    steps = []
    s = start
    while s <= end:
        steps.append(_snap(s))
        s += stride
    # Always include the four canonical regime checkpoints so the
    # dense grid is a strict superset of the headline E7 sampling.
    canonical = [
        _snap(e_first * 0.5),
        _snap((events[0] + events[1]) / 2)
            if events[1] - events[0] >= 1000 else None,
        _snap((events[1] + events[2]) / 2)
            if events[2] - events[1] >= 1000 else None,
        _snap(min(max_steps, events[2] + span * 0.25)),
    ]
    for c in canonical:
        if c is not None and c not in steps:
            steps.append(c)
    return sorted(set(steps))


def _regime_label_for_step(step: int, cell: dict) -> str:
    """Auto-assign a regime label based on which interval step falls in."""
    events = sorted([
        ("circ", cell["tau_circuit"]),
        ("gen", cell["tau_gen"]),
        ("F", cell["tau_F"]),
    ], key=lambda e: e[1])
    e0_n, e0 = events[0]
    e1_n, e1 = events[1]
    e2_n, e2 = events[2]
    if step < e0:
        return "pre_all"
    if step < e1:
        return f"between_{e0_n}_{e1_n}"
    if step < e2:
        return f"between_{e1_n}_{e2_n}"
    return "post_all"


def run_one(label: str, cell: dict, outdir: Path, max_steps: int,
            stride: int) -> dict:
    target_steps = _dense_target_steps(cell, stride, max_steps)
    print(f"[{label}] cell={cell}")
    print(f"  dense targets ({len(target_steps)}): {target_steps}")

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

    target_set = set(target_steps)
    saved_cps: dict[str, dict] = {}
    train_log: list[dict] = []

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
            save_now = step in target_set

            if log_now or save_now:
                model.eval()
                with torch.no_grad():
                    tr_acc = (model(train_x.to(device)).argmax(1).cpu()
                              == train_y).float().mean().item()
                    te_acc = (model(test_x.to(device)).argmax(1).cpu()
                              == test_y).float().mean().item()
                emb = get_token_embeddings(model, PRIME)
                if step % 5000 == 0 or step == 1:
                    cur_null_emb = compute_fourier_null_p95(
                        emb, PRIME, n_perms=100, rng=null_rng_emb)
                f_emb_raw, _ = compute_fourier_alignment(emb, PRIME)
                f_corr = max(0.0, f_emb_raw - cur_null_emb)
                train_log.append({
                    "step": step,
                    "train_acc": tr_acc, "test_acc": te_acc,
                    "f_emb_raw": float(f_emb_raw),
                    "f_emb_corr": float(f_corr),
                })

                if save_now:
                    regime = _regime_label_for_step(step, cell)
                    cp_path = cp_outdir / f"step_{step:06d}.pt"
                    torch.save({
                        "model_state": model.state_dict(),
                        "step": step,
                        "regime": regime,
                        "train_acc": tr_acc,
                        "test_acc": te_acc,
                    }, cp_path)
                    # Use unique key so multiple checkpoints in same regime
                    # both survive in meta.json.
                    cp_key = f"{regime}_step{step:06d}"
                    saved_cps[cp_key] = {
                        "step": step, "regime": regime,
                        "train_acc": tr_acc, "test_acc": te_acc,
                        "path": str(cp_path.relative_to(outdir)),
                    }
                    print(f"  [save] step={step} regime={regime} "
                          f"train={tr_acc:.3f} test={te_acc:.3f}")

            if step >= max_steps:
                break

    # Write CSV log
    csv_path = cell_outdir / "train_log.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(train_log[0].keys()))
        w.writeheader(); w.writerows(train_log)

    # Write meta
    meta = {
        "label": label,
        "cell": cell,
        "stride": stride,
        "max_steps": max_steps,
        "n_target_checkpoints": len(target_steps),
        "saved_checkpoints": saved_cps,
        "elapsed_sec": time.time() - t0,
    }
    (cell_outdir / "meta.json").write_text(json.dumps(meta, indent=2))
    print(f"  done in {time.time() - t0:.0f}s; "
          f"{len(saved_cps)} checkpoints saved -> {cell_outdir}")
    return meta


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--outdir", default="runs/causal_probe_dense")
    parser.add_argument("--cells", nargs="+", default=list(CELLS.keys()),
                        choices=list(CELLS.keys()),
                        help="Subset of cells to run.")
    parser.add_argument("--dense-stride", type=int, default=1000,
                        help="Spacing between consecutive dense checkpoints.")
    parser.add_argument("--max-steps", type=int, default=50_000)
    parser.add_argument("--smoke", action="store_true",
                        help="Smoke run: print plan only, do not train.")
    args = parser.parse_args()

    outdir = Path(args.outdir); outdir.mkdir(parents=True, exist_ok=True)

    if args.smoke:
        for label in args.cells:
            cell = CELLS[label]
            steps = _dense_target_steps(cell, args.dense_stride, args.max_steps)
            print(f"[{label}] would save {len(steps)} checkpoints at: {steps}")
        return

    print(f"Dense causal probe: cells={args.cells}, "
          f"stride={args.dense_stride}, max_steps={args.max_steps:,}")
    print(f"Output: {outdir}\n")

    summary = {}
    for label in args.cells:
        cell = CELLS[label]
        meta = run_one(label, cell, outdir, args.max_steps, args.dense_stride)
        summary[label] = {
            "n_saved": len(meta["saved_checkpoints"]),
            "elapsed_sec": meta["elapsed_sec"],
        }
    (outdir / "run_summary.json").write_text(json.dumps(summary, indent=2))
    total = sum(s["elapsed_sec"] for s in summary.values())
    print(f"\nAll done in {total:.0f}s total")
    for k, s in summary.items():
        print(f"  {k:18s} {s['n_saved']:>3} checkpoints  {s['elapsed_sec']:.0f}s")


if __name__ == "__main__":
    main()
