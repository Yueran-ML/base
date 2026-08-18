#!/usr/bin/env python3
"""
E7 expansion: extend causal-probe from 5 to 12 cells.

This script is a defensive companion to ``causal_probe_checkpoints.py``.
Its sole purpose is to address the predicted reviewer attack
"E7 is a 5-cell causal-localization probe, which is too small to
support a causal-dissociation claim."

Design principle: sample *diversity* over sample *count*. The 7 new
cells vary along five independent axes, so each axis cell answers a
different reviewer question:

  axis        | cell label              | reviewer question it answers
  ------------|-------------------------|---------------------------------
  seed        | canonical_GF_s42        | "is the effect seed-specific?"
  seed        | canonical_GF_s2025      | "is the effect seed-specific?"
  wd          | wd_low_GF (wd=1.5223)   | "is the effect wd-specific?"
  wd          | wd_high_GF (wd=3.1075)  | "is the effect wd-specific?"
  lr          | lr_low_GF (lr=9.259e-4) | "is the effect lr-specific?"
  task        | mul_GF (a*b mod 53)     | "is the effect addition-specific?"
  prime       | p97_GF (p=97 add)       | "is the effect p=53-specific?"

Cell-design rationale
---------------------
- canonical_GF_s42 / canonical_GF_s2025: clone the existing
  ``canonical_GF`` cell at (lr=1.6e-3, wd=1.7145) and change only the
  seed. These cells force any "the effect comes from one lucky
  random init" attack to require an unusually large number of
  coincidences across seeds.

- wd_low_GF / wd_high_GF: stay at canonical lr=1.6e-3 but move to
  nearby Stage-2 wd-grid corners (1.5223 and 3.1075). These cells
  have pre-detected tau_gen and tau_F, but not tau_circuit, so the
  checkpoint labels are marked as partial.

- lr_low_GF: stay at canonical wd=2.5 but lower lr to 9.259e-4,
  the nearest existing Stage-3 lr-grid point below 1e-3. It has
  pre-detected tau_gen and tau_F, but not tau_circuit.

- mul_GF: Stage 5A canonical (a*b mod 53) cell. The logit-Fourier
  intervention applies directly without discrete-log reindexing
  because the LOGITS are still indexed by the answer class 0..p-1;
  only the embedding-side Fourier analysis (which we are not the
  primary focus of E7) requires reindexing.

- p97_GF: Stage 5B p=97 cell at the nearest existing wd-grid point
  to canonical. Median tau_F is much later for p=97 (~28k steps per
  paper), so this cell uses max_steps=100k by default.

For the new cells the pre-detected (tau_circuit, tau_gen, tau_F)
values may need to be looked up from your existing per-stage results
CSVs and pasted into the CELLS_EXPANSION dict below. If they are
left as None, the script falls back to a fixed-fraction checkpoint
schedule (15% / 35% / 55% / 80% of max_steps) so the experiment is
still runnable, but the regime labels will then be generic and you
should not interpret "between_<a>_<b>" as a real event-bracketing
checkpoint until you fill in the tau values.

Output
------
  runs/causal_probe_expansion/{label}/
       checkpoints/step_{step:06d}_{regime}.pt
       train_log.csv
       meta.json

After running, point the existing analysis scripts at the new probe
root:

  python src/analysis/causal_fourier_intervention.py \
         --probe-root runs/causal_probe_expansion \
         --outdir results/causal_probe_expansion
  python src/analysis/causal_fourier_topk_spectrum.py \
         --probe-root runs/causal_probe_expansion \
         --outdir results/causal_probe_expansion
  python src/analysis/causal_fourier_freq_selectivity.py \
         --probe-root runs/causal_probe_expansion \
         --outdir results/causal_probe_expansion

Then merge the results CSVs from runs/causal_probe and
runs/causal_probe_expansion to produce the n=12 headline numbers.
``tools/merge_causal_probe_results.py`` can do this in one line; see
its docstring for usage.

Compute estimate
----------------
7 cells x 50k-100k steps on RTX 4080:
  5 add p=53 cells (50k-80k steps each): ~ 2-3 GPU-hours
  1 mul p=53 cell (50k steps):            ~ 25-30 min
  1 add p=97 cell (100k steps):           ~ 45-55 min
Total: ~ 3-4 GPU-hours.

Usage
-----
  # Smoke test (plan only, no training):
  python src/sweeps/causal_probe_expansion.py --smoke

  # Single cell, useful for first verification:
  python src/sweeps/causal_probe_expansion.py --cells canonical_GF_s42

  # Full sweep:
  python src/sweeps/causal_probe_expansion.py

  # If you've filled in tau values and want to re-run a single cell:
  rm -rf runs/causal_probe_expansion/canonical_GF_s42
  python src/sweeps/causal_probe_expansion.py --cells canonical_GF_s42
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
)

# Reuse helpers from the canonical 5-cell script so behaviour matches
# byte-for-byte.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from causal_probe_checkpoints import _snap, LOG_INTERVAL, EMBED_LR  # noqa: E402


# ---------------------------------------------------------------------------
# 7 new cells extending the original 5.
#
# tau values left as None will trigger fallback fixed-fraction checkpoint
# scheduling. If you have detected tau values from results/step2_circuit/,
# results/stage5a/, results/stage5b/, or results/stage6/, paste them in
# here and the regime-boundary checkpoint logic in the original 5-cell
# script will be used directly.
# ---------------------------------------------------------------------------
CELLS_EXPANSION = {
    # ---- seed diversity at canonical (lr, wd) ----
    "canonical_GF_s42": dict(
        lr=1.6e-3, wd=1.7145, seed=42,
        task="add", prime=53,
        tau_circuit=11_000.0, tau_gen=18_000.0, tau_F=32_500.0,
        ordering="C<G<F",  # expected
        max_steps=80_000,
    ),
    "canonical_GF_s2025": dict(
        lr=1.6e-3, wd=1.7145, seed=2025,
        task="add", prime=53,
        tau_circuit=8_000.0, tau_gen=16_000.0, tau_F=36_000.0,
        ordering="C<G<F",
        max_steps=80_000,
    ),

    # ---- wd corners at canonical lr ----
    "wd_low_GF": dict(
        lr=1.6e-3, wd=1.5223, seed=42,
        task="add", prime=53,
        tau_circuit=None, tau_gen=14_000.0, tau_F=21_500.0,
        ordering="G<F",
        max_steps=80_000,
    ),
    "wd_high_GF": dict(
        lr=1.6e-3, wd=3.1075, seed=42,
        task="add", prime=53,
        tau_circuit=None, tau_gen=12_000.0, tau_F=24_500.0,
        ordering="G<F",
        max_steps=80_000,
    ),

    # ---- lr corner at canonical wd ----
    "lr_low_GF": dict(
        lr=9.259e-4, wd=2.5, seed=42,
        task="add", prime=53,
        tau_circuit=None, tau_gen=21_500.0, tau_F=29_000.0,
        ordering="G<F",
        max_steps=80_000,
    ),

    # ---- task diversity: multiplication (Stage 5A canonical) ----
    "mul_GF": dict(
        lr=1.6e-3, wd=2.4497, seed=42,
        task="mul", prime=53,
        tau_circuit=None, tau_gen=8_500.0, tau_F=14_000.0,
        ordering="G<F",
        max_steps=50_000,
    ),

    # ---- prime diversity: p=97 (Stage 5B canonical) ----
    "p97_GF": dict(
        lr=1.6e-3, wd=2.4497, seed=42,
        task="add", prime=97,
        tau_circuit=None, tau_gen=2_500.0, tau_F=30_500.0,
        ordering="G<F",
        max_steps=100_000,
    ),
}


def _pick_checkpoints_with_taus(cell: dict, max_steps: int) -> dict:
    """Same regime-boundary logic as ``causal_probe_checkpoints._pick_checkpoints``,
    requires all three taus to be set."""
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

    seen = set()
    unique = {}
    for k, v in cps.items():
        if v not in seen:
            unique[k] = v
            seen.add(v)
    return unique


def _pick_checkpoints_fallback(max_steps: int) -> dict:
    """Fallback when tau values aren't available: pick 4 checkpoints
    at fixed fractions of max_steps. The regime labels are generic
    placeholders; you should NOT interpret ``between_*`` literally
    until tau values are filled in."""
    fractions = {
        "pre_all":             0.15,
        "between_a_b":         0.35,  # placeholder regime label
        "between_b_c":         0.55,  # placeholder regime label
        "post_all":            0.80,
    }
    return {k: _snap(max_steps * f) for k, f in fractions.items()}


def _pick_checkpoints_partial(cell: dict, max_steps: int) -> dict:
    """Use the known event times without pretending unknown tau_circuit is
    available. These labels are intentionally generic enough that downstream
    summaries do not treat them as full three-stage regime boundaries."""
    known = sorted(
        [(name, cell[name]) for name in ("tau_circuit", "tau_gen", "tau_F")
         if cell[name] is not None],
        key=lambda item: item[1],
    )
    if len(known) < 2:
        return _pick_checkpoints_fallback(max_steps)

    e1_name, e1 = known[0]
    e2_name, e2 = known[-1]
    e1_short = e1_name.removeprefix("tau_")
    e2_short = e2_name.removeprefix("tau_")
    span = max(LOG_INTERVAL * 2, e2 - e1)
    cps = {
        "pre_known_events": _snap(e1 * 0.5),
        f"between_{e1_short}_{e2_short}_partial": _snap((e1 + e2) / 2),
        "post_known_events": _snap(min(max_steps, e2 + span * 0.25)),
    }
    return cps


def _pick_checkpoints(cell: dict, max_steps: int) -> tuple[dict, str]:
    """Returns (checkpoint_dict, schedule_mode)."""
    if (cell["tau_circuit"] is not None
            and cell["tau_gen"] is not None
            and cell["tau_F"] is not None):
        return _pick_checkpoints_with_taus(cell, max_steps), "full"
    if sum(cell[name] is not None
           for name in ("tau_circuit", "tau_gen", "tau_F")) >= 2:
        return _pick_checkpoints_partial(cell, max_steps), "partial"
    return _pick_checkpoints_fallback(max_steps), "fallback"


def run_one(label: str, cell: dict, outdir: Path) -> dict:
    max_steps = cell.get("max_steps", 80_000)
    target_cps, schedule_mode = _pick_checkpoints(cell, max_steps)
    if schedule_mode == "fallback":
        print(f"[{label}] cell={cell}")
        print(f"  [WARNING] No tau values given; using fixed-fraction "
              f"checkpoint fallback. Regime labels are placeholders.")
        print(f"  target checkpoints: {target_cps}")
    elif schedule_mode == "partial":
        print(f"[{label}] cell={cell}")
        print("  [INFO] Partial tau schedule: tau_circuit is unavailable, "
              "so labels use known tau_gen/tau_F events only.")
        print(f"  target checkpoints: {target_cps}")
    else:
        print(f"[{label}] cell={cell}; target checkpoints: {target_cps}")

    set_seed(cell["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    prime = cell["prime"]
    task = cell["task"]
    x, y = make_dataset(prime, task, cell["seed"])
    train_x, train_y, test_x, test_y = split_dataset(x, y, train_fraction=0.3)

    model = GrokkingTransformer(
        prime=prime, d_model=256, n_heads=4, n_layers=2, d_ff=1024,
        dropout=0.0,
    ).to(device)

    loader = DataLoader(
        TensorDataset(train_x, train_y),
        batch_size=len(train_x), shuffle=True,
        generator=torch.Generator().manual_seed(cell["seed"]),
    )
    criterion = nn.CrossEntropyLoss()
    optimizer = build_optimizer(
        model, decoder_lr=cell["lr"], embed_lr=EMBED_LR,
        decoder_weight_decay=cell["wd"],
    )
    scheduler = build_scheduler(
        optimizer, warmup_steps=10, lr_schedule="constant",
        lr_min_ratio=0.05, max_steps=max_steps,
    )

    cell_outdir = outdir / label
    cp_outdir = cell_outdir / "checkpoints"
    cp_outdir.mkdir(parents=True, exist_ok=True)

    target_steps_set = set(target_cps.values())
    saved_cps: dict[str, dict] = {}
    train_log = []
    null_rng_emb = np.random.default_rng(cell["seed"] ^ 0xDEAD)
    cur_null_emb = 0.0

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
            save_now = step in target_steps_set

            if log_now or save_now:
                model.eval()
                with torch.no_grad():
                    tr_acc = (model(train_x.to(device)).argmax(1).cpu()
                              == train_y).float().mean().item()
                    te_acc = (model(test_x.to(device)).argmax(1).cpu()
                              == test_y).float().mean().item()
                emb = get_token_embeddings(model, prime)
                if step % 5000 == 0 or step == 1:
                    cur_null_emb = compute_fourier_null_p95(
                        emb, prime, n_perms=100, rng=null_rng_emb,
                    )
                f_emb_raw, _ = compute_fourier_alignment(emb, prime)
                f_corr = max(0.0, f_emb_raw - cur_null_emb)
                train_log.append({
                    "step": step, "train_acc": tr_acc, "test_acc": te_acc,
                    "f_emb_corr": f_corr,
                })

            if save_now:
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
        "used_fallback_taus": schedule_mode == "fallback",
        "checkpoint_schedule": schedule_mode,
        "runtime_sec": time.time() - t0,
    }
    with open(cell_outdir / "meta.json", "w") as f:
        json.dump(meta, f, indent=2)
    print(f"[{label}] done, {len(saved_cps)} checkpoints saved, "
          f"max_test_acc={max(r['test_acc'] for r in train_log):.3f}, "
          f"{meta['runtime_sec']:.0f}s")
    return meta


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--outdir", default="runs/causal_probe_expansion")
    parser.add_argument("--cells", nargs="+",
                        default=list(CELLS_EXPANSION.keys()),
                        choices=list(CELLS_EXPANSION.keys()))
    parser.add_argument("--smoke", action="store_true",
                        help="Print the plan (target steps, taus, fallback "
                             "status) without training.")
    args = parser.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    if args.smoke:
        print("Smoke plan (no training):\n")
        for label in args.cells:
            cell = CELLS_EXPANSION[label]
            max_steps = cell.get("max_steps", 80_000)
            cps, schedule_mode = _pick_checkpoints(cell, max_steps)
            taus = (cell["tau_circuit"], cell["tau_gen"], cell["tau_F"])
            print(f"[{label}] task={cell['task']} prime={cell['prime']} "
                  f"lr={cell['lr']:g} wd={cell['wd']:g} seed={cell['seed']} "
                  f"max_steps={max_steps:,}")
            print(f"  taus = {taus}  schedule={schedule_mode}")
            for regime, step in cps.items():
                print(f"    {regime:25s} step={step:>6}")
            print()
        return

    print(f"E7 expansion: cells={args.cells}")
    print(f"Output: {outdir}\n")

    metas = []
    for label in args.cells:
        cell_meta_path = outdir / label / "meta.json"
        if cell_meta_path.exists():
            print(f"  [skip] {label} already has meta.json; "
                  f"delete to re-run.")
            continue
        meta = run_one(label, CELLS_EXPANSION[label], outdir)
        metas.append(meta)

    summary = {
        "n_cells_run": len(metas),
        "cell_labels": [m["label"] for m in metas],
        "any_fallback": any(m["used_fallback_taus"] for m in metas),
        "total_runtime_sec": sum(m["runtime_sec"] for m in metas),
    }
    with open(outdir / "run_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nAll done: {summary['n_cells_run']} cells in "
          f"{summary['total_runtime_sec']:.0f}s.")
    if summary["any_fallback"]:
        print("\n  NOTE: One or more cells used the fixed-fraction "
              "checkpoint fallback because tau values were not "
              "supplied. Regime labels in these cells are placeholders; "
              "you should fill in detected taus and re-run those cells "
              "before reporting headline numbers.")


if __name__ == "__main__":
    main()
