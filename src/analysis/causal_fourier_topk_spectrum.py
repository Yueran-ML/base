#!/usr/bin/env python3
"""
E7 top-K Fourier ablation SPECTRUM.

This script is a wrapper around ``causal_fourier_intervention.py`` that runs
the same I1 / I2 / I3 / I4 interventions for several K values
(K in {1, 3, 5, 10} by default) on the same probe checkpoints.

Purpose
-------
The headline E7 result uses K = 3 (the dominant top-3 logit Fourier
harmonics). A reviewer can reasonably push: "is K = 3 ad hoc?" This
spectrum sweep is a *defensive* check, not a claim of K = 3 being
uniquely privileged. We aim to verify:

    The causal effect is localized to low-order logit-Fourier
    structure, rather than to arbitrary low-rank damage.

Concretely we want to see:
  - small K (1, 3, 5): I2 drop strong, I3 random-control drop negligible,
                       specificity ratio large
  - large K (10)     : I2 drop saturates (most of the structure removed);
                       I3 random-control drop grows as the removed energy
                       grows; specificity ratio compresses but I2 still
                       dominates I3

The matched-energy random control (I3) automatically scales with K via
the existing code (Q_rand has dimension 2K matching the top-K Fourier
subspace), so the comparison stays methodologically apples-to-apples
within each K.

Output
------
  results/causal_probe/intervention_topk_spectrum.csv
    one row per (cell, regime checkpoint, K)

Usage
-----
  python src/analysis/causal_fourier_topk_spectrum.py
  python src/analysis/causal_fourier_topk_spectrum.py --top-k-values 1 3 5 10
  python src/analysis/causal_fourier_topk_spectrum.py --probe-root runs/causal_probe
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import torch

# Use the helpers from the existing intervention module.
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))           # for grokking_baseline
sys.path.insert(0, str(HERE))                  # for sibling import

from causal_fourier_intervention import (  # noqa: E402
    PRIME,
    N_RANDOM_CONTROL_REPS,
    _build_fourier_basis_p,
    _identify_topk_harmonics,
    _evaluate_with_logit_intervention,
    _evaluate_with_embedding_intervention,
)
from grokking_baseline import (  # noqa: E402
    GrokkingTransformer,
    make_dataset,
    set_seed,
    split_dataset,
)


DEFAULT_TOP_K_VALUES = [1, 3, 5, 10]


def analyse_checkpoint_topk(label: str, cell: dict, cp_meta: dict,
                            cp_path: Path, basis_pairs: np.ndarray,
                            device, rng, top_k_values: list[int]) -> list[dict]:
    """For one (cell, checkpoint), run the intervention for each K in
    top_k_values and return one row per K."""
    set_seed(cell["seed"])
    prime = int(cell.get("prime", PRIME))
    task = cell.get("task", "add")
    x, y = make_dataset(prime, task, cell["seed"])
    _, _, test_x, test_y = split_dataset(x, y, train_fraction=0.3)

    model = GrokkingTransformer(
        prime=prime, d_model=256, n_heads=4, n_layers=2, d_ff=1024,
        dropout=0.0,
    ).to(device)
    state = torch.load(cp_path, map_location=device)
    model.load_state_dict(state["model_state"])

    test_x_dev = test_x.to(device)
    test_y_np = test_y.numpy()

    # Identify top-K_max harmonics ONCE, then take prefixes for each K.
    # (Prevents K=1 from picking a different harmonic than K=3 does.)
    K_max = max(top_k_values)
    model.eval()
    with torch.no_grad():
        L_init = model(test_x_dev).cpu().numpy()
    L_init_c = L_init - L_init.mean(0)
    ranked_harmonics = _identify_topk_harmonics(L_init_c, basis_pairs,
                                                top_k=K_max)

    rows = []
    for K in top_k_values:
        top_k = ranked_harmonics[:K]
        logit_results = _evaluate_with_logit_intervention(
            model, test_x_dev, test_y_np, basis_pairs, top_k, device, rng)
        acc_I4 = _evaluate_with_embedding_intervention(
            model, test_x_dev, test_y_np, basis_pairs, top_k, device, prime)

        # Derived diagnostics
        i2_drop = logit_results["base_acc"] - logit_results["acc_I2_remove_fourier"]
        i3_drop = logit_results["base_acc"] - logit_results["acc_I3_random_ctrl_mean"]
        i4_drop = logit_results["base_acc"] - acc_I4
        specificity_ratio = (i2_drop / i3_drop) if abs(i3_drop) > 1e-6 else float("inf")

        rows.append({
            "label": label,
            "task": task, "prime": prime,
            "lr": cell["lr"], "wd": cell["wd"], "seed": cell["seed"],
            "ordering": cell["ordering"],
            "tau_circ": cell["tau_circuit"],
            "tau_gen": cell["tau_gen"],
            "tau_F": cell["tau_F"],
            "regime": state["regime"],
            "step": state["step"],
            "saved_test_acc": state["test_acc"],
            "K": K,
            **logit_results,
            "acc_I4_remove_emb_fourier": acc_I4,
            "i2_drop": i2_drop,
            "i3_random_drop_mean": i3_drop,
            "i4_drop": i4_drop,
            "specificity_ratio_i2_over_i3": specificity_ratio,
        })

    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--probe-root", default="runs/causal_probe")
    parser.add_argument("--outdir", default="results/causal_probe")
    parser.add_argument("--top-k-values", type=int, nargs="+",
                        default=DEFAULT_TOP_K_VALUES,
                        help="Spectrum of K values to evaluate.")
    args = parser.parse_args()

    probe_root = Path(args.probe_root)
    outdir = Path(args.outdir); outdir.mkdir(parents=True, exist_ok=True)

    if not probe_root.exists():
        print(f"  MISSING: {probe_root}")
        print("  Run: python src/sweeps/causal_probe_checkpoints.py first.")
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    basis_cache: dict[int, np.ndarray] = {}
    rng = np.random.default_rng(20260522)

    print(f"E7 top-K spectrum: K in {args.top_k_values}, device={device}")
    print(f"Probe root: {probe_root}\n")

    all_rows = []
    for cell_dir in sorted(probe_root.iterdir()):
        if not cell_dir.is_dir():
            continue
        meta_path = cell_dir / "meta.json"
        if not meta_path.exists():
            continue
        meta = json.loads(meta_path.read_text())
        label = meta["label"]
        cell = meta["cell"]
        prime = int(cell.get("prime", PRIME))
        if prime not in basis_cache:
            basis_cache[prime] = _build_fourier_basis_p(prime)
        basis_pairs = basis_cache[prime]
        for regime, cp_info in meta["saved_checkpoints"].items():
            cp_path = probe_root / cp_info["path"]
            if not cp_path.exists():
                print(f"  [skip] missing checkpoint {cp_path}")
                continue
            print(f"[topk-spectrum] {label} regime={regime} step={cp_info['step']}")
            rows = analyse_checkpoint_topk(
                label, cell, cp_info, cp_path,
                basis_pairs, device, rng, args.top_k_values)
            all_rows.extend(rows)
            for r in rows:
                print(f"  K={r['K']:>2}  "
                      f"base={r['base_acc']:.3f}  "
                      f"I2_drop={r['i2_drop']:+.3f}  "
                      f"I3_ctrl_drop={r['i3_random_drop_mean']:+.3f}  "
                      f"specificity={r['specificity_ratio_i2_over_i3']:.1f}x  "
                      f"I4_emb_drop={r['i4_drop']:+.3f}")

    if not all_rows:
        print("No checkpoints analysed.")
        return

    out_csv = outdir / "intervention_topk_spectrum.csv"
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(all_rows[0].keys()))
        w.writeheader()
        w.writerows(all_rows)
    print(f"\nWrote {len(all_rows)} rows -> {out_csv}")
    print(f"  (= {len(all_rows) // len(args.top_k_values)} (cell, regime) checkpoints"
          f" x {len(args.top_k_values)} K values)")

    # Quick pivot summary on stdout: median I2 drop and specificity per K
    print("\n=== summary per K (median across all (cell, regime) checkpoints) ===")
    print(f"{'K':>3} | {'med I2 drop':>12} | {'med I3 ctrl drop':>16} | "
          f"{'med specificity':>15}")
    print("-" * 60)
    for K in sorted(set(r["K"] for r in all_rows)):
        bucket = [r for r in all_rows if r["K"] == K]
        med_i2 = float(np.median([r["i2_drop"] for r in bucket]))
        med_i3 = float(np.median([r["i3_random_drop_mean"] for r in bucket]))
        med_sp = float(np.median([r["specificity_ratio_i2_over_i3"]
                                  for r in bucket
                                  if r["specificity_ratio_i2_over_i3"] != float("inf")]))
        print(f"{K:>3} | {med_i2:>+12.3f} | {med_i3:>+16.3f} | {med_sp:>14.1f}x")


if __name__ == "__main__":
    main()
