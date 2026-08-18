#!/usr/bin/env python3
"""
E7 frequency-selectivity control.

The headline E7 result removes the TOP-K Fourier harmonics of the
centered logits (by power). A natural follow-up attack is: "would
removing ANY K Fourier harmonics damage accuracy equally? Maybe the
result isn't about *low-order* Fourier structure, just about removing
K harmonic directions of *any* kind."

This script tests the FREQUENCY-SELECTIVITY claim:

    For each (cell, between-events checkpoint), compare:
      I2_top   = remove the top-K Fourier harmonics (ranked by power)
      I2_alt   = remove K Fourier harmonics drawn from below the
                 top-K_max=10 cutoff (random K-of-(26-K_max))
      I2_bot   = remove the bottom-K Fourier harmonics by power

If top-K removal is dramatically more damaging than alt/bottom-K
removal, the causal effect is genuinely localised to the SPECIFIC
high-power Fourier harmonics, not "any K Fourier directions".

This is a stronger control than the existing matched-energy random
subspace control (I3): I3 draws from the orthogonal complement of
the Fourier+constant subspace (i.e., non-Fourier directions). I2_alt
stays IN the Fourier basis but picks different frequencies.

Output
------
  results/causal_probe/intervention_freq_selectivity.csv
  one row per (cell, regime checkpoint, K), with columns for I2_top,
  I2_alt (averaged over 5 random alt-K draws), I2_bot, and the
  matched I3 random-subspace control for comparison.

Usage
-----
  python src/analysis/causal_fourier_freq_selectivity.py
  python src/analysis/causal_fourier_freq_selectivity.py --K 3 --n-alt-draws 5
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))

from causal_fourier_intervention import (  # noqa: E402
    PRIME,
    _build_fourier_basis_p,
    _identify_topk_harmonics,
    _build_topk_projection,
    _build_random_orthogonal_subspace,
)
from grokking_baseline import (  # noqa: E402
    GrokkingTransformer,
    make_dataset,
    set_seed,
    split_dataset,
)


N_ALT_DRAWS_DEFAULT = 5
N_RANDOM_CTRL_REPS = 5


def _rank_all_harmonics_by_power(L_centered: np.ndarray,
                                 basis_pairs: np.ndarray) -> list[int]:
    """Return harmonic indices (1..(p-1)/2) sorted by descending power
    in the centered logit matrix."""
    half = basis_pairs.shape[0]
    powers = []
    for k in range(half):
        Q = basis_pairs[k].T
        proj = L_centered @ Q
        powers.append(float(np.sum(proj ** 2)))
    return [i + 1 for i in sorted(range(half), key=lambda i: powers[i],
                                  reverse=True)]


def _accuracy(L: np.ndarray, y: np.ndarray) -> float:
    return float((L.argmax(1) == y).mean())


def _remove_harmonics_acc(L: np.ndarray, Lc: np.ndarray, Lc_mean: np.ndarray,
                          harmonics: list[int], basis_pairs: np.ndarray,
                          y_np: np.ndarray, p: int) -> float:
    """Remove the subspace spanned by the listed Fourier harmonics
    (plus constant), forward y_np accuracy."""
    _, P_remove, _ = _build_topk_projection(harmonics, basis_pairs, p)
    L2 = (P_remove @ Lc.T).T + Lc_mean
    return _accuracy(L2, y_np)


def analyse_freq_selectivity(label: str, cell: dict, cp_path: Path,
                             basis_pairs: np.ndarray, device, rng,
                             K: int, K_topcut: int,
                             n_alt_draws: int) -> dict | None:
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

    model.eval()
    with torch.no_grad():
        L = model(test_x_dev).cpu().numpy()
    p = L.shape[1]
    Lc_mean = L.mean(0)
    Lc = L - Lc_mean

    ranked = _rank_all_harmonics_by_power(Lc, basis_pairs)
    half = basis_pairs.shape[0]
    if K_topcut > half:
        K_topcut = half

    base_acc = _accuracy(L, test_y_np)
    if base_acc < 0.5:
        # below-threshold cells don't have a meaningful "load-bearing
        # subspace" yet; record but flag
        pass

    top_K = ranked[:K]
    bot_K = ranked[-K:]

    # Alternative: K harmonics drawn from positions K_topcut..half-K
    # (i.e. neither in the top-K_topcut nor in the bottom-K bucket).
    alt_pool = ranked[K_topcut:-K] if K_topcut < half - K else ranked[K:]
    if len(alt_pool) < K:
        alt_pool = ranked[K:]
    alt_accs = []
    alt_harm_lists = []
    for _ in range(n_alt_draws):
        idx = rng.choice(len(alt_pool), size=K, replace=False)
        alt_K = [alt_pool[i] for i in idx]
        alt_harm_lists.append(",".join(str(h) for h in sorted(alt_K)))
        alt_accs.append(_remove_harmonics_acc(L, Lc, Lc_mean, alt_K,
                                              basis_pairs, test_y_np, p))

    acc_I2_top = _remove_harmonics_acc(L, Lc, Lc_mean, top_K,
                                       basis_pairs, test_y_np, p)
    acc_I2_bot = _remove_harmonics_acc(L, Lc, Lc_mean, bot_K,
                                       basis_pairs, test_y_np, p)
    acc_I2_alt_mean = float(np.mean(alt_accs))
    acc_I2_alt_std = float(np.std(alt_accs))

    # Existing matched-energy random subspace control (I3) for comparison
    _, _, Q_keep = _build_topk_projection(top_K, basis_pairs, p)
    K_dim = 2 * K
    i3_accs = []
    for _ in range(N_RANDOM_CTRL_REPS):
        Q_rand = _build_random_orthogonal_subspace(Q_keep, K_dim, rng, p)
        proj_rand = Q_rand @ (Q_rand.T @ Lc.T)
        L3 = (Lc.T - proj_rand).T + Lc_mean
        i3_accs.append(_accuracy(L3, test_y_np))
    acc_I3_mean = float(np.mean(i3_accs))

    return {
        "label": label,
        "task": task,
        "prime": prime,
        "lr": cell["lr"], "wd": cell["wd"], "seed": cell["seed"],
        "ordering": cell["ordering"],
        "regime": state["regime"],
        "step": state["step"],
        "K": K,
        "K_topcut": K_topcut,
        "base_acc": base_acc,
        "top_K_harmonics": ",".join(str(h) for h in top_K),
        "bot_K_harmonics": ",".join(str(h) for h in bot_K),
        "alt_K_harmonics_examples": "; ".join(alt_harm_lists[:3]),
        "acc_I2_top": acc_I2_top,
        "acc_I2_alt_mean": acc_I2_alt_mean,
        "acc_I2_alt_std": acc_I2_alt_std,
        "acc_I2_bot": acc_I2_bot,
        "acc_I3_random_subspace_mean": acc_I3_mean,
        "i2_top_drop": base_acc - acc_I2_top,
        "i2_alt_drop": base_acc - acc_I2_alt_mean,
        "i2_bot_drop": base_acc - acc_I2_bot,
        "i3_random_subspace_drop": base_acc - acc_I3_mean,
        "selectivity_top_vs_alt":
            ((base_acc - acc_I2_top) / max(base_acc - acc_I2_alt_mean, 1e-6))
            if (base_acc - acc_I2_alt_mean) > 0 else float("inf"),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--probe-root", default="runs/causal_probe")
    parser.add_argument("--outdir", default="results/causal_probe")
    parser.add_argument("--K", type=int, default=3,
                        help="How many harmonics to remove in each variant.")
    parser.add_argument("--K-topcut", type=int, default=10,
                        help="Alt harmonics drawn from ranks below this top-cut.")
    parser.add_argument("--n-alt-draws", type=int, default=N_ALT_DRAWS_DEFAULT)
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

    print(f"E7 frequency-selectivity: K={args.K}, K_topcut={args.K_topcut}, "
          f"n_alt_draws={args.n_alt_draws}, device={device}")

    rows = []
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
                continue
            print(f"[freq-selectivity] {label} regime={regime} "
                  f"step={cp_info['step']}")
            r = analyse_freq_selectivity(label, cell, cp_path, basis_pairs,
                                         device, rng, args.K, args.K_topcut,
                                         args.n_alt_draws)
            if r is None:
                continue
            rows.append(r)
            print(f"  base={r['base_acc']:.3f}  "
                  f"top-{args.K} drop={r['i2_top_drop']:+.3f}  "
                  f"alt-{args.K} drop={r['i2_alt_drop']:+.3f}  "
                  f"bot-{args.K} drop={r['i2_bot_drop']:+.3f}  "
                  f"random ss drop={r['i3_random_subspace_drop']:+.3f}  "
                  f"selectivity top/alt={r['selectivity_top_vs_alt']:.1f}x")

    if not rows:
        print("No checkpoints analysed.")
        return

    out_csv = outdir / "intervention_freq_selectivity.csv"
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"\nWrote {len(rows)} rows -> {out_csv}")

    # Summary: focus on between-events checkpoints with base_acc > 0.5
    bw = {"between_circ_gen", "between_gen_circ", "between_gen_F",
          "between_circ_F"}
    subset = [r for r in rows if r["regime"] in bw and r["base_acc"] > 0.5]
    if subset:
        print(f"\n=== summary on n={len(subset)} between-events checkpoints"
              f" with base_acc>0.5 (K={args.K}) ===")
        med = lambda key: float(np.median([r[key] for r in subset]))
        print(f"  median top-{args.K}  drop : {med('i2_top_drop'):+.3f}")
        print(f"  median alt-{args.K}  drop : {med('i2_alt_drop'):+.3f}  "
              f"(harmonics drawn from rank {args.K_topcut}..)")
        print(f"  median bot-{args.K}  drop : {med('i2_bot_drop'):+.3f}")
        print(f"  median random-ss drop : {med('i3_random_subspace_drop'):+.3f}")
        sel = [r["selectivity_top_vs_alt"] for r in subset
               if r["selectivity_top_vs_alt"] != float("inf")]
        if sel:
            print(f"  median selectivity (top / alt) : {float(np.median(sel)):.1f}x")


if __name__ == "__main__":
    main()
