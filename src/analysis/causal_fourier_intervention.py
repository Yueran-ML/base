#!/usr/bin/env python3
"""
Causal Fourier-subspace intervention analysis on saved checkpoints.

For every (cell, checkpoint) saved by causal_probe_checkpoints.py, we apply
four interventions and measure resulting test accuracy:

  I1 (logit_keep_topK)   : project logit (p,) onto the top-K Fourier
                            harmonics of the answer-class index s = (a+b) mod p,
                            zero out the rest. Keeps only the dominant
                            cyclic-group structure.
  I2 (logit_remove_topK) : the orthogonal complement of I1. Removes the
                            dominant Fourier subspace; if the model's circuit
                            is implemented through these harmonics, this
                            should tank accuracy.
  I3 (logit_random_ctrl) : a matched-energy random control. Removes a
                            random orthonormal subspace of dimension 2K from
                            the centered logit space, with the same Frobenius
                            energy as the top-K Fourier removal would have
                            removed. Provides a baseline for I2.
  I4 (embed_remove_topK) : project token embedding rows onto the orthogonal
                            complement of the top-K Fourier subspace
                            (basis indexed by token), then forward.

Top-K is chosen at the LOGIT level: at each checkpoint we identify the K
harmonics k in {1, ..., (p-1)/2} that explain the most centered-logit
variance, with K=3 (matches the dynamic top-K used by F_logit).

The "matched energy" I3 control draws a random 2K-dim orthonormal basis from
the (p-2K)-dim subspace orthogonal to the union of all Fourier basis pairs.
We then project the centered logit onto that random subspace and remove an
energy-matched component. This isolates whether the accuracy drop in I2 is
specific to Fourier structure rather than just generic energy removal.

Output:
  results/causal_probe/intervention_results.csv

Usage:
  python src/analysis/causal_fourier_intervention.py
  python src/analysis/causal_fourier_intervention.py --probe-root runs/causal_probe
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from grokking_baseline import (  # noqa: E402
    GrokkingTransformer,
    make_dataset,
    set_seed,
    split_dataset,
)


PRIME = 53
TOP_K = 3
N_RANDOM_CONTROL_REPS = 5  # average over 5 random subspace draws for I3


def _build_fourier_basis_p(p: int = PRIME):
    """Real-valued Fourier basis over indices 0..p-1.

    Returns:
        basis_pairs : np.ndarray of shape ((p-1)//2, 2, p)
                      basis_pairs[k-1, 0] = cos(2*pi*k*j/p) / norm
                      basis_pairs[k-1, 1] = sin(2*pi*k*j/p) / norm
    """
    half = (p - 1) // 2
    j = np.arange(p, dtype=np.float64)
    basis = np.zeros((half, 2, p), dtype=np.float64)
    for k in range(1, half + 1):
        c = np.cos(2 * np.pi * k * j / p)
        s = np.sin(2 * np.pi * k * j / p)
        c -= c.mean(); s -= s.mean()
        c /= np.linalg.norm(c) + 1e-12
        s /= np.linalg.norm(s) + 1e-12
        # Gram-Schmidt
        s -= s @ c * c
        s /= np.linalg.norm(s) + 1e-12
        basis[k - 1, 0] = c
        basis[k - 1, 1] = s
    return basis


def _identify_topk_harmonics(L_centered: np.ndarray, basis_pairs: np.ndarray,
                             top_k: int = TOP_K) -> list[int]:
    """Identify the K Fourier harmonics with most variance in the centered
    logit matrix L_centered (n_pairs, p)."""
    half = basis_pairs.shape[0]
    powers = []
    for k in range(half):
        Q = basis_pairs[k].T  # (p, 2)
        proj = L_centered @ Q  # (n_pairs, 2)
        powers.append(float(np.sum(proj ** 2)))
    top_idx = sorted(range(half), key=lambda i: powers[i], reverse=True)[:top_k]
    return [i + 1 for i in top_idx]  # return 1-indexed harmonic numbers


def _build_topk_projection(top_k_harmonics: list[int],
                           basis_pairs: np.ndarray, p: int | None = None):
    """Return (P_keep, P_remove) where each is a (p, p) projection matrix.

    P_keep projects R^p onto the subspace spanned by the 2K Fourier basis
    vectors for the chosen harmonics, plus the constant direction.
    P_remove = I - P_keep.
    """
    if p is None:
        p = basis_pairs.shape[-1]
    Q_cols = [np.ones(p) / np.sqrt(p)]  # constant direction
    for k in top_k_harmonics:
        Q_cols.append(basis_pairs[k - 1, 0])
        Q_cols.append(basis_pairs[k - 1, 1])
    Q = np.stack(Q_cols, axis=1)  # (p, 1+2K)
    Q, _ = np.linalg.qr(Q)
    P_keep = Q @ Q.T
    P_remove = np.eye(p) - P_keep
    return P_keep, P_remove, Q


def _build_random_orthogonal_subspace(Q_fourier: np.ndarray, dim: int,
                                      rng: np.random.Generator,
                                      p: int | None = None):
    """Build a random 'dim'-dimensional orthonormal subspace within the
    orthogonal complement of Q_fourier (which is (1+2K) dim).

    Returns Q_rand of shape (p, dim).
    """
    if p is None:
        p = Q_fourier.shape[0]
    P_orth = np.eye(p) - Q_fourier @ Q_fourier.T
    M = rng.standard_normal((p, dim))
    M = P_orth @ M
    Q_rand, _ = np.linalg.qr(M)
    return Q_rand[:, :dim]


def _accuracy_from_logits(logits: np.ndarray, y: np.ndarray) -> float:
    return float((logits.argmax(1) == y).mean())


def _evaluate_with_logit_intervention(model, test_x_dev, test_y_np,
                                      basis_pairs, top_k_harmonics,
                                      device, rng):
    """Run model on test_x and apply 4 logit-space interventions."""
    model.eval()
    with torch.no_grad():
        L = model(test_x_dev).cpu().numpy()  # (n_test, p)
    p = L.shape[1]
    Lc_mean = L.mean(0)
    Lc = L - Lc_mean  # column-centered

    # Identify top-K harmonics from centered logits
    K_dim = 2 * len(top_k_harmonics)  # 2 basis vectors per harmonic
    P_keep, P_remove, Q_keep = _build_topk_projection(
        top_k_harmonics, basis_pairs, p)

    base_acc = _accuracy_from_logits(L, test_y_np)

    # I1: keep only Fourier subspace
    L1 = (P_keep @ Lc.T).T + Lc_mean
    acc_I1 = _accuracy_from_logits(L1, test_y_np)

    # I2: remove Fourier subspace (orthogonal complement)
    L2 = (P_remove @ Lc.T).T + Lc_mean
    acc_I2 = _accuracy_from_logits(L2, test_y_np)

    # Energy of the Fourier-subspace component (relative)
    energy_keep = float(np.sum((P_keep @ Lc.T) ** 2))
    energy_total = float(np.sum(Lc ** 2)) + 1e-12
    frac_keep = energy_keep / energy_total

    # I3: matched-energy random subspace control (average over reps)
    accs_I3 = []
    for _ in range(N_RANDOM_CONTROL_REPS):
        Q_rand = _build_random_orthogonal_subspace(Q_keep, K_dim, rng, p)
        # Project centered logits onto Q_rand and remove that component
        proj_rand = Q_rand @ (Q_rand.T @ Lc.T)  # (p, n_test)
        L3 = (Lc.T - proj_rand).T + Lc_mean
        accs_I3.append(_accuracy_from_logits(L3, test_y_np))
    acc_I3_mean = float(np.mean(accs_I3))
    acc_I3_std = float(np.std(accs_I3))

    return {
        "base_acc": base_acc,
        "acc_I1_keep_fourier": acc_I1,
        "acc_I2_remove_fourier": acc_I2,
        "acc_I3_random_ctrl_mean": acc_I3_mean,
        "acc_I3_random_ctrl_std": acc_I3_std,
        "frac_logit_energy_in_topK_fourier": frac_keep,
        "top_k_harmonics": ",".join(str(k) for k in top_k_harmonics),
    }


def _evaluate_with_embedding_intervention(model, test_x_dev, test_y_np,
                                          basis_pairs, top_k_harmonics,
                                          device, p: int | None = None):
    """I4: remove top-K Fourier components from token-embedding rows, then
    forward."""
    model.eval()
    if p is None:
        p = basis_pairs.shape[-1]
    P_keep, P_remove, _ = _build_topk_projection(
        top_k_harmonics, basis_pairs, p)
    P_remove_t = torch.from_numpy(P_remove).float().to(device)

    # Save original embedding weights
    orig_weight = model.token_embed.weight.data.clone()
    try:
        E = orig_weight.clone()  # (p, d_model)
        # Project rows: indexed by token (the first axis is over p tokens)
        E_modified = P_remove_t @ E
        model.token_embed.weight.data.copy_(E_modified)
        with torch.no_grad():
            L = model(test_x_dev).cpu().numpy()
    finally:
        model.token_embed.weight.data.copy_(orig_weight)

    return _accuracy_from_logits(L, test_y_np)


def analyse_checkpoint(label: str, cell: dict, cp_meta: dict,
                       cp_path: Path, basis_pairs: np.ndarray,
                       device, rng) -> dict:
    set_seed(cell["seed"])
    prime = int(cell.get("prime", PRIME))
    task = cell.get("task", "add")
    x, y = make_dataset(prime, task, cell["seed"])
    train_x, train_y, test_x, test_y = split_dataset(x, y, train_fraction=0.3)

    model = GrokkingTransformer(
        prime=prime, d_model=256, n_heads=4, n_layers=2, d_ff=1024,
        dropout=0.0,
    ).to(device)
    state = torch.load(cp_path, map_location=device)
    model.load_state_dict(state["model_state"])

    test_x_dev = test_x.to(device)
    test_y_np = test_y.numpy()

    # Compute centered logits to identify top-K harmonics
    model.eval()
    with torch.no_grad():
        L_init = model(test_x_dev).cpu().numpy()
    L_init_c = L_init - L_init.mean(0)
    top_k = _identify_topk_harmonics(L_init_c, basis_pairs)

    logit_results = _evaluate_with_logit_intervention(
        model, test_x_dev, test_y_np, basis_pairs, top_k, device, rng)
    acc_I4 = _evaluate_with_embedding_intervention(
        model, test_x_dev, test_y_np, basis_pairs, top_k, device, prime)

    return {
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
        **logit_results,
        "acc_I4_remove_emb_fourier": acc_I4,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--probe-root", default="runs/causal_probe")
    parser.add_argument("--outdir", default="results/causal_probe")
    args = parser.parse_args()

    probe_root = Path(args.probe_root)
    outdir = Path(args.outdir); outdir.mkdir(parents=True, exist_ok=True)

    if not probe_root.exists():
        print(f"  MISSING: {probe_root}")
        print("  Run: python src/sweeps/causal_probe_checkpoints.py first.")
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    basis_cache: dict[int, np.ndarray] = {}
    rng = np.random.default_rng(20240509)

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
                print(f"  [skip] missing checkpoint {cp_path}")
                continue
            print(f"[analyse] {label} regime={regime} step={cp_info['step']}")
            r = analyse_checkpoint(label, cell, cp_info, cp_path,
                                   basis_pairs, device, rng)
            rows.append(r)
            print(f"  base={r['base_acc']:.3f} "
                  f"I1_keep={r['acc_I1_keep_fourier']:.3f} "
                  f"I2_remove={r['acc_I2_remove_fourier']:.3f} "
                  f"I3_ctrl={r['acc_I3_random_ctrl_mean']:.3f}"
                  f"+/-{r['acc_I3_random_ctrl_std']:.3f} "
                  f"I4_emb_remove={r['acc_I4_remove_emb_fourier']:.3f}")

    if not rows:
        print("No checkpoints analysed.")
        return

    out_csv = outdir / "intervention_results.csv"
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    print(f"\nWrote {len(rows)} rows -> {out_csv}")


if __name__ == "__main__":
    main()
