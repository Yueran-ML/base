#!/usr/bin/env python3
"""
Label-free, component-level Fourier activation patching on saved checkpoints.

Motivation (F4 in the paper's Future Work): the E7 interventions act on
output logits with a basis indexed by the ground-truth answer class, and on
embedding rows with a basis indexed by the token value. They establish
functional necessity of those subspaces but do not localize WHERE inside the
network the answer-aligned structure is computed. This probe ablates
Fourier-frequency content of INTERNAL activations, with no use of labels
anywhere in the construction:

  For a hook site with activation A of shape (p^2, 2, d) over the full input
  grid (a, b) in Z_p x Z_p, we view each (position, channel) pair as a scalar
  function on the grid, take its 2D DFT over (a, b), and define "harmonic k"
  as the set of 2D frequencies (ka, kb) with ka in {k, p-k} or kb in
  {k, p-k}. This includes the product terms cos/sin(2*pi*k*(a+-b)/p) that
  the grokked circuit uses (Nanda et al., 2023), but is constructed purely
  from the inputs.

Interventions per (checkpoint, site):
  ablate  : zero the grid-Fourier coefficients of the top-K harmonics
            (selected by activation power at that site, label-free),
            inverse-DFT, and replace the site's activation.
  random  : matched-energy control. Remove the component of each grid
            function along a random orthonormal subspace of the same
            dimension (drawn orthogonal to the ablated Fourier subspace),
            scaled so the total removed energy matches the Fourier ablation.
            Averaged over N_RANDOM_CONTROL_REPS draws.

Hook sites:
  embed      : token_embed output (per-token embedding, both positions)
  attn_L{i}  : encoder.layers[i].self_attn output
  mlp_L{i}   : encoder.layers[i].linear2 output (MLP-block output)
  resid_final: encoder output (final residual stream)

Implementation is two-pass per intervention: pass 1 records the site's
activation over the full p^2 input grid; the patched tensor is computed
offline; pass 2 replaces the site's output with the patched tensor. The
model runs in eval mode with dropout 0, so the replacement is exact.

Scope note mirroring E7: the patch basis is computed from the activation
over the full input grid, and the same fixed linear projection is then
applied to every sample; no ground-truth label enters the construction.

Output:
  results/activation_patching/activation_patching_results.csv

Usage:
  python src/analysis/activation_patching_probe.py
  python src/analysis/activation_patching_probe.py \
      --probe-roots runs/causal_probe runs/causal_probe_expansion \
      --top-k 3 --smoke
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

N_RANDOM_CONTROL_REPS = 5


# ------------------------------------------------------------------
# Grid-Fourier machinery (label-free: indexed by inputs (a, b) only)
# ------------------------------------------------------------------

def _harmonic_mask(p: int, harmonics: list[int]) -> np.ndarray:
    """Boolean mask over 2D DFT frequencies (ka, kb) for the given
    harmonics: True where ka in {k, p-k} or kb in {k, p-k} for some k.
    The DC row/column (0) is never masked."""
    mask = np.zeros((p, p), dtype=bool)
    for k in harmonics:
        rows = {k % p, (-k) % p} - {0}
        for r in rows:
            mask[r, :] = True
            mask[:, r] = True
    return mask


def _per_harmonic_power(G_hat: np.ndarray, p: int) -> np.ndarray:
    """Power attributable to each harmonic k in 1..(p-1)//2.

    G_hat: complex DFT coefficients of shape (p, p, C).
    A coefficient (ka, kb) contributes to harmonic k if ka in {k, p-k}
    or kb in {k, p-k}; coefficients touching two harmonics contribute
    to both (selection heuristic only; the ablation itself uses the
    exact union mask, so double counting cannot bias the intervention).
    """
    half = (p - 1) // 2
    coeff_pow = np.abs(G_hat) ** 2  # (p, p, C)
    coeff_pow = coeff_pow.sum(axis=2)  # (p, p)
    powers = np.zeros(half)
    for k in range(1, half + 1):
        m = _harmonic_mask(p, [k])
        powers[k - 1] = coeff_pow[m].sum()
    return powers


def _ablate_harmonics(A: np.ndarray, p: int, harmonics: list[int]):
    """Zero the grid-Fourier content of the given harmonics.

    A: (p*p, C) activation rows ordered so that row index = a*p + b.
    Returns (A_ablated, removed, frac_energy_removed) where removed is the
    (p*p, C) component taken out.
    """
    C = A.shape[1]
    G = A.reshape(p, p, C)
    G_hat = np.fft.fft2(G, axes=(0, 1))
    mask = _harmonic_mask(p, harmonics)
    G_hat_ablated = G_hat.copy()
    G_hat_ablated[mask, :] = 0.0
    G_ablated = np.real(np.fft.ifft2(G_hat_ablated, axes=(0, 1)))
    A_ablated = G_ablated.reshape(p * p, C)
    removed = A - A_ablated
    total = float(np.sum((A - A.mean(0)) ** 2)) + 1e-12
    frac = float(np.sum(removed ** 2)) / total
    return A_ablated, removed, frac


def _fourier_subspace_basis(p: int, harmonics: list[int],
                            rng: np.random.Generator) -> np.ndarray:
    """Orthonormal basis (p*p, m) of the ablated grid-Fourier subspace.

    The harmonic mask is conjugate-symmetric, so the real dimension equals
    the number of masked coefficients. We obtain the basis by projecting
    random probes onto the masked subspace and orthonormalising.
    """
    n = p * p
    mask = _harmonic_mask(p, harmonics)
    m = int(mask.sum())
    R = rng.standard_normal((n, m))
    G_hat = np.fft.fft2(R.reshape(p, p, m), axes=(0, 1))
    keep = np.zeros_like(G_hat)
    keep[mask, :] = G_hat[mask, :]
    P = np.real(np.fft.ifft2(keep, axes=(0, 1))).reshape(n, m)
    Q, _ = np.linalg.qr(P)
    return Q[:, :m]


def _onmanifold_control(A: np.ndarray, p: int, harmonics: list[int],
                        rng: np.random.Generator,
                        energy_keep: float = 0.99) -> np.ndarray:
    """Random control drawn inside the span the activations actually occupy.

    A grid-space random direction is a poor control at sites whose
    activation is a (partially) separable function of a single input token:
    it moves the activation off the low-dimensional manifold the network
    ever produces, so the accuracy drop measures manifold departure rather
    than Fourier specificity. Here we restrict the random subspace to the
    principal subspace of the observed activations, and match its dimension
    to the ablated Fourier subspace *as seen from that same span*, so the
    control removes an equally large slice of live activation variance.
    """
    n, _ = A.shape
    Ac = A - A.mean(0, keepdims=True)
    U, S, _ = np.linalg.svd(Ac, full_matrices=False)
    energy = np.cumsum(S ** 2) / (float(np.sum(S ** 2)) + 1e-12)
    r = int(np.searchsorted(energy, energy_keep) + 1)
    U = U[:, :r]

    QF = _fourier_subspace_basis(p, harmonics, rng)
    PU_QF = U @ (U.T @ QF)
    Uf, sf, _ = np.linalg.svd(PU_QF, full_matrices=False)
    m_eff = int((sf > 1e-8).sum())
    if m_eff == 0:
        return A.copy()
    QF_in_U = Uf[:, :m_eff]

    M = U @ rng.standard_normal((r, m_eff))
    M = M - QF_in_U @ (QF_in_U.T @ M)
    Q, _ = np.linalg.qr(M)
    Q = Q[:, :m_eff]
    return A - Q @ (Q.T @ A)


def _random_matched_control(A: np.ndarray, removed: np.ndarray, p: int,
                            harmonics: list[int],
                            rng: np.random.Generator,
                            ctrl_mode: str = "dim") -> np.ndarray:
    """Random-subspace control in grid-function space.

    Draws a random orthonormal subspace of R^{p^2} with the same dimension
    as the ablated Fourier subspace, orthogonal to it, and removes the
    component of each channel along that subspace.

    ctrl_mode:
      "dim"         : dimension-matched in grid space, remove the natural
                      component only (mirrors E7's I3 control).
      "energy"      : additionally rescale so removed energy matches the
                      Fourier ablation. Stricter, but pushes activations
                      off-manifold at sites with separable structure.
      "onmanifold"  : handled by _onmanifold_control (see above).
    """
    if ctrl_mode == "onmanifold":
        return _onmanifold_control(A, p, harmonics, rng)
    n = p * p
    mask = _harmonic_mask(p, harmonics)
    m_dim = int(mask.sum())  # real dimension of the ablated subspace

    # Random directions, then project out the ablated Fourier subspace by
    # zeroing their own masked DFT coefficients (exact orthogonality).
    M = rng.standard_normal((n, m_dim))
    G = M.reshape(p, p, m_dim)
    G_hat = np.fft.fft2(G, axes=(0, 1))
    G_hat[mask, :] = 0.0
    M = np.real(np.fft.ifft2(G_hat, axes=(0, 1))).reshape(n, m_dim)
    Q, _ = np.linalg.qr(M)

    comp = Q @ (Q.T @ A)  # (n, C) component along random subspace
    if ctrl_mode == "energy":
        e_comp = float(np.sum(comp ** 2)) + 1e-12
        e_target = float(np.sum(removed ** 2))
        comp = np.sqrt(e_target / e_comp) * comp
    return A - comp


# ------------------------------------------------------------------
# Hook plumbing
# ------------------------------------------------------------------

def _site_modules(model) -> dict[str, torch.nn.Module]:
    sites = {"embed": model.token_embed}
    for i, layer in enumerate(model.encoder.layers):
        sites[f"attn_L{i}"] = layer.self_attn
        sites[f"mlp_L{i}"] = layer.linear2
    sites["resid_final"] = model.encoder
    return sites


def _extract_tensor(out):
    """MultiheadAttention returns (attn_output, weights); others a tensor."""
    return out[0] if isinstance(out, tuple) else out


def _capture_activation(model, site_mod, grid_x_dev) -> np.ndarray:
    """Pass 1: record the site's output over the full input grid.

    Returns (p*p, 2*d) with the two sequence positions flattened into
    channels.
    """
    store = {}

    def hook(mod, inp, out):
        store["act"] = _extract_tensor(out).detach()

    h = site_mod.register_forward_hook(hook)
    try:
        with torch.no_grad():
            model(grid_x_dev)
    finally:
        h.remove()
    act = store["act"]  # (n, 2, d)
    n = act.shape[0]
    return act.reshape(n, -1).cpu().numpy().astype(np.float64)


def _forward_with_replacement(model, site_mod, grid_x_dev,
                              patched: np.ndarray) -> np.ndarray:
    """Pass 2: replace the site's output with the patched tensor and
    return logits over the full grid."""
    device = grid_x_dev.device
    n = grid_x_dev.shape[0]
    patched_t = torch.from_numpy(
        patched.reshape(n, 2, -1)).float().to(device)

    def hook(mod, inp, out):
        if isinstance(out, tuple):
            return (patched_t, *out[1:])
        return patched_t

    h = site_mod.register_forward_hook(hook)
    try:
        with torch.no_grad():
            logits = model(grid_x_dev).cpu().numpy()
    finally:
        h.remove()
    return logits


# ------------------------------------------------------------------
# Per-checkpoint analysis
# ------------------------------------------------------------------

def analyse_checkpoint(label: str, cell: dict, cp_path: Path,
                       state: dict, device, rng,
                       top_k: int, ctrl_mode: str = "dim") -> list[dict]:
    set_seed(cell["seed"])
    prime = int(cell.get("prime", 53))
    task = cell.get("task", "add")

    # Full input grid in row order a*p + b.
    aa, bb = np.meshgrid(np.arange(prime), np.arange(prime), indexing="ij")
    grid_x = torch.from_numpy(
        np.stack([aa.ravel(), bb.ravel()], axis=1)).long()

    # Labels and test-split membership come from make_dataset so any
    # task-specific label convention (e.g. discrete-log reindexing for
    # mul) is inherited rather than re-derived; numbers stay comparable
    # with E7.
    x, y = make_dataset(prime, task, cell["seed"])
    x_np, y_all = x.numpy(), y.numpy()
    label_of = {int(r[0]) * prime + int(r[1]): int(lab)
                for r, lab in zip(x_np, y_all)}
    y_np = np.array([label_of[i] for i in range(prime * prime)],
                    dtype=np.int64)
    _, _, test_x, test_y = split_dataset(x, y, train_fraction=0.3)
    test_keys = set((int(r[0]) * prime + int(r[1])) for r in test_x.numpy())
    test_idx = np.array(
        [i for i in range(prime * prime) if i in test_keys], dtype=np.int64)

    model = GrokkingTransformer(
        prime=prime, d_model=256, n_heads=4, n_layers=2, d_ff=1024,
        dropout=0.0,
    ).to(device)
    model.load_state_dict(state["model_state"])
    model.eval()

    grid_x_dev = grid_x.to(device)

    with torch.no_grad():
        base_logits = model(grid_x_dev).cpu().numpy()
    base_acc = float(
        (base_logits[test_idx].argmax(1) == y_np[test_idx]).mean())

    ctrl_modes = [m.strip() for m in ctrl_mode.split(",") if m.strip()]

    rows = []
    for site_name, site_mod in _site_modules(model).items():
        A = _capture_activation(model, site_mod, grid_x_dev)  # (p^2, 2d)

        # Label-free top-K harmonic ranking by activation power.
        G_hat = np.fft.fft2(
            A.reshape(prime, prime, -1), axes=(0, 1))
        powers = _per_harmonic_power(G_hat, prime)
        ranked = (np.argsort(powers)[::-1] + 1).tolist()

        for k_val in top_k:
            harmonics = [int(h) for h in ranked[:k_val]]

            A_ablated, removed, frac = _ablate_harmonics(
                A, prime, harmonics)
            logits_ab = _forward_with_replacement(
                model, site_mod, grid_x_dev, A_ablated)
            acc_ablate = float(
                (logits_ab[test_idx].argmax(1) == y_np[test_idx]).mean())

            for mode in ctrl_modes:
                ctrl_accs = []
                for _ in range(N_RANDOM_CONTROL_REPS):
                    A_ctrl = _random_matched_control(
                        A, removed, prime, harmonics, rng, mode)
                    logits_ct = _forward_with_replacement(
                        model, site_mod, grid_x_dev, A_ctrl)
                    ctrl_accs.append(float(
                        (logits_ct[test_idx].argmax(1)
                         == y_np[test_idx]).mean()))
                rows.append(_make_row(
                    label, task, prime, cell, state, site_name, k_val,
                    mode, harmonics, base_acc, acc_ablate, ctrl_accs, frac))
                r = rows[-1]
                print(f"    {site_name:12s} K={k_val:<2d} {mode:11s} "
                      f"base={base_acc:.3f} ablate={acc_ablate:.3f} "
                      f"ctrl={r['acc_random_ctrl_mean']:.3f}"
                      f"+/-{r['acc_random_ctrl_std']:.3f}")
    return rows


def _make_row(label, task, prime, cell, state, site_name, k_val, mode,
              harmonics, base_acc, acc_ablate, ctrl_accs, frac) -> dict:
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
            "site": site_name,
            "K": k_val,
            "ctrl_mode": mode,
            "top_harmonics": ",".join(str(k) for k in harmonics),
            "base_acc": base_acc,
            "acc_ablate_fourier": acc_ablate,
            "acc_random_ctrl_mean": float(np.mean(ctrl_accs)),
            "acc_random_ctrl_std": float(np.std(ctrl_accs)),
            "frac_energy_removed": frac,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--probe-roots", nargs="+",
                        default=["runs/causal_probe",
                                 "runs/causal_probe_expansion"])
    parser.add_argument("--outdir", default="results/activation_patching")
    parser.add_argument("--top-k", type=int, nargs="+", default=[3],
                        help="one or more K values; each is run separately")
    parser.add_argument("--ctrl-mode", default="dim",
                        help="comma-separated list from {dim, energy, "
                             "onmanifold}. dim mirrors E7's I3; onmanifold "
                             "draws the random subspace inside the span the "
                             "activations actually occupy, which is the "
                             "meaningful control at sites whose activation "
                             "is a separable function of one input token")
    parser.add_argument("--smoke", action="store_true",
                        help="analyse only the first checkpoint found")
    args = parser.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rng = np.random.default_rng(20260717)

    rows = []
    done = False
    for root in args.probe_roots:
        probe_root = Path(root)
        if not probe_root.exists():
            print(f"  [skip root] {probe_root} does not exist")
            continue
        for cell_dir in sorted(probe_root.iterdir()):
            if done or not cell_dir.is_dir():
                continue
            meta_path = cell_dir / "meta.json"
            if not meta_path.exists():
                continue
            meta = json.loads(meta_path.read_text())
            label, cell = meta["label"], meta["cell"]
            for regime, cp_info in meta["saved_checkpoints"].items():
                cp_path = probe_root / cp_info["path"]
                if not cp_path.exists():
                    print(f"  [skip] missing checkpoint {cp_path}")
                    continue
                print(f"[patch] {label} regime={regime} "
                      f"step={cp_info['step']}")
                state = torch.load(cp_path, map_location=device)
                rows.extend(analyse_checkpoint(
                    label, cell, cp_path, state, device, rng, args.top_k,
                    args.ctrl_mode))
                if args.smoke:
                    done = True
                    break

    if not rows:
        print("No checkpoints analysed.")
        return

    out_csv = outdir / "activation_patching_results.csv"
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"\nWrote {len(rows)} rows -> {out_csv}")


if __name__ == "__main__":
    main()
