"""Multi-harmonic Fourier-alignment recompute on saved trajectories.

For each cell that has a saved per-step embedding trajectory, recompute
F_raw^{(m)}(t) = R^2 explained by the top-m Fourier harmonics, for
m in {1, 2, 3, 5}. Apply the same null correction (LOCF refreshed every
5,000 steps) and the same EMA + 1-break BIC pipeline. Report:

  tau_F^{1h}     := canonical single-harmonic onset (matches paper's tau_F)
  tau_F^{(m)}    := top-m onset
  delta_F^{(m)}  := tau_F^{(m)} - tau_F^{1h}

Inputs:
  results/posthoc/traj_*.npz                (8 cells, full emb trajectory)
  runs/causal_probe/{cell}/checkpoints/*.pt (5 cells, 4 checkpoints, models)

Outputs:
  results/posthoc/multi_harmonic_recompute_emb.csv
  results/posthoc/multi_harmonic_recompute_logit.csv
  results/posthoc/multi_harmonic_summary.md
"""

from __future__ import annotations

import csv
import json
import re
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from grok_metrics import estimate_changepoint  # noqa: E402

PRIME = 53
M_VALUES = [1, 2, 3, 5]
NULL_REFRESH_STEPS = 5_000
N_NULL_PERMS = 100
EMA_ALPHA = 0.15


def _build_fourier_pair_basis(p: int = PRIME):
    """Return shape ((p-1)//2, p, 2): orthonormal 2D basis per harmonic k."""
    half = (p - 1) // 2
    j = np.arange(p, dtype=np.float64)
    bases = np.zeros((half, p, 2), dtype=np.float64)
    for k in range(1, half + 1):
        c = np.cos(2 * np.pi * k * j / p)
        s = np.sin(2 * np.pi * k * j / p)
        c -= c.mean()
        s -= s.mean()
        c /= np.linalg.norm(c) + 1e-12
        s /= np.linalg.norm(s) + 1e-12
        s -= s @ c * c
        s /= np.linalg.norm(s) + 1e-12
        bases[k - 1, :, 0] = c
        bases[k - 1, :, 1] = s
    return bases  # (half, p, 2)


def _per_harmonic_fractions(M: np.ndarray, basis_pairs: np.ndarray) -> np.ndarray:
    """Return per-harmonic R^2 fractions for matrix M (rows centered)."""
    Mc = M - M.mean(axis=0, keepdims=True)
    total = float(np.sum(Mc * Mc)) + 1e-12
    half = basis_pairs.shape[0]
    fracs = np.zeros(half, dtype=np.float64)
    for k in range(half):
        Q = basis_pairs[k]                      # (p, 2)
        proj = Q @ (Q.T @ Mc)                   # (p, d)
        fracs[k] = float(np.sum(proj * proj)) / total
    return fracs


def _topm_fraction(fracs: np.ndarray, m: int) -> float:
    return float(np.sort(fracs)[-m:].sum())


def _ema(values: np.ndarray, alpha: float = EMA_ALPHA) -> np.ndarray:
    out = values.astype(np.float64).copy()
    for i in range(1, len(out)):
        out[i] = alpha * values[i] + (1 - alpha) * out[i - 1]
    return out


def _build_null_trajectory(emb_traj: np.ndarray, basis_pairs: np.ndarray,
                           m: int, seed: int) -> np.ndarray:
    """LOCF null at p95 over 100 random row-permutations, refreshed every
    NULL_REFRESH_STEPS log points."""
    T, p, d = emb_traj.shape
    rng = np.random.default_rng(seed)
    null = np.zeros(T, dtype=np.float64)
    last = 0.0
    refresh_every_log = max(1, NULL_REFRESH_STEPS // 500)
    for t in range(T):
        if t % refresh_every_log == 0:
            E = emb_traj[t].astype(np.float64)
            samples = []
            for _ in range(N_NULL_PERMS):
                perm = rng.permutation(p)
                E_shuf = E[perm]
                fracs = _per_harmonic_fractions(E_shuf, basis_pairs)
                samples.append(_topm_fraction(fracs, m))
            last = float(np.percentile(samples, 95))
        null[t] = last
    return null


def _multi_harmonic_trajectory(emb_traj: np.ndarray, basis_pairs: np.ndarray,
                               m: int):
    T = emb_traj.shape[0]
    raw = np.zeros(T, dtype=np.float64)
    for t in range(T):
        fracs = _per_harmonic_fractions(emb_traj[t].astype(np.float64),
                                        basis_pairs)
        raw[t] = _topm_fraction(fracs, m)
    return raw


def _process_emb_traj(npz_path: Path, label: str, basis_pairs: np.ndarray):
    d = np.load(npz_path, allow_pickle=True)
    steps = d["steps"].astype(int)
    emb_traj = d["emb"].astype(np.float64)  # (T, p, d)

    rows = []
    seed = abs(hash(label)) % (2 ** 31)
    for m in M_VALUES:
        raw = _multi_harmonic_trajectory(emb_traj, basis_pairs, m)
        null = _build_null_trajectory(emb_traj, basis_pairs, m, seed=seed + m)
        corr = np.maximum(0.0, raw - null)
        smoothed = _ema(corr)
        tau = estimate_changepoint(steps.tolist(), smoothed.tolist(),
                                   slope_rel_threshold=0.01)
        rows.append({
            "label": label, "m": m,
            "tau_F_topm": tau,
            "f_raw_max": float(raw.max()),
            "f_corr_max": float(corr.max()),
            "f_null_max": float(null.max()),
        })
    return rows


def _logit_per_class_fractions(L: np.ndarray, basis_pairs: np.ndarray,
                                target_indices: np.ndarray) -> np.ndarray:
    """For logit matrix L of shape (n_pairs, p), with target class indices
    target_indices of shape (n_pairs,), measure how well a single
    Fourier harmonic of (a+b) mod p explains the per-pair logit pattern.
    Returns per-harmonic fractions of dimension half."""
    p = L.shape[1]
    Lc = L - L.mean(axis=0, keepdims=True)
    total = float(np.sum(Lc * Lc)) + 1e-12
    half = basis_pairs.shape[0]
    fracs = np.zeros(half, dtype=np.float64)
    for k in range(half):
        # Build basis indexed BY input pair: c[ab]=cos(2*pi*k*s/p), s[ab]=sin
        s_idx = target_indices  # (n_pairs,)
        c = np.cos(2 * np.pi * k * s_idx / p)
        s = np.sin(2 * np.pi * k * s_idx / p)
        c -= c.mean(); s -= s.mean()
        c /= np.linalg.norm(c) + 1e-12
        s /= np.linalg.norm(s) + 1e-12
        s -= s @ c * c; s /= np.linalg.norm(s) + 1e-12
        Q = np.column_stack([c, s])  # (n_pairs, 2)
        proj = Q @ (Q.T @ Lc)        # (n_pairs, p)
        fracs[k] = float(np.sum(proj * proj)) / total
    return fracs


def _process_causal_probe_logits(probe_root: Path, basis_pairs: np.ndarray):
    """Compute multi-harmonic logit alignment at the 4 saved checkpoints
    per causal-probe cell. Requires loading the model and forwarding all
    p^2 input pairs."""
    sys.path.insert(0,
                    str(Path(__file__).resolve().parent.parent))
    from grokking_baseline import (  # noqa: WPS433
        GrokkingTransformer, make_dataset, set_seed,
    )

    if not probe_root.exists():
        return []

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rows = []
    p = PRIME
    a_vals = torch.arange(p)
    b_vals = torch.arange(p)
    aa, bb = torch.meshgrid(a_vals, b_vals, indexing="ij")
    all_pairs = torch.stack([aa.flatten(), bb.flatten()], dim=1).to(device)
    target_classes = ((aa + bb) % p).flatten().numpy()

    for cell_dir in sorted(probe_root.iterdir()):
        if not cell_dir.is_dir():
            continue
        meta_path = cell_dir / "meta.json"
        if not meta_path.exists():
            continue
        meta = json.loads(meta_path.read_text())
        cell = meta["cell"]
        set_seed(cell["seed"])
        model = GrokkingTransformer(
            prime=p, d_model=256, n_heads=4, n_layers=2, d_ff=1024,
            dropout=0.0,
        ).to(device)
        for regime, cp_info in meta["saved_checkpoints"].items():
            cp_path = probe_root / cp_info["path"]
            state = torch.load(cp_path, map_location=device)
            model.load_state_dict(state["model_state"])
            model.eval()
            with torch.no_grad():
                logits = model(all_pairs).cpu().numpy()  # (p^2, p)
            fracs = _logit_per_class_fractions(
                logits, basis_pairs, target_classes)
            for m in M_VALUES:
                rows.append({
                    "label": meta["label"], "regime": regime,
                    "step": cp_info["step"], "m": m,
                    "f_logit_topm_raw": _topm_fraction(fracs, m),
                    "ordering": cell["ordering"],
                })
    return rows


def main():
    posthoc_dir = Path("results/posthoc")
    probe_root = Path("runs/causal_probe")
    out_dir = posthoc_dir
    basis_pairs = _build_fourier_pair_basis(PRIME)

    # ------------------------------------------------------------------
    # Embedding multi-harmonic on posthoc trajectories
    # ------------------------------------------------------------------
    emb_rows = []
    traj_files = sorted(posthoc_dir.glob("traj_*.npz"))
    print(f"Found {len(traj_files)} posthoc trajectory files.")

    canonical_taus: dict[str, dict[int, float]] = {}
    for traj_path in traj_files:
        m = re.match(r"traj_(.+)\.npz", traj_path.name)
        if not m:
            continue
        label = m.group(1)
        print(f"[emb] processing {label} ...")
        rows = _process_emb_traj(traj_path, label, basis_pairs)
        emb_rows.extend(rows)
        canonical_taus[label] = {r["m"]: r["tau_F_topm"] for r in rows}

    out_emb_csv = out_dir / "multi_harmonic_recompute_emb.csv"
    if emb_rows:
        with open(out_emb_csv, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(emb_rows[0].keys()))
            w.writeheader()
            w.writerows(emb_rows)
        print(f"  [csv] {out_emb_csv}")

    # ------------------------------------------------------------------
    # Logit multi-harmonic on causal-probe checkpoints
    # ------------------------------------------------------------------
    logit_rows = _process_causal_probe_logits(probe_root, basis_pairs)
    out_logit_csv = out_dir / "multi_harmonic_recompute_logit.csv"
    if logit_rows:
        with open(out_logit_csv, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(logit_rows[0].keys()))
            w.writeheader()
            w.writerows(logit_rows)
        print(f"  [csv] {out_logit_csv}")

    # ------------------------------------------------------------------
    # Summary markdown
    # ------------------------------------------------------------------
    out_md = out_dir / "multi_harmonic_summary.md"
    with open(out_md, "w", encoding="utf-8") as f:
        f.write("# Multi-harmonic Fourier-alignment recompute\n\n")
        f.write("## Embedding multi-harmonic ($\\tau_F^{(m)}$) "
                f"across {len(traj_files)} posthoc cells\n\n")
        f.write("| cell | m=1 (canonical) | m=2 | m=3 | m=5 | "
                "delta(5-1) [steps] |\n")
        f.write("|---|---|---|---|---|---|\n")
        for label, taus in canonical_taus.items():
            tau1 = taus.get(1)
            row = [str(label),
                   f"{int(tau1)}" if tau1 else "--"]
            for m in [2, 3, 5]:
                t = taus.get(m)
                row.append(f"{int(t)}" if t else "--")
            d = (taus.get(5) - tau1) if (taus.get(5) and tau1) else None
            row.append(f"{int(d):+d}" if d is not None else "--")
            f.write("| " + " | ".join(row) + " |\n")

        # Aggregate shift summary
        if canonical_taus:
            shifts = [taus.get(5) - taus.get(1)
                      for taus in canonical_taus.values()
                      if taus.get(5) and taus.get(1)]
            if shifts:
                shifts = np.array(shifts)
                f.write(f"\n**Median |delta(tau_F^5 - tau_F^1)|** "
                        f"= {int(np.median(np.abs(shifts)))} steps "
                        f"across {len(shifts)} cells.\n")
                f.write(f"**Max |delta|** = {int(np.max(np.abs(shifts)))} "
                        f"steps. **Within 1 measurement bin (500 steps)**: "
                        f"{int(np.sum(np.abs(shifts) <= 500))}/"
                        f"{len(shifts)} cells.\n")

        if logit_rows:
            f.write("\n## Logit multi-harmonic at causal-probe checkpoints\n\n")
            f.write("Per-cell, per-regime top-m logit Fourier R^2 (raw, no "
                    "null correction):\n\n")
            f.write("| cell | regime | step | m=1 | m=2 | m=3 | m=5 |\n")
            f.write("|---|---|---|---|---|---|---|\n")
            grouped: dict[tuple, dict[int, float]] = {}
            for r in logit_rows:
                k = (r["label"], r["regime"], r["step"])
                grouped.setdefault(k, {})[r["m"]] = r["f_logit_topm_raw"]
            for (label, regime, step), ms in grouped.items():
                vals = [ms.get(m, float('nan')) for m in [1, 2, 3, 5]]
                f.write(f"| {label} | {regime} | {step} | "
                        + " | ".join(f"{v:.3f}" for v in vals) + " |\n")
    print(f"  [md] {out_md}")


if __name__ == "__main__":
    main()
