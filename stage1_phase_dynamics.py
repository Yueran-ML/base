#!/usr/bin/env python3
"""
Stage 1: Phase Dynamics Comparison
====================================
3 cells × 3 seeds, tracking revised metric suite to answer:

  Q1: 修订指标能否区分 Grokking 和 Memorization？
  Q2: Collapse 的动力学是否与普通 Memorization 不同？

Cell design (same lr=1.6e-3, only wd varies — clean single-axis contrast):
  grokking   : lr=1.6e-3, wd=1.0   (left-cluster, confirmed in Stage 0 v2/v3)
  memorization: lr=1.6e-3, wd=0.04  (confirmed in Stage 0 v2)
  collapse   : lr=1.6e-3, wd=10.0  (wd >> lr → expected catastrophic weight collapse)

Revised metric suite (every 100 steps):
  G(t)      : test accuracy + test NLL → tau_gen (sustained >0.9)
  F(t)      : Fourier alignment R² (best harmonic)  → tau_F (changepoint)
  eS(t)     : effective dimensionality e^S (MIT paper)
  emb_norm  : mean L2 norm of token embedding rows  (embedding layer, wd=0)
  dec_norm  : mean L2 norm of weight matrices in decoder (subject to wd)
  train_nll : training NLL
  gen_gap   : test_nll - train_nll  (generalization gap)

Usage:
  python stage1_phase_dynamics.py
  python stage1_phase_dynamics.py --max-steps 50000 --outdir runs/stage1_dynamics
  python stage1_phase_dynamics.py --cells grokking memorization --seeds 42  # quick check
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, str(Path(__file__).parent))
from grokking_baseline import (
    GrokkingTransformer,
    build_optimizer,
    build_scheduler,
    compute_effective_dim,
    get_token_embeddings,
    make_dataset,
    set_seed,
    split_dataset,
)

# ---------------------------------------------------------------------------
# Cell definitions — same lr, only wd varies for clean comparison
# ---------------------------------------------------------------------------

ALL_CELLS: dict[str, dict] = {
    "grokking":    dict(lr=1.6e-3, wd=1.0,  expected="Grokking"),
    "memorization": dict(lr=1.6e-3, wd=0.04, expected="Memorization"),
    "collapse":    dict(lr=1.6e-3, wd=10.0, expected="Collapse/Confusion"),
}

DEFAULT_SEEDS = [42, 7, 2025]


# ---------------------------------------------------------------------------
# Fourier alignment F(t)
# ---------------------------------------------------------------------------

def compute_fourier_alignment(embeddings: np.ndarray, prime: int) -> tuple[float, int]:
    """Fraction of centered embedding variance explained by best cyclic harmonic pair."""
    E = embeddings - embeddings.mean(0)
    total_var = float(np.sum(E ** 2))
    if total_var < 1e-12:
        return 0.0, 1
    thetas = 2.0 * np.pi * np.arange(prime) / prime
    best_r2, best_k = 0.0, 1
    for k in range(1, prime // 2 + 1):
        B = np.stack([np.cos(k * thetas), np.sin(k * thetas)], axis=1)
        Q, _ = np.linalg.qr(B)
        proj = Q @ (Q.T @ E)
        res_var = float(np.sum((E - proj) ** 2))
        r2 = 1.0 - res_var / total_var
        if r2 > best_r2:
            best_r2, best_k = r2, k
    return float(best_r2), best_k


# ---------------------------------------------------------------------------
# Decoder weight norm — mean L2 norm of weight matrices (not biases/LN)
# ---------------------------------------------------------------------------

def compute_decoder_norm(model: GrokkingTransformer) -> float:
    """Mean L2 norm of decoder weight matrices (attention + ff + head).

    Excludes: token_embed, pos_embed, LayerNorm parameters, biases.
    These are the parameters subject to weight decay.
    """
    norms = []
    skip_names = {"token_embed.weight", "pos_embed"}
    for name, param in model.named_parameters():
        if name in skip_names:
            continue
        if "norm" in name:      # LayerNorm weights/biases
            continue
        if "bias" in name:      # all biases
            continue
        norms.append(param.data.norm(2).item())
    return float(np.mean(norms)) if norms else 0.0


# ---------------------------------------------------------------------------
# Changepoint estimator (same as Stage 0)
# ---------------------------------------------------------------------------

def estimate_changepoint(
    steps: list[int],
    values: list[float],
    min_frac: float = 1 / 6,
    max_frac: float = 5 / 6,
    min_post_points: int = 5,
    slope_rel_threshold: float = 0.01,
) -> Optional[float]:
    n = len(steps)
    if n < 12:
        return None
    log_t = np.log1p(np.array(steps, dtype=float))
    v = np.array(values, dtype=float)
    smoothed = v.copy()
    alpha = 0.15
    for i in range(1, n):
        smoothed[i] = alpha * v[i] + (1.0 - alpha) * smoothed[i - 1]

    def ols_rss(x, y):
        A = np.column_stack([x, np.ones(len(x))])
        coeffs, res, _, _ = np.linalg.lstsq(A, y, rcond=None)
        pred = A @ coeffs
        rss = float(np.sum((y - pred) ** 2))
        return float(coeffs[0]), rss

    lo = max(3, int(min_frac * n))
    hi = min(n - 3, int(max_frac * n))
    _, rss0 = ols_rss(log_t, smoothed)
    bic_nobreak = n * np.log(max(rss0 / n, 1e-15)) + 2.0 * np.log(n)
    best_bic, best_bp = bic_nobreak, None

    for bp in range(lo, hi + 1):
        x1, y1 = log_t[:bp], smoothed[:bp]
        x2, y2 = log_t[bp:], smoothed[bp:]
        if len(x1) < 2 or len(x2) < 2:
            continue
        _, rss1 = ols_rss(x1, y1)
        _, rss2 = ols_rss(x2, y2)
        total_rss = rss1 + rss2
        bic = n * np.log(max(total_rss / n, 1e-15)) + 4.0 * np.log(n)
        if bic < best_bic:
            best_bic, best_bp = bic, bp

    if best_bp is None or (n - best_bp) < min_post_points:
        return None
    slope2, _ = ols_rss(log_t[best_bp:], smoothed[best_bp:])
    val_range = max(float(np.max(smoothed) - np.min(smoothed)), 1e-9)
    if abs(slope2) < slope_rel_threshold * val_range:
        return None
    return float(steps[best_bp])


def find_tau_sustained(
    steps: list[int],
    values: list[float],
    threshold: float,
    n_sustained: int = 3,
) -> Optional[float]:
    n = len(steps)
    for i in range(n - n_sustained + 1):
        if all(values[i + k] >= threshold for k in range(n_sustained)):
            return float(steps[i])
    return None


# ---------------------------------------------------------------------------
# Cell result dataclass
# ---------------------------------------------------------------------------

@dataclass
class CellResult:
    cell_name: str
    lr: float
    wd: float
    seed: int
    expected_phase: str
    steps: list[int] = field(default_factory=list)
    train_acc:   list[float] = field(default_factory=list)
    test_acc:    list[float] = field(default_factory=list)
    train_nll:   list[float] = field(default_factory=list)
    test_nll:    list[float] = field(default_factory=list)
    gen_gap:     list[float] = field(default_factory=list)   # test_nll - train_nll
    fourier_r2:  list[float] = field(default_factory=list)
    fourier_k:   list[int]   = field(default_factory=list)
    eff_dim:     list[float] = field(default_factory=list)
    emb_norm:    list[float] = field(default_factory=list)
    dec_norm:    list[float] = field(default_factory=list)
    # Onset estimates
    tau_gen:     Optional[float] = None
    tau_fourier: Optional[float] = None
    tau_es:      Optional[float] = None   # changepoint in -eS (drop = decrease in eS)
    # Summary
    observed_phase: str = ""
    runtime_sec: float = 0.0


def classify_phase(steps, train_acc, test_acc):
    t_train = next((s for s, a in zip(steps, train_acc) if a >= 0.9), None)
    t_test  = next((s for s, a in zip(steps, test_acc)  if a >= 0.9), None)
    if t_train is None:
        return "Confusion"
    if t_test is None:
        return "Memorization"
    return "Grokking" if (t_test - t_train) >= 1000 else "Comprehension"


# ---------------------------------------------------------------------------
# Main training loop
# ---------------------------------------------------------------------------

def run_cell(
    cell_name: str,
    lr: float,
    wd: float,
    seed: int,
    expected_phase: str,
    prime: int = 53,
    train_fraction: float = 0.3,
    max_steps: int = 50_000,
    log_interval: int = 100,
    embed_lr: float = 1e-3,
    d_model: int = 256,
    n_heads: int = 4,
    n_layers: int = 2,
    d_ff: int = 1024,
) -> CellResult:
    t0 = time.time()
    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    x, y = make_dataset(prime, "add", seed)
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
    optimizer = build_optimizer(model, decoder_lr=lr, embed_lr=embed_lr, decoder_weight_decay=wd)
    scheduler = build_scheduler(
        optimizer, warmup_steps=10, lr_schedule="constant",
        lr_min_ratio=0.05, max_steps=max_steps,
    )

    result = CellResult(
        cell_name=cell_name, lr=lr, wd=wd, seed=seed, expected_phase=expected_phase,
    )

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
                    tr_logits = model(train_x.to(device))
                    tr_loss = criterion(tr_logits, train_y.to(device)).item()
                    tr_acc  = (tr_logits.argmax(1).cpu() == train_y).float().mean().item()
                    te_logits = model(test_x.to(device))
                    te_loss = criterion(te_logits, test_y.to(device)).item()
                    te_acc  = (te_logits.argmax(1).cpu() == test_y).float().mean().item()

                emb = get_token_embeddings(model, prime)
                f_r2, best_k = compute_fourier_alignment(emb, prime)
                ed = compute_effective_dim(emb)
                e_norm = float(np.mean(np.linalg.norm(emb, axis=1)))
                d_norm = compute_decoder_norm(model)

                result.steps.append(step)
                result.train_acc.append(tr_acc)
                result.test_acc.append(te_acc)
                result.train_nll.append(tr_loss)
                result.test_nll.append(te_loss)
                result.gen_gap.append(te_loss - tr_loss)
                result.fourier_r2.append(f_r2)
                result.fourier_k.append(best_k)
                result.eff_dim.append(ed)
                result.emb_norm.append(e_norm)
                result.dec_norm.append(d_norm)

            if step >= max_steps:
                break

    result.observed_phase = classify_phase(result.steps, result.train_acc, result.test_acc)
    result.tau_gen     = find_tau_sustained(result.steps, result.test_acc, 0.9, n_sustained=3)
    result.tau_fourier = estimate_changepoint(result.steps, result.fourier_r2)
    # For eS: detect when it drops (use negative eS for changepoint)
    neg_es = [-v for v in result.eff_dim]
    result.tau_es = estimate_changepoint(result.steps, neg_es)
    result.runtime_sec = time.time() - t0
    return result


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_cell(result: CellResult, outdir: Path) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return

    steps = result.steps
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    title = (
        f"{result.cell_name} | lr={result.lr:.2e} wd={result.wd:.2f} seed={result.seed}\n"
        f"observed={result.observed_phase}  "
        f"τ_gen={result.tau_gen}  τ_F={result.tau_fourier}  τ_eS={result.tau_es}"
    )
    fig.suptitle(title, fontsize=10)

    def _vlines(ax):
        for tau, color, label in [
            (result.tau_gen,     "tab:orange", "τ_gen"),
            (result.tau_fourier, "tab:green",  "τ_F"),
            (result.tau_es,      "tab:purple", "τ_eS"),
        ]:
            if tau:
                ax.axvline(tau, ls=":", color=color, lw=1.2, label=label)

    ax = axes[0, 0]
    ax.plot(steps, result.train_acc, label="train", color="tab:blue", lw=1.5)
    ax.plot(steps, result.test_acc,  label="test",  color="tab:orange", lw=1.5)
    ax.axhline(0.9, ls="--", color="gray", lw=0.8)
    ax.set_title("Accuracy"); ax.legend(fontsize=8); ax.grid(True, alpha=0.3)
    _vlines(ax)

    ax = axes[0, 1]
    ax.plot(steps, result.train_nll, label="train NLL", color="tab:blue",   lw=1.2)
    ax.plot(steps, result.test_nll,  label="test NLL",  color="tab:orange", lw=1.2)
    ax.set_title("NLL (loss)"); ax.legend(fontsize=8); ax.grid(True, alpha=0.3)
    _vlines(ax)

    ax = axes[0, 2]
    ax.plot(steps, result.gen_gap, color="tab:red", lw=1.5)
    ax.axhline(0, ls="--", color="gray", lw=0.8)
    ax.set_title("Generalization Gap (test−train NLL)"); ax.grid(True, alpha=0.3)
    _vlines(ax)

    ax = axes[1, 0]
    ax.plot(steps, result.fourier_r2, color="tab:green", lw=1.5)
    ax.set_title("Fourier Alignment F(t)"); ax.set_ylabel("R²"); ax.grid(True, alpha=0.3)
    _vlines(ax)

    ax = axes[1, 1]
    ax.plot(steps, result.eff_dim, color="tab:purple", lw=1.5)
    ax.set_title("Effective Dimensionality e^S"); ax.grid(True, alpha=0.3)
    _vlines(ax)

    ax = axes[1, 2]
    ax2 = ax.twinx()
    ax.plot(steps, result.emb_norm,  label="emb_norm",  color="tab:cyan",   lw=1.5)
    ax2.plot(steps, result.dec_norm, label="dec_norm",  color="tab:brown",  lw=1.5, ls="--")
    ax.set_title("Weight Norms")
    ax.set_ylabel("emb_norm", color="tab:cyan")
    ax2.set_ylabel("dec_norm", color="tab:brown")
    lines1, labs1 = ax.get_legend_handles_labels()
    lines2, labs2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labs1 + labs2, fontsize=8)
    ax.grid(True, alpha=0.3)
    _vlines(ax)

    for panel in axes.flatten():
        panel.set_xlabel("step")

    fig.tight_layout()
    fig.savefig(outdir / "plot.png", dpi=130, bbox_inches="tight")
    plt.close(fig)


def plot_comparison(all_results: dict[str, list[CellResult]], outdir: Path) -> None:
    """Multi-cell overlay plot — one panel per metric, seeds overlaid per cell."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return

    COLORS = {
        "grokking":    "tab:blue",
        "memorization": "tab:orange",
        "collapse":    "tab:red",
    }

    metrics = [
        ("test_acc",   "Test Accuracy"),
        ("fourier_r2", "Fourier Alignment F(t)"),
        ("eff_dim",    "Effective Dim e^S"),
        ("emb_norm",   "Embedding Norm"),
        ("dec_norm",   "Decoder Weight Norm"),
        ("gen_gap",    "Gen Gap (test−train NLL)"),
    ]

    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    fig.suptitle("Phase Dynamics Comparison: Grokking / Memorization / Collapse", fontsize=11)

    for ax, (attr, title) in zip(axes.flatten(), metrics):
        for cell_name, results in all_results.items():
            color = COLORS.get(cell_name, "gray")
            for i, r in enumerate(results):
                vals = getattr(r, attr, [])
                lbl = cell_name if i == 0 else None
                ax.plot(r.steps, vals, color=color, lw=1.2, alpha=0.6, label=lbl)
        ax.set_title(title)
        ax.set_xlabel("step")
        ax.grid(True, alpha=0.3)
        if attr == "test_acc":
            ax.axhline(0.9, ls="--", color="gray", lw=0.8)
            ax.legend(fontsize=8)

    fig.tight_layout()
    fig.savefig(outdir / "comparison.png", dpi=130, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Diagnostic report
# ---------------------------------------------------------------------------

def write_report(all_results: dict[str, list[CellResult]], outdir: Path) -> None:
    lines = ["# Stage 1: Phase Dynamics Report\n"]

    # Table of onset times per cell × seed
    lines.append("## Onset Estimates\n")
    lines.append("| Cell | Seed | Observed Phase | τ_gen | τ_F | τ_eS | F@50k | eS_drop |\n")
    lines.append("|------|------|---------------|-------|-----|------|-------|--------|\n")

    for cell_name, results in all_results.items():
        for r in results:
            f_final = f"{r.fourier_r2[-1]:.3f}" if r.fourier_r2 else "—"
            es_drop = "—"
            if r.eff_dim:
                es_drop = f"{r.eff_dim[0]:.1f}→{r.eff_dim[-1]:.1f}"
            lines.append(
                f"| {cell_name} | {r.seed} | {r.observed_phase} | "
                f"{r.tau_gen or '—'} | {r.tau_fourier or '—'} | {r.tau_es or '—'} | "
                f"{f_final} | {es_drop} |\n"
            )

    # Q1 analysis
    lines.append("\n## Q1: 修订指标能否区分 Grokking 和 Memorization？\n")

    grok_results = all_results.get("grokking", [])
    memo_results = all_results.get("memorization", [])

    def median_final(results, attr):
        vals = [getattr(r, attr, []) for r in results]
        finals = [v[-1] for v in vals if v]
        return float(np.median(finals)) if finals else None

    def median_tau(results, attr):
        vals = [getattr(r, attr) for r in results if getattr(r, attr) is not None]
        return float(np.median(vals)) if vals else None

    for label, metric, attr in [
        ("F(t) final value", "fourier_r2", "fourier_r2"),
        ("eS final value", "eff_dim", "eff_dim"),
        ("dec_norm final", "dec_norm", "dec_norm"),
        ("emb_norm final", "emb_norm", "emb_norm"),
        ("gen_gap final", "gen_gap", "gen_gap"),
    ]:
        g_val = median_final(grok_results, attr)
        m_val = median_final(memo_results, attr)
        sep = "DISCRIMINATIVE" if g_val is not None and m_val is not None and abs(g_val - m_val) > 0.05 * max(abs(g_val), abs(m_val), 1e-9) else "similar"
        g_str = f"{g_val:.3f}" if g_val is not None else "N/A"
        m_str = f"{m_val:.3f}" if m_val is not None else "N/A"
        lines.append(f"- **{label}**: Grokking={g_str}, Memorization={m_str} -> {sep}\n")

    g_tau = median_tau(grok_results, "tau_fourier")
    m_tau = median_tau(memo_results, "tau_fourier")
    lines.append(f"- **τ_F (median)**: Grokking={g_tau}, Memorization={m_tau}\n")

    g_es = median_tau(grok_results, "tau_es")
    m_es = median_tau(memo_results, "tau_es")
    lines.append(f"- **τ_eS (median)**: Grokking={g_es}, Memorization={m_es}\n")

    # Q2 analysis
    lines.append("\n## Q2: Collapse 与普通 Memorization 的动力学差异\n")
    coll_results = all_results.get("collapse", [])

    coll_phase = [r.observed_phase for r in coll_results]
    memo_phase = [r.observed_phase for r in memo_results]
    lines.append(f"- **Collapse observed phases**: {coll_phase}\n")
    lines.append(f"- **Memorization observed phases**: {memo_phase}\n")

    c_dec = median_final(coll_results, "dec_norm")
    m_dec = median_final(memo_results, "dec_norm")
    c_emb = median_final(coll_results, "emb_norm")
    m_emb = median_final(memo_results, "emb_norm")
    lines.append(f"- **dec_norm final**: Collapse={f'{c_dec:.4f}' if c_dec is not None else 'N/A'}, Memorization={f'{m_dec:.4f}' if m_dec is not None else 'N/A'}\n")
    lines.append(f"- **emb_norm final**: Collapse={f'{c_emb:.4f}' if c_emb is not None else 'N/A'}, Memorization={f'{m_emb:.4f}' if m_emb is not None else 'N/A'}\n")

    c_acc = median_final(coll_results, "train_acc")
    m_acc = median_final(memo_results, "train_acc")
    lines.append(f"- **train_acc final**: Collapse={f'{c_acc:.3f}' if c_acc is not None else 'N/A'}, Memorization={f'{m_acc:.3f}' if m_acc is not None else 'N/A'}\n")
    lines.append(
        "\n注：Collapse 的核心特征应是 dec_norm 快速归零（weight decay >> lr），"
        "导致 train_acc 无法提升，与 Memorization（train_acc→1 但 test_acc 不提升）形成对比。\n"
    )

    # Runtime
    lines.append("\n## Runtime\n")
    for cell_name, results in all_results.items():
        for r in results:
            lines.append(f"- {cell_name} seed={r.seed}: {r.runtime_sec:.1f}s\n")

    report_path = outdir / "report.md"
    report_path.write_text("".join(lines), encoding="utf-8")
    print(f"Report: {report_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--cells",     nargs="+", default=list(ALL_CELLS.keys()),
                   choices=list(ALL_CELLS.keys()),
                   help="Which cells to run (default: all 3)")
    p.add_argument("--seeds",     nargs="+", type=int, default=DEFAULT_SEEDS)
    p.add_argument("--max-steps", type=int, default=50_000)
    p.add_argument("--log-interval", type=int, default=100)
    p.add_argument("--outdir",    default="runs/stage1_dynamics")
    p.add_argument("--train-fraction", type=float, default=0.3)
    return p.parse_args()


def main():
    args = parse_args()
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    all_results: dict[str, list[CellResult]] = {}
    total_runs = len(args.cells) * len(args.seeds)
    run_idx = 0

    for cell_name in args.cells:
        cfg = ALL_CELLS[cell_name]
        cell_results = []
        for seed in args.seeds:
            run_idx += 1
            print(f"\n[{run_idx}/{total_runs}] {cell_name} seed={seed} "
                  f"(lr={cfg['lr']:.2e} wd={cfg['wd']:.2f}) "
                  f"max_steps={args.max_steps}", flush=True)

            r = run_cell(
                cell_name=cell_name,
                lr=cfg["lr"],
                wd=cfg["wd"],
                seed=seed,
                expected_phase=cfg["expected"],
                max_steps=args.max_steps,
                log_interval=args.log_interval,
                train_fraction=args.train_fraction,
            )
            print(f"  → phase={r.observed_phase}  τ_gen={r.tau_gen}  "
                  f"τ_F={r.tau_fourier}  τ_eS={r.tau_es}  "
                  f"F_final={r.fourier_r2[-1]:.3f}  "
                  f"dec_norm={r.dec_norm[-1]:.4f}  "
                  f"emb_norm={r.emb_norm[-1]:.2f}  "
                  f"t={r.runtime_sec:.0f}s", flush=True)

            cell_dir = outdir / f"cell_{cell_name}_seed{seed}"
            cell_dir.mkdir(exist_ok=True)
            with open(cell_dir / "metrics.json", "w") as f:
                json.dump(asdict(r), f, indent=2)
            plot_cell(r, cell_dir)
            cell_results.append(r)

        all_results[cell_name] = cell_results

    # Comparison plot and report
    plot_comparison(all_results, outdir)
    write_report(all_results, outdir)
    print(f"\nAll done. Results in {outdir}/")
    print(f"  comparison.png  — overlay plot")
    print(f"  report.md       — diagnostic answers to Q1 and Q2")


if __name__ == "__main__":
    main()
