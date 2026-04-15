#!/usr/bin/env python3
"""
Stage 2: Boundary Local Grid
==============================
沿 wd 轴精确定位 Grokking/Memorization 相界，固定 lr=1.6e-3。

决策背景（自动决策模式触发）：
  Stage 1 Q1 = 3/3 PASS（dec_norm / e^S / gen_gap 全部区分 G vs M）
  Stage 1 Q2 = 0/3 FAIL（wd=10 并非瞬时崩塌；collapse seed=7 意外 Grokked）

Cell 设计（wd 轴边界两侧各 3 点）：
  Memorization 侧: wd ∈ {0.01, 0.02, 0.04}  (已知 wd=0.04 → Memo)
  Grokking 侧:     wd ∈ {0.25, 1.0, 2.5}    (已知 wd∈[0.25,2.5] → Grokking)

步数策略：
  base = 30,000 步
  延长至 50k：tau_gen 未出现 且 train_acc > 0.9 (Grokking 进行中)
  延长至 80k：50k 时仍未泛化 且 train_acc > 0.9
  提前停止：tau_gen 已确认 且 稳定 ≥ 5 个 checkpoint

Usage:
  python stage2_boundary_grid.py
  python stage2_boundary_grid.py --wd-extra 0.1 0.15 0.5  # 追加 wd 值
  python stage2_boundary_grid.py --seeds 42 --base-steps 20000   # 快速预览
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
from stage1_phase_dynamics import (
    CellResult,
    classify_phase,
    compute_decoder_norm,
    compute_fourier_alignment,
    estimate_changepoint,
    find_tau_sustained,
    plot_cell,
)

# ---------------------------------------------------------------------------
# Grid definition
# ---------------------------------------------------------------------------

FIXED_LR = 1.6e-3
DEFAULT_WD_GRID = [0.01, 0.02, 0.04, 0.25, 1.0, 2.5]
DEFAULT_SEEDS   = [42, 7, 2025]

BASE_STEPS    = 30_000
EXTEND1_STEPS = 50_000   # 延长阈值 1（运行时由 base_steps 动态设定）
EXTEND2_STEPS = 80_000   # 延长阈值 2
EARLY_STOP_N  = 5        # tau_gen 稳定 N 个 checkpoint 后可提前停止


# ---------------------------------------------------------------------------
# Training loop with dynamic extension
# ---------------------------------------------------------------------------

def run_cell_adaptive(
    wd: float,
    seed: int,
    prime: int = 53,
    train_fraction: float = 0.3,
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
    optimizer = build_optimizer(model, decoder_lr=FIXED_LR, embed_lr=embed_lr,
                                decoder_weight_decay=wd)
    # 初始 scheduler 按最大可能步数设置（不影响 constant 调度）
    scheduler = build_scheduler(optimizer, warmup_steps=10, lr_schedule="constant",
                                lr_min_ratio=0.05, max_steps=EXTEND2_STEPS)

    cell_name = f"wd{wd:.4f}"
    result = CellResult(
        cell_name=cell_name, lr=FIXED_LR, wd=wd, seed=seed,
        expected_phase="unknown",
    )

    step = 0
    max_steps = BASE_STEPS
    tau_gen_confirmed_count = 0
    extension_log = []

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

                # ---- 动态步数决策 ----
                current_tau_gen = find_tau_sustained(result.steps, result.test_acc, 0.9)

                # 提前停止：tau_gen 已稳定
                if current_tau_gen is not None:
                    tau_gen_confirmed_count += 1
                    if tau_gen_confirmed_count >= EARLY_STOP_N and step >= current_tau_gen + 2000:
                        result.observed_phase = "Grokking"
                        max_steps = step  # 触发退出
                        break

                # 延长判定（仅在 milestone 节点检查）
                if step == BASE_STEPS and current_tau_gen is None:
                    if tr_acc > 0.9:
                        max_steps = EXTEND1_STEPS
                        extension_log.append(f"step={step}: extend to {EXTEND1_STEPS} (train_acc={tr_acc:.3f}, tau_gen=None)")
                    # else: Confusion/Collapse，不延长

                elif step == EXTEND1_STEPS and current_tau_gen is None:
                    if tr_acc > 0.9:
                        max_steps = EXTEND2_STEPS
                        extension_log.append(f"step={step}: extend to {EXTEND2_STEPS} (train_acc={tr_acc:.3f}, tau_gen=None)")

            if step >= max_steps:
                break

    # 最终计算
    result.observed_phase = result.observed_phase or classify_phase(
        result.steps, result.train_acc, result.test_acc
    )
    result.tau_gen     = find_tau_sustained(result.steps, result.test_acc, 0.9)
    result.tau_fourier = estimate_changepoint(result.steps, result.fourier_r2)
    neg_es = [-v for v in result.eff_dim]
    result.tau_es = estimate_changepoint(result.steps, neg_es)
    result.runtime_sec = time.time() - t0

    # 将 extension_log 写入 cell_name（方便追踪）
    if extension_log:
        result.cell_name = cell_name + " [extended]"
    return result, extension_log


# ---------------------------------------------------------------------------
# Summary metrics (Stage 1 判别指标计算)
# ---------------------------------------------------------------------------

def compute_discriminant_metrics(result: CellResult) -> dict:
    steps = result.steps
    gap = result.gen_gap
    dec = result.dec_norm
    es  = result.eff_dim

    # gen_gap 闭合比例
    idx0 = next((i for i,s in enumerate(steps) if s >= 500), 0)
    init_gap = gap[idx0] if gap else 0.0
    final_gap = float(np.mean(gap[-5:])) if len(gap) >= 5 else (gap[-1] if gap else 0.0)
    gap_ratio = final_gap / max(abs(init_gap), 1e-6)

    # e^S 相对变化
    es_ratio = (es[-1] / max(es[0], 1e-6)) if es else 1.0

    return {
        "dec_norm_final": dec[-1] if dec else 0.0,
        "es_final":       es[-1]  if es  else 0.0,
        "es_ratio":       es_ratio,
        "gap_ratio":      gap_ratio,
        "gap_final":      final_gap,
        "f_final":        result.fourier_r2[-1] if result.fourier_r2 else 0.0,
        "tau_gen":        result.tau_gen,
        "tau_es":         result.tau_es,
        "max_steps_used": result.steps[-1] if result.steps else 0,
    }


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

def write_report(all_results: dict[float, list[CellResult]], outdir: Path) -> None:
    lines = ["# Stage 2: Boundary Grid Report\n\n"]
    lines.append(f"固定 lr={FIXED_LR:.2e}，沿 wd 轴扫描相界\n\n")

    lines.append("## 相区分类\n\n")
    lines.append("| wd | seed=42 | seed=7 | seed=2025 | 多数相区 |\n")
    lines.append("|----|---------|--------|-----------|--------|\n")

    boundary_wd = None
    boundary_prev_wd = None
    prev_phase = None
    prev_wd_loop = None
    for wd in sorted(all_results.keys()):
        results = all_results[wd]
        phases = [r.observed_phase for r in results]
        seeds  = {r.seed: r.observed_phase for r in results}
        majority = max(set(phases), key=phases.count)
        lines.append(
            f"| {wd:.3f} | {seeds.get(42,'—')} | {seeds.get(7,'—')} | {seeds.get(2025,'—')} | **{majority}** |\n"
        )
        if prev_phase and prev_phase != majority:
            boundary_wd = wd
            boundary_prev_wd = prev_wd_loop
        prev_phase = majority
        prev_wd_loop = wd

    if boundary_wd:
        lines.append(f"\n**相界估计**：wd ∈ ({boundary_prev_wd:.3f}, {boundary_wd:.3f})（多数相区翻转点）\n")

    lines.append("\n## 判别指标（中位数 across seeds）\n\n")
    lines.append("| wd | 相区 | dec_norm | e^S | gap_ratio | F@final | τ_gen |\n")
    lines.append("|----|------|---------|-----|-----------|---------|------|\n")

    prev_wd = None
    for wd in sorted(all_results.keys()):
        results = all_results[wd]
        phases = [r.observed_phase for r in results]
        majority = max(set(phases), key=phases.count)
        dm_list = [compute_discriminant_metrics(r) for r in results]
        dec = np.median([d["dec_norm_final"] for d in dm_list])
        es  = np.median([d["es_final"]       for d in dm_list])
        gr  = np.median([d["gap_ratio"]      for d in dm_list])
        ff  = np.median([d["f_final"]        for d in dm_list])
        tg_vals = [d["tau_gen"] for d in dm_list if d["tau_gen"]]
        tg  = f"{np.median(tg_vals):.0f}" if tg_vals else "—"
        lines.append(f"| {wd:.3f} | {majority} | {dec:.2f} | {es:.1f} | {gr:.3f} | {ff:.3f} | {tg} |\n")
        prev_wd = wd

    lines.append("\n## 步数使用（延长情况）\n\n")
    lines.append("| wd | seed | 使用步数 | 运行时间 |\n")
    lines.append("|----|------|---------|--------|\n")
    for wd in sorted(all_results.keys()):
        for r in all_results[wd]:
            lines.append(f"| {wd:.3f} | {r.seed} | {r.steps[-1] if r.steps else 0} | {r.runtime_sec:.0f}s |\n")

    (outdir / "report.md").write_text("".join(lines), encoding="utf-8")


def plot_boundary_summary(all_results: dict[float, list[CellResult]], outdir: Path) -> None:
    try:
        import matplotlib.pyplot as plt
        import matplotlib.cm as cm
    except ImportError:
        return

    wd_list = sorted(all_results.keys())
    colors = cm.coolwarm(np.linspace(0, 1, len(wd_list)))

    metrics = [
        ("dec_norm",  "Decoder Weight Norm"),
        ("eff_dim",   "Effective Dim e^S"),
        ("gen_gap",   "Gen Gap (test−train NLL)"),
        ("test_acc",  "Test Accuracy"),
        ("fourier_r2","Fourier Alignment F(t)"),
        ("emb_norm",  "Embedding Norm"),
    ]

    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    fig.suptitle(f"Stage 2: Boundary Grid  lr={FIXED_LR:.2e}", fontsize=11)

    for ax, (attr, title) in zip(axes.flatten(), metrics):
        for wd, color in zip(wd_list, colors):
            for r in all_results[wd]:
                vals = getattr(r, attr, [])
                lbl = f"wd={wd}" if r.seed == 42 else None
                ax.plot(r.steps, vals, color=color, lw=1.0, alpha=0.55, label=lbl)
        ax.set_title(title)
        ax.set_xlabel("step")
        ax.grid(True, alpha=0.3)
        if attr == "test_acc":
            ax.axhline(0.9, ls="--", color="gray", lw=0.8)

    handles, labels = axes.flatten()[3].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=6,
               fontsize=7, bbox_to_anchor=(0.5, -0.02))
    fig.tight_layout(rect=[0, 0.04, 1, 1])
    fig.savefig(outdir / "boundary_summary.png", dpi=130, bbox_inches="tight")
    plt.close(fig)

    # Phase boundary heatmap — dec_norm final vs wd
    fig2, ax2 = plt.subplots(figsize=(7, 4))
    for wd in wd_list:
        results = all_results[wd]
        dec_vals = [r.dec_norm[-1] for r in results if r.dec_norm]
        phases = [r.observed_phase for r in results]
        majority = max(set(phases), key=phases.count)
        color = "tab:blue" if "Grokking" in majority else "tab:orange"
        ax2.scatter([wd] * len(dec_vals), dec_vals, color=color, s=60, zorder=3)
        ax2.errorbar(wd, np.mean(dec_vals), yerr=np.std(dec_vals),
                     fmt="none", color=color, capsize=4)
    ax2.set_xscale("log")
    ax2.set_xlabel("wd (log scale)")
    ax2.set_ylabel("dec_norm (final)")
    ax2.set_title("Phase Boundary: dec_norm vs wd  (blue=Grokking, orange=Memorization)")
    ax2.grid(True, alpha=0.3)
    fig2.tight_layout()
    fig2.savefig(outdir / "phase_boundary.png", dpi=130, bbox_inches="tight")
    plt.close(fig2)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--wd-grid",    nargs="+", type=float, default=DEFAULT_WD_GRID)
    p.add_argument("--wd-extra",   nargs="+", type=float, default=[],
                   help="追加额外 wd 值（例如精细化边界）")
    p.add_argument("--seeds",      nargs="+", type=int,   default=DEFAULT_SEEDS)
    p.add_argument("--base-steps", type=int, default=BASE_STEPS)
    p.add_argument("--log-interval", type=int, default=100)
    p.add_argument("--outdir",     default="runs/stage2_boundary")
    p.add_argument("--train-fraction", type=float, default=0.3)
    return p.parse_args()


def main():
    global BASE_STEPS, EXTEND1_STEPS, EXTEND2_STEPS
    args = parse_args()
    BASE_STEPS    = args.base_steps
    EXTEND1_STEPS = int(args.base_steps * 5 / 3)   # ~1.67× base（30k→50k）
    EXTEND2_STEPS = int(args.base_steps * 8 / 3)   # ~2.67× base（30k→80k）

    wd_values = sorted(set(args.wd_grid + args.wd_extra))
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    all_results: dict[float, list[CellResult]] = {}
    total = len(wd_values) * len(args.seeds)
    run_idx = 0

    for wd in wd_values:
        wd_results = []
        for seed in args.seeds:
            run_idx += 1
            print(f"\n[{run_idx}/{total}] lr={FIXED_LR:.2e} wd={wd:.4f} seed={seed} "
                  f"(base={BASE_STEPS})", flush=True)

            r, ext_log = run_cell_adaptive(
                wd=wd, seed=seed,
                log_interval=args.log_interval,
                train_fraction=args.train_fraction,
            )
            used_steps = r.steps[-1] if r.steps else 0
            print(f"  → phase={r.observed_phase}  τ_gen={r.tau_gen}  τ_F={r.tau_fourier}"
                  f"  τ_eS={r.tau_es}  dec_norm={r.dec_norm[-1]:.2f}"
                  f"  gap_ratio={r.gen_gap[-1]/max(abs(r.gen_gap[0]),1e-6):.3f}"
                  f"  steps_used={used_steps}  t={r.runtime_sec:.0f}s", flush=True)
            if ext_log:
                for msg in ext_log:
                    print(f"  [延长] {msg}", flush=True)

            cell_dir = outdir / f"cell_wd{wd:.4f}_seed{seed}"
            cell_dir.mkdir(exist_ok=True)
            with open(cell_dir / "metrics.json", "w") as f:
                json.dump(asdict(r), f, indent=2)
            plot_cell(r, cell_dir)
            wd_results.append(r)

        all_results[wd] = wd_results

    plot_boundary_summary(all_results, outdir)
    write_report(all_results, outdir)
    print(f"\nAll done. Results in {outdir}/")
    print(f"  boundary_summary.png  — 6 指标轨迹叠加图")
    print(f"  phase_boundary.png    — dec_norm vs wd 相界图")
    print(f"  report.md             — 相区分类 + 判别指标表")


if __name__ == "__main__":
    main()
