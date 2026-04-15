#!/usr/bin/env python3
"""
Stage 3: τ Ordering Crossing Scan
===================================
在 wd ∈ (0.04, 0.25) 之间做致密扫描，寻找 Δτ = τ_F − τ_gen 的 crossing。

核心记录：
  Δτ = τ_F − τ_gen  （正：泛化先于 Fourier；负：Fourier 先于泛化）
  gap_ratio          （gen gap 闭合比例）
  dec_norm           （最终解码器权重范数）

输出：
  delta_tau.png      — Δτ vs wd（含 Stage 2 数据）
  gap_ratio.png      — gap_ratio vs wd（含 Stage 2 数据）
  report.md          — 数值表格

Primary wd : [0.06, 0.08, 0.12, 0.16, 0.20]  (--wd-grid 默认)
Optional wd: [0.10, 0.14, 0.18]               (--wd-extra 追加)

步数策略：base=60k，延长至 100k（train_acc>0.9 且 tau_gen=None），最大 100k。
比 Stage 2 更长，因为边界附近 Grokking 可能非常缓慢（wd=0.25 时 τ_gen≈50-62k）。

Usage:
  python stage3_tau_crossing.py
  python stage3_tau_crossing.py --wd-extra 0.10 0.14 0.18
  python stage3_tau_crossing.py --seeds 42 --base-steps 500 --log-interval 200  # smoketest
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict
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
)
from stage2_boundary_grid import compute_discriminant_metrics

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

FIXED_LR = 1.6e-3
DEFAULT_WD_PRIMARY  = [0.06, 0.08, 0.12, 0.16, 0.20]
DEFAULT_SEEDS       = [42, 7, 2025]
STAGE2_OUTDIR       = Path("runs/stage2_boundary")   # 加载已有数据用于联合作图

# ---------------------------------------------------------------------------
# Training (same as stage2, but with 60k→100k step budget)
# ---------------------------------------------------------------------------

def run_cell(
    wd: float,
    seed: int,
    base_steps: int,
    log_interval: int = 100,
    prime: int = 53,
    train_fraction: float = 0.3,
    embed_lr: float = 1e-3,
    d_model: int = 256,
    n_heads: int = 4,
    n_layers: int = 2,
    d_ff: int = 1024,
) -> tuple[CellResult, list[str]]:
    extend_steps = int(base_steps * 5 / 3)   # 60k → 100k
    early_stop_n = 5

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
    scheduler = build_scheduler(optimizer, warmup_steps=10, lr_schedule="constant",
                                lr_min_ratio=0.05, max_steps=extend_steps)

    result = CellResult(
        cell_name=f"wd{wd:.4f}", lr=FIXED_LR, wd=wd, seed=seed,
        expected_phase="unknown",
    )

    step = 0
    max_steps = base_steps
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

                current_tau_gen = find_tau_sustained(result.steps, result.test_acc, 0.9)

                # 提前停止：tau_gen 已稳定
                if current_tau_gen is not None:
                    tau_gen_confirmed_count += 1
                    if tau_gen_confirmed_count >= early_stop_n and step >= current_tau_gen + 2000:
                        result.observed_phase = "Grokking"
                        max_steps = step
                        break

                # 唯一延长节点
                if step == base_steps and current_tau_gen is None and tr_acc > 0.9:
                    max_steps = extend_steps
                    extension_log.append(
                        f"step={step}: extend to {extend_steps} "
                        f"(train_acc={tr_acc:.3f}, tau_gen=None)"
                    )

            if step >= max_steps:
                break

    result.observed_phase = result.observed_phase or classify_phase(
        result.steps, result.train_acc, result.test_acc
    )
    result.tau_gen     = find_tau_sustained(result.steps, result.test_acc, 0.9)
    result.tau_fourier = estimate_changepoint(result.steps, result.fourier_r2)
    neg_es = [-v for v in result.eff_dim]
    result.tau_es      = estimate_changepoint(result.steps, neg_es)
    result.runtime_sec = time.time() - t0
    return result, extension_log


# ---------------------------------------------------------------------------
# Load existing Stage 2 data for combined plots
# ---------------------------------------------------------------------------

def load_stage2_results() -> dict[float, list[CellResult]]:
    """读取 runs/stage2_boundary/ 已保存的 metrics.json，返回 {wd: [CellResult...]}"""
    if not STAGE2_OUTDIR.exists():
        return {}
    all_results: dict[float, list[CellResult]] = {}
    for cell_dir in sorted(STAGE2_OUTDIR.glob("cell_wd*_seed*")):
        jf = cell_dir / "metrics.json"
        if not jf.exists():
            continue
        d = json.loads(jf.read_text(encoding="utf-8"))
        wd = d["wd"]
        r = CellResult(**{k: d[k] for k in CellResult.__dataclass_fields__ if k in d})
        all_results.setdefault(wd, []).append(r)
    return all_results


# ---------------------------------------------------------------------------
# Δτ computation
# ---------------------------------------------------------------------------

def compute_delta_tau(r: CellResult) -> Optional[float]:
    """Δτ = τ_F − τ_gen。若任一为 None 则返回 None。"""
    if r.tau_gen is None or r.tau_fourier is None:
        return None
    return r.tau_fourier - r.tau_gen


# ---------------------------------------------------------------------------
# Plots: Δτ vs wd  &  gap_ratio vs wd
# ---------------------------------------------------------------------------

def plot_crossing(
    new_results: dict[float, list[CellResult]],
    stage2_results: dict[float, list[CellResult]],
    outdir: Path,
) -> None:
    try:
        import matplotlib.pyplot as plt
        import matplotlib.patches as mpatches
    except ImportError:
        print("matplotlib 不可用，跳过绘图")
        return

    # 合并所有数据
    combined: dict[float, list[CellResult]] = {}
    for wd, rs in stage2_results.items():
        combined.setdefault(wd, []).extend(rs)
    for wd, rs in new_results.items():
        combined.setdefault(wd, []).extend(rs)

    wd_sorted = sorted(combined.keys())

    # ---- 汇总统计 ----
    delta_tau_med: list[Optional[float]] = []
    delta_tau_lo:  list[Optional[float]] = []
    delta_tau_hi:  list[Optional[float]] = []
    gap_ratio_med: list[float] = []
    gap_ratio_lo:  list[float] = []
    gap_ratio_hi:  list[float] = []
    dec_norm_med:  list[float] = []
    phases:        list[str]   = []

    for wd in wd_sorted:
        rs = combined[wd]
        ph_list = [r.observed_phase for r in rs]
        phases.append(max(set(ph_list), key=ph_list.count))

        dts = [dt for r in rs if (dt := compute_delta_tau(r)) is not None]
        if dts:
            delta_tau_med.append(float(np.median(dts)))
            delta_tau_lo.append(float(np.percentile(dts, 25)))
            delta_tau_hi.append(float(np.percentile(dts, 75)))
        else:
            delta_tau_med.append(None)
            delta_tau_lo.append(None)
            delta_tau_hi.append(None)

        dm_list = [compute_discriminant_metrics(r) for r in rs]
        grs = [d["gap_ratio"] for d in dm_list]
        gap_ratio_med.append(float(np.median(grs)))
        gap_ratio_lo.append(float(np.percentile(grs, 25)))
        gap_ratio_hi.append(float(np.percentile(grs, 75)))
        dec_norm_med.append(float(np.median([d["dec_norm_final"] for d in dm_list])))

    # 颜色：Grokking=蓝，Memorization=橙
    colors = ["tab:blue" if "Grokking" in p else "tab:orange" for p in phases]

    # ========== 图 1: Δτ vs wd ==========
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.axhline(0, color="black", lw=1.0, ls="-", zorder=1, label="Δτ = 0 (crossing)")

    for i, wd in enumerate(wd_sorted):
        med = delta_tau_med[i]
        lo  = delta_tau_lo[i]
        hi  = delta_tau_hi[i]
        c   = colors[i]
        if med is not None:
            ax.scatter(wd, med, color=c, s=70, zorder=5)
            ax.errorbar(wd, med, yerr=[[med - lo], [hi - med]],
                        fmt="none", color=c, capsize=5, lw=1.5)
        else:
            # Memorization（τ_gen=None）：在底部画 × 标记
            ax.scatter(wd, ax.get_ylim()[0] if ax.get_ylim()[0] < -1 else -5000,
                       marker="x", color=c, s=80, zorder=5, linewidths=1.5)

    ax.set_xscale("log")
    ax.set_xlabel("wd (log scale)", fontsize=11)
    ax.set_ylabel("Δτ = τ_F − τ_gen  (steps)", fontsize=11)
    ax.set_title(f"τ Ordering: Δτ vs wd  (lr={FIXED_LR:.2e})\n"
                 "Δτ > 0: generalization first | Δτ < 0: Fourier first", fontsize=10)
    ax.grid(True, alpha=0.3)

    grok_patch = mpatches.Patch(color="tab:blue",   label="Grokking")
    memo_patch = mpatches.Patch(color="tab:orange",  label="Memorization / τ_gen=None")
    ax.legend(handles=[grok_patch, memo_patch], fontsize=9)

    fig.tight_layout()
    fig.savefig(outdir / "delta_tau.png", dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"  delta_tau.png saved")

    # ========== 图 2: gap_ratio vs wd ==========
    fig2, ax2 = plt.subplots(figsize=(8, 5))

    for i, wd in enumerate(wd_sorted):
        c = colors[i]
        med = gap_ratio_med[i]
        lo  = gap_ratio_lo[i]
        hi  = gap_ratio_hi[i]
        ax2.scatter(wd, med, color=c, s=70, zorder=5)
        ax2.errorbar(wd, med, yerr=[[med - lo], [hi - med]],
                     fmt="none", color=c, capsize=5, lw=1.5)

    ax2.set_xscale("log")
    ax2.set_yscale("log")
    ax2.set_xlabel("wd (log scale)", fontsize=11)
    ax2.set_ylabel("gap_ratio = gap_final / gap_init  (log scale)", fontsize=11)
    ax2.set_title(f"Gen Gap Closure: gap_ratio vs wd  (lr={FIXED_LR:.2e})\n"
                  "gap_ratio → 0: generalization gap closed", fontsize=10)
    ax2.axhline(1.0, color="black", lw=1.0, ls="--", alpha=0.5, label="ratio = 1 (no change)")
    ax2.grid(True, alpha=0.3, which="both")
    ax2.legend(handles=[grok_patch, memo_patch,
                        mpatches.Patch(color="none", label="— dashed: ratio=1")],
               fontsize=9)

    fig2.tight_layout()
    fig2.savefig(outdir / "gap_ratio.png", dpi=140, bbox_inches="tight")
    plt.close(fig2)
    print(f"  gap_ratio.png saved")

    # ========== 图 3 (bonus): dec_norm vs wd — 便于与 gap_ratio 对比 ==========
    fig3, ax3 = plt.subplots(figsize=(8, 5))
    for i, wd in enumerate(wd_sorted):
        ax3.scatter(wd, dec_norm_med[i], color=colors[i], s=70, zorder=5)
    ax3.set_xscale("log")
    ax3.set_xlabel("wd (log scale)", fontsize=11)
    ax3.set_ylabel("dec_norm final (median across seeds)", fontsize=11)
    ax3.set_title(f"dec_norm vs wd  (lr={FIXED_LR:.2e})", fontsize=10)
    ax3.grid(True, alpha=0.3)
    ax3.legend(handles=[grok_patch, memo_patch], fontsize=9)
    fig3.tight_layout()
    fig3.savefig(outdir / "dec_norm.png", dpi=140, bbox_inches="tight")
    plt.close(fig3)
    print(f"  dec_norm.png saved")


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

def write_report(
    new_results: dict[float, list[CellResult]],
    stage2_results: dict[float, list[CellResult]],
    outdir: Path,
) -> None:
    combined: dict[float, list[CellResult]] = {}
    for wd, rs in stage2_results.items():
        combined.setdefault(wd, []).extend(rs)
    for wd, rs in new_results.items():
        combined.setdefault(wd, []).extend(rs)

    lines = ["# Stage 3: τ Ordering Crossing Report\n\n"]
    lines.append(f"固定 lr={FIXED_LR:.2e}，扫描 wd 轴寻找 Δτ = τ_F − τ_gen 的 crossing\n\n")
    lines.append("> Stage 2 数据（wd ∈ {0.01, 0.02, 0.04, 0.25, 1.0, 2.5}）已包含在联合图中\n\n")

    lines.append("## 新增 wd 点结果\n\n")
    lines.append("| wd | 来源 | 相区 (多数) | τ_gen (中位) | τ_F (中位) | Δτ (中位) | gap_ratio | dec_norm |\n")
    lines.append("|----|------|------------|------------|-----------|----------|-----------|----------|\n")

    def row(wd: float, rs: list[CellResult], src: str) -> str:
        phases = [r.observed_phase for r in rs]
        majority = max(set(phases), key=phases.count)
        tg_vals = [r.tau_gen for r in rs if r.tau_gen is not None]
        tf_vals = [r.tau_fourier for r in rs if r.tau_fourier is not None]
        dt_vals = [dt for r in rs if (dt := compute_delta_tau(r)) is not None]
        dm_list = [compute_discriminant_metrics(r) for r in rs]

        tg_s  = f"{np.median(tg_vals):.0f}" if tg_vals else "—"
        tf_s  = f"{np.median(tf_vals):.0f}" if tf_vals else "—"
        dt_s  = f"{np.median(dt_vals):+.0f}" if dt_vals else "—"
        gr_s  = f"{np.median([d['gap_ratio'] for d in dm_list]):.4f}"
        dn_s  = f"{np.median([d['dec_norm_final'] for d in dm_list]):.2f}"
        return f"| {wd:.4f} | {src} | **{majority}** | {tg_s} | {tf_s} | {dt_s} | {gr_s} | {dn_s} |\n"

    for wd in sorted(new_results.keys()):
        lines.append(row(wd, new_results[wd], "Stage 3"))

    lines.append("\n## 参考：Stage 2 结果（用于 crossing 定位）\n\n")
    lines.append("| wd | 相区 | τ_gen | τ_F | Δτ | gap_ratio |\n")
    lines.append("|----|------|-------|-----|-----|----------|\n")
    for wd in sorted(stage2_results.keys()):
        rs = stage2_results[wd]
        phases = [r.observed_phase for r in rs]
        majority = max(set(phases), key=phases.count)
        tg_vals = [r.tau_gen for r in rs if r.tau_gen is not None]
        tf_vals = [r.tau_fourier for r in rs if r.tau_fourier is not None]
        dt_vals = [dt for r in rs if (dt := compute_delta_tau(r)) is not None]
        dm_list = [compute_discriminant_metrics(r) for r in rs]
        tg_s = f"{np.median(tg_vals):.0f}" if tg_vals else "—"
        tf_s = f"{np.median(tf_vals):.0f}" if tf_vals else "—"
        dt_s = f"{np.median(dt_vals):+.0f}" if dt_vals else "—"
        gr_s = f"{np.median([d['gap_ratio'] for d in dm_list]):.4f}"
        lines.append(f"| {wd:.4f} | **{majority}** | {tg_s} | {tf_s} | {dt_s} | {gr_s} |\n")

    lines.append("\n## Δτ Crossing 分析\n\n")
    # 找 crossing：Δτ 符号翻转的 wd 区间
    crossing_intervals = []
    wd_sorted = sorted(combined.keys())
    prev_sign = None
    prev_wd = None
    for wd in wd_sorted:
        rs = combined[wd]
        dt_vals = [dt for r in rs if (dt := compute_delta_tau(r)) is not None]
        if not dt_vals:
            prev_sign = None
            prev_wd = wd
            continue
        med = float(np.median(dt_vals))
        sign = 1 if med > 0 else -1
        if prev_sign is not None and sign != prev_sign:
            crossing_intervals.append((prev_wd, wd))
        prev_sign = sign
        prev_wd = wd

    if crossing_intervals:
        for lo, hi in crossing_intervals:
            lines.append(f"- **Δτ crossing detected**: wd ∈ ({lo:.4f}, {hi:.4f})\n")
        lines.append("\n  在此区间内 τ 顺序从「Fourier 先于泛化」翻转为「泛化先于 Fourier」。\n\n")
    else:
        lines.append("- crossing 未检测到（可能在当前 wd 范围之外，或数据不足）\n\n")

    lines.append("## 步数使用\n\n")
    lines.append("| wd | seed | 步数 | 运行时间 | 相区 |\n")
    lines.append("|----|------|------|---------|------|\n")
    for wd in sorted(new_results.keys()):
        for r in new_results[wd]:
            lines.append(
                f"| {wd:.4f} | {r.seed} | {r.steps[-1] if r.steps else 0} "
                f"| {r.runtime_sec:.0f}s | {r.observed_phase} |\n"
            )

    (outdir / "report.md").write_text("".join(lines), encoding="utf-8")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--wd-grid",      nargs="+", type=float, default=DEFAULT_WD_PRIMARY,
                   help="Primary wd values (default: 0.06 0.08 0.12 0.16 0.20)")
    p.add_argument("--wd-extra",     nargs="+", type=float, default=[],
                   help="Optional extra wd values (e.g. 0.10 0.14 0.18)")
    p.add_argument("--seeds",        nargs="+", type=int,   default=DEFAULT_SEEDS)
    p.add_argument("--base-steps",   type=int,  default=60_000)
    p.add_argument("--log-interval", type=int,  default=100)
    p.add_argument("--outdir",       default="runs/stage3_tau_crossing")
    p.add_argument("--no-stage2",    action="store_true",
                   help="不加载 Stage 2 数据（仅用于调试）")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    wd_values = sorted(set(args.wd_grid + args.wd_extra))
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    stage2_results = {} if args.no_stage2 else load_stage2_results()
    if stage2_results:
        print(f"  加载 Stage 2 数据：{len(stage2_results)} 个 wd 值")
    else:
        print("  未找到 Stage 2 数据，联合图仅含新数据")

    new_results: dict[float, list[CellResult]] = {}
    total = len(wd_values) * len(args.seeds)
    run_idx = 0

    for wd in wd_values:
        wd_results = []
        for seed in args.seeds:
            run_idx += 1
            print(f"\n[{run_idx}/{total}] lr={FIXED_LR:.2e} wd={wd:.4f} seed={seed} "
                  f"(base={args.base_steps})", flush=True)

            r, ext_log = run_cell(
                wd=wd, seed=seed,
                base_steps=args.base_steps,
                log_interval=args.log_interval,
            )

            for msg in ext_log:
                print(f"  [延长] {msg}", flush=True)

            dt = compute_delta_tau(r)
            dt_str = f"{dt:+.0f}" if dt is not None else "None"
            dm = compute_discriminant_metrics(r)
            print(
                f"  phase={r.observed_phase}  τ_gen={r.tau_gen}  τ_F={r.tau_fourier}"
                f"  Δτ={dt_str}  dec_norm={dm['dec_norm_final']:.2f}"
                f"  gap_ratio={dm['gap_ratio']:.3f}  steps_used={r.steps[-1]}  t={r.runtime_sec:.0f}s",
                flush=True,
            )

            # 保存 per-cell JSON
            cell_dir = outdir / f"cell_wd{wd:.4f}_seed{seed}"
            cell_dir.mkdir(exist_ok=True)
            (cell_dir / "metrics.json").write_text(
                json.dumps(asdict(r), indent=2), encoding="utf-8"
            )

            wd_results.append(r)

        new_results[wd] = wd_results

    # 生成图和报告
    print("\nGenerating plots and report...", flush=True)
    plot_crossing(new_results, stage2_results, outdir)
    write_report(new_results, stage2_results, outdir)

    print(f"\nAll done. Results in {outdir}/")
    print(f"  delta_tau.png   — Δτ vs wd（crossing 图）")
    print(f"  gap_ratio.png   — gap_ratio vs wd")
    print(f"  dec_norm.png    — dec_norm vs wd")
    print(f"  report.md       — 数值表格 + crossing 分析")


if __name__ == "__main__":
    main()
