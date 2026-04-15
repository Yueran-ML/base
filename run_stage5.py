#!/usr/bin/env python3
"""
run_stage5.py — Stage 5 全流程调度脚本
=======================================
按顺序运行两个多任务实验，支持断点续连：
  Stage 5A: (a·b) mod 53，离散对数 Fourier 指标
  Stage 5B: (a+b) mod 97，不同质数

断点续连机制：
  - 每个 cell 完成后立即写入 CSV
  - 重启时读取 CSV，跳过已完成的 (wd, seed) 对
  - 通过 --start-from {5a|5b} 可跳过已完成的阶段

用法：
  python run_stage5.py                    # 全部跑（断点续连）
  python run_stage5.py --start-from 5b   # 跳过 5A 直接跑 5B
  python run_stage5.py --stage 5a        # 只跑 5A
  python run_stage5.py --stage 5b        # 只跑 5B
"""

from __future__ import annotations

import argparse
import csv
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

BASE = Path(__file__).parent
# Auto-detect Python executable: use sys.executable (works on both Windows and Linux)
PYTHON = sys.executable

# ---------------------------------------------------------------------------
# CSV 工具
# ---------------------------------------------------------------------------

def load_csv(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with open(path) as f:
        return list(csv.DictReader(f))


def print_summary(rows: list[dict], label: str) -> None:
    grok = [r for r in rows if r.get("observed_phase") == "Grokking"]
    if not grok:
        from collections import Counter
        phases = Counter(r.get("observed_phase", "?") for r in rows)
        print(f"\n{'='*55}")
        print(f"  {label}: {len(rows)} runs 完成，无 Grokking")
        print(f"  相区分布: {dict(phases)}")
        print(f"{'='*55}")
        return

    n_gf  = sum(1 for r in grok if r.get("ordering") == "G<F")
    n_fg  = sum(1 for r in grok if r.get("ordering") == "F<G")
    n_co  = sum(1 for r in grok if r.get("ordering") == "coincident")
    n_fo  = sum(1 for r in grok if r.get("ordering") == "F_only")
    n_go  = sum(1 for r in grok if r.get("ordering") == "G_only")
    deltas = [float(r["delta"]) for r in grok
              if r.get("delta") not in (None, "", "None")]
    med = int(np.median(deltas)) if deltas else None

    print(f"\n{'='*55}")
    print(f"  {label} 汇总")
    print(f"  总 runs: {len(rows)}  Grokking: {len(grok)}")
    print(f"  G<F: {n_gf}  F<G: {n_fg}  coincident: {n_co}  "
          f"F_only: {n_fo}  G_only: {n_go}")
    if len(grok) > 0:
        rate = 100 * n_gf / len(grok)
        print(f"  G<F 比例: {n_gf}/{len(grok)} = {rate:.1f}%")
    if med is not None:
        print(f"  中位数 Δτ: {med:,} 步")
    print(f"{'='*55}\n")


# ---------------------------------------------------------------------------
# 单阶段运行器（调用子脚本，支持断点续连）
# ---------------------------------------------------------------------------

def run_stage(
    script: str,
    outdir: str,
    csv_name: str,
    label: str,
    max_steps: int,
    wd_values: list[float],
    seeds: list[int],
) -> list[dict]:
    csv_path = BASE / outdir / csv_name
    existing = load_csv(csv_path)
    done_keys = {(float(r["wd"]), int(r["seed"])) for r in existing}
    total = len(wd_values) * len(seeds)
    done_count = len(done_keys)

    print(f"\n{'─'*55}")
    print(f"  {label}")
    print(f"  进度: {done_count}/{total} 已完成，{total-done_count} 待运行")
    print(f"  输出: {csv_path}")
    print(f"{'─'*55}")

    if done_count == total:
        print(f"  → 全部完成，跳过运行")
        print_summary(existing, label)
        return existing

    # 逐格运行
    for wd in wd_values:
        for seed in seeds:
            if (wd, seed) in done_keys:
                continue
            t0 = time.time()
            wd_idx = wd_values.index(wd)
            cmd = [
                PYTHON, script,
                "--outdir", str(BASE / outdir),
                "--wd-idx", str(wd_idx),
                "--seeds", str(seed),
                "--max-steps", str(max_steps),
            ]
            print(f"  [run] wd={wd:.4f} seed={seed} ...", end=" ", flush=True)
            result = subprocess.run(cmd, capture_output=True, text=True)
            elapsed = int(time.time() - t0)
            if result.returncode != 0:
                print(f"FAILED (exit {result.returncode})")
                print(result.stderr[-500:] if result.stderr else "(no stderr)")
            else:
                # 解析最后一行结果
                lines = [l for l in result.stdout.strip().splitlines() if "phase=" in l]
                summary = lines[-1] if lines else "（无输出）"
                print(f"done [{elapsed}s]")
                print(f"         {summary}")

            # 更新已完成集合
            done_keys.add((wd, seed))

    # 读取最终 CSV
    final = load_csv(csv_path)
    print_summary(final, label)
    return final


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    wd_values = list(np.round(
        np.logspace(np.log10(1.2), np.log10(3.5), 10), 4
    ).tolist())
    seeds = [42, 7, 2025]

    parser = argparse.ArgumentParser(description="Stage 5 全流程调度")
    parser.add_argument(
        "--start-from", choices=["5a", "5b"], default="5a",
        help="从哪个阶段开始（5a=mul dlog，5b=p97）"
    )
    parser.add_argument(
        "--stage", choices=["5a", "5b"], default=None,
        help="只跑指定阶段（覆盖 --start-from）"
    )
    args = parser.parse_args()

    run_5a = True
    run_5b = True
    if args.stage == "5a":
        run_5b = False
    elif args.stage == "5b":
        run_5a = False
    elif args.start_from == "5b":
        run_5a = False

    t_total = time.time()

    # ── Stage 5A: (a·b) mod 53, dlog metric ─────────────────────────────
    rows_5a = []
    if run_5a:
        rows_5a = run_stage(
            script=str(BASE / "stage5_mul_dlog_sweep.py"),
            outdir="runs/stage5_mul_dlog",
            csv_name="results_stage5_mul_dlog.csv",
            label="Stage 5A — (a·b) mod 53（离散对数 Fourier 指标）",
            max_steps=50_000,
            wd_values=wd_values,
            seeds=seeds,
        )

    # ── Stage 5B: (a+b) mod 97 ──────────────────────────────────────────
    rows_5b = []
    if run_5b:
        rows_5b = run_stage(
            script=str(BASE / "stage5_p97_sweep.py"),
            outdir="runs/stage5_p97",
            csv_name="results_stage5_p97.csv",
            label="Stage 5B — (a+b) mod 97（不同质数）",
            max_steps=80_000,
            wd_values=wd_values,
            seeds=seeds,
        )

    # ── 合并汇总 ─────────────────────────────────────────────────────────
    total_elapsed = int(time.time() - t_total)
    print(f"\n{'='*55}")
    print(f"  Stage 5 全部完成，总耗时 {total_elapsed//3600}h {(total_elapsed%3600)//60}m")

    all_rows = rows_5a + rows_5b
    grok_all = [r for r in all_rows if r.get("observed_phase") == "Grokking"]
    if grok_all:
        n_gf = sum(1 for r in grok_all if r.get("ordering") == "G<F")
        print(f"  合并 Grokking runs: {len(grok_all)}")
        print(f"  合并 G<F: {n_gf}/{len(grok_all)} = {100*n_gf/len(grok_all):.1f}%")
    print(f"{'='*55}\n")


if __name__ == "__main__":
    main()
