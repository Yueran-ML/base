"""
spot_test_comprehension.py
--------------------------
对 8 个高 wd 候选细胞做快速点测，寻找 Comprehension 相区。

候选坐标选取依据：
- Step A 相图显示 Grokking 左簇在 lr~1.6e-3, wd=0.63~2.5
- Step A 相图显示 wd=10 时大多为 Memorization 或 Confusion
- Comprehension 在 MIT 论文中通常出现在高 wd（>10）区域
- wd 上限扩展至 30（符合 CLAUDE.md 规格）

用法：
    python spot_test_comprehension.py
    python spot_test_comprehension.py --max-steps 30000
    python spot_test_comprehension.py --seed 0
"""

import argparse
import sys
import time
from dataclasses import dataclass

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

# 直接导入 baseline 中的组件（不修改原文件）
sys.path.insert(0, ".")
from grokking_baseline import (
    GrokkingTransformer,
    build_optimizer,
    build_scheduler,
    evaluate,
    make_dataset,
    set_seed,
    split_dataset,
)


# ──────────────────────────────────────────────────────
# 候选细胞定义
# ──────────────────────────────────────────────────────
CANDIDATES = [
    # 名称              lr        wd      注释
    ("left_wd12",   1.6e-3,   12.0,  "左簇上方，wd 翻倍于 Grokking"),
    ("left_wd20",   1.6e-3,   20.0,  "左簇高 wd"),
    ("left_wd30",   1.6e-3,   30.0,  "左簇最高 wd"),
    ("mid_lr4_wd10", 4.0e-3,  10.0,  "中间 lr，wd=10"),
    ("mid_lr4_wd20", 4.0e-3,  20.0,  "中间 lr，wd=20"),
    ("right_wd10",  1.0e-2,   10.0,  "右簇上方 wd=10"),
    ("right_wd20",  1.0e-2,   20.0,  "右簇 wd=20"),
    ("fast_wd15",   2.5e-2,   15.0,  "高 lr，wd=15"),
]


@dataclass
class SpotResult:
    name: str
    lr: float
    wd: float
    phase: str
    train_acc_final: float
    test_acc_final: float
    train_acc_90_step: int | None   # 训练准确率首次 >=90% 的步数
    test_acc_90_step: int | None    # 测试准确率首次 >=90% 的步数
    elapsed_s: float


def classify_phase(train_90, test_90, max_steps, grok_min_gap=2000) -> str:
    """与 baseline 保持一致的相区判定逻辑。"""
    if train_90 is None:
        return "Confusion"
    if test_90 is None:
        return "Memorization"
    gap = test_90 - train_90
    if gap >= grok_min_gap:
        return "Grokking"
    return "Comprehension"


def run_spot(name: str, lr: float, wd: float, max_steps: int,
             seed: int, device: torch.device) -> SpotResult:
    t0 = time.time()
    prime = 53

    set_seed(seed)
    x, y = make_dataset(prime, "add", seed)
    train_x, train_y, test_x, test_y = split_dataset(x, y, 0.3)

    model = GrokkingTransformer(
        prime=prime, d_model=256, n_heads=4, n_layers=2, d_ff=1024, dropout=0.0,
    ).to(device)

    batch_size = train_x.shape[0]
    dataset = TensorDataset(train_x, train_y)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True,
                        generator=torch.Generator().manual_seed(seed))
    criterion = nn.CrossEntropyLoss()
    optimizer = build_optimizer(model, decoder_lr=lr, embed_lr=1e-3, decoder_weight_decay=wd)
    scheduler = build_scheduler(optimizer, warmup_steps=10, lr_schedule="constant",
                                lr_min_ratio=0.05, max_steps=max_steps)

    eval_interval = max(1, max_steps // 200)  # 约 200 个检查点
    train_acc_90_step = None
    test_acc_90_step = None
    train_acc_final = 0.0
    test_acc_final = 0.0

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

            if step % eval_interval == 0 or step >= max_steps:
                _, train_acc = evaluate(model, train_x, train_y, device, criterion)
                _, test_acc = evaluate(model, test_x, test_y, device, criterion)
                train_acc_final = train_acc
                test_acc_final = test_acc

                if train_acc >= 0.9 and train_acc_90_step is None:
                    train_acc_90_step = step
                if test_acc >= 0.9 and test_acc_90_step is None:
                    test_acc_90_step = step

                # Comprehension 细胞两者都早期达到 —— 检测到后无需继续
                if train_acc_90_step and test_acc_90_step:
                    # 但要确保记录的是真正早期泛化，gap 很小才提前停
                    gap = test_acc_90_step - train_acc_90_step
                    if gap < 2000:   # Comprehension
                        step = max_steps  # 触发退出
                        break

            if step >= max_steps:
                break
        if step >= max_steps:
            break

    phase = classify_phase(train_acc_90_step, test_acc_90_step, max_steps)

    return SpotResult(
        name=name, lr=lr, wd=wd, phase=phase,
        train_acc_final=train_acc_final,
        test_acc_final=test_acc_final,
        train_acc_90_step=train_acc_90_step,
        test_acc_90_step=test_acc_90_step,
        elapsed_s=time.time() - t0,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-steps", type=int, default=20000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="auto")
    args = parser.parse_args()

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    print(f"Device: {device} | max_steps: {args.max_steps} | seed: {args.seed}")
    print(f"{'Name':<18} {'lr':>8} {'wd':>6} | {'Phase':<15} {'tr_acc':>6} {'te_acc':>6} "
          f"{'tr@90':>7} {'te@90':>7} {'sec':>5}")
    print("-" * 85)

    results = []
    for name, lr, wd, note in CANDIDATES:
        print(f"  running {name} (lr={lr:.1e}, wd={wd:.1f}) ...", flush=True)
        r = run_spot(name, lr, wd, args.max_steps, args.seed, device)
        results.append(r)

        tr90 = f"{r.train_acc_90_step:>7}" if r.train_acc_90_step else "     --"
        te90 = f"{r.test_acc_90_step:>7}"  if r.test_acc_90_step  else "     --"
        print(f"  {r.name:<18} {r.lr:>8.1e} {r.wd:>6.1f} | {r.phase:<15} "
              f"{r.train_acc_final:>6.3f} {r.test_acc_final:>6.3f} "
              f"{tr90} {te90} {r.elapsed_s:>5.0f}s")

    print("\n" + "=" * 85)
    print("SUMMARY — Comprehension candidates (train_90 early AND gap < 2000 steps):")
    comp_found = False
    for r in results:
        if r.phase == "Comprehension":
            print(f"  [COMP] {r.name:<18} lr={r.lr:.1e}  wd={r.wd:.1f}  "
                  f"tr@90={r.train_acc_90_step}  te@90={r.test_acc_90_step}")
            comp_found = True
    if not comp_found:
        print("  None found — all candidates are Memorization/Grokking/Confusion.")
        print("  Try increasing wd further or adjusting lr range.")
    print()


if __name__ == "__main__":
    main()
