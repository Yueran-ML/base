"""
final_spot_test_highlr.py
--------------------------
最后一轮高-lr 证伪点测。共 6 个候选点，跑完即止。

如果这 6 个候选中仍无 Comprehension，研究框架立刻转向
Option B：Grokking vs Memorization 两相对比。

候选选取依据：
- 上一轮点测证明 lr=1.6e-3 时，wd 提高只导致权重崩塌，没有 Comprehension
- Comprehension 需要"梯度信号强 + 足够正则化"的平衡
- 高 lr（5e-2 ~ 2e-1）可以提供强梯度信号，使模型在权重被大量衰减之前快速学到泛化解
- wd = 1~3：比 Grokking 左簇的 wd=2.5 稍低或相当，但 lr 高出一个数量级

结论准则（严格）：
  Comprehension = train@90 存在 AND test@90 存在 AND (test@90 - train@90) < 2000 步
"""

import argparse
import sys
import time
from dataclasses import dataclass

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

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
# 6 个最终候选点（lr × wd 矩阵：3 lr × 2 wd）
# ──────────────────────────────────────────────────────
CANDIDATES = [
    # 名称              lr        wd       注释
    ("highlr_A",    5.0e-2,   1.0,   "高 lr，wd 与 Grokking 左簇相当"),
    ("highlr_B",    5.0e-2,   3.0,   "高 lr，wd 稍高"),
    ("highlr_C",    1.0e-1,   1.0,   "很高 lr，wd 中等"),
    ("highlr_D",    1.0e-1,   3.0,   "很高 lr，wd 稍高"),
    ("highlr_E",    2.0e-1,   0.5,   "极高 lr，低 wd（测试是否存在快速泛化）"),
    ("highlr_F",    2.0e-1,   2.0,   "极高 lr，中等 wd"),
]


@dataclass
class SpotResult:
    name: str
    lr: float
    wd: float
    phase: str
    train_acc_final: float
    test_acc_final: float
    train_acc_90_step: int | None
    test_acc_90_step: int | None
    elapsed_s: float


def classify_phase(train_90, test_90, grok_min_gap=2000) -> str:
    if train_90 is None:
        return "Confusion"
    if test_90 is None:
        return "Memorization"
    return "Comprehension" if (test_90 - train_90) < grok_min_gap else "Grokking"


def run_spot(name, lr, wd, max_steps, seed, device) -> SpotResult:
    t0 = time.time()
    prime = 53
    set_seed(seed)
    x, y = make_dataset(prime, "add", seed)
    train_x, train_y, test_x, test_y = split_dataset(x, y, 0.3)

    model = GrokkingTransformer(
        prime=prime, d_model=256, n_heads=4, n_layers=2, d_ff=1024, dropout=0.0,
    ).to(device)

    loader = DataLoader(
        TensorDataset(train_x, train_y),
        batch_size=train_x.shape[0], shuffle=True,
        generator=torch.Generator().manual_seed(seed),
    )
    criterion = nn.CrossEntropyLoss()
    optimizer = build_optimizer(model, decoder_lr=lr, embed_lr=1e-3, decoder_weight_decay=wd)
    scheduler = build_scheduler(optimizer, warmup_steps=10, lr_schedule="constant",
                                lr_min_ratio=0.05, max_steps=max_steps)

    eval_interval = max(1, max_steps // 200)
    train_90 = test_90 = None
    tr_final = te_final = 0.0

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
                _, tr = evaluate(model, train_x, train_y, device, criterion)
                _, te = evaluate(model, test_x, test_y, device, criterion)
                tr_final, te_final = tr, te
                if tr >= 0.9 and train_90 is None:
                    train_90 = step
                if te >= 0.9 and test_90 is None:
                    test_90 = step
                # 提前退出：两者都已到达 90%
                if train_90 and test_90:
                    step = max_steps
                    break
            if step >= max_steps:
                break

    phase = classify_phase(train_90, test_90)
    return SpotResult(name, lr, wd, phase, tr_final, te_final, train_90, test_90,
                      time.time() - t0)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-steps", type=int, default=30000,
                        help="高 lr 收敛更快，30k 步足够判断")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="auto")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") \
             if args.device == "auto" else torch.device(args.device)

    print("=" * 80)
    print("最终高-lr 证伪点测（共 6 个候选点，跑完即止）")
    print(f"Device: {device} | max_steps: {args.max_steps} | seed: {args.seed}")
    print("=" * 80)
    print(f"{'Name':<14} {'lr':>7} {'wd':>5} | {'Phase':<15} "
          f"{'tr_acc':>6} {'te_acc':>6} {'tr@90':>7} {'te@90':>7} {'sec':>5}")
    print("-" * 80)

    results = []
    for name, lr, wd, note in CANDIDATES:
        print(f"  {name} (lr={lr:.0e}, wd={wd}) — {note}", flush=True)
        r = run_spot(name, lr, wd, args.max_steps, args.seed, device)
        results.append(r)
        tr90 = f"{r.train_acc_90_step:>7}" if r.train_acc_90_step else "     --"
        te90 = f"{r.test_acc_90_step:>7}"  if r.test_acc_90_step  else "     --"
        print(f"  {r.name:<14} {r.lr:>7.0e} {r.wd:>5.1f} | {r.phase:<15} "
              f"{r.train_acc_final:>6.3f} {r.test_acc_final:>6.3f} "
              f"{tr90} {te90} {r.elapsed_s:>5.0f}s")

    print("\n" + "=" * 80)
    comp = [r for r in results if r.phase == "Comprehension"]
    if comp:
        print(f"[PASS] Comprehension 细胞找到 {len(comp)} 个：")
        for r in comp:
            gap = (r.test_acc_90_step - r.train_acc_90_step) if r.test_acc_90_step else None
            print(f"  {r.name}  lr={r.lr:.0e}  wd={r.wd}  "
                  f"tr@90={r.train_acc_90_step}  te@90={r.test_acc_90_step}  gap={gap}步")
        print("\n  -> 将以上坐标加入 Stage 0 pilot cells，执行 Step C。")
    else:
        print("[FINAL] 所有高-lr 候选均无 Comprehension。")
        print("        立刻转向 Option B：Grokking vs Memorization 两相对比框架。")
        print("        Stage 0 pilot cells 重新设计：")
        print("          2 个 Grokking  + 2 个 Memorization（不再包含 Comprehension）")
        print("          Memorization 作为无泛化对照组，检验 tau_ring 是否在 Grokking 中独有。")
    print()


if __name__ == "__main__":
    main()
