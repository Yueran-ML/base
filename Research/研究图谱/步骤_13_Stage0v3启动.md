---
tags: [步骤, Stage0]
date: 2026-03-28
---

# 步骤 13：Stage 0 v3 启动（三项修复）

**前置步骤**: [[步骤_12_Stage0v2失败]]
**后续步骤**: [[步骤_14_Stage0v3结果]]

---

## 三项修复

### 修复 1：grok_C/D 使用 Step A 精确网格坐标

```python
# v2（错误）
"grok_C": lr=1e-2,   wd=0.16  # 实际 Memorization
"grok_D": lr=6.3e-3, wd=0.3   # 实际 Memorization

# v3（修复后）
"grok_C": lr=6.3e-3, wd=0.63  # Step A 确认 Grokking
"grok_D": lr=2.5e-2, wd=0.16  # Step A 确认 Grokking
```

同步更新 memo_B 配对：`memo_B: lr=6.3e-3, wd=0.01`

### 修复 2：P1 阈值放宽到 25000 步

左簇 grok_A/B 的 tau_F 分布在 12k-25k，放宽到 25k 才能覆盖慢细胞。

### 修复 3：max_steps 增加到 50000 步

30k 步内 tau_ring 从未出现，需要更多步数等待圆形结构形成（最终仍未出现 → [[发现_tau_ring失效]]）。

---

## 执行

```bash
python stage0_metric_validation.py --max-steps 50000 --outdir runs/stage0_v3
```

6 cells × 3 seeds = 18 runs，50000 步，预计 2-2.5 小时。

---

## 完整 v3 ALL_CELLS

```python
ALL_CELLS = {
    "grok_A": dict(lr=1.6e-3, wd=1.0,  expected="Grokking"),
    "grok_B": dict(lr=1.6e-3, wd=2.5,  expected="Grokking"),
    "grok_C": dict(lr=6.3e-3, wd=0.63, expected="Grokking"),
    "grok_D": dict(lr=2.5e-2, wd=0.16, expected="Grokking"),
    "memo_A": dict(lr=1.6e-3, wd=0.04, expected="Memorization"),
    "memo_B": dict(lr=6.3e-3, wd=0.01, expected="Memorization"),
}
```
