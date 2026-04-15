---
tags: [步骤, Comprehension搜索]
date: 2026-03-27
---

# 步骤 07：Comprehension 细胞点测（第一轮高-wd）

**前置步骤**: [[步骤_06_StepA结果确认]]
**后续步骤**: [[步骤_08_高wd点测结果]]

---

## 背景

Step A 的 wd 上限（10）不够，无法找到 Comprehension 细胞。根据理论，Comprehension 需要"高 wd"：强正则化迫使模型从一开始就走泛化路径，而非记忆路径。

策略：先做**点测**（而非重跑整张相图），更高效。

---

## 脚本

`spot_test_comprehension.py`，30000 步，seed=42，8 个候选点。

---

## 8 个候选细胞

| 候选名 | lr | wd | 选取依据 |
|--------|----|----|----------|
| left_wd12 | 1.6e-3 | 12 | Grokking 左簇正上方 |
| left_wd20 | 1.6e-3 | 20 | 继续加大 wd |
| left_wd30 | 1.6e-3 | 30 | wd 最大值（CLAUDE.md 规格）|
| mid_lr4_wd10 | 4e-3 | 10 | 左右两簇之间 |
| mid_lr4_wd20 | 4e-3 | 20 | 中间 lr + 高 wd |
| right_wd10 | 1e-2 | 10 | 右簇正上方 |
| right_wd20 | 1e-2 | 20 | 右簇 + 高 wd |
| fast_wd15 | 2.5e-2 | 15 | 高 lr 区域 |

---

## 理论预测

当 wd 足够大时，L2 正则化对权重约束强，Memorization 解的代价太高，模型被迫寻找泛化解 → Comprehension。

**结果见**：[[步骤_08_高wd点测结果]]
