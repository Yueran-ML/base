---
tags: [步骤, Comprehension搜索, 转折点]
date: 2026-03-27
---

# 步骤 10：最终高-lr 点测结果 → 判定转 Option B

**前置步骤**: [[步骤_09_高lr点测]]
**后续步骤**: [[步骤_11_OptionB框架]]

---

## 完整结果

| 候选名 | lr | wd | 相区 | tr_acc | te_acc | 说明 |
|--------|----|----|------|--------|--------|------|
| highlr_A | 5e-2 | 1.0 | Memorization | 1.000 | 0.006 | **第 150 步极速过拟合** |
| highlr_B | 5e-2 | 3.0 | Memorization | 0.909 | 0.005 | 极速过拟合 |
| highlr_C | 1e-1 | 1.0 | Memorization | 0.859 | 0.076 | |
| highlr_D | 1e-1 | 3.0 | Memorization | 0.473 | 0.015 | 权重崩塌迹象 |
| highlr_E | 2e-1 | 0.5 | Confusion | 0.425 | 0.003 | |
| highlr_F | 2e-1 | 2.0 | Confusion | 0.124 | 0.048 | |

**关键观察**：高 lr 不但没有带来 Comprehension，反而导致**更快的过拟合**。

---

## 最终结论

在完整测试范围（lr ∈ [1e-4, 2e-1]，wd ∈ [1e-2, 30]，14 个候选点）内：

> **Comprehension 相区不存在于此配置的测试范围内。**

推测原因：p=53 训练样本仅 ~840，模型容量相对过大，几乎总选记忆路径。p=97（MIT 论文）训练样本 ~2800，Comprehension 区间更宽。

→ [[发现_Comprehension缺失]] — 将作为附加结果在论文中报告

---

## 决策：转 Option B（两相对比）

从"三相对比"转为"Grokking vs Memorization 两相对比"。

详见 [[步骤_11_OptionB框架]]。
