---
tags: [发现, 附加结果]
---

# 发现：p=53 配置下无 Comprehension 相区

**来源实验**: [[步骤_04_StepA基线相图]] → [[步骤_08_高wd点测结果]] → [[步骤_10_最终判定OptionB]]
**相关背景**: [[背景_Grokking]] | [[背景_MIT相图]]

---

## 搜索范围

完整测试范围：
- lr ∈ [1e-4, 2e-1]（跨越 3 个数量级）
- wd ∈ [1e-2, 30]（跨越 3 个数量级）
- 14 个候选点穷举测试 + 6×6 基线相图

**结果：所有候选点均为 Memorization 或 Confusion，无 Comprehension。**

---

## 相区结构（lr ≈ 1.6e-3 截面）

```
wd = 0.01~0.04  → Memorization（弱正则，直接记忆）
wd = 0.63~2.5   → Grokking（延迟泛化）
wd = 12          → 权重崩塌区（lr×wd 乘积过大）
wd ≥ 20          → Confusion（过强正则，无法学习）
```

Grokking 和 Confusion 之间**直接跳过 Comprehension**，没有过渡带。

---

## 推测原因

| 因素 | MIT（p=97）| 本研究（p=53）|
|------|-----------|-------------|
| 训练样本 | ~2800 | ~840 |
| 模型记忆难度 | 较难 | 较易 |
| Comprehension 参数窗口 | 宽 | 极窄或不存在 |

p=53 的训练集更小（840 vs 2800），模型容量相对过大，几乎总选记忆路径。

---

## 对研究的影响

研究转为**两相对比**（Grokking vs Memorization）框架，见 [[步骤_11_OptionB框架]]。Comprehension 缺失本身将作为**附加结果**在论文中报告：对 p 的依赖性分析有助于理解 Comprehension 相的成因。
