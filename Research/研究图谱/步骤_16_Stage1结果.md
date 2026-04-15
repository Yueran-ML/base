---
tags: [步骤, Stage1]
date: 2026-03-30
---

# 步骤 16：Stage 1 结果分析

**前置步骤**: [[步骤_15_Stage1设计]]
**后续步骤**: [[步骤_17_统一分类器]]

---

## Phase Map（5×5 网格，mode across 3 seeds）

```
          wd=0.04  wd=0.16  wd=0.63  wd=2.5   wd=10
lr=1.0e-3   Memo     Memo     Memo    Grok     Conf
lr=1.6e-3   Memo     Memo     Memo    Comp*    Memo
lr=2.5e-3   Memo     Memo     Memo    Comp*    Memo
lr=4.0e-3   Memo     Grok     Memo    Grok     Conf
lr=6.3e-3   Memo     Memo     Memo    Memo     Conf
```

*注：Comp 标记来自 Stage 1 旧分类器，统一分类器下这些细胞应为 Grokking（见 [[步骤_17_统一分类器]]）

---

## 主要发现：G<F 排序严格局限在 wd=2.5

| lr | Phase | G<F fraction | median Δ |
|----|-------|-------------|---------|
| 1.0×10⁻³ | Grokking | **3/3** | +9,500 步 |
| 1.6×10⁻³ | Grokking | **3/3** | +5,000 步 |
| 2.5×10⁻³ | Grokking | **3/3** | +5,000 步 |
| 4.0×10⁻³ | Grokking | **3/3** | +4,500 步 |
| 6.3×10⁻³ | Memo/Grokking | 0/3 | ~0 |

**12/12 runs 在 wd=2.5 均为 G<F**。其余 63 runs 零例 G<F。→ [[发现_GF排序]]

---

## 次要发现：Δ 随 lr 单调递减

$$\Delta: 9.5k \xrightarrow{lr\uparrow} 5k \xrightarrow{} 5k \xrightarrow{} 4.5k \xrightarrow{} 0k$$

---

## 第三个发现：F_only 广泛存在

**39/75 runs** 有 tau_F 但无 tau_gen → [[发现_F_only现象]]
- wd=10 全部 15 runs：tau_F 在 ~8k，但权重在结构形成前就崩塌
- 低 wd 记忆化细胞：Fourier 结构可作为记忆化的短暂副产品出现

---

## 第四个发现：唯一 F<G 点

lr=4e-3, wd=0.16：Δ=−18,500 步（结构先于泛化 18k 步）。位于 Memorization/Grokking 相界附近，置信度低（3 seeds）。

---

## 注记：Phase 标签不一致

lr=1.6e-3 和 lr=2.5e-3 在 wd=2.5 被 Stage 1 旧分类器标记为 Comprehension，但 Stage 0 标记同一细胞为 Grokking。根本原因是两个脚本的 `_classify_phase` 实现不同。→ [[步骤_17_统一分类器]]
