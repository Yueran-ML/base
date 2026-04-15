---
tags: [步骤, Stage0]
date: 2026-03-27
---

# 步骤 12：Stage 0 v2 失败分析

**前置步骤**: [[步骤_11_OptionB框架]]
**后续步骤**: [[步骤_13_Stage0v3启动]]

---

## 判决：FAIL（P1、P3、P4 均未通过）

| 判据 | 结果 | 说明 |
|------|------|------|
| P1：F > 0.02 by step 15k | **FAIL** | 左簇 tau_F 在 12k-25k，15k 阈值太早 |
| P2：CS 初始 < 0.05 | **PASS** | ✓ |
| P3：std(tau_F) < 8000 | **FAIL** | grok_A seeds 21100/missing/24800，方差极大 |
| P4：Grokking 有 tau_gen+tau_F | **FAIL** | grok_C/D 全部呈 Memorization，没有 tau_gen |

---

## 根本原因

### 原因 1：grok_C/D 坐标错误（主因）

| 细胞 | v2 坐标（错误）| 实际相区 | Step A 正确坐标 |
|------|--------------|---------|----------------|
| grok_C | lr=1e-2, wd=0.16 | **Memorization** | lr=6.3e-3, wd=0.63 |
| grok_D | lr=6.3e-3, wd=0.3 | **Memorization** | lr=2.5e-2, wd=0.16 |

插值点落在 Grokking 区域外，导致 P4 直接失败。

### 原因 2：P1 阈值过早（次因）

左簇细胞（grok_A/B）tau_F 在 12k-25k 步，而 P1 检测阈值为 15k 步。

---

## 关键发现（尽管整体 FAIL）

grok_B 三个 seed 一致显示：

```
tau_gen (10000-13200) < tau_F (12200-14900)
```

这是本实验的**第一个实质性发现**——泛化先于 Fourier 结构对齐。→ [[发现_GF排序]]

---

## 修复方案

→ [[步骤_13_Stage0v3启动]]：三项修复（坐标、P1 阈值、max_steps）
