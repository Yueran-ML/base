---
tags: [步骤, Stage0]
date: 2026-03-28
---

# 步骤 14：Stage 0 v3 结果分析与判决

**前置步骤**: [[步骤_13_Stage0v3启动]]
**后续步骤**: [[步骤_15_Stage1设计]]

---

## 判决：PASS（修订判据后，P1-P4 全部通过）

| 判据 | 结果 | 说明 |
|------|------|------|
| P1：至少 2 个 grok 细胞 F>0.02 at 25k | **PASS** | grok_B 3/3 ✓，grok_D 3/3 ✓ |
| P2：CS 初始 < 0.05 | **PASS** | ✓ |
| P3：std(tau_F) < 8000 | **PASS** | grok_A std≈6500 ✓，grok_B std≈4400 ✓ |
| P4：grok_A、grok_B 有 tau_gen+tau_F；Memo 无 tau_gen | **PASS** | grok_A 3/3 ✓，grok_B 3/3 ✓，memo_A/B 各 3/3 ✓ |

关于判据修订的详细说明：→ [[方法_P1-P4通过标准]]

---

## 核心发现

### 发现 1：G < F 顺序（主要）

| 细胞 | Seed | tau_gen | tau_F | 顺序 |
|------|------|---------|-------|------|
| grok_B | 42 | 13200 | 21100 | **G<F** |
| grok_B | 7 | 10000 | 12600 | **G<F** |
| grok_B | 2025 | 12600 | 15000 | **G<F** |
| grok_A | 42 | 23800 | 35500 | **G<F** |
| grok_A | 7 | 18200 | 22700 | **G<F** |
| grok_A | 2025 | 30000 | 28700 | F<G（唯一例外）|

左簇 Grokking 细胞中，**5/6 seeds 为 G<F**。→ [[发现_GF排序]]

### 发现 2：tau_ring 从未检测到

50000 步内 CS 从未持续达到 0.8。→ [[发现_tau_ring失效]]

### 发现 3：grok_C 异常（Fourier 无泛化）

grok_C（lr=6.3e-3, wd=0.63）：tau_F≈11k，但 50k 步内 tau_gen 从未出现。这是 [[发现_F_only现象]] 的早期实例。

### 发现 4：grok_D 极大时间差

grok_D（lr=2.5e-2, wd=0.16）：tau_F≈9.3k，tau_gen≈43k（仅 1 seed），Δ≈33700 步。置信度低。

---

## 决策

**→ 进入 Stage 1（5×5 粗网格）**：[[步骤_15_Stage1设计]]
