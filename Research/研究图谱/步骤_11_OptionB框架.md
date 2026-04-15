---
tags: [步骤, Stage0, 框架转变]
date: 2026-03-27
---

# 步骤 11：Option B 框架确立 + Stage 0 v2 启动（Step C）

**前置步骤**: [[步骤_10_最终判定OptionB]]
**后续步骤**: [[步骤_12_Stage0v2失败]]

---

## 框架转变

**旧**：三相对比（Grokking / Comprehension / Memorization）
**新**：两相对比（Grokking vs Memorization）

---

## 新研究问题

1. tau_F 和 tau_ring 是否**仅出现于 Grokking 细胞**，在 Memorization 细胞中缺席？
2. 在 Grokking 细胞中，tau_gen 和 tau_F 的相对顺序是什么？
3. Memorization 细胞能否作为结构性指标的"阴性对照"？

---

## 新 ALL_CELLS 定义（已更新到 stage0_metric_validation.py）

```python
# 4 个 Grokking 细胞（来自两个已确认的 Grokking 簇）
"grok_A": lr=1.6e-3, wd=1.0    # 左簇中心
"grok_B": lr=1.6e-3, wd=2.5    # 左簇上边缘
"grok_C": lr=1e-2,   wd=0.16   # 右簇（此时还是错误坐标！）
"grok_D": lr=6.3e-3, wd=0.3    # 右簇（此时还是错误坐标！）

# 2 个 Memorization 细胞（与 Grokking 配对，仅 wd 不同）
"memo_A": lr=1.6e-3, wd=0.04   # 与 grok_A/B 同 lr
"memo_B": lr=1e-2,   wd=0.01   # 与 grok_C 同 lr
```

**配对设计**：grok_A 和 memo_A 共享相同 lr=1.6e-3，仅 wd 不同，排除 lr 的混淆效应。

---

## Stage 0 v2 执行

```bash
python stage0_metric_validation.py --max-steps 30000 --outdir runs/stage0_v2
```

6 cells × 3 seeds = 18 runs，30000 步，预计 60-90 分钟。

---

## 注意

grok_C/D 在此步骤仍使用了**插值坐标**而非 Step A 精确网格点，这是 v2 失败的主因 → [[步骤_12_Stage0v2失败]]
