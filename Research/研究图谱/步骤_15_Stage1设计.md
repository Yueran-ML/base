---
tags: [步骤, Stage1]
date: 2026-03-29
---

# 步骤 15：Stage 1 设计与启动（5×5 粗网格）

**前置步骤**: [[步骤_14_Stage0v3结果]]
**后续步骤**: [[步骤_16_Stage1结果]]

---

## 设计决策

- **去掉 Circle Score**：tau_ring 在 Stage 0 从未出现，节省 ~30% 运行时间
- **log_interval = 500 步**（Stage 0 为 100 步）：100 个 checkpoint/run，足够 BIC 变化点
- **max_steps = 50000**（不变）
- **3 seeds/cell**：与 Stage 0 一致

## 网格定义

```
lr  ∈ [1.0e-3, 1.6e-3, 2.5e-3, 4.0e-3, 6.3e-3]  （5 值，对数间隔）
wd  ∈ [0.04, 0.16, 0.63, 2.5, 10]                  （5 值，对数间隔）
seeds = [42, 7, 2025]
```

25 cells × 3 seeds = 75 runs，预计 8-9 小时。

## 脚本

`stage1_coarse_sweep.py`（当时的本地函数版本，后来统一到 grok_metrics.py → [[步骤_17_统一分类器]]）

## 主要指标

- `tau_gen`：`find_tau_sustained(test_acc, 0.9, n=3)`
- `tau_F`：`estimate_changepoint(fourier_corr)`
- `delta = tau_F - tau_gen`（正值 = G<F）
- `ordering`：G<F / F<G / F_only / G_only / none

## 输出

`runs/stage1_coarse/results.csv` + 3 张热图（phase、delta、timing）
