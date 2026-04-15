---
tags: [发现, 方法决策]
---

# 发现：tau_ring 从未检测到——Circle Score 放弃

**来源实验**: [[步骤_12_Stage0v2失败]] | [[步骤_14_Stage0v3结果]] | [[步骤_16_Stage1结果]]
**相关概念**: [[概念_CircleScore]]

---

## 数据

**Stage 0 + Stage 1 共 93 个 run，50,000 步内：**
- Circle Score 从未在任何 run 中持续达到 0.8
- CS 在 Grokking 细胞中有上升趋势，但未成型（不饱和）

---

## 含义

tau_ring（Circle Score ≥ 0.8 持续 3 个 checkpoint）在当前训练预算内**无法被可靠测量**。

---

## 决策

**tau_ring 降级为辅助观察指标**，不再作为主要指标。本研究主要指标改为：
- `tau_gen`：测试准确率 ≥ 0.9（sustained 3 checkpoints）
- `tau_F`：[[概念_BIC变化点估计器]] 检测 Fourier 对齐的起始时刻

这一调整在 Stage 1 已完全落实（stage1_coarse_sweep.py 不再计算 Circle Score）。

---

## 未来方向

如果要重新测量 tau_ring，可能需要：
- 更长训练（100k+ 步）
- 更低的 CS 阈值（< 0.8）
- 更细的 log_interval（< 500 步）

Stage 1 后续计划（方向 3）：100k 步训练，检查 G<F 细胞中 CS 是否最终饱和。
