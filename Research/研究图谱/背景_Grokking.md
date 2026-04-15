---
tags: [背景, 概念]
---

# Grokking 现象

**相关**: [[背景_MIT相图]] | [[概念_Fourier对齐]] | [[发现_GF排序]]

---

## 什么是 Grokking？

Grokking 是深度学习中的**延迟泛化现象**：模型在训练集上很快达到接近完美准确率，但在测试集的泛化能力要过很久才突然出现。这种"先记忆、后理解"的模式由 Power 等人（2022）在模块化算术任务中系统研究。

本研究任务：**模块化加法**——给定整数 a、b，预测 (a + b) mod p，素数 p = 53。

---

## 四个相区

| 相区 | 定义 |
|------|------|
| **Comprehension（理解相）** | train_acc 和 test_acc 几乎同步上升，快速泛化 |
| **Grokking（慢泛化相）** | train_acc 先达到 90%，test_acc 要过很久才达到 90% |
| **Memorization（记忆相）** | train_acc 达到 90%，test_acc 始终无法达到 90% |
| **Confusion（混乱相）** | train_acc 也无法达到 90% |

---

## 相位分类的规范实现

见 [[概念_相位分类器]]。

**规范判据（grok_metrics.py）**：
- tau_train = 训练准确率首次持续 ≥0.9 的步数（sustained n=3）
- tau_gen = 测试准确率首次持续 ≥0.9 的步数（sustained n=3）
- Grokking：tau_gen - tau_train ≥ 2000 步
- Comprehension：tau_gen - tau_train < 2000 步
- Memorization：tau_train 检测到，tau_gen 未检测到
- Confusion：tau_train 未检测到

---

## 本研究的核心问题

三个事件的时序关系：

| 事件 | 符号 | 定义 |
|------|------|------|
| 泛化突破 | tau_gen | test_acc ≥ 0.9 持续 3 个 checkpoint |
| Fourier 结构出现 | tau_F | [[概念_BIC变化点估计器]] 检测到 F(t) 斜率变化 |
| 可见环形出现 | tau_ring | Circle Score ≥ 0.8 持续 3 个 checkpoint |

原始假设：**F < G < R**（Fourier 先于泛化，泛化先于环形）
实际发现：**G < F**（泛化先于 Fourier 对齐）→ [[发现_GF排序]]
