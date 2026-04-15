---
tags: [发现, 次要结果]
---

# 发现：F_only 现象——Fourier 结构不足以保证泛化

**来源实验**: [[步骤_14_Stage0v3结果]] | [[步骤_16_Stage1结果]]
**相关概念**: [[概念_Fourier对齐]] | [[发现_GF排序]]

---

## 定义

**F_only**：tau_F 被检测到（Fourier 对齐发生）但 tau_gen 从未出现（test_acc 始终 < 0.9）。

---

## 数据

**Stage 1：39/75 runs（52%）为 F_only**

| 区域 | F_only 情况 |
|------|------------|
| wd=10（全部 15 runs）| Fourier 结构在 ~8k 步出现，但随后权重崩塌，无法泛化 |
| 低 wd Memorization 细胞 | Fourier 结构作为记忆化的短暂副产品出现 |

**Stage 0：grok_C（lr=6.3e-3, wd=0.63）** — tau_F≈11k，但 50k 步内 tau_gen 从未出现

---

## 科学意义

**Fourier 对齐是泛化的必要条件，但不是充分条件。**

模型可以发展出"正确"的内部表示（嵌入向量在 Fourier 谐波方向上对齐），但计算能力未能迁移到测试集。这说明：
1. Fourier 嵌入结构 ≠ 正确的读出电路（readout circuit）
2. 泛化可能还需要其他结构组件同时到位

---

## 待解决问题

- F_only 细胞中 Fourier 结构的具体形态与 G<F 细胞中一样吗？（方向 3：F_only 细胞几何分析）
- wd=10 的 F_only 是权重崩塌造成的，还是泛化本来就需要更长训练时间？
