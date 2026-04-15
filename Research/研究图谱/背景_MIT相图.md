---
tags: [背景, 文献]
---

# MIT 相图（Liu 2022）

**相关**: [[背景_Grokking]] | [[方法_模型架构]] | [[步骤_04_StepA基线相图]]

---

## 核心发现

Liu 等人（2022）发现，改变解码器的学习率（decoder_lr）和权重衰减（decoder_wd），可将模型行为分成四个相区（见 [[背景_Grokking]]）。这四个相区在 (decoder_lr, decoder_wd) 对数坐标网格上形成**相图**。

---

## 本研究与 MIT 论文的区别

| 论文 | 他们做了什么 | 我们的区别 |
|------|-------------|-----------|
| Liu 2022 | 静态相图，仅分类超参数 | 测量各事件的时序，构建"时序地图" |
| Nanda 2023 | 在泛化时刻识别 Fourier 电路 | 动态追踪 F(t)，比较整张相图中的顺序 |
| Musat 2025 | 2 层网络中证明范数最小化导致环形结构 | 完整 Transformer，跨超参数，有因果干预实验 |
| He 2026 | 2 层网络 Fourier 竞争理论 | 经验测量，有顺序概率图 |

---

## 本研究的 Step A 相图

运行 6×6 网格（20000 步），结果：

```
wd=10  Conf  Conf  Conf  Conf  Conf  Conf
wd=2.5 Memo  Grok  Memo  Memo  Memo  Conf
wd=0.63 Memo Memo  Grok  Memo  Memo  Conf
wd=0.16 Memo Memo  Memo  Grok  Memo  Memo
wd=0.04 Memo Memo  Memo  Memo  Memo  Memo
wd=0.01 Memo Memo  Memo  Memo  Memo  Memo
       1e-4  4e-4  1.6e-3 6.3e-3 2.5e-2 1e-1  (lr)
```

**Grokking 区域**：
- 左簇：lr ≈ 1.6e-3，wd = 0.63 ~ 2.5
- 右簇：lr ≈ 6.3e-3 ~ 2.5e-2，wd = 0.04 ~ 0.63

**关键发现**：整张相图中**无 Comprehension 相区** → [[发现_Comprehension缺失]]

---

## 为什么 p=53 没有 Comprehension？

MIT 使用 p=97（训练样本 ≈2800），本研究 p=53（训练样本 ≈840）。训练集更小，模型更容易走记忆路径，Comprehension 的参数窗口极窄甚至不存在。
