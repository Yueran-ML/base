# Project: Generalization Precedes Fourier Alignment in Transformer Grokking

## 研究核心

**研究问题**：在 Transformer grokking（模运算任务）中，token embedding 的 Fourier 几何对齐（τ_F）相对于泛化（τ_gen）的时序关系是什么？

**核心发现**：G<F 在 55 次 Grokking runs 中有 52/55（94.5%）成立，中位数 Δτ = τ_F − τ_gen ≈ 7000 步。结论在 lr 和 wd 两轴上均为宽平台，36 种 detector 配置下 RS=0.94。

**一句话解读**：可见的 Fourier 环形结构是 grokking 计算完成后的"收尾整合"产物，而非泛化的前提条件。

---

## 当前实验完整理解

### 任务与模型

- 任务：$(a+b) \bmod 53$，训练集 30%（843 对），共 2809 对
- 架构：Decoder-only Transformer，d_model=256，4头，2层，FFN=1024，无 dropout
- 优化器：AdamW，10步线性 warmup，梯度裁剪 norm=1.0
- 嵌入层：lr=1e-3，wd=0（固定）；解码器参数进行超参数扫描

### 两个核心指标

**τ_gen（泛化时刻）**
- 每 500 步评估一次 test accuracy
- 连续 3 次 ≥ 0.9 才触发（防止瞬时峰值误报）

**τ_F（Fourier 几何时刻）**
- 计算 F_raw(t)：当前 embedding 矩阵与最优 Fourier 谐波的方差解释率
- Null correction：每 5000 步用 100 次随机置换估计 F_null_p95，令 F_corr = max(0, F_raw − F_null_p95)
- BIC 变点检测：在 log 时间轴上对 F_corr 做 EMA 平滑（α=0.15），用 BIC 选择 1 个变点
- 需满足：BIC 改善 + 变点后斜率 > 总值域的 1%

**Null correction 的意义**：p=53，d=256 时，随机 embedding 已有相当 F_raw（高维随机对齐），不纠正会产生误报

### 实验阶段

| 阶段 | 内容 | 结果 |
|------|------|------|
| Stage 0 | 指标验证，发现 Circle Score 永远不达标 → 研究 pivot | 确认 G<F 方向 |
| Stage 1 | 5×5 粗网格（25 cells × 1 seed） | G<F 初步信号 |
| Stage 2 | lr=1.6e-3，wd∈[1.2,3.5]，10×3 seeds | 29/30 G<F，wd 轴宽平台 |
| Stage 3 | wd=2.5，lr∈[5e-4,8e-3]，10×3 seeds | 23/25 G<F，相变在 lr≈5.9e-3 |
| Sensitivity | 8 cells × 36 detector 配置 | RS=0.94，0/8 ordering flip |

### 关键数值结果

| 实验 | Grokking runs | G<F | F<G | coincident |
|------|--------------|-----|-----|-----------|
| Stage 2（wd 轴） | 30/30 | 29 (97%) | 0 | 1 |
| Stage 3（lr 轴） | 25/30 | 23 (92%) | 2 | 0 |
| **合计** | **55** | **52 (94.5%)** | **2 (3.6%)** | **1 (1.8%)** |

中位数 Δτ ≈ 7000 步，IQR ≈ 4000 步，范围 1000–25500 步。

### 两个 F<G 异常案例

- **lr=6.8e-4, seed=7**：Δτ=−1500步，F_corr 在 τ_gen 时仅 0.015（接近检测下限），wd/lr 比值≈3700（远高于典型的≈1070），处于 Grokking 相区低 lr 边缘
- **lr=4.32e-3, seed=42**：Δτ=−54500步，τ_F=13000 但 τ_gen=67500，max test_acc 仅 0.965，处于 Grokking/Memorization 相界

两个异常均位于 Grokking 相区边界，G<F 在相区内部最为可靠。

### 额外发现

- **Comprehension 相区不存在**：在 p=53, d=256, train_frac=0.3 配置下，仅有 Grokking 和 Memorization/Confusion，Comprehension 相区未出现（推测因训练样本约 840 个，模型容量相对过剩）
- **F_only 现象**：lr≈5.9e-3 处存在 F_only runs（Fourier 几何出现但无泛化），说明 Fourier embedding 结构既不是泛化的充分条件，也不是必要条件

### 三阶段 Grokking 图景（当前理解）

```
(1) 电路形成（0 → τ_circuit）
    Fourier 计算机制在 attention/MLP 层面形成
    ← 来自 Nanda et al. 2023，本实验未直接测量

(2) 泛化突破（τ_gen）
    test accuracy 跳跃至 ≥ 0.9
    ← 本实验直接测量

(3) Embedding 几何整合（τ_gen → τ_F，约 7000 步后）
    Token embedding 的 Fourier 环形几何完成整合
    ← 本实验直接测量
```

**当前论文只直接测量了阶段 (2) 和 (3) 之间的 gap，阶段 (1) 借用了 Nanda 的结论。**

---

## 论文现状评价

### 作为 MIT 授课硕论文

**非常强，属于该层次优秀作品。** 方法严谨、结论清晰、敏感性分析超出预期，写作达到会议论文水准。预期成绩：High Distinction。

### 作为学术投稿

- **Workshop（MechInterp/MINT）**：小改即可尝试
- **ICLR/NeurIPS Findings**：需补多任务实验
- **ICLR/NeurIPS 主会**：当前差距较大，需完整扩展

**核心问题**：发现在 Nanda 2023 的框架下有一定预期性；范围仅限单一任务；三阶段图景的第一阶段借用他人数据。

---

## 后续优化路径（投稿方向）

### 第一优先级：必须做（任何投稿都需要）

#### 1. 多任务实验 ⭐ 最高优先级

**目标**：证明 G<F 是 grokking 的普遍性质，而非单一任务配置的特定现象

推荐新增任务：
- $(a \times b) \bmod 53$：乘法，同样有 Fourier 电路，对比意义最强
- $(a+b) \bmod 97$：不同素数，验证 p 不影响结论
- $(a-b) \bmod 53$：减法，对称性检验

实现方式：代码改动极小（修改数据生成函数），主要是计算时间。每个任务约 60 runs（10 lr 值 × 3 seeds 或 10 wd 值 × 3 seeds）。

**预期收益**：如果 G<F 跨任务成立，论文价值翻倍，从"有趣观察"变成"可复现规律"。
**预计工作量**：2–3 周（含计算时间）

#### 2. 直接测量三阶段完整时序

**目标**：把借来的第一阶段（电路形成）变成自己测量的数据，完整呈现 τ_circuit < τ_gen < τ_F

实现方式：实现 Nanda 的 **Fourier squared loss**（restricted logit loss，只保留 task-relevant 谐波频率的贡献），在每个 checkpoint 计算，用 BIC 变点检测 τ_circuit。可复用现有的 grok_metrics.py 框架。

产出：一张完整的三时序图，全部数据都是自己的。

**预计工作量**：1–2 周实现 + 重新跑实验

#### 3. 写清楚与 Nanda 2023 的区别（写作）

不需要新实验，是写作问题。在 Related Work 和 Introduction 里明确加一段：

> "Nanda et al. 测量的是 Fourier **计算机制**（logit/weight 层面）在泛化前形成；本文测量的是 Fourier **embedding 几何**（token 表示层面）在泛化后整合。这是两个不同层次的不同量。本文首次量化了 embedding-geometry lag（Δτ≈7000步）及其跨超参数的稳健性。"

**预计工作量**：3 天

---

### 第二优先级：显著提升质量

#### 4. 统计分析结构（Δτ / ordering 预测因子）

**目标**：从"Δτ 存在"推进到"Δτ 由什么决定"，同时避免构造性耦合和内生性问题。

**⚠️ 关键约束（已确认）**

- **不能**直接回归 `Δτ ~ τ_gen`：因为 Δτ = τ_F − τ_gen，τ_gen 出现在两边，负相关是数学构造产物，无解释价值
- **不能**做 logistic 回归 G<F vs 非G<F：55 runs 中仅 3 个非G<F，quasi-complete separation，模型不可识别
- **不能**把 τ_train 称为"外生变量"：τ_train 是训练过程内部 outcome，受 wd/lr/seed 共同决定
- **不能**同时放 wd + lr + wd/lr 进同一回归：三者强共线，系数不稳定
- **术语**："within-cell illustrative contrast"，不用"自然实验"（seed 改变整个随机轨迹，非单一处理）

**确定的分析结构（最终版）**

| 位置 | 内容 | 方法 | 定位 |
|------|------|------|------|
| 主文分析 1 | `log τ_F ~ log wd + log lr`（或单独 `log(wd/lr)`，二选一） | OLS 回归 | τ_F 时点随外生超参数系统变化 |
| 主文分析 2 | 按 log(wd/lr) 分三组，比较 τ_F 分布 | Kruskal-Wallis + η²（pairwise: Dunn + Holm） | 非参数佐证 |
| 主文分析 3 | G<F rate = 94.5%，bootstrap 95% CI | 区间估计，不做 logistic | 诚实量化不确定性 |
| 主文 case study | subtraction 同超参数 seed 对比（wd=2.18, seed=42 vs 7） | 描述性 | within-cell contrast，弱化"ordering 纯由超参数决定"的备择解释 |
| 补充分析 | τ_gen 分三档（fast/medium/slow），比较各档 P(G<F) 和 Δτ 分布 | 描述性分层，标注 *descriptive only* | 速度假说图示 |
| Appendix | `log τ_F ~ τ_gen + log wd + log lr` | OLS，标注 *exploratory; τ_gen may be bad control* | 探索性 |

**log wd + log lr vs log(wd/lr) 选择方法**：
先用 Stage 2+3 已有数据跑两个规格的 R²，选拟合更好的作为主规格。若两轴效应方向相反（wd↑晚，lr↑早），用两变量；若效应对称，用单一 ratio。

**τ_F 未检测到的 runs 的处理（必须说明）**：
不能无声删除。二选一：
- 明确说明只在 τ_F detected 子样本上做回归，报告未检测率及其超参数分布
- 将"未检测"视为右删失，使用 Tobit 模型或生存分析框架

**预计工作量**：1 周（数据现成，主要是分析脚本和写作）

#### 5. F_only 区域系统研究

**目标**：F_only（Fourier 几何出现但没有泛化）是一个独立有趣的发现，值得专门分析

具体做法：在 lr≈5.9e-3 附近做更密集的扫描，分析 F_only runs 的轨迹特征（dec_norm、gen_gap 等），与 Grokking runs 对比。

**核心论点**：Fourier embedding 结构既不是泛化的必要条件（G<F 时，泛化发生时 F_corr 接近 0），也不是充分条件（F_only runs 有 Fourier 几何但无泛化）。

**预计工作量**：1 周

#### 6. 2D 超参数网格（可选）

**目标**：完整的相图，回答"G<F 在整个 Grokking 区域都成立吗"

内容：5×5 或 7×7 的 (lr, wd) 网格，每格 3 seeds，产出完整 2D 热图。

**预计工作量**：计算量大（150–300 runs），约 3–4 周，建议最后做

 4. Codex 提出的新优化路径（project.md 未列的）

  ┌──────────────────────────────┬─────────┬───────────────────────────────────────────────────────────────────────┐
  │            新路径            │ 重要性  │                                 说明                                  │
  ├──────────────────────────────┼─────────┼───────────────────────────────────────────────────────────────────────┤
  │ ⑦ 验证 τ_circuit 指标等价性  │ ⭐⭐⭐  │ 在代表性 runs 上对比 flogit changepoint 与 Nanda 的 Fourier squared   │
  │                              │ 最高    │ loss。这是强 claim 的测量等价性漏洞，优先级高于④⑤                     │
  ├──────────────────────────────┼─────────┼───────────────────────────────────────────────────────────────────────┤
  │ ⑧ 删减/降格 speed-dependent  │ ⭐⭐ 高 │ 移入 appendix 或 future work，数学构造成分已被承认，不应与主结论并列  │
  │ hypothesis                   │         │                                                                       │
  ├──────────────────────────────┼─────────┼───────────────────────────────────────────────────────────────────────┤
  │ ⑨ 显式处理未检测/边界样本    │ ⭐⭐ 高 │ F_only/G_only/not detected 的分布要系统报告，不能只报 52/55           │
  ├──────────────────────────────┼─────────┼───────────────────────────────────────────────────────────────────────┤
  │ ⑩ 架构/数据量 robustness     │ ⭐ 中   │ 改一个 train_frac 或 d_model，比继续雕统计更能挡 reviewer             │
  │ check                        │         │ 的普适性质疑                                                          │
  └──────────────────────────────┴─────────┴───────────────────────────────────────────────────────────────────────┘

---

## 工作量总结

```
最小投稿路径（Workshop 级别）：
  ① 多任务实验（+2个任务）          2–3 周
  ② 与 Nanda 区别写清楚（写作）       3 天
  ③ 三阶段直接测量                  1–2 周
  合计：约 1–1.5 个月

完整投稿路径（主会级别）：
  以上 ① ② ③ +
  ④ Δτ 预测因子分析                1 周
  ⑤ F_only 系统研究                1 周
  ⑥ 论文重写（定位、related work）   2 周
  合计：约 2.5–3 个月
```

---

## 代码文件索引

| 文件 | 用途 |
|------|------|
| `grok_metrics.py` | 核心指标库（canonical，所有脚本 import 此文件） |
| `grokking_baseline.py` | 基线训练脚本 |
| `stage0_metric_validation.py` | Stage 0 指标验证 |
| `stage1_coarse_sweep.py` | Stage 1 粗网格扫描 |
| `stage2_wd_sweep.py` | Stage 2 wd 轴细扫 |
| `stage3_lr_sweep.py` | Stage 3 lr 轴细扫 |
| `sensitivity_analysis.py` | 36 种 detector 配置敏感性分析 |
| `make_timeline_figure.py` | 时序图生成 |
| `paper/main.tex` | ICLR 格式论文正文（v2） |
| `paper/refs.bib` | 参考文献（7条已验证） |

## 实验数据

所有实验输出在 `runs/` 目录下，按阶段分子目录：
`stage0_*/`, `stage1_*/`, `stage2_wd/`, `stage3_lr/`, `sensitivity/`
