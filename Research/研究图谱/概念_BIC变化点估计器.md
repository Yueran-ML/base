---
tags: [概念, 算法]
---

# BIC 变化点估计器

**相关**: [[概念_Fourier对齐]] | [[概念_CircleScore]] | [[步骤_05_修复脚本]]

用于从时序轨迹中估计"起始时刻"，替代简单的阈值穿越方法。

---

## 算法流程

**Step 1：平滑**
指数移动平均（EMA，α = 0.15）对轨迹去噪：
$$\tilde{v}_t = 0.15 \cdot v_t + 0.85 \cdot \tilde{v}_{t-1}$$

**Step 2：对数时间尺度**
在 $\log(1 + t)$ 尺度操作，使早期步数密集、后期步数稀疏的采样均匀化。

**Step 3：分段线性拟合**
在内部区间（去掉头尾各 1/6）遍历所有候选折点，对每个候选折点分别拟合折点前后两段线性回归，计算总 RSS。

**Step 4：BIC 选择**
$$\text{BIC}_{\text{break}} = n \log(\text{RSS}/n) + 4 \log(n) \quad \text{（4个参数）}$$
$$\text{BIC}_{\text{nobreak}} = n \log(\text{RSS}_0/n) + 2 \log(n) \quad \text{（2个参数）}$$

仅当 $\text{BIC}_{\text{break}} < \text{BIC}_{\text{nobreak}}$ 时才接受折点。

**Step 5：持续性检验**
- 折点后斜率必须显著：$|\text{slope}_2| > 0.01 \times \text{value\_range}$
- 折点后至少还有 5 个评估点

---

## 为什么不用阈值穿越？

阈值设定是任意的，不同细胞轨迹尺度不同，同一阈值含义不同。BIC 变化点是"无阈值"的，自适应找到斜率持续变化的时刻。

---

## 实现位置

`grok_metrics.py` 中的 `estimate_changepoint()`。

所有脚本（stage0、stage1、stage2）均从此导入，禁止重新定义本地版本。

---

## 注意：tau_gen 不用此方法

tau_gen（测试准确率阈值穿越）使用 `find_tau_sustained()` 而非变化点估计器。早期曾尝试对测试 NLL 使用变化点，结果在 ~5000 步误触发（捕捉到训练初期的快速改善而非真正的泛化突破）。→ [[步骤_02_Stage0v1运行]]
