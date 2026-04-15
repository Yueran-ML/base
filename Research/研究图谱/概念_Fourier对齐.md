---
tags: [概念, 指标]
---

# Fourier 对齐分数 F(t)

**相关**: [[概念_BIC变化点估计器]] | [[发现_GF排序]] | [[步骤_05_修复脚本]]

---

## 定义

嵌入矩阵 E（维度 p × d）在**最佳 Fourier 谐波对**上的方差解释比例（R²），再减去置换零假设的第 95 百分位数。

$$F(t) = \max(0,\ R^2_{\text{best}}(t) - \text{null}_{95}(t))$$

---

## 计算过程

**Step 1：中心化**
$$E \leftarrow E - \text{mean}(E, \text{axis}=0)$$

**Step 2：搜索最优谐波 k = 1, ..., p//2**

对每个 k，构造正交基：
$$B_k = [\cos(2\pi k i/p),\ \sin(2\pi k i/p)],\quad i=0,\ldots,p-1 \quad (p \times 2)$$

QR 分解得正交规范化基 Q，计算投影方差：
$$R^2(k) = 1 - \frac{\|E - Q Q^T E\|^2}{\|E\|^2}$$

取最大 R²：$R^2_{\text{best}} = \max_k R^2(k)$

**Step 3：置换校正（permutation null p95）**

随机打乱词元顺序 100 次，计算 100 个 R² 值的第 95 百分位数作为"随机水平"。

**为什么置换校正？** p=53，d=256，随机嵌入也可能在某方向有偶然高投影。置换校正确保测量的是**结构性**对齐，而非偶然。

**Step 4：修正值**
$$F_{\text{corr}}(t) = \max(0,\ R^2_{\text{best}}(t) - \text{null}_{95}(t))$$

---

## tau_F 的确定

对 $F_{\text{corr}}(t)$ 轨迹使用 [[概念_BIC变化点估计器]]，返回 F(t) 开始持续上升的步数。

---

## 实现位置

`grok_metrics.py` 中的 `compute_fourier_alignment()` 和 `compute_fourier_null_p95()`。

---

## 关键发现

Stage 1 中：
- **G<F 顺序**：tau_gen < tau_F（泛化先于 Fourier 结构对齐）在 wd=2.5 的所有 12 个 run 中成立 → [[发现_GF排序]]
- **F_only**：tau_F 检测到但 tau_gen 未出现（结构存在但不泛化）在 39/75 run 中出现 → [[发现_F_only现象]]
