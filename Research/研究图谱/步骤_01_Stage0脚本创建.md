---
tags: [步骤, Stage0]
date: 2026-03-27
---

# 步骤 01：Stage 0 脚本创建（初始版本）

**前置步骤**: [[步骤_00_研究方向确定]]
**后续步骤**: [[步骤_02_Stage0v1运行]]

---

## 执行内容

创建 `stage0_metric_validation.py`，包含：
- 6 个 pilot cells 定义
- 三个指标计算函数
- 变化点估计器（→ [[概念_BIC变化点估计器]]）
- P1-P4 通过标准（→ [[方法_P1-P4通过标准]]）
- 每细胞 2×2 面板可视化

## 初始 Pilot Cells（错误的）

```python
"grok_A": lr=1e-3,  wd=1.0   # 期望 Grokking
"grok_B": lr=2e-3,  wd=3.0   # 期望 Grokking
"comp_A": lr=1e-2,  wd=0.1   # 期望 Comprehension
"comp_B": lr=5e-3,  wd=0.5   # 期望 Comprehension
"memo_A": lr=1e-3,  wd=0.01  # 期望 Memorization
"memo_B": lr=2e-3,  wd=0.05  # 期望 Memorization
```

**问题**：超参数仅凭 MIT 论文文字描述估计，未经实际相图验证。→ 步骤 2 发现大多数细胞落错相区。

## 初始指标实现

- `compute_procrustes_ring`：Procrustes 相似度（后来发现全部为 0，废弃 → [[步骤_02_Stage0v1运行]]）
- `compute_fourier_alignment`：[[概念_Fourier对齐]]
- `estimate_changepoint`：[[概念_BIC变化点估计器]]

## 同步操作

创建 `grokking_baseline_orig.py` 作为原始基线只读备份。
