---
tags: [步骤, Stage0]
date: 2026-03-27
---

# 步骤 02：Stage 0 v1 运行（有问题的版本）

**前置步骤**: [[步骤_01_Stage0脚本创建]]
**后续步骤**: [[步骤_03_v1初步分析]]

---

## 配置

```bash
python stage0_metric_validation.py --max-steps 30000 --outdir runs/stage0_validation
```

6 cells × 3 seeds = 18 runs，每次 30000 步，GPU RTX 4080 Laptop。

---

## 三个根本问题

### 问题 A：Procrustes 相似度全部为零

所有 18 次 run 中，`procrustes_r2` 始终为 0，包括已 Grokking 的 grok_B。

**原因**：Fourier 结构在 256 维嵌入空间中分散在多个谐波维度，无法集中在 PCA top-2 方向。PCA 投影无法捕捉圆形结构。

**修复**：改用 [[概念_CircleScore]]（Circle Score），直接在全维度空间检验平行四边形条件。→ [[步骤_05_修复脚本]]

### 问题 B：tau_gen 定义错误

变化点估计器作用于测试 NLL 时，在所有细胞约第 5000 步给出 tau_gen，捕捉的是训练初期的快速改善，而非真正的泛化突破（通常在 10000-30000 步）。

**修复**：tau_gen 改为"test_acc ≥ 0.9 持续 3 个 checkpoint"→ [[步骤_05_修复脚本]]

### 问题 C：大多数 Pilot Cells 相区标签错误

通过仔细分析输出日志：
- grok_A：实际 **Memorization**
- grok_B：实际 **Grokking**（唯一正确的）
- comp_A/comp_B：实际 **Grokking 右簇**（原本期望 Comprehension！）
- memo_A/memo_B：Memorization（符合预期）

**根本原因**：超参数仅凭 MIT 论文文字描述估计，未经实际相图验证。

**修复**：运行 6×6 基线相图（Step A）确认实际相区位置 → [[步骤_04_StepA基线相图]]
