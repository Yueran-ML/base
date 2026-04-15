---
tags: [步骤, StepA]
date: 2026-03-27
---

# 步骤 06：Step A 结果确认与问题发现

**前置步骤**: [[步骤_05_修复脚本]]
**后续步骤**: [[步骤_07_Comprehension点测]]

---

## Step A 完整结论

**Grokking 区域（已确认）**：
- 左簇：lr = 1.6e-3，wd = 0.63 和 2.5
- 右簇：lr = 6.3e-3 ~ 2.5e-2，wd = 4e-2 ~ 6.3e-1

**Comprehension 区域：完全未出现** → [[发现_Comprehension缺失]]

---

## Stage 0 超参数诊断

- grok_A、memo_A/B：在 Memorization 区（符合预期）
- grok_B：在 Grokking 左簇边界（基本符合）
- **comp_A（lr=1e-2, wd=0.1）**：实际在 **Grokking 右簇**（原期望 Comprehension！）
- **comp_B（lr=5e-3, wd=0.5）**：实际在 **Grokking 右簇**（原期望 Comprehension！）

**含义**：Stage 0 v1 中标为"Memorization"的 comp_A/comp_B，给更多步数（>30000 步）可能会 Grokk。实际上它们是 Grokking 细胞，只是在 30000 步内还没来得及 Grokk。

---

## 待解决问题

1. 整张相图没有 Comprehension → 需要搜索更高 wd：[[步骤_07_Comprehension点测]]
2. Stage 0 需要重新选取正确坐标的细胞：[[步骤_11_OptionB框架]]
