---
tags: [步骤, StepA]
date: 2026-03-27
---

# 步骤 04：Step A——基线相图扫描（6×6 网格）

**前置步骤**: [[步骤_03_v1初步分析]]
**后续步骤**: [[步骤_05_修复脚本]]

---

## 执行

```bash
python grokking_baseline.py --phase-diagram --phase-grid-size 6 \
  --phase-max-steps 20000 --outdir runs/phase_diagram_stepA --seed 42
```

时间：2026-03-27 16:47 ~ 18:51（约 2 小时）

---

## 结果（参见 runs/phase_diagram_stepA/phase_diagram.png）

**Grokking 区域**（已确认）：
- 左簇：lr ≈ 1.6e-3，wd = 0.63 ~ 2.5
- 右簇：lr ≈ 6.3e-3 ~ 2.5e-2，wd = 0.04 ~ 0.63

**关键发现：整张相图中没有 Comprehension 相区** → [[发现_Comprehension缺失]]

原因：wd 最大值仅 10，未覆盖到 Comprehension 区域（通常在更高 wd）。

---

## 对 Stage 0 Pilot Cells 的影响

| Stage 0 细胞 | 期望相区 | 实际相区 |
|-------------|---------|---------|
| grok_A (lr=1e-3, wd=1.0) | Grokking | **Memorization** |
| grok_B (lr=2e-3, wd=3.0) | Grokking | 接近 Grokking 左簇 |
| comp_A (lr=1e-2, wd=0.1) | Comprehension | **Grokking 右簇！** |
| comp_B (lr=5e-3, wd=0.5) | Comprehension | **Grokking 右簇！** |

→ 必须搜索 Comprehension 细胞：[[步骤_07_Comprehension点测]]
→ 必须重新选取 Stage 0 细胞坐标：[[步骤_11_OptionB框架]]
