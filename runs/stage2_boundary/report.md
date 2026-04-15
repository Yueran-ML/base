# Stage 2: Boundary Grid Report

固定 lr=1.60e-03，沿 wd 轴扫描相界

## 相区分类

| wd | seed=42 | seed=7 | seed=2025 | 多数相区 |
|----|---------|--------|-----------|--------|
| 0.010 | Memorization | Memorization | Memorization | **Memorization** |
| 0.020 | Memorization | Memorization | Memorization | **Memorization** |
| 0.040 | Memorization | Memorization | Memorization | **Memorization** |
| 0.250 | Grokking | Grokking | Grokking | **Grokking** |
| 1.000 | Grokking | Grokking | Grokking | **Grokking** |
| 2.500 | Grokking | Grokking | Grokking | **Grokking** |

**相界估计**：wd ∈ (0.040, 0.250)（多数相区翻转点）

## 判别指标（中位数 across seeds）

| wd | 相区 | dec_norm | e^S | gap_ratio | F@final | τ_gen |
|----|------|---------|-----|-----------|---------|------|
| 0.010 | Memorization | 77.67 | 47.8 | 17.930 | 0.043 | — |
| 0.020 | Memorization | 65.14 | 47.8 | 8.600 | 0.044 | — |
| 0.040 | Memorization | 55.69 | 47.2 | 4.674 | 0.047 | — |
| 0.250 | Grokking | 26.61 | 44.2 | 0.032 | 0.043 | 56700 |
| 1.000 | Grokking | 14.20 | 36.3 | 0.023 | 0.051 | 23800 |
| 2.500 | Grokking | 12.32 | 31.8 | 0.028 | 0.063 | 12600 |

## 步数使用（延长情况）

| wd | seed | 使用步数 | 运行时间 |
|----|------|---------|--------|
| 0.010 | 2025 | 80000 | 8009s |
| 0.010 | 42 | 80000 | 1768s |
| 0.010 | 7 | 80000 | 1698s |
| 0.020 | 2025 | 80000 | 749s |
| 0.020 | 42 | 80000 | 1243s |
| 0.020 | 7 | 80000 | 795s |
| 0.040 | 2025 | 80000 | 893s |
| 0.040 | 42 | 80000 | 799s |
| 0.040 | 7 | 80000 | 826s |
| 0.250 | 2025 | 58700 | 618s |
| 0.250 | 42 | 63900 | 651s |
| 0.250 | 7 | 52200 | 518s |
| 1.000 | 2025 | 32000 | 368s |
| 1.000 | 42 | 25800 | 295s |
| 1.000 | 7 | 20200 | 234s |
| 2.500 | 2025 | 14600 | 163s |
| 2.500 | 42 | 15200 | 181s |
| 2.500 | 7 | 12000 | 155s |
