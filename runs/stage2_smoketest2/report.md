# Stage 2: Boundary Grid Report

固定 lr=1.60e-03，沿 wd 轴扫描相界

## 相区分类

| wd | seed=42 | seed=7 | seed=2025 | 多数相区 |
|----|---------|--------|-----------|--------|
| 0.010 | Memorization | — | — | **Memorization** |
| 0.020 | Memorization | — | — | **Memorization** |
| 0.040 | Memorization | — | — | **Memorization** |
| 0.250 | Memorization | — | — | **Memorization** |
| 1.000 | Memorization | — | — | **Memorization** |
| 2.500 | Memorization | — | — | **Memorization** |

## 判别指标（中位数 across seeds）

| wd | 相区 | dec_norm | e^S | gap_ratio | F@final | τ_gen |
|----|------|---------|-----|-----------|---------|------|
| 0.010 | Memorization | 17.45 | 46.7 | 1.008 | 0.042 | — |
| 0.020 | Memorization | 17.13 | 46.7 | 1.010 | 0.042 | — |
| 0.040 | Memorization | 16.52 | 46.7 | 1.015 | 0.042 | — |
| 0.250 | Memorization | 12.91 | 46.7 | 1.159 | 0.042 | — |
| 1.000 | Memorization | 14.32 | 46.2 | 1.054 | 0.042 | — |
| 2.500 | Memorization | 14.88 | 45.7 | 0.931 | 0.042 | — |

## 步数使用（延长情况）

| wd | seed | 使用步数 | 运行时间 |
|----|------|---------|--------|
| 0.010 | 42 | 1333 | 27s |
| 0.020 | 42 | 1333 | 27s |
| 0.040 | 42 | 1333 | 29s |
| 0.250 | 42 | 1333 | 19s |
| 1.000 | 42 | 1333 | 18s |
| 2.500 | 42 | 1333 | 18s |
