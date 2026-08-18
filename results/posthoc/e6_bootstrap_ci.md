# E6 — Bootstrap 95% CIs (n_boot = 1000)

Percentile bootstrap over per-run rows. Δτ = τ_F − τ_gen computed only for rows where both changepoints are detected; proportions are computed over *all* rows in the source.

## Δτ and ordering rates

| source | n | n_grok | n_Δ | median Δτ (steps) | mean Δτ (steps) | P(G<F) | P(F<G) | P(F_only) |
| --- | ---: | ---: | ---: | --- | --- | --- | --- | --- |
| Stage 2 — wd sweep (lr=1.6e-3) | 30 | 30 | 30 | 7750 [6000, 9750] | 8783 [6917, 10750] | 0.97 [0.90, 1.00] | 0.00 [0.00, 0.00] | 0.00 [0.00, 0.00] |
| Stage 3 — lr sweep (wd=2.5) | 30 | 25 | 25 | 4000 [2500, 9500] | 3560 [-1803, 7360] | 0.77 [0.60, 0.90] | 0.07 [0.00, 0.17] | 0.17 [0.03, 0.30] |
| Stage 6 — 2D (lr, wd) grid | 147 | 130 | 126 | 6000 [5000, 6500] | 5317 [3777, 6520] | 0.75 [0.68, 0.82] | 0.09 [0.05, 0.14] | 0.09 [0.05, 0.14] |
| Step 2 — circuit-formation three-way | 60 | 54 | 54 | 6000 [5000, 8500] | 7194 [5786, 8750] | 0.87 [0.78, 0.95] | 0.03 [0.00, 0.08] | 0.10 [0.03, 0.18] |
| E4 — decoupled weight-decay ablation | 21 | 11 | 11 | 4500 [2500, 10000] | 4909 [135, 9592] | 0.43 [0.24, 0.67] | 0.10 [0.00, 0.24] | 0.33 [0.14, 0.52] |

## Timing medians

| source | median τ_gen (steps) | median τ_F (steps) |
| --- | --- | --- |
| Stage 2 — wd sweep (lr=1.6e-3) | 13750 [11500, 15000] | 23500 [20750, 25750] |
| Stage 3 — lr sweep (wd=2.5) | 18000 [11500, 26000] | 21500 [16500, 33000] |
| Stage 6 — 2D (lr, wd) grid | 20250 [17000, 24000] | 28250 [24250, 32000] |
| Step 2 — circuit-formation three-way | 15000 [12500, 17500] | 21750 [19750, 26000] |
| E4 — decoupled weight-decay ablation | 13000 [10000, 13500] | 17000 [15000, 23500] |
