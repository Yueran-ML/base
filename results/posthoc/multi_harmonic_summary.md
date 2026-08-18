# Multi-harmonic Fourier-alignment recompute

## Embedding multi-harmonic ($\tau_F^{(m)}$) across 8 posthoc cells

| cell | m=1 (canonical) | m=2 | m=3 | m=5 | delta(5-1) [steps] |
|---|---|---|---|---|---|
| lr6.8e4_s7 | 42000 | 42000 | 42000 | 42000 | +0 |
| wd1.52_s42 | 23000 | 28000 | 29000 | 29500 | +6500 |
| wd1.71_s7 | 25500 | 25500 | 27500 | 28500 | +3000 |
| wd2.18_s42 | 26500 | 27500 | 28500 | 29500 | +3000 |
| wd2.45_s7 | 18500 | 17000 | 16000 | 18500 | +0 |
| wd2.76_s42 | 29500 | 30000 | 30000 | 30000 | +500 |
| wd3.11_s2025 | 14000 | 15000 | 16000 | 17500 | +3500 |
| wd3.50_s7 | 23000 | 23000 | 23500 | 24000 | +1000 |

**Median |delta(tau_F^5 - tau_F^1)|** = 2000 steps across 8 cells.
**Max |delta|** = 6500 steps. **Within 1 measurement bin (500 steps)**: 3/8 cells.

## Logit multi-harmonic at causal-probe checkpoints

Per-cell, per-regime top-m logit Fourier R^2 (raw, no null correction):

| cell | regime | step | m=1 | m=2 | m=3 | m=5 |
|---|---|---|---|---|---|---|
| canonical_GF | pre_all | 5000 | 0.035 | 0.053 | 0.061 | 0.073 |
| canonical_GF | between_circ_gen | 10000 | 0.104 | 0.193 | 0.249 | 0.298 |
| canonical_GF | between_gen_F | 18500 | 0.242 | 0.468 | 0.627 | 0.849 |
| canonical_GF | post_all | 30000 | 0.280 | 0.462 | 0.615 | 0.822 |
| coincident | pre_all | 4000 | 0.107 | 0.159 | 0.207 | 0.223 |
| coincident | between_circ_gen | 9000 | 0.187 | 0.356 | 0.520 | 0.666 |
| coincident | between_gen_F | 14500 | 0.201 | 0.376 | 0.540 | 0.801 |
| coincident | post_all | 22500 | 0.178 | 0.351 | 0.510 | 0.761 |
| FG_slow | pre_all | 16000 | 0.016 | 0.029 | 0.042 | 0.062 |
| FG_slow | between_circ_F | 37000 | 0.060 | 0.106 | 0.147 | 0.205 |
| FG_slow | between_F_gen | 43000 | 0.068 | 0.124 | 0.174 | 0.262 |
| FG_slow | post_all | 48000 | 0.088 | 0.172 | 0.247 | 0.373 |
| G_CF_boundary | pre_all | 3000 | 0.122 | 0.233 | 0.246 | 0.268 |
| G_CF_boundary | between_gen_circ | 7000 | 0.240 | 0.409 | 0.535 | 0.723 |
| G_CF_boundary | between_circ_F | 10000 | 0.169 | 0.306 | 0.434 | 0.634 |
| G_CF_boundary | post_all | 13500 | 0.204 | 0.396 | 0.515 | 0.734 |
| strong_CGF | pre_all | 5000 | 0.062 | 0.072 | 0.082 | 0.091 |
| strong_CGF | between_circ_gen | 11000 | 0.196 | 0.337 | 0.453 | 0.651 |
| strong_CGF | between_gen_F | 15500 | 0.185 | 0.316 | 0.447 | 0.667 |
| strong_CGF | post_all | 20500 | 0.143 | 0.261 | 0.378 | 0.600 |
