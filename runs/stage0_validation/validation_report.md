# Stage 0 Metric Validation Report

**Verdict: FAIL — diagnose before Stage 1**

## Onset Estimates per Run

| Cell | Seed | Phase (expected->observed) | tau_Fourier | tau_gen | tau_ring | Ordering | Censored |
|------|------|--------------------------|-----------|-------|--------|----------|---------|
| comp_A | 42 | Comprehension→Memorization | 25000 | 5000 | — | G<F,R_missing | ok |
| comp_A | 7 | Comprehension→Memorization | 25000 | 5000 | — | G<F,R_missing | ok |
| comp_A | 2025 | Comprehension→Memorization | 24100 | 5000 | — | G<F,R_missing | ok |
| comp_B | 42 | Comprehension→Memorization | — | 5000 | — | censored | ok |
| comp_B | 7 | Comprehension→Memorization | — | 5000 | — | censored | ok |
| comp_B | 2025 | Comprehension→Memorization | 24100 | 5000 | — | G<F,R_missing | ok |
| grok_A | 42 | Grokking→Memorization | — | 8700 | — | censored | ok |
| grok_A | 7 | Grokking→Memorization | — | 5000 | — | censored | ok |
| grok_A | 2025 | Grokking→Memorization | 23300 | 5000 | — | G<F,R_missing | ok |
| grok_B | 42 | Grokking→Grokking | 16700 | 7500 | — | G<F,R_missing | !! |
| grok_B | 7 | Grokking→Grokking | 12600 | 6600 | — | G<F,R_missing | !! |
| grok_B | 2025 | Grokking→Grokking | 12500 | 5000 | — | G<F,R_missing | !! |
| memo_A | 42 | Memorization→Memorization | — | 12700 | — | censored | ok |
| memo_A | 7 | Memorization→Memorization | — | 12900 | — | censored | ok |
| memo_A | 2025 | Memorization→Memorization | — | 22400 | — | censored | ok |
| memo_B | 42 | Memorization→Memorization | — | 6700 | — | censored | ok |
| memo_B | 7 | Memorization→Memorization | — | 6200 | — | censored | ok |
| memo_B | 2025 | Memorization→Memorization | — | 5900 | — | censored | ok |

## Pass/Fail Criteria

### [FAIL] P1_fourier_fires_grokking
*F_corrected > 0.02 in Grokking cells by step 15k, >=2/3 seeds*

### [PASS] P2_ring_low_at_init
*R(t) < 0.15 at step 1 in all 6 cells*

### [FAIL] P3_fourier_onset_stable
*std(tau_Fourier) < 8000 steps within each Grokking cell*

### [FAIL] P4_ordering_pattern
*F<G<R in >=2/3 Grokking seeds; collapsed in >=2/3 Comprehension seeds*


## Runtime
- comp_A seed=42: 283.1s
- comp_A seed=7: 283.4s
- comp_A seed=2025: 283.8s
- comp_B seed=42: 282.4s
- comp_B seed=7: 281.3s
- comp_B seed=2025: 280.2s
- grok_A seed=42: 280.8s
- grok_A seed=7: 291.0s
- grok_A seed=2025: 291.5s
- grok_B seed=42: 281.4s
- grok_B seed=7: 282.9s
- grok_B seed=2025: 284.1s
- memo_A seed=42: 280.7s
- memo_A seed=7: 2268.5s
- memo_A seed=2025: 293.4s
- memo_B seed=42: 315.7s
- memo_B seed=7: 321.7s
- memo_B seed=2025: 317.1s

## Interpretation Guide
- **P1 fail**: F(t) is too noisy or doesn't separate phases. Try computing F on top-10 PCA.
- **P2 fail**: Ring score is non-zero at init — check PCA normalization or metric bug.
- **P3 fail**: tau_Fourier is highly variable across seeds — changepoint estimator is unstable.
  Try wider LOWESS bandwidth or coarser logging interval.
- **P4 fail**: Ordering not as predicted — check if grok_A/grok_B are actually Grokking cells.
  Recheck hyperparameters or run the baseline phase diagram first.