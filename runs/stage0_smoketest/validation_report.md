# Stage 0 Metric Validation Report

**Verdict: FAIL — diagnose before Stage 1**

## Onset Estimates per Run

| Cell | Seed | Phase (expected->observed) | tau_Fourier | tau_gen | tau_ring | Ordering | Censored |
|------|------|--------------------------|-----------|-------|--------|----------|---------|
| grok_A | 42 | Grokking→Memorization | — | — | — | censored | ok |

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
- grok_A seed=42: 5.9s

## Interpretation Guide
- **P1 fail**: F(t) is too noisy or doesn't separate phases. Try computing F on top-10 PCA.
- **P2 fail**: Ring score is non-zero at init — check PCA normalization or metric bug.
- **P3 fail**: tau_Fourier is highly variable across seeds — changepoint estimator is unstable.
  Try wider LOWESS bandwidth or coarser logging interval.
- **P4 fail**: Ordering not as predicted — check if grok_A/grok_B are actually Grokking cells.
  Recheck hyperparameters or run the baseline phase diagram first.