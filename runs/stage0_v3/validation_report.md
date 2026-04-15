# Stage 0 Metric Validation Report

**Verdict: FAIL — diagnose before Stage 1**

## Onset Estimates per Run

| Cell | Seed | Phase (expected->observed) | tau_Fourier | tau_gen | tau_ring | Ordering | Censored |
|------|------|--------------------------|-----------|-------|--------|----------|---------|
| grok_A | 42 | Grokking→Grokking | 35500 | 23800 | — | G<F,R_missing | !! |
| grok_A | 7 | Grokking→Grokking | 22700 | 18200 | — | G<F,R_missing | !! |
| grok_A | 2025 | Grokking→Grokking | 28700 | 30000 | — | F<G,R_missing | !! |
| grok_B | 42 | Grokking→Grokking | 21100 | 13200 | — | G<F,R_missing | !! |
| grok_B | 7 | Grokking→Grokking | 12600 | 10000 | — | G<F,R_missing | !! |
| grok_B | 2025 | Grokking→Grokking | 15000 | 12600 | — | G<F,R_missing | !! |
| grok_C | 42 | Grokking→Memorization | 10700 | — | — | censored | ok |
| grok_C | 7 | Grokking→Memorization | 10500 | — | — | censored | ok |
| grok_C | 2025 | Grokking→Memorization | 11700 | — | — | censored | ok |
| grok_D | 42 | Grokking→Grokking | 9500 | — | — | censored | !! |
| grok_D | 7 | Grokking→Grokking | 9600 | 43000 | — | F<G,R_missing | !! |
| grok_D | 2025 | Grokking→Grokking | 9100 | — | — | censored | !! |
| memo_A | 42 | Memorization→Memorization | — | — | — | censored | ok |
| memo_A | 7 | Memorization→Memorization | — | — | — | censored | ok |
| memo_A | 2025 | Memorization→Memorization | 41700 | — | — | censored | ok |
| memo_B | 42 | Memorization→Memorization | — | — | — | censored | ok |
| memo_B | 7 | Memorization→Memorization | 41700 | — | — | censored | ok |
| memo_B | 2025 | Memorization→Memorization | — | — | — | censored | ok |

## Pass/Fail Criteria

### [FAIL] P1_fourier_fires_grokking
*F_corrected > 0.02 in Grokking cells by step 25k, >=2/3 seeds*

### [PASS] P2_ring_low_at_init
*Circle Score < 0.05 at step 1 in all 6 cells (ring not pre-formed)*

### [PASS] P3_fourier_onset_stable
*std(tau_Fourier) < 8000 steps within each Grokking cell*

### [FAIL] P4_ordering_pattern
*Grokking cells: tau_gen+tau_F both present in >=2/3 seeds; Memorization cells: tau_gen absent (never generalizes) in >=2/3 seeds*


## Runtime
- grok_A seed=42: 534.0s
- grok_A seed=7: 526.7s
- grok_A seed=2025: 583.0s
- grok_B seed=42: 594.4s
- grok_B seed=7: 574.7s
- grok_B seed=2025: 516.1s
- grok_C seed=42: 571.5s
- grok_C seed=7: 635.6s
- grok_C seed=2025: 689.6s
- grok_D seed=42: 603.1s
- grok_D seed=7: 551.5s
- grok_D seed=2025: 535.4s
- memo_A seed=42: 531.2s
- memo_A seed=7: 528.9s
- memo_A seed=2025: 529.0s
- memo_B seed=42: 562.8s
- memo_B seed=7: 661.9s
- memo_B seed=2025: 636.2s

## Interpretation Guide
- **P1 fail**: F(t) is too noisy or doesn't separate phases. Try computing F on top-10 PCA.
- **P2 fail**: Circle Score non-negligible at init — model may have loaded wrong checkpoint, or check avg_norm in compute_circle_score.
- **P3 fail**: tau_Fourier is highly variable across seeds — changepoint estimator is unstable.
  Try wider LOWESS bandwidth or coarser logging interval.
- **P4 fail**: Ordering not as predicted — check if grok_A/grok_B are actually Grokking cells.
  Recheck hyperparameters or run the baseline phase diagram first.