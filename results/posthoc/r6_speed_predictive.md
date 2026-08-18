# R6 - Predictive test of the speed-dependent ordering hypothesis

Source: `results/stage6_2d/results_stage6_2d.csv` (123 cells with both tau_gen and tau_F detected and |delta_tau| > 500).

Class prior P(G<F) = 0.894; baseline majority-class accuracy = 0.894.

Leave-one-out logistic regression on standardised features.

| feature set | LOO AUC | 95% CI | LOO acc |
| --- | --- | --- | --- |
| log10(lr), log10(wd), log1p(tau_gen) | 0.945 | [0.896, 0.981] | 0.894 |
| log1p(tau_gen) only | 0.925 | -- | -- |
| log10(lr), log10(wd) only | 0.549 | -- | 0.894 |

Brier score (full model) = 0.070.

Interpretation:
- AUC > baseline implies (lr, wd, tau_gen) carry leave-one-out
  predictive signal about whether a held-out grokking cell is
  G<F or F<G; this elevates Section 7's correlation into a
  predictive statement.
- The tau_gen-only ablation isolates the speed contribution; if
  it accounts for most of the AUC, the speed-dependent
  hypothesis is the dominant driver.
