# W5 - tau_gen threshold sensitivity

Canonical threshold: 0.90 (sustained over 3 consecutive 500-step log points).  delta_T = tau_gen(T) - tau_gen(0.90), in training steps.

| threshold | n cells | median |delta| | max |delta| |
| --- | --- | --- | --- |
| 0.80 | 8 | 1250 | 4000 |
| 0.85 | 8 | 750 | 3000 |
| 0.95 | 8 | 750 | 9500 |

Interpretation: if |delta| stays within the 500-step measurement bin across all four thresholds, the tau_gen=0.90 choice is not load-bearing for the G<F ordering (Delta_tau changes by a single bin at most).
