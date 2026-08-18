# W1 - 1-break vs 2-break BIC sanity check

Cells analysed: 8
Preferred by BIC: M1 (1-break) in 1 cells, M2 (2-break) in 7 cells.

Across cells where both bp1_M1 and bp1_M2 exist: median |delta_bp1| = 1500 steps; max |delta_bp1| = 4000 steps.

Interpretation:
- If M1 wins on BIC for the bulk of cells, the 1-break formulation is BIC-justified and not hiding a secondary changepoint.
- If |delta_bp1| stays within the 500-step measurement bin, the early changepoint reported as tau_F is robust under richer models.
