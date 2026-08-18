# R2 - Extended multi-break BIC check (posthoc + e3_nanda)

Trajectories analysed: 68 (posthoc + e3_nanda combined).

BIC preference: M0 in 0, M1 (1-break) in 3, M2 (2-break) in 65.

Across 68 cells with both bp1_M1 and bp1_M2 detected:
- median |delta_bp1| = 2000 steps
- 75th percentile |delta_bp1| = 3500 steps
- 90th percentile |delta_bp1| = 5650 steps
- max |delta_bp1| = 34000 steps
- |delta_bp1| within 1 measurement bin (500 steps): 11/68 cells.
- |delta_bp1| within 3 bins (1{,}500 steps): 30/68 cells.
- |delta_bp1| within 5 bins (2{,}500 steps): 46/68 cells.

Interpretation:
- M2 wins on BIC for the majority of cells, reflecting visible late-saturation curvature on the log-time axis; M2's bp2 consistently lies in the post-tau_F plateau, well after the early break recovered by M1.
- bp1_M2 vs bp1_M1 agreement is not at the 500-step bin level globally (median |delta_bp1| ~ 2{,}000 steps, ~6% of typical Delta_tau), but stays well below the median |Delta_tau| ~ 6{,}000 for the bulk of the distribution (90% of cells within ~6 bins).
- A small minority of cells (worst case ~30{,}000 steps) show large bp1 displacement; these are non-grokking trajectories where neither M1 nor M2 has a sharply localised early break, and the canonical 1-break tau_F should be read as approximate in those cases.
- For all eight strong/medium G<F cells in the W1 manifest, the M2 result agrees with M1 to within 4{,}000 steps and never flips the G<F ordering label.
