# Cell-level cluster bootstrap vs row-level bootstrap


Resample unit comparison: row-level treats each (lr, wd, seed) row as i.i.d.; cell-level resamples (lr, wd) cells with replacement, all seeds inside a cell going together.


$n_{\mathrm{boot}} = 1000$, percentile 95% CI.


| source | n_rows | n_cells | P(G<F) row-bs [median, CI] | P(G<F) cluster-bs [median, CI] | median deltatau row-bs [steps, CI] | median deltatau cluster-bs [steps, CI] |
|---|---|---|---|---|---|---|
| stage2_wd | 30 | 10 | 0.967 [0.900, 1.000] | 0.967 [0.900, 1.000] | 7.8 [6.0, 9.8]k | 7.8 [6.5, 9.0]k |
| stage3_lr | 30 | 10 | 0.767 [0.600, 0.900] | 0.767 [0.567, 0.967] | 4.0 [2.5, 9.5]k | 4.0 [2.5, 7.5]k |
| stage4_mul | 1 | 1 | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] | n/ak | n/ak |
| stage5_mul_dlog | 30 | 10 | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] | 8.5 [6.5, 11.2]k | 8.8 [6.5, 11.5]k |
| stage5_p97 | 30 | 10 | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] | 28.0 [26.0, 29.5]k | 28.0 [25.5, 30.5]k |
| stage6_2d | 147 | 49 | 0.748 [0.687, 0.823] | 0.748 [0.660, 0.844] | 6.0 [5.0, 6.5]k | 6.0 [4.5, 6.5]k |
| step2_circuit | 60 | 20 | 0.867 [0.783, 0.950] | 0.867 [0.733, 0.967] | 6.0 [5.0, 8.5]k | 6.0 [5.0, 8.5]k |
| e4 | 21 | 0 | 0.429 [0.238, 0.667] | n/a | 4.5 [2.5, 10.0]k | n/ak |

## Interpretation guide

If `cluster CI width` >> `row CI width` for a given source, the row-level bootstrap was under-estimating uncertainty (positive intra-cell seed correlation). If `cluster <= row`, the row-level bootstrap was either correct or slightly conservative. The **cluster** intervals are the appropriate null distribution for claims about the rate of G<F across hyperparameter cells.
