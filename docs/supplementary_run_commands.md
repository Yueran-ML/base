# Reproduction commands

All commands assume the supplementary archive is unzipped into a
working directory with `supplementary/` as the current working
directory and that `pip install -r requirements.txt` has been run
inside the chosen Python environment.

Outputs land under `results/` (for sweeps) and
`results/posthoc/` (for post-hoc analyses).

---

## 1. Main 1D sweeps (Stages 2 + 3)

```bash
python src/sweeps/stage2_wd_sweep.py
python src/sweeps/stage3_lr_sweep.py
```

Each writes `results/stage2_wd/results.csv` and
`results/stage3_lr/results.csv`.

## 2. Stage 4 (circuit onset, three-stage ordering)

```bash
python src/sweeps/step2_circuit_sweep.py
python src/analysis/make_step2_figures.py
```

Produces `results/step2_circuit/results.csv` and the Stage 4
scatter / boxplot figures under `figures/`.

## 3. Robustness — Stage 5A, 5B, Stage 6

```bash
python src/sweeps/stage5_mul_dlog_sweep.py
python src/sweeps/stage5_p97_sweep.py
python src/sweeps/stage5a_fpstar_recompute.py
python src/sweeps/stage6_2d_sweep.py
python src/analysis/make_stage6_heatmap.py
```

## 4. Decoupled-WD ablation (E4) and data-subsampling (E5)

```bash
python src/sweeps/e4_decoupled_wd.py
python src/sweeps/e5_data_subsample.py
```

## 5. Slow-grokking falsification sweep

```bash
python src/sweeps/slow_grokking_sweep.py
python src/analysis/slow_grokking_figure.py
python src/analysis/slow_grokking_summary.py
```

## 6. Post-hoc trajectory dump (prereq for Q2/Q3/Q8)

```bash
python src/sweeps/posthoc_trajectory_dump.py --max-steps 50000
```

Writes 8 `results/posthoc/traj_<cell>.npz` files (~5 MB each)
plus `results/posthoc/manifest.csv`.

## 7. Post-hoc analyses (Q2 / Q3 / Q8)

Pure CPU analysis on the `.npz` files above; runs in minutes.

```bash
python src/analysis/q2_null_sensitivity.py
python src/analysis/q3_multiharmonic.py
python src/analysis/q8_embedding_geometry.py
python src/analysis/multi_harmonic_recompute.py
```

## 8. Bootstrap CIs (E6 + cluster bootstrap)

```bash
python src/analysis/e6_bootstrap_ci.py
python src/analysis/cluster_bootstrap.py
```

## 9. E3 Nanda comparison

```bash
python src/sweeps/e3_nanda_sweep.py
python src/analysis/e3_nanda_analysis.py
```

## 10. E7 causal Fourier-subspace intervention

```bash
python src/sweeps/causal_probe_checkpoints.py
python src/analysis/causal_fourier_intervention.py
python src/analysis/causal_fourier_summary.py
python src/analysis/causal_fourier_figure.py
```

## 11. Detector and post-hoc robustness panels

```bash
python src/analysis/sensitivity_analysis.py
python src/analysis/sensitivity_figures.py
python src/analysis/raw_vs_corr_analysis.py
python src/analysis/r2_multibreak_extended.py
python src/analysis/w1_multibreak_check.py
python src/analysis/w5_taugen_threshold.py
```

## 12. Speed-ordering predictive analysis

```bash
python src/analysis/r6_speed_ordering_predictive.py
python src/analysis/make_speed_ordering_plot.py
```

## 13. Final timeline figure

```bash
python src/analysis/make_timeline_figure.py
```

---

## Determinism and resource notes

- Each modular-arithmetic training run terminates in
  20--45 minutes at 50 000 steps and 30--75 minutes at 80 000
  steps on the local 4080; cloud runs are correspondingly faster.
- The full set of experiments totals approximately 280 runs and
  ~170 GPU-hours (aggregated across hardware).
- Seeds are `{42, 7, 2025}` per cell, applied uniformly to
  PyTorch, NumPy, and Python `random`. The permutation-null
  estimator uses an independent NumPy generator seeded with
  `run_seed XOR 0xDEAD`.
