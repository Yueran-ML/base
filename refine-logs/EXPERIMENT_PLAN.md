# Experiment Plan
**Proposal**: "The Ring Is Not the Algorithm"
**Date**: 2026-03-26
**GPU Budget**: RTX 4080 12GB, ~8–9 hours total

---

## Block 0: Metric Validation Pilot (~30 min, 4 cells)

**Goal**: Verify that F(t), R(t), G(t) all fire and produce clean trajectories before investing in the full sweep.

**Runs**: 4 cells × 2 seeds
- Cell A: known Grokking cell (e.g., decoder_lr=1e-3, decoder_wd=1.0)
- Cell B: known Comprehension cell (e.g., decoder_lr=1e-2, decoder_wd=0.1)
- 2 seeds each (seed=42, seed=7)

**Logging every 100 steps**:
- test NLL (smoothed with window=5) → G(t)
- Fourier alignment: project each of the p=53 token embeddings onto best-fit sin(2πk/p)/cos(2πk/p) for k=1..p/2; report fraction of embedding variance explained by top harmonic pair → F(t)
- Procrustes R²: fit canonical circle [cos(2πi/p), sin(2πi/p)] to top-2 PCA of embedding table → R(t)
- Also log: Circle Score (original parallelogram metric, for comparison with MIT paper's e^S)

**Pass criterion**: In Cell A (Grokking), F(t) rises detectably before test NLL drop. In Cell B (Comprehension), F(t) and G(t) rise together.

**If fail**: Diagnose metric (is F(t) too noisy? Is Procrustes R² scale-invariant?), adjust parameters, re-pilot.

---

## Block 1: 5×5 Coarse Grid (~2 GPU hours)

**Goal**: Get a first event-ordering map before committing to the full 10×10 sweep.

**Runs**: 25 cells × 1 seed (seed=42)
- Coarse (decoder_lr, decoder_wd): 5 values each log-spaced within the known ranges
- 30k steps each, logging every 200 steps

**Compute**: ~5 min/cell × 25 = ~2 hours

**Outputs**:
- Raw trajectories G(t), F(t), R(t) for all 25 cells
- Apply changepoint estimator (segmented regression, BIC) → τ_Fourier, τ_gen, τ_ring for each cell
- Tag each cell with ordering: {F<G<R, F≈G≈R, F<G≈R, other, censored}
- Overlay on known phase diagram

**Decision gate**: If ≥8/25 Grokking cells show F<G<R ordering, proceed to Block 2. Otherwise stop and diagnose.

---

## Block 2: Phase-Boundary Multi-Seed (~2.5 GPU hours)

**Goal**: Establish statistical significance of the ordering at the grokking/comprehension boundary.

**Runs**: 10 cells near grokking/comprehension boundary × 3 seeds each = 30 runs
- Select from Block 1 output: 5 cells on each side of the visible phase boundary
- Seeds: 42, 7, 2025

**Outputs**:
- Ordering probability P(τ_F < τ_gen < τ_R) per cell with bootstrap CIs
- Compare ordering probability to e^S onset timing as predictor of phase identity
- Phase classification performance: can (τ_F − τ_gen, τ_R − τ_gen) predict Grokking vs. Comprehension as a binary classifier?

---

## Block 3: Causal Interventions (~1.5 GPU hours, 2 cells)

**Cell selection**: 1 confirmed Grokking cell (strong F<G<R ordering), 1 Comprehension cell (F≈G≈R).

### Necessity Test (checkpoint analysis, ~30 min per cell)
For each selected cell:
1. Load checkpoint at τ_gen + 1000 steps (ring not yet visible)
2. Decompose E = E_cyc + E_perp via projection onto best-fit harmonic pair
3. Run 4 ablation variants:
   a. Replace E_cyc with E_cyc from τ_gen − 2000 steps (pre-generalization cyclic component)
   b. Remove random equal-norm 2D subspace (control)
   c. Remove non-task harmonic (next best harmonic) (control)
   d. Add same-norm Gaussian noise to E
4. Evaluate test accuracy after each swap (no retraining)

### Sufficiency Test (~45 min per cell)
For each selected cell:
1. Load checkpoint at τ_gen − 3000 steps (pre-generalization)
2. Transplant E_cyc from final checkpoint into current embedding table
3. Freeze all parameters except non-embedding components; train for 2000 steps
4. Compare: steps-to-gen with transplant vs. baseline continuation vs. random-subspace transplant control

---

## Block 4: Full 10×10 Sweep (optional, ~4–5 GPU hours)

Only run if GPU budget allows after Blocks 0–3 (estimated ~6.5 hours total for Blocks 0–3).

**Runs**: 100 cells × 1 seed
- Full (decoder_lr, decoder_wd) grid as specified in CLAUDE.md
- 30k steps each, logging every 200 steps

**Outputs**:
- Final Δ heatmap: τ_ring − τ_gen for all 100 cells
- Event-ordering map at full resolution
- results.csv with columns: cell_id, decoder_lr, decoder_wd, phase, t_gen, t_fourier, t_ring, delta, ordering, CS_trajectory

---

## Metric Implementation Notes

### F(t) — Fourier Alignment Score
```python
def fourier_alignment(embeddings, prime):
    """Fraction of embedding variance explained by best cyclic harmonic pair."""
    # embeddings: (prime, d_model)
    thetas = 2 * np.pi * np.arange(prime) / prime
    best_r2 = 0
    for k in range(1, prime // 2 + 1):
        basis = np.stack([np.cos(k * thetas), np.sin(k * thetas)], axis=1)  # (p, 2)
        proj = basis @ np.linalg.lstsq(basis, embeddings, rcond=None)[0]
        ss_res = np.sum((embeddings - proj)**2)
        ss_tot = np.sum((embeddings - embeddings.mean(0))**2)
        r2 = 1 - ss_res / ss_tot
        best_r2 = max(best_r2, r2)
    return best_r2
```

### R(t) — Procrustes R²
```python
def procrustes_ring_score(embeddings, prime):
    """Procrustes R^2 between top-2 PCA and canonical circle."""
    from sklearn.decomposition import PCA
    pca = PCA(n_components=2)
    coords = pca.fit_transform(embeddings)  # (p, 2)
    # Canonical circle ordered by token index
    thetas = 2 * np.pi * np.arange(prime) / prime
    canonical = np.stack([np.cos(thetas), np.sin(thetas)], axis=1)
    # Procrustes alignment
    from scipy.spatial import procrustes
    _, canonical_aligned, disparity = procrustes(coords, canonical)
    return 1 - disparity  # R^2-like score
```

### Changepoint Estimator
```python
def estimate_onset(times, values, min_persistence=5):
    """BIC-selected one-break segmented regression onset."""
    import pwlf
    smoother = lowess(values, np.log1p(times), frac=0.2)
    smooth_vals = smoother[:, 1]
    # Fit piecewise linear with 1 break
    my_pwlf = pwlf.PiecewiseLinFit(np.log1p(times), smooth_vals)
    breaks = my_pwlf.fit(2)
    breakpoint_t = np.expm1(breaks[1])
    # Validate persistence
    break_idx = np.searchsorted(times, breakpoint_t)
    post_break_slope = np.diff(smooth_vals[break_idx:break_idx+min_persistence]).mean()
    if abs(post_break_slope) < 1e-4:
        return None  # Not persistent
    return breakpoint_t
```

---

## Run Order Summary

| Block | Cells | Seeds | Est. Time | Decision Gate |
|-------|-------|-------|-----------|---------------|
| 0: Pilot | 4 | 2 | 30 min | F(t) fires in Grokking cell |
| 1: 5×5 coarse | 25 | 1 | 2 hr | ≥8/25 Grokking show F<G<R |
| 2: Boundary multi-seed | 10 | 3 | 2.5 hr | P(F<G<R) > 0.7 in Grokking |
| 3: Causal interventions | 2 | — | 1.5 hr | Necessity: control ≠ cyclic ablation |
| 4: Full 10×10 | 100 | 1 | 4–5 hr | Optional; only if budget remains |

**Total (Blocks 0–3)**: ~6.5 GPU hours ✓ within budget
