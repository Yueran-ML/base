# Final Proposal: "The Ring Is Not the Algorithm"
**Refined from**: Signed-Δ Atlas + Fourier-to-Circle Bridge (Ideas 1+2)
**Date**: 2026-03-26
**Reviewer score (pre-refine)**: 4/10 → target 7–8/10 after refinement

---

## Problem Anchor (3 sentences — frozen, no scope drift)

Prior work shows that grokking transformers eventually form structured, Fourier-like representations, but does not identify which internal event separates delayed generalization from immediate comprehension within the same phase diagram. We ask whether a task-aligned cyclic subspace appears *before* behavioral generalization while the visibly circular embedding geometry emerges only *later*, and whether that event ordering is the defining signature of grokking cells specifically. We test this only in the existing modular-addition transformer sweep, using threshold-free onset estimation and semantics-preserving subspace interventions rather than new tasks, larger models, or broader architecture searches.

---

## Method Thesis (one sentence)

Grokking and comprehension differ not in delay magnitude but in **internal event ordering**: in grokking cells τ_Fourier < τ_gen < τ_ring (the hidden harmonic subspace precedes generalization; the visible ring follows), while in comprehension cells these events collapse — and this ordering predicts phase identity better than e^S alone.

---

## Dominant Contribution

**The visible PCA ring is not the algorithm.** What predicts generalization is the earlier emergence of a task-aligned cyclic harmonic subspace in embedding space. The familiar circular PCA visualization is a *cleanup* artifact that lags the actual functional representation. The lag is specific to grokking cells; in comprehension cells it vanishes. This reframes the existing "ring → generalization" narrative as wrong in the causal direction.

---

## Three Signals to Track (every 100–200 steps)

| Signal | Definition | Role |
|--------|-----------|------|
| **G(t)** | Smoothed test NLL (not accuracy threshold) | Generalization onset τ_gen |
| **F(t)** | Fraction of embedding variance explained by best task-aligned cyclic harmonic pair (sin/cos 2πk/p) in full d-dimensional space | Fourier/cyclic subspace onset τ_Fourier |
| **R(t)** | Procrustes R² between top-2 PCA projection of embeddings and canonical circle indexed by token order modp | Visible ring onset τ_ring |

### Onset Estimator (threshold-free)
For each trajectory:
1. Smooth on log(1+t) with robust LOWESS
2. Fit one-break segmented regression with Huber loss
3. Accept breakpoint τ̂ only if post-break slope persists ≥ 5 evaluation intervals (BIC-selected)
4. Bootstrap across seeds for CIs
5. Report ordering probabilities: P(τ_F < τ_gen < τ_R) per cell

---

## Causal Experiments

### Necessity Test
At a post-generalization checkpoint, decompose E = E_cyc + E_perp where E_cyc is the fitted task-aligned Fourier component (top-2 cyclic harmonic projection).
- **Ablation**: replace E_cyc with E_cyc from a checkpoint before τ_gen; compare test accuracy
- **Controls**: (a) equal-norm random 2D subspace removal, (b) non-task harmonic removal, (c) same-norm Gaussian noise
- Hypothesis: accuracy collapses specifically when E_cyc (not matched controls) is reverted

### Sufficiency Test
At a pre-generalization checkpoint (before τ_gen), transplant the late E_cyc into the early model.
- Resume training for a fixed short budget (10% of original steps)
- Compare: steps-to-generalization with vs. without transplant
- Hypothesis: transplant shortens τ_gen specifically in grokking cells, not in comprehension cells

---

## Key Figures

1. **Phase diagram** (10×10, known) — baseline context
2. **Event-ordering map** — each cell labeled by ordering: {F<G<R (grokking), F≈G≈R (comprehension), other}
3. **Δ heatmap** — τ_ring − τ_gen signed, colored; showing grokking/comprehension contrast
4. **Representative timelines** — G(t), F(t), R(t) on same axis for 4 cells (2 grokking, 2 comprehension)
5. **Causal test plot** — test accuracy vs. ablation condition; steps-to-gen with/without transplant

---

## Closest Prior Work and Differentiation

| Paper | What they do | How we differ |
|-------|-------------|---------------|
| Liu et al. 2022 (2205.10343) | e^S drops at τ_gen; 4-phase diagram | We use threshold-free Procrustes R² and Fourier variance; map event ordering, not just e^S |
| Nanda et al. 2023 (2301.05217) | Identify Fourier circuit at generalization | We track F(t) and R(t) dynamically, compare their ordering across the (lr,wd) grid |
| Musat 2025 (2511.01938) | Prove norm minimization causes circle in 2-layer net | Full transformer; phase-conditioned event ordering; causal subspace intervention |
| He et al. 2026 (2602.16849) | Fourier competition theory in 2-layer net | Full transformer sweep; empirical ordering probability map; not theoretical |
| Yıldırım 2026 (2603.05228) | Architecture interventions to bypass grokking | Hyperparameter-space mapping; not architectural surgery |

---

## Success Criteria

**Strong paper (7–8/10)**: τ_Fourier < τ_gen < τ_ring confirmed in ≥70% of grokking cells with P(ordering) > 0.8; ordering collapses in comprehension cells; event ordering predicts phase identity better than e^S; cyclic-component transplant shortens τ_gen vs. matched controls

**Publishable fallback**: Ordering holds but causal test is inconclusive → framed as "diagnostic progress measure that distinguishes grokking from comprehension" (ICLR workshop)

**Negative result / pivot**: If F(t) and R(t) are always co-incident → the "ring IS the algorithm" story is strengthened; publishable as falsification of the lag hypothesis
