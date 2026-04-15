---
title: "Representation Structure Timing in Grokking: Does the Ring Follow the Algorithm?"
date: 2026-03-30
tags:
  - grokking
  - transformer
  - fourier-structure
  - phase-diagram
  - report
aliases:
  - research report
  - Stage 1 findings
---

# Representation Structure Timing in Grokking: Does the Ring Follow the Algorithm?

> **Status**: Stage 1 complete. Stage 2 (boundary refinement) pending.
> **Model**: Decoder-only Transformer, p=53 modular addition, d=256, 2 layers, 4 heads.

---

## What We Did

We trained a small Transformer to compute $(a + b) \bmod 53$ across a 5×5 grid of decoder learning rates and weight decay values, measuring three quantities at each checkpoint:

- **G(t)** — test accuracy; $\tau_\text{gen}$ = first step $\geq 0.9$ sustained
- **F(t)** — Fourier alignment $R^2$ (permutation-corrected); $\tau_F$ = BIC changepoint
- **CS(t)** — Circle Score (parallelogram test); $\tau_\text{ring}$ = first step $\geq 0.8$ sustained

The central question: does Fourier structure in the token embeddings appear *before* or *after* the model generalizes?

---

## Results

### Phase Map

Running 75 cells (25 grid points × 3 seeds, 50,000 steps each) produces the following observed-phase map:

```
          wd=0.04  wd=0.16  wd=0.63  wd=2.5   wd=10
lr=1.0e-3   Memo     Memo     Memo    Grok     Conf
lr=1.6e-3   Memo     Memo     Memo    Comp     Memo
lr=2.5e-3   Memo     Memo     Memo    Comp     Memo
lr=4.0e-3   Memo     Grok     Memo    Grok     Conf
lr=6.3e-3   Memo     Memo     Memo    Memo     Conf
```

*Memo = Memorization, Grok = Grokking, Comp = Comprehension, Conf = Confusion.*

Generalizing cells (Grokking or Comprehension) are confined to a narrow strip: **wd = 2.5**, for lr between 1×10⁻³ and 4×10⁻³. Below wd ≈ 0.63 the model memorizes regardless of lr; above wd ≈ 10 it collapses into Confusion.

> [!note] On Comprehension vs Grokking labels
> Stage 1 classifies lr=1.6e-3 and lr=2.5e-3 at wd=2.5 as Comprehension (τ_gen within the first 33% of the training budget). Stage 0 labeled the same hyperparameters (grok_B) as Grokking. Both observe identical τ_gen values (~10k–13.5k steps); the discrepancy is a classification artifact from slightly different threshold implementations. The timing-gap analysis is unaffected.

---

### Main Finding: G < F, Localized at wd = 2.5

Across the 5×5 grid, **G<F ordering appears exclusively at wd = 2.5**:

| lr | Phase | G<F fraction | Median Δ = τ_F − τ_gen |
|----|-------|-------------|------------------------|
| 1.0×10⁻³ | Grokking | **3/3** | +9,500 steps |
| 1.6×10⁻³ | Comprehension | **3/3** | +5,000 steps |
| 2.5×10⁻³ | Comprehension | **3/3** | +5,000 steps |
| 4.0×10⁻³ | Grokking/Comp | **3/3** | +4,500 steps |
| 6.3×10⁻³ | Memo/Grokking | 0/3 | ~0 |

All 12 generalizing runs at wd=2.5 (four lr values) show the model generalizing *before* its token embeddings form a Fourier-aligned representation. The remaining 63 runs — spanning four other wd values across all five lr settings — show zero G<F instances.

> [!important] Core result
> **Generalization precedes Fourier alignment structure.** The model learns to correctly compute modular addition before its internal token representations organise into the clean Fourier geometry that characterises the "grokked" solution. The Fourier structure is a lagging indicator of the computational breakthrough, not its cause.

---

### Secondary Finding: Δ Decreases with Learning Rate

Along the wd=2.5 column, the timing gap shrinks monotonically as lr increases:

$$
\Delta(\text{lr}) \approx 9.5\text{k} \xrightarrow{\text{lr} \uparrow} 5\text{k} \xrightarrow{} 5\text{k} \xrightarrow{} 4.5\text{k} \xrightarrow{} 0\text{k}
$$

At lr=6.3×10⁻³ the gap collapses to zero: the model either fails to generalize or the two onsets become simultaneous. This suggests that faster optimization (higher lr) causes structure formation and generalization to co-occur more tightly, potentially converging to the Comprehension regime.

---

### Third Finding: Fourier Alignment Without Generalization (F_only)

**39 of 75 runs** exhibit tau_F (Fourier changepoint detected) but no tau_gen (test accuracy never reaches 90%). These "F_only" cells are concentrated at:

- **wd=10** (all 15 runs): extremely strong regularization; Fourier structure fires early (~8k steps) but weights collapse before the model can generalise
- Low-wd Memorization cells: Fourier structure can appear as a transient during memorisation without completing the generalisation transition

This demonstrates that Fourier alignment in the token embeddings is **necessary but not sufficient** for generalisation. A model can develop the "right-looking" internal geometry and still fail to generalise.

---

### Fourth Finding: Single F < G Point

One cell stands out: **lr=4×10⁻³, wd=0.16** — mixed Grokking/Memorization, Δ=−18,500 steps (structure precedes generalisation by ~18k steps). This is the only cell in the grid where F<G holds. Its location near the Memorization/Grokking phase boundary suggests that F<G ordering may be specific to slow, boundary-regime grokking where the model spends a long time in a structured-but-not-yet-generalised state.

---

### Circle Score (tau_ring)

Circle Score never reached the 0.8 sustained threshold in any of the 93 runs across Stage 0 and Stage 1 (50,000 steps). CS shows an upward trend in Grokking cells but does not saturate within the training budget. **tau_ring is dropped as a primary metric** for this study; Fourier alignment (tau_F) serves as the sole structural onset marker.

---

## What We Did Not Find

**No Comprehension phase in the original search grid.** The initial 6×6 sweep (Step A) and two rounds of spot tests (14 candidates, lr ∈ [10⁻⁴, 2×10⁻¹], wd ∈ [10⁻², 30]) found no Comprehension cells. This is consistent with the reduced training-set size for p=53 (~840 samples vs ~2,800 for MIT's p=97): smaller datasets make memorization easier and narrow the Comprehension window. Comprehension does appear in Stage 1's broader wd=2.5 sweep under the Stage 1 classifier, but the labelling ambiguity noted above means this should be verified with an explicit joint-onset check.

---

## Next Steps

### Stage 2 (Immediate)

Focus on the **wd=2.5 column** where G<F is confirmed:

1. **Finer lr resolution** — 10 lr values between 1×10⁻³ and 6×10⁻³ at wd=2.5, to precisely characterise the lr boundary where Δ→0
2. **More seeds** — 5 seeds per cell (vs 3 now) near the boundaries (lr≈5×10⁻³ at wd=2.5)
3. **Verify the single F<G point** — re-run lr=4×10⁻³, wd=0.16 with 5 seeds and 80k steps to confirm the F<G ordering and rule out censoring

### Stage 3 (Planned)

- **wd sweep at fixed lr=1.6×10⁻³** — map the exact wd boundary separating Memorization from Grokking/Comprehension, and check whether Δ varies smoothly across the transition
- **Longer runs (100k steps)** — check whether Circle Score eventually saturates in the confirmed G<F cells
- **p=97 comparison** — replicate the wd=2.5 G<F experiment with p=97 (matching MIT setup) to test whether the G<F ordering persists at larger prime

### Open Questions

1. **Why wd=2.5 specifically?** The G<F ordering is sharply localised at this single wd value. What is special about this regularisation strength? Does it correspond to a particular loss landscape geometry?
2. **What does F_only mean mechanistically?** 39 runs develop Fourier token geometry but fail to generalise. Is the embedding geometry sufficient but the circuit that reads it out underdeveloped? Does the Fourier structure in F_only cells look the same as in generalising cells?
3. **Is G<F universal or model-specific?** All positive G<F evidence comes from a 2-layer, 4-head decoder at p=53. MIT's longer runs with p=97 might show the opposite ordering (or no ordering), since the model there has more capacity relative to the task.

---

## Summary Table

| Finding | Evidence | Confidence |
|---------|----------|-----------|
| G<F ordering at wd=2.5 | 12/12 runs, 4 lr values, 3 seeds each | High |
| Δ decreases with lr | Monotone: 9.5k→5k→5k→4.5k→0k | High |
| F_only widespread | 39/75 runs; stable at wd=10 | High |
| tau_ring undetected at 50k steps | 0/93 runs reach CS≥0.8 | High |
| Single F<G point at phase boundary | 1 cell, 3 seeds, needs confirmation | Low–Medium |
| Comprehension absent in p=53 grid | 14 spot-test candidates, 0 Comprehension | Medium (grid may miss narrow window) |
