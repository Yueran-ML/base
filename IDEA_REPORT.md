# Idea Discovery Report

**Direction**: Transformer grokking phase diagram — does visible circular structure in embeddings lag behind generalization across (decoder_lr, decoder_wd) configurations
**Date**: 2026-03-26
**Pipeline**: research-lit → idea-creator (GPT-4o via Codex MCP)

---

## Executive Summary

The MIT paper's phase diagram is static — it classifies hyperparameter cells but does not measure the *timing* of geometric structure relative to generalization. The cleanest novel contribution is a **Signed-Δ Atlas**: extend the existing sweep to track Circle Score (RQI parallelogram test) alongside e^S, then plot Δ = t_ring − t_gen as a heatmap on the same 10×10 grid. This directly answers whether the visible ring lags, leads, or coincides with generalization as a function of the four phase regions. Supporting experiments (Fourier bridge at near-zero compute, post-grokking destruction test) sharpen the causal story. Full sweep pilots are flagged as **needs manual pilot** (estimated 8–9 GPU hours > PILOT_MAX_HOURS = 2).

---

## Ranked Ideas

### 🏆 Idea 1: Signed-Δ Atlas
**Hypothesis**: The four-phase map has temporal subphases; the largest positive Δ = t_ring − t_gen concentrates near the grokking/memorization boundary rather than being uniform across the grokking region.

**Experiment**:
1. Add `compute_circle_score(embeddings, prime, delta=0.1) → float` to `grokking_baseline.py` — parallelogram test on admissible (i+j≡m+n mod p) quadruples
2. Modify `run_single_for_phase` to return a dataclass: `{phase, t_gen, t_eS, t_ring, CS_trajectory}`
3. Run 10×10 (decoder_lr, decoder_wd) grid, collecting embedding snapshots every 500 steps
4. Plot: (A) phase diagram, (B) Δ heatmap colored by t_ring − t_gen, (C) event-order labels {ring→gen, gen→ring, coincident, neither}, (D) 3-4 representative timelines

**Why interesting**: First direct measurement of geometric circularity timing across the full phase space. Upgrades the MIT phase diagram from static classification to temporal-geometric characterization. Directly answers the open question the MIT paper acknowledges.

**Feasibility**: MEDIUM compute — 100 runs × 30k steps ≈ 8–9 GPU hours (exceeds PILOT_MAX_HOURS; **needs manual pilot**). Code changes are modest (~100 lines new code on top of existing sweep).

**Risk**: Circle Score may never reach threshold 0.8 in any cell (high-dimensional embedding squashes the parallelogram residuals). Mitigation: also report threshold-free CS trajectory curves; use PCA-reduced embeddings for the CS computation.

**Novelty**: HIGH — no paper maps Δ across the full 2D phase space.

---

### 🥈 Idea 2: Fourier-to-Circle Bridge
**Hypothesis**: t_gen is more tightly coupled to concentration onto the first Fourier harmonic pair of Z_p than to visually obvious PCA circularity; large positive Δ occurs when spectral alignment precedes clean geometric separation.

**Experiment**:
1. At each checkpoint, project embeddings onto sin(2πk/p), cos(2πk/p) for the dominant harmonic k and measure first-harmonic alignment score
2. Track t_fourier (step when harmonic alignment first exceeds threshold) alongside t_ring and t_gen
3. Test whether t_fourier < t_gen < t_ring (or other orderings) across the grid
4. This adds ~20 lines to the logging loop — no sweep overhead

**Why interesting**: Bridges the visual-geometric (RQI circle) and spectral-algebraic (Fourier circuit, Nanda 2023) accounts of grokking. Clarifies which signal is the "real" leading indicator. Very low extra compute.

**Feasibility**: LOW-MEDIUM compute — purely a checkpoint analysis addition, almost zero training overhead. Can be added to the same sweep as Idea 1 at no extra cost.

**Risk**: Learned embedding rotations may align poorly with the canonical Fourier basis; discrete Fourier structure in 256D may be distributed across many harmonics.

**Novelty**: MEDIUM-HIGH — Nanda 2023 identified the Fourier circuit, but no paper maps harmonic alignment timing against geometric circularity across the (lr, wd) grid.

---

### 🥉 Idea 3: Post-Grokking Destruction Test
**Hypothesis**: If visible circular structure is causally necessary for generalization post-grokking, targeted angular perturbations to saved embeddings should immediately collapse test accuracy in positive-Δ cells; if not, test accuracy is robust.

**Experiment**:
1. Save checkpoints at t_ring − ε and t_ring + ε for representative cells
2. Apply offline perturbations: (a) angular permutation of embedding vectors, (b) projection off the top PCA plane, (c) radial-only noise
3. Evaluate test accuracy before/after without retraining
4. Compare robustness across phase regions (Grokking vs Comprehension vs Memorization)

**Why interesting**: Converts the correlational Δ measurement into a causal necessity test. Extremely low extra compute (checkpoint analysis only).

**Feasibility**: LOW compute — uses saved checkpoints, no extra training beyond the main sweep.

**Risk**: Perturbations may be criticized as out-of-distribution damage rather than targeted geometry removal. The test cannot distinguish "geometric structure is necessary" from "any perturbation hurts."

**Novelty**: MEDIUM-HIGH — no paper tests the causal necessity of embedding circularity post-grokking across phase regions.

---

### Idea 4: Representation Cascade Map
**Hypothesis**: Circularity appears first in later hidden states or the output head, and only later in the token embedding table; embedding-level lag overstates the true representational lag.

**Experiment**:
1. Extend Circle Score computation to: token embeddings, output head weight rows, layer-0 and layer-1 token-conditioned hidden states (average hidden rep per token across all contexts)
2. Track t_ring at each layer for a subset of grid cells (one per phase)
3. Report whether the ring "migrates inward" from output→embedding or appears everywhere simultaneously

**Why interesting**: Mechanistically distinguishes "geometry lags" from "geometry migrates from head to embedding," which is a stronger and more surprising story.

**Feasibility**: MEDIUM compute — analysis requires storing per-step hidden states (memory overhead), but model is small (d_model=256, p=53 so only 53 token reps).

**Risk**: Contextual hidden states averaged across all (a,b) inputs may be too noisy to form a clean per-token circle.

**Novelty**: MEDIUM — no paper tracks the spatial origin of circular structure across layers during grokking.

---

### Idea 5: Hyperparameter Quench Test
**Hypothesis**: Δ depends on training trajectory, not just endpoint hyperparameters; switching (lr, wd) mid-training across a phase boundary produces a different lag profile than training at the target from scratch.

**Experiment**:
1. Choose 4–6 source/target pairs: e.g., Memorization → Grokking, Grokking → Comprehension
2. Switch (decoder_lr, decoder_wd) at checkpoint times t = 2k, 5k, 10k steps
3. Measure post-switch t_gen and t_ring, compare to baseline training at target from step 0
4. Report: does early memorization pre-training "poison" the ring lag?

**Why interesting**: Causal, phase-transition-style test of whether Δ is a property of the basin/trajectory or a static property of the hyperparameter region. Directly relevant to training practice.

**Feasibility**: MEDIUM compute — small number of extra runs (~20–30), not the full grid.

**Risk**: Switch dynamics may be unstable; results may be hard to summarize cleanly without many more runs.

**Novelty**: MEDIUM — hyperparameter quenching is known in physics but has not been applied to grokking geometry timing.

---

### Idea 6: Geometry-as-Cause Intervention
**Hypothesis**: Adding a weak parallelogram-consistency auxiliary loss (RQI loss) shortens grokking delay specifically in cells where baseline Δ > 0, with little effect in Comprehension cells.

**Experiment**:
1. Add `rqi_loss(embeddings, prime, lambda_rqi)` — sample random admissible parallelograms, penalize residual norm
2. Sweep λ ∈ {0, 0.01, 0.1, 1.0} at a few representative cells per phase
3. Compare shifts in t_gen, t_ring, Δ, and final accuracy

**Why interesting**: Turns the correlational Δ observation into a causal sufficiency claim — if forcing the ring early moves t_gen earlier, the ring is causally helpful.

**Feasibility**: MEDIUM compute — small targeted runs, not a full grid.

**Risk**: Any speed-up may be dismissed as generic regularization rather than geometry-specific causation; may simply speed up all training uniformly.

**Novelty**: MEDIUM — auxiliary structural losses have been used before, but not specifically tied to the phase-specific Δ prediction.

---

### Idea 7: Local-to-Global Circle Emergence
**Hypothesis**: Generalization starts once *local* additive consistency appears (short-range parallelograms pass the RQI test), while the globally visible ring forms later; small-radius Circle Score predicts t_gen earlier than global CS.

**Experiment**:
1. Define multiscale Circle Scores by restricting admissible parallelograms to |i−m| ≤ r (mod p) for r ∈ {2, 5, 10, 26}
2. Track t_ring(r) for each radius at representative grid cells
3. Test whether t_ring(r_small) < t_gen < t_ring(global)

**Why interesting**: Provides a finer-grained metric that could serve as an early warning signal for generalization — practically valuable if it leads.

**Feasibility**: MEDIUM compute — purely a metric modification, no extra training.

**Risk**: The notion of "local" on Z_p may not align with what PCA visually calls a ring; Z_p has no natural metric "distance."

**Novelty**: MEDIUM — multiscale geometric metrics have not been applied to grokking embeddings.

---

### Idea 8: Algebra-Aware Cross-Operation Mapping
**Hypothesis**: The ring-lag phenomenon generalizes to multiplication/division after reindexing symbols by discrete logarithm (mapping Z_p* → Z_{p-1}) because the underlying structure is always cyclic.

**Experiment**:
1. Run the Δ heatmap for mul and div operations using existing task variants
2. For mul/div, remap nonzero token IDs by discrete log before computing Circle Score
3. Compare Δ maps across operations: does the same phase-dependent lag appear?

**Why interesting**: Tests whether the geometry tracks algebraic structure rather than raw token IDs — a much stronger universality claim.

**Feasibility**: HIGH compute — requires separate full sweeps per operation (~3× total compute).

**Risk**: Zero-handling and coordinate choice ambiguities may make negative results hard to interpret.

**Novelty**: HIGH — no paper compares geometric timing across operations with structure-aware reindexing.

---

### Idea 9: Finite-Size Scaling of Δ
**Hypothesis**: The signed lag Δ obeys a scaling law with prime size p and train fraction, with larger p or smaller train fraction expanding the positive-Δ region.

**Experiment**:
1. Repeat coarse (5×5) phase sweeps for p ∈ {29, 53, 97} and train_fraction ∈ {0.2, 0.3, 0.4}
2. Normalize time by tokens seen per epoch
3. Test whether Δ maps collapse under a simple rescaling

**Why interesting**: Turns a single-benchmark observation into a scaling-law claim, fitting the trend in grokking theory papers.

**Feasibility**: HIGH compute — ~9 extra sweeps.

**Risk**: Noise near phase boundaries may obscure any clean scaling pattern.

**Novelty**: MEDIUM — scaling laws for grokking timing exist, but not for geometric structure lag.

---

### Idea 10: Architecture Mediation of Δ
**Hypothesis**: Δ is sensitive to specific architectural choices in the baseline: post-concat LayerNorm, tied/untied output head, and positional embeddings.

**Experiment**:
1. Run reduced (5×5) phase sweeps for baseline variants: no LayerNorm, tied output head, no positional embeddings, 1-layer
2. Compare the Δ heatmaps

**Why interesting**: Tests whether the phenomenon is robust across architectures or a baseline-specific artifact.

**Feasibility**: HIGH compute — 4 variants × 25 cells each.

**Risk**: Results may be fragmented without a dominant factor; hard to tell a clean story.

**Novelty**: MEDIUM — architectural sensitivity of grokking has been studied but not for geometric timing.

---

## Novelty Verdict (Phase 3)

### Idea 1 (Signed-Δ Atlas) — CONFIRMED NOVEL
- 2511.01938 (Musat 2025): proves norm minimization → circle in 2-layer net; does NOT map Δ across the transformer (lr,wd) phase diagram
- 2602.16849 (He et al. 2026): analyzes Fourier competition theoretically in 2-layer net; does NOT measure t_fourier vs t_gen across the full grid
- **No paper** maps τ_ring − τ_gen as a 2D heatmap on the transformer phase space

### Idea 2 (Fourier-to-Circle Bridge) — CONFIRMED NOVEL (with sharper framing needed)
- 2301.05217 (Nanda 2023): identifies Fourier circuit at generalization; no timing sweep
- 2602.16849 (He 2026): Fourier competition theory in 2-layer net; no empirical ordering across (lr,wd) grid
- **Critical sharpening**: must frame as "phase-conditioned timing law" (F<G<R in grokking, collapsed in comprehension), not just "Fourier before circle"

## Critical Review Results (Phase 4)
**Reviewer score**: 4/10 (as-is framing) → target 7–8/10 (after refinement)

**Original weaknesses identified**:
1. Novelty too thin — overlay story, no sharp falsifiable claim
2. Threshold artifacts — first-crossing times not robust scientific objects
3. Causal experiment invalid — angular permutation breaks semantics regardless

**Refined framing**: "The Ring Is Not the Algorithm" — grokking and comprehension differ by **event ordering**, not delay magnitude. The key claim: τ_Fourier < τ_gen < τ_ring specifically in grokking, collapsed in comprehension.

**Key improvements**:
1. Changepoint detection (segmented regression, BIC) replaces first-crossing thresholds
2. Subspace intervention (E_cyc transplant) replaces angular permutation
3. Ordering probability P(τ_F < τ_gen < τ_R) as the primary scientifically robust object

See `refine-logs/FINAL_PROPOSAL.md` for full refined proposal.

---

## Pilot Status

**All pilots flagged as NEEDS MANUAL PILOT** — the minimum meaningful experiment (10×10 grid × 30k steps) requires ~8–9 GPU hours, exceeding PILOT_MAX_HOURS = 2.

Recommended pilot order (ascending compute):
1. **Ideas 2+3** (Fourier Bridge + Destruction Test): run a single representative Grokking cell (~1 run × 30k steps ≈ 5 min on GPU), add CS + Fourier logging, verify metrics fire correctly. **True pilot cost: ~10 min.**
2. **Idea 1** (Signed-Δ Atlas): 5×5 coarse grid (25 cells) ≈ 2 GPU hours — borderline feasible as pilot.
3. Full 10×10 sweep: manual.

---

## Eliminated Ideas (Phase 2 filter)
None eliminated yet — all ideas are technically valid. Ideas 8, 9, 10 are deprioritized due to HIGH compute and secondary relevance; they work better as follow-on experiments after the core result is established.

---

## Next Steps
- Phase 3: Novelty check on Ideas 1, 2, 3 against recent arXiv
- Phase 4: External critical review of top idea
- Phase 4.5: Method refinement + experiment plan
