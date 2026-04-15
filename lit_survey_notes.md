# Literature Survey Notes
**Direction**: Transformer grokking phase diagram — circular structure timing vs. generalization
**Date**: 2026-03-26

---

## Pillar 1: Phase Diagram Foundation (the reference)

**Liu et al. (2022) — "Towards Understanding Grokking: An Effective Theory of Representation Learning"**
- arXiv: [2205.10343](https://arxiv.org/abs/2205.10343)
- Establishes 4-phase diagram: Comprehension / Grokking / Memorization / Confusion
- Uses e^S (exponential of embedding entropy) as proxy for circular structure
- Shows e^S drops *at* generalization — does not track timing lag
- Uses only e^S, not geometric RQI, in the transformer setting
- **This is the reference paper we aim to extend**

---

## Pillar 2: Circular/Ring Structure in Grokking

**Nanda et al. (2023) — "Progress measures for grokking via mechanistic interpretability"**
- arXiv: [2301.05217](https://arxiv.org/abs/2301.05217) — ICLR 2023
- Fourier features analysis: transformers learn discrete Fourier transforms + trig identities to map addition → rotation on a circle
- Established that PCA of token embeddings shows circular structure at generalization
- Does not quantify timing of circle emergence relative to t_gen
- Does not sweep (lr, wd) phase space

**Gromov (2023) — "Grokking modular arithmetic"**
- arXiv: [2301.02679](https://arxiv.org/abs/2301.02679)
- Fourier circuit structure in modular arithmetic tasks

---

## Pillar 3: Embedding-Specific Mechanisms

**AlQuabeh et al. (2025) — "Mechanistic Insights into Grokking from the Embedding Layer"**
- arXiv: [2505.15624](https://arxiv.org/abs/2505.15624)
- Identifies two causes of grokking delay: (1) sparse embedding gradient updates, (2) bilinear coupling between embeddings and downstream weights
- Proposes solutions: uniform resampling + elevated embedding LR
- Does **not** study geometric circularity or t_ring timing
- Focuses on causes of delay, not measurement of representation structure onset

---

## Pillar 4: Geometric/Topological Inductive Bias

**Yıldırım (2026) — "The Geometric Inductive Bias of Grokking: Bypassing Phase Transitions via Architectural Topology"**
- arXiv: [2603.05228](https://arxiv.org/abs/2603.05228)
- Key finding: "delayed generalization coincides with a rapid decrease in effective radius and dimensionality of the representation manifold"
- Introduces spherical topology (L2 norm throughout residual stream) → removes grokking phase, 20x speedup
- Closest work to ours on timing of geometric compression vs. generalization
- However: studies architecture intervention, **not** phase-space mapping of Δ across (lr, wd)
- Single-point intervention, not a systematic grid sweep

**Lei & Xu — "Grokking as construct-then-compress"** (referenced in search results)
- Characterizes grokking as: memorization = build disjoint representations; generalization = compress into coherent algorithmic geometry
- "Commutator defect onset reliably precedes generalization" — serves as early warning signal
- Timing relationship studied, but not across (lr, wd) phase diagram

---

## Pillar 5: Multi-Task Geometry + Weight Decay Structure

**"The Geometry of Multi-Task Grokking: Transverse Instability, Superposition, and Weight Decay Phase Structure"**
- arXiv: [2602.18523](https://arxiv.org/abs/2602.18523)
- Extends geometric analysis to multi-task modular arithmetic
- Grokking timescale, curvature depth, defect lead **covary systematically with weight decay**
- Reveals distinct dynamical regimes and a sharp no-decay failure
- Most closely related to phase-space mapping, but: multi-task, different architecture, no Circle Score / t_ring measurement

---

## Pillar 6: Complexity / Information Dynamics

**"The Complexity Dynamics of Grokking"** — arXiv: [2412.09810](https://arxiv.org/abs/2412.09810)
- Complexity rises during memorization, falls at generalization
- "Construct-then-compress" consistent framing

**"Information-Theoretic Progress Measures reveal Grokking is an Emergent Phase Transition"** — arXiv: [2408.08944](https://arxiv.org/abs/2408.08944)
- Higher-order mutual information; grokking as emergent synergistic phase transition

**Li² — "Provable Scaling Laws of Feature Emergence from Learning Dynamics of Grokking"** — arXiv: [2509.21519](https://arxiv.org/abs/2509.21519)
- Three stages: Lazy learning → independent feature learning → interactive feature learning
- Provable scaling laws for weight decay, lr, sample size effects

---

## Pillar 7: Acceleration Methods (context)

**GrokAlign (2506.12284)** — Jacobian regularization induces grokking 7.56x faster; centroid alignment tracks stages
**Geometric Inductive Bias (2603.05228)** — Spherical topology removes grokking
**AlQuabeh et al. (2505.15624)** — Uniform sampling + LR adjustment

---

## THE KEY GAP

**What exists:**
- e^S drops at t_gen (MIT paper) — but this is an *information-theoretic* proxy, not a direct geometric measurement
- PCA circle exists at generalization (Nanda 2023)
- Delayed generalization coincides with compression of representation manifold (2603.05228)
- Weight decay affects grokking timescale (2602.18523, Li²)

**What does NOT exist:**
1. A 2D (decoder_lr, decoder_wd) grid map of Δ = t_ring − t_gen for the transformer setting
2. A dedicated Circle Score (CS) metric — the parallelogram/RQI test — applied to full transformers (only used in toy models in MIT paper)
3. Systematic comparison: does CS timing agree with e^S timing? In all four phase regions?
4. Whether Δ varies smoothly across phase boundaries or shows a sharp transition
5. Whether t_ring can *precede* t_gen in any phase region (is the circle ever predictive?)

**Verdict**: The research direction in the brief is **genuinely unexplored**. The closest works (2603.05228, 2602.18523) address geometric timing or (lr, wd) structure but not their combination as a phase-space Δ heatmap.

---

## Identified Sub-Directions for Idea Generation

1. **Core direction (brief)**: Map Δ across the full (dec_lr, dec_wd) phase diagram
2. **Metric agreement**: Compare CS and e^S timing — do they tell the same story or diverge by phase region?
3. **Predictive lag**: Can CS onset *precede* t_gen in Comprehension zone? (Is the ring ever an early warning?)
4. **Construct-then-compress alignment**: Does t_ring coincide with the "compress" onset in the Li²/construct-then-compress narrative?
5. **3D extension**: Vary embedding LR too — does the (emb_lr, dec_lr, dec_wd) cube reveal additional structure?
6. **Pessimistic fallback as contribution**: Document failure modes of RQI in high-D transformer embeddings (gap between toy model and transformer is itself novel)
