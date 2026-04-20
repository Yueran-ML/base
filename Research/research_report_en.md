---
title: "A Three-Stage Picture of Grokking: Circuit Formation, Generalization, and Embedding Geometry Consolidation"
date: 2026-04-20
tags:
  - grokking
  - transformer
  - fourier-structure
  - phase-diagram
  - report
aliases:
  - research report
  - three-stage report
---

# A Three-Stage Picture of Grokking

> **Status**: full thesis complete. Stages 0–6 + Step 2 (τ_circuit) done; E3 Nanda-comparison code complete but not yet run.
> **Model**: decoder-only transformer, $p=53$ modular addition, $d_\text{model}=256$, 2 layers, 4 heads.

---

## Question

Do the three temporal landmarks of grokking — (i) the internal *Fourier circuit* forming in the logits, (ii) *generalization* crossing the test-accuracy threshold, and (iii) the token embedding's *Fourier geometry* consolidating — happen simultaneously, or in a specific order?

We claim:

$$\tau_\text{circuit} \;<\; \tau_\text{gen} \;<\; \tau_F$$

and directly measure all three on the same runs, for the first time.

---

## Metrics

| Symbol | Name | Definition |
|--------|------|-----------|
| $\tau_\text{circuit}$ | Circuit formation | BIC changepoint on $F_L(t)$ — Fourier alignment of the full logit matrix onto 2-freq subspaces |
| $\tau_\text{gen}$ | Generalization | First step with test acc $\ge 0.9$ sustained over 3 checkpoints |
| $\tau_F$ | Embedding geometry | BIC changepoint on $F_\text{corr}(t)$ — permutation-null-corrected Fourier alignment of the token embedding |

Changepoint detector: 1-breakpoint segmented regression selected by BIC (see `src/grok_metrics.py::estimate_changepoint`). All detectors robust to 36 config variations (sensitivity analysis).

---

## Main results

Five experiment stages, totalling 236 Grokking-phase runs:

| Experiment | n (Grokking) | G<F rate | Median Δτ = τ_F − τ_gen |
|------------|--------------|----------|-------------------------|
| Stage 2 (wd sweep, lr=1.6e-3) | 29/30 | 96.7% | ~8,500 steps |
| Stage 3 (lr sweep, wd=2.5) | 23/30 | 91.3% | ~5,500 steps |
| Stage 5A (mult mod 53) | 30/30 | **100%** | 8,750 |
| Stage 5B (add mod 97) | 21/30 | **100%** | 28,000 |
| Stage 6 (7×7 2-D grid) | 130/147 | 85.4% | 6,000 |
| **Total** | **236** | **89.8%** | — |

### Three-stage ordering (Step 2, 60 runs)

Direct measurement of $\tau_\text{circuit}$ via $F_L(t)$:

| Comparison | Fraction |
|------------|----------|
| C<G ($\tau_\text{circuit} < \tau_\text{gen}$) | 50/54 = 92.6% |
| G<F ($\tau_\text{gen} < \tau_F$) | 52/54 = 96.3% |
| **Full C<G<F** | **48/54 = 88.9%** |

Interval medians: $\tau_\text{gen} - \tau_\text{circuit} = 6{,}250$ steps; $\tau_F - \tau_\text{gen} = 6{,}000$ steps; total $\tau_F - \tau_\text{circuit} = 13{,}000$ steps. The three intervals are nearly equal.

### Boundary behavior

The few F<G runs (14/130 in Stage 6) concentrate at phase boundaries — low-lr and high-lr edges of the Grokking region — supporting a *speed-dependent ordering* hypothesis: very fast or very slow generalization disturbs the canonical ordering.

---

## Implications

1. **Generalization is not the last step of grokking.** Embedding geometry continues to consolidate for ~6,000 more steps after test accuracy saturates.
2. **Fourier circuit forms first.** Independent of the embedding's visible ring structure, the logits acquire their algorithmic structure before generalization — consistent with Nanda 2023's mechanistic account.
3. **The ordering is robust across task and prime.** Addition mod 53, multiplication mod 53, addition mod 97 — all give ≥85% G<F; mult and p=97 give 100%.
4. **The weight-decay plateau is wide, not a knife-edge.** The G<F plateau spans wd ∈ [1.2, 3.5] at lr=1.6e-3; it is not localized to a narrow boundary.

---

## What we did not find

- A Circle Score threshold that fires reliably — in 0/93 pilot runs did CS cross 0.8 within 50k steps. We therefore adopted the BIC changepoint on $F_\text{corr}$ as the canonical $\tau_F$ detector.
- Monotonic Δτ dependence on lr or wd within the Grokking plateau — the relationship is flat, with deviations only at phase boundaries.

---

## Code and artifacts

| Artifact | Location |
|----------|----------|
| Full progress log (28 steps, English) | [research_notes_en.md](research_notes_en.md) |
| Full progress log (28 steps, Chinese) | [研究说明文档.md](研究说明文档.md) |
| Thesis LaTeX source | `paper/` |
| Sweep scripts | `src/sweeps/` |
| Analysis scripts | `src/analysis/` |
| Summary CSVs per experiment | `results/` |

Reproduction instructions: see [../README.md](../README.md#reproduce).

---

## Next steps

1. Launch E3 Nanda-comparison sweep (~10 wall-clock hours on RTX 4080 with `--parallel 6`); produces run-by-run agreement between our $F_L$ changepoint and Nanda's restricted / excluded logit loss changepoints.
2. Statistical regression of Δτ on (lr, wd, seed) predictors.
3. Systematic F_only study: when does Fourier alignment occur *without* generalization?
4. Architecture and train_frac robustness (optional).
