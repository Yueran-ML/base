---
name: Reference Paper Summary — Phase Diagram Grokking Brief
type: reference
---

# Reference Paper Summary

**Title**: Transformer Phase Diagram with Representation Structure Timing (Research Brief)
**Authors**: [Internal project brief — docs/phase_diagram_brief.md]
**Venue**: N/A (pre-experiment research brief)

## What They Did

The MIT paper (referenced therein) establishes a four-phase diagram (Comprehension / Grokking / Memorization / Confusion) for decoder-only transformers on modular arithmetic, governed by (decoder_lr, decoder_wd). The brief proposes extending this by mapping the **timing relationship** between the emergence of visible circular structure in token embeddings (t_ring) and generalization (t_gen) across the full 10×10 phase grid.

## Key Results (from MIT baseline)

- Four phases recovered by competition between representation learning rate and decoder capacity
- e^S (effective dimensionality) drops **at** generalization — but whether visible geometric structure (the circle) lags has not been tested in transformers
- RQI (Ring Quality Index / Circle Score) was used only in toy models, not in the full transformer setting

## Limitations & Open Questions

1. MIT paper uses only e^S as a circularity proxy — not geometric RQI in the transformer
2. The **timing relationship** Δ = t_ring − t_gen across the full phase space is unmapped
3. It is unresolved whether the circle is a cause or consequence of generalization
4. The paper admits "it is challenging to show explicitly that generalization only occurs when a structure exists"
5. The circle may exist in a decoder-induced metric but not in Euclidean PCA (high-dim issue)

## Potential Improvement Directions

- Adapt RQI/Circle Score to transformer embeddings and map Δ across the full phase diagram
- Test whether Δ varies systematically across phase boundaries (Comprehension vs Grokking zones)
- Use multi-seed validation near phase boundaries to establish statistical significance of Δ
- Explore alternative circularity metrics robust to high-dimensional embedding spaces

## Core Hypothesis

In the Grokking zone: t_ring >> t_gen (circle lags generalization — structure is a consequence)
In the Comprehension zone: t_ring ≈ t_gen (circle coincides with generalization)
The lag Δ = t_ring − t_gen varies smoothly across phase boundaries

## Codebase

Base repo at `C:\Users\ASUS\Desktop\文件\学术\7841\base` — contains phase diagram sweep infrastructure
