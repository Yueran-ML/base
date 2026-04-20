# Research Brief: Transformer Phase Diagram with Representation Structure Timing

## 1. Research Question

In grokking, does the emergence of **visible circular structure** in token embeddings coincide with generalization, or does it systematically **lag behind**?

Specifically, for different (decoder_lr, decoder_wd) configurations in the phase diagram:
- In which phase regions does t_ring ≈ t_gen (circle appears with generalization)?
- In which regions does t_ring >> t_gen (circle lags behind)?
- Does the lag Δ = t_ring − t_gen vary smoothly across phase boundaries?

## 2. Motivation

The MIT paper (Sec 4.2, Figure 6/7) establishes a **four-phase diagram** (Comprehension / Grokking / Memorization / Confusion) governed by the competition between representation learning rate and decoder capacity. However, the paper:

- Uses only **e^S (effective dimensionality)** as a quantitative proxy for circular structure
- Acknowledges that "no choice of dimensionality reduction is guaranteed to find any structure"
- Does **not** apply RQI (used only in toy models) to the transformer setting
- Shows e^S drops **at** generalization, not before — leaving open whether the visible geometric structure (the circle) might lag

This creates a gap: the **timing relationship** between the quantifiable circle and generalization across the full phase space has not been mapped.

## 3. Method

### 3.1 Architecture (matching MIT paper Sec 4.2)

| Parameter | Value | Source |
|-----------|-------|--------|
| p (prime) | 53 | MIT paper Sec 4.2 |
| d_model | 256 | "encode p=53 integers into 256D learnable embeddings" |
| Architecture | Decoder-only transformer | "decoder-only transformer architecture" |
| Input | [a, b] (2 tokens, no op token) | "we do not encode the operation symbols" |
| Output | Concatenate both positions → linear | "outputs from last layer are concatenated" |
| n_layers | 2 | Following Power et al. (2022) default |
| n_heads | 4 | Following Power et al. (2022) default |
| d_ff | 1024 | 4 × d_model (standard) |
| Embedding weight decay | 0.0 | "zero weight decay" on embeddings |
| Embedding lr | 1e-3 (fixed) | "learning rate of embeddings kept fixed at 10^-3" |

### 3.2 Phase Diagram Sweep

| Parameter | Range | Scale |
|-----------|-------|-------|
| Decoder learning rate | 3e-7 → 4e-2 | Log-spaced |
| Decoder weight decay | 1e-2 → 30 | Log-spaced |
| Grid resolution | 10 × 10 (= 100 runs) | — |
| Max steps per run | 30,000 | MIT paper tracks up to ~20k |
| Embedding lr | 1e-3 (fixed) | — |
| Seed | Fixed seed=42 (+ multi-seed for key cells) | — |

### 3.3 Metrics Collected Per Grid Cell

For each (decoder_lr, decoder_wd) pair, record:

1. **Phase label**: Comprehension / Grokking / Memorization / Confusion
   - Based on train_acc > 90% timing and test_acc > 90% timing

2. **t_gen**: Step when test_acc first exceeds 90%
   - ∞ if never reached (Memorization / Confusion)

3. **t_eS**: Step when e^S first drops below a threshold
   - Threshold: 50% of initial e^S (≈ 53/2 ≈ 26)
   - MIT paper says this coincides with t_gen; verify

4. **Circle Score (CS)**: A quantitative "circularity" metric on the embedding matrix
   - Computed every 500 steps from embedding snapshots
   - **Definition**: For all admissible parallelograms (i+j ≡ m+n mod p), compute:
     ```
     CS = fraction of (i,j,m,n) where |E_i + E_j - E_m - E_n| / avg_norm < δ
     ```
     where δ = 0.1 (tunable threshold)
   - This is the RQI concept from the toy model, adapted to transformers
   - t_ring: Step when CS first exceeds 0.8

5. **PCA explained variance ratio**: Top-2 explained variance at each snapshot
   - A clean circle → PC1 ≈ PC2 ≈ 50% each

6. **Δ = t_ring − t_gen**: The "structure lag" — the key output variable

### 3.4 Outputs

1. **Phase diagram** (Figure A): 10×10 heatmap colored by phase (replicating MIT Figure 7 Right)
2. **Δ heatmap** (Figure B): Same grid, colored by t_ring − t_gen (blue = sync, red = lag)
3. **Timeline comparison** (Figure C): For 3-4 representative cells (one per phase), plot test_acc, e^S, and CS on the same time axis

## 4. Implementation Plan

### Step 1: Implement Circle Score (CS) metric
- Add `compute_circle_score(embeddings, prime) → float` function
- ~30 lines: enumerate admissible parallelograms, compute vector residuals
- Test on final embeddings from existing run (should be high if grokking occurred)

### Step 2: Extend `run_single_for_phase` to return rich metrics
- Currently returns only phase label string
- Modify to return a dataclass with: phase, t_gen, t_eS, t_ring, CS trajectory
- Add embedding snapshot collection (every 500 steps) inside the phase sweep loop

### Step 3: Extend `run_phase_diagram` to produce Δ heatmap
- Adjust grid range to match MIT paper (lr: 3e-7→4e-2, wd: 1e-2→30)
- Increase grid to 10×10
- Save raw results to CSV for post-hoc analysis
- Generate both phase diagram and Δ heatmap

### Step 4: Multi-seed validation (optional)
- For 5-6 interesting grid cells (near phase boundaries), run 5 seeds
- Report mean ± std of Δ

## 5. Compute Budget Estimate

- 100 grid cells × 30k steps × ~0.01s/step (GPU) ≈ **8-9 hours on single GPU**
- Can reduce to ~3 hours with 20k steps (MIT paper's range)
- Multi-seed (30 extra runs) adds ~3 hours
- Total: **~12 hours GPU time**

## 6. Expected Outcomes & Risks

### Optimistic scenario
- Δ > 0 in Grokking zone (circle lags generalization)
- Δ ≈ 0 in Comprehension zone (circle coincides)
- Clean gradient of Δ across phase boundaries
- → Strong thesis contribution: "visible geometric structure is a *consequence*, not a *cause*, of generalization"

### Pessimistic scenario
- Circle score never reaches threshold in transformer (too high-dimensional)
- → Fallback: use e^S as primary metric, report that RQI fails to transfer from toy model to transformers (itself a contribution)

### Key risk
- MIT paper admits "it is challenging to show explicitly that generalization only occurs when a structure exists"
- The circle may exist in a decoder-induced metric but not in Euclidean PCA
- Mitigation: also plot normalized PCA and report explained variance ratios
