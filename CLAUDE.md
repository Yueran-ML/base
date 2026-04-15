# Project: Transformer Phase Diagram — Representation Structure Timing

## Research Goal
Investigate whether visible circular structure in token embeddings (measured by Circle Score) lags behind generalization in grokking, across the (decoder_lr, decoder_wd) phase diagram.

See `docs/phase_diagram_brief.md` for full research brief.

## Working Directory
All code and outputs live in this directory:
`C:\Users\ASUS\Desktop\文件\学术\7841\base\`

- `grokking_baseline.py` — existing baseline implementation
- `docs/phase_diagram_brief.md` — full research brief with metrics and method
- `runs/` — experiment outputs

## Compute Environment

### Local GPU (primary)
- GPU: NVIDIA GeForce RTX 4080 Laptop GPU (12 GB VRAM)
- OS: Windows 10, Python 3.14
- Run experiments locally using `python` directly
- No SSH or rsync needed — all experiments run on this machine

### Running Experiments
```bash
python grokking_baseline.py [args]
```

For phase diagram sweep (100 grid cells, ~8-9 hours):
```bash
python phase_diagram_sweep.py
```

Save outputs to `runs/` directory with timestamped subfolders.

## Key Metrics (from brief)
- **t_gen**: step when test_acc first > 90%
- **t_ring**: step when Circle Score (CS) first > 0.8
- **Δ = t_ring − t_gen**: structure lag (main output variable)
- **e^S**: effective dimensionality (MIT paper metric)
- **CS**: Circle Score — fraction of admissible parallelograms satisfying |E_i + E_j - E_m - E_n| / avg_norm < δ=0.1

## Model Architecture (MIT paper Sec 4.2)
- p = 53, d_model = 256, n_layers = 2, n_heads = 4, d_ff = 1024
- Decoder-only transformer, input: [a, b] (2 tokens)
- Embedding: lr=1e-3, wd=0 (fixed)
- Decoder sweep: lr ∈ [3e-7, 4e-2], wd ∈ [1e-2, 30] (log-spaced, 10×10 grid)

## Phase Labels
- **Comprehension**: train_acc > 90% and test_acc > 90% early
- **Grokking**: train_acc > 90% first, test_acc > 90% later
- **Memorization**: train_acc > 90%, test_acc never > 90%
- **Confusion**: neither reaches 90%

## Expected Outputs
1. `phase_diagram.png` — 10×10 heatmap colored by phase
2. `delta_heatmap.png` — 10×10 heatmap colored by t_ring − t_gen
3. `timeline_comparison.png` — test_acc, e^S, CS on same time axis for 3-4 representative cells
4. `results.csv` — raw metrics for all 100 grid cells

## Notes
- Save embedding snapshots every 500 steps for CS computation
- Use seed=42 for all runs; run 5 seeds for cells near phase boundaries
- Fallback if CS never reaches threshold: report e^S as primary metric (also a valid contribution)
- 回答全部使用简体中文
- 每一步决策让codex审核一遍
- 每次压缩上下文时，筛选重要的部分更新 CLAUDE.md

---

## Current Research State (updated 2026-03-31)

### Pivot: Research Question Has Changed
Original goal (Circle Score lag) was abandoned because **Circle Score never reaches 0.8 in any of 93 runs at 50k steps**. Research pivoted to:

> **Does Fourier alignment in token embeddings appear before or after generalization? (G<F ordering)**

### Key Findings (Stage 0 + Stage 1 + Stage 2)
- **G<F ordering confirmed broadly**: 29/30 Stage 2 runs are G<F across wd∈[1.2, 3.5] at lr=1.6e-3
- **G<F is a wide plateau**: Stage 1's "localized to wd=2.5" was a coarse-grid artifact — G<F holds across entire tested wd range
- **Δ = tau_F − tau_gen**: ranges 5,500–13,000 steps in Stage 2, no clear monotonic trend
- **F_only widespread** (Stage 1): 39/75 runs have tau_F but no tau_gen — Fourier alignment necessary but not sufficient
- **tau_ring dropped**: Circle Score never detected; tau_F (BIC changepoint) is the sole structural metric
- **Stage 2 sole exception**: wd=3.11 seed=2025, Δ=0 (tau_gen=tau_F=11000, simultaneous — tie not genuine F<G)

### Canonical Phase Classifier (grok_metrics.py)
All scripts import from `grok_metrics.py` — do NOT redefine locally:
```python
# Grokking if tau_gen - tau_train >= 2000 steps (sustained detection, n=3)
from grok_metrics import classify_phase, find_tau_sustained, estimate_changepoint,
                         compute_fourier_alignment, compute_fourier_null_p95, compute_ordering
```

### Completed Scripts
| Script | Purpose | Status |
|--------|---------|--------|
| `grok_metrics.py` | Shared metric utilities (canonical) | active |
| `stage0_metric_validation.py` | Pilot (6 cells × 3 seeds) | done |
| `stage1_coarse_sweep.py` | 5×5 grid sweep | done → `runs/stage1_coarse/` |
| `stage2_wd_sweep.py` | lr=1.6e-3, wd∈[1.2,3.5], 10×3 | **done** → `runs/stage2_wd/` |

### Next Step: TBD
Stage 2 complete (answer: G<F is a wide plateau, not a peak). Options: (a) sweep lr axis at fixed wd, (b) write up findings, (c) other experiment.

### All Documents in Research/ (Obsidian vault)
- `Research/研究说明文档.md` — Chinese research log (Steps 1–18)
- `Research/research_notes_en.md` — English research log (Steps 1–18)
- `Research/research_report_en.md` — English research report (Stage 0+1 findings)
- `Research/讲解文档.md` — Non-specialist explanation of code and theory
