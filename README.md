# Three-Stage Grokking: Dissociating Logit Structure from Embedding Consolidation

UQ Master of IT thesis project (REIT7841) investigating the temporal ordering
of three events during grokking on modular arithmetic:

$$\tau_\text{circuit} \;<\; \tau_\text{gen} \;<\; \tau_F$$

where $\tau_\text{circuit}$ is when the Fourier logit structure forms (BIC
changepoint), $\tau_\text{gen}$ is when test accuracy crosses 90% (sustained),
and $\tau_F$ is when the embedding Fourier alignment consolidates.

**Core finding:** the visible Fourier ring in token embeddings is a
*post-generalization* structural consequence, not the load-bearing computation.
Across 236 Grokking runs spanning two tasks (addition, multiplication), two
primes (53, 97), and a dense (lr, wd) grid, **212/236 (89.8%)** show the G<F
ordering: embedding structure consolidates *after* generalization.

## Repository layout

```
src/
  grok_metrics.py           # shared metric utilities (canonical)
  grokking_baseline.py      # decoder-only transformer + training loop
  sweeps/                   # experiment drivers (stage0 - stage6, e3 - e5)
  analysis/                 # figure/table generation + post-hoc analyses
paper_tmlr/                 # TMLR paper LaTeX source + figures
paper/                      # earlier ICLR draft
plan/                       # UQ thesis plan LaTeX
proposal/                   # UQ proposal LaTeX
docs/                       # research brief + baseline explainer
Research/                   # Obsidian vault: progress log + canvases
results/                    # summary CSVs per canonical experiment
runs/                       # per-run metadata + training logs (no checkpoints)
tools/                      # bibliography/figure audit utilities
notebooks/                  # exploratory notebooks
```

Raw checkpoints (`.pt`) and per-seed trajectory `.npz` are **not** tracked.
Regenerate via the sweep scripts (see Reproduce below).

## Setup

Python 3.11+, Windows 11 + NVIDIA RTX 4080 (12 GB) is the primary target.

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
# For GPU PyTorch, follow https://pytorch.org/get-started/ to pick the right
# CUDA wheel, e.g.:
#   pip install torch --index-url https://download.pytorch.org/whl/cu121
```

## Reproduce

Each sweep writes a `results*.csv` under `results/<name>/`.

| Stage | Entry point | Runtime (RTX 4080) |
|-------|-------------|-------------------|
| Metric pilot | `python src/sweeps/stage0_metric_validation.py` | ~30 min |
| Coarse 5x5 grid | `python src/sweeps/stage1_coarse_sweep.py` | ~2 h |
| wd sweep (lr=1.6e-3) | `python src/sweeps/stage2_wd_sweep.py` | ~3 h |
| lr sweep (wd=2.5) | `python src/sweeps/stage3_lr_sweep.py` | ~3 h |
| Three-stage timing | `python src/sweeps/step2_circuit_sweep.py` | ~5 h |
| Multiplication (mod 53) | `python src/sweeps/stage4_mul_sweep.py` | ~2 h |
| Addition (mod 97) | `python src/sweeps/stage5_p97_sweep.py` | ~2 h |
| Dense 2D grid | `python src/sweeps/stage6_2d_sweep.py` | ~4 h |
| Cross-hardware (A800) | `python src/sweeps/e3_nanda_sweep.py` | ~10 h |

After any sweep, generate figures with the matching script in `src/analysis/`.

## Key results

| Metric | Value | Source |
|--------|-------|--------|
| G<F (embedding lags generalization) | 212/236 = 89.8% | aggregated across Stages 2-6 |
| C<G<F (full three-stage ordering) | 48/54 (primary) + 44/55 (cross-hardware) | Stage 4 + E3 |
| Causal specificity (logit Fourier removal) | 48x to 240x vs random control | E7 (8 cells) |

Research logs:
- English: [Research/research_notes_en.md](Research/research_notes_en.md)
- Chinese: [Research/研究说明文档.md](Research/研究说明文档.md)

## License

See [LICENSE](LICENSE).
