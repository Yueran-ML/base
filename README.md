# Grokking Phase Diagram — Representation Structure Timing

UQ EECS thesis project investigating the temporal ordering of three events
during grokking on modular arithmetic:

$$\tau_\text{circuit} \;<\; \tau_\text{gen} \;<\; \tau_F$$

where $\tau_\text{circuit}$ is when the Fourier logit structure forms,
$\tau_\text{gen}$ is when test accuracy crosses 90%, and $\tau_F$ is when
the embedding Fourier alignment $F_\text{corr}$ saturates.

Key finding: across 55 Grokking-phase runs sweeping learning rate and weight
decay at $p=53$, **52/55 (94.5%)** show the G<F ordering — embedding
structure visibly consolidates *after* generalization, not before.

Research brief: [docs/research_brief.md](docs/research_brief.md).

## Repository layout

```
src/
├── grok_metrics.py         # shared metric utilities (canonical)
├── grokking_baseline.py    # decoder-only transformer + training loop
├── sweeps/                 # experiment drivers (stage0 → stage6, step2, e3)
└── analysis/               # figure/table generation
paper/                      # thesis LaTeX (IEEEtran, \graphicspath{{figures/}})
proposal/                   # UQ proposal LaTeX
docs/                       # research brief + baseline explainer
Research/                   # Obsidian vault: progress log (中文 + EN) + canvases
results/                    # summary CSVs per canonical experiment
notebooks/                  # exploratory notebooks
```

Raw checkpoints (`.pt`) and per-seed trajectory `.npz` are **not** tracked
— regenerate via the sweep scripts (see §Reproduce).

## Setup

Python 3.11, Windows 11 + NVIDIA RTX 4080 (12 GB) is the primary target.

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
# For GPU PyTorch, follow https://pytorch.org/get-started/ to pick the right
# CUDA wheel, e.g.:
#   pip install torch --index-url https://download.pytorch.org/whl/cu121
```

## Reproduce

Each sweep writes a `results*.csv` under `runs/<name>/` (recreated on first run).

| Stage | Entry point | Runtime (4080) |
|-------|-------------|---------------|
| Metric pilot | `python src/sweeps/stage0_metric_validation.py` | ~30 min |
| Coarse 5×5 grid | `python src/sweeps/stage1_coarse_sweep.py` | ~2 h |
| wd sweep at lr=1.6e-3 | `python src/sweeps/stage2_wd_sweep.py` | ~3 h |
| lr sweep at wd=2.5 | `python src/sweeps/stage3_lr_sweep.py` | ~3 h |
| Stage 4 Grokking measurement | `python src/sweeps/step2_circuit_sweep.py` | ~5 h |
| Op generalization (×, p=97) | `python src/sweeps/stage5_p97_sweep.py` | ~2 h |
| 2-D grid | `python src/sweeps/stage6_2d_sweep.py` | ~4 h |
| Nanda (2023) comparison | `python src/sweeps/e3_nanda_sweep.py --parallel 6 --grokking-only` | ~10 h |

After any sweep, generate figures with the matching script in
`src/analysis/` (e.g. `src/analysis/make_step2_figures.py`).

## Canonical findings

| Metric | Value | Source |
|--------|-------|--------|
| G<F fraction (Grokking only) | 52/55 = 94.5% | `results/step2_circuit/` |
| Δτ = τ_F − τ_gen range | 1,000 – 25,500 steps | `paper/figures/delta_histogram.png` |
| C<G<F fraction | 53/55 = 96.4% | `results/step2_circuit/` |

See [Research/研究说明文档.md](Research/研究说明文档.md) for the full
chronological log (Steps 1–28).

## Citation

If you build on this code, cite the thesis (available in `paper/`).
AI assistance: see [CLAUDE.md](CLAUDE.md) for the collaboration protocol
used throughout the project.
