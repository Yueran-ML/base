# Supplementary Material — Anonymous TMLR Submission

This package accompanies the anonymous TMLR submission
*A Three-Stage Timing Picture of Grokking: Logit Fourier Alignment,
Generalization, and Embedding Geometry Consolidation*.

It contains the source code, per-run metric trajectories, and
post-hoc analysis scripts needed to reproduce every figure and
table in the paper. Reviewers may inspect any subset; nothing in
the main paper depends on the appendix or this archive.

---

## Layout

```
supplementary.zip
├── README.md                          (this file)
├── requirements.txt                    (top-level Python deps)
├── run_commands.md                     (commands behind every result)
├── src/
│   ├── grokking_baseline.py           (training loop)
│   ├── grok_metrics.py                (canonical metric utilities)
│   ├── sweeps/                        (one script per experimental stage)
│   └── analysis/                      (post-hoc analysis & figure scripts)
└── results/
    ├── stage0_v3/, stage1_coarse/, stage2_wd/, stage3_lr/,
    ├── stage4_mul/, stage4_sub/, stage5_mul_dlog/, stage5_p97/,
    ├── stage5a_fpstar/, stage6_2d/, step2_circuit/,
    ├── e4/, e5/, slow_grokking/, sensitivity/, causal_probe/,
    └── posthoc/                       (dense .npz trajectories + post-hoc CSVs)
```

`results/posthoc/` contains the 8 per-cell `.npz` trajectory dumps
plus the per-CSV outputs of the post-hoc analyses (Q2/Q3/Q8, E6
bootstrap, multi-harmonic recompute, cluster bootstrap).

## Software environment

Reproduced on:

- Python `3.11.9`
- PyTorch `2.5.1+cu121`
- NumPy `2.4.3`
- CUDA `12.1`, cuDNN `9.1.0`

Top-level pinned versions are in `requirements.txt`. PyTorch should
be installed first to match the user's CUDA toolkit. See
`run_commands.md` for stage-by-stage invocation.

## Hardware

- Local: NVIDIA RTX 4080 Laptop (12 GB VRAM)
- Cloud (subset of long sweeps): NVIDIA RTX PRO 6000 / NVIDIA A800

`torch.backends.cudnn.deterministic=True` is set in the seeding
routine. Bit-exact reproducibility holds within a single
GPU/CUDA configuration but not across them; the cross-hardware
variance band ($\sim$9 percentage points on single-run point
estimates) is documented in the main paper, Section "Post-hoc
Robustness", E3.

## Anonymity

This archive is anonymized for double-blind review:

- No author names, affiliations, emails, ORCIDs, or commit hashes
  identifying contributors are included.
- The paper's GitHub URL placeholder is `ANONYMISED`; this archive
  is the supplementary equivalent. A de-anonymized public
  repository will be released after acceptance.

## License

Code and CSVs in this archive are released for review purposes
under the same terms as the main project (`LICENSE` in the
upstream repository). After acceptance the de-anonymized
repository will carry an explicit OSI license.
