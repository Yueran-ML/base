# Stage 1: Phase Dynamics Report
## Onset Estimates
| Cell | Seed | Observed Phase | τ_gen | τ_F | τ_eS | F@50k | eS_drop |
|------|------|---------------|-------|-----|------|-------|--------|
| grokking | 42 | Grokking | 23800.0 | 34800.0 | 15800.0 | 0.142 | 46.7→25.5 |
| grokking | 7 | Grokking | 18200.0 | 22000.0 | 19300.0 | 0.162 | 47.0→24.2 |
| grokking | 2025 | Grokking | 30000.0 | 28100.0 | 14600.0 | 0.150 | 46.9→26.6 |
| memorization | 42 | Memorization | — | 17100.0 | 8600.0 | 0.043 | 46.7→47.3 |
| memorization | 7 | Memorization | — | 17200.0 | 8500.0 | 0.044 | 47.0→47.5 |
| memorization | 2025 | Memorization | — | 23400.0 | 8300.0 | 0.045 | 46.9→47.4 |
| collapse | 42 | Memorization | — | 9400.0 | 9500.0 | 0.745 | 46.7→1.8 |
| collapse | 7 | Grokking | 24900.0 | 8900.0 | — | 0.462 | 47.0→4.4 |
| collapse | 2025 | Memorization | — | 8300.0 | — | 0.049 | 46.9→1.1 |

## Q1: 修订指标能否区分 Grokking 和 Memorization？
- **F(t) final value**: Grokking=0.150, Memorization=0.044 -> DISCRIMINATIVE
- **eS final value**: Grokking=25.535, Memorization=47.431 -> DISCRIMINATIVE
- **dec_norm final**: Grokking=10.467, Memorization=62.194 -> DISCRIMINATIVE
- **emb_norm final**: Grokking=4.589, Memorization=15.817 -> DISCRIMINATIVE
- **gen_gap final**: Grokking=0.025, Memorization=149.058 -> DISCRIMINATIVE
- **τ_F (median)**: Grokking=28100.0, Memorization=17200.0
- **τ_eS (median)**: Grokking=15800.0, Memorization=8500.0

## Q2: Collapse 与普通 Memorization 的动力学差异
- **Collapse observed phases**: ['Memorization', 'Grokking', 'Memorization']
- **Memorization observed phases**: ['Memorization', 'Memorization', 'Memorization']
- **dec_norm final**: Collapse=4.8538, Memorization=62.1935
- **emb_norm final**: Collapse=1.6369, Memorization=15.8172
- **train_acc final**: Collapse=0.844, Memorization=1.000

注：Collapse 的核心特征应是 dec_norm 快速归零（weight decay >> lr），导致 train_acc 无法提升，与 Memorization（train_acc→1 但 test_acc 不提升）形成对比。

## Runtime
- grokking seed=42: 467.4s
- grokking seed=7: 525.7s
- grokking seed=2025: 578.1s
- memorization seed=42: 566.4s
- memorization seed=7: 476.0s
- memorization seed=2025: 467.2s
- collapse seed=42: 2672.6s
- collapse seed=7: 586.6s
- collapse seed=2025: 530.4s
