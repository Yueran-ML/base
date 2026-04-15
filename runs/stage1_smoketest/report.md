# Stage 1: Phase Dynamics Report
## Onset Estimates
| Cell | Seed | Observed Phase | τ_gen | τ_F | τ_eS | F@50k | eS_drop |
|------|------|---------------|-------|-----|------|-------|--------|
| grokking | 42 | Memorization | — | — | — | 0.042 | 46.7→46.6 |

## Q1: 修订指标能否区分 Grokking 和 Memorization？
- **F(t) final value**: Grokking=0.042, Memorization=N/A -> similar
- **eS final value**: Grokking=46.589, Memorization=N/A -> similar
- **dec_norm final**: Grokking=11.520, Memorization=N/A -> similar
- **emb_norm final**: Grokking=15.726, Memorization=N/A -> similar
- **gen_gap final**: Grokking=8.557, Memorization=N/A -> similar
- **τ_F (median)**: Grokking=None, Memorization=None
- **τ_eS (median)**: Grokking=None, Memorization=None

## Q2: Collapse 与普通 Memorization 的动力学差异
- **Collapse observed phases**: []
- **Memorization observed phases**: []
- **dec_norm final**: Collapse=N/A, Memorization=N/A
- **emb_norm final**: Collapse=N/A, Memorization=N/A
- **train_acc final**: Collapse=N/A, Memorization=N/A

注：Collapse 的核心特征应是 dec_norm 快速归零（weight decay >> lr），导致 train_acc 无法提升，与 Memorization（train_acc→1 但 test_acc 不提升）形成对比。

## Runtime
- grokking seed=42: 5.5s
