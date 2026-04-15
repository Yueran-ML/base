# Auto Review Log — G<F Grokking Paper

**Project**: Generalization Precedes Fourier Alignment in Grokking
**Loop started**: 2026-04-13
**MAX_ROUNDS**: 4
**Reviewer model**: gpt-5.4 (via Codex MCP)

---

## Round 1 (2026-04-13)

### Assessment (Summary)
- **Score**: 4/10
- **Verdict**: Not ready
- **Key criticisms**:
  1. Too narrow: single task, one modulus, two 1D slices
  2. Overreaches from embedding proxy to mechanistic claim (circuit not measured)
  3. Detector-sensitivity validated on only 8 cells
  4. "Phase diagram" language overstated for 1D slices
  5. F<G outliers need regime-conditional framing
  6. "Not necessary/not sufficient" language too strong

### Reviewer Raw Response

<details>
<summary>Click to expand full reviewer response</summary>

**Score**: `4/10` for NeurIPS/ICLR/ICML main track.

This is a careful and interesting mechanistic empirical note, but not yet a top-tier paper. The core result is plausible and the robustness analysis is better than average for grokking work, but the contribution is still too narrow, too proxy-dependent, and too weakly connected to mechanism to clear the bar.

**Critical Weaknesses, Ranked**

1. **The contribution is too narrow for a full top-tier paper.**  
   Everything rides on one toy task, one modulus, one small decoder-only architecture, and two 1D hyperparameter slices. "Generalization precedes embedding-level Fourier alignment in modular addition" is interesting, but as stated it is still a small fact about one canonical grokking setup, not a broadly important scientific result.  
   **Minimum fix**: Show the same ordering on at least one materially different grokking setting. The minimum acceptable add would be one more task or family, not just another modulus.

2. **The paper overreaches from an embedding proxy to a mechanistic claim.**  
   You do not measure the Fourier circuit directly. You measure an embedding-geometry statistic. That is not the same thing. So the stronger story about "circuit formation -> generalization -> embedding consolidation" is not established by this paper; stage 1 is imported from prior work and stage 2 is only a proxy.  
   **Minimum fix**: Either add direct circuit-level probes on a subset of runs, or aggressively reframe the paper as a result about *embedding-level Fourier geometry*, not about the grokking mechanism broadly.

3. **The headline ordering could still be detector-driven.**  
   `τ_gen` and `τ_F` are both operational definitions with many moving parts: thresholds, sustain rules, EMA, null correction, BIC changepoints, log-time fitting, 500-step sampling, and null updates every 5k steps. A median lag of ~7k is real, but some cells are close enough that detector design matters. Sensitivity on 8 cells helps, but it is not enough for a paper whose main claim is entirely about event ordering.  
   **Minimum fix**: Run denser temporal logging around the transition region, report uncertainty intervals for `Δτ` across all cells, and extend the sensitivity analysis beyond 8 representative cells.

4. **"Hyperparameter phase diagram" is overstated.**  
   Two 1D slices are not a convincing phase diagram. That language makes the paper sound more complete than it is, and it weakens trust in the boundary story.  
   **Minimum fix**: Either rename this as "hyperparameter slices" or add a sparse 2D grid around the grokking boundary.

5. **The outliers are more damaging than the paper seems to admit.**  
   The two `F<G` cases, especially the large negative-lag high-lr outlier, plus the `F_only` regime, imply the main ordering is not universal and may fail exactly where the dynamics are most informative. That should be a central part of the story, not a footnote.  
   **Minimum fix**: Add targeted reruns and more seeds around the low-lr and high-lr boundaries, then make the claim explicitly regime-conditional.

6. **The "not necessary / not sufficient" language is too strong.**  
   What you can support is that *this measured robust embedding-alignment metric* is neither synchronized with nor predictive enough for generalization. That is weaker than saying Fourier alignment itself is neither necessary nor sufficient.  
   **Minimum fix**: Narrow the claim to the measured metric and stop generalizing from proxy failure to mechanism failure.

**Ready for submission?** `No.`

For a workshop or a focused interpretability venue, this is much closer. For NeurIPS/ICLR/ICML main track, it still needs at minimum: one extra task or setting, tighter event-ordering validation, and either direct circuit evidence or a much more modest framing.

</details>

### Actions Taken (Round 1)

**Writing fixes (no compute)**:
- [ ] Reframe: title → "hyperparameter slices" or add grid; claim → "embedding-level Fourier geometry" not "mechanism"
- [ ] Narrow "not necessary / not sufficient" to measured metric only
- [ ] Reframe outliers as regime-conditional claim prominently

**Experiment fixes (compute required)**:
- [ ] Add second task: (a·b) mod 53 or (a+b) mod 97 sweep (Stage 2 equivalent, ~3h GPU)
- [ ] Extend sensitivity to all 55 cells (post-hoc, moderate compute)
- [ ] Add 2 extra seeds at phase boundary cells (low-lr and high-lr edges)

### Status
- Round 1 完成，写作修复已实施，进入 Round 2

---

## Round 2 (2026-04-13)

### Assessment (Summary)
- **Score**: 5/10（较 Round 1 提升 1 分）
- **Verdict**: Almost（但仍未达 NeurIPS/ICLR 主赛道水准）
- **关键结论**：写作修复有实质效果；核心瓶颈是实验范围太窄

### Reviewer Raw Response

<details>
<summary>点击展开完整审阅意见</summary>

**Score**: `5/10`

The writing fixes are real improvements. They remove the worst overclaims, clarify the proxy/mechanism distinction, and make the paper more intellectually honest. That matters. But they do not fully solve the main-track issue: the empirical claim is still narrow, and the measured object is still a task-specific embedding proxy rather than the circuit itself.

**Remaining Critical Weaknesses**

1. `Generality is still too weak.`
   Right now the result is still basically "in modular addition, under these sweeps, this embedding-alignment metric typically trails generalization." That is cleaner than before, but still narrow for a full NeurIPS/ICLR paper.
   **Minimum fix**: Finish the full `(a-b) mod 53` sweep. If it replicates strongly, you can support a scoped claim over the modular-addition/subtraction family.

2. `The metric is still family-specific, which limits the scientific reach.`
   Your multiplication pilot exposes a problem: the main metric does not transfer cleanly beyond the cyclic-addition family without nontrivial redesign.
   **Minimum fix**: Either adapt the metric to one non-addition setting, or explicitly narrow the paper's scope to tasks with the same cyclic Fourier structure.

3. `The central ordering claim is still detector/proxy-based.`
   Sensitivity on 8 cells helps, but this remains the paper's main vulnerability.
   **Minimum fix**: Extend sensitivity analysis beyond 8 cells; report uncertainty bands for Δτ due to 500-step sampling.

4. `Boundary behavior remains under-characterized.`
   F<G outliers and F_only regime still look like acknowledged exceptions rather than explained phenomena.
   **Minimum fix**: Extra seeds at low-lr and high-lr margins; present claim as interior-of-grokking-region behavior.

5. `Impact limited by no direct circuit-level evidence.`
   **Minimum fix (cheap)**: Reframe as "robust empirical dissociation"; or add circuit-level probe on subset of runs.

**Ready for submission?** `Almost`, but not for NeurIPS/ICLR main track.

**On (a-b) mod 53**: Partially yes, fully no for weakness #1.
- If subtraction replicates → enough to scope claim to cyclic-Fourier modular arithmetic family
- Not enough for broader grokking claim
- Highest-ROI next step; probably enough to make the paper credible in its narrowed form

</details>

### Actions Taken (Round 2)

**写作修复（已完成）**：
- [x] 明确将论文范围限定为"循环 Fourier 结构的模运算任务"
- [x] 乘法 pilot 结果注记（G_only；F_corr 不适用于乘法）
- [ ] 进一步缩窄 scope 到 cyclic-Fourier 任务族

**实验（进行中）**：
- [x] sub pilot（wd=2.18, seed=42）— 运行中
- [ ] 完整 sub 扫描（30 runs，~5h）— 待 pilot 结果确认后决定

### Status
- Round 2 完成，进入 Round 3

---

## Round 3 (2026-04-13)

### Assessment (Summary)
- **Score**: 5/10（持平）
- **Verdict**: Almost（workshop 可投；TMLR 需完成 sub 种子+速度分析）
- 速度假说方向正确但证据不足；减法 pilot（F<G）是有价值的新发现

### Reviewer Raw Response
<details>
<summary>展开</summary>

Score: 5/10. The speed-dependent ordering hypothesis would strengthen the paper if validated, but with current evidence (2 outliers + 1 pilot) it is post-hoc and confounded. Framing as "suggestive" is correct. Best fit: workshop now, TMLR after more evidence. Minimum fix: finish sub seeds, add τ_gen vs Δτ correlation analysis directly.

</details>

### Actions Taken (Round 3)
- [x] 生成 speed_ordering_scatter.png（τ_gen vs Δτ，r=-0.62，56 runs）
- [x] 论文 Limitations 节加入速度假说 + 图表引用
- [x] Caption 明确标注数学虚假相关的免责声明
- [x] 启动 sub 种子 7, 2025（后台运行）

### Status
- 进入 Round 4（最终轮）

---

## Round 4 (2026-04-13) — 最终轮

### Assessment (Summary)
- **Score**: 6/10
- **Verdict**: **TMLR 可投** ✅（达到 POSITIVE_THRESHOLD）
- 关键条件：保持 main claim 明确（G<F in modular addition interior），速度假说仅作 suggestive secondary

### Reviewer Raw Response
<details>
<summary>展开完整意见</summary>

**Final score**: 6/10

For TMLR specifically, this is now in plausible submission territory. The paper's strongest claim is no longer overstated: technically careful, clearly scoped empirical result about embedding-level Fourier alignment lagging generalization in modular addition across two sweep axes, with detector robustness and explicit limits. That matches TMLR's standard.

**Ready for TMLR?** Yes — if submitted with current narrow scope and speed-dependent hypothesis not promoted beyond "suggestive."

**Top 1-2 changes for TMLR acceptance**:
1. Finish sub seeds, stabilize secondary observation (if seeds don't replicate, keep subtraction out of main narrative)
2. Uncertainty analysis table across all runs (which remain G<F/F<G/ambiguous under reasonable perturbations)

**Hedging level**: Appropriate. Keep firm on addition result, hedged on speed interpretation, firm on scope limits.

**Bottom line**: "This is no longer a top-conference paper, but it is now a credible TMLR paper if it stays disciplined about what it has actually shown."

</details>

### Actions Taken (Round 4)
- 所有 4 轮审阅完成
- 论文已更新至可投稿状态

### Final Status: LOOP COMPLETE ✅

---

## Method Description

The paper studies the temporal relationship between two events during transformer grokking on (a+b) mod 53:
- **τ_gen**: first checkpoint where test acc ≥ 0.9 for 3 consecutive 500-step evaluations
- **τ_F**: BIC changepoint on EMA-smoothed (α=0.15), permutation-null-corrected Fourier alignment score F_corr(t)

F_corr(t) = max(0, F_raw(t) - F_null_p95(t)), where F_raw measures the fraction of embedding variance explained by the best single Fourier harmonic, and F_null_p95 is the 95th percentile under 100 random token-index permutations.

Two hyperparameter sweeps: wd ∈ [1.2, 3.5] at lr=1.6e-3 (Stage 2, 30 runs); lr ∈ [5e-4, 8e-3] at wd=2.5 (Stage 3, 30 runs). Sensitivity: 36 detector configs on 8 representative cells.

Main finding: G<F ordering (τ_gen < τ_F) in 52/55 Grokking runs (94.5%), median Δτ ≈ 7,000 steps.

---

## Score Progression

| Round | Score | Verdict | Key Action |
|-------|-------|---------|-----------|
| 1 | 4/10 | Not ready | 初始审阅 |
| 2 | 5/10 | Almost | 写作修复（6项） |
| 3 | 5/10 | Almost | Sub pilot + 速度假说 |
| 4 | **6/10** | **TMLR 可投** ✅ | 速度散点图 + 最终修订 |

---
