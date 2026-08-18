"""Summarise causal Fourier-intervention results and emit a paper-ready
verdict on the proxy-vs-mechanism question.

Reads:  results/causal_probe/intervention_results.csv

Produces a per-cell, per-regime table of accuracy drops:
  drop_I2 = base_acc - acc_I2_remove_fourier      (drop when removing Fourier)
  drop_I3 = base_acc - acc_I3_random_ctrl_mean    (drop under random control)
  drop_I4 = base_acc - acc_I4_remove_emb_fourier  (drop when removing
                                                   embedding Fourier)
  spec_ratio_logit = drop_I2 / max(drop_I3, 1e-3)  (Fourier-specificity)

Decision rules for the four reviewer-relevant regime patterns (per cell):

  (A) STRONG MECHANISM:
      drop_I2 >= 0.40 AND spec_ratio_logit >= 3.0 in regime 'between_circ_*'
      (between tau_circuit and the next event)
      AND drop_I2 < 0.10 in regime 'pre_all'
      => Fourier subspace is causally load-bearing once tau_circ is reached.

  (B) LAGGING / POST-HOC:
      drop_I2 < 0.20 in regime between tau_circ and tau_gen (Fourier alignment
      detected but accuracy doesn't depend on it yet) AND drop_I2 >= 0.40
      after tau_F.
      => Fourier subspace is a post-hoc readout, not the active mechanism.

  (C) PROXY FAILURE:
      drop_I2 ~ drop_I3 (spec_ratio_logit < 1.5) at all regimes.
      => Model isn't preferentially using Fourier; logit-Fourier is a
      correlate, not a mechanism.

  (D) NOISY: none of (A)/(B)/(C) cleanly applies.

The script prints the verdict per cell, the aggregate across cells, and a
suggested paragraph for §6 of the paper.
"""

from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path


CSV_PATH = "results/causal_probe/intervention_results.csv"
RESULTS_DIR = Path("results/causal_probe")


def _f(s):
    if s in (None, "", "None"):
        return None
    try:
        return float(s)
    except Exception:
        return None


def classify_regime(regime: str) -> str:
    """Map the raw regime label produced by causal_probe_checkpoints.py to a
    canonical bucket."""
    if regime == "pre_all":
        return "pre_all"
    if regime == "post_all":
        return "post_all"
    # Between events: extract first event name
    # regime looks like 'between_circ_gen' or 'between_F_gen' etc
    parts = regime.replace("between_", "").split("_")
    if len(parts) == 2:
        return f"between({parts[0]}->{parts[1]})"
    return regime


def per_cell_verdict(cell_rows: list[dict]) -> tuple[str, str]:
    """Return (verdict, reason) for one cell's rows."""
    rows_by_regime = {classify_regime(r["regime"]): r for r in cell_rows}

    def drop(r, intervention_field):
        return _f(r["base_acc"]) - _f(r[intervention_field])

    # Pre-all baseline
    pre = rows_by_regime.get("pre_all")
    post = rows_by_regime.get("post_all")
    if pre is None or post is None:
        return ("D", "missing pre_all or post_all checkpoint")

    drop_I2_pre = drop(pre, "acc_I2_remove_fourier")
    drop_I2_post = drop(post, "acc_I2_remove_fourier")
    drop_I3_post = drop(post, "acc_I3_random_ctrl_mean")
    spec_ratio_post = (drop_I2_post / max(drop_I3_post, 1e-3))

    # Between checkpoints (sorted by step)
    between_rows = sorted(
        [(k, v) for k, v in rows_by_regime.items() if k.startswith("between(")],
        key=lambda kv: int(kv[1]["step"]),
    )

    # (A) STRONG MECHANISM: post-circuit drop is large AND specific, pre-all is small
    if drop_I2_post >= 0.40 and spec_ratio_post >= 3.0 and drop_I2_pre < 0.10:
        if between_rows:
            first_between = between_rows[0][1]
            drop_I2_between = drop(first_between, "acc_I2_remove_fourier")
            drop_I3_between = drop(first_between, "acc_I3_random_ctrl_mean")
            spec_between = drop_I2_between / max(drop_I3_between, 1e-3)
            if drop_I2_between >= 0.30 and spec_between >= 2.0:
                return ("A_STRONG_MECHANISM",
                        f"drop_I2 grows from {drop_I2_pre:.2f} (pre) "
                        f"to {drop_I2_between:.2f} "
                        f"(between, spec ratio {spec_between:.1f}) to "
                        f"{drop_I2_post:.2f} (post, spec ratio "
                        f"{spec_ratio_post:.1f})")
        return ("A_STRONG_MECHANISM",
                f"drop_I2 grows from {drop_I2_pre:.2f} pre to "
                f"{drop_I2_post:.2f} post (spec ratio {spec_ratio_post:.1f})")

    # (B) LAGGING / POST-HOC
    if (drop_I2_post >= 0.40 and spec_ratio_post >= 2.0
            and any(drop(r, "acc_I2_remove_fourier") < 0.20
                    for _, r in between_rows)):
        return ("B_LAGGING_INDICATOR",
                f"between-regime drop_I2 stays low (<0.20) but post-all is "
                f"{drop_I2_post:.2f} (spec ratio {spec_ratio_post:.1f})")

    # (C) PROXY FAILURE: I2 drop ~ I3 drop everywhere
    if all(abs(drop(r, "acc_I2_remove_fourier")
               - drop(r, "acc_I3_random_ctrl_mean")) < 0.10
           for _, r in between_rows + [("post_all", post)]
           if r is not None):
        return ("C_PROXY_FAILURE",
                f"I2 drop tracks I3 drop at all regimes "
                f"(spec ratio post = {spec_ratio_post:.1f})")

    # (D) NOISY
    return ("D_NOISY",
            f"pre={drop_I2_pre:.2f} post={drop_I2_post:.2f} "
            f"spec_post={spec_ratio_post:.1f}")


def main():
    p = Path(CSV_PATH)
    if not p.exists():
        print(f"  MISSING: {CSV_PATH}")
        print("  Run causal_probe_checkpoints.py + "
              "causal_fourier_intervention.py first.")
        return
    rows = list(csv.DictReader(open(p)))
    if not rows:
        print("Empty CSV.")
        return

    by_cell = defaultdict(list)
    for r in rows:
        by_cell[r["label"]].append(r)

    print(f"=== Causal Fourier intervention summary ===")
    print(f"Cells analysed: {len(by_cell)}")
    print()

    verdicts = {}
    for label in sorted(by_cell):
        cell_rows = sorted(by_cell[label], key=lambda r: int(r["step"]))
        print(f"--- {label} ({cell_rows[0]['ordering']}) ---")
        for r in cell_rows:
            base = _f(r["base_acc"])
            i1 = _f(r["acc_I1_keep_fourier"])
            i2 = _f(r["acc_I2_remove_fourier"])
            i3 = _f(r["acc_I3_random_ctrl_mean"])
            i4 = _f(r["acc_I4_remove_emb_fourier"])
            efrac = _f(r["frac_logit_energy_in_topK_fourier"])
            print(f"  step={r['step']:>6s} regime={r['regime']:<25s} "
                  f"base={base:.3f}  "
                  f"I1_keep={i1:.3f}  "
                  f"I2_rm={i2:.3f}  "
                  f"I3_ctrl={i3:.3f}  "
                  f"I4_emb={i4:.3f}  "
                  f"E_frac={efrac:.3f}  "
                  f"top_k={r['top_k_harmonics']}")
        verdict, reason = per_cell_verdict(cell_rows)
        verdicts[label] = (verdict, reason)
        print(f"  verdict: {verdict}  ({reason})")
        print()

    # Aggregate verdict
    counts = defaultdict(int)
    for v, _ in verdicts.values():
        counts[v.split("_")[0]] += 1
    print("=== Aggregate verdict counts ===")
    for v, n in sorted(counts.items()):
        print(f"  {v}: {n}/{len(verdicts)}")

    # Suggest paper paragraph based on most common verdict
    most_common = max(counts, key=counts.get)
    print()
    print("=== Suggested paper §6 paragraph (draft, edit before commit) ===")
    if most_common == "A":
        print("STRONG_MECHANISM: removing the dominant top-K Fourier")
        print("subspace from logits collapses test accuracy in the regime")
        print("between tau_circuit and tau_gen, and the drop is specific")
        print("(matched-energy random ablation has substantially smaller")
        print("effect). This upgrades tau_circuit from a logit-level proxy")
        print("to a load-bearing mechanism.")
    elif most_common == "B":
        print("LAGGING_INDICATOR: removing the dominant Fourier subspace")
        print("from logits leaves test accuracy nearly intact in the")
        print("regime between tau_circuit and tau_gen, but the same")
        print("intervention tanks accuracy after tau_F. This is consistent")
        print("with the lagging-indicator interpretation: F_logit detects")
        print("an emerging Fourier readout that is not yet load-bearing")
        print("during generalization onset, but becomes the dominant")
        print("decoding axis post-tau_F.")
    elif most_common == "C":
        print("PROXY_FAILURE: removing the dominant Fourier subspace")
        print("drops accuracy by amounts indistinguishable from a")
        print("matched-energy random ablation. F_logit is therefore a")
        print("correlated readout, not a mechanism. The C<G<F claim must")
        print("be retracted to a logit-Fourier-alignment claim only.")
    else:
        print("NOISY: results do not cleanly identify a single mechanism")
        print("regime. Consider running additional cells or seeds before")
        print("interpreting causally.")

    out_md = RESULTS_DIR / "intervention_summary.md"
    with open(out_md, "w") as f:
        f.write("# Causal Fourier intervention summary\n\n")
        f.write(f"Cells analysed: {len(by_cell)}\n\n")
        f.write("| cell | ordering | verdict | reason |\n")
        f.write("|---|---|---|---|\n")
        for label, (v, r) in verdicts.items():
            ordering = by_cell[label][0]["ordering"]
            f.write(f"| {label} | {ordering} | {v} | {r} |\n")
        f.write(f"\n**Aggregate**: " + ", ".join(
            f"{k}={n}" for k, n in sorted(counts.items())) + "\n")
    print(f"\nWrote summary to {out_md}")


if __name__ == "__main__":
    main()
