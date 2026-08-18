"""Summarise the slow-grokking targeted sweep and emit a verdict on the
strong form of the speed-dependent ordering hypothesis (Section 7).

Strong form prediction:
  As tau_gen grows past ~35,000 steps (the slow-grokking margin), the
  F<G ordering should become systematic rather than anecdotal.

Decision rules:
  STRONG_SUPPORTED   : >=2/3 of cells with tau_gen >= 40,000 show F<G,
                       AND <= 1/N cells with tau_gen <= 25,000 show F<G.
  WEAK_SUPPORTED     : F<G fraction increases monotonically with tau_gen
                       binned into terciles, but the high-tau_gen tercile
                       does not exceed 50% F<G.
  NOT_SUPPORTED      : F<G is no more frequent in the high-tau_gen tercile
                       than in the low-tau_gen tercile.
  UNPOWERED          : fewer than 6 Grokking cells reach tau_gen >= 35,000.

Inputs : results/slow_grokking/results.csv
Outputs:
  console summary
  results/slow_grokking/summary.md
"""

from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path

CSV_PATH = Path("results/slow_grokking/results.csv")
OUTDIR = CSV_PATH.parent

SLOW_MARGIN = 35_000
HIGH_TAU_GEN = 40_000
LOW_TAU_GEN = 25_000


def _f(s):
    if s in (None, "", "None"):
        return None
    try:
        return float(s)
    except Exception:
        return None


def main():
    if not CSV_PATH.exists():
        print(f"  MISSING: {CSV_PATH}")
        print("  Run: python src/sweeps/slow_grokking_sweep.py first.")
        return

    rows = list(csv.DictReader(open(CSV_PATH, encoding="utf-8")))
    if not rows:
        print("Empty CSV.")
        return

    print(f"Loaded {len(rows)} rows from {CSV_PATH}")

    # Phase distribution
    phases = defaultdict(int)
    for r in rows:
        phases[r.get("observed_phase", "?")] += 1
    print(f"Phase distribution: {dict(phases)}")

    grok = [r for r in rows if r.get("observed_phase") == "Grokking"]
    print(f"Grokking cells with both tau_gen and tau_F detected: ", end="")
    grok_full = [r for r in grok
                 if _f(r["tau_gen"]) is not None
                 and _f(r["tau_F"]) is not None]
    print(f"{len(grok_full)}/{len(grok)}")

    if not grok_full:
        print("No Grokking cells with both onsets. Nothing to test.")
        return

    # Per-cell ordering breakdown
    counts = defaultdict(int)
    for r in grok_full:
        counts[r["ordering_gf"]] += 1
    print(f"Ordering: {dict(counts)}")

    # Slow-grokking cells
    slow = [r for r in grok_full
            if _f(r["tau_gen"]) >= SLOW_MARGIN]
    fast = [r for r in grok_full
            if _f(r["tau_gen"]) < SLOW_MARGIN]
    print(f"\ntau_gen >= {SLOW_MARGIN} (slow margin): {len(slow)} cells")
    print(f"tau_gen <  {SLOW_MARGIN}                    : {len(fast)} cells")

    def fg_rate(cells):
        if not cells:
            return float("nan"), 0
        n = len(cells)
        n_fg = sum(1 for r in cells if r["ordering_gf"] == "F<G")
        return n_fg / n, n_fg

    rate_slow, n_fg_slow = fg_rate(slow)
    rate_fast, n_fg_fast = fg_rate(fast)
    print(f"  F<G in slow: {n_fg_slow}/{len(slow)} "
          f"({rate_slow*100:.1f}%)" if slow else
          "  F<G in slow: n/a (no slow cells)")
    print(f"  F<G in fast: {n_fg_fast}/{len(fast)} "
          f"({rate_fast*100:.1f}%)" if fast else
          "  F<G in fast: n/a (no fast cells)")

    # Tercile binning
    tau_gens = sorted([_f(r["tau_gen"]) for r in grok_full])
    n = len(tau_gens)
    if n >= 6:
        t1 = tau_gens[n // 3 - 1]
        t2 = tau_gens[2 * n // 3 - 1]
        low = [r for r in grok_full if _f(r["tau_gen"]) <= t1]
        mid = [r for r in grok_full if t1 < _f(r["tau_gen"]) <= t2]
        high = [r for r in grok_full if _f(r["tau_gen"]) > t2]
        rate_low, _ = fg_rate(low)
        rate_mid, _ = fg_rate(mid)
        rate_high, _ = fg_rate(high)
        print(f"\nTercile breakdown:")
        print(f"  low  (tau_gen <= {int(t1)}):     "
              f"F<G {rate_low*100:.0f}% ({sum(1 for r in low if r['ordering_gf']=='F<G')}/{len(low)})")
        print(f"  mid  ({int(t1)} < tau_gen <= {int(t2)}): "
              f"F<G {rate_mid*100:.0f}% ({sum(1 for r in mid if r['ordering_gf']=='F<G')}/{len(mid)})")
        print(f"  high (tau_gen > {int(t2)}):     "
              f"F<G {rate_high*100:.0f}% ({sum(1 for r in high if r['ordering_gf']=='F<G')}/{len(high)})")

    # Verdict
    print()
    high_cells = [r for r in grok_full if _f(r["tau_gen"]) >= HIGH_TAU_GEN]
    low_cells = [r for r in grok_full if _f(r["tau_gen"]) <= LOW_TAU_GEN]
    rate_high_explicit, n_fg_h = fg_rate(high_cells)
    rate_low_explicit, n_fg_l = fg_rate(low_cells)

    if len(slow) < 6:
        verdict = "UNPOWERED"
        reason = (f"only {len(slow)} cells reach tau_gen >= {SLOW_MARGIN}; "
                  f"need >= 6 for a meaningful test.")
    elif (rate_high_explicit >= 2/3
          and (not low_cells or rate_low_explicit <= 1/max(len(low_cells), 1))):
        verdict = "STRONG_SUPPORTED"
        reason = (f"F<G in {n_fg_h}/{len(high_cells)} of cells with "
                  f"tau_gen >= {HIGH_TAU_GEN}; only {n_fg_l}/"
                  f"{len(low_cells)} below {LOW_TAU_GEN}.")
    elif n >= 6 and rate_high > rate_low > 0:
        verdict = "WEAK_SUPPORTED"
        reason = (f"F<G frequency increases with tau_gen (low {rate_low*100:.0f}%, "
                  f"mid {rate_mid*100:.0f}%, high {rate_high*100:.0f}%) but "
                  f"high tercile does not exceed 50%.")
    elif rate_high_explicit <= rate_low_explicit:
        verdict = "NOT_SUPPORTED"
        reason = (f"F<G fraction at tau_gen >= {HIGH_TAU_GEN} is not "
                  f"greater than at tau_gen <= {LOW_TAU_GEN}.")
    else:
        verdict = "AMBIGUOUS"
        reason = (f"Pattern does not cleanly fit any rule; manual review "
                  f"recommended.")

    print(f"=== Verdict: {verdict} ===")
    print(f"Reason: {reason}")

    # Markdown summary
    out_md = OUTDIR / "summary.md"
    with open(out_md, "w", encoding="utf-8") as f:
        f.write("# Slow-grokking targeted sweep — summary\n\n")
        f.write(f"**Verdict**: {verdict}\n\n")
        f.write(f"_Reason_: {reason}\n\n")
        f.write("## Counts\n\n")
        f.write(f"- Total cells: {len(rows)}\n")
        f.write(f"- Grokking with both onsets: {len(grok_full)}\n")
        f.write(f"- F<G in slow (tau_gen >= {SLOW_MARGIN}): "
                f"{n_fg_slow}/{len(slow)}\n")
        f.write(f"- F<G in fast (tau_gen <  {SLOW_MARGIN}): "
                f"{n_fg_fast}/{len(fast)}\n\n")
        if n >= 6:
            f.write("## Tercile breakdown\n\n")
            f.write("| tercile | tau_gen range | F<G rate |\n")
            f.write("|---|---|---|\n")
            f.write(f"| low  | <= {int(t1)} | {rate_low*100:.0f}% "
                    f"({sum(1 for r in low if r['ordering_gf']=='F<G')}/{len(low)}) |\n")
            f.write(f"| mid  | ({int(t1)}, {int(t2)}] | {rate_mid*100:.0f}% "
                    f"({sum(1 for r in mid if r['ordering_gf']=='F<G')}/{len(mid)}) |\n")
            f.write(f"| high | > {int(t2)} | {rate_high*100:.0f}% "
                    f"({sum(1 for r in high if r['ordering_gf']=='F<G')}/{len(high)}) |\n")
        f.write("\n## Per-cell rows\n\n")
        f.write("| lr | seed | tau_gen | tau_F | delta | ordering | phase |\n")
        f.write("|---|---|---|---|---|---|---|\n")
        for r in sorted(grok_full,
                        key=lambda r: (_f(r["lr"]), int(r["seed"]))):
            tg = _f(r["tau_gen"]); tf = _f(r["tau_F"])
            delta = tf - tg if (tg is not None and tf is not None) else None
            f.write(f"| {float(r['lr']):.2e} | {r['seed']} | "
                    f"{int(tg) if tg else '--'} | "
                    f"{int(tf) if tf else '--'} | "
                    f"{int(delta) if delta else '--'} | "
                    f"{r['ordering_gf']} | {r['observed_phase']} |\n")
    print(f"\nWrote summary to {out_md}")


if __name__ == "__main__":
    main()
