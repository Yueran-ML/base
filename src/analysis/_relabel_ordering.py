"""Re-derive `ordering` column under canonical |delta|<=500 -> coincident rule.

Scans all results CSVs; for each row with both tau_gen and tau_F detected,
recomputes ordering as one of {G<F, F<G, coincident, F_only, G_only, none}
using the same conventions as the paper text:
    coincident if |delta| <= 500
    G<F        if delta >  500
    F<G        if delta < -500
    G_only     if tau_gen but not tau_F
    F_only     if tau_F but not tau_gen
    none       if neither
Reports any row whose stored label disagrees with the canonical recomputation.
Writes back the canonical label to a sibling .recanon.csv (does not overwrite).
"""
from __future__ import annotations

import csv
from pathlib import Path

CSVS = [
    "results/stage2_wd/results.csv",
    "results/stage3_lr/results.csv",
    "results/stage4_mul/results_stage4_mul.csv",
    "results/stage4_sub/results_stage4_sub.csv",
    "results/stage5_mul_dlog/results_stage5_mul_dlog.csv",
    "results/stage5_p97/results_stage5_p97.csv",
    "results/stage6_2d/results_stage6_2d.csv",
    "results/sensitivity/results.csv",
    "results/e4/results.csv",
    "results/e5/results.csv",
]

COINC_BIN = 500


def _f(s):
    if s == "" or s is None:
        return None
    try:
        return float(s)
    except Exception:
        return None


def canonical_ordering(tau_gen, tau_F):
    if tau_gen is None and tau_F is None:
        return "none"
    if tau_gen is None:
        return "F_only"
    if tau_F is None:
        return "G_only"
    delta = tau_F - tau_gen
    if abs(delta) <= COINC_BIN:
        return "coincident"
    return "G<F" if delta > 0 else "F<G"


def relabel(csv_path: Path):
    rows = list(csv.DictReader(open(csv_path, encoding="utf-8")))
    if not rows or "ordering" not in rows[0]:
        return None
    diffs = []
    for r in rows:
        tg = _f(r.get("tau_gen"))
        tf = _f(r.get("tau_F"))
        canon = canonical_ordering(tg, tf)
        old = r.get("ordering", "")
        if canon != old:
            diffs.append((r, old, canon))
        r["ordering"] = canon
    if diffs:
        out = csv_path.with_suffix(".recanon.csv")
        with open(out, "w", newline="", encoding="utf-8") as fout:
            w = csv.DictWriter(fout, fieldnames=rows[0].keys())
            w.writeheader()
            w.writerows(rows)
        return diffs, out
    return [], None


def main():
    total_changes = 0
    for c in CSVS:
        p = Path(c)
        if not p.exists():
            print(f"  MISSING {c}")
            continue
        result = relabel(p)
        if result is None:
            print(f"  NO ordering col {c}")
            continue
        diffs, out = result
        if not diffs:
            print(f"  CLEAN {c}")
            continue
        print(f"  {c}: {len(diffs)} rows relabelled -> {out}")
        total_changes += len(diffs)
        for r, old, new in diffs[:5]:
            tg = r.get("tau_gen")
            tf = r.get("tau_F")
            wd = r.get("wd", "?")
            seed = r.get("seed", "?")
            print(f"    wd={wd} seed={seed}: tau_gen={tg} tau_F={tf} -> {old} ==> {new}")
    print(f"\nTOTAL changes: {total_changes}")


if __name__ == "__main__":
    main()
