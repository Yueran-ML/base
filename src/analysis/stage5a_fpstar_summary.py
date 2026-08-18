"""Summarise Stage 5A F_p* recompute results and emit paper-ready numbers.

Reads results/stage5a_fpstar/results.csv and reports:
  - n cells with both tau_F and tau_gen_full / tau_gen_fpstar detected
  - distribution of (tau_gen_fpstar - tau_gen_full)
  - Delta_tau distribution under both evaluation domains
  - G<F count under both
  - cells whose ordering label flips between the two domains
  - the smallest |Delta_tau_fpstar| (the cell most at risk of flipping)
"""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np


def _f(s):
    if s in (None, "", "None"):
        return None
    try:
        return float(s)
    except Exception:
        return None


def main(csv_path="results/stage5a_fpstar/results.csv"):
    p = Path(csv_path)
    if not p.exists():
        print(f"  MISSING: {csv_path}")
        print("  Run: .venv/Scripts/python.exe src/sweeps/stage5a_fpstar_recompute.py")
        return
    rows = list(csv.DictReader(open(p, encoding="utf-8")))
    if not rows:
        print("  Empty CSV.")
        return

    n = len(rows)
    grok = [r for r in rows if r.get("observed_phase") == "Grokking"]
    print(f"Total cells: {n}; Grokking: {len(grok)}")

    detected_both = []
    for r in grok:
        tg_full = _f(r.get("tau_gen_full"))
        tg_fps = _f(r.get("tau_gen_fpstar"))
        tF = _f(r.get("tau_F"))
        if tg_full is None or tg_fps is None or tF is None:
            continue
        detected_both.append({
            "lr": r["lr"], "wd": r["wd"], "seed": r["seed"],
            "tau_gen_full": tg_full, "tau_gen_fpstar": tg_fps, "tau_F": tF,
            "delta_full": tF - tg_full,
            "delta_fpstar": tF - tg_fps,
            "ordering_full": r.get("ordering_full"),
            "ordering_fpstar": r.get("ordering_fpstar"),
        })

    if not detected_both:
        print("  No cells with all three of (tau_gen_full, tau_gen_fpstar, tau_F).")
        return

    n_both = len(detected_both)
    print(f"\nCells with all three onsets detected: {n_both}/{len(grok)}")

    shifts = np.array([r["tau_gen_fpstar"] - r["tau_gen_full"]
                       for r in detected_both])
    print(f"\ntau_gen_fpstar - tau_gen_full (steps):")
    print(f"  median: {int(np.median(shifts))}")
    print(f"  mean:   {int(np.mean(shifts))}")
    print(f"  min:    {int(np.min(shifts))}")
    print(f"  max:    {int(np.max(shifts))}")
    print(f"  fraction >= 0:  {float(np.mean(shifts >= 0)):.2f}")
    print(f"  fraction == 0:  {float(np.mean(shifts == 0)):.2f}")

    deltas_full = np.array([r["delta_full"] for r in detected_both])
    deltas_fps = np.array([r["delta_fpstar"] for r in detected_both])

    def _summary(arr, name):
        return (f"  {name}: median={int(np.median(arr))}, "
                f"min={int(np.min(arr))}, max={int(np.max(arr))}, "
                f"GF (Δ>500)={int(np.sum(arr > 500))}/{len(arr)}, "
                f"FG (Δ<-500)={int(np.sum(arr < -500))}/{len(arr)}, "
                f"coincident={int(np.sum(np.abs(arr) <= 500))}/{len(arr)}")

    print("\nDelta_tau under each evaluation domain:")
    print(_summary(deltas_full, "delta_full"))
    print(_summary(deltas_fps, "delta_fpstar"))

    # Cells with ordering label flips
    flips = [r for r in detected_both
             if r["ordering_full"] != r["ordering_fpstar"]
             and r["ordering_full"] and r["ordering_fpstar"]]
    print(f"\nCells whose ordering label flips between full and F_p*: "
          f"{len(flips)}/{n_both}")
    for r in flips:
        print(f"  wd={r['wd']} seed={r['seed']}: "
              f"{r['ordering_full']} -> {r['ordering_fpstar']} "
              f"(Δ_full={r['delta_full']:.0f} Δ_fps={r['delta_fpstar']:.0f})")

    # Smallest |delta_fpstar| -- the cell closest to coincident under F_p*
    min_idx = int(np.argmin(np.abs(deltas_fps)))
    min_cell = detected_both[min_idx]
    print(f"\nSmallest |Delta_tau_fpstar|: "
          f"|Δ|={abs(deltas_fps[min_idx]):.0f} steps "
          f"at wd={min_cell['wd']} seed={min_cell['seed']} "
          f"(tau_gen_fpstar={min_cell['tau_gen_fpstar']:.0f}, "
          f"tau_F={min_cell['tau_F']:.0f})")


if __name__ == "__main__":
    main()
