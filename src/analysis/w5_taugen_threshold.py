#!/usr/bin/env python3
"""
W5 - tau_gen threshold sensitivity.

For each post-hoc trajectory (.npz), we recompute tau_gen at four
test-accuracy thresholds {0.80, 0.85, 0.90, 0.95} using the canonical
sustained-detection rule (n_sustained = 3 consecutive log points above
the threshold).  Canonical reference is 0.90.

Output:
  results/posthoc/w5_taugen_threshold.csv
  results/posthoc/w5_taugen_threshold.md

Usage:
  python src/analysis/w5_taugen_threshold.py
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from grok_metrics import find_tau_sustained  # noqa: E402

THRESHOLDS = [0.80, 0.85, 0.90, 0.95]
CANONICAL = 0.90


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--indir", default="results/posthoc")
    args = ap.parse_args()

    indir = Path(args.indir)
    manifest = indir / "manifest.csv"
    if not manifest.exists():
        print(f"ERROR: manifest not found at {manifest}")
        return

    with open(manifest, newline="", encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))

    out_csv = indir / "w5_taugen_threshold.csv"
    fieldnames = ["label", "category"] \
        + [f"tau_gen_{t:.2f}" for t in THRESHOLDS] \
        + [f"delta_{t:.2f}" for t in THRESHOLDS if t != CANONICAL]

    per_thresh_deltas: dict[float, list[int]] = {
        t: [] for t in THRESHOLDS if t != CANONICAL
    }

    with open(out_csv, "w", newline="", encoding="utf-8") as fcsv:
        w = csv.DictWriter(fcsv, fieldnames=fieldnames)
        w.writeheader()

        for row in rows:
            npz_path = indir / row["file"]
            if not npz_path.exists():
                print(f"  SKIP {row['label']}")
                continue
            print(f"Analysing {row['label']} ({row['category']}) ...")
            data = np.load(npz_path, allow_pickle=True)
            steps = data["steps"].astype(int).tolist()
            test_acc = data["test_acc"].astype(float).tolist()

            tau_by_thresh: dict[float, float | None] = {}
            for t in THRESHOLDS:
                tau = find_tau_sustained(steps, test_acc,
                                         threshold=t, n_sustained=3)
                tau_by_thresh[t] = tau

            tau_canon = tau_by_thresh[CANONICAL]
            out_row = {"label": row["label"], "category": row["category"]}
            for t in THRESHOLDS:
                tau = tau_by_thresh[t]
                out_row[f"tau_gen_{t:.2f}"] = (
                    f"{tau:.0f}" if tau is not None else ""
                )
            for t in THRESHOLDS:
                if t == CANONICAL:
                    continue
                tau = tau_by_thresh[t]
                if tau is None or tau_canon is None:
                    out_row[f"delta_{t:.2f}"] = ""
                else:
                    delta = int(tau - tau_canon)
                    out_row[f"delta_{t:.2f}"] = str(delta)
                    per_thresh_deltas[t].append(delta)
            w.writerow(out_row)

    print(f"\nCSV -> {out_csv}")

    # Markdown summary
    out_md = indir / "w5_taugen_threshold.md"
    lines = [
        "# W5 - tau_gen threshold sensitivity",
        "",
        "Canonical threshold: 0.90 (sustained over 3 consecutive 500-step "
        "log points).  delta_T = tau_gen(T) - tau_gen(0.90), in training "
        "steps.",
        "",
        "| threshold | n cells | median |delta| | max |delta| |",
        "| --- | --- | --- | --- |",
    ]
    for t in THRESHOLDS:
        if t == CANONICAL:
            continue
        deltas = per_thresh_deltas[t]
        if not deltas:
            lines.append(f"| {t:.2f} | 0 | -- | -- |")
            continue
        arr = np.array(deltas)
        lines.append(
            f"| {t:.2f} | {len(deltas)} | "
            f"{float(np.median(np.abs(arr))):.0f} | "
            f"{int(np.max(np.abs(arr)))} |"
        )
    lines += [
        "",
        "Interpretation: if |delta| stays within the 500-step measurement "
        "bin across all four thresholds, the tau_gen=0.90 choice is not "
        "load-bearing for the G<F ordering (Delta_tau changes by a single "
        "bin at most).",
    ]
    out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"MD  -> {out_md}")


if __name__ == "__main__":
    main()
