#!/usr/bin/env python3
"""
W1 - Multi-break sanity check on F_corr trajectories.

For each post-hoc trajectory (.npz), we compare three segmented regression
models on EMA-smoothed F_corr(log t):

  M0 : no break       (2 params)
  M1 : 1 break        (4 params)  -- canonical detector used in the paper
  M2 : 2 breaks       (6 params)

Models are scored by BIC; we report:

  bp1_M1               canonical 1-break tau_F
  bp1_M2, bp2_M2       early/late breakpoints from M2
  bic_M0, bic_M1, bic_M2
  preferred            argmin BIC across the three
  delta_bp1            bp1_M2 - bp1_M1   (steps; how the early break shifts
                                          when a second break is allowed)

Purpose: check that the 1-break formulation is not hiding a secondary
changepoint that would shift tau_F if the model were richer.

Output:
  results/posthoc/w1_multibreak.csv
  results/posthoc/w1_multibreak.md

Usage:
  python src/analysis/w1_multibreak_check.py
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from grok_metrics import estimate_changepoint  # noqa: E402

EMA_ALPHA = 0.15
SLOPE_THRESH = 0.01
MIN_FRAC = 1 / 6
MAX_FRAC = 5 / 6
MIN_POST_POINTS = 5


def _ema(values: np.ndarray, alpha: float) -> np.ndarray:
    out = values.copy().astype(float)
    for i in range(1, len(out)):
        out[i] = alpha * values[i] + (1.0 - alpha) * out[i - 1]
    return out


def _ols_rss(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    A = np.column_stack([x, np.ones(len(x))])
    coeffs, res, _, _ = np.linalg.lstsq(A, y, rcond=None)
    if len(res) == 0:
        pred = A @ coeffs
        rss = float(np.sum((y - pred) ** 2))
    else:
        rss = float(res[0])
    return float(coeffs[0]), rss


def _bic(n: int, total_rss: float, k_params: int) -> float:
    return n * np.log(max(total_rss / n, 1e-15)) + k_params * np.log(n)


def fit_one_break(steps: np.ndarray, smoothed: np.ndarray) -> dict:
    n = len(steps)
    log_t = np.log1p(steps.astype(float))
    lo = max(3, int(MIN_FRAC * n))
    hi = min(n - 3, int(MAX_FRAC * n))
    v_range = float(np.ptp(smoothed))

    _, rss0 = _ols_rss(log_t, smoothed)
    bic0 = _bic(n, rss0, 2)

    best_bic = np.inf
    best_bp = None
    for bp in range(lo, hi + 1):
        s1, rss1 = _ols_rss(log_t[:bp], smoothed[:bp])
        s2, rss2 = _ols_rss(log_t[bp:], smoothed[bp:])
        if (n - bp) < MIN_POST_POINTS:
            continue
        if abs(s2) <= SLOPE_THRESH * max(v_range, 1e-9):
            continue
        bic = _bic(n, rss1 + rss2, 4)
        if bic < best_bic:
            best_bic = bic
            best_bp = bp
    return {"bic_nobreak": bic0,
            "bic_1break": best_bic if best_bp is not None else np.inf,
            "bp1": int(steps[best_bp]) if best_bp is not None else None}


def fit_two_breaks(steps: np.ndarray, smoothed: np.ndarray) -> dict:
    n = len(steps)
    log_t = np.log1p(steps.astype(float))
    lo = max(3, int(MIN_FRAC * n))
    hi = min(n - 3, int(MAX_FRAC * n))

    best_bic = np.inf
    best_pair = (None, None)
    for bp1 in range(lo, hi - 1):
        for bp2 in range(bp1 + 3, hi + 1):
            if (n - bp2) < MIN_POST_POINTS:
                continue
            _, rss1 = _ols_rss(log_t[:bp1], smoothed[:bp1])
            _, rss2 = _ols_rss(log_t[bp1:bp2], smoothed[bp1:bp2])
            _, rss3 = _ols_rss(log_t[bp2:], smoothed[bp2:])
            bic = _bic(n, rss1 + rss2 + rss3, 6)
            if bic < best_bic:
                best_bic = bic
                best_pair = (bp1, bp2)
    bp1, bp2 = best_pair
    return {"bic_2break": best_bic if bp1 is not None else np.inf,
            "bp1_M2": int(steps[bp1]) if bp1 is not None else None,
            "bp2_M2": int(steps[bp2]) if bp2 is not None else None}


def analyse(npz_path: Path) -> dict:
    data = np.load(npz_path, allow_pickle=True)
    steps = data["steps"].astype(int)
    f_raw = data["f_raw"].astype(float)
    f_null = data["f_null_p95"].astype(float)
    f_corr = np.maximum(0.0, f_raw - f_null)
    smoothed = _ema(f_corr, EMA_ALPHA)

    canonical = estimate_changepoint(
        steps.tolist(), f_corr.tolist(),
        slope_rel_threshold=SLOPE_THRESH,
    )
    m1 = fit_one_break(steps, smoothed)
    m2 = fit_two_breaks(steps, smoothed)

    bic_vec = {
        "M0": m1["bic_nobreak"],
        "M1": m1["bic_1break"],
        "M2": m2["bic_2break"],
    }
    preferred = min(bic_vec, key=bic_vec.get)

    delta_bp1 = None
    if m2["bp1_M2"] is not None and m1["bp1"] is not None:
        delta_bp1 = m2["bp1_M2"] - m1["bp1"]

    return {
        "canonical_tauF": canonical,
        "bp1_M1": m1["bp1"],
        "bp1_M2": m2["bp1_M2"],
        "bp2_M2": m2["bp2_M2"],
        "bic_M0": m1["bic_nobreak"],
        "bic_M1": m1["bic_1break"],
        "bic_M2": m2["bic_2break"],
        "preferred": preferred,
        "delta_bp1": delta_bp1,
    }


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

    out_csv = indir / "w1_multibreak.csv"
    fieldnames = ["label", "category", "canonical_tauF",
                  "bp1_M1", "bp1_M2", "bp2_M2",
                  "bic_M0", "bic_M1", "bic_M2",
                  "preferred", "delta_bp1"]
    n_prefer_M1 = 0
    n_prefer_M2 = 0
    deltas = []

    with open(out_csv, "w", newline="", encoding="utf-8") as fcsv:
        w = csv.DictWriter(fcsv, fieldnames=fieldnames)
        w.writeheader()
        for row in rows:
            npz_path = indir / row["file"]
            if not npz_path.exists():
                print(f"  SKIP {row['label']}")
                continue
            print(f"Analysing {row['label']} ({row['category']}) ...")
            res = analyse(npz_path)
            if res["preferred"] == "M1":
                n_prefer_M1 += 1
            elif res["preferred"] == "M2":
                n_prefer_M2 += 1
            if res["delta_bp1"] is not None:
                deltas.append(res["delta_bp1"])
            w.writerow({
                "label": row["label"],
                "category": row["category"],
                "canonical_tauF":
                    f"{res['canonical_tauF']:.0f}"
                    if res["canonical_tauF"] is not None else "",
                "bp1_M1": res["bp1_M1"] if res["bp1_M1"] is not None else "",
                "bp1_M2": res["bp1_M2"] if res["bp1_M2"] is not None else "",
                "bp2_M2": res["bp2_M2"] if res["bp2_M2"] is not None else "",
                "bic_M0": f"{res['bic_M0']:.2f}",
                "bic_M1": f"{res['bic_M1']:.2f}",
                "bic_M2": f"{res['bic_M2']:.2f}",
                "preferred": res["preferred"],
                "delta_bp1":
                    f"{res['delta_bp1']:.0f}"
                    if res["delta_bp1"] is not None else "",
            })

    print(f"\nCSV -> {out_csv}")

    out_md = indir / "w1_multibreak.md"
    lines = [
        "# W1 - 1-break vs 2-break BIC sanity check",
        "",
        f"Cells analysed: {n_prefer_M1 + n_prefer_M2}",
        f"Preferred by BIC: M1 (1-break) in {n_prefer_M1} cells, "
        f"M2 (2-break) in {n_prefer_M2} cells.",
        "",
    ]
    if deltas:
        arr = np.array(deltas)
        lines += [
            f"Across cells where both bp1_M1 and bp1_M2 exist: "
            f"median |delta_bp1| = {float(np.median(np.abs(arr))):.0f} steps; "
            f"max |delta_bp1| = {int(np.max(np.abs(arr)))} steps.",
            "",
        ]
    lines += [
        "Interpretation:",
        "- If M1 wins on BIC for the bulk of cells, the 1-break formulation "
        "is BIC-justified and not hiding a secondary changepoint.",
        "- If |delta_bp1| stays within the 500-step measurement bin, the "
        "early changepoint reported as tau_F is robust under richer models.",
    ]
    out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"MD  -> {out_md}")


if __name__ == "__main__":
    main()
