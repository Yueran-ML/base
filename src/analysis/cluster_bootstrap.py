"""Cell-level cluster bootstrap, comparing against the existing per-row
bootstrap.

The existing bootstrap (`e6_bootstrap_ci.py`) resamples per-run rows as
i.i.d. units. Under within-cell seed clustering this can either inflate
or deflate the apparent variance. A cleaner uncertainty quantification is
a cluster bootstrap that resamples (lr, wd) cells with replacement; for
each cell drawn, all of its seed rows go in together. This preserves the
hierarchical structure (cell -> seeds-within-cell) and its CI is the
appropriate null-hypothesis distribution for "the rate of G<F across
cells".

Inputs (auto-detected, only files that exist are used):
  results/stage2_wd/results.csv
  results/stage3_lr/results.csv
  results/stage4_mul/results_stage4_mul.csv     (NB: small, may have 1 cell)
  results/stage5_mul_dlog/results_stage5_mul_dlog.csv
  results/stage5_p97/results_stage5_p97.csv
  results/stage6_2d/results_stage6_2d.csv
  results/step2_circuit/results_step2_circuit.csv
  results/e4/results.csv

Outputs:
  results/posthoc/cluster_bootstrap_ci.csv
  results/posthoc/cluster_bootstrap_ci.md
"""

from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path

import numpy as np

N_BOOT = 1000
RNG_SEED = 20260509


SOURCES = [
    ("stage2_wd", "results/stage2_wd/results.csv"),
    ("stage3_lr", "results/stage3_lr/results.csv"),
    ("stage4_mul", "results/stage4_mul/results_stage4_mul.csv"),
    ("stage5_mul_dlog", "results/stage5_mul_dlog/results_stage5_mul_dlog.csv"),
    ("stage5_p97", "results/stage5_p97/results_stage5_p97.csv"),
    ("stage6_2d", "results/stage6_2d/results_stage6_2d.csv"),
    ("step2_circuit", "results/step2_circuit/results_step2_circuit.csv"),
    ("e4", "results/e4/results.csv"),
]


def _f(s):
    if s in (None, "", "None"):
        return None
    try:
        return float(s)
    except Exception:
        return None


def _ordering_label(row):
    # Try several possible column names
    for col in ("ordering", "ordering_gf", "ordering_3stage"):
        if col in row and row[col]:
            return row[col]
    return None


def _row_delta(row):
    d = _f(row.get("delta")) or _f(row.get("delta_gf"))
    if d is not None:
        return d
    tg = _f(row.get("tau_gen"))
    tF = _f(row.get("tau_F"))
    if tg is not None and tF is not None:
        return tF - tg
    return None


def _is_gf(row):
    o = _ordering_label(row)
    if o is None:
        return None
    return o == "G<F"


def _row_bootstrap(rows, n_boot=N_BOOT, rng=None):
    """Resample rows independently."""
    rng = rng or np.random.default_rng(RNG_SEED)
    n = len(rows)
    p_gf_samples = []
    median_delta_samples = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        boot = [rows[i] for i in idx]
        gfs = [_is_gf(r) for r in boot if _is_gf(r) is not None]
        if gfs:
            p_gf_samples.append(np.mean(gfs))
        deltas = [d for d in (_row_delta(r) for r in boot) if d is not None]
        if deltas:
            median_delta_samples.append(np.median(deltas))
    return p_gf_samples, median_delta_samples


def _cluster_bootstrap(rows, n_boot=N_BOOT, rng=None):
    """Resample (lr, wd) cells with replacement; all seeds inside each cell
    go together."""
    rng = rng or np.random.default_rng(RNG_SEED ^ 0xCE11)
    cells = defaultdict(list)
    for r in rows:
        try:
            key = (round(float(r["lr"]), 8), round(float(r["wd"]), 8))
        except (KeyError, ValueError, TypeError):
            continue
        cells[key].append(r)
    cell_keys = list(cells.keys())
    n_cells = len(cell_keys)

    p_gf_samples = []
    median_delta_samples = []
    for _ in range(n_boot):
        idx = rng.integers(0, n_cells, size=n_cells)
        boot_rows = []
        for j in idx:
            boot_rows.extend(cells[cell_keys[j]])
        gfs = [_is_gf(r) for r in boot_rows if _is_gf(r) is not None]
        if gfs:
            p_gf_samples.append(np.mean(gfs))
        deltas = [d for d in (_row_delta(r) for r in boot_rows)
                  if d is not None]
        if deltas:
            median_delta_samples.append(np.median(deltas))
    return p_gf_samples, median_delta_samples, n_cells


def _ci(samples, alpha=0.05):
    if not samples:
        return (float("nan"), float("nan"), float("nan"))
    arr = np.asarray(samples)
    return (float(np.median(arr)),
            float(np.percentile(arr, 100 * alpha / 2)),
            float(np.percentile(arr, 100 * (1 - alpha / 2))))


def main():
    out_dir = Path("results/posthoc")
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(RNG_SEED)

    out_rows = []
    md_lines = ["# Cell-level cluster bootstrap vs row-level bootstrap\n",
                "",
                "Resample unit comparison: row-level treats each "
                "(lr, wd, seed) row as i.i.d.; cell-level resamples "
                "(lr, wd) cells with replacement, all seeds inside a "
                "cell going together.\n",
                "",
                f"$n_{{\\mathrm{{boot}}}} = {N_BOOT}$, percentile 95% CI.\n",
                "",
                "| source | n_rows | n_cells | "
                "P(G<F) row-bs [median, CI] | "
                "P(G<F) cluster-bs [median, CI] | "
                "median deltatau row-bs [steps, CI] | "
                "median deltatau cluster-bs [steps, CI] |",
                "|---|---|---|---|---|---|---|"]

    for source_label, path in SOURCES:
        p = Path(path)
        if not p.exists():
            print(f"  [skip] {path} not found")
            continue
        rows = list(csv.DictReader(open(p, encoding="utf-8")))
        n_rows = len(rows)
        if n_rows == 0:
            continue

        p_row, d_row = _row_bootstrap(rows, rng=rng)
        p_cl, d_cl, n_cells = _cluster_bootstrap(rows, rng=rng)
        m_p_row, lo_p_row, hi_p_row = _ci(p_row)
        m_p_cl, lo_p_cl, hi_p_cl = _ci(p_cl)
        m_d_row, lo_d_row, hi_d_row = _ci(d_row)
        m_d_cl, lo_d_cl, hi_d_cl = _ci(d_cl)

        out_rows.append({
            "source": source_label, "n_rows": n_rows, "n_cells": n_cells,
            "p_gf_row_median": m_p_row, "p_gf_row_lo95": lo_p_row,
            "p_gf_row_hi95": hi_p_row,
            "p_gf_cluster_median": m_p_cl, "p_gf_cluster_lo95": lo_p_cl,
            "p_gf_cluster_hi95": hi_p_cl,
            "median_delta_row_median": m_d_row,
            "median_delta_row_lo95": lo_d_row,
            "median_delta_row_hi95": hi_d_row,
            "median_delta_cluster_median": m_d_cl,
            "median_delta_cluster_lo95": lo_d_cl,
            "median_delta_cluster_hi95": hi_d_cl,
        })

        def _fmt(med, lo, hi, prec=2, scale=1.0):
            if med != med:
                return "n/a"
            return f"{med*scale:.{prec}f} [{lo*scale:.{prec}f}, {hi*scale:.{prec}f}]"

        md_lines.append(
            f"| {source_label} | {n_rows} | {n_cells} | "
            f"{_fmt(m_p_row, lo_p_row, hi_p_row, 3)} | "
            f"{_fmt(m_p_cl, lo_p_cl, hi_p_cl, 3)} | "
            f"{_fmt(m_d_row/1000.0, lo_d_row/1000.0, hi_d_row/1000.0, 1)}k | "
            f"{_fmt(m_d_cl/1000.0, lo_d_cl/1000.0, hi_d_cl/1000.0, 1)}k |"
        )

        # Compute ratio of CI widths
        row_w = (hi_p_row - lo_p_row) if hi_p_row == hi_p_row else float("nan")
        cl_w = (hi_p_cl - lo_p_cl) if hi_p_cl == hi_p_cl else float("nan")
        ratio = cl_w / row_w if row_w and row_w > 1e-9 else float("nan")
        print(f"[{source_label}] n_rows={n_rows} n_cells={n_cells} | "
              f"P(G<F): row CI width={row_w:.3f}, cluster CI width={cl_w:.3f} "
              f"(cluster/row ratio={ratio:.2f})")

    # CSV
    out_csv = out_dir / "cluster_bootstrap_ci.csv"
    if out_rows:
        with open(out_csv, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(out_rows[0].keys()))
            w.writeheader()
            w.writerows(out_rows)
        print(f"\n[csv] {out_csv}")

    # MD
    md_lines.append("")
    md_lines.append("## Interpretation guide")
    md_lines.append("")
    md_lines.append(
        "If `cluster CI width` >> `row CI width` for a given source, the "
        "row-level bootstrap was under-estimating uncertainty (positive "
        "intra-cell seed correlation). If `cluster <= row`, the row-level "
        "bootstrap was either correct or slightly conservative. The "
        "**cluster** intervals are the appropriate null distribution for "
        "claims about the rate of G<F across hyperparameter cells."
    )
    out_md = out_dir / "cluster_bootstrap_ci.md"
    out_md.write_text("\n".join(md_lines) + "\n", encoding="utf-8")
    print(f"[md] {out_md}")


if __name__ == "__main__":
    main()
