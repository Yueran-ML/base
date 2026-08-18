#!/usr/bin/env python3
"""
Post-hoc Trajectory Dump — re-run 8 sensitivity cells and save dense
(step, embedding) snapshots to .npz for downstream analyses (Q2/Q3/Q8).

For each cell we save (every 500 steps):
  - steps          (T,)  int
  - emb            (T, p, d_model)  float32   (token embedding matrix)
  - train_acc      (T,)  float
  - test_acc       (T,)  float
  - f_raw          (T,)  float    (single-harmonic max, for reference)
  - f_null_p95     (T,)  float    (perm null recomputed every log step, 100 perms)

Output:
  results/posthoc/traj_<cell_label>.npz   (~5-6 MB per cell, ~45 MB total)
  results/posthoc/manifest.csv            (cell → file + metadata)

The same 8 cells are used as `sensitivity_analysis.py` so results line up
with the published Robustness table.

Usage:
  python src/sweeps/posthoc_trajectory_dump.py
  python src/sweeps/posthoc_trajectory_dump.py --cells wd1.71_s7 lr6.8e4_s7
  python src/sweeps/posthoc_trajectory_dump.py --max-steps 2000  # smoketest
"""

from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from grokking_baseline import (
    GrokkingTransformer,
    build_optimizer,
    build_scheduler,
    get_token_embeddings,
    make_dataset,
    set_seed,
    split_dataset,
)
from grok_metrics import (
    compute_fourier_alignment,
    compute_fourier_null_p95,
)

# ---------------------------------------------------------------------------
# 8-cell definition (must match sensitivity_analysis.py)
# ---------------------------------------------------------------------------

CELLS = [
    # (label,        lr,       wd,     seed,  category)
    ("wd1.71_s7",    1.6e-3,   1.7145, 7,     "strong_GF"),
    ("wd2.76_s42",   1.6e-3,   2.7591, 42,    "strong_GF"),
    ("wd1.52_s42",   1.6e-3,   1.5223, 42,    "medium_GF"),
    ("wd2.18_s42",   1.6e-3,   2.175,  42,    "medium_GF"),
    ("wd2.45_s7",    1.6e-3,   2.4497, 7,     "weak_GF"),
    ("wd3.50_s7",    1.6e-3,   3.5,    7,     "weak_GF"),
    ("wd3.11_s2025", 1.6e-3,   3.1075, 2025,  "coincident"),
    ("lr6.8e4_s7",   6.804e-4, 2.5,    7,     "FG"),
]

PRIME = 53
D_MODEL = 256
N_HEADS = 4
N_LAYERS = 2
D_FF = 1024
TRAIN_FRACTION = 0.3
EMBED_LR = 1e-3
LOG_INTERVAL = 500
N_NULL_PERMS = 100


def run_and_dump(
    label: str,
    lr: float,
    wd: float,
    seed: int,
    category: str,
    outdir: Path,
    max_steps: int,
) -> dict:
    t0 = time.time()
    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    x, y = make_dataset(PRIME, "add", seed)
    train_x, train_y, test_x, test_y = split_dataset(x, y, TRAIN_FRACTION)

    model = GrokkingTransformer(
        prime=PRIME, d_model=D_MODEL, n_heads=N_HEADS,
        n_layers=N_LAYERS, d_ff=D_FF, dropout=0.0,
    ).to(device)

    loader = DataLoader(
        TensorDataset(train_x, train_y),
        batch_size=len(train_x), shuffle=True,
        generator=torch.Generator().manual_seed(seed),
    )
    criterion = nn.CrossEntropyLoss()
    optimizer = build_optimizer(
        model, decoder_lr=lr, embed_lr=EMBED_LR, decoder_weight_decay=wd,
    )
    scheduler = build_scheduler(
        optimizer, warmup_steps=10, lr_schedule="constant",
        lr_min_ratio=0.05, max_steps=max_steps,
    )

    null_rng = np.random.default_rng(seed ^ 0xDEAD)

    steps_log:      list[int]   = []
    train_acc_log:  list[float] = []
    test_acc_log:   list[float] = []
    f_raw_log:      list[float] = []
    f_null_log:     list[float] = []
    emb_snapshots:  list[np.ndarray] = []

    step = 0
    while step < max_steps:
        for xb, yb in loader:
            model.train()
            logits = model(xb.to(device))
            loss = criterion(logits, yb.to(device))
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            step += 1

            if step % LOG_INTERVAL == 0 or step == 1 or step >= max_steps:
                model.eval()
                with torch.no_grad():
                    tr_acc = (model(train_x.to(device)).argmax(1).cpu()
                              == train_y).float().mean().item()
                    te_acc = (model(test_x.to(device)).argmax(1).cpu()
                              == test_y).float().mean().item()

                emb = get_token_embeddings(model, PRIME)  # (p, d)
                f_raw, _ = compute_fourier_alignment(emb, PRIME)
                f_null = compute_fourier_null_p95(
                    emb, PRIME, n_perms=N_NULL_PERMS, rng=null_rng,
                )

                steps_log.append(step)
                train_acc_log.append(tr_acc)
                test_acc_log.append(te_acc)
                f_raw_log.append(float(f_raw))
                f_null_log.append(float(f_null))
                emb_snapshots.append(emb.astype(np.float32))

            if step >= max_steps:
                break

    emb_arr = np.stack(emb_snapshots, axis=0)  # (T, p, d)
    npz_path = outdir / f"traj_{label}.npz"
    np.savez_compressed(
        npz_path,
        steps=np.array(steps_log, dtype=np.int32),
        emb=emb_arr,
        train_acc=np.array(train_acc_log, dtype=np.float32),
        test_acc=np.array(test_acc_log, dtype=np.float32),
        f_raw=np.array(f_raw_log, dtype=np.float32),
        f_null_p95=np.array(f_null_log, dtype=np.float32),
        meta=np.array(
            [label, category, f"{lr}", f"{wd}", f"{seed}"], dtype=object
        ),
    )

    runtime = time.time() - t0
    size_mb = npz_path.stat().st_size / (1024 * 1024)
    return {
        "label": label, "category": category, "lr": lr, "wd": wd, "seed": seed,
        "file": npz_path.name, "size_mb": f"{size_mb:.2f}",
        "T": len(steps_log), "runtime_sec": f"{runtime:.1f}",
        "max_test_acc": f"{max(test_acc_log):.4f}",
    }


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--outdir", default="results/posthoc")
    p.add_argument("--max-steps", type=int, default=50_000)
    p.add_argument("--cells", nargs="+", default=None,
                   help="Subset of cell labels to run (default: all 8)")
    return p.parse_args()


def main():
    args = parse_args()
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    cells = list(CELLS) if args.cells is None else [
        c for c in CELLS if c[0] in set(args.cells)
    ]
    if not cells:
        print("No cells selected; exiting.")
        return

    manifest_path = outdir / "manifest.csv"
    fieldnames = ["label", "category", "lr", "wd", "seed", "file",
                  "size_mb", "T", "runtime_sec", "max_test_acc"]

    done = set()
    if manifest_path.exists():
        with open(manifest_path, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                done.add(row["label"])

    mode = "a" if manifest_path.exists() else "w"
    mf = open(manifest_path, mode, newline="", encoding="utf-8")
    writer = csv.DictWriter(mf, fieldnames=fieldnames)
    if mode == "w":
        writer.writeheader()

    n = len(cells)
    print(f"Trajectory dump: {n} cells, max_steps={args.max_steps:,}, "
          f"log_interval={LOG_INTERVAL} → T={args.max_steps // LOG_INTERVAL + 1}")
    print(f"Output dir: {outdir}\n")

    for i, (label, lr, wd, seed, category) in enumerate(cells, 1):
        if label in done:
            print(f"[{i}/{n}] {label} — SKIPPED (manifest already has it)")
            continue

        print(f"[{i}/{n}] {label} (lr={lr:.3e}, wd={wd:.4g}, seed={seed}, "
              f"cat={category}) ...", flush=True)
        row = run_and_dump(label, lr, wd, seed, category, outdir,
                           args.max_steps)
        writer.writerow(row)
        mf.flush()
        print(f"    → {row['file']}  ({row['size_mb']} MB, "
              f"T={row['T']}, {row['runtime_sec']}s, "
              f"test_acc_max={row['max_test_acc']})")

    mf.close()
    print(f"\nDone. Manifest: {manifest_path}")


if __name__ == "__main__":
    main()
