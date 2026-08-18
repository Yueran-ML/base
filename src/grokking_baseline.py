#!/usr/bin/env python3
"""
Grokking reproduction with paper-faithful visualizations.

Reproduces special visualization structures from:
- Paper 1: "Towards Understanding Grokking: An Effective Theory of Representation Learning"
- Paper 2: "Grokking: Generalization Beyond Overfitting on Small Algorithmic Datasets"

Visualizations implemented:
1. PCA Embedding Evolution  (Paper 1, Fig 1) - circular structure emergence
2. Effective Dimensionality  (Paper 1, Fig 7) - e^S entropy metric
3. Phase Diagram             (Paper 1, Fig 6) - 4-phase hyperparameter map
4. Operation Table           (Paper 2, Fig 1) - Sudoku-style binary op matrix
5. t-SNE Output Weights      (Paper 2, Fig 3) - output layer representation

Default setup (following MIT paper, Section 4.2 / Figure 7):
- task: a + b mod p (prime p=53)
- model: 2-token decoder-only transformer (d_model=256, n_heads=4, n_layers=2, d_ff=1024)
- optimizer: AdamW with separate parameter groups
  embeddings: lr=1e-3, wd=0
  transformer blocks + head: lr=1e-3, wd=1.0
- linear warmup=10, constant LR afterwards, full-batch training, dropout=0.0
- split: small train fraction to encourage memorization first
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def mod_inverse(x: int, p: int) -> int:
    return pow(x, p - 2, p)


def make_dataset(prime: int, operation: str, seed: int) -> tuple[torch.Tensor, torch.Tensor]:
    op_fn = {
        "add": lambda a, b: (a + b) % prime,
        "sub": lambda a, b: (a - b) % prime,
        "mul": lambda a, b: (a * b) % prime,
        "div": lambda a, b: (a * mod_inverse(b, prime)) % prime,
    }[operation]

    samples: list[tuple[int, int, int]] = []
    b_values: Iterable[int] = range(1, prime) if operation == "div" else range(prime)
    for a in range(prime):
        for b in b_values:
            y = op_fn(a, b)
            samples.append((a, b, y))

    rng = random.Random(seed)
    rng.shuffle(samples)

    # MIT paper Section 4.2: "we do not encode the operation symbols here"
    x = torch.tensor([[a, b] for a, b, _ in samples], dtype=torch.long)
    y = torch.tensor([c for _, _, c in samples], dtype=torch.long)
    return x, y


def split_dataset(
    x: torch.Tensor,
    y: torch.Tensor,
    train_fraction: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    n = x.shape[0]
    n_train = max(1, int(n * train_fraction))
    train_x = x[:n_train]
    train_y = y[:n_train]
    test_x = x[n_train:]
    test_y = y[n_train:]
    return train_x, train_y, test_x, test_y


class GrokkingTransformer(nn.Module):
    def __init__(
        self,
        prime: int,
        d_model: int,
        n_heads: int,
        n_layers: int,
        d_ff: int,
        dropout: float,
    ) -> None:
        super().__init__()
        vocab_size = prime  # numbers 0..p-1 (no op token per MIT paper Sec 4.2)
        self.d_model = d_model
        self.token_embed = nn.Embedding(vocab_size, d_model)
        self.pos_embed = nn.Parameter(torch.zeros(2, d_model))  # 2 positions: [a, b]

        # Decoder-only transformer with causal masking (MIT paper Sec 4.2)
        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_ff,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=n_layers)
        self.norm = nn.LayerNorm(d_model * 2)
        # MIT paper: "outputs from the last layer are concatenated and passed to a linear layer"
        self.head = nn.Linear(d_model * 2, prime, bias=False)

        nn.init.normal_(self.pos_embed, std=0.02)

        # Causal mask: token b cannot attend to token a (decoder-only)
        self.register_buffer(
            "causal_mask",
            torch.triu(torch.ones(2, 2, dtype=torch.bool), diagonal=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.token_embed(x) + self.pos_embed.unsqueeze(0)
        h = self.encoder(h, mask=self.causal_mask)
        # Concatenate outputs from both positions (MIT paper Sec 4.2)
        h = torch.cat([h[:, 0, :], h[:, 1, :]], dim=-1)
        h = self.norm(h)
        return self.head(h)


@dataclass
class Config:
    prime: int
    operation: str
    train_fraction: float
    batch_size: int
    max_steps: int
    eval_interval: int
    lr: float  # decoder / transformer+head learning rate
    embed_lr: float
    weight_decay: float
    d_model: int
    n_heads: int
    n_layers: int
    d_ff: int
    dropout: float
    seed: int
    device: str
    outdir: str
    stop_test_acc: float
    stop_extra_steps: int = 0  # extra steps after threshold (0 = run full max_steps, just log milestones)
    warmup_steps: int = 10
    grad_clip: float = 1.0
    lr_schedule: str = "constant"  # "constant" (paper, needed for circular structure) or "cosine"
    lr_min_ratio: float = 0.05  # cosine schedule decays to lr * lr_min_ratio (not 0)
    embed_snapshot_steps: list[int] | None = None
    pca_interval: int = 2000  # auto-snapshot every N steps (0 = disabled, use manual list only)
    phase_diagram: bool = False
    phase_grid_size: int = 8
    phase_max_steps: int = 20000


def build_optimizer(
    model: GrokkingTransformer,
    decoder_lr: float,
    embed_lr: float,
    decoder_weight_decay: float,
    embed_weight_decay: float = 0.0,
) -> torch.optim.AdamW:
    """Match the paper's split between embeddings and the rest of the decoder.

    embed_weight_decay defaults to 0 (paper-canonical); set non-zero for the
    embedding-wd ablation (Q1 / E4).
    """
    embed_params = list(model.token_embed.parameters()) + [model.pos_embed]
    decoder_params = list(model.encoder.parameters()) + list(model.norm.parameters()) \
        + list(model.head.parameters())
    # Power et al. uses β1=0.9, β2=0.98 (not PyTorch default 0.999)
    return torch.optim.AdamW([
        {"params": embed_params, "lr": embed_lr, "weight_decay": embed_weight_decay},
        {"params": decoder_params, "lr": decoder_lr, "weight_decay": decoder_weight_decay},
    ], betas=(0.9, 0.98))


def set_decoder_weight_decay(
    optimizer: torch.optim.Optimizer, new_wd: float
) -> None:
    """In-place update of the decoder parameter group's weight_decay.

    Assumes group 0 = embeddings (unchanged), group 1 = decoder (updated).
    Used by E4 to turn decoder wd off at a specified step.
    """
    if len(optimizer.param_groups) < 2:
        raise ValueError("expected 2 param groups (embed, decoder)")
    optimizer.param_groups[1]["weight_decay"] = float(new_wd)


def build_scheduler(
    optimizer: torch.optim.Optimizer,
    warmup_steps: int,
    lr_schedule: str,
    lr_min_ratio: float,
    max_steps: int,
) -> torch.optim.lr_scheduler.LambdaLR:
    """Shared LR multiplier for both parameter groups."""
    def lr_lambda(current_step: int) -> float:
        if current_step < warmup_steps:
            return current_step / max(1, warmup_steps)
        if lr_schedule == "cosine":
            progress = (current_step - warmup_steps) / max(1, max_steps - warmup_steps)
            cosine_val = 0.5 * (1.0 + math.cos(math.pi * progress))
            return lr_min_ratio + (1.0 - lr_min_ratio) * cosine_val
        return 1.0

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


@torch.inference_mode()
def evaluate(
    model: nn.Module,
    x: torch.Tensor,
    y: torch.Tensor,
    device: torch.device,
    criterion: nn.Module,
) -> tuple[float, float]:
    model.eval()
    logits = model(x.to(device))
    loss = criterion(logits, y.to(device)).item()
    preds = logits.argmax(dim=-1).cpu()
    acc = (preds == y).float().mean().item()
    return loss, acc


# ---------------------------------------------------------------------------
# Visualization 1: PCA Embedding Evolution (Paper 1, Figure 1)
# Tracks token embedding geometry at key training stages.
# Tokens should form a circular structure for modular addition after grokking.
# ---------------------------------------------------------------------------

def get_token_embeddings(model: GrokkingTransformer, prime: int) -> np.ndarray:
    """Extract input token embeddings for tokens 0..p-1 (excluding op token)."""
    with torch.no_grad():
        emb = model.token_embed.weight[:prime].cpu().numpy()
    return emb


def compute_pca_2d(embeddings: np.ndarray, normalize: bool = False) -> np.ndarray:
    """Project embeddings to 2D via PCA (centered, no sklearn dependency).

    Args:
        normalize: If True, L2-normalize each embedding to unit length before PCA.
            This reveals angular/directional structure (e.g. Fourier circles)
            even when embedding norms are collapsing.
    """
    X = embeddings.copy()
    if normalize:
        norms = np.linalg.norm(X, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-8)  # avoid division by zero
        X = X / norms
    X = X - X.mean(axis=0, keepdims=True)
    _, s, Vt = np.linalg.svd(X, full_matrices=False)
    return X @ Vt[:2].T


def _select_pca_keyframes(
    snapshots: dict[int, np.ndarray],
    max_panels: int = 12,
) -> dict[int, np.ndarray]:
    """Select up to *max_panels* keyframes from all snapshots.

    Strategy: always keep first & last; distribute the rest evenly.
    """
    if len(snapshots) <= max_panels:
        return snapshots
    sorted_steps = sorted(snapshots)
    # Always include first and last
    selected = {sorted_steps[0], sorted_steps[-1]}
    # Fill remaining slots evenly
    remaining = max_panels - 2
    if remaining > 0:
        indices = np.linspace(1, len(sorted_steps) - 2, remaining, dtype=int)
        for idx in indices:
            selected.add(sorted_steps[idx])
    return {s: snapshots[s] for s in sorted(selected)}


def plot_pca_evolution(
    snapshots: dict[int, np.ndarray],
    prime: int,
    out_path: Path,
    max_panels: int = 8,
) -> None:
    """Plot PCA projections of token embeddings at multiple training stages.

    Reproduces Paper 1, Figure 1: shows how embeddings evolve from random
    initialization -> overfitting (scattered) -> grokking (circular structure).
    Points are colored by token value to reveal the geometric ordering.

    If there are more snapshots than *max_panels*, keyframes are selected
    automatically (first, last, + evenly spaced).  A full version with all
    panels is also saved as ``*_full.png``.
    """
    try:
        import matplotlib.pyplot as plt
        from matplotlib import cm
    except Exception:
        return

    # Always save a normalized (unit-vector) version to reveal angular structure
    # even when embedding norms collapse during training
    if len(snapshots) > max_panels:
        _plot_pca_grid(snapshots, prime, out_path.with_name("pca_evolution_full.png"),
                       title_suffix=" (all snapshots)")
        _plot_pca_grid(snapshots, prime, out_path.with_name("pca_evolution_normed_full.png"),
                       title_suffix=" (normalized, all snapshots)", normalize=True)
        keyframes = _select_pca_keyframes(snapshots, max_panels)
    else:
        keyframes = snapshots

    _plot_pca_grid(keyframes, prime, out_path)
    _plot_pca_grid(keyframes, prime, out_path.with_name("pca_evolution_normed.png"),
                   title_suffix=" (normalized)", normalize=True)


def _plot_pca_grid(
    snapshots: dict[int, np.ndarray],
    prime: int,
    out_path: Path,
    title_suffix: str = "",
    normalize: bool = False,
) -> None:
    """Render PCA snapshots in a multi-row grid layout."""
    try:
        import matplotlib.pyplot as plt
        from matplotlib import cm
    except Exception:
        return

    n = len(snapshots)
    ncols = min(n, 6)
    nrows = math.ceil(n / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.5 * ncols, 4.5 * nrows))
    if n == 1:
        axes = np.array([axes])
    axes = np.atleast_2d(axes).flatten()

    colors = cm.hsv(np.linspace(0, 1, prime, endpoint=False))

    for idx, (step, emb) in enumerate(sorted(snapshots.items())):
        ax = axes[idx]
        proj = compute_pca_2d(emb, normalize=normalize)
        ax.scatter(proj[:, 0], proj[:, 1], c=colors, s=30, zorder=3)
        label_stride = max(1, prime // 20)
        for i in range(0, prime, label_stride):
            ax.annotate(
                str(i), (proj[i, 0], proj[i, 1]),
                fontsize=6, ha="center", va="bottom", alpha=0.7,
            )
        ax.set_title(f"Step {step:,}", fontsize=11)
        ax.set_xlabel("PC 1")
        ax.set_ylabel("PC 2")
        if not normalize:
            ax.set_aspect("equal")
        ax.grid(True, alpha=0.3)

    # Hide unused axes
    for idx in range(n, len(axes)):
        axes[idx].set_visible(False)

    fig.suptitle(f"PCA Embedding Evolution{title_suffix}", fontsize=14, y=1.02)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Visualization 2: Effective Dimensionality (Paper 1, Figure 7)
# e_S = exp(S) where S = -sum(sigma_i * log(sigma_i))
# sigma_i = fractional PCA variance of token embeddings
# ---------------------------------------------------------------------------

def compute_effective_dim(embeddings: np.ndarray) -> float:
    """Compute effective dimensionality e^S of token embeddings.

    S = Shannon entropy of normalized PCA eigenvalues (explained variance ratios).
    Returns e^S. High value = spread across many dimensions; low = concentrated.
    """
    X = embeddings - embeddings.mean(axis=0, keepdims=True)
    _, s, _ = np.linalg.svd(X, full_matrices=False)
    eigenvalues = s ** 2
    sigma = eigenvalues / eigenvalues.sum()
    sigma = sigma[sigma > 1e-12]  # avoid log(0)
    entropy = -np.sum(sigma * np.log(sigma))
    return float(np.exp(entropy))


def plot_effective_dim(
    steps: list[int],
    eff_dims: list[float],
    out_path: Path,
) -> None:
    """Plot effective dimensionality over training steps.

    Reproduces Paper 1, Figure 7 (left panel): e_S should drop as the model
    learns a low-dimensional circular representation during grokking.
    """
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(steps, eff_dims, color="tab:blue", linewidth=1.5)
    ax.set_xlabel("Optimization Step")
    ax.set_ylabel(r"Effective Dimensionality $e^S$")
    ax.set_title("Effective Dimensionality of Token Embeddings (Paper 1, Fig 7)")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Visualization 3: Operation Table (Paper 2, Figure 1 / Figure 5)
# Sudoku-style matrix showing the binary operation a op b mod p.
# ---------------------------------------------------------------------------

def plot_operation_table(
    prime: int,
    operation: str,
    out_path: Path,
) -> None:
    """Plot the modular arithmetic operation as a colored matrix.

    Reproduces Paper 2, Figure 1 (right panel) and Figure 5: a visual
    representation of the binary operation table used as the dataset.
    """
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return

    op_fn = {
        "add": lambda a, b: (a + b) % prime,
        "sub": lambda a, b: (a - b) % prime,
        "mul": lambda a, b: (a * b) % prime,
        "div": lambda a, b: (a * mod_inverse(b, prime)) % prime,
    }[operation]

    b_range = range(1, prime) if operation == "div" else range(prime)
    b_list = list(b_range)
    table = np.zeros((prime, len(b_list)), dtype=int)
    for i, a in enumerate(range(prime)):
        for j, b in enumerate(b_list):
            table[i, j] = op_fn(a, b)

    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(table, cmap="twilight", interpolation="nearest", aspect="equal")
    ax.set_xlabel("b")
    ax.set_ylabel("a")
    op_sym = {"add": "+", "sub": "-", "mul": "*", "div": "/"}[operation]
    ax.set_title(f"Operation Table: a {op_sym} b mod {prime} (Paper 2, Fig 1)")

    tick_stride = max(1, prime // 10)
    ax.set_xticks(range(0, len(b_list), tick_stride))
    ax.set_xticklabels([str(b_list[i]) for i in range(0, len(b_list), tick_stride)], fontsize=7)
    ax.set_yticks(range(0, prime, tick_stride))
    ax.set_yticklabels([str(i) for i in range(0, prime, tick_stride)], fontsize=7)

    fig.colorbar(im, ax=ax, label="Result", shrink=0.8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Visualization 4: t-SNE of Output Layer Weights (Paper 2, Figure 3)
# Dimensionality reduction of the output (head) layer weight rows.
# Should show clusters corresponding to cosets or residue classes.
# ---------------------------------------------------------------------------

def plot_output_tsne(
    model: GrokkingTransformer,
    prime: int,
    out_path: Path,
) -> None:
    """Apply t-SNE to output layer weight rows and plot.

    Reproduces Paper 2, Figure 3: row vectors of the output layer matrix
    projected to 2D, revealing group-theoretic structure (rings, clusters).
    Falls back to PCA if sklearn is not installed.
    """
    try:
        import matplotlib.pyplot as plt
        from matplotlib import cm
    except Exception:
        return

    with torch.no_grad():
        W = model.head.weight.cpu().numpy()  # shape (prime, 2 * d_model)

    colors = cm.hsv(np.linspace(0, 1, prime, endpoint=False))

    # Try t-SNE, fall back to PCA
    try:
        from sklearn.manifold import TSNE
        perplexity = min(30, prime - 1)
        try:
            # sklearn >= 1.5 uses max_iter
            tsne = TSNE(n_components=2, perplexity=perplexity, random_state=42, max_iter=1000)
        except TypeError:
            # sklearn < 1.5 uses n_iter
            tsne = TSNE(n_components=2, perplexity=perplexity, random_state=42, n_iter=1000)
        proj = tsne.fit_transform(W)
        method = "t-SNE"
    except ImportError:
        proj = compute_pca_2d(W)
        method = "PCA (sklearn unavailable for t-SNE)"

    fig, ax = plt.subplots(figsize=(7, 7))
    ax.scatter(proj[:, 0], proj[:, 1], c=colors, s=40, zorder=3)
    label_stride = max(1, prime // 20)
    for i in range(0, prime, label_stride):
        ax.annotate(
            str(i), (proj[i, 0], proj[i, 1]),
            fontsize=6, ha="center", va="bottom", alpha=0.7,
        )
    ax.set_title(f"{method} of Output Layer Weights (Paper 2, Fig 3)")
    ax.set_xlabel(f"{method} 1")
    ax.set_ylabel(f"{method} 2")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Visualization 5: Phase Diagram (Paper 1, Figure 6)
# Grid sweep over two hyperparameters, classify each run into one of 4 phases:
# Comprehension, Grokking, Memorization, Confusion.
# Thresholds from paper: 90% accuracy within 10^5 steps, delay >= 10^3 steps.
# ---------------------------------------------------------------------------

def classify_phase(
    train_acc_90_step: int | None,
    test_acc_90_step: int | None,
    delay_threshold: int = 1000,
) -> str:
    """Classify a training run into one of 4 grokking phases.

    Based on Paper 1, Figure 6 criteria:
    - Comprehension: both > 90%, delay < 1000 steps
    - Grokking: both > 90%, delay >= 1000 steps
    - Memorization: train > 90%, test <= 90%
    - Confusion: train <= 90%
    """
    if train_acc_90_step is None:
        return "Confusion"
    if test_acc_90_step is None:
        return "Memorization"
    delay = test_acc_90_step - train_acc_90_step
    if delay >= delay_threshold:
        return "Grokking"
    return "Comprehension"


def run_single_for_phase(
    prime: int,
    operation: str,
    train_fraction: float,
    lr: float,
    embed_lr: float,
    weight_decay: float,
    d_model: int,
    n_heads: int,
    n_layers: int,
    d_ff: int,
    max_steps: int,
    seed: int,
    device: torch.device,
) -> str:
    """Run a single training to classify its phase. Returns phase label."""
    set_seed(seed)
    x, y = make_dataset(prime, operation, seed)
    train_x, train_y, test_x, test_y = split_dataset(x, y, train_fraction)

    model = GrokkingTransformer(
        prime=prime, d_model=d_model, n_heads=n_heads,
        n_layers=n_layers, d_ff=d_ff, dropout=0.0,
    ).to(device)

    batch_size = train_x.shape[0]
    dataset = TensorDataset(train_x, train_y)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True,
                        generator=torch.Generator().manual_seed(seed))
    criterion = nn.CrossEntropyLoss()
    optimizer = build_optimizer(
        model,
        decoder_lr=lr,
        embed_lr=embed_lr,
        decoder_weight_decay=weight_decay,
    )
    _scheduler = build_scheduler(
        optimizer,
        warmup_steps=10,
        lr_schedule="constant",
        lr_min_ratio=0.05,
        max_steps=max_steps,
    )

    eval_interval = max(1, max_steps // 200)
    train_acc_90_step = None
    test_acc_90_step = None

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
            _scheduler.step()
            step += 1

            if step % eval_interval == 0 or step >= max_steps:
                _, train_acc = evaluate(model, train_x, train_y, device, criterion)
                _, test_acc = evaluate(model, test_x, test_y, device, criterion)
                if train_acc >= 0.9 and train_acc_90_step is None:
                    train_acc_90_step = step
                if test_acc >= 0.9 and test_acc_90_step is None:
                    test_acc_90_step = step
                if train_acc_90_step and test_acc_90_step:
                    break
            if step >= max_steps:
                break
        if train_acc_90_step and test_acc_90_step:
            break

    return classify_phase(train_acc_90_step, test_acc_90_step)


def run_phase_diagram(
    prime: int,
    operation: str,
    train_fraction: float,
    d_model: int,
    n_heads: int,
    n_layers: int,
    d_ff: int,
    max_steps: int,
    seed: int,
    device: torch.device,
    outdir: Path,
    embed_lr: float,
    lr_range: tuple[float, float] = (1e-4, 1e-1),
    wd_range: tuple[float, float] = (1e-2, 1e1),
    grid_size: int = 8,
) -> None:
    """Generate a phase diagram by sweeping decoder lr and decoder weight decay.

    Reproduces Paper 1, Figure 6: each cell in the grid is classified into
    Comprehension / Grokking / Memorization / Confusion.
    """
    try:
        import matplotlib.pyplot as plt
        from matplotlib.colors import ListedColormap
        from matplotlib.patches import Patch
    except Exception:
        print("matplotlib not available, skip phase diagram.")
        return

    lrs = np.logspace(np.log10(lr_range[0]), np.log10(lr_range[1]), grid_size)
    wds = np.logspace(np.log10(wd_range[0]), np.log10(wd_range[1]), grid_size)

    phase_map = {"Comprehension": 0, "Grokking": 1, "Memorization": 2, "Confusion": 3}
    phase_colors = ["#2ecc71", "#f39c12", "#e74c3c", "#95a5a6"]
    grid = np.zeros((grid_size, grid_size), dtype=int)

    total = grid_size * grid_size
    for i, wd in enumerate(wds):
        for j, lr_val in enumerate(lrs):
            run_idx = i * grid_size + j + 1
            print(f"Phase diagram: run {run_idx}/{total} (lr={lr_val:.1e}, wd={wd:.1e})")
            phase = run_single_for_phase(
                prime=prime, operation=operation, train_fraction=train_fraction,
                lr=lr_val, embed_lr=embed_lr, weight_decay=wd,
                d_model=d_model, n_heads=n_heads,
                n_layers=n_layers, d_ff=d_ff, max_steps=max_steps,
                seed=seed, device=device,
            )
            grid[i, j] = phase_map[phase]

    fig, ax = plt.subplots(figsize=(8, 7))
    cmap = ListedColormap(phase_colors)
    ax.imshow(grid, cmap=cmap, vmin=0, vmax=3, origin="lower",
              aspect="auto", interpolation="nearest")

    ax.set_xticks(range(grid_size))
    ax.set_xticklabels([f"{lr_val:.0e}" for lr_val in lrs], rotation=45, fontsize=7)
    ax.set_yticks(range(grid_size))
    ax.set_yticklabels([f"{wd:.0e}" for wd in wds], fontsize=7)
    ax.set_xlabel("Decoder Learning Rate")
    ax.set_ylabel("Decoder Weight Decay")
    ax.set_title(f"Phase Diagram (embed_lr fixed at {embed_lr:.0e})")

    legend_elements = [
        Patch(facecolor=c, label=name)
        for name, c in zip(phase_map.keys(), phase_colors)
    ]
    ax.legend(handles=legend_elements, loc="upper left", fontsize=9)

    fig.tight_layout()
    fig.savefig(outdir / "phase_diagram.png", dpi=150)
    plt.close(fig)
    print(f"Phase diagram saved to {outdir / 'phase_diagram.png'}")


# ---------------------------------------------------------------------------
# Combined plotting: original curves + effective dimensionality
# ---------------------------------------------------------------------------

def _ema_smooth(values: list[float], alpha: float = 0.9) -> list[float]:
    """Exponential moving average for curve smoothing."""
    smoothed = []
    s = values[0]
    for v in values:
        s = alpha * s + (1 - alpha) * v
        smoothed.append(s)
    return smoothed


def maybe_plot(metrics_csv: Path, out_png: Path) -> None:
    try:
        import matplotlib.pyplot as plt
    except Exception:
        print("matplotlib not available, skip plot generation.")
        return

    steps: list[int] = []
    train_accs: list[float] = []
    test_accs: list[float] = []
    train_losses: list[float] = []
    test_losses: list[float] = []
    eff_dims: list[float] = []

    with metrics_csv.open("r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            steps.append(int(row["step"]))
            train_losses.append(float(row["train_loss"]))
            test_losses.append(float(row["test_loss"]))
            train_accs.append(float(row["train_acc"]))
            test_accs.append(float(row["test_acc"]))
            if "eff_dim" in row and row["eff_dim"]:
                eff_dims.append(float(row["eff_dim"]))

    alpha = 0.9  # EMA smoothing factor

    n_plots = 3 if eff_dims else 2
    fig, axes = plt.subplots(1, n_plots, figsize=(5 * n_plots, 4))

    # Accuracy: raw (faint) + smoothed (bold)
    axes[0].plot(steps, train_accs, color="tab:blue", alpha=0.2, linewidth=0.8)
    axes[0].plot(steps, test_accs, color="tab:orange", alpha=0.2, linewidth=0.8)
    axes[0].plot(steps, _ema_smooth(train_accs, alpha), color="tab:blue", linewidth=1.8, label="train")
    axes[0].plot(steps, _ema_smooth(test_accs, alpha), color="tab:orange", linewidth=1.8, label="test")
    axes[0].set_title("Accuracy")
    axes[0].set_xlabel("step")
    axes[0].set_ylabel("acc")
    axes[0].set_ylim(0.0, 1.05)
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Loss: raw (faint) + smoothed (bold), log scale
    axes[1].plot(steps, train_losses, color="tab:blue", alpha=0.2, linewidth=0.8)
    axes[1].plot(steps, test_losses, color="tab:orange", alpha=0.2, linewidth=0.8)
    axes[1].plot(steps, _ema_smooth(train_losses, alpha), color="tab:blue", linewidth=1.8, label="train")
    axes[1].plot(steps, _ema_smooth(test_losses, alpha), color="tab:orange", linewidth=1.8, label="test")
    axes[1].set_title("Cross-Entropy Loss")
    axes[1].set_xlabel("step")
    axes[1].set_ylabel("loss")
    axes[1].set_yscale("log")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    if eff_dims:
        axes[2].plot(steps, eff_dims, color="tab:purple", alpha=0.2, linewidth=0.8)
        axes[2].plot(steps, _ema_smooth(eff_dims, alpha), color="tab:purple", linewidth=1.8)
        axes[2].set_title(r"Effective Dimensionality $e^S$")
        axes[2].set_xlabel("step")
        axes[2].set_ylabel(r"$e^S$")
        axes[2].grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_png, dpi=140)
    plt.close(fig)


def parse_args() -> Config:
    parser = argparse.ArgumentParser(description="Grokking with paper-faithful visualizations.")
    parser.add_argument("--prime", type=int, default=53)
    parser.add_argument("--operation", choices=["add", "sub", "mul", "div"], default="add")
    parser.add_argument("--train-fraction", type=float, default=0.3)
    parser.add_argument(
        "--batch-size", type=int, default=0,
        help="0 means full-batch training over the train split.",
    )
    parser.add_argument("--max-steps", type=int, default=100000)
    parser.add_argument("--eval-interval", type=int, default=100)
    parser.add_argument(
        "--lr", type=float, default=1e-3,
        help="Learning rate for transformer blocks + output head (decoder side).",
    )
    parser.add_argument(
        "--embed-lr", type=float, default=1e-3,
        help="Learning rate for token/position embeddings. Paper Figure 7 fixes this at 1e-3.",
    )
    parser.add_argument("--weight-decay", type=float, default=1.0)
    parser.add_argument("--d-model", type=int, default=256)
    parser.add_argument("--n-heads", type=int, default=4)
    parser.add_argument("--n-layers", type=int, default=2)
    parser.add_argument("--d-ff", type=int, default=1024)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda", "mps"])
    parser.add_argument("--outdir", type=str, default="runs/grokking_baseline")
    parser.add_argument(
        "--stop-test-acc", type=float, default=0.99,
        help="Stop after test accuracy reaches this threshold (+ extra steps). Default 0.99.",
    )
    parser.add_argument(
        "--stop-extra-steps", type=int, default=0,
        help="Extra steps after stop_test_acc threshold. 0 = run full max_steps (just log milestones). "
             ">0 = stop N steps after threshold for early termination.",
    )
    # --- Training stability ---
    parser.add_argument(
        "--warmup-steps", type=int, default=10,
        help="Linear LR warmup over first N steps (paper uses 10).",
    )
    parser.add_argument(
        "--grad-clip", type=float, default=1.0,
        help="Max gradient norm for clipping. 0 disables clipping.",
    )
    parser.add_argument(
        "--lr-schedule", type=str, default="constant",
        choices=["constant", "cosine"],
        help="LR schedule after warmup. 'constant' (paper, needed for circular PCA structure) or 'cosine' decay.",
    )
    parser.add_argument(
        "--lr-min-ratio", type=float, default=0.05,
        help="Cosine schedule decays LR to lr * lr_min_ratio (default 0.05 = 5%% floor). "
             "Keeps weight decay effective throughout training.",
    )
    # --- New visualization arguments ---
    parser.add_argument(
        "--embed-snapshot-steps", type=str, default="",
        help="Comma-separated extra steps for PCA snapshots (in addition to auto-interval). "
             "Empty = auto-interval only.",
    )
    parser.add_argument(
        "--pca-interval", type=int, default=2000,
        help="Auto-snapshot token embeddings every N steps for PCA plot. 0 = disabled. Default 2000.",
    )
    parser.add_argument(
        "--phase-diagram", action="store_true",
        help="Run phase diagram sweep instead of single training.",
    )
    parser.add_argument("--phase-grid-size", type=int, default=8)
    parser.add_argument("--phase-max-steps", type=int, default=20000)

    args = parser.parse_args()
    snap_steps = [int(s) for s in args.embed_snapshot_steps.split(",") if s.strip()] \
        if args.embed_snapshot_steps.strip() else []

    return Config(
        prime=args.prime,
        operation=args.operation,
        train_fraction=args.train_fraction,
        batch_size=args.batch_size,
        max_steps=args.max_steps,
        eval_interval=args.eval_interval,
        lr=args.lr,
        embed_lr=args.embed_lr,
        weight_decay=args.weight_decay,
        d_model=args.d_model,
        n_heads=args.n_heads,
        n_layers=args.n_layers,
        d_ff=args.d_ff,
        dropout=args.dropout,
        seed=args.seed,
        device=args.device,
        outdir=args.outdir,
        stop_test_acc=args.stop_test_acc,
        stop_extra_steps=args.stop_extra_steps,
        warmup_steps=args.warmup_steps,
        grad_clip=args.grad_clip,
        lr_schedule=args.lr_schedule,
        lr_min_ratio=args.lr_min_ratio,
        embed_snapshot_steps=snap_steps,
        pca_interval=args.pca_interval,
        phase_diagram=args.phase_diagram,
        phase_grid_size=args.phase_grid_size,
        phase_max_steps=args.phase_max_steps,
    )


def resolve_device(name: str) -> torch.device:
    if name == "cpu":
        return torch.device("cpu")
    if name == "cuda":
        return torch.device("cuda")
    if name == "mps":
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def main() -> None:
    cfg = parse_args()
    set_seed(cfg.seed)

    if cfg.operation == "div" and cfg.prime < 3:
        raise ValueError("prime must be >= 3 for division task.")
    if cfg.n_heads < 1 or cfg.d_model % cfg.n_heads != 0:
        raise ValueError("d_model must be divisible by n_heads.")
    if not (0.0 < cfg.train_fraction < 1.0):
        raise ValueError("train_fraction must be in (0, 1).")

    outdir = Path(cfg.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    (outdir / "config.json").write_text(json.dumps(asdict(cfg), indent=2))

    device = resolve_device(cfg.device)
    print(f"Using device: {device}")

    # --- Visualization: Operation Table (Paper 2, Fig 1) ---
    plot_operation_table(cfg.prime, cfg.operation, outdir / "operation_table.png")
    print(f"Operation table saved to {outdir / 'operation_table.png'}")

    # --- Phase diagram mode ---
    if cfg.phase_diagram:
        run_phase_diagram(
            prime=cfg.prime, operation=cfg.operation,
            train_fraction=cfg.train_fraction,
            d_model=cfg.d_model, n_heads=cfg.n_heads,
            n_layers=cfg.n_layers, d_ff=cfg.d_ff,
            max_steps=cfg.phase_max_steps, seed=cfg.seed,
            device=device, outdir=outdir,
            embed_lr=cfg.embed_lr,
            grid_size=cfg.phase_grid_size,
        )
        return

    x, y = make_dataset(cfg.prime, cfg.operation, cfg.seed)
    train_x, train_y, test_x, test_y = split_dataset(x, y, cfg.train_fraction)
    print(f"dataset size={x.shape[0]}, train={train_x.shape[0]}, test={test_x.shape[0]}")

    model = GrokkingTransformer(
        prime=cfg.prime,
        d_model=cfg.d_model,
        n_heads=cfg.n_heads,
        n_layers=cfg.n_layers,
        d_ff=cfg.d_ff,
        dropout=cfg.dropout,
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"model params={n_params:,}")

    batch_size = cfg.batch_size if cfg.batch_size > 0 else train_x.shape[0]
    dataset = TensorDataset(train_x, train_y)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=False,
        generator=torch.Generator().manual_seed(cfg.seed),
    )

    criterion = nn.CrossEntropyLoss()
    optimizer = build_optimizer(
        model,
        decoder_lr=cfg.lr,
        embed_lr=cfg.embed_lr,
        decoder_weight_decay=cfg.weight_decay,
    )
    scheduler = build_scheduler(
        optimizer,
        warmup_steps=cfg.warmup_steps,
        lr_schedule=cfg.lr_schedule,
        lr_min_ratio=cfg.lr_min_ratio,
        max_steps=cfg.max_steps,
    )

    # Tracking state for new visualizations
    embed_snapshots: dict[int, np.ndarray] = {}
    eff_dim_steps: list[int] = []
    eff_dim_values: list[float] = []
    snapshot_set = set(cfg.embed_snapshot_steps) if cfg.embed_snapshot_steps else set()

    def _maybe_snapshot(s: int) -> None:
        """Take PCA snapshot if step matches any trigger."""
        if s in embed_snapshots:
            return
        # Manual list
        if s in snapshot_set:
            embed_snapshots[s] = get_token_embeddings(model, cfg.prime)
            snapshot_set.discard(s)
            return
        # Auto-interval
        if cfg.pca_interval > 0 and s % cfg.pca_interval == 0:
            embed_snapshots[s] = get_token_embeddings(model, cfg.prime)

    # Snapshot at step 0 (initialization)
    embed_snapshots[0] = get_token_embeddings(model, cfg.prime)
    snapshot_set.discard(0)

    metrics_path = outdir / "metrics.csv"
    with metrics_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["step", "train_loss", "test_loss", "train_acc", "test_acc", "eff_dim"])

    step = 0
    best_test = -math.inf
    threshold_hit_step: int | None = None  # step when test_acc first >= stop_test_acc
    effective_max_steps = cfg.max_steps  # may extend by stop_extra_steps
    train_acc_99_logged = False
    test_acc_99_logged = False

    while step < effective_max_steps:
        for xb, yb in loader:
            model.train()
            xb = xb.to(device)
            yb = yb.to(device)
            logits = model(xb)
            loss = criterion(logits, yb)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            if cfg.grad_clip > 0:
                nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            optimizer.step()
            scheduler.step()

            step += 1
            _maybe_snapshot(step)

            if step == 1 or step % cfg.eval_interval == 0 or step >= effective_max_steps:
                train_loss, train_acc = evaluate(model, train_x, train_y, device, criterion)
                test_loss, test_acc = evaluate(model, test_x, test_y, device, criterion)
                best_test = max(best_test, test_acc)

                # Compute effective dimensionality
                emb = get_token_embeddings(model, cfg.prime)
                eff_dim = compute_effective_dim(emb)
                eff_dim_steps.append(step)
                eff_dim_values.append(eff_dim)

                with metrics_path.open("a", newline="") as f:
                    writer = csv.writer(f)
                    writer.writerow([step, train_loss, test_loss, train_acc, test_acc,
                                     f"{eff_dim:.4f}"])

                # Milestone PCA snapshots
                if train_acc >= 0.99 and not train_acc_99_logged:
                    train_acc_99_logged = True
                    if step not in embed_snapshots:
                        embed_snapshots[step] = get_token_embeddings(model, cfg.prime)
                    print(f"  ** Memorization milestone: train_acc >= 0.99 at step={step}")
                if test_acc >= 0.99 and not test_acc_99_logged:
                    test_acc_99_logged = True
                    if step not in embed_snapshots:
                        embed_snapshots[step] = get_token_embeddings(model, cfg.prime)
                    print(f"  ** Grokking milestone: test_acc >= 0.99 at step={step}")

                extra_info = ""
                if threshold_hit_step is not None:
                    remaining = effective_max_steps - step
                    extra_info = f" [consolidating, {remaining} steps left]"

                print(
                    f"step={step:6d} "
                    f"train_loss={train_loss:.4f} test_loss={test_loss:.4f} "
                    f"train_acc={train_acc:.4f} test_acc={test_acc:.4f} "
                    f"eff_dim={eff_dim:.2f}{extra_info}"
                )

                # Stopping logic: log threshold hit; if stop_extra_steps > 0, stop early
                if threshold_hit_step is None and test_acc >= cfg.stop_test_acc:
                    threshold_hit_step = step
                    if cfg.stop_extra_steps > 0:
                        effective_max_steps = min(
                            step + cfg.stop_extra_steps, cfg.max_steps
                        )
                        print(
                            f"  >> test_acc >= {cfg.stop_test_acc:.4f} at step={step}. "
                            f"Early stop in {cfg.stop_extra_steps} extra steps "
                            f"(until step {effective_max_steps})."
                        )
                    else:
                        print(
                            f"  >> test_acc >= {cfg.stop_test_acc:.4f} at step={step}. "
                            f"Continuing to max_steps={cfg.max_steps} (stop_extra_steps=0)."
                        )

            if step >= effective_max_steps:
                # Snapshot at final step
                if step not in embed_snapshots:
                    embed_snapshots[step] = get_token_embeddings(model, cfg.prime)
                break

        if step >= effective_max_steps:
            break

    # Save checkpoint
    ckpt_path = outdir / "model.pt"
    torch.save({"model_state_dict": model.state_dict(), "config": asdict(cfg)}, ckpt_path)

    # --- Generate all visualizations ---
    print("\nGenerating visualizations...")

    # 1. Combined metrics curve (accuracy + loss + effective dim)
    maybe_plot(metrics_path, outdir / "curve.png")
    print(f"  curve.png")

    # 2. PCA embedding evolution (Paper 1, Fig 1)
    if embed_snapshots:
        plot_pca_evolution(embed_snapshots, cfg.prime, outdir / "pca_evolution.png")
        print(f"  pca_evolution.png")

    # 3. Effective dimensionality (Paper 1, Fig 7)
    if eff_dim_steps:
        plot_effective_dim(eff_dim_steps, eff_dim_values, outdir / "effective_dim.png")
        print(f"  effective_dim.png")

    # 4. t-SNE / PCA of output weights (Paper 2, Fig 3)
    plot_output_tsne(model, cfg.prime, outdir / "output_weights_tsne.png")
    print(f"  output_weights_tsne.png")

    print(f"\nbest_test_acc={best_test:.4f}")
    print(f"saved metrics to {metrics_path}")
    print(f"saved checkpoint to {ckpt_path}")
    print("done")


if __name__ == "__main__":
    main()
