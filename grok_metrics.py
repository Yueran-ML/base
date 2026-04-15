#!/usr/bin/env python3
"""
grok_metrics.py — Canonical metric utilities shared across all stages.

All Grokking phase-diagram scripts must import from here so that
_classify_phase and tau estimators are guaranteed identical everywhere.

Phase classification contract (GROK_GAP = 2000 steps):
  Grokking      : tau_train detected, tau_gen detected, (tau_gen - tau_train) >= GROK_GAP
  Comprehension : tau_train detected, tau_gen detected, (tau_gen - tau_train) <  GROK_GAP
  Memorization  : tau_train detected, tau_gen NOT detected
  Confusion     : tau_train NOT detected

tau_train / tau_gen are both estimated with _find_tau_sustained (n_sustained=3
consecutive checkpoints ≥ threshold), so the classifier is robust to transient
spikes.

History
-------
2026-03-30  Created to unify Stage 0 and Stage 1 implementations.
            Stage 0 used single-point detection + 1000-step gap.
            Stage 1 used sustained detection + early_frac=0.33 criterion.
            Both were wrong in different ways for the boundary cells at
            lr=1.6e-3, wd=2.5 (Stage 1 mis-labelled them Comprehension).
            Canonical rule: sustained detection + absolute GROK_GAP = 2000.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

GROK_GAP: int = 2000        # minimum tau_gen − tau_train gap to be called Grokking
ACC_THRESHOLD: float = 0.9  # train/test accuracy threshold
N_SUSTAINED: int = 3        # consecutive checkpoints required for tau detection


# ---------------------------------------------------------------------------
# Tau estimators
# ---------------------------------------------------------------------------

def find_tau_sustained(
    steps: list[int] | np.ndarray,
    values: list[float] | np.ndarray,
    threshold: float = ACC_THRESHOLD,
    n_sustained: int = N_SUSTAINED,
) -> Optional[float]:
    """First step where values >= threshold for n_sustained consecutive checkpoints.

    Returns None if the threshold is never sustained.
    Used for tau_gen (test_acc >= 0.9) and tau_train (train_acc >= 0.9).
    """
    n = len(steps)
    for i in range(n - n_sustained + 1):
        if all(values[i + k] >= threshold for k in range(n_sustained)):
            return float(steps[i])
    return None


def estimate_changepoint(
    steps: list[int] | np.ndarray,
    values: list[float] | np.ndarray,
    min_frac: float = 1 / 6,
    max_frac: float = 5 / 6,
    min_post_points: int = 5,
    slope_rel_threshold: float = 0.01,
) -> Optional[float]:
    """BIC changepoint estimator for Fourier alignment onset (tau_F).

    Fits 1-break segmented linear regression on log(1+t) scale.
    Returns the breakpoint step only if it strictly improves BIC over no-break
    and the post-break segment has a non-negligible slope.

    Returns None if no credible changepoint is found.
    """
    n = len(steps)
    if n < 12:
        return None
    log_t = np.log1p(np.array(steps, dtype=float))
    v = np.array(values, dtype=float)

    # Exponential moving average smoothing
    smoothed = v.copy()
    alpha = 0.15
    for i in range(1, n):
        smoothed[i] = alpha * v[i] + (1.0 - alpha) * smoothed[i - 1]

    def ols_rss(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
        A = np.column_stack([x, np.ones(len(x))])
        coeffs, res, _, _ = np.linalg.lstsq(A, y, rcond=None)
        if len(res) == 0:
            pred = A @ coeffs
            rss = float(np.sum((y - pred) ** 2))
        else:
            rss = float(res[0])
        return float(coeffs[0]), rss

    lo = max(3, int(min_frac * n))
    hi = min(n - 3, int(max_frac * n))
    _, rss0 = ols_rss(log_t, smoothed)
    bic_nobreak = n * np.log(max(rss0 / n, 1e-15)) + 2.0 * np.log(n)
    best_bic = bic_nobreak
    best_bp = None
    v_range = float(np.ptp(smoothed))

    for bp in range(lo, hi + 1):
        x1, y1 = log_t[:bp], smoothed[:bp]
        x2, y2 = log_t[bp:], smoothed[bp:]
        if len(x1) < 2 or len(x2) < 2:
            continue
        slope1, rss1 = ols_rss(x1, y1)
        slope2, rss2 = ols_rss(x2, y2)
        total_rss = rss1 + rss2
        bic = n * np.log(max(total_rss / n, 1e-15)) + 4.0 * np.log(n)
        if bic < best_bic:
            if abs(slope2) > slope_rel_threshold * max(v_range, 1e-9):
                if (n - bp) >= min_post_points:
                    best_bic = bic
                    best_bp = bp

    if best_bp is None:
        return None
    return float(steps[best_bp])


# ---------------------------------------------------------------------------
# Phase classifier (canonical)
# ---------------------------------------------------------------------------

def classify_phase(
    steps: list[int] | np.ndarray,
    train_acc: list[float] | np.ndarray,
    test_acc: list[float] | np.ndarray,
    acc_threshold: float = ACC_THRESHOLD,
    grok_gap: int = GROK_GAP,
    n_sustained: int = N_SUSTAINED,
) -> str:
    """Canonical phase label for a single (lr, wd, seed) run.

    Parameters
    ----------
    steps       : checkpoint step indices
    train_acc   : training accuracy at each checkpoint
    test_acc    : test accuracy at each checkpoint
    acc_threshold : threshold defining "high accuracy" (default 0.9)
    grok_gap    : minimum tau_gen - tau_train gap (steps) to call Grokking
                  vs Comprehension (default 2000)
    n_sustained : consecutive checkpoints required for tau detection (default 3)

    Returns
    -------
    One of: "Grokking", "Comprehension", "Memorization", "Confusion"
    """
    tau_train = find_tau_sustained(steps, train_acc, acc_threshold, n_sustained)
    tau_gen   = find_tau_sustained(steps, test_acc,  acc_threshold, n_sustained)

    if tau_train is None:
        return "Confusion"
    if tau_gen is None:
        return "Memorization"
    if (tau_gen - tau_train) >= grok_gap:
        return "Grokking"
    return "Comprehension"


# ---------------------------------------------------------------------------
# Fourier alignment
# ---------------------------------------------------------------------------

def compute_fourier_alignment(embeddings: np.ndarray, prime: int) -> tuple[float, int]:
    """Best R² of centered embeddings onto any single harmonic pair sin/cos(2πk/p).

    Returns (best_r2, best_k).
    """
    E = embeddings - embeddings.mean(0)
    total_var = float(np.sum(E ** 2))
    if total_var < 1e-12:
        return 0.0, 1
    thetas = 2.0 * np.pi * np.arange(prime) / prime
    best_r2, best_k = 0.0, 1
    for k in range(1, prime // 2 + 1):
        B = np.stack([np.cos(k * thetas), np.sin(k * thetas)], axis=1)
        Q, _ = np.linalg.qr(B)
        proj = Q @ (Q.T @ E)
        res_var = float(np.sum((E - proj) ** 2))
        r2 = 1.0 - res_var / total_var
        if r2 > best_r2:
            best_r2, best_k = r2, k
    return float(best_r2), best_k


def compute_fourier_null_p95(
    embeddings: np.ndarray,
    prime: int,
    n_perms: int = 100,
    rng: Optional[np.random.Generator] = None,
) -> float:
    """95th percentile of permutation-null Fourier alignment scores."""
    if rng is None:
        rng = np.random.default_rng(0)
    emb = embeddings.copy()
    idx = np.arange(prime)
    scores = []
    for _ in range(n_perms):
        rng.shuffle(idx)
        score, _ = compute_fourier_alignment(emb[idx], prime)
        scores.append(score)
    return float(np.percentile(scores, 95))


# ---------------------------------------------------------------------------
# Discrete-log Fourier alignment (for multiplicative tasks)
# ---------------------------------------------------------------------------

def _primitive_root(p: int) -> int:
    """Find the smallest primitive root of prime p."""
    from math import gcd
    order = p - 1
    # factorise order
    factors = set()
    n = order
    d = 2
    while d * d <= n:
        while n % d == 0:
            factors.add(d)
            n //= d
        d += 1
    if n > 1:
        factors.add(n)
    for g in range(2, p):
        if all(pow(g, order // f, p) != 1 for f in factors):
            return g
    raise ValueError(f"No primitive root found for p={p}")


def _discrete_log_table(p: int) -> dict[int, int]:
    """Return dlog[i] = k  s.t.  g^k ≡ i (mod p), for i = 1 .. p-1."""
    g = _primitive_root(p)
    table: dict[int, int] = {}
    x = 1
    for k in range(p - 1):
        table[x] = k
        x = (x * g) % p
    return table


def compute_fourier_alignment_mul(
    embeddings: np.ndarray,
    prime: int,
) -> tuple[float, int]:
    """Fourier alignment for the *multiplication* task via discrete-log reindexing.

    For (a·b) mod p the token embeddings should develop a cyclic structure
    indexed by discrete logarithm, not by token index directly.
    Steps:
      1. Exclude token 0 (multiplicatively degenerate).
      2. Reorder tokens 1..p-1 by their discrete log position k = log_g(i).
      3. Compute standard R² Fourier alignment on the (p-1)-element sequence.

    Returns (best_r2, best_k).
    """
    dlog = _discrete_log_table(prime)
    order = prime - 1          # cyclic group order
    E_raw = embeddings         # shape (prime, d)

    # Build reordered embedding: position k ← embedding of token g^k
    E = np.zeros((order, E_raw.shape[1]), dtype=float)
    for tok in range(1, prime):
        k = dlog[tok]
        E[k] = E_raw[tok]

    E = E - E.mean(0)
    total_var = float(np.sum(E ** 2))
    if total_var < 1e-12:
        return 0.0, 1

    thetas = 2.0 * np.pi * np.arange(order) / order
    best_r2, best_k = 0.0, 1
    for k in range(1, order // 2 + 1):
        B = np.stack([np.cos(k * thetas), np.sin(k * thetas)], axis=1)
        Q, _ = np.linalg.qr(B)
        proj = Q @ (Q.T @ E)
        res_var = float(np.sum((E - proj) ** 2))
        r2 = 1.0 - res_var / total_var
        if r2 > best_r2:
            best_r2, best_k = r2, k
    return float(best_r2), best_k


def compute_fourier_null_p95_mul(
    embeddings: np.ndarray,
    prime: int,
    n_perms: int = 100,
    rng: Optional[np.random.Generator] = None,
) -> float:
    """Permutation-null p95 for the discrete-log Fourier alignment (mul task).

    Permutes the *dlog-reordered* rows (i.e., shuffles the mapping from
    dlog position to embedding), preserving the token 0 exclusion.
    """
    if rng is None:
        rng = np.random.default_rng(0)

    dlog = _discrete_log_table(prime)
    order = prime - 1
    E_raw = embeddings

    # Build baseline reordered embedding
    E_base = np.zeros((order, E_raw.shape[1]), dtype=float)
    for tok in range(1, prime):
        k = dlog[tok]
        E_base[k] = E_raw[tok]

    idx = np.arange(order)
    scores = []
    for _ in range(n_perms):
        rng.shuffle(idx)
        score, _ = compute_fourier_alignment(E_base[idx], order)
        scores.append(score)
    return float(np.percentile(scores, 95))


# ---------------------------------------------------------------------------
# Ordering label
# ---------------------------------------------------------------------------

def compute_ordering(tau_gen: Optional[float], tau_F: Optional[float]) -> str:
    """Classify the timing relationship between generalization and Fourier onset.

    Returns one of: "G<F", "F<G", "F_only", "G_only", "none"
    """
    if tau_gen is not None and tau_F is not None:
        return "G<F" if tau_gen < tau_F else "F<G"
    if tau_F is not None:
        return "F_only"
    if tau_gen is not None:
        return "G_only"
    return "none"


# ---------------------------------------------------------------------------
# Fourier logit alignment (τ_circuit)
# ---------------------------------------------------------------------------

def compute_fourier_logit_alignment(
    logits: np.ndarray,
    prime: int,
    operation: str = "add",
) -> tuple[float, int]:
    """R² of all output logits explained by a single Fourier frequency of (a op b) mod p.

    Measures how much of the logit variance across all (a,b) input pairs is
    captured by a harmonic component of the sum/product class, quantifying
    the degree to which the circuit routes through Fourier modes.

    Parameters
    ----------
    logits    : shape (prime*prime, prime) — output logits for every (a,b) pair,
                in row-major order (row i = pair (a=i//p, b=i%p)).
    prime     : the modulus p.
    operation : "add" or "mul".  For "mul", token 0 rows are excluded and
                the frequency is indexed over (a*b) mod p classes.

    Returns
    -------
    (best_r2, best_k)
    """
    p = prime
    L = np.array(logits, dtype=float).reshape(p * p, p)

    if operation == "mul":
        # Exclude rows where a=0 or b=0 (multiplicatively degenerate)
        a_idx = np.arange(p * p) // p
        b_idx = np.arange(p * p) % p
        mask = (a_idx != 0) & (b_idx != 0)
        L = L[mask]
        sums_flat = (a_idx[mask] * b_idx[mask]) % p
    else:
        a_idx = np.arange(p * p) // p
        b_idx = np.arange(p * p) % p
        sums_flat = (a_idx + b_idx) % p

    L_c = L - L.mean(0)
    total_var = float(np.sum(L_c ** 2))
    if total_var < 1e-12:
        return 0.0, 1

    best_r2, best_k = 0.0, 1
    for k in range(1, p // 2 + 1):
        angles = 2.0 * np.pi * k * sums_flat / p
        F = np.stack([np.cos(angles), np.sin(angles)], axis=1)  # (n, 2)
        Q, _ = np.linalg.qr(F)
        proj = Q @ (Q.T @ L_c)
        res_var = float(np.sum((L_c - proj) ** 2))
        r2 = 1.0 - res_var / total_var
        if r2 > best_r2:
            best_r2, best_k = r2, k

    return float(best_r2), best_k


def compute_fourier_logit_null_p95(
    logits: np.ndarray,
    prime: int,
    operation: str = "add",
    n_perms: int = 100,
    rng: Optional[np.random.Generator] = None,
) -> float:
    """95th percentile of permutation-null Fourier logit alignment scores.

    Permutes the (a+b) mod p label assignment while keeping the logit matrix
    fixed, giving the baseline R² under no Fourier structure.
    """
    if rng is None:
        rng = np.random.default_rng(0)

    p = prime
    L = np.array(logits, dtype=float).reshape(p * p, p)

    if operation == "mul":
        a_idx = np.arange(p * p) // p
        b_idx = np.arange(p * p) % p
        mask = (a_idx != 0) & (b_idx != 0)
        L = L[mask]
        sums_flat = (a_idx[mask] * b_idx[mask]) % p
    else:
        a_idx = np.arange(p * p) // p
        b_idx = np.arange(p * p) % p
        sums_flat = (a_idx + b_idx) % p

    L_c = L - L.mean(0)
    total_var = float(np.sum(L_c ** 2))
    if total_var < 1e-12:
        return 0.0

    scores = []
    perm_sums = sums_flat.copy()
    for _ in range(n_perms):
        rng.shuffle(perm_sums)
        best_r2 = 0.0
        for k in range(1, p // 2 + 1):
            angles = 2.0 * np.pi * k * perm_sums / p
            F = np.stack([np.cos(angles), np.sin(angles)], axis=1)
            Q, _ = np.linalg.qr(F)
            proj = Q @ (Q.T @ L_c)
            r2 = 1.0 - float(np.sum((L_c - proj) ** 2)) / total_var
            if r2 > best_r2:
                best_r2 = r2
        scores.append(best_r2)

    return float(np.percentile(scores, 95))
