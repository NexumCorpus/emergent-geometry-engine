"""Adapter N — neural representational similarity.

Turns a set of probe activations into a symmetric similarity matrix for the kernel.
We use cosine similarity / linear CKA, NOT estimated mutual information: MI estimators
are badly biased on high-dimensional neural activations (Constitution robustness fix #2).
"""
from __future__ import annotations

import numpy as np


def _center_columns(X: np.ndarray) -> np.ndarray:
    return X - X.mean(axis=0, keepdims=True)


def linear_cka(X: np.ndarray, Y: np.ndarray) -> float:
    """Linear Centered Kernel Alignment between two activation sets (same n rows).

    CKA(X, Y) = ||Yc^T Xc||_F^2 / (||Xc^T Xc||_F * ||Yc^T Yc||_F), columns centered.
    Invariant to orthogonal transforms and isotropic scaling; in [0, 1]; 1 for X~X.
    """
    Xc = _center_columns(np.asarray(X, dtype=float))
    Yc = _center_columns(np.asarray(Y, dtype=float))
    hsic_xy = np.linalg.norm(Yc.T @ Xc, "fro") ** 2
    norm_x = np.linalg.norm(Xc.T @ Xc, "fro")
    norm_y = np.linalg.norm(Yc.T @ Yc, "fro")
    denom = norm_x * norm_y
    if denom < 1e-30:
        return 0.0
    return float(hsic_xy / denom)


def cosine_similarity_matrix(A: np.ndarray) -> np.ndarray:
    """(n, n) cosine-similarity matrix among the n rows of A (nodes = rows).

    This is the natural node-similarity object for the kernel's graph: each node is a
    probe item (e.g. a token's activation vector); the entry is the cosine of the angle
    between their representation vectors. Diagonal is 1; zero-norm rows map to 0.
    """
    A = np.asarray(A, dtype=float)
    norms = np.linalg.norm(A, axis=1, keepdims=True)
    norms = np.where(norms < 1e-30, 1.0, norms)
    An = A / norms
    S = An @ An.T
    return np.clip(S, -1.0, 1.0)
