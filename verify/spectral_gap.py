"""Spectral-gap >= curvature certification (discrete Lichnerowicz / Lin-Lu-Yau).

The bound is on the NORMALIZED Laplacian L = I - D^{-1/2} A D^{-1/2}, NOT the
combinatorial D - A (for a d-regular graph they differ by a factor d). Using the
wrong Laplacian is the single most common way this check is silently wrong, so the
test suite asserts tightness on K_n / Q_n where only the normalized gap equals the
curvature.

Pure numpy + scipy-free; no kernel imports.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

_ZERO_TOL = 1e-9


@dataclass
class CheckResult:
    name: str
    ok: bool
    slack: float
    detail: str = ""


def normalized_laplacian(A: np.ndarray) -> np.ndarray:
    d = np.asarray(A).sum(axis=1)
    if np.any(d <= 0):
        raise ValueError("isolated vertex (zero degree) — normalized Laplacian undefined")
    dis = 1.0 / np.sqrt(d)
    return np.eye(A.shape[0]) - (dis[:, None] * A * dis[None, :])


def combinatorial_laplacian(A: np.ndarray) -> np.ndarray:
    d = np.asarray(A).sum(axis=1)
    return np.diag(d) - A


def spectral_gap(A: np.ndarray, normalized: bool = True) -> float:
    """First nonzero Laplacian eigenvalue lambda_1. Raises if the graph is
    disconnected (more than one near-zero eigenvalue), since lambda_1 is then
    ill-defined and the Lichnerowicz framework assumes connectivity."""
    L = normalized_laplacian(A) if normalized else combinatorial_laplacian(A)
    w = np.sort(np.linalg.eigvalsh(L).real)
    n_zero = int(np.sum(w < _ZERO_TOL))
    if n_zero != 1:
        raise ValueError(
            f"expected exactly one zero eigenvalue (connected graph), found {n_zero}; "
            "graph is disconnected or has an isolated component")
    return float(w[1])


def check_lichnerowicz(A: np.ndarray, kappa_min: float, expect_tight: bool = False,
                       tol: float = 1e-9) -> CheckResult:
    """Confirm lambda_1 >= kappa_min on the normalized Laplacian. If expect_tight,
    additionally require equality (the K_n / Q_n tight cases)."""
    lam1 = spectral_gap(A, normalized=True)
    slack = lam1 - kappa_min
    ok = slack >= -tol
    detail = f"lambda_1={lam1:.9f}, kappa_min={kappa_min:.9f}, slack={slack:.2e}"
    if expect_tight:
        ok = ok and abs(slack) < tol
        detail += " (expect tight)"
    return CheckResult("lichnerowicz_bound", ok, slack, detail)
