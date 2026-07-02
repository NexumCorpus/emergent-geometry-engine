"""Finite-size-scaling tools for the Phase 3 stretch (Constitution Q1).

Two standard instruments, deliberately simple and fully deterministic:

- `crossing_points`: pairwise crossings of O(p; L) curves. If O is a scale-
  invariant quantity at the transition, curves for different L intersect at
  p_c (up to corrections to scaling); the spread of pairwise crossings is an
  honest systematic-error estimate.
- `fit_collapse`: grid-search fit of (p_c, nu) for the one-parameter scaling
  form O(p, L) = f((p - p_c) * L^(1/nu)) by minimizing the master-curve
  residual (each size's points vs the interpolated curve of all other sizes,
  normalized by the pooled variance — Houdayer-Hartmann in spirit). No
  amplitude exponent is fitted: appropriate for bounded/dimensionless
  observables; a poor best-fit quality is itself the honest answer for
  observables that do not scale this way.

Graded by tests/test_fss.py on synthetic data with known (p_c, nu), including
a structureless negative control that must NOT produce a good collapse.
"""
from __future__ import annotations

import numpy as np


def _smooth(y: np.ndarray, w: int = 5) -> np.ndarray:
    if w <= 1 or len(y) < w:
        return np.asarray(y, dtype=float)
    k = np.ones(w) / w
    return np.convolve(np.asarray(y, dtype=float), k, mode="same")


def crossing_points(data: dict) -> list[float]:
    """Pairwise crossing locations of O(p; L) curves.

    data: {L: (ps, ys)}. For each size pair the curves are lightly smoothed,
    interpolated onto a common fine grid, and the sign change of their
    difference with the STEEPEST local slope is taken as the crossing (end-of-
    range sign flips where saturating curves converge have near-zero slope and
    are rejected by this criterion).
    """
    sizes = sorted(data)
    out = []
    for a in range(len(sizes)):
        for b in range(a + 1, len(sizes)):
            p1, y1 = data[sizes[a]]
            p2, y2 = data[sizes[b]]
            lo = max(np.min(p1), np.min(p2))
            hi = min(np.max(p1), np.max(p2))
            grid = np.linspace(lo, hi, 600)
            d = (np.interp(grid, p1, _smooth(y1))
                 - np.interp(grid, p2, _smooth(y2)))
            sign = np.sign(d)
            flips = np.nonzero(sign[:-1] * sign[1:] < 0)[0]
            if flips.size == 0:
                continue
            slopes = np.abs(np.gradient(d, grid))
            best = max(flips, key=lambda i: slopes[i])
            # linear interpolation of the root inside the flip interval
            x0, x1 = grid[best], grid[best + 1]
            y0, y1_ = d[best], d[best + 1]
            out.append(float(x0 - y0 * (x1 - x0) / (y1_ - y0)))
    return out


def collapse_quality(data: dict, p_c: float, nu: float,
                     min_overlap: int = 10) -> float:
    """Master-curve residual of the scaling form at (p_c, nu).

    Each size's rescaled points are compared to the interpolated master curve
    built from all OTHER sizes over the overlapping x-range; the mean squared
    deviation is normalized by the pooled variance of O. Returns np.inf when
    fewer than `min_overlap` points land in overlap (no evidence either way).
    """
    sizes = sorted(data)
    resc = {}
    for L in sizes:
        p, y = data[L]
        resc[L] = ((np.asarray(p, float) - p_c) * L ** (1.0 / nu),
                   np.asarray(y, float))
    pooled = np.concatenate([resc[L][1] for L in sizes])
    var = float(np.var(pooled))
    if var <= 0:
        return float("inf")

    total, cnt = 0.0, 0
    for L in sizes:
        xo = np.concatenate([resc[M][0] for M in sizes if M != L])
        yo = np.concatenate([resc[M][1] for M in sizes if M != L])
        order = np.argsort(xo)
        xo, yo = xo[order], yo[order]
        xL, yL = resc[L]
        inside = (xL >= xo[0]) & (xL <= xo[-1])
        if not inside.any():
            continue
        yhat = np.interp(xL[inside], xo, yo)
        total += float(np.sum((yL[inside] - yhat) ** 2))
        cnt += int(inside.sum())
    if cnt < min_overlap:
        return float("inf")
    return (total / cnt) / var


def fit_collapse(data: dict):
    """Grid-search (p_c, nu) minimizing `collapse_quality`; coarse pass then a
    local refinement. Returns (p_c, nu, quality)."""
    ps_all = np.concatenate([np.asarray(p, float) for p, _ in data.values()])
    lo, hi = float(ps_all.min()), float(ps_all.max())
    span = hi - lo

    def search(pcs, nus):
        best = (None, None, float("inf"))
        for pc in pcs:
            for nu in nus:
                q = collapse_quality(data, pc, nu)
                if q < best[2]:
                    best = (float(pc), float(nu), float(q))
        return best

    pcs = np.linspace(lo + 0.05 * span, hi - 0.05 * span, 60)
    nus = np.linspace(0.5, 3.0, 51)
    pc0, nu0, _ = search(pcs, nus)
    pcs = np.linspace(pc0 - 0.03 * span, pc0 + 0.03 * span, 41)
    nus = np.linspace(max(0.3, nu0 - 0.3), nu0 + 0.3, 41)
    return search(pcs, nus)
