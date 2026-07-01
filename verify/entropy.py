"""Numpy-only quantum-information primitives for the verifier.

No QuTiP, no kernel imports. Everything here is independently re-derivable so it can
serve as an honest grader for the rest of the project.
"""
from __future__ import annotations

import string

import numpy as np

# Eigenvalues below this are treated as exact zero (they contribute 0 to entropy in
# the limit x->0+, x ln x -> 0). Guards against log of tiny-negative numerical noise.
_EIG_FLOOR = 1e-12


def von_neumann_entropy(rho: np.ndarray, base: float | str | None = None) -> float:
    """S(rho) = -Tr(rho ln rho), computed from the Hermitian eigenvalues.

    base: None or 'e' -> natural log (nats); 2 -> bits; 10 -> dits.
    Tiny negative eigenvalues (floating-point noise) are clipped to 0.
    """
    vals = np.linalg.eigvalsh(_hermitian(rho)).real
    vals = np.clip(vals, 0.0, None)
    nz = vals[vals > _EIG_FLOOR]
    s = float(-np.sum(nz * np.log(nz)))
    if base in (None, "e"):
        return s
    return s / np.log(float(base))


def partial_trace(rho: np.ndarray, dims: list[int], keep: list[int]) -> np.ndarray:
    """Partial trace of a density matrix over the subsystems not in ``keep``.

    dims: subsystem dimensions, in order. keep: indices of subsystems to retain.
    Uses an einsum that contracts the row/column index of each traced subsystem.
    """
    dims = list(dims)
    n = len(dims)
    keep = sorted(keep)
    if int(np.prod(dims)) != rho.shape[0]:
        raise ValueError(f"prod(dims)={np.prod(dims)} != rho dim {rho.shape[0]}")

    t = np.asarray(rho).reshape(dims + dims)
    letters = list(string.ascii_letters)
    if 2 * n > len(letters):
        raise ValueError("too many subsystems for einsum-letter scheme")

    row_idx = [letters[i] for i in range(n)]
    # Traced subsystems reuse the row letter in the column slot -> contracted.
    col_idx = [letters[n + i] if i in keep else letters[i] for i in range(n)]
    out = [row_idx[i] for i in keep] + [col_idx[i] for i in keep]
    subscripts = "".join(row_idx) + "".join(col_idx) + "->" + "".join(out)

    res = np.einsum(subscripts, t)
    d_keep = int(np.prod([dims[i] for i in keep])) if keep else 1
    return res.reshape(d_keep, d_keep)


def _hermitian(rho: np.ndarray) -> np.ndarray:
    """Symmetrize away tiny anti-Hermitian numerical drift before eigendecomp."""
    rho = np.asarray(rho)
    return 0.5 * (rho + rho.conj().T)
