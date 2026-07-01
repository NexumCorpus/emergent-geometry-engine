"""Entropy-inequality suite (Constitution sec 2.2), numpy-only.

Each check returns a CheckResult carrying the slack by which the inequality is
satisfied (>= 0 means satisfied). For von Neumann entropy S(rho) = -Tr(rho ln rho):

  trace          Tr(rho) = 1
  psd            rho >= 0
  S>=0           S(rho) >= 0
  subadditivity  S(AB) <= S(A) + S(B)
  SSA            S(ABC) + S(B) <= S(AB) + S(BC)
  Araki-Lieb     |S(A) - S(B)| <= S(AB)
  purity symm    pure global state => S(A) = S(complement)
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from verify.entropy import von_neumann_entropy, partial_trace

TOL = 1e-9


@dataclass
class CheckResult:
    name: str
    ok: bool
    slack: float  # amount the inequality is satisfied by; ~0 for equalities
    detail: str = ""


def check_trace_one(rho: np.ndarray, tol: float = TOL) -> CheckResult:
    dev = abs(complex(np.trace(rho)).real - 1.0) + abs(complex(np.trace(rho)).imag)
    return CheckResult("trace=1", dev < tol, tol - dev, f"|Tr-1|={dev:.2e}")


def check_positive_semidefinite(rho: np.ndarray, tol: float = TOL) -> CheckResult:
    rho = 0.5 * (rho + rho.conj().T)
    min_eig = float(np.linalg.eigvalsh(rho).min())
    return CheckResult("rho>=0", min_eig >= -tol, min_eig, f"min eig={min_eig:.2e}")


def check_entropy_nonnegative(rho: np.ndarray, tol: float = TOL) -> CheckResult:
    s = von_neumann_entropy(rho)
    return CheckResult("S>=0", s >= -tol, s, f"S={s:.6f}")


def check_subadditivity(rho_AB: np.ndarray, dims: list[int], tol: float = TOL) -> CheckResult:
    s_ab = von_neumann_entropy(rho_AB)
    s_a = von_neumann_entropy(partial_trace(rho_AB, dims, keep=[0]))
    s_b = von_neumann_entropy(partial_trace(rho_AB, dims, keep=[1]))
    slack = s_a + s_b - s_ab
    return CheckResult("subadditivity", slack >= -tol, slack,
                       f"S(A)+S(B)-S(AB)={slack:.6f}")


def check_strong_subadditivity(rho_ABC: np.ndarray, dims: list[int], tol: float = TOL) -> CheckResult:
    s_abc = von_neumann_entropy(rho_ABC)
    s_b = von_neumann_entropy(partial_trace(rho_ABC, dims, keep=[1]))
    s_ab = von_neumann_entropy(partial_trace(rho_ABC, dims, keep=[0, 1]))
    s_bc = von_neumann_entropy(partial_trace(rho_ABC, dims, keep=[1, 2]))
    slack = s_ab + s_bc - s_abc - s_b
    return CheckResult("strong_subadditivity", slack >= -tol, slack,
                       f"S(AB)+S(BC)-S(ABC)-S(B)={slack:.6f}")


def check_araki_lieb(rho_AB: np.ndarray, dims: list[int], tol: float = TOL) -> CheckResult:
    s_ab = von_neumann_entropy(rho_AB)
    s_a = von_neumann_entropy(partial_trace(rho_AB, dims, keep=[0]))
    s_b = von_neumann_entropy(partial_trace(rho_AB, dims, keep=[1]))
    slack = s_ab - abs(s_a - s_b)
    return CheckResult("araki_lieb", slack >= -tol, slack,
                       f"S(AB)-|S(A)-S(B)|={slack:.6f}")


def check_pure_state_symmetry(psi: np.ndarray, dims: list[int], partition: list[int],
                              tol: float = TOL) -> CheckResult:
    rho = np.outer(psi, np.conj(psi))
    complement = [i for i in range(len(dims)) if i not in partition]
    s_a = von_neumann_entropy(partial_trace(rho, dims, keep=partition))
    s_abar = von_neumann_entropy(partial_trace(rho, dims, keep=complement))
    dev = s_a - s_abar
    return CheckResult("pure_state_symmetry", abs(dev) < tol, dev,
                       f"S(A)-S(Abar)={dev:.2e}")
