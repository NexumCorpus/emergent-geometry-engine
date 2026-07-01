"""Optional QuTiP cross-check — a THIRD independent computation of the XX-chain
entanglement entropy via exact many-body ptrace + entropy_vn (Constitution sec 2.2:
"QuTiP entropy_vn must match the correlation-matrix method to <1e-9").

Gated behind a soft import so the verifier never breaks when QuTiP is absent; the
project's correctness does not depend on it (the numpy ED path in verify/ed.py is the
guaranteed independent cross-check).
"""
from __future__ import annotations

try:
    import qutip as _qt
    _HAVE_QUTIP = True
except Exception:  # pragma: no cover - environment dependent
    _qt = None
    _HAVE_QUTIP = False


def qutip_available() -> bool:
    return _HAVE_QUTIP


def _jw_annihilators(N: int):
    """Jordan-Wigner fermion annihilation operators on N sites (open chain)."""
    sm, sz, I = _qt.sigmam(), _qt.sigmaz(), _qt.qeye(2)
    return [
        _qt.tensor([sz if l < j else sm if l == j else I for l in range(N)])
        for j in range(N)
    ]


def xx_entropy_qutip(N: int, L: int) -> float:
    """Entanglement entropy (nats) of the first L sites of the open XX chain, via
    QuTiP exact diagonalization, ptrace, and entropy_vn (default base e)."""
    if not _HAVE_QUTIP:
        raise RuntimeError("QuTiP is not installed")
    c = _jw_annihilators(N)
    H = sum(-(c[i].dag() * c[i + 1] + c[i + 1].dag() * c[i]) for i in range(N - 1))
    _, psi = H.groundstate()
    rho_A = psi.ptrace(list(range(L)))
    return float(_qt.entropy_vn(rho_A))  # base e -> nats
