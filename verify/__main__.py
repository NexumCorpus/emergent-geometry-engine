"""`python -m verify` — run the pure physics/math grader and print PASS/FAIL per
check with the actual numbers. This entry point imports ONLY verify/ (no kernel),
preserving the generator-not-grader independence.

For the full Phase-0 acceptance gate (physics + kernel robustness + pytest), use the
top-level phase0_gate.py instead.
"""
from __future__ import annotations

import sys

import numpy as np
import networkx as nx

from verify.inequalities import (
    check_trace_one, check_positive_semidefinite, check_entropy_nonnegative,
    check_subadditivity, check_strong_subadditivity, check_araki_lieb,
    check_pure_state_symmetry,
)
from verify.peschel import xx_entanglement_entropy
from verify.ed import xx_reduced_entropy_ed
from verify.spectral_gap import check_lichnerowicz

# Anchor (open BC, half filling, A=first 4): float64-precise, confirmed by the
# independent ED + QuTiP routes and a blind external re-derivation (< 5e-15).
PESCHEL_ANCHOR_N8_L4 = 0.5697484844216253


def _rng_density(dim, rank, seed):
    rng = np.random.default_rng(seed)
    M = rng.standard_normal((dim, rank)) + 1j * rng.standard_normal((dim, rank))
    rho = M @ M.conj().T
    return rho / np.trace(rho).real


def run_physics_checks():
    """Return a list of (name, ok, detail) tuples."""
    out = []

    rho3 = _rng_density(8, 8, 4)
    for res in (
        check_trace_one(rho3),
        check_positive_semidefinite(rho3),
        check_entropy_nonnegative(rho3),
        check_subadditivity(_rng_density(4, 4, 3), dims=[2, 2]),
        check_strong_subadditivity(rho3, dims=[2, 2, 2]),
        check_araki_lieb(_rng_density(4, 3, 5), dims=[2, 2]),
    ):
        out.append((f"inequality:{res.name}", res.ok, res.detail))

    rng = np.random.default_rng(6)
    psi = rng.standard_normal(8) + 1j * rng.standard_normal(8)
    psi /= np.linalg.norm(psi)
    sym = check_pure_state_symmetry(psi, dims=[2, 2, 2], partition=[0])
    out.append((f"inequality:{sym.name}", sym.ok, sym.detail))

    # Peschel: three-way agreement (correlation-matrix vs ED vs literature anchor).
    s_peschel = xx_entanglement_entropy(8, 4, bc="open")
    s_ed = xx_reduced_entropy_ed(8, 4, bc="open")
    out.append(("peschel:anchor_N8_L4",
                abs(s_peschel - PESCHEL_ANCHOR_N8_L4) < 1e-9,
                f"S_peschel={s_peschel:.15f} vs anchor {PESCHEL_ANCHOR_N8_L4}"))
    out.append(("peschel:vs_independent_ED",
                abs(s_peschel - s_ed) < 1e-9,
                f"|S_peschel - S_ED| = {abs(s_peschel - s_ed):.2e}"))

    # Spectral-gap >= curvature (Lichnerowicz), tight on K_5 and Q_3.
    A_k5 = nx.to_numpy_array(nx.complete_graph(5))
    r_k5 = check_lichnerowicz(A_k5, kappa_min=5.0 / 4.0, expect_tight=True)
    out.append(("spectral_gap:K5_tight", r_k5.ok, r_k5.detail))
    A_q3 = nx.to_numpy_array(nx.hypercube_graph(3),
                             nodelist=sorted(nx.hypercube_graph(3).nodes()))
    r_q3 = check_lichnerowicz(A_q3, kappa_min=2.0 / 3.0, expect_tight=True)
    out.append(("spectral_gap:Q3_tight", r_q3.ok, r_q3.detail))

    return out


def main() -> int:
    checks = run_physics_checks()
    width = max(len(name) for name, _, _ in checks)
    print("=" * (width + 30))
    print("EGE verifier — physics/math grader (verify/ only, no kernel)")
    print("=" * (width + 30))
    all_ok = True
    for name, ok, detail in checks:
        tag = "PASS" if ok else "FAIL"
        all_ok = all_ok and ok
        print(f"  [{tag}] {name:<{width}}  {detail}")
    print("-" * (width + 30))
    print(f"OVERALL: {'PASS' if all_ok else 'FAIL'}  ({sum(o for _, o, _ in checks)}/{len(checks)} checks)")
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
