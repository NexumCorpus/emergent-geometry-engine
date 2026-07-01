"""Phase-0 acceptance gate (Constitution sec 6).

The single command that decides whether Phase 0 passes. Prints PASS/FAIL per gate
item with the ACTUAL numbers (evidence before claims). This is the top-level
orchestrator, so it is allowed to import BOTH the verifier and the kernel — unlike
verify/, which stays independent.

Gate (Constitution): Peschel <-> independent agreement < 1e-9; full inequality suite
passes; curvature stable under the fixed recipe; spectral-gap bound lambda_1 >=
kappa_min confirmed (and tight) on test graphs; robustness harness runs.

    python phase0_gate.py
"""
from __future__ import annotations

import sys

import numpy as np
import networkx as nx

from verify.__main__ import run_physics_checks, PESCHEL_ANCHOR_N8_L4
from verify.peschel import xx_entanglement_entropy
from verify.ed import xx_reduced_entropy_ed
from verify.qutip_check import qutip_available, xx_entropy_qutip
from geometry.config import FROZEN_RECIPE
from geometry.robustness import robustness_sweep


def _ring_distance_matrix(n: int) -> np.ndarray:
    ang = 2 * np.pi * np.arange(n) / n
    pts = np.column_stack([np.cos(ang), np.sin(ang)])
    return np.linalg.norm(pts[:, None, :] - pts[None, :, :], axis=2)


def _status(ok: bool) -> str:
    return "PASS" if ok else "FAIL"


def gate():
    items = []  # (name, status, detail) where status in {PASS, FAIL, SKIP}

    # --- Gate 1: free-fermion entropy, independent routes agree < 1e-9 ---
    s_p = xx_entanglement_entropy(8, 4, bc="open")
    s_ed = xx_reduced_entropy_ed(8, 4, bc="open")
    items.append(("free-fermion: Peschel vs anchor",
                  _status(abs(s_p - PESCHEL_ANCHOR_N8_L4) < 1e-9),
                  f"|S - anchor| = {abs(s_p - PESCHEL_ANCHOR_N8_L4):.2e}"))
    items.append(("free-fermion: Peschel vs independent ED",
                  _status(abs(s_p - s_ed) < 1e-9),
                  f"|S_Peschel - S_ED| = {abs(s_p - s_ed):.2e}"))
    if qutip_available():
        s_q = xx_entropy_qutip(8, 4)
        items.append(("free-fermion: Peschel vs QuTiP",
                      _status(abs(s_p - s_q) < 1e-9),
                      f"|S_Peschel - S_QuTiP| = {abs(s_p - s_q):.2e}"))
    else:
        # Distinct SKIP — does NOT count toward the passing tally (an absent QuTiP
        # means one fewer real cross-check, and the gate says so plainly).
        items.append(("free-fermion: Peschel vs QuTiP", "SKIP", "QuTiP not installed"))

    # --- Gates 2 & 3: inequality suite + spectral-gap bound (from the pure grader) ---
    for name, ok, detail in run_physics_checks():
        if name.startswith("inequality:") or name.startswith("spectral_gap:"):
            items.append((name, _status(ok), detail))

    # --- Gate 4: robustness harness runs and reports stability ---
    res = robustness_sweep(_ring_distance_matrix(20), k_values=(2, 3, 4), alpha_values=(0.0, 0.5))
    sdim = res.stability["spectral_dim"]
    ran = len(res.configs) == 6 and all(np.isfinite(c["delta"]) for c in res.configs)
    items.append(("robustness: harness runs over (k, alpha) grid", _status(ran),
                  f"{len(res.configs)} configs; spectral_dim mean={sdim['mean']:.3f} cov={sdim['cov']:.3f}"))

    return items


def main() -> int:
    items = gate()
    width = max(len(n) for n, _, _ in items)
    bar = "=" * (width + 34)
    print(bar)
    print("EMERGENT GEOMETRY ENGINE -- PHASE 0 ACCEPTANCE GATE")
    print(f"frozen recipe: {FROZEN_RECIPE}")
    print(bar)
    n_pass = n_fail = n_skip = 0
    for name, status, detail in items:
        n_pass += status == "PASS"
        n_fail += status == "FAIL"
        n_skip += status == "SKIP"
        print(f"  [{status}] {name:<{width}}  {detail}")
    print("-" * (width + 34))
    verdict = "PASS" if n_fail == 0 else "FAIL"
    skip_note = f", {n_skip} skipped" if n_skip else ""
    print(f"PHASE 0 GATE: {verdict}   ({n_pass} passed, {n_fail} failed{skip_note})")
    print(bar)
    return 0 if n_fail == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
