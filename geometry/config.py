"""Frozen kernel constants — robustness fix #1 (Constitution sec 3).

These are fixed BEFORE looking at any result. ORC curvature is hyperparameter
sensitive; tuning these per-dataset to make a signal appear is exactly the failure
mode the Constitution forbids. Report under this one recipe. Demonstrate robustness
only AFTER a signal exists, by SWEEPING these in verify/robustness.py — never by
quietly editing them here.
"""
from __future__ import annotations

# --- graph construction ---
KNN_K: int = 8                       # neighbours kept per node in the kNN graph
SIMILARITY_FLOOR: float = 1e-12      # similarities/distances below this are treated as 0

# --- Ollivier-Ricci curvature ---
OLLIVIER_ALPHA: float = 0.5          # idleness (mass kept at source) for fixed-alpha ORC
LLY_ALPHA_NEAR_ONE: float = 0.999    # alpha used to estimate the Lin-Lu-Yau (alpha->1) limit

# --- shortcut-edge pruning (ORC-MANL style, robustness fix #3) ---
# Negative Ollivier-Ricci curvature is the theoretical signature of a bottleneck /
# shortcut edge (Topping et al., over-squashing, ICLR 2022). So any negatively-curved
# edge is a shortcut CANDIDATE (threshold 0.0); the redundancy guard below — remove
# only if a parallel path keeps the endpoints within SHORTCUT_DISTANCE_THRESHOLD hops —
# is what prevents over-pruning a genuine bridge. Grounded in theory, not fit to data.
SHORTCUT_CURVATURE: float = 0.0      # edges with ORC below this are shortcut candidates
SHORTCUT_DISTANCE_THRESHOLD: int = 3  # remove a candidate only if its endpoints stay within
                                      # this many hops in the pruned graph (i.e. it was redundant)

# --- spectral dimension ---
SPECTRAL_DIM_NPTS: int = 60          # number of log-spaced heat-trace samples

# A single immutable record of the frozen recipe, for provenance in reports.
FROZEN_RECIPE = {
    "KNN_K": KNN_K,
    "OLLIVIER_ALPHA": OLLIVIER_ALPHA,
    "LLY_ALPHA_NEAR_ONE": LLY_ALPHA_NEAR_ONE,
    "SHORTCUT_CURVATURE": SHORTCUT_CURVATURE,
    "SHORTCUT_DISTANCE_THRESHOLD": SHORTCUT_DISTANCE_THRESHOLD,
}
