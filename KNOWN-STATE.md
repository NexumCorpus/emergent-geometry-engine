# Known state at git freeze (2026-07-01)

Suite run with `E:\_phase0_test_venv` (networkx 3.6.1, numpy 2.5.0):
**83 passed, 1 failed.**

Failing: `tests/test_curvature.py::test_forman_augmented_matches_library` —
in-repo augmented Forman value is 0.0 where GraphRicciCurvature reports 101.72.
Suspected dependency-version drift versus the original certification environment
(Phase 0 was certified with 71 tests; suite has since grown to 84). Not diagnosed.

**Must be resolved (or attributed to env drift with evidence) before any capstone
Phase 5 campaign cites EGE results.**
