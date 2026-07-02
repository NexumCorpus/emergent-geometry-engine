# Known state (2026-07-02)

Suite run with `E:\_phase0_test_venv` (networkx 3.6.1, numpy 2.5.0,
GraphRicciCurvature 0.5.3.2): **84 passed, 0 failed.**

## Resolved: `test_forman_augmented_matches_library` (2026-07-02)

The 2026-07-01 freeze recorded this test failing (in-repo 0.0 vs library
101.72 on a karate edge). Diagnosed as dependency drift, with evidence:

- The test was written against GraphRicciCurvature 0.4.5, which ignored
  pre-existing edge weights; the venv has 0.5.3.2, which consumes them when
  present (its docstring still claims "edge weight will be ignored" — only
  the weight-*injection* default survived).
- `nx.karate_club_graph()` carries edge weights (e.g. `(0,1) weight=4`), so
  0.5.3.2 silently computed the WEIGHTED augmented variant → non-integer
  101.72. Unweighted Forman on any graph is integral under both formulas.
- On a weight-free tiny graph the library matches the in-repo 3T+4-deg-deg
  formula exactly (3, 2, 2, 0 on triangle+tail); on a weight-stripped karate
  club, **0 mismatches on all 78 edges**.

Verdict: the EGE kernel was correct throughout; the test now strips weights
before the library call, pinning the intended unweighted comparison. The
Phase 5 blocker recorded at freeze is lifted.
