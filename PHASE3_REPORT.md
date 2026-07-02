# Phase 3 — MIPT-geometry sweep (Track A)

**Date:** 2026-07-02 · **Branch:** `phase3/mipt-sweep` · **Artifacts:** `artifacts/mipt_sweep_v1/`
**Gate (Constitution §6):** "Per-trajectory curvature/δ/spectral-dim vs p, ≥3 sizes, with I₃
control; h-MDS stress vs p. Clean comparison to I₃; documented signal or honest null."

## Verdict: SIGNAL — the critical point is geometrically distinguished

Every geometric observable of the per-trajectory block-MI graph shows an extremum at the
transition located independently by the I₃ control, and the two most physical ones
strengthen with system size.

## Setup

- Monitored Clifford brickwork chain: C1⊗C1·CNOT bricks (uniform single-qubit Cliffords,
  enumerated exactly), per-site Z-measurement rate p after each layer, T = 2L, open chain.
  **Ensemble-specific p_c located by I₃ — no literature value assumed.**
- Sizes L = 32, 48, 64; 13 rates p ∈ [0.02, 0.30] densified near the transition;
  **200 trajectories per point**, every trajectory reproducible via
  `SeedSequence([base_seed=20260702, L, p_index, k])`.
- Geometry per trajectory (Constitution §4.2 crux), then mean/sem — never annealed MI.
- Graph: MI between contiguous 4-site blocks (instrument repair — site-level pairwise MI
  is degenerate at ALL p because scrambled entanglement is multipartite; probe documented
  in `block_mi_matrix`). MI_FLOOR non-edge rule; frozen kernel recipe (kNN k=8, ORC-MANL
  pruning); hop metric. Pre-registered before the sweep; geometric observables' p-dependence
  was NOT examined pre-sweep (connectivity diagnostics were, and are labeled as such).

## I₃ control (the transition locator)

I₃ is strongly negative in the volume-law phase (−16.4 bits at L=64, p=0.02), rises to ~0
at **p_c ≈ 0.11–0.13**, and the rise sharpens with L (finite-size trend in the right
direction). This is the trusted diagnostic all geometric claims are compared against.

## Findings (per observable, mean over 200 trajectories, sem bars in figure)

| observable | behavior at the I₃-located transition | size trend |
|---|---|---|
| spectral dimension d_s | **peak** at p ≈ 0.12–0.14 | peak **grows** with L: 0.54 → 0.64 → 0.72 |
| mean Forman curvature | **minimum** at p ≈ 0.08–0.13 | minimum **deepens** with L: 0.67 → 0.58 → 0.52 |
| Gromov δ / diameter | peak at p ≈ 0.11–0.14 | height non-monotone in L (0.052/0.033/0.039) |
| h-MDS distortion (rank 2) | peak at transition | stable ~0.04 |
| Euclidean MDS stress (dim 2) | peak at transition | stable ~0.05 |
| largest-component fraction | peak at p ≈ 0.10–0.13 | fraction falls with L (block count grows) |
| long-range MI edges | peak at p ≈ 0.13–0.14 | **grows** with L: 1.8 → 2.5 → 3.0 |
| mean block MI | peak at p ≈ 0.11–0.14 | falls with L (normalization) |

**Q1 (are curvature / δ / d_s order parameters?): qualitatively supported.** All three
locate p_c as an extremum without being told where it is; d_s and Forman strengthen with
system size. A quantitative finite-size-scaling collapse (critical exponents) was NOT
attempted and is required before any "order parameter" language leaves the qualitative
register.

**Q2 (is the critical point geometrically distinguished / maximally hyperbolic?):
"distinguished" — yes, cleanly. "Maximally hyperbolic" — NOT established.** Two honest
obstructions: (a) off-critical graphs are near-chains or fragments, whose δ = 0 is
*trivially* hyperbolic, so "δ minimal at p_c" cannot be tested against these baselines as
posed; observed δ/diameter in fact peaks at p_c because only there does the graph have
cycles at all. (b) h-MDS distortion and Euclidean stress are not like-for-like metrics, so
their comparison at p_c does not yet decide hyperbolic-vs-flat. A like-for-like comparison
(same distortion functional, plus a scale-learned h-MDS per Sala et al. §4) is the concrete
next step if Q2 matters downstream.

## Physics side-findings from instrument validation

1. Site-level pairwise MI graphs are blind at every p — multipartite scrambling; this is
   the same fact that makes I₃ (not pairwise MI) the standard diagnostic.
2. Long-range block-MI connectivity peaks precisely at p_c — consistent with power-law MI
   at criticality. The MI graph is *richest exactly at the transition*, which is what makes
   per-trajectory geometry readable there and trivial away from it.

## Caveats (load-bearing)

- p_c ≈ 0.11–0.13 is ensemble-specific (C1⊗C1·CNOT bricks, not uniform 2-qubit Cliffords).
- Block graphs are small (8/12/16 nodes): geometric observables are coarse; peak positions
  carry ±0.01–0.02 resolution in p.
- T = 2L steady-state protocol; no purification-time check at the smallest p.
- 1D chain: some geometric hypotheses (ring embedding, higher-d structure) are not
  addressable in this geometry at all.

## Addendum (2026-07-02 evening): Q2 like-for-like embedding comparison — POST-HOC EXPLORATORY

The Q2 obstruction (b) above is addressable: score BOTH embeddings with the same
functional (`avg_distortion`) on the same per-trajectory LCC hop metric — Euclidean MDS
dim 2 (scale-invariant, no search) vs h-MDS rank 2 with a per-trajectory scale search
τ ∈ {0.25 … 4} (h-MDS is scale-sensitive; Sala et al. §4). Fresh trajectories (base seed
777, NOT the sweep's), 100 per point (`experiments/mipt/q2_embedding.py`):

- **Hyperbolic fits better at every (L, p) probed** (euclid − hyper > 0, 5–10σ per point).
- **The hyperbolic advantage is largest near the transition**: at p=0.12 the gap is
  +0.017…+0.022 vs +0.008…+0.010 at p=0.08 (all three sizes agree).

Reading: where the emergent metric is nontrivial, it is better described by negatively
curved space than flat space, most strongly near criticality — directionally the Q2 /
holographic expectation. Caveats: post-hoc (informs Q2 language only, feeds no certified
claim); the τ search gives the hyperbolic side a discrete free parameter the Euclidean
side does not need (mitigated by Euclidean scale-invariance, but the df asymmetry stands);
graphs remain small. A pre-registered version belongs in any follow-up sweep.

## Reproduce

```
E:\_phase0_test_venv\Scripts\python.exe -m experiments.mipt.run \
  --sizes 32,48,64 --ps 0.02,0.05,0.08,0.10,0.11,0.12,0.13,0.14,0.15,0.17,0.20,0.25,0.30 \
  --ntraj 200 --out artifacts/mipt_sweep_v1
python -m experiments.mipt.analyze artifacts/mipt_sweep_v1
```
Suite: 102 green (`python -m pytest`). Figure: `artifacts/mipt_sweep_v1/phase3_gate.png`.
