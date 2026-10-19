<div align="center">
  <img src="./nexum-hero.svg" alt="Emergent Geometry Engine — geometry for phase change" width="100%" />

  <p><strong>Turn correlation into geometry—and geometry into an instrument for phase change.</strong></p>

  <p><a href="#what-it-is">What it is</a> · <a href="#five-minutes">Five minutes</a> · <a href="#evidence">Evidence</a> · <a href="#phase-gates">Phase gates</a> · <a href="https://github.com/NexumCorpus">Nexum Corpus</a></p>
</div>

---

## What it is

The Emergent Geometry Engine is a substrate-agnostic instrument that constructs
an emergent geometry from any correlation / similarity matrix, computes its discrete-geometry invariants
(Ollivier / Lin–Lu–Yau & Forman–Ricci curvature, Gromov δ-hyperbolicity, spectral
dimension), embeds it (Euclidean + hyperbolic MDS), and uses those invariants as
**order parameters / early-warning signals** for integration↔fragmentation
transitions.

## Why it matters

Phase transitions in complex systems (a quantum circuit crossing from volume-law to
area-law entanglement, a network that suddenly generalizes) are usually read off
one bespoke statistic per field. If the *shape* of the correlation graph moves at
the transition, one instrument can watch many substrates. This repo tests that idea
with a verifier that never imports the code it grades, so a passing number is not
the kernel agreeing with itself.

## Five minutes

Python 3.12+ (`numpy==2.5.0` needs it; developed and CI-tested on 3.13). From a fresh clone:

```bash
python -m venv .venv
# Windows:      .venv\Scripts\activate
# macOS/Linux:  source .venv/bin/activate
pip install -r requirements.txt                         # ~1-2 min

python -m verify                                        # 11 physics/math checks, ~3 s, prints PASS/FAIL each
python -m pytest -q                                     # 104 pass, 4 skip (QuTiP optional), ~15 s
python artifacts/mipt_sweep_v1/bundle/check.py 777      # re-derive the Phase-3 claim on a seed you pick, ~10 s
python phase0_gate.py                                   # Phase-0 acceptance gate: 13/13 with QuTiP, 12 pass + 1 skip without
```

The last command draws a *fresh* 50-trajectory ensemble from your seed and checks
ordering assertions (no stored numbers), so it cannot pass on memorized output.
QuTiP is an **optional** third cross-check (`pip install -r requirements-quantum.txt`);
the core verifier runs without it via the independent numpy exact-diagonalization path.

## Evidence

| Phase | Result | Where |
|---|---|---|
| 0 — kernel + verifier | **PASS**, acceptance gate 13/13 with the actual numbers | [docs/PHASE0_REPORT.md](docs/PHASE0_REPORT.md) |
| 3 — measurement-induced transition (Clifford circuits, L = 32/48/64, 200 traj/point) | **SIGNAL**: every geometric observable is extremal at the transition located independently by the I₃ control | [PHASE3_REPORT.md](PHASE3_REPORT.md), `artifacts/mipt_sweep_v1/` |
| 4 — grokking head-to-head (does geometry move *before* test accuracy?) | **NO**, under the frozen pre-registered recipe | [docs/PHASE4_REPORT.md](docs/PHASE4_REPORT.md) |

The Phase-4 null is reported with the same weight as the Phase-3 signal; that is the point.

## What it does not claim

- Not a grand-unified theory. It is *one shared mathematical object* (graph curvature as
  an integration–fragmentation order parameter) applied to two substrates (quantum
  measurement-induced transitions; neural grokking). An analogy with a shared object.
- Not an early-warning signal for grokking: Phase 4 says geometry did not lead in that
  configuration.
- Phase 3's maximally-hyperbolic reading (Q2) is **not** established; only the
  qualitative extremum (Q1) is. Sweep v2 (four sizes, analysis frozen before the data)
  found the geometric observables locate the transition and sharpen with size but do
  **not** collapse under one- or two-exponent finite-size scaling, so "order parameter"
  stays qualitative: no critical-exponent claim.
- "Haven't seen this done," never "first."

## Where it sits

One of the research instruments of [Nexum Corpus](https://github.com/NexumCorpus) —
instruments for verifiable machine work. Its generator-≠-grader discipline is the same
one used across the estate (e.g. the
[Recursive Discovery Engine](https://github.com/NexumCorpus/recursive-discovery-engine)'s
frozen holdout). Built by Claude, directed by Nexum Corpus. MIT licensed.

## Architecture — one kernel, two adapters, two front-ends

```
                 ┌──────────────────────────────────────────────┐
   Adapter Q ──▶ │  KERNEL                                       │ ──▶ Front-end V
 (quantum MI)    │  matrix → kNN graph (+ shortcut-edge prune)   │     (audiovisual)
                 │        → Forman / Ollivier-LLY curvature      │
   Adapter N ──▶ │        → Gromov δ, spectral dimension         │ ──▶ Front-end A
 (neural CKA)    │        → Euclidean MDS + hyperbolic h-MDS      │     (analysis/EWS)
                 └──────────────────────────────────────────────┘
                                    ▲
                                    │ certified by
                 ┌──────────────────────────────────────────────┐
                 │  VERIFIER (separate code path, no kernel imports)
                 │  • entropy-inequality suite
                 │  • Peschel free-fermion cross-check (numpy-only)
                 │  • independent exact-diagonalization cross-check
                 │  • spectral-gap ≥ curvature bound (Lin–Lu–Yau)
                 │  • graph-construction robustness harness
                 └──────────────────────────────────────────────┘
```

## Governing discipline (non-negotiable)

1. **Generator ≠ grader.** The code that *computes* a quantity and the code that
   *certifies* it live on separate paths. The verifier does **not** import the
   kernel. The free-fermion entropy is checked *two mathematically independent
   ways* (correlation-matrix eigenvalues vs. full reduced-density-matrix
   diagonalization) and must agree to `< 1e-9`.
2. **Evidence before claims.** Show the test output / the actual number. Never
   assert success.
3. **Phase-gated.** Each phase has an acceptance gate; do not advance until it
   passes.
4. **Build the verifier alongside the feature.** Failing tests committed first.

## The four robustness fixes (kernel requirements, not options)

1. **Freeze** the Ollivier idleness `α` and the distance threshold as fixed
   constants *before* looking at any results (see `geometry/config.py`).
2. **Zero the dissimilarity diagonal**; use unbiased / cross-validated
   dissimilarity (diagonal entries and within-run correlation fake structure).
3. **k-NN graph + shortcut-edge pruning** *before* computing global curvature / δ.
4. **Forman / effective-resistance curvature for trajectories** (cheap); exact
   Ollivier / LLY only at anchor checkpoints (Wasserstein-LP is costly).

## Phase gates

| Phase | Deliverable | Gate |
|------|-------------|------|
| **0** | Kernel + Verifier + four robustness fixes | Peschel↔independent-ED agreement `<1e-9`; full inequality suite passes; curvature stable under the fixed recipe; spectral-gap bound `λ₁ ≥ κ_min` confirmed on test graphs; robustness harness runs |
| 1 | Track-A hero visual (ring → tear → bridge) | money-shot plays; product state shows no structure; critical chain embeds as a ring/disk |
| 2 | Sonification + interactivity | the tear is audiovisual and demo-able |
| 3 | Track-A MIPT-geometry sweep | clean comparison to I₃; documented signal or honest null |
| 4 | Track-B head-to-head plot (front-loadable) | the deciding figure exists; lead-time assessed honestly |
| 5 | Emergent-misalignment target; hyperbolic hardening; OTOC (stretch) | — |

## Layout

```
geometry/      # the kernel (graph build, curvature, δ, spectral dim, MDS)
verify/        # the verifier — SEPARATE code path, must not import geometry/
adapters/      # Adapter Q (quantum MI), Adapter N (neural CKA)
tests/         # pytest suite (failing tests committed first)
artifacts/     # committed evidence: sweep summaries, plots, the Phase-3 claim bundle
.claude/       # physics-verifier subagent + verification hooks
```
