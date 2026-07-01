# Phase 4 — Track-B grokking head-to-head: evidence report

**The deciding question (Constitution §5.2):** does graph-curvature geometry deflect
*before* test accuracy / before the competitor's 2RDM spectral-heat-capacity signal?

**Answer (honest, this configuration): NO.** Under the frozen pre-registered recipe,
graph curvature is a **lagging / concurrent** indicator of grokking — it reorganizes
during the *cleanup* phase, at or after generalization. The **2RDM spectral heat
capacity (the competitor) carries the genuine early-warning signal**, elevated
throughout the pre-generalization plateau. This is a **null for the "curvature deflects
first" thesis**, and at best a complementary/downstream geometric *description* — exactly
the fallback the Constitution names (§5.3). "Earlier is empirical" — the lead-time plot
*was* the test, and it came back negative.

## Setup (all on the Phase-0 certified kernel)

- **Model:** hand-rolled 1-layer transformer (d_model=128, 4 heads, d_mlp=512, no
  LayerNorm), modular addition (a+b) mod p, p=59, loss on the `=` position. AdamW
  lr=1e-3, **weight_decay=1.0**, full-batch. (`experiments/grokking/model.py`)
- **Geometric signal:** at log-spaced checkpoints, the token-embedding rows `W_E`
  (p nodes) → cosine similarity (Adapter N) → kNN graph (frozen recipe) → mean Forman
  curvature, curvature variance, Gromov δ, spectral dimension. (`analysis/signals.py`)
- **Competitor (faithful):** 2RDM = n×n covariance of per-sample **losses** over a W=20
  checkpoint window; **SHC = Var(eigenvalues)**, computed on the test-block probe set
  (Hennick-Corlouer arXiv:2603.29805 — definition verified, not guessed).
- **Controls:** gradient L2 norm; train/test accuracy.

## Results (two regimes)

| regime | memorize | grok (test>0.9) | SHC peak | geometry (δ) commit-knee | verdict |
|---|---|---|---|---|---|
| frac=0.5 (fast) | 100 | 900 | 1100 | 3250 | δ **lags** grok by 2350, lags SHC by 2150 |
| frac=0.3 (wide gap) | 100 | 13800 | 2000 | 13750 | δ ≈ concurrent with grok; **SHC leads δ by 11750** |

- **mean Forman curvature is null** (it is `4−deg−deg+3T`, degree-dominated on a fixed-k
  kNN graph; it barely moves — reported transparently, not hidden).
- In the wide-gap regime the SHC is elevated (1e4–1e6) across the entire pre-grokking
  plateau and collapses ~10¹⁵× at generalization — a strong early warning. The geometric
  signals (δ, var-Forman) stay noisy through the plateau and only settle to their final
  values around/after the test-accuracy rise (the cleanup phase).

Figures: `artifacts/grokking_headtohead_frac050.png`, `..._frac030.png`.

## Seed ensemble (error bars) — frac=0.3, seeds {0..4}

The single-seed direction holds **across seeds, unanimously** where it counts
(`artifacts/grokking_ensemble_frac030.png`, `..._frac030.json`):

| metric | median | [min, max] |
|---|---|---|
| grok step (4/5 seeds; 1 stalled) | 13700 | [9100, 17050] |
| SHC peak step | **2000** | [1950, 3500] |
| **SHC lead over grok** (>0 = early warning) | **+11700** | [7150, 15050] — positive in all 4 |
| geometry (δ) lead over grok (≤0 = lags) | −250 | [−5900, +1700] — straddles 0 |
| **geometry lead over SHC** (<0 = later) | **−14450** | [−18450, −6400] — negative in all 5 |

- The **2RDM SHC fires at step ~2000 in every seed** (tight band) and leads grokking by a
  median **+11,700 steps** — a robust early-warning signal.
- The **geometric signal does not lead**: its knee is roughly concurrent with grokking
  (median −250, straddling zero) and **lags the SHC by a median ~14,500 steps in all 5
  seeds** (unanimous). High across-seed variance (wide band in the figure).
- **1/5 seeds (s3) stalled** — did not reach test_acc>0.9 within 25k steps — the known
  seed-variance of grok timing (budget was ~2× median grok; the research recommends
  ≥3–5×). Its SHC still fired early (step 3500); it is excluded from grok-based leads.

The ensemble converts "the direction is clear" into a statistically grounded null: with
error bars, graph curvature is not an earlier predictor of grokking than the spectral
signal — it trails it every time.

## A methodological honesty moment (evidence before claims, applied to my own analysis)

My first automated summary reported **"δ leads grokking by 11,850 steps"** in the
wide-gap regime — a headline-grade positive. **It was a false positive.** Inspecting the
raw δ trajectory (`inspect_run.py`) showed δ bounces around its plateau (~0.6) the whole
time and only durably drops *after* grokking; a single transient noise dip at step 2000
had fooled the naive first-crossing knee detector. Fix: the knee is now the **commit
step** — the point after which the smoothed signal stays deflected for the remainder —
which correctly places δ's reorganization at ~step 13750 (concurrent with grok), not
1950. The fragile detector is exactly the kind of self-deception the project is built to
prevent; the trajectory plot caught it.

## Honest caveats / scope

- ~~Single seed per regime.~~ **DONE** — frac=0.3 seed ensemble {0..4} above; the
  direction holds unanimously (geometry lags SHC in all 5). One seed stalled within
  budget (report it; don't drop it silently).
- **One instantiation of the geometric signal** (raw `W_E` cosine kNN, cheap invariants).
  A Fourier-projected probe, or exact Ollivier/LLY at anchor checkpoints, might behave
  differently — but under the frozen, pre-registered recipe the result is a null, and
  retuning the probe *after* seeing the null to manufacture a lead would be the forbidden
  move (§3 robustness fix #1).
- **Laptop scale** (p=59). The qualitative grokking phenomenology is standard; scaling is
  untested.

## What this is worth

A clean, certified-kernel-based head-to-head with a **faithful** competitor that **wins**.
That is a more valuable artifact than a flattering null-competitor result: it tells us the
graph-curvature lens, as posed, is a mechanistic *description* of the grokking
reorganization rather than an *earlier predictor* of it. The Constitution anticipated this
exact outcome and called it publishable.
