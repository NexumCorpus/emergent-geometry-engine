"""Render THE figure (Constitution Phase 4): the geometric order parameter vs the
2RDM test-block SHC vs gradient-norm, overlaid against train/test accuracy, with
honest lead-time annotations. Reads artifacts/grokking_results_<TAG>.npz.

Set GROK_TAG to pick the run (default frac050).
"""
from __future__ import annotations

import os
import json

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ART = os.path.join(os.path.dirname(__file__), "..", "..", "artifacts")
TAG = os.environ.get("GROK_TAG", "frac050")


def _norm(v, invert=False):
    v = np.asarray(v, dtype=float)
    m = ~np.isnan(v)
    out = np.full_like(v, np.nan)
    if m.sum() < 2:
        return out
    lo, hi = np.nanmin(v[m]), np.nanmax(v[m])
    if hi <= lo:
        return out
    out[m] = (v[m] - lo) / (hi - lo)
    return (1.0 - out) if invert else out


def main():
    d = np.load(os.path.join(ART, f"grokking_results_{TAG}.npz"))
    with open(os.path.join(ART, f"grokking_summary_{TAG}.json")) as f:
        summ = json.load(f)
    step = d["step"]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8.5), sharex=True)

    ax1.plot(step, d["train_acc"], color="#888", lw=2, label="train acc")
    ax1.plot(step, d["test_acc"], color="#1f77b4", lw=2.5, label="test acc (grokking)")
    ax1.set_ylabel("accuracy"); ax1.set_ylim(-0.03, 1.03)
    ax1.legend(loc="center right")
    ax1.set_title(f"Grokking head-to-head ({TAG}): does graph geometry deflect before the 2RDM signal / test accuracy?")

    # geometry as "ring-formation" = 1 - normalized Gromov delta (rises as the graph
    # becomes less tree-like / more ring-like). Plus the competitor SHC and grad-norm.
    ax2.plot(step, _norm(d["gromov_delta"], invert=True), color="#d62728", lw=2.5,
             label="ring-formation = 1 - norm(Gromov $\\delta$)  [geometry]")
    ax2.plot(step, _norm(d["var_forman"], invert=True), color="#9467bd", lw=1.6, alpha=0.8,
             label="curvature uniformity = 1 - norm(var Forman)  [geometry]")
    ax2.plot(step, _norm(d["shc_test"]), color="#2ca02c", lw=2,
             label="2RDM test-block SHC  [competitor]")
    ax2.plot(step, _norm(d["grad_norm"]), color="#ff7f0e", lw=1.4, alpha=0.7,
             label="gradient norm  [control]")
    ax2.set_ylabel("signal (min-max normalized)")
    ax2.set_xlabel("training step"); ax2.legend(loc="center right", fontsize=8)

    for key, color, lab in [
        ("delta_knee_step", "#d62728", "$\\delta$ knee"),
        ("grok_step_test_acc>0.9", "#1f77b4", "grok"),
        ("shc_test_peak_step", "#2ca02c", "SHC peak"),
    ]:
        s = summ.get(key)
        if s is not None:
            for ax in (ax1, ax2):
                ax.axvline(s, color=color, ls="--", alpha=0.7, lw=1.4)
            ax1.text(s, 1.05, lab, color=color, fontsize=8, rotation=30, ha="left")

    lead_g = summ.get("delta_lead_over_grok")
    lead_s = summ.get("delta_lead_over_shc_peak")
    sub = (f"$\\delta$ knee={summ.get('delta_knee_step')}  grok={summ.get('grok_step_test_acc>0.9')}  "
           f"SHC peak={summ.get('shc_test_peak_step')}  |  "
           f"geometry lead over grok = {lead_g}, over SHC peak = {lead_s} steps  |  "
           f"mean-Forman deflects: {summ.get('mean_forman_deflects')} (degree-dominated, reported as null)")
    fig.text(0.5, 0.005, sub, ha="center", fontsize=8.5, style="italic")

    fig.tight_layout(rect=[0, 0.03, 1, 1])
    out = os.path.join(ART, f"grokking_headtohead_{TAG}.png")
    fig.savefig(out, dpi=130)
    print("wrote", os.path.abspath(out))


if __name__ == "__main__":
    main()
