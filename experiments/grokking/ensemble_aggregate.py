"""Aggregate the frac=0.3 seed ensemble into a result with error bars.

Reads grokking_results_frac030_s{0..4}.npz / _summary_..., reports per-seed and
median[min,max] lead times, and renders an ensemble figure with median +/- across-seed
band for the geometric signal vs the 2RDM SHC vs test accuracy.

Honest by construction: it reports whatever the seeds show. SHC lead > 0 means the
spectral competitor fires before grokking; geometry lead <= 0 means graph curvature lags.
"""
from __future__ import annotations

import os
import json
import glob

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ART = os.path.join(os.path.dirname(__file__), "..", "..", "artifacts")
PREFIX = "frac030_s"


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


def _stat(xs):
    xs = [x for x in xs if x is not None]
    if not xs:
        return None
    return {"median": float(np.median(xs)), "min": int(np.min(xs)), "max": int(np.max(xs)),
            "values": [int(x) for x in xs]}


def main():
    tags = sorted(os.path.basename(p)[len("grokking_results_"):-4]
                  for p in glob.glob(os.path.join(ART, f"grokking_results_{PREFIX}*.npz")))
    if not tags:
        print("no ensemble runs found")
        return
    print("seeds found:", tags)

    runs, summaries = {}, {}
    for t in tags:
        runs[t] = dict(np.load(os.path.join(ART, f"grokking_results_{t}.npz")))
        summaries[t] = json.load(open(os.path.join(ART, f"grokking_summary_{t}.json")))

    grok = [summaries[t].get("grok_step_test_acc>0.9") for t in tags]
    shc_peak = [summaries[t].get("shc_test_peak_step") for t in tags]
    dknee = [summaries[t].get("delta_knee_step") for t in tags]

    def lead(a_list, b_list):  # a - b per seed (positive = b before a... we pass grok, knee)
        return [(a - b) if (a is not None and b is not None) else None
                for a, b in zip(a_list, b_list)]

    agg = {
        "n_seeds": len(tags),
        "grok_step": _stat(grok),
        "shc_peak_step": _stat(shc_peak),
        "delta_knee_step": _stat(dknee),
        "SHC_lead_over_grok (positive = competitor early-warning)": _stat(lead(grok, shc_peak)),
        "geometry_lead_over_grok (<=0 = geometry lags)": _stat(lead(grok, dknee)),
        "geometry_lead_over_SHC (negative = geometry later than SHC)": _stat(lead(shc_peak, dknee)),
        "mean_forman_deflects_per_seed": [summaries[t].get("mean_forman_deflects") for t in tags],
    }
    json.dump(agg, open(os.path.join(ART, "grokking_ensemble_frac030.json"), "w"), indent=2)
    print("\n=== ENSEMBLE (frac=0.3) ===")
    print(json.dumps(agg, indent=2))

    # --- figure: median +/- across-seed band ---
    step = runs[tags[0]]["step"]
    def stack(key, invert=False):
        return np.vstack([_norm(runs[t][key], invert=invert) for t in tags])

    test_acc = np.vstack([runs[t]["test_acc"] for t in tags])
    shc = stack("shc_test")
    ring = stack("gromov_delta", invert=True)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8.5), sharex=True)
    for t in tags:
        ax1.plot(step, runs[t]["test_acc"], color="#1f77b4", alpha=0.25, lw=1)
    ax1.plot(step, np.nanmedian(test_acc, 0), color="#1f77b4", lw=2.5, label="test acc (median)")
    ax1.set_ylabel("test accuracy"); ax1.set_ylim(-0.03, 1.03); ax1.legend(loc="center right")
    ax1.set_title(f"Grokking head-to-head, frac=0.3, {len(tags)}-seed ensemble: SHC (competitor) vs graph geometry")

    for arr, color, lab in [(ring, "#d62728", "ring-formation = 1-norm Gromov $\\delta$ [geometry]"),
                            (shc, "#2ca02c", "2RDM test-block SHC [competitor]")]:
        med = np.nanmedian(arr, 0); lo = np.nanmin(arr, 0); hi = np.nanmax(arr, 0)
        ax2.fill_between(step, lo, hi, color=color, alpha=0.15)
        ax2.plot(step, med, color=color, lw=2.4, label=lab)
    ax2.set_ylabel("signal (per-seed min-max norm)"); ax2.set_xlabel("training step")
    ax2.legend(loc="center right", fontsize=8.5)

    for key, color, lab in [("delta_knee_step", "#d62728", "$\\delta$ knee"),
                            ("grok_step", "#1f77b4", "grok"),
                            ("shc_peak_step", "#2ca02c", "SHC peak")]:
        m = agg[key]["median"] if agg.get(key) else None
        if m is not None:
            for ax in (ax1, ax2):
                ax.axvline(m, color=color, ls="--", alpha=0.7, lw=1.4)
            ax1.text(m, 1.05, lab, color=color, fontsize=8, rotation=30, ha="left")

    sl = agg["SHC_lead_over_grok (positive = competitor early-warning)"]
    gl = agg["geometry_lead_over_grok (<=0 = geometry lags)"]
    fig.text(0.5, 0.005,
             f"median SHC lead over grok = {sl['median']:.0f} steps [{sl['min']}, {sl['max']}]  |  "
             f"median geometry lead over grok = {gl['median']:.0f} steps [{gl['min']}, {gl['max']}]  "
             f"(<=0 = geometry lags)", ha="center", fontsize=9, style="italic")
    fig.tight_layout(rect=[0, 0.03, 1, 1])
    out = os.path.join(ART, "grokking_ensemble_frac030.png")
    fig.savefig(out, dpi=130)
    print("\nwrote", os.path.abspath(out))


if __name__ == "__main__":
    main()
