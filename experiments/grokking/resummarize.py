"""Recompute lead-time summaries from saved npz runs (after fixing the knee detector)."""
import os
import json

import numpy as np

from experiments.grokking.run import _lead_time_summary

ART = os.path.join(os.path.dirname(__file__), "..", "..", "artifacts")
CONFIGS = {"frac050": dict(P=59, frac_train=0.5, num_steps=15000),
           "frac030": dict(P=59, frac_train=0.3, num_steps=25000)}

for tag, cfg in CONFIGS.items():
    path = os.path.join(ART, f"grokking_results_{tag}.npz")
    if not os.path.exists(path):
        print("skip", tag)
        continue
    d = dict(np.load(path))
    summ = _lead_time_summary(d)
    summ["config"] = {**cfg, "tag": tag}
    json.dump(summ, open(os.path.join(ART, f"grokking_summary_{tag}.json"), "w"), indent=2)
    print(f"=== {tag} ===")
    for k in ("grok_step_test_acc>0.9", "shc_test_peak_step", "delta_knee_step",
              "var_forman_knee_step", "mean_forman_deflects",
              "delta_lead_over_grok", "delta_lead_over_shc_peak"):
        print(f"  {k}: {summ.get(k)}")
