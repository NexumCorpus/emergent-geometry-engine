"""Track-B head-to-head (Constitution Phase 4): does the geometry signal deflect
before test accuracy / before the 2RDM spectral-heat-capacity signal?

Trains a 1-layer transformer on modular addition until it groks, and at log-spaced
checkpoints computes — on ALIGNED x-axes — (1) the geometric order parameter (mean
Forman curvature + Gromov delta + spectral dim of the W_E cosine graph), (2) the
competitor 2RDM test-block SHC, (3) gradient norm, and (4) train/test accuracy.

Saves artifacts/grokking_results.npz and prints the lead-time table. Honest by
construction: it reports whatever the data shows, including a null.
"""
from __future__ import annotations

import os
import json
from collections import deque

import numpy as np
import torch
import torch.nn.functional as F

from experiments.grokking.model import OneLayerTransformer, make_modular_addition_data
from analysis.signals import spectral_heat_capacity, embedding_graph_signal

# --- frozen config (overridable via env for the regime sweep) ---
P = int(os.environ.get("GROK_P", "59"))
FRAC_TRAIN = float(os.environ.get("GROK_FRAC", "0.5"))
NUM_STEPS = int(os.environ.get("GROK_STEPS", "15000"))
TAG = os.environ.get("GROK_TAG", f"frac{int(round(FRAC_TRAIN * 100)):03d}")
LOG_EVERY = 50
SHC_WINDOW = 20
N_PROBE = 150          # per block (train / test)
KNN_K = 6
SEED = int(os.environ.get("GROK_SEED", "0"))
ART = os.path.join(os.path.dirname(__file__), "..", "..", "artifacts")


def _loss_acc(model, X, Y):
    logits = model(X)[:, -1, :]
    loss = F.cross_entropy(logits, Y)
    acc = (logits.argmax(-1) == Y).float().mean()
    return loss, acc


def _grad_norm(model):
    sq = 0.0
    for p in model.parameters():
        if p.grad is not None:
            sq += float(p.grad.detach().pow(2).sum())
    return sq ** 0.5


def run():
    torch.set_num_threads(int(os.environ.get("GROK_THREADS", str(max(1, os.cpu_count() or 2)))))
    os.makedirs(ART, exist_ok=True)

    X, Y = make_modular_addition_data(P)
    g = torch.Generator().manual_seed(SEED)
    perm = torch.randperm(P * P, generator=g)
    n_tr = int(FRAC_TRAIN * P * P)
    tr, te = perm[:n_tr], perm[n_tr:]
    Xtr, Ytr, Xte, Yte = X[tr], Y[tr], X[te], Y[te]

    # fixed probe sets for the 2RDM (deterministic subsamples of train / test).
    p_tr = tr[:N_PROBE]
    p_te = te[:N_PROBE]
    Xp_tr, Yp_tr = X[p_tr], Y[p_tr]
    Xp_te, Yp_te = X[p_te], Y[p_te]

    model = OneLayerTransformer(d_vocab=P + 1, d_vocab_out=P, seed=SEED)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1.0, betas=(0.9, 0.98))

    win_te = deque(maxlen=SHC_WINDOW)
    win_tr = deque(maxlen=SHC_WINDOW)
    rec = {k: [] for k in ("step", "train_acc", "test_acc", "train_loss", "test_loss",
                            "grad_norm", "shc_test", "shc_train",
                            "mean_forman", "var_forman", "gromov_delta", "spectral_dim")}

    for step in range(NUM_STEPS + 1):
        model.train()
        opt.zero_grad()
        loss, _ = _loss_acc(model, Xtr, Ytr)
        loss.backward()
        gnorm = _grad_norm(model)
        opt.step()

        if step % LOG_EVERY != 0:
            continue

        model.eval()
        with torch.no_grad():
            ltr, atr = _loss_acc(model, Xtr, Ytr)
            lte, ate = _loss_acc(model, Xte, Yte)
            # per-sample probe losses -> 2RDM window
            pl_te = F.cross_entropy(model(Xp_te)[:, -1, :], Yp_te, reduction="none").numpy()
            pl_tr = F.cross_entropy(model(Xp_tr)[:, -1, :], Yp_tr, reduction="none").numpy()
            win_te.append(pl_te)
            win_tr.append(pl_tr)
            shc_te = spectral_heat_capacity(np.array(win_te)) if len(win_te) == SHC_WINDOW else np.nan
            shc_tr = spectral_heat_capacity(np.array(win_tr)) if len(win_tr) == SHC_WINDOW else np.nan
            # geometric signal on the p number-token embeddings (exclude '=' token)
            W_E_num = model.W_E.detach().numpy()[:P]
            geo = embedding_graph_signal(W_E_num, k=KNN_K)

        rec["step"].append(step)
        rec["train_acc"].append(float(atr)); rec["test_acc"].append(float(ate))
        rec["train_loss"].append(float(ltr)); rec["test_loss"].append(float(lte))
        rec["grad_norm"].append(gnorm)
        rec["shc_test"].append(float(shc_te)); rec["shc_train"].append(float(shc_tr))
        rec["mean_forman"].append(geo["mean_forman"]); rec["var_forman"].append(geo["var_forman"])
        rec["gromov_delta"].append(geo["gromov_delta"]); rec["spectral_dim"].append(geo["spectral_dim"])

        if step % 500 == 0:
            print(f"step {step:6d}  tr_acc={float(atr):.3f}  te_acc={float(ate):.3f}  "
                  f"mean_forman={geo['mean_forman']:+.2f}  shc_te={shc_te:.3e}  gnorm={gnorm:.2f}",
                  flush=True)

    out = {k: np.array(v, dtype=float) for k, v in rec.items()}
    np.savez(os.path.join(ART, f"grokking_results_{TAG}.npz"), **out)
    summary = _lead_time_summary(out)
    summary["config"] = {"P": P, "frac_train": FRAC_TRAIN, "num_steps": NUM_STEPS, "tag": TAG}
    with open(os.path.join(ART, f"grokking_summary_{TAG}.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print("\n=== LEAD-TIME SUMMARY ===")
    print(json.dumps(summary, indent=2))
    return out, summary


def _first_cross(steps, values, thresh, rising=True):
    for s, v in zip(steps, values):
        if np.isnan(v):
            continue
        if (rising and v >= thresh) or (not rising and v <= thresh):
            return int(s)
    return None


def _smooth(v, w=7):
    """Edge-preserving centered moving average (shrinks the window at the boundaries,
    no zero-padding artifacts)."""
    n = len(v)
    if n < w or w < 2:
        return v
    half = w // 2
    return np.array([v[max(0, i - half):min(n, i + half + 1)].mean() for i in range(n)])


def _half_step(steps, values, rising):
    """COMMIT step: the first step after which the (smoothed) signal stays past the
    halfway level between its initial and final plateau for the entire remainder of
    training — i.e. where it DURABLY reorganizes, not where a transient noise dip first
    grazes the threshold. This is the honest 'knee': the naive first-crossing falsely
    reported an 11k-step 'lead' on a delta that bounces around its plateau and only
    settles post-grokking. Returns None if the signal does not move that direction."""
    v = np.asarray(values, dtype=float)
    m = ~np.isnan(v)
    if m.sum() < 6:
        return None
    vv = _smooth(v[m], min(7, len(v[m]) // 2))
    ss = np.asarray(steps)[m]
    edge = max(2, len(vv) // 10)
    init = np.median(vv[:edge]); final = np.median(vv[-edge:])
    if rising and final <= init:
        return None
    if (not rising) and final >= init:
        return None
    target = init + 0.5 * (final - init)
    past = (vv >= target) if rising else (vv <= target)
    n = len(vv)
    stays = np.cumsum(past[::-1])[::-1] == np.arange(n, 0, -1)  # True where all-past from here
    idx = np.where(stays)[0]
    return int(ss[idx[0]]) if len(idx) else None


def _moves(values, min_range):
    v = np.asarray(values, dtype=float)
    m = ~np.isnan(v)
    return bool(m.sum() > 2 and (np.nanmax(v[m]) - np.nanmin(v[m])) >= min_range)


def _lead_time_summary(out):
    steps = out["step"]
    grok = _first_cross(steps, out["test_acc"], 0.9, rising=True)
    memorize = _first_cross(steps, out["train_acc"], 0.99, rising=True)
    shc_peak = int(steps[int(np.nanargmax(out["shc_test"]))]) if np.any(~np.isnan(out["shc_test"])) else None
    gnorm_peak = int(steps[int(np.nanargmax(out["grad_norm"]))])

    # Geometric order parameter = Gromov delta (declining = graph becoming ring-like);
    # curvature variance is a secondary declining channel. mean Forman is reported
    # transparently as the (pre-registered) channel that does NOT deflect.
    delta_knee = _half_step(steps, out["gromov_delta"], rising=False)
    varf_knee = _half_step(steps, out["var_forman"], rising=False)
    return {
        "memorization_step_train_acc>0.99": memorize,
        "grok_step_test_acc>0.9": grok,
        "shc_test_peak_step": shc_peak,
        "gradnorm_peak_step": gnorm_peak,
        "geometric_order_param": "gromov_delta (falling = less tree-like / more ring-like)",
        "delta_knee_step": delta_knee,
        "var_forman_knee_step": varf_knee,
        "mean_forman_deflects": _moves(out["mean_forman"], 2.0),
        "delta_lead_over_grok": (grok - delta_knee) if (grok and delta_knee) else None,
        "delta_lead_over_shc_peak": (shc_peak - delta_knee) if (shc_peak and delta_knee) else None,
        "note": "positive lead = geometric knee occurs BEFORE the comparison event. 'earlier' is EMPIRICAL — this is the test, not an assumption. mean_forman is degree-dominated on a kNN graph and does not deflect (reported honestly).",
    }


if __name__ == "__main__":
    run()
