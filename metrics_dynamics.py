"""Survival metrics emphasising the v2 differentiators: calibration of S(t)."""
from __future__ import annotations
import numpy as np


def harrell_c(risk, time, event):
    risk = np.asarray(risk); time = np.asarray(time); event = np.asarray(event)
    n = len(time); num = den = 0.0
    for i in range(n):
        if event[i] == 0:
            continue
        m = time > time[i]
        den += m.sum()
        num += (risk[i] > risk[m]).sum() + 0.5 * (risk[i] == risk[m]).sum()
    return float(num / den) if den else 0.5


def restricted_mean_survival(S_curve, grid):
    return np.trapz(S_curve, grid, axis=-1)


def integrated_brier_score(S_curve, grid, time, event):
    """IBS with IPCW-free (simple) Brier; lower is better. S_curve:(N,len(grid))."""
    time = np.asarray(time); event = np.asarray(event)
    bs = np.zeros(len(grid))
    for k, t in enumerate(grid):
        y = (time > t).astype(float)                     # 1 if still event-free at t
        pred = S_curve[:, k]
        mask = ~((time <= t) & (event == 0))             # drop censored-before-t
        if mask.sum():
            bs[k] = np.mean((y[mask] - pred[mask]) ** 2)
    return float(np.trapz(bs, grid) / (grid[-1] - grid[0]))


def d_calibration(S_at_event, n_bins=10):
    """Distribution calibration: S(T_i) for events should be ~Uniform(0,1).
    Returns chi-square statistic over bins (lower = better calibrated)."""
    S_at_event = np.asarray(S_at_event)
    counts, _ = np.histogram(S_at_event, bins=n_bins, range=(0, 1))
    expected = len(S_at_event) / n_bins
    return float(((counts - expected) ** 2 / max(expected, 1e-9)).sum())


# --------------------------------------------------------------------------- #
# Censoring-correct threshold analysis (the "false positive" tools).
# Added for H1_RUNNER_SPEC.md §5: only meaningful once you threshold risk.
# --------------------------------------------------------------------------- #
def _km(time, event):
    """Kaplan-Meier S(t) at unique times. Returns (times_sorted, surv_step)."""
    time = np.asarray(time, float); event = np.asarray(event, float)
    uniq = np.unique(time); surv = 1.0; out = []
    for ut in uniq:
        at_risk = (time >= ut).sum()
        d = ((time == ut) & (event == 1)).sum()
        if at_risk > 0:
            surv *= (1.0 - d / at_risk)
        out.append(surv)
    return uniq, np.asarray(out)


def _step(xs, ys, q):
    if len(xs) == 0:
        return 1.0
    idx = np.searchsorted(xs, q, side="right") - 1
    return float(ys[np.clip(idx, 0, len(ys) - 1)]) if idx >= 0 else 1.0


def uno_tdauc(risk, time, event, horizons):
    """Uno-type IPCW cumulative/dynamic time-dependent AUC at each horizon.

    At horizon t: cases = {T<=t, event=1} weighted by 1/G(T) (G = KM of the
    CENSORING distribution); controls = {T>t}. This is the censoring-correct way
    to read a 'false-positive rate' for a survival model at a fixed horizon.
    Returns {t: auc or None}.
    """
    risk = np.asarray(risk, float); time = np.asarray(time, float); event = np.asarray(event, float)
    gt, gs = _km(time, 1.0 - event)                      # censoring survival G
    Gw = np.array([max(_step(gt, gs, ti), 1e-3) for ti in time])
    out = {}
    for t in np.atleast_1d(horizons):
        cases = np.where((time <= t) & (event == 1))[0]
        ctrls = np.where(time > t)[0]
        if len(cases) == 0 or len(ctrls) == 0:
            out[float(t)] = None; continue
        num = den = 0.0
        for i in cases:
            wi = 1.0 / Gw[i]
            comp = risk[i] > risk[ctrls]
            tie = risk[i] == risk[ctrls]
            num += wi * (comp.sum() + 0.5 * tie.sum()); den += wi * len(ctrls)
        out[float(t)] = float(num / den) if den > 0 else None
    return out


def decision_curve(pred_event_prob, time, event, horizon, thresholds=None):
    """Vickers net-benefit decision curve for survival (censoring-adjusted via KM).

    `pred_event_prob` = predicted P(event by `horizon`) = 1 - S(horizon|x). For each
    threshold pt, 'treat' = pred>=pt; event rate among treated estimated by KM at the
    horizon. Net benefit weighs true- vs false-positive treatment by pt/(1-pt) -- the
    clinically honest way to trade off over- vs under-treatment. Returns curves for
    the model, treat-all, and treat-none (0).
    """
    p = np.asarray(pred_event_prob, float); time = np.asarray(time, float); event = np.asarray(event, float)
    if thresholds is None:
        thresholds = np.linspace(0.01, 0.6, 30)
    ut, us = _km(time, event); ev_all = 1.0 - _step(ut, us, horizon)
    nb_model, nb_all = [], []
    for pt in thresholds:
        treat = p >= pt
        if treat.sum() == 0:
            nb_model.append(0.0)
        else:
            ut2, us2 = _km(time[treat], event[treat])
            ev_t = 1.0 - _step(ut2, us2, horizon); ntreat = treat.mean()
            nb_model.append(ev_t * ntreat - (1 - ev_t) * ntreat * (pt / (1 - pt)))
        nb_all.append(ev_all - (1 - ev_all) * (pt / (1 - pt)))
    return {"thresholds": [float(x) for x in thresholds],
            "net_benefit_model": [float(x) for x in nb_model],
            "net_benefit_treat_all": [float(x) for x in nb_all],
            "net_benefit_treat_none": [0.0] * len(thresholds)}