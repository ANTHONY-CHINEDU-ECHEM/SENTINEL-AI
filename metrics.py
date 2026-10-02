"""Evaluation metrics, all aware of deployment prevalence.

The corpus is enriched with theft so the model has something to learn from. In
a real store theft is rare, and precision collapses if you forget that. Every
headline number here is therefore computed with sample weights that restore a
realistic base rate (see :func:`shelfsentinel.models.policy.prevalence_weights`).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, confusion_matrix, roc_auc_score

from ..config import EconomicsConfig
from ..schema import N_CLASSES, SCENARIO_TITLES, Scenario


def expected_calibration_error(p: np.ndarray, y: np.ndarray, w: np.ndarray | None = None, bins: int = 15) -> float:
    """Weighted expected calibration error of a binary probability."""
    w = np.ones_like(p) if w is None else w
    edges = np.linspace(0.0, 1.0, bins + 1)
    idx = np.clip(np.digitize(p, edges[1:-1]), 0, bins - 1)
    ece = 0.0
    for b in range(bins):
        m = idx == b
        if m.any():
            wb = w[m].sum()
            ece += wb * abs(np.average(p[m], weights=w[m]) - np.average(y[m], weights=w[m]))
    return float(ece / w.sum())


def calibration_curve(p: np.ndarray, y: np.ndarray, w: np.ndarray | None = None, bins: int = 12) -> pd.DataFrame:
    w = np.ones_like(p) if w is None else w
    edges = np.linspace(0.0, 1.0, bins + 1)
    idx = np.clip(np.digitize(p, edges[1:-1]), 0, bins - 1)
    rows = []
    for b in range(bins):
        m = idx == b
        if m.sum() >= 20:
            rows.append({"predicted": np.average(p[m], weights=w[m]), "observed": np.average(y[m], weights=w[m]), "n": int(m.sum())})
    return pd.DataFrame(rows)


def multiclass_report(y: np.ndarray, pred: np.ndarray) -> dict:
    """Per class precision, recall and F1 on the enriched test set."""
    cm = confusion_matrix(y, pred, labels=list(range(N_CLASSES)))
    out = {"per_class": {}, "confusion_matrix": cm.tolist()}
    f1s = []
    for k in range(N_CLASSES):
        tp = cm[k, k]
        prec = tp / max(cm[:, k].sum(), 1)
        rec = tp / max(cm[k].sum(), 1)
        f1 = 2 * prec * rec / max(prec + rec, 1e-12)
        f1s.append(f1)
        out["per_class"][SCENARIO_TITLES[Scenario(k)]] = {
            "precision": round(float(prec), 4),
            "recall": round(float(rec), 4),
            "f1": round(float(f1), 4),
            "support": int(cm[k].sum()),
        }
    out["accuracy"] = round(float(np.trace(cm) / cm.sum()), 4)
    out["macro_f1"] = round(float(np.mean(f1s)), 4)
    out["theft_macro_f1"] = round(float(np.mean(f1s[1:])), 4)
    return out


def ranking_metrics(y: np.ndarray, proba: np.ndarray, w: np.ndarray) -> dict:
    """Threshold free quality: theft vs normal and one vs rest per scenario."""
    theft = (y != 0).astype(int)
    risk = 1.0 - proba[:, 0]
    out = {
        "theft_roc_auc": round(float(roc_auc_score(theft, risk)), 4),
        "theft_ap_enriched": round(float(average_precision_score(theft, risk)), 4),
        "theft_ap_deployment": round(float(average_precision_score(theft, risk, sample_weight=w)), 4),
        "per_class_ap_deployment": {},
    }
    for k in range(1, N_CLASSES):
        ap = average_precision_score((y == k).astype(int), proba[:, k], sample_weight=w)
        out["per_class_ap_deployment"][SCENARIO_TITLES[Scenario(k)]] = round(float(ap), 4)
    return out


def alert_report(y: np.ndarray, alert: np.ndarray, flag_class: np.ndarray, w: np.ndarray) -> dict:
    """Precision and recall of an alerting policy at deployment prevalence."""
    theft = y != 0
    total = w.sum()
    n_alert = (w * alert).sum()
    tp = (w * (alert & theft)).sum()
    out = {
        "alerts_per_10k_clips": round(float(n_alert / total * 10_000), 1),
        "precision": round(float(tp / n_alert), 4) if n_alert > 0 else None,
        "recall": round(float(tp / max((w * theft).sum(), 1e-12)), 4),
        "false_alerts_per_10k_normal": round(float((w * (alert & ~theft)).sum() / max((w * ~theft).sum(), 1e-12) * 10_000), 2),
        "per_class": {},
    }
    for k in range(1, N_CLASSES):
        is_k = y == k
        flagged_k = alert & (flag_class == k)
        prec = (w * (flagged_k & theft)).sum() / max((w * flagged_k).sum(), 1e-12)
        out["per_class"][SCENARIO_TITLES[Scenario(k)]] = {
            "recall": round(float((alert & is_k).sum() / max(is_k.sum(), 1)), 4),
            "precision": round(float(prec), 4) if flagged_k.any() else None,
            "support": int(is_k.sum()),
        }
    return out


def false_alert_sources(subtype: pd.Series, y: np.ndarray, alert: np.ndarray) -> pd.DataFrame:
    """Which benign behaviours trigger alerts, as a rate per behaviour."""
    df = pd.DataFrame({"subtype": subtype.to_numpy(), "alert": alert, "normal": y == 0})
    df = df[df["normal"]]
    out = df.groupby("subtype")["alert"].agg(clips="size", false_alerts="sum")
    out["rate_per_1000"] = out["false_alerts"] / out["clips"] * 1000
    return out.sort_values("rate_per_1000", ascending=False).reset_index()


def slice_report(groups: pd.Series, y: np.ndarray, alert: np.ndarray) -> pd.DataFrame:
    """Theft recall and false alert rate inside each slice of the test set."""
    df = pd.DataFrame({"slice": groups.to_numpy(), "theft": y != 0, "alert": alert})
    rows = []
    for name, g in df.groupby("slice"):
        rows.append(
            {
                "slice": str(name),
                "clips": len(g),
                "theft_recall": float(g.loc[g["theft"], "alert"].mean()) if g["theft"].any() else np.nan,
                "false_alerts_per_1000_normal": float(g.loc[~g["theft"], "alert"].mean() * 1000),
            }
        )
    return pd.DataFrame(rows)


def economics(
    y: np.ndarray, alert: np.ndarray, w: np.ndarray, value: np.ndarray, cfg: EconomicsConfig, per: int = 100_000
) -> dict:
    """Pounds per ``per`` track clips at deployment prevalence.

    The assumptions (review cost, recovery rate) are illustrative and live in
    the config so they can be replaced with real figures.
    """
    theft = y != 0
    scale = per / w.sum()
    alerts = (w * alert).sum() * scale
    false_alerts = (w * (alert & ~theft)).sum() * scale
    value_total = (w * value * theft).sum() * scale
    value_flagged = (w * value * (alert & theft)).sum() * scale
    recovered = cfg.recovery_rate * value_flagged
    cost = alerts * cfg.review_cost_gbp + false_alerts * cfg.false_alert_friction_gbp
    return {
        "alerts": round(float(alerts), 1),
        "false_alerts": round(float(false_alerts), 1),
        "value_at_risk_gbp": round(float(value_total), 0),
        "value_flagged_gbp": round(float(value_flagged), 0),
        "share_of_value_flagged": round(float(value_flagged / max(value_total, 1e-9)), 4),
        "recovered_gbp": round(float(recovered), 0),
        "review_cost_gbp": round(float(cost), 0),
        "net_value_gbp": round(float(recovered - cost), 0),
        "alerts_per_store_day": round(float(alerts / per * cfg.clips_per_store_day), 1),
        "net_value_per_store_year_gbp": round(float((recovered - cost) / per * cfg.clips_per_store_day * 364), 0),
    }


def economics_curve(
    y: np.ndarray, risk: np.ndarray, w: np.ndarray, value: np.ndarray, cfg: EconomicsConfig
) -> pd.DataFrame:
    """Sweep one global risk threshold and report workload and net value."""
    rows = []
    for thr in np.r_[np.linspace(0.02, 0.9, 45), np.linspace(0.91, 0.995, 18)]:
        e = economics(y, risk >= thr, w, value, cfg)
        rows.append({"threshold": float(thr), **e})
    return pd.DataFrame(rows)


def precision_vs_prevalence(y: np.ndarray, alert: np.ndarray, prevalences: np.ndarray) -> pd.DataFrame:
    """How alert precision moves as theft gets rarer or more common."""
    theft = y != 0
    tpr = alert[theft].mean()
    fpr = alert[~theft].mean()
    rows = [{"prevalence": float(p), "precision": float(tpr * p / max(tpr * p + fpr * (1 - p), 1e-12))} for p in prevalences]
    return pd.DataFrame(rows)
