"""Alert policy: turning calibrated probabilities into review tiers.

A loss prevention alert is an interruption for a colleague and, if handled
badly, an insult to a customer. So the policy is precision first: for every
scenario it finds the lowest threshold that still meets a precision target
*at the theft prevalence expected in production*, not at the enriched
prevalence of the training corpus.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field

import numpy as np
import pandas as pd

from ..config import PolicyConfig
from ..schema import N_CLASSES, SCENARIO_TITLES, THEFT_CLASSES, Scenario

TIERS = ("clear", "review", "priority")


def prevalence_weights(y: np.ndarray, theft_prevalence: float) -> np.ndarray:
    """Sample weights that make theft clips ``theft_prevalence`` of the total.

    Theft clips keep weight one and their relative mix; normal clips are
    weighted up. All weighted metrics in the project use these weights.
    """
    y = np.asarray(y)
    theft = y != int(Scenario.NORMAL)
    n_theft, n_normal = int(theft.sum()), int((~theft).sum())
    w = np.ones(len(y), dtype=np.float64)
    if n_theft and n_normal:
        w[~theft] = (n_theft / n_normal) * (1.0 - theft_prevalence) / theft_prevalence
    return w


def top_theft(proba: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Most likely theft scenario per clip and its probability."""
    theft_cols = [int(c) for c in THEFT_CLASSES]
    idx = np.argmax(proba[:, theft_cols], axis=1)
    cls = np.asarray(theft_cols)[idx]
    return cls, proba[np.arange(len(proba)), cls]


@dataclass
class AlertPolicy:
    review: dict[int, float] = field(default_factory=dict)
    priority: dict[int, float] = field(default_factory=dict)
    info: dict = field(default_factory=dict)

    def decide(self, proba: np.ndarray) -> pd.DataFrame:
        """Return the flagged scenario, its risk and the review tier for every clip."""
        cls, risk = top_theft(proba)
        t_review = np.array([self.review.get(int(c), 1.01) for c in cls])
        t_priority = np.array([self.priority.get(int(c), 1.01) for c in cls])
        tier = np.where(risk >= t_priority, 2, np.where(risk >= t_review, 1, 0))
        return pd.DataFrame(
            {
                "flag_class": cls,
                "risk": risk,
                "tier": np.asarray(TIERS)[tier],
                "tier_id": tier,
            }
        )

    def to_dict(self) -> dict:
        return {
            "review": {SCENARIO_TITLES[Scenario(k)]: round(v, 4) for k, v in self.review.items()},
            "priority": {SCENARIO_TITLES[Scenario(k)]: round(v, 4) for k, v in self.priority.items()},
            "info": self.info,
        }

    def as_state(self) -> dict:
        return asdict(self)


def _threshold_for_precision(
    risk: np.ndarray, correct: np.ndarray, w: np.ndarray, target: float, floor: float, phantom: float = 0.0
) -> tuple[float, float]:
    """Lowest threshold whose weighted precision meets ``target``.

    ``phantom`` is extra false alert weight added to every candidate threshold.
    At a realistic theft rate one honest clip outweighs dozens of theft clips,
    so a threshold can rest on a handful of validation errors. Requiring the
    target to hold even if one more honest clip had been flagged keeps the
    policy away from thresholds that are supported by too little evidence.

    Returns ``(threshold, achieved_precision)``. If no threshold reaches the
    target the scenario is effectively switched off (threshold above one).
    """
    order = np.argsort(-risk)
    r, c, ww = risk[order], correct[order], w[order]
    tp = np.cumsum(ww * c)
    total = np.cumsum(ww)
    precision = tp / np.maximum(total, 1e-12)
    guarded = tp / np.maximum(total + phantom, 1e-12)
    # Only evaluate at the last element of each run of equal scores.
    last = np.r_[r[1:] != r[:-1], True]
    ok = last & (guarded >= target) & (r >= floor)
    if not ok.any():
        return 1.01, float("nan")
    i = np.flatnonzero(ok)[-1]
    return float(r[i]), float(precision[i])


def fit_policy(proba: np.ndarray, y: np.ndarray, cfg: PolicyConfig) -> AlertPolicy:
    """Choose per scenario thresholds on validation data."""
    y = np.asarray(y)
    w = prevalence_weights(y, cfg.deployment_theft_prevalence)
    cls, risk = top_theft(proba)
    policy = AlertPolicy(info={"deployment_theft_prevalence": cfg.deployment_theft_prevalence})
    normal_w = w[y == int(Scenario.NORMAL)]
    phantom = cfg.phantom_false_alerts * (float(normal_w[0]) if len(normal_w) else 0.0)
    achieved = {}
    for c in range(1, N_CLASSES):
        m = cls == c
        # An alert is useful when the clip really is theft, even if the scenario label is off.
        correct = (y[m] != int(Scenario.NORMAL)).astype(float)
        t_rev, p_rev = _threshold_for_precision(risk[m], correct, w[m], cfg.review_precision_target, cfg.min_threshold, phantom)
        # The margin is applied to the review tier only. At a realistic theft rate one honest
        # clip outweighs dozens of theft clips, so a 95 percent target plus a margin cannot be
        # certified on a validation set of this size and would switch the priority tier off.
        t_pri, p_pri = _threshold_for_precision(risk[m], correct, w[m], cfg.priority_precision_target, cfg.min_threshold)
        policy.review[c] = t_rev
        policy.priority[c] = max(t_pri, t_rev)
        achieved[SCENARIO_TITLES[Scenario(c)]] = {"review_precision": p_rev, "priority_precision": p_pri}
    policy.info["validation_precision"] = achieved
    policy.info["targets"] = {"review": cfg.review_precision_target, "priority": cfg.priority_precision_target}
    policy.info["phantom_false_alerts"] = cfg.phantom_false_alerts
    return policy
