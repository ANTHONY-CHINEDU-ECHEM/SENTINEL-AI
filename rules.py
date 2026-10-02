"""Transparent rule baseline.

This is the kind of logic a loss prevention team would write by hand: a short
decision list over reconciliation features. It is easy to audit and it sets the
bar the learned model has to clear. It is also kept as a fallback detector.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from ..schema import Scenario


@dataclass(frozen=True)
class RuleThresholds:
    sweep_min_picks: int = 5
    sweep_min_unaccounted: int = 3
    conceal_min_unaccounted: int = 1
    sco_min_unscanned: int = 1
    till_min_unscanned: int = 2
    exit_max_dwell_s: float = 20.0
    exit_min_items: int = 3


class RuleBaseline:
    """Decision list over interpretable features. No training required."""

    def __init__(self, thresholds: RuleThresholds | None = None) -> None:
        self.th = thresholds or RuleThresholds()

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        th = self.th
        pred = np.full(len(X), int(Scenario.NORMAL), dtype=int)
        col = lambda name: X[name].to_numpy()  # noqa: E731

        aisle = col("scene_aisle") > 0.5
        sco = col("scene_self_checkout") > 0.5
        till = col("scene_staffed_till") > 0.5
        exit_ = col("scene_exit") > 0.5

        # Aisle: later assignments take priority, so the most specific rule goes last.
        unaccounted = col("unaccounted_items")
        clothing = (
            aisle
            & (col("vanish_unexplained_near_waist") >= 1)
            & (unaccounted >= th.conceal_min_unaccounted)
            & (col("hand_item_end") < 0.5)
        )
        bag = aisle & (col("vanish_unexplained_near_bag") >= 1) & (unaccounted >= th.conceal_min_unaccounted)
        sweep = aisle & (col("shelf_picks") >= th.sweep_min_picks) & (unaccounted >= th.sweep_min_unaccounted)
        pred[clothing] = int(Scenario.CONCEALMENT_CLOTHING)
        pred[bag] = int(Scenario.CONCEALMENT_BAG)
        pred[sweep] = int(Scenario.SHELF_SWEEP)

        # Self checkout.
        skip = sco & (
            (col("unscanned_bagging") >= th.sco_min_unscanned)
            | (col("bypass_transfers") >= 1)
            | ((col("vanish_unexplained_near_bag") >= 1) & (col("flow_gap") >= 1))
        )
        switch = sco & (col("suspect_scans") >= 1)
        pred[skip] = int(Scenario.SKIP_SCAN)
        pred[switch] = int(Scenario.TICKET_SWITCH)

        # Staffed till.
        pred[till & (col("unscanned_bagging") >= th.till_min_unscanned)] = int(Scenario.SWEETHEARTING)

        # Exit.
        carrying = (col("container_max") >= th.exit_min_items) | (col("item_frames_frac") > 0.5)
        push = exit_ & (col("txn_linked") < 0.5) & (col("checkout_dwell_s") < th.exit_max_dwell_s) & carrying
        pred[push] = int(Scenario.PUSH_OUT)
        return pred
