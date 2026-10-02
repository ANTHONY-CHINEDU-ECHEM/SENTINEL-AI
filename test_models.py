import numpy as np
import pandas as pd

from shelfsentinel.config import PolicyConfig, SplitConfig
from shelfsentinel.features import FEATURE_NAMES
from shelfsentinel.models import AlertPolicy, RuleBaseline, fit_policy, prevalence_weights, split_by_store, top_theft
from shelfsentinel.models.policy import _threshold_for_precision
from shelfsentinel.schema import N_CLASSES, Scenario


def _row(**values) -> pd.DataFrame:
    base = {name: 0.0 for name in FEATURE_NAMES}
    base.update(values)
    return pd.DataFrame([base])


def test_rules_fire_on_textbook_cases():
    rules = RuleBaseline()
    cases = {
        Scenario.NORMAL: _row(scene_aisle=1, shelf_picks=2, container_gain=2),
        Scenario.CONCEALMENT_BAG: _row(scene_aisle=1, shelf_picks=1, unaccounted_items=1, vanish_unexplained_near_bag=1),
        Scenario.CONCEALMENT_CLOTHING: _row(scene_aisle=1, shelf_picks=1, unaccounted_items=1, vanish_unexplained_near_waist=1),
        Scenario.SHELF_SWEEP: _row(scene_aisle=1, shelf_picks=7, unaccounted_items=6, vanish_unexplained_near_bag=3),
        Scenario.SKIP_SCAN: _row(scene_self_checkout=1, unscanned_bagging=2),
        Scenario.TICKET_SWITCH: _row(scene_self_checkout=1, suspect_scans=1),
        Scenario.SWEETHEARTING: _row(scene_staffed_till=1, unscanned_bagging=3),
        Scenario.PUSH_OUT: _row(scene_exit=1, txn_linked=0, checkout_dwell_s=3, container_max=9),
    }
    for expected, row in cases.items():
        assert rules.predict(row)[0] == int(expected), expected.name
    paid = _row(scene_exit=1, txn_linked=1, checkout_dwell_s=90, container_max=9)
    assert rules.predict(paid)[0] == int(Scenario.NORMAL)


def test_prevalence_weights_hit_the_target():
    y = np.array([0] * 900 + [1] * 60 + [7] * 40)
    w = prevalence_weights(y, 0.01)
    assert np.isclose(w[y != 0].sum() / w.sum(), 0.01)
    assert np.all(w[y != 0] == 1.0)


def test_threshold_search_finds_lowest_qualifying_score():
    risk = np.array([0.9, 0.8, 0.7, 0.6, 0.5, 0.4])
    correct = np.array([1, 1, 1, 0, 1, 0], dtype=float)
    w = np.ones(6)
    thr, prec = _threshold_for_precision(risk, correct, w, target=0.8, floor=0.0)
    assert thr == 0.5 and np.isclose(prec, 0.8)
    thr, _ = _threshold_for_precision(risk, np.zeros(6), w, target=0.8, floor=0.0)
    assert thr > 1.0  # scenario is switched off when the target cannot be met


def test_policy_assigns_tiers():
    policy = AlertPolicy(review={c: 0.5 for c in range(1, N_CLASSES)}, priority={c: 0.9 for c in range(1, N_CLASSES)})
    proba = np.full((3, N_CLASSES), 0.0)
    proba[0, 0] = 1.0
    proba[1, [0, 3]] = [0.3, 0.7]
    proba[2, [0, 7]] = [0.05, 0.95]
    out = policy.decide(proba)
    assert out["tier"].tolist() == ["clear", "review", "priority"]
    assert out["flag_class"].tolist()[1:] == [3, 7]
    cls, risk = top_theft(proba)
    assert risk[2] == 0.95 and cls[2] == 7


def test_fit_policy_respects_precision_target():
    rng = np.random.default_rng(0)
    n = 6000
    y = np.where(rng.random(n) < 0.15, 1, 0)
    score = np.clip(np.where(y == 1, rng.normal(0.8, 0.15, n), rng.normal(0.15, 0.15, n)), 0, 1)
    proba = np.zeros((n, N_CLASSES))
    proba[:, 1] = score
    proba[:, 0] = 1 - score
    cfg = PolicyConfig(deployment_theft_prevalence=0.02, review_precision_target=0.8, priority_precision_target=0.95)
    loose = fit_policy(proba, y, PolicyConfig(deployment_theft_prevalence=0.02, phantom_false_alerts=0.0))
    policy = fit_policy(proba, y, cfg)
    assert policy.review[1] >= loose.review[1]  # the safety margin can only raise a threshold
    w = prevalence_weights(y, 0.02)
    alert = policy.decide(proba)["tier_id"].to_numpy() >= 1
    precision = (w * (alert & (y == 1))).sum() / (w * alert).sum()
    assert precision >= 0.8 - 1e-9
    assert policy.priority[1] >= policy.review[1]


def test_store_split_is_disjoint_and_stratified():
    df = pd.DataFrame({"store_id": np.repeat(np.arange(30), 5), "store_format": np.repeat(["a", "b", "c"], 50)})
    split = split_by_store(df, SplitConfig())
    per_store = df.assign(split=split).groupby("store_id")["split"].nunique()
    assert (per_store == 1).all()
    assert set(split.unique()) == {"train", "val", "test"}
    formats = df.assign(split=split).groupby("split")["store_format"].nunique()
    assert (formats == 3).all()
