import numpy as np
import pandas as pd

from shelfsentinel.features import FEATURE_GROUPS, FEATURE_NAMES, extract_features, frame_primitives
from shelfsentinel.features.catalog import FEATURE_LABELS, feature_dictionary_markdown
from shelfsentinel.features.extractor import _dilate, _falling, _fill, _majority3, _rising
from shelfsentinel.simulation import generate_batch
from shelfsentinel.simulation.generator import build_table


def test_helpers():
    b = np.array([[0, 0, 1, 0, 0, 0, 0]], dtype=bool)
    assert _dilate(b, 1, 2)[0].tolist() == [True, True, True, True, False, False, False]  # window is t-1 .. t+2
    assert _rising(b)[0].tolist() == [False, False, True, False, False, False, False]
    assert _falling(b)[0].tolist() == [False, False, False, True, False, False, False]
    assert _majority3(np.array([[1, 0, 1, 1, 0, 0]], dtype=bool))[0].tolist() == [True, True, True, True, False, False]
    x = np.arange(5, dtype=np.float32).reshape(1, 5, 1)
    valid = np.array([[False, True, False, False, True]])
    assert _fill(x, valid)[0, :, 0].tolist() == [1.0, 1.0, 1.0, 1.0, 4.0]
    assert np.isnan(_fill(x, np.zeros((1, 5), dtype=bool))).all()


def test_feature_table_shape_and_cleanliness(batch, table):
    feats = extract_features(batch)
    assert list(feats.columns) == list(FEATURE_NAMES)
    assert len(feats) == len(batch)
    assert np.isfinite(feats.to_numpy()).all()
    assert sum(len(v) for v in FEATURE_GROUPS.values()) == len(FEATURE_NAMES) == len(set(FEATURE_NAMES))
    assert not table.columns.duplicated().any()


def test_catalog_describes_every_feature():
    assert set(FEATURE_LABELS) == set(FEATURE_NAMES)
    doc = feature_dictionary_markdown()
    assert all(f"`{name}`" in doc for name in FEATURE_NAMES)


def test_prefixes_work_down_to_a_few_frames(batch):
    for n in (3, 8, 31):
        feats = extract_features(batch.prefix(n))
        assert np.isfinite(feats.to_numpy()).all()


def _mean_by_subtype(subtypes, n=240, seed=5):
    batch = generate_batch(n, seed=seed, subtypes=subtypes, quality="hd", crowding=0)
    return build_table(batch.meta, extract_features(batch)).groupby("subtype").mean(numeric_only=True)


def test_aisle_features_separate_theft_from_lookalikes():
    m = _mean_by_subtype(["concealment_bag", "scan_and_go", "pick_to_basket", "concealment_clothing", "shelf_sweep", "bulk_buyer"])
    assert m.loc["concealment_bag", "vanish_unexplained_near_bag"] > 5 * m.loc["scan_and_go", "vanish_unexplained_near_bag"]
    assert m.loc["scan_and_go", "scan_before_bag"] > 0.5
    assert m.loc["concealment_bag", "unaccounted_items"] > m.loc["pick_to_basket", "unaccounted_items"] + 0.5
    assert m.loc["concealment_clothing", "vanish_unexplained_near_waist"] > 0.4
    assert m.loc["shelf_sweep", "unaccounted_items"] > m.loc["bulk_buyer", "unaccounted_items"] + 2
    assert m.loc["shelf_sweep", "min_pick_interval_s"] < 1.5


def test_checkout_features_reconcile_scans():
    m = _mean_by_subtype(["sco_normal", "skip_scan", "ticket_switch", "sco_produce", "till_normal", "sweethearting"])
    assert m.loc["skip_scan", "unscanned_bagging"] > m.loc["sco_normal", "unscanned_bagging"] + 0.8
    assert m.loc["sweethearting", "unscanned_bagging"] > m.loc["till_normal", "unscanned_bagging"] + 1.5
    assert m.loc["ticket_switch", "suspect_scans"] > 0.8
    assert m.loc["sco_produce", "suspect_scans"] < 0.1
    assert m.loc["sco_normal", "transfers"] > 3


def test_exit_features():
    m = _mean_by_subtype(["exit_paid_till", "push_out"])
    assert m.loc["exit_paid_till", "txn_linked"] > 0.8
    assert m.loc["push_out", "txn_linked"] < 0.15
    assert m.loc["push_out", "speed_max"] > m.loc["exit_paid_till", "speed_max"]


def test_frame_primitives_match_feature_counts(batch):
    prim = frame_primitives(batch)
    feats = extract_features(batch)
    total = prim.vanish[0].sum(axis=1) + prim.vanish[1].sum(axis=1)
    pd.testing.assert_series_equal(pd.Series(total.astype(np.float32)), feats["vanish_total"], check_names=False)
