import json

import numpy as np

from shelfsentinel.evaluation import metrics as M
from shelfsentinel.evaluation.replay import risk_curves
from shelfsentinel.features import extract_features
from shelfsentinel.models.sentinel import SentinelModel
from shelfsentinel.simulation import generate_batch
from shelfsentinel.simulation.generator import load_features


def test_manifest(tiny_project, tiny_cfg):
    m = tiny_project["manifest"]
    assert m["n_clips"] == tiny_cfg.data.n_clips
    assert m["total_frames"] == tiny_cfg.data.n_clips * tiny_cfg.data.n_frames
    table = load_features(tiny_project["root"], tiny_cfg.data.out_dir)
    assert len(table) == m["n_clips"]
    assert table["clip_id"].is_unique


def test_metrics_are_complete_and_sane(tiny_project):
    metrics = tiny_project["metrics"]
    for key in ("dataset", "model", "rules", "logistic", "matched_alert_budget"):
        assert key in metrics
    model = metrics["model"]
    assert model["classification"]["macro_f1"] > 0.75
    assert model["ranking"]["theft_roc_auc"] > 0.95
    assert model["classification"]["macro_f1"] > metrics["rules"]["classification"]["macro_f1"]
    saved = json.loads((tiny_project["root"] / "reports" / "metrics.json").read_text())
    assert saved["model"]["classification"]["macro_f1"] == model["classification"]["macro_f1"]
    for name in ("false_alert_sources", "robustness_slices", "economics_curve", "calibration", "precision_vs_prevalence"):
        assert (tiny_project["root"] / "reports" / "tables" / f"{name}.csv").exists()


def test_model_roundtrip_and_probabilities(tiny_project, batch, tmp_path):
    model = tiny_project["model"]
    feats = extract_features(batch)
    proba = model.predict_proba(feats)
    assert proba.shape == (len(batch), 8)
    np.testing.assert_allclose(proba.sum(axis=1), 1.0, atol=1e-6)
    again = SentinelModel.load(model.save(tmp_path / "copy.joblib"))
    np.testing.assert_allclose(again.predict_proba(feats), proba)
    out = model.assess(feats)
    assert set(out["tier"]) <= {"clear", "review", "priority"}
    assert model.reference  # per scene statistics for explanations


def test_streaming_replay_shapes(tiny_project):
    clips = generate_batch(6, seed=3, subtypes=["concealment_bag", "pick_to_basket"])
    counts, risk, tier = risk_curves(tiny_project["model"], clips, min_frames=16, step=16)
    assert counts.tolist() == [16, 32, 48, 64]
    assert risk.shape == tier.shape == (6, 4)
    assert ((risk >= 0) & (risk <= 1)).all()


def test_calibration_error_and_economics():
    p = np.array([0.1] * 100 + [0.9] * 100)
    y = np.array([0] * 90 + [1] * 10 + [1] * 90 + [0] * 10, dtype=float)
    assert M.expected_calibration_error(p, y) < 1e-9
    assert M.expected_calibration_error(1 - p, y) > 0.7

    from shelfsentinel.config import EconomicsConfig

    y = np.array([0, 0, 0, 1, 1])
    alert = np.array([True, False, False, True, False])
    value = np.array([0, 0, 0, 100.0, 50.0])
    e = M.economics(y, alert, np.ones(5), value, EconomicsConfig(review_cost_gbp=1, recovery_rate=0.5, false_alert_friction_gbp=1), per=5)
    assert e["alerts"] == 2 and e["false_alerts"] == 1
    assert e["value_flagged_gbp"] == 100 and e["recovered_gbp"] == 50
    assert e["review_cost_gbp"] == 3 and e["net_value_gbp"] == 47


def test_precision_falls_with_prevalence():
    y = np.array([1] * 100 + [0] * 900)
    alert = np.array([True] * 90 + [False] * 10 + [True] * 9 + [False] * 891)
    curve = M.precision_vs_prevalence(y, alert, np.array([0.001, 0.01, 0.1]))
    assert curve["precision"].is_monotonic_increasing
    assert curve["precision"].iloc[0] < 0.1 < 0.9 < curve["precision"].iloc[-1]
