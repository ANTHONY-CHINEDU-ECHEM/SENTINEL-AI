"""End to end training and evaluation.

``run_training`` reads the feature shards, splits by store, fits the detector
and both baselines, calibrates, chooses alert thresholds, and writes every
number the README quotes to ``reports/``. Nothing in the write up is typed by
hand: it all comes from the files produced here.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import log_loss

from .config import Config
from .evaluation import metrics as M
from .evaluation.replay import latency_report, streaming_report
from .features.extractor import FEATURE_GROUPS, FEATURE_NAMES
from .models.policy import fit_policy, prevalence_weights, top_theft
from .models.rules import RuleBaseline
from .models.sentinel import SentinelModel, split_by_store, train_gbm, train_logistic
from .schema import CAMERA_QUALITIES, N_CLASSES, SCENARIO_TITLES, SCENE_TITLES, Scenario, Scene
from .simulation.generator import generate_batch, load_features, load_prefix_features, make_stores


def _say(verbose: bool, msg: str) -> None:
    if verbose:
        print(msg, flush=True)


def _dataset_summary(df: pd.DataFrame) -> dict:
    by_split = df.groupby("split").size().to_dict()
    by_class = {SCENARIO_TITLES[Scenario(int(k))]: int(v) for k, v in df["scenario"].value_counts().sort_index().items()}
    by_scene = {SCENE_TITLES[Scene(int(k))]: int(v) for k, v in df["scene"].value_counts().sort_index().items()}
    return {
        "clips": int(len(df)),
        "stores": int(df["store_id"].nunique()),
        "stores_by_split": df.groupby("split")["store_id"].nunique().to_dict(),
        "clips_by_split": {k: int(v) for k, v in by_split.items()},
        "clips_by_scenario": by_class,
        "clips_by_scene": by_scene,
        "benign_subtypes": int(df.loc[df["scenario"] == 0, "subtype"].nunique()),
        "theft_share": round(float((df["scenario"] != 0).mean()), 4),
        "camera_quality_share": {q: round(float((df["camera_quality"] == q).mean()), 3) for q in CAMERA_QUALITIES},
    }


def _permutation_importance(model: SentinelModel, test: pd.DataFrame, seed: int, max_rows: int = 12_000) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Increase in log loss when a feature (or a whole sensor group) is shuffled."""
    rng = np.random.default_rng(seed)
    sample = test.sample(min(max_rows, len(test)), random_state=seed)
    y = sample["scenario"].to_numpy()
    labels = list(range(N_CLASSES))
    base = log_loss(y, model.predict_proba(sample), labels=labels)

    def shuffled_loss(cols: list[str]) -> float:
        perm = sample.copy()
        order = rng.permutation(len(perm))
        perm[cols] = perm[cols].to_numpy()[order]
        return log_loss(y, model.predict_proba(perm), labels=labels) - base

    feature_rows = [{"feature": name, "importance": shuffled_loss([name])} for name in FEATURE_NAMES]
    group_rows = [{"group": g, "importance": shuffled_loss(list(cols))} for g, cols in FEATURE_GROUPS.items()]
    return (
        pd.DataFrame(feature_rows).sort_values("importance", ascending=False).reset_index(drop=True),
        pd.DataFrame(group_rows).sort_values("importance", ascending=False).reset_index(drop=True),
    )


def _ablation(train: pd.DataFrame, test: pd.DataFrame, w: np.ndarray, cfg: Config) -> pd.DataFrame:
    """What does each sensor family buy? Train on growing feature sets."""
    sub = train.sample(min(60_000, len(train)), random_state=cfg.model.seed)
    ctx = FEATURE_GROUPS["context"]
    stages = [
        ("Pose only", FEATURE_GROUPS["pose"] + ctx),
        ("Pose + item flow", FEATURE_GROUPS["pose"] + FEATURE_GROUPS["item_flow"] + ctx),
        ("Pose + item flow + POS", FEATURE_NAMES),
    ]
    y = test["scenario"].to_numpy()
    rows = []
    for name, cols in stages:
        m = train_gbm(sub, cfg.model, feature_names=tuple(cols), max_iter=150)
        proba = m.predict_raw(test)
        rep = M.multiclass_report(y, proba.argmax(axis=1))
        rank = M.ranking_metrics(y, proba, w)
        rows.append(
            {
                "features": name,
                "n_features": len(cols),
                "macro_f1": rep["macro_f1"],
                "theft_ap_deployment": rank["theft_ap_deployment"],
                **{f"recall_{k}": v["recall"] for k, v in rep["per_class"].items() if k != SCENARIO_TITLES[Scenario.NORMAL]},
            }
        )
    return pd.DataFrame(rows)


def run_training(cfg: Config, root: str | Path = ".", verbose: bool = True, light: bool = False) -> dict:
    """Train, calibrate, evaluate and persist. Returns the metrics dictionary.

    ``light=True`` skips the slow extras (ablation, permutation importance and
    streaming replay) and is used by the test suite and the quick demo.
    """
    root = Path(root)
    reports = root / "reports"
    tables = reports / "tables"
    tables.mkdir(parents=True, exist_ok=True)
    started = time.time()

    df = load_features(root, cfg.data.out_dir)
    df["split"] = split_by_store(df, cfg.split)
    train, val, test = (df[df["split"] == s].reset_index(drop=True) for s in ("train", "val", "test"))
    _say(verbose, f"clips  train {len(train):,}  val {len(val):,}  test {len(test):,}")

    # --------------------------- main model --------------------------- #
    # Partial clips from the training stores teach the detector what an unfinished
    # action looks like, which is what makes it safe to score clips as they unfold.
    prefix = load_prefix_features(root, cfg.data.out_dir)
    fit_rows = train
    if prefix is not None:
        train_stores = set(train["store_id"])
        prefix_train = prefix[prefix["store_id"].isin(train_stores)]
        fit_rows = pd.concat([train, prefix_train], ignore_index=True)
        _say(verbose, f"streaming aware training: {len(prefix_train):,} partial clip rows added")
    model = train_gbm(fit_rows, cfg.model)
    _say(verbose, f"gradient boosting fitted in {model.info['n_iter']} iterations ({time.time() - started:.0f}s)")
    rng = np.random.default_rng(cfg.split.seed)
    half = rng.random(len(val)) < 0.5
    val_cal, val_pol = val[half], val[~half]
    y_test = test["scenario"].to_numpy()
    w_test = prevalence_weights(y_test, cfg.policy.deployment_theft_prevalence)
    raw_test = model.predict_raw(test)
    # Calibrate on one half of validation and keep the calibrators only if they
    # reduce calibration error on the other half. Boosted trees with log loss are
    # often well calibrated already, and isotonic maps can overfit rare classes.
    y_pol = val_pol["scenario"].to_numpy()
    w_pol = prevalence_weights(y_pol, cfg.policy.deployment_theft_prevalence)
    ece_raw = M.expected_calibration_error(1 - model.predict_raw(val_pol)[:, 0], (y_pol != 0).astype(float), w_pol)
    model.calibrate(val_cal, val_cal["scenario"].to_numpy())
    ece_cal = M.expected_calibration_error(1 - model.predict_proba(val_pol)[:, 0], (y_pol != 0).astype(float), w_pol)
    calibrated = ece_cal < ece_raw
    if not calibrated:
        model.calibrators = None
    model.info["isotonic_calibration_kept"] = bool(calibrated)
    model.fit_reference(train)
    model.policy = fit_policy(model.predict_proba(val_pol), val_pol["scenario"].to_numpy(), cfg.policy)
    proba = model.predict_proba(test)
    decisions = model.policy.decide(proba)
    alert = decisions["tier_id"].to_numpy() >= 1
    priority = decisions["tier_id"].to_numpy() >= 2
    flag = decisions["flag_class"].to_numpy()
    value = test["value_at_risk_gbp"].to_numpy()
    theft = (y_test != 0).astype(float)

    metrics: dict = {"dataset": _dataset_summary(df), "config": cfg.to_dict()}
    metrics["model"] = {
        "classification": M.multiclass_report(y_test, proba.argmax(axis=1)),
        "ranking": M.ranking_metrics(y_test, proba, w_test),
        "calibration": {
            "isotonic_kept": bool(calibrated),
            "validation_ece_raw": round(ece_raw, 5),
            "validation_ece_isotonic": round(ece_cal, 5),
            "test_ece_raw": round(M.expected_calibration_error(1 - raw_test[:, 0], theft, w_test), 5),
            "test_ece_final": round(M.expected_calibration_error(1 - proba[:, 0], theft, w_test), 5),
        },
        "review_tier": M.alert_report(y_test, alert, flag, w_test),
        "priority_tier": M.alert_report(y_test, priority, flag, w_test),
        "economics_review": M.economics(y_test, alert, w_test, value, cfg.economics),
        "economics_priority": M.economics(y_test, priority, w_test, value, cfg.economics),
        "policy": model.policy.to_dict(),
        "n_iter": model.info["n_iter"],
    }

    # ---------------------------- baselines ---------------------------- #
    rule_pred = RuleBaseline().predict(test)
    rule_alert = rule_pred != 0
    metrics["rules"] = {
        "classification": M.multiclass_report(y_test, rule_pred),
        "alerts": M.alert_report(y_test, rule_alert, rule_pred, w_test),
        "economics": M.economics(y_test, rule_alert, w_test, value, cfg.economics),
    }
    logit = train_logistic(train.sample(min(80_000, len(train)), random_state=cfg.model.seed), cfg.model)
    logit_proba = logit.predict_raw(test)
    logit_cls, logit_risk = top_theft(logit_proba)
    metrics["logistic"] = {
        "classification": M.multiclass_report(y_test, logit_proba.argmax(axis=1)),
        "ranking": M.ranking_metrics(y_test, logit_proba, w_test),
    }
    _say(verbose, f"baselines done ({time.time() - started:.0f}s)")

    # Matched workload comparison: give every detector the same number of alerts.
    budget = float((w_test * alert).sum())
    matched = {}
    for name, risk in (("model", decisions["risk"].to_numpy()), ("logistic", logit_risk)):
        order = np.argsort(-risk)
        cut = np.searchsorted(np.cumsum(w_test[order]), budget)
        chosen = np.zeros(len(risk), dtype=bool)
        chosen[order[: max(cut, 1)]] = True
        matched[name] = M.alert_report(y_test, chosen, flag if name == "model" else logit_cls, w_test)
    metrics["matched_alert_budget"] = matched

    # ----------------------------- tables ------------------------------ #
    df.groupby(["scene", "scenario", "subtype"]).size().rename("clips").reset_index().to_csv(tables / "dataset_composition.csv", index=False)
    M.false_alert_sources(test["subtype"], y_test, alert).to_csv(tables / "false_alert_sources.csv", index=False)
    slices = pd.concat(
        [
            M.slice_report(test["camera_quality"], y_test, alert).assign(dimension="camera_quality"),
            M.slice_report(test["crowding"].map({0: "quiet", 1: "busy", 2: "crowded"}), y_test, alert).assign(dimension="crowding"),
            M.slice_report(test["store_format"], y_test, alert).assign(dimension="store_format"),
        ]
    )
    slices.to_csv(tables / "robustness_slices.csv", index=False)
    M.economics_curve(y_test, decisions["risk"].to_numpy(), w_test, value, cfg.economics).to_csv(tables / "economics_curve.csv", index=False)
    pd.concat(
        [
            M.calibration_curve(1 - raw_test[:, 0], theft, w_test).assign(stage="raw"),
            M.calibration_curve(1 - proba[:, 0], theft, w_test).assign(stage="final"),
        ]
    ).to_csv(tables / "calibration.csv", index=False)
    M.precision_vs_prevalence(y_test, alert, np.array([0.001, 0.002, 0.005, 0.01, 0.02, 0.03, 0.05, 0.10, 0.14])).to_csv(
        tables / "precision_vs_prevalence.csv", index=False
    )

    if not light:
        feat_imp, group_imp = _permutation_importance(model, test, cfg.model.seed)
        feat_imp.to_csv(tables / "feature_importance.csv", index=False)
        group_imp.to_csv(tables / "group_importance.csv", index=False)
        metrics["top_features"] = feat_imp.head(12).round(4).to_dict(orient="records")
        _say(verbose, f"permutation importance done ({time.time() - started:.0f}s)")
        ablation = _ablation(train, test, w_test, cfg)
        ablation.to_csv(tables / "sensor_ablation.csv", index=False)
        metrics["sensor_ablation"] = ablation.round(4).to_dict(orient="records")
        _say(verbose, f"sensor ablation done ({time.time() - started:.0f}s)")
        test_stores = [s for s in make_stores(cfg.data.n_stores, cfg.data.seed) if s.store_id in set(test["store_id"])]
        live = generate_batch(4000, seed=cfg.data.seed, shard=950, stores=test_stores, n_frames=cfg.data.n_frames, fps=cfg.data.fps)
        metrics["streaming"] = {"streaming_aware_model": streaming_report(model, live)}
        if prefix is not None:  # the same detector trained on whole clips only, for comparison
            naive = train_gbm(train, cfg.model)
            naive.policy = fit_policy(naive.predict_proba(val_pol), y_pol, cfg.policy)
            metrics["streaming"]["whole_clip_only_model"] = streaming_report(naive, live)
            naive_proba = naive.predict_proba(test)
            metrics["streaming"]["whole_clip_only_model"]["macro_f1"] = M.multiclass_report(y_test, naive_proba.argmax(axis=1))["macro_f1"]
        _say(verbose, f"streaming comparison done ({time.time() - started:.0f}s)")
        latency = latency_report(model, test_stores, cfg.data.seed, n_frames=cfg.data.n_frames, fps=cfg.data.fps)
        latency.to_csv(tables / "detection_latency.csv", index=False)
        metrics["detection_latency"] = latency.round(3).to_dict(orient="records")
        _say(verbose, f"streaming replay done ({time.time() - started:.0f}s)")

    # --------------------------- persistence --------------------------- #
    model.info.update({"trained_on_clips": int(len(train)), "feature_count": len(FEATURE_NAMES)})
    model.save(root / cfg.model.out_dir / "sentinel.joblib")
    artifacts = root / "artifacts"
    artifacts.mkdir(exist_ok=True)
    pred = test[["clip_id", "store_id", "scene", "scenario", "subtype", "camera_quality", "crowding", "value_at_risk_gbp"]].copy()
    for k in range(N_CLASSES):
        pred[f"p{k}"] = proba[:, k]
    pred = pd.concat([pred, decisions], axis=1)
    pred.to_parquet(artifacts / "test_predictions.parquet", index=False)
    metrics["runtime_seconds"] = round(time.time() - started, 1)
    (reports / "metrics.json").write_text(json.dumps(metrics, indent=2, default=float))
    _say(verbose, f"metrics written to {reports / 'metrics.json'}")
    return metrics
