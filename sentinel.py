"""The learned detector: gradient boosted trees with per class calibration."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from ..config import ModelConfig, SplitConfig
from ..features.extractor import FEATURE_NAMES
from ..schema import N_CLASSES, SCENARIO_TITLES, Scenario
from .policy import AlertPolicy


def split_by_store(df: pd.DataFrame, cfg: SplitConfig) -> pd.Series:
    """Assign whole stores to train, validation or test.

    Splitting by store (not by clip) means the test set measures how the model
    travels to stores, cameras and theft mixes it has never seen. The split is
    stratified by store format so every format appears in every partition.
    """
    rng = np.random.default_rng(cfg.seed)
    stores = df[["store_id", "store_format"]].drop_duplicates().sort_values("store_id")
    mapping: dict[int, str] = {}
    for _, group in stores.groupby("store_format"):
        ids = group["store_id"].to_numpy().copy()
        rng.shuffle(ids)
        n = len(ids)
        n_test = max(1, int(round(n * (1.0 - cfg.train_frac - cfg.val_frac)))) if n >= 3 else 0
        n_val = max(1, int(round(n * cfg.val_frac))) if n >= 3 else 0
        for i, sid in enumerate(ids):
            mapping[int(sid)] = "test" if i < n_test else ("val" if i < n_test + n_val else "train")
    return df["store_id"].map(mapping).rename("split")


@dataclass
class SentinelModel:
    """Fitted estimator plus everything needed to score and explain a clip."""

    estimator: object
    feature_names: tuple[str, ...] = FEATURE_NAMES
    calibrators: list[IsotonicRegression] | None = None
    reference: dict = field(default_factory=dict)  # per scene feature statistics of normal clips
    policy: AlertPolicy | None = None
    info: dict = field(default_factory=dict)

    def _matrix(self, X: pd.DataFrame) -> np.ndarray:
        return X.loc[:, list(self.feature_names)].to_numpy(dtype=np.float32)

    def predict_raw(self, X: pd.DataFrame) -> np.ndarray:
        """Uncalibrated class probabilities with one column per scenario."""
        proba = self.estimator.predict_proba(self._matrix(X))
        full = np.zeros((len(X), N_CLASSES), dtype=np.float64)
        full[:, np.asarray(self.estimator.classes_, dtype=int)] = proba
        return full

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Calibrated class probabilities (rows sum to one)."""
        raw = self.predict_raw(X)
        if not self.calibrators:
            return raw
        cal = np.column_stack([c.predict(raw[:, k]) for k, c in enumerate(self.calibrators)])
        total = cal.sum(axis=1, keepdims=True)
        return np.where(total > 0, cal / np.maximum(total, 1e-12), raw)

    def assess(self, X: pd.DataFrame) -> pd.DataFrame:
        """Score clips end to end: probabilities, flagged scenario, risk and tier."""
        if self.policy is None:
            raise RuntimeError("this model has no alert policy; call fit_policy first")
        proba = self.predict_proba(X)
        out = self.policy.decide(proba)
        out["flag_title"] = [SCENARIO_TITLES[Scenario(int(c))] for c in out["flag_class"]]
        out["p_normal"] = proba[:, 0]
        return out

    def calibrate(self, X: pd.DataFrame, y: np.ndarray) -> None:
        """Fit one isotonic map per class on held out data."""
        raw = self.predict_raw(X)
        self.calibrators = []
        for k in range(N_CLASSES):
            iso = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip")
            iso.fit(raw[:, k], (y == k).astype(float))
            self.calibrators.append(iso)

    def fit_reference(self, df: pd.DataFrame) -> None:
        """Remember what normal looks like in each scene, for explanations."""
        normal = df[df["scenario"] == int(Scenario.NORMAL)]
        for scene, grp in normal.groupby("scene"):
            feats = grp.loc[:, list(self.feature_names)]
            self.reference[int(scene)] = {
                "median": feats.median().to_dict(),
                "p05": feats.quantile(0.05).to_dict(),
                "p95": feats.quantile(0.95).to_dict(),
            }

    def save(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path, compress=3)
        return path

    @staticmethod
    def load(path: str | Path) -> SentinelModel:
        model = joblib.load(path)
        if not isinstance(model, SentinelModel):
            raise TypeError(f"{path} does not contain a SentinelModel")
        return model


def train_gbm(
    train: pd.DataFrame,
    cfg: ModelConfig,
    feature_names: tuple[str, ...] = FEATURE_NAMES,
    max_iter: int | None = None,
) -> SentinelModel:
    """Fit the gradient boosted detector on the training stores."""
    est = HistGradientBoostingClassifier(
        max_iter=max_iter or cfg.max_iter,
        learning_rate=cfg.learning_rate,
        max_leaf_nodes=cfg.max_leaf_nodes,
        l2_regularization=cfg.l2_regularization,
        min_samples_leaf=cfg.min_samples_leaf,
        early_stopping=True,
        n_iter_no_change=cfg.early_stopping_rounds,
        validation_fraction=0.1,
        random_state=cfg.seed,
    )
    X = train.loc[:, list(feature_names)].to_numpy(dtype=np.float32)
    est.fit(X, train["scenario"].to_numpy())
    return SentinelModel(estimator=est, feature_names=tuple(feature_names), info={"n_iter": int(est.n_iter_)})


def train_logistic(train: pd.DataFrame, cfg: ModelConfig) -> SentinelModel:
    """Linear baseline on standardised features."""
    est = make_pipeline(StandardScaler(), LogisticRegression(max_iter=300, C=1.0, random_state=cfg.seed))
    X = train.loc[:, list(FEATURE_NAMES)].to_numpy(dtype=np.float32)
    est.fit(X, train["scenario"].to_numpy())
    return SentinelModel(estimator=est, info={"kind": "logistic"})
