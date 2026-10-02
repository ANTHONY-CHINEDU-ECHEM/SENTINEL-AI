"""Shared fixtures. The tiny project is built once per test session."""

from __future__ import annotations

import pytest

from shelfsentinel.config import load_config
from shelfsentinel.features import extract_features
from shelfsentinel.models.sentinel import SentinelModel
from shelfsentinel.pipeline import run_training
from shelfsentinel.simulation import generate_batch, generate_dataset
from shelfsentinel.simulation.generator import build_table


@pytest.fixture(scope="session")
def batch():
    """600 clips drawn from the natural behaviour mix."""
    return generate_batch(600, seed=123)


@pytest.fixture(scope="session")
def table(batch):
    return build_table(batch.meta, extract_features(batch))


@pytest.fixture(scope="session")
def tiny_cfg():
    return load_config(
        overrides={
            "data": {"n_clips": 8000, "shard_size": 2000, "n_stores": 12, "out_dir": "data/tiny", "prefix_clips": 2000},
            "model": {"max_iter": 60},
            # A tiny validation set cannot support the production safety margin.
            "policy": {"phantom_false_alerts": 0.0, "review_precision_target": 0.6, "priority_precision_target": 0.8},
        }
    )


@pytest.fixture(scope="session")
def tiny_project(tmp_path_factory, tiny_cfg):
    """A complete miniature run: corpus, trained model and metrics."""
    root = tmp_path_factory.mktemp("project")
    manifest = generate_dataset(tiny_cfg.data, root, verbose=False)
    metrics = run_training(tiny_cfg, root, verbose=False, light=True)
    model = SentinelModel.load(root / "models" / "sentinel.joblib")
    return {"root": root, "manifest": manifest, "metrics": metrics, "model": model}
