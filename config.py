"""Typed configuration loaded from YAML.

Every number that shapes the data, the model or the business case lives here so
an experiment is fully described by one file under ``configs/``.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass
class DataConfig:
    n_clips: int = 250_000
    shard_size: int = 5_000
    n_frames: int = 64
    fps: float = 5.0
    n_stores: int = 40
    seed: int = 2026
    raw_sample_shards: int = 1  # how many shards also keep raw keypoints on disk
    prefix_clips: int = 50_000  # extra clips cut into partial windows for streaming aware training
    prefixes_per_clip: int = 2
    out_dir: str = "data/generated"


@dataclass
class SplitConfig:
    """Stores are split, never clips, so evaluation is on unseen stores."""

    train_frac: float = 0.65
    val_frac: float = 0.175
    seed: int = 7


@dataclass
class ModelConfig:
    max_iter: int = 300
    learning_rate: float = 0.08
    max_leaf_nodes: int = 31
    l2_regularization: float = 0.5
    min_samples_leaf: int = 40
    early_stopping_rounds: int = 20
    seed: int = 11
    out_dir: str = "models"


@dataclass
class PolicyConfig:
    """How calibrated probabilities become review alerts."""

    deployment_theft_prevalence: float = 0.005
    review_precision_target: float = 0.80
    priority_precision_target: float = 0.95
    min_threshold: float = 0.20
    # Safety margin for the review tier: a threshold must meet its target even
    # if this many more honest validation clips had been flagged at it.
    phantom_false_alerts: float = 1.0


@dataclass
class EconomicsConfig:
    """Illustrative assumptions. Replace with figures from your own estate."""

    review_cost_gbp: float = 0.45  # one analyst minute per alert
    recovery_rate: float = 0.30  # share of flagged value recovered or deterred
    false_alert_friction_gbp: float = 0.30  # extra cost of a wasted review
    clips_per_store_day: int = 2_000


@dataclass
class NarratorConfig:
    provider: str = "template"  # "template" or "anthropic"
    model: str = "claude-sonnet-5-5"
    max_tokens: int = 700
    attach_keyframes: bool = True


@dataclass
class Config:
    data: DataConfig = field(default_factory=DataConfig)
    split: SplitConfig = field(default_factory=SplitConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    policy: PolicyConfig = field(default_factory=PolicyConfig)
    economics: EconomicsConfig = field(default_factory=EconomicsConfig)
    narrator: NarratorConfig = field(default_factory=NarratorConfig)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _merge(obj: Any, updates: dict[str, Any]) -> None:
    valid = {f.name for f in fields(obj)}
    for key, value in updates.items():
        if key not in valid:
            raise KeyError(f"unknown config key '{key}' for {type(obj).__name__}")
        current = getattr(obj, key)
        if is_dataclass(current) and isinstance(value, dict):
            _merge(current, value)
        else:
            setattr(obj, key, type(current)(value) if current is not None else value)


def load_config(path: str | Path | None = None, overrides: dict[str, Any] | None = None) -> Config:
    """Load defaults, then a YAML file, then a dictionary of overrides."""
    cfg = Config()
    if path is not None:
        with open(path, encoding="utf8") as fh:
            _merge(cfg, yaml.safe_load(fh) or {})
    if overrides:
        _merge(cfg, overrides)
    return cfg
