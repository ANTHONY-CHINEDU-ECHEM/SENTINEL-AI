"""Streaming replay: how early in a clip does the alert fire?

The detector is trained on whole clips, but features are defined on any prefix.
Replaying a clip frame by frame therefore gives a risk curve over time and a
detection latency: seconds between the theft event and the first alert.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..features.extractor import extract_features
from ..models.sentinel import SentinelModel
from ..schema import SCENARIO_TITLES, ClipBatch, Scenario
from ..simulation.behaviours import REGISTRY
from ..simulation.generator import StoreProfile, generate_batch


def risk_curves(model: SentinelModel, batch: ClipBatch, min_frames: int = 8, step: int = 2) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Replay every clip. Returns ``(frame_counts, risk, tier_id)`` with shape (N, steps)."""
    counts = np.arange(min_frames, batch.n_frames + 1, step)
    if counts[-1] != batch.n_frames:
        counts = np.r_[counts, batch.n_frames]
    risk = np.zeros((len(batch), len(counts)))
    tier = np.zeros((len(batch), len(counts)), dtype=int)
    cls = np.zeros((len(batch), len(counts)), dtype=int)
    for j, n in enumerate(counts):
        out = model.assess(extract_features(batch.prefix(int(n))))
        risk[:, j] = out["risk"].to_numpy()
        tier[:, j] = out["tier_id"].to_numpy()
        cls[:, j] = out["flag_class"].to_numpy()
    return counts, risk, tier


def confirmed(tier: np.ndarray, confirm: int = 2) -> np.ndarray:
    """Alert state after debouncing: an alert must hold for ``confirm`` evaluations in a row."""
    fired = tier >= 1
    out = fired.copy()
    for k in range(1, confirm):
        out[:, k:] &= fired[:, :-k]
        out[:, :k] = False
    return out


def streaming_report(model: SentinelModel, batch: ClipBatch, confirm: int = 2) -> dict:
    """Compare scoring a clip once it is complete with scoring it as it unfolds."""
    counts, _, tier = risk_curves(model, batch)
    normal = batch.meta["scenario"].to_numpy() == int(Scenario.NORMAL)
    live = confirmed(tier, confirm)
    return {
        "clips": int(len(batch)),
        "honest_clips": int(normal.sum()),
        "whole_clip_false_alerts_per_1000": round(float((tier[normal, -1] >= 1).mean() * 1000), 2),
        "streaming_false_alerts_per_1000": round(float(live[normal].any(axis=1).mean() * 1000), 2),
        "streaming_false_alerts_unconfirmed_per_1000": round(float((tier[normal] >= 1).any(axis=1).mean() * 1000), 2),
        "whole_clip_theft_recall": round(float((tier[~normal, -1] >= 1).mean()), 4),
        "streaming_theft_recall": round(float(live[~normal].any(axis=1).mean()), 4),
        "confirm_evaluations": confirm,
    }


def latency_report(
    model: SentinelModel,
    stores: list[StoreProfile],
    seed: int,
    n_per_scenario: int = 150,
    n_frames: int = 64,
    fps: float = 5.0,
    confirm: int = 2,
) -> pd.DataFrame:
    """Detection latency per theft scenario on freshly simulated clips."""
    theft_subtypes = [name for name, spec in REGISTRY.items() if spec.scenario != Scenario.NORMAL]
    batch = generate_batch(
        n_per_scenario * len(theft_subtypes), seed=seed, shard=900, stores=stores, n_frames=n_frames, fps=fps, subtypes=theft_subtypes
    )
    counts, _, tier = risk_curves(model, batch)
    fired = confirmed(tier, confirm)
    first = np.where(fired.any(axis=1), counts[np.argmax(fired, axis=1)], -1)
    df = batch.meta[["scenario", "event_time_s"]].copy()
    df["alert_time_s"] = np.where(first > 0, first / fps, np.nan)
    df["latency_s"] = df["alert_time_s"] - df["event_time_s"]
    rows = []
    for scen, g in df.groupby("scenario"):
        has_event = g["event_time_s"].notna()
        det = g.loc[has_event, "latency_s"].dropna()
        rows.append(
            {
                "scenario": SCENARIO_TITLES[Scenario(int(scen))],
                "clips": int(has_event.sum()),
                "detected_in_clip": float(g.loc[has_event, "alert_time_s"].notna().mean()),
                "median_latency_s": float(det.median()) if len(det) else np.nan,
                "p90_latency_s": float(det.quantile(0.9)) if len(det) else np.nan,
            }
        )
    return pd.DataFrame(rows)
