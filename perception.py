"""Sensor model: what the perception stack actually reports.

The behaviour scripts describe what truly happened. Real perception is messier:
pose estimators jitter and drop joints, item detectors miss products that are
held against the body, shelf sensors skip events, and item counters drift. This
module degrades the ground truth so the detector has to earn its accuracy, and
so camera quality and crowding can be studied as robustness slices.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..schema import KP, SIG


@dataclass(frozen=True)
class NoiseConfig:
    """Noise levels indexed by camera quality: hd, sd, low light."""

    kp_sigma: tuple[float, float, float] = (0.004, 0.008, 0.013)
    kp_dropout: tuple[float, float, float] = (0.02, 0.05, 0.10)
    item_miss: tuple[float, float, float] = (0.05, 0.10, 0.18)
    item_false: tuple[float, float, float] = (0.008, 0.015, 0.03)
    body_occlusion_miss: float = 0.30
    shelf_miss: tuple[float, float, float] = (0.04, 0.07, 0.12)
    shelf_spurious: float = 0.03
    count_drift: tuple[float, float, float] = (0.05, 0.08, 0.12)
    count_flicker: float = 0.03
    full_trolley_knee: float = 8.0
    full_trolley_slope: float = 0.6
    bag_miss: tuple[float, float, float] = (0.05, 0.08, 0.14)
    occlusion_burst: tuple[float, float, float] = (0.0, 0.20, 0.45)  # by crowding level
    pos_clock_skew_frames: int = 2


def _per_clip(values: tuple[float, float, float], quality: np.ndarray) -> np.ndarray:
    return np.asarray(values, dtype=np.float32)[quality]


def apply_sensor_noise(
    xy: np.ndarray,
    visibility: np.ndarray,
    signals: np.ndarray,
    scale: np.ndarray,
    quality: np.ndarray,
    crowding: np.ndarray,
    lower_occluded: np.ndarray,
    rng: np.random.Generator,
    trolley_scene: np.ndarray | None = None,
    back_view: np.ndarray | None = None,
    cfg: NoiseConfig | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Return observed ``keypoints (N, T, 17, 3)`` and ``signals (N, T, S)``."""
    cfg = cfg or NoiseConfig()
    n, t = xy.shape[:2]
    out_sig = signals.copy()

    # ------------------------------------------------------------------ #
    # Keypoints
    # ------------------------------------------------------------------ #
    sigma = _per_clip(cfg.kp_sigma, quality) * (1.0 + 0.3 * crowding)
    kp_xy = xy + rng.normal(0.0, 1.0, xy.shape).astype(np.float32) * sigma[:, None, None, None]
    conf = visibility * rng.uniform(0.85, 1.0, visibility.shape).astype(np.float32)

    drop = rng.random(visibility.shape) < _per_clip(cfg.kp_dropout, quality)[:, None, None]

    # Counter scenes hide the legs and partly hide the hips.
    legs = [KP[k] for k in ("left_knee", "right_knee", "left_ankle", "right_ankle")]
    hips = [KP[k] for k in ("left_hip", "right_hip")]
    occ = lower_occluded.astype(bool)
    if occ.any():
        leg_mask = np.zeros_like(drop)
        leg_mask[np.ix_(occ, np.arange(t), legs)] = True
        drop |= leg_mask
        conf[np.ix_(occ, np.arange(t), hips)] *= 0.6

    # Crowding: another shopper walks through and hides one arm for a while.
    burst_p = np.asarray(cfg.occlusion_burst, dtype=np.float32)[crowding]
    has_burst = rng.random(n) < burst_p
    burst_len = rng.integers(5, 16, size=n)
    burst_start = rng.integers(0, max(t - 5, 1), size=n)
    burst_side = rng.integers(0, 2, size=n)
    frames = np.arange(t)[None, :]
    in_burst = has_burst[:, None] & (frames >= burst_start[:, None]) & (frames < (burst_start + burst_len)[:, None])
    for side, names in enumerate((("left_elbow", "left_wrist"), ("right_elbow", "right_wrist"))):
        mask = in_burst & (burst_side == side)[:, None]
        for name in names:
            drop[:, :, KP[name]] |= mask

    conf = np.where(drop, rng.uniform(0.02, 0.25, conf.shape), conf).astype(np.float32)
    kp_xy = np.where(drop[..., None], kp_xy + rng.normal(0.0, 0.04, kp_xy.shape), kp_xy).astype(np.float32)
    keypoints = np.concatenate([kp_xy, conf[..., None]], axis=-1).astype(np.float32)

    # ------------------------------------------------------------------ #
    # Item in hand
    # ------------------------------------------------------------------ #
    hip_mid = 0.5 * (xy[:, :, KP["left_hip"]] + xy[:, :, KP["right_hip"]])
    sh_mid = 0.5 * (xy[:, :, KP["left_shoulder"]] + xy[:, :, KP["right_shoulder"]])
    torso = hip_mid + 0.35 * (sh_mid - hip_mid)
    miss_p = _per_clip(cfg.item_miss, quality)[:, None]
    false_p = _per_clip(cfg.item_false, quality)[:, None]
    for side, (chan, wrist) in enumerate((("hand_item_l", "left_wrist"), ("hand_item_r", "right_wrist"))):
        truth = signals[:, :, SIG[chan]] > 0.5
        dist = np.linalg.norm(xy[:, :, KP[wrist]] - torso, axis=-1) / scale[:, None]
        # Seen from behind, a hand held in front of the torso is hidden by the body.
        behind_body = (dist < 0.12) & (back_view[:, None] if back_view is not None else True)
        p_miss = miss_p + cfg.body_occlusion_miss * behind_body
        hidden = in_burst & (burst_side == side)[:, None]
        seen = truth & (rng.random((n, t)) >= p_miss) & ~hidden
        ghost = ~truth & (rng.random((n, t)) < false_p)
        obs = np.where(seen, rng.uniform(0.62, 0.98, (n, t)), rng.uniform(0.0, 0.22, (n, t)))
        obs = np.where(ghost, rng.uniform(0.55, 0.80, (n, t)), obs)
        out_sig[:, :, SIG[chan]] = obs

    # ------------------------------------------------------------------ #
    # Shelf sensor
    # ------------------------------------------------------------------ #
    shelf = signals[:, :, SIG["shelf_delta"]].copy()
    missed = (shelf != 0) & (rng.random((n, t)) < _per_clip(cfg.shelf_miss, quality)[:, None])
    shelf[missed] = 0.0
    spurious = np.flatnonzero(rng.random(n) < cfg.shelf_spurious)
    shelf[spurious, rng.integers(0, t, size=len(spurious))] += rng.choice([-1.0, 1.0], size=len(spurious))
    out_sig[:, :, SIG["shelf_delta"]] = shelf

    # ------------------------------------------------------------------ #
    # Item counters
    # ------------------------------------------------------------------ #
    drift_p = _per_clip(cfg.count_drift, quality)
    for chan, p_scale in (("container_count", 1.0), ("bagging_count", 0.7)):
        truth = signals[:, :, SIG[chan]]
        # A heaped trolley hides items from the counter; flat counters do not.
        saturates = (trolley_scene[:, None] if trolley_scene is not None else True) & (chan == "container_count")
        obs = np.where(
            (truth > cfg.full_trolley_knee) & saturates,
            cfg.full_trolley_knee + cfg.full_trolley_slope * (truth - cfg.full_trolley_knee),
            truth,
        )
        drifts = rng.random(n) < drift_p * p_scale
        d_start = rng.integers(0, t, size=n)
        d_sign = rng.choice([-1.0, 1.0], size=n)
        obs = obs + (drifts[:, None] & (frames >= d_start[:, None])) * d_sign[:, None]
        flicker = rng.random((n, t)) < cfg.count_flicker
        obs = obs + flicker * rng.choice([-1.0, 1.0], size=(n, t))
        present = truth.max(axis=1, keepdims=True) > 0
        out_sig[:, :, SIG[chan]] = np.where(present, np.maximum(np.round(obs), 0.0), 0.0)

    # ------------------------------------------------------------------ #
    # Point of sale: exact content, slightly skewed clock
    # ------------------------------------------------------------------ #
    k = cfg.pos_clock_skew_frames
    if k > 0:
        shift = rng.integers(-k, k + 1, size=n)
        idx = np.clip(frames - shift[:, None], 0, t - 1)
        valid = (frames - shift[:, None] >= 0) & (frames - shift[:, None] < t)
        rows = np.arange(n)[:, None]
        for chan in ("pos_qty", "scan_match", "price_ratio"):
            out_sig[:, :, SIG[chan]] = np.where(valid, signals[rows, idx, SIG[chan]], 0.0)
    scanned = out_sig[:, :, SIG["scan_match"]] > 0
    noisy_match = np.clip(out_sig[:, :, SIG["scan_match"]] + rng.normal(0.0, 0.05, (n, t)), 0.01, 1.0)
    out_sig[:, :, SIG["scan_match"]] = np.where(scanned, noisy_match, 0.0)

    # ------------------------------------------------------------------ #
    # Personal bag and container detections
    # ------------------------------------------------------------------ #
    bag_missed = rng.random(n) < _per_clip(cfg.bag_miss, quality)
    for prefix, missed_clip in (("bag", bag_missed), ("cont", np.zeros(n, dtype=bool))):
        present = signals[:, :, SIG[f"{prefix}_conf"]] > 0.5
        seen = present & ~missed_clip[:, None]
        conf_obs = np.where(seen, rng.uniform(0.60, 0.95, (n, t)), rng.uniform(0.0, 0.12, (n, t)))
        conf_obs = np.where(present & missed_clip[:, None], rng.uniform(0.05, 0.25, (n, t)), conf_obs)
        out_sig[:, :, SIG[f"{prefix}_conf"]] = conf_obs
        for axis in ("x", "y"):
            c = SIG[f"{prefix}_{axis}"]
            out_sig[:, :, c] = np.where(present, signals[:, :, c] + rng.normal(0.0, 0.008, (n, t)), 0.0)

    return keypoints, out_sig.astype(np.float32)
