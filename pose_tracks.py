"""Bridge from real pose estimators to the Shelf Sentinel clip schema.

Any tracker that emits COCO 17 keypoints per person per frame can feed the
pipeline: Ultralytics YOLO pose, MMPose, OpenPifPaf, or a pose only dataset
such as PoseLift. This module converts such tracks into :class:`ClipBatch`
windows at the working frame rate.

Pose alone activates the pose features. The item flow and point of sale
channels need their own sources (an item detector, shelf sensors, the till
feed); pass them through ``signals`` when you have them. Without them the
channels are zero and the detector behaves like the pose only ablation.
"""

from __future__ import annotations

from collections import defaultdict

import numpy as np
import pandas as pd

from ..schema import DEFAULT_FPS, DEFAULT_FRAMES, N_KEYPOINTS, N_SIGNALS, ClipBatch, Scene


def tracks_from_frames(frames: list[dict], image_size: tuple[int, int]) -> dict[int, tuple[np.ndarray, np.ndarray]]:
    """Group per frame detections into per person tracks.

    Parameters
    ----------
    frames : list of ``{"frame": int, "persons": [{"track_id": int,
        "keypoints": [[x, y, conf], ... 17 rows]}]}`` in pixel coordinates.
    image_size : ``(width, height)`` in pixels, used to normalise coordinates.

    Returns
    -------
    ``{track_id: (frame_numbers, keypoints)}`` with keypoints shaped
    ``(T, 17, 3)`` in normalised coordinates.
    """
    w, h = image_size
    by_track: dict[int, list[tuple[int, np.ndarray]]] = defaultdict(list)
    for frame in frames:
        for person in frame.get("persons", []):
            kp = np.asarray(person["keypoints"], dtype=np.float32)
            if kp.shape != (N_KEYPOINTS, 3):
                raise ValueError(f"expected keypoints of shape (17, 3), got {kp.shape}")
            kp = kp.copy()
            kp[:, 0] /= w
            kp[:, 1] /= h
            by_track[int(person["track_id"])].append((int(frame["frame"]), kp))
    out = {}
    for tid, items in by_track.items():
        items.sort(key=lambda it: it[0])
        out[tid] = (np.array([it[0] for it in items]), np.stack([it[1] for it in items]))
    return out


def resample_track(frame_numbers: np.ndarray, keypoints: np.ndarray, fps_in: float, fps_out: float = DEFAULT_FPS) -> np.ndarray:
    """Resample a track to the working frame rate, marking gaps as unseen."""
    t_in = frame_numbers / fps_in
    t_out = np.arange(t_in[0], t_in[-1] + 1e-9, 1.0 / fps_out)
    nearest = np.clip(np.searchsorted(t_in, t_out), 0, len(t_in) - 1)
    left = np.clip(nearest - 1, 0, len(t_in) - 1)
    use = np.where(np.abs(t_in[left] - t_out) <= np.abs(t_in[nearest] - t_out), left, nearest)
    out = keypoints[use].copy()
    missing = np.abs(t_in[use] - t_out) > 0.75 / fps_out
    out[missing, :, 2] = 0.0  # no detection near this instant: confidence zero
    return out


def clips_from_track(
    keypoints: np.ndarray,
    scene: Scene,
    track_id: int = 0,
    store_id: int = 0,
    n_frames: int = DEFAULT_FRAMES,
    stride: int = DEFAULT_FRAMES // 2,
    fps: float = DEFAULT_FPS,
    signals: np.ndarray | None = None,
    txn_linked: int = 0,
    checkout_dwell_s: float = 0.0,
    hour: int = 12,
) -> ClipBatch:
    """Cut one resampled track into overlapping windows in the clip schema."""
    t = keypoints.shape[0]
    if t < n_frames:
        pad = np.zeros((n_frames - t, N_KEYPOINTS, 3), dtype=np.float32)
        keypoints = np.concatenate([keypoints, pad])
        t = n_frames
    if signals is None:
        signals = np.zeros((t, N_SIGNALS), dtype=np.float32)
    elif signals.shape[0] < t:
        signals = np.concatenate([signals, np.zeros((t - signals.shape[0], N_SIGNALS), dtype=np.float32)])
    starts = list(range(0, t - n_frames + 1, stride))
    kp = np.stack([keypoints[s : s + n_frames] for s in starts]).astype(np.float32)
    sg = np.stack([signals[s : s + n_frames] for s in starts]).astype(np.float32)
    hips = kp[:, :, 11:13, :2].mean(axis=2)
    shoulders = kp[:, :, 5:7, :2].mean(axis=2)
    scale = np.median(np.linalg.norm(shoulders - hips, axis=-1), axis=1) / 0.29
    meta = pd.DataFrame(
        {
            "clip_id": [f"T{track_id:04d}W{i:04d}" for i in range(len(starts))],
            "store_id": store_id,
            "store_format": "unknown",
            "scene": int(scene),
            "scenario": 0,
            "subtype": "unlabelled",
            "variant": "",
            "camera_quality": "unknown",
            "crowding": 0,
            "hour": hour,
            "body_scale": scale,
            "txn_linked": txn_linked,
            "checkout_dwell_s": checkout_dwell_s,
            "event_time_s": np.nan,
            "value_at_risk_gbp": 0.0,
            "events_json": "[]",
        }
    )
    return ClipBatch(keypoints=kp, signals=sg, meta=meta, fps=fps)


def batch_to_frames(batch: ClipBatch, index: int, image_size: tuple[int, int], track_id: int = 1) -> list[dict]:
    """Inverse of :func:`tracks_from_frames` for one clip (used in tests and demos)."""
    w, h = image_size
    frames = []
    for f in range(batch.n_frames):
        kp = batch.keypoints[index, f].copy()
        kp[:, 0] *= w
        kp[:, 1] *= h
        frames.append({"frame": f, "persons": [{"track_id": track_id, "keypoints": kp.tolist()}]})
    return frames
