"""Core data contracts shared by every stage of the pipeline.

Everything downstream of perception (simulation, real footage adapters,
feature extraction, rendering and narration) speaks the same language:

* ``keypoints``  float32 array of shape ``(N, T, 17, 3)`` holding COCO pose
  keypoints as ``x, y, confidence`` in normalised image coordinates.
* ``signals``    float32 array of shape ``(N, T, S)`` holding the item flow and
  point of sale channels listed in :data:`SIGNAL_NAMES`.
* ``meta``       one row per clip with context that does not vary per frame.

A clip is a short window (12.8 seconds by default) that follows ONE tracked
person. No pixels, faces or demographic attributes are ever stored.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum
from pathlib import Path

import numpy as np
import pandas as pd

# --------------------------------------------------------------------------- #
# Pose
# --------------------------------------------------------------------------- #
KEYPOINT_NAMES: tuple[str, ...] = (
    "nose",
    "left_eye",
    "right_eye",
    "left_ear",
    "right_ear",
    "left_shoulder",
    "right_shoulder",
    "left_elbow",
    "right_elbow",
    "left_wrist",
    "right_wrist",
    "left_hip",
    "right_hip",
    "left_knee",
    "right_knee",
    "left_ankle",
    "right_ankle",
)
KP: dict[str, int] = {name: i for i, name in enumerate(KEYPOINT_NAMES)}
N_KEYPOINTS = len(KEYPOINT_NAMES)

SKELETON_EDGES: tuple[tuple[int, int], ...] = (
    (KP["left_shoulder"], KP["right_shoulder"]),
    (KP["left_shoulder"], KP["left_elbow"]),
    (KP["left_elbow"], KP["left_wrist"]),
    (KP["right_shoulder"], KP["right_elbow"]),
    (KP["right_elbow"], KP["right_wrist"]),
    (KP["left_shoulder"], KP["left_hip"]),
    (KP["right_shoulder"], KP["right_hip"]),
    (KP["left_hip"], KP["right_hip"]),
    (KP["left_hip"], KP["left_knee"]),
    (KP["left_knee"], KP["left_ankle"]),
    (KP["right_hip"], KP["right_knee"]),
    (KP["right_knee"], KP["right_ankle"]),
)

LEFT, RIGHT = 0, 1  # hand indices used throughout the code base


# --------------------------------------------------------------------------- #
# Scenes and scenarios
# --------------------------------------------------------------------------- #
class Scene(IntEnum):
    """Camera context. Known at deployment time from the camera registry."""

    AISLE = 0
    SELF_CHECKOUT = 1
    STAFFED_TILL = 2
    EXIT = 3


class Scenario(IntEnum):
    """Target classes. ``NORMAL`` covers every benign behaviour."""

    NORMAL = 0
    CONCEALMENT_BAG = 1
    CONCEALMENT_CLOTHING = 2
    SHELF_SWEEP = 3
    SKIP_SCAN = 4
    TICKET_SWITCH = 5
    SWEETHEARTING = 6
    PUSH_OUT = 7


N_CLASSES = len(Scenario)
THEFT_CLASSES: tuple[Scenario, ...] = tuple(s for s in Scenario if s != Scenario.NORMAL)

SCENARIO_TITLES: dict[Scenario, str] = {
    Scenario.NORMAL: "Normal shopping",
    Scenario.CONCEALMENT_BAG: "Concealment in a bag",
    Scenario.CONCEALMENT_CLOTHING: "Concealment in clothing",
    Scenario.SHELF_SWEEP: "Shelf sweep",
    Scenario.SKIP_SCAN: "Self checkout skip scan",
    Scenario.TICKET_SWITCH: "Ticket switch",
    Scenario.SWEETHEARTING: "Sweethearting at the till",
    Scenario.PUSH_OUT: "Trolley push out",
}

SCENARIO_SCENE: dict[Scenario, Scene] = {
    Scenario.CONCEALMENT_BAG: Scene.AISLE,
    Scenario.CONCEALMENT_CLOTHING: Scene.AISLE,
    Scenario.SHELF_SWEEP: Scene.AISLE,
    Scenario.SKIP_SCAN: Scene.SELF_CHECKOUT,
    Scenario.TICKET_SWITCH: Scene.SELF_CHECKOUT,
    Scenario.SWEETHEARTING: Scene.STAFFED_TILL,
    Scenario.PUSH_OUT: Scene.EXIT,
}

SCENE_TITLES: dict[Scene, str] = {
    Scene.AISLE: "Aisle",
    Scene.SELF_CHECKOUT: "Self checkout",
    Scene.STAFFED_TILL: "Staffed till",
    Scene.EXIT: "Store exit",
}

CAMERA_QUALITIES: tuple[str, ...] = ("hd", "sd", "low_light")

# --------------------------------------------------------------------------- #
# Signal channels (item flow, personal containers, point of sale)
# --------------------------------------------------------------------------- #
SIGNAL_NAMES: tuple[str, ...] = (
    "hand_item_l",  # detector confidence that the left hand holds merchandise
    "hand_item_r",  # same for the right hand
    "shelf_delta",  # shelf sensor: +1 item removed from shelf, minus 1 item put back
    "container_count",  # visible merchandise count in basket or trolley
    "bagging_count",  # item count on the bagging area or till output
    "pos_qty",  # quantity registered by the point of sale in this frame
    "scan_match",  # visual vs scanned product similarity at scan frames (0..1)
    "price_ratio",  # scanned price divided by visually expected price at scan frames
    "bag_x",  # personal bag centre (handbag, tote, own shopping bag)
    "bag_y",
    "bag_conf",
    "cont_x",  # basket or trolley centre
    "cont_y",
    "cont_conf",
    "eas_alarm",  # electronic article surveillance gate alarm
)
SIG: dict[str, int] = {name: i for i, name in enumerate(SIGNAL_NAMES)}
N_SIGNALS = len(SIGNAL_NAMES)

DEFAULT_FPS = 5.0
DEFAULT_FRAMES = 64

META_COLUMNS: tuple[str, ...] = (
    "clip_id",
    "store_id",
    "scene",
    "scenario",
    "subtype",
    "camera_quality",
    "crowding",
    "hour",
    "body_scale",
    "txn_linked",
    "checkout_dwell_s",
    "event_time_s",
    "value_at_risk_gbp",
)


@dataclass
class ClipBatch:
    """A batch of single person track clips in the shared schema."""

    keypoints: np.ndarray  # (N, T, 17, 3)
    signals: np.ndarray  # (N, T, S)
    meta: pd.DataFrame  # N rows
    fps: float = DEFAULT_FPS

    def __post_init__(self) -> None:
        n = len(self.meta)
        if self.keypoints.ndim != 4 or self.keypoints.shape[2:] != (N_KEYPOINTS, 3):
            raise ValueError(f"keypoints must be (N, T, 17, 3), got {self.keypoints.shape}")
        if self.signals.ndim != 3 or self.signals.shape[2] != N_SIGNALS:
            raise ValueError(f"signals must be (N, T, {N_SIGNALS}), got {self.signals.shape}")
        if not (self.keypoints.shape[0] == self.signals.shape[0] == n):
            raise ValueError("keypoints, signals and meta must describe the same clips")
        if self.keypoints.shape[1] != self.signals.shape[1]:
            raise ValueError("keypoints and signals must share the time axis")

    def __len__(self) -> int:
        return len(self.meta)

    @property
    def n_frames(self) -> int:
        return int(self.keypoints.shape[1])

    @property
    def duration_s(self) -> float:
        return self.n_frames / self.fps

    def select(self, index) -> ClipBatch:
        """Return a new batch holding the clips at ``index`` (ints, slice or mask)."""
        idx = np.atleast_1d(np.arange(len(self))[index])
        return ClipBatch(
            keypoints=self.keypoints[idx],
            signals=self.signals[idx],
            meta=self.meta.iloc[idx].reset_index(drop=True),
            fps=self.fps,
        )

    def prefix(self, n_frames: int) -> ClipBatch:
        """Return the first ``n_frames`` of every clip (used for streaming replay)."""
        n_frames = int(np.clip(n_frames, 1, self.n_frames))
        return ClipBatch(
            keypoints=self.keypoints[:, :n_frames],
            signals=self.signals[:, :n_frames],
            meta=self.meta.copy(),
            fps=self.fps,
        )

    def save(self, path: str | Path) -> Path:
        """Persist the batch as a compressed ``.npz`` plus a sibling parquet file."""
        path = Path(path).with_suffix(".npz")
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            path,
            keypoints=self.keypoints.astype(np.float16),
            signals=self.signals.astype(np.float32),
            fps=np.float32(self.fps),
        )
        self.meta.to_parquet(path.with_suffix(".meta.parquet"), index=False)
        return path

    @classmethod
    def load(cls, path: str | Path) -> ClipBatch:
        path = Path(path).with_suffix(".npz")
        with np.load(path) as data:
            keypoints = data["keypoints"].astype(np.float32)
            signals = data["signals"].astype(np.float32)
            fps = float(data["fps"])
        meta = pd.read_parquet(path.with_suffix(".meta.parquet"))
        return cls(keypoints=keypoints, signals=signals, meta=meta, fps=fps)

    @classmethod
    def concat(cls, batches: list[ClipBatch]) -> ClipBatch:
        if not batches:
            raise ValueError("cannot concatenate an empty list of batches")
        return cls(
            keypoints=np.concatenate([b.keypoints for b in batches]),
            signals=np.concatenate([b.signals for b in batches]),
            meta=pd.concat([b.meta for b in batches], ignore_index=True),
            fps=batches[0].fps,
        )
