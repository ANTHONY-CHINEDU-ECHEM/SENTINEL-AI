"""Scene geometry for the four camera contexts.

All coordinates are normalised image coordinates: ``x`` grows to the right and
``y`` grows downward, both in the range 0 to 1. In a live deployment these
zones come from a one time camera calibration step; here they are fixed so the
simulator, the feature extractor and the renderer agree on where things are.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .schema import Scene


@dataclass(frozen=True)
class Rect:
    """Axis aligned rectangle in normalised image coordinates."""

    x0: float
    y0: float
    x1: float
    y1: float

    @property
    def center(self) -> tuple[float, float]:
        return (0.5 * (self.x0 + self.x1), 0.5 * (self.y0 + self.y1))

    @property
    def width(self) -> float:
        return self.x1 - self.x0

    @property
    def height(self) -> float:
        return self.y1 - self.y0

    def contains(self, x, y, margin: float = 0.0):
        """Vectorised point in rectangle test (NaN coordinates are outside)."""
        x = np.asarray(x)
        y = np.asarray(y)
        with np.errstate(invalid="ignore"):
            return (
                (x >= self.x0 - margin)
                & (x <= self.x1 + margin)
                & (y >= self.y0 - margin)
                & (y <= self.y1 + margin)
            )

    def sample(self, rng: np.random.Generator, shrink: float = 0.2) -> tuple[float, float]:
        """Random point inside the rectangle, kept away from the border."""
        dx, dy = self.width * shrink, self.height * shrink
        return (
            float(rng.uniform(self.x0 + dx, self.x1 - dx)),
            float(rng.uniform(self.y0 + dy, self.y1 - dy)),
        )


@dataclass(frozen=True)
class SceneLayout:
    scene: Scene
    camera_label: str
    zones: dict[str, Rect] = field(default_factory=dict)
    # +1 when the person has their back to the camera, minus 1 when facing it.
    mirror: int = 1
    lower_body_occluded: bool = False
    typical_scale: tuple[float, float] = (0.50, 0.60)
    typical_hip_y: tuple[float, float] = (0.66, 0.70)


LAYOUTS: dict[Scene, SceneLayout] = {
    Scene.AISLE: SceneLayout(
        scene=Scene.AISLE,
        camera_label="AISLE",
        zones={
            "shelf": Rect(0.04, 0.10, 0.96, 0.60),
            "floor": Rect(0.00, 0.60, 1.00, 1.00),
        },
        mirror=1,
        typical_scale=(0.50, 0.60),
        typical_hip_y=(0.66, 0.70),
    ),
    Scene.SELF_CHECKOUT: SceneLayout(
        scene=Scene.SELF_CHECKOUT,
        camera_label="SELF CHECKOUT",
        zones={
            "input": Rect(0.20, 0.58, 0.38, 0.80),
            "scanner": Rect(0.41, 0.58, 0.59, 0.80),
            "bagging": Rect(0.62, 0.58, 0.80, 0.80),
        },
        mirror=-1,
        lower_body_occluded=True,
        typical_scale=(0.80, 0.90),
        typical_hip_y=(0.66, 0.68),
    ),
    Scene.STAFFED_TILL: SceneLayout(
        scene=Scene.STAFFED_TILL,
        camera_label="TILL",
        zones={
            "input": Rect(0.16, 0.58, 0.38, 0.80),
            "scanner": Rect(0.41, 0.58, 0.59, 0.80),
            "bagging": Rect(0.62, 0.58, 0.84, 0.80),
        },
        mirror=-1,
        lower_body_occluded=True,
        typical_scale=(0.80, 0.90),
        typical_hip_y=(0.66, 0.68),
    ),
    Scene.EXIT: SceneLayout(
        scene=Scene.EXIT,
        camera_label="EXIT",
        zones={
            "walkway": Rect(0.00, 0.46, 0.80, 1.00),
            "exit": Rect(0.80, 0.40, 1.00, 1.00),
        },
        mirror=1,
        typical_scale=(0.42, 0.52),
        typical_hip_y=(0.52, 0.70),
    ),
}


def layout_for(scene: Scene | int) -> SceneLayout:
    return LAYOUTS[Scene(int(scene))]
