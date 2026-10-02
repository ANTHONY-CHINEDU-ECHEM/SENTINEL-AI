"""A tiny keyframe language for scripting one tracked person.

Behaviours are written as readable sequences of actions ("reach to the shelf,
hold the item at chest height, move the hand to the bag"). :class:`ClipScript`
records those actions as keyframes and event markers, then :meth:`build`
resamples them at the camera frame rate to produce the latent state that the
kinematics and sensor noise models consume.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ..scenes import SceneLayout
from ..schema import LEFT, N_SIGNALS, RIGHT, SIG


def ease_interp(tq: np.ndarray, key_t, key_v) -> np.ndarray:
    """Piecewise interpolation with smoothstep easing between keyframes."""
    key_t = np.asarray(key_t, dtype=np.float64)
    key_v = np.asarray(key_v, dtype=np.float64)
    if key_v.ndim == 1:
        key_v = key_v[:, None]
    if len(key_t) == 1:
        return np.repeat(key_v[:1], len(tq), axis=0)
    idx = np.clip(np.searchsorted(key_t, tq, side="right") - 1, 0, len(key_t) - 2)
    t0, t1 = key_t[idx], key_t[idx + 1]
    u = np.clip((tq - t0) / np.maximum(t1 - t0, 1e-6), 0.0, 1.0)
    u = u * u * (3.0 - 2.0 * u)
    return key_v[idx] + (key_v[idx + 1] - key_v[idx]) * u[:, None]


@dataclass
class LatentClip:
    """Noise free state of one clip, ready for kinematics and sensor models."""

    hip: np.ndarray  # (T, 2)
    scale: float
    width: np.ndarray  # (T,)
    left_hand: np.ndarray  # (T, 2)
    right_hand: np.ndarray  # (T, 2)
    yaw: np.ndarray  # (T,)
    gait_phase: np.ndarray  # (T,)
    walk: np.ndarray  # (T,)
    mirror: int
    signals: np.ndarray  # (T, S) ground truth channels
    events: list[dict] = field(default_factory=list)
    meta: dict = field(default_factory=dict)


class ClipScript:
    """Record actions for one person and resample them into frames."""

    # Body relative hand poses as (lateral, down) in units of body height.
    REST = (0.12, 0.04)
    CARRY = (0.05, -0.20)
    WAIST = (0.02, -0.07)
    POCKET = (0.09, 0.01)
    PHONE = (0.03, -0.22)

    def __init__(
        self,
        rng: np.random.Generator,
        layout: SceneLayout,
        n_frames: int = 64,
        fps: float = 5.0,
    ) -> None:
        self.rng = rng
        self.layout = layout
        self.n_frames = int(n_frames)
        self.fps = float(fps)
        self.duration = self.n_frames / self.fps
        self.mirror = int(layout.mirror)
        self.scale = float(rng.uniform(*layout.typical_scale))

        self._body: list[tuple[float, float, float]] = []
        self._width: list[tuple[float, float]] = []
        self._hand: dict[int, list[tuple[float, bool, float, float]]] = {LEFT: [], RIGHT: []}
        self._glances: list[tuple[float, float, float]] = []
        self._item_open: dict[int, float | None] = {LEFT: None, RIGHT: None}

        t = self.n_frames
        self.item = np.zeros((2, t), dtype=np.float32)
        self.shelf = np.zeros(t, dtype=np.float32)
        self.container = np.zeros(t, dtype=np.float32)
        self.bagging = np.zeros(t, dtype=np.float32)
        self.pos = np.zeros(t, dtype=np.float32)
        self.match = np.zeros(t, dtype=np.float32)
        self.ratio = np.zeros(t, dtype=np.float32)
        self.eas = np.zeros(t, dtype=np.float32)

        self.bag_rel: tuple[float, float] | None = None
        self.bag_abs: tuple[float, float] | None = None
        self.cont_rel: tuple[float, float] | None = None
        self.cont_abs: tuple[float, float] | None = None

        self.head_wander = float(rng.uniform(0.05, 0.25))
        self.events: list[dict] = []
        self.meta: dict = {}
        # Free form scratch space for behaviour code (hand targets, free hands).
        self.ctx: dict = {}

    # ------------------------------------------------------------------ #
    # Helpers
    # ------------------------------------------------------------------ #
    def frame(self, t: float) -> int:
        return int(round(t * self.fps))

    def hand_side(self, hand: int) -> int:
        """Image x direction (+1 right, minus 1 left) on which a hand hangs."""
        return -self.mirror if hand == LEFT else self.mirror

    def image_hand(self, side: int) -> int:
        """Which hand appears on the given image side (+1 right, minus 1 left)."""
        return LEFT if self.hand_side(LEFT) == side else RIGHT

    def hip_at(self, t: float) -> tuple[float, float]:
        if not self._body:
            raise RuntimeError("set the body position before scripting hands")
        kt = [k[0] for k in self._body]
        kv = [(k[1], k[2]) for k in self._body]
        x, y = ease_interp(np.array([t]), kt, kv)[0]
        return float(x), float(y)

    def log(self, t: float, kind: str, **detail) -> None:
        if 0.0 <= t < self.duration:
            self.events.append({"t": round(float(t), 2), "kind": kind, **detail})

    # ------------------------------------------------------------------ #
    # Body
    # ------------------------------------------------------------------ #
    def body(self, t: float, x: float, y: float) -> None:
        if self._body:
            t = max(t, self._body[-1][0] + 1e-3)
        self._body.append((float(t), float(x), float(y)))

    def width(self, t: float, w: float) -> None:
        if self._width:
            t = max(t, self._width[-1][0] + 1e-3)
        self._width.append((float(t), float(np.clip(w, 0.25, 1.0))))

    def glance(self, t: float, amplitude: float, duration: float = 0.8) -> None:
        self._glances.append((float(t), float(amplitude), float(duration)))
        self.log(t, "glance")

    # ------------------------------------------------------------------ #
    # Hands
    # ------------------------------------------------------------------ #
    def _push_hand(self, hand: int, t: float, rel: bool, x: float, y: float) -> float:
        keys = self._hand[hand]
        if keys:
            t = max(t, keys[-1][0] + 1e-3)
        keys.append((float(t), rel, float(x), float(y)))
        return t

    def hand_abs(self, hand: int, t: float, x: float, y: float) -> float:
        """At time ``t`` the hand is at an absolute image position."""
        return self._push_hand(hand, t, False, x, y)

    def hand_rel(
        self, hand: int, t: float, lateral: float, down: float, side: int | None = None
    ) -> float:
        """At time ``t`` the hand is at a body relative pose (units of height)."""
        side = self.hand_side(hand) if side is None else side
        return self._push_hand(hand, t, True, side * lateral, down)

    def hold(self, hand: int, t: float) -> float:
        """Keep the hand where it was until time ``t``."""
        keys = self._hand[hand]
        if not keys:
            return self.hand_rel(hand, t, *self.REST)
        _, rel, x, y = keys[-1]
        return self._push_hand(hand, t, rel, x, y)

    def rest(self, hand: int, t: float) -> float:
        return self.hand_rel(hand, t, *self.REST)

    def last_hand_time(self, hand: int) -> float:
        keys = self._hand[hand]
        return keys[-1][0] if keys else 0.0

    # ------------------------------------------------------------------ #
    # Item flow and point of sale channels
    # ------------------------------------------------------------------ #
    def item_on(self, hand: int, t: float) -> None:
        if self._item_open[hand] is None:
            self._item_open[hand] = t

    def item_off(self, hand: int, t: float) -> None:
        start = self._item_open[hand]
        if start is None:
            return
        f0 = int(np.clip(self.frame(start), 0, self.n_frames))
        f1 = int(np.clip(max(self.frame(t), f0 + 1), 0, self.n_frames))
        self.item[hand, f0:f1] = 1.0
        self._item_open[hand] = None

    def _impulse(self, channel: np.ndarray, t: float, value: float) -> bool:
        f = self.frame(t)
        if 0 <= f < self.n_frames:
            channel[f] += value
            return True
        return False

    def _step(self, channel: np.ndarray, t: float, value: float) -> None:
        f = self.frame(t)
        if f < self.n_frames:
            channel[max(f, 0) :] += value

    def shelf_event(self, t: float, delta: int) -> None:
        self._impulse(self.shelf, t, delta)

    def container_add(self, t: float, n: int = 1) -> None:
        self._step(self.container, t, n)

    def bagging_add(self, t: float, n: int = 1) -> None:
        self._step(self.bagging, t, n)

    def pos_scan(self, t: float, qty: int = 1, match: float = 0.9, ratio: float = 1.0) -> None:
        f = self.frame(t)
        if 0 <= f < self.n_frames:
            self.pos[f] += qty
            if qty > 0:
                self.match[f] = float(np.clip(match, 0.01, 1.0))
                self.ratio[f] = float(max(ratio, 0.01))
            self.log(t, "scan", qty=int(qty))

    def eas_alarm(self, t: float, duration: float = 1.0) -> None:
        f0 = self.frame(t)
        f1 = self.frame(t + duration)
        if f0 < self.n_frames:
            self.eas[max(f0, 0) : max(f1, f0 + 1)] = 1.0
            self.log(t, "eas_alarm")

    # ------------------------------------------------------------------ #
    # Resampling
    # ------------------------------------------------------------------ #
    def build(self) -> LatentClip:
        if not self._body:
            raise RuntimeError("clip script has no body position")
        tq = np.arange(self.n_frames) / self.fps
        s = self.scale

        hip = ease_interp(tq, [k[0] for k in self._body], [(k[1], k[2]) for k in self._body])
        # Gentle postural sway so a standing person is never perfectly still.
        sway_phase = self.rng.uniform(0, 2 * np.pi, size=2)
        hip[:, 0] += 0.004 * s * np.sin(2 * np.pi * 0.31 * tq + sway_phase[0])
        hip[:, 1] += 0.002 * s * np.sin(2 * np.pi * 0.23 * tq + sway_phase[1])

        if not self._width:
            self._width.append((0.0, 1.0))
        width = ease_interp(tq, [k[0] for k in self._width], [k[1] for k in self._width])[:, 0]

        hands = []
        for hand in (LEFT, RIGHT):
            keys = self._hand[hand]
            if not keys:
                keys = [(0.0, True, self.hand_side(hand) * self.REST[0], self.REST[1])]
            kt = np.array([k[0] for k in keys])
            hip_k = ease_interp(kt, [k[0] for k in self._body], [(k[1], k[2]) for k in self._body])
            offsets = np.array(
                [
                    (k[2] * s, k[3] * s) if k[1] else (k[2] - hip_k[i, 0], k[3] - hip_k[i, 1])
                    for i, k in enumerate(keys)
                ]
            )
            hands.append(hip + ease_interp(tq, kt, offsets))

        # Head yaw: slow wander plus scripted glances.
        ph = self.rng.uniform(0, 2 * np.pi, size=2)
        yaw = self.head_wander * (
            0.7 * np.sin(2 * np.pi * 0.11 * tq + ph[0]) + 0.3 * np.sin(2 * np.pi * 0.29 * tq + ph[1])
        )
        for tc, amp, dur in self._glances:
            u = np.clip((tq - (tc - dur / 2)) / dur, 0.0, 1.0)
            yaw += amp * np.sin(np.pi * u) ** 2
        yaw = np.clip(yaw, -1.0, 1.0)

        # Gait from hip speed (body heights per second).
        vel = np.gradient(hip, axis=0) * self.fps
        speed = np.linalg.norm(vel, axis=1) / s
        walk = np.clip(speed / 0.22, 0.0, 1.0)
        gait_phase = np.cumsum(speed / self.fps / 0.7 * 2 * np.pi)

        # Close any item that is still held when the clip ends.
        for hand in (LEFT, RIGHT):
            if self._item_open[hand] is not None:
                self.item_off(hand, self.duration + 1.0)

        sig = np.zeros((self.n_frames, N_SIGNALS), dtype=np.float32)
        sig[:, SIG["hand_item_l"]] = self.item[LEFT]
        sig[:, SIG["hand_item_r"]] = self.item[RIGHT]
        sig[:, SIG["shelf_delta"]] = self.shelf
        sig[:, SIG["container_count"]] = np.maximum(self.container, 0.0)
        sig[:, SIG["bagging_count"]] = np.maximum(self.bagging, 0.0)
        sig[:, SIG["pos_qty"]] = self.pos
        sig[:, SIG["scan_match"]] = self.match
        sig[:, SIG["price_ratio"]] = self.ratio
        sig[:, SIG["eas_alarm"]] = self.eas
        for prefix, rel, absolute in (
            ("bag", self.bag_rel, self.bag_abs),
            ("cont", self.cont_rel, self.cont_abs),
        ):
            if rel is not None:
                sig[:, SIG[f"{prefix}_x"]] = hip[:, 0] + rel[0] * s
                sig[:, SIG[f"{prefix}_y"]] = hip[:, 1] + rel[1] * s
                sig[:, SIG[f"{prefix}_conf"]] = 1.0
            elif absolute is not None:
                sig[:, SIG[f"{prefix}_x"]] = absolute[0]
                sig[:, SIG[f"{prefix}_y"]] = absolute[1]
                sig[:, SIG[f"{prefix}_conf"]] = 1.0

        return LatentClip(
            hip=hip.astype(np.float32),
            scale=s,
            width=width.astype(np.float32),
            left_hand=hands[LEFT].astype(np.float32),
            right_hand=hands[RIGHT].astype(np.float32),
            yaw=yaw.astype(np.float32),
            gait_phase=gait_phase.astype(np.float32),
            walk=walk.astype(np.float32),
            mirror=self.mirror,
            signals=sig,
            events=sorted(self.events, key=lambda e: e["t"]),
            meta=dict(self.meta),
        )
