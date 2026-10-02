"""Forward kinematics for a 17 keypoint COCO skeleton in the image plane.

The simulator drives a small latent state (hip position, body scale, apparent
torso width, two hand targets, head yaw and gait phase) and this module turns
that state into keypoints. The model is deliberately simple, but it respects
reach: arms are solved with two link inverse kinematics so a wrist can never be
further from the shoulder than the arm is long, and a bent elbow is drawn
foreshortened, the way a camera sees it.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..schema import KP, N_KEYPOINTS


@dataclass(frozen=True)
class BodyProportions:
    """Segment sizes as a fraction of standing height."""

    shoulder_above_hip: float = 0.29
    nose_above_hip: float = 0.415
    eye_above_hip: float = 0.432
    ear_above_hip: float = 0.42
    shoulder_half_width: float = 0.11
    hip_half_width: float = 0.07
    arm_length: float = 0.34
    knee_below_hip: float = 0.245
    ankle_below_hip: float = 0.48
    eye_half_gap: float = 0.018
    ear_half_gap: float = 0.045


BODY = BodyProportions()


def _solve_arm(
    shoulder: np.ndarray, target: np.ndarray, arm: np.ndarray, outward: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Two link inverse kinematics with equal upper and lower arm lengths.

    Parameters
    ----------
    shoulder, target : (..., 2) arrays
    arm : (..., 1) total arm length
    outward : (..., 1) sign of the image x direction that points away from the torso

    Returns
    -------
    elbow, wrist : (..., 2) arrays
    """
    v = target - shoulder
    d = np.linalg.norm(v, axis=-1, keepdims=True)
    d_safe = np.maximum(d, 1e-6)
    reach = np.minimum(d_safe, arm * 0.999)
    direction = v / d_safe
    wrist = shoulder + direction * reach
    half = arm / 2.0
    bend = np.sqrt(np.maximum(half**2 - (reach / 2.0) ** 2, 0.0))
    perp = np.stack([-direction[..., 1], direction[..., 0]], axis=-1)
    # Elbows bend away from the torso and, when ambiguous, downward.
    score = perp[..., :1] * outward + 0.35 * perp[..., 1:2]
    sign = np.where(score >= 0, 1.0, -1.0)
    # A bent arm folds partly out of the image plane, so only part of the bend is
    # visible sideways; the rest shows up as the elbow dropping towards the floor.
    down = np.zeros_like(direction)
    down[..., 1] = 1.0
    elbow = shoulder + direction * (reach / 2.0) + perp * sign * bend * 0.5 + down * bend * 0.45
    return elbow, wrist


def build_skeleton(
    hip: np.ndarray,
    scale: np.ndarray,
    width: np.ndarray,
    left_hand: np.ndarray,
    right_hand: np.ndarray,
    yaw: np.ndarray,
    gait_phase: np.ndarray,
    walk: np.ndarray,
    mirror: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Turn latent body state into keypoints.

    Parameters
    ----------
    hip : (N, T, 2) hip centre
    scale : (N,) standing height in image units
    width : (N, T) apparent torso width factor, 1 when square on to the camera
    left_hand, right_hand : (N, T, 2) hand targets in image coordinates
    yaw : (N, T) head yaw in the range minus 1 to 1
    gait_phase : (N, T) radians
    walk : (N, T) walking intensity from 0 (standing) to 1
    mirror : (N,) +1 when the back is to the camera, minus 1 when facing it

    Returns
    -------
    xy : (N, T, 17, 2) keypoint coordinates
    visibility : (N, T, 17) prior detection confidence before sensor noise
    """
    n, t = hip.shape[:2]
    s = scale[:, None].astype(np.float32)
    m = mirror[:, None].astype(np.float32)
    cx, cy = hip[..., 0], hip[..., 1]
    xy = np.zeros((n, t, N_KEYPOINTS, 2), dtype=np.float32)

    def put(name: str, x: np.ndarray, y: np.ndarray) -> None:
        xy[:, :, KP[name], 0] = x
        xy[:, :, KP[name], 1] = y

    # Torso. The person's left appears on image left when their back is to the camera.
    sh_y = cy - BODY.shoulder_above_hip * s
    sh_hw = BODY.shoulder_half_width * s * width
    hip_hw = BODY.hip_half_width * s * width
    put("left_shoulder", cx - m * sh_hw, sh_y)
    put("right_shoulder", cx + m * sh_hw, sh_y)
    put("left_hip", cx - m * hip_hw, cy)
    put("right_hip", cx + m * hip_hw, cy)

    # Head. Yaw slides the face keypoints sideways relative to the neck.
    nose_x = cx + yaw * 0.075 * s
    put("nose", nose_x, cy - BODY.nose_above_hip * s)
    eye_gap = BODY.eye_half_gap * s * (1.0 - 0.5 * np.abs(yaw))
    put("left_eye", nose_x - m * eye_gap, cy - BODY.eye_above_hip * s)
    put("right_eye", nose_x + m * eye_gap, cy - BODY.eye_above_hip * s)
    ear_shift = -yaw * 0.012 * s
    put("left_ear", cx - m * BODY.ear_half_gap * s + ear_shift, cy - BODY.ear_above_hip * s)
    put("right_ear", cx + m * BODY.ear_half_gap * s + ear_shift, cy - BODY.ear_above_hip * s)

    # Arms.
    arm = (BODY.arm_length * s)[..., None]
    for side, target, outward in (
        ("left", left_hand, -m),
        ("right", right_hand, m),
    ):
        shoulder = xy[:, :, KP[f"{side}_shoulder"]]
        elbow, wrist = _solve_arm(shoulder, target, arm, outward[..., None])
        xy[:, :, KP[f"{side}_elbow"]] = elbow
        xy[:, :, KP[f"{side}_wrist"]] = wrist

    # Legs with a simple alternating gait.
    stride = 0.07 * s * walk * np.sin(gait_phase)
    lift_l = 0.025 * s * walk * np.clip(np.cos(gait_phase), 0.0, 1.0)
    lift_r = 0.025 * s * walk * np.clip(np.cos(gait_phase + np.pi), 0.0, 1.0)
    l_hip_x = xy[:, :, KP["left_hip"], 0]
    r_hip_x = xy[:, :, KP["right_hip"], 0]
    ankle_y = cy + BODY.ankle_below_hip * s
    knee_y = cy + BODY.knee_below_hip * s
    put("left_ankle", l_hip_x + stride, ankle_y - lift_l)
    put("right_ankle", r_hip_x - stride, ankle_y - lift_r)
    put("left_knee", l_hip_x + 0.55 * stride + 0.015 * s * walk, knee_y - 0.5 * lift_l)
    put("right_knee", r_hip_x - 0.55 * stride + 0.015 * s * walk, knee_y - 0.5 * lift_r)

    # Prior visibility. With the back to the camera the face is only seen in a glance.
    visibility = np.full((n, t, N_KEYPOINTS), 0.92, dtype=np.float32)
    face = [KP[k] for k in ("nose", "left_eye", "right_eye")]
    back_view = (m > 0).astype(np.float32)
    face_vis = back_view * (0.22 + 0.65 * np.abs(yaw)) + (1.0 - back_view) * 0.92
    visibility[:, :, face] = face_vis[..., None]
    return xy, visibility
