"""Interpretable clip level features.

The extractor turns ``(keypoints, signals)`` into about sixty numbers that a
loss prevention analyst can read: how many items left the shelf, how many of
them can be accounted for, whether an item vanished while the hand was at a
bag, how many items reached the bagging area without a scan, and so on.

Everything is vectorised over the batch, tolerant of missing keypoints, and
works on a clip prefix, which is what makes streaming replay possible.
"""

from __future__ import annotations

import warnings
from types import SimpleNamespace

import numpy as np
import pandas as pd

from ..scenes import LAYOUTS
from ..schema import KP, SIG, ClipBatch, Scene

MIN_CONF = 0.30
VANISH_PERSIST_FRAMES = 4

FEATURE_GROUPS: dict[str, tuple[str, ...]] = {
    "pose": (
        "speed_mean",
        "speed_max",
        "still_frac",
        "path_length",
        "width_mean",
        "width_min",
        "width_range",
        "yaw_abs_p90",
        "glance_count",
        "face_visible_frac",
        "reach_count",
        "reach_frac",
        "near_waist_frac",
        "near_bag_frac",
        "hands_together_frac",
        "kp_conf_mean",
        "arm_occluded_frac",
    ),
    "item_flow": (
        "shelf_picks",
        "shelf_returns",
        "min_pick_interval_s",
        "item_frames_frac",
        "item_onsets",
        "mean_hold_s",
        "hand_item_end",
        "vanish_total",
        "vanish_unexplained",
        "vanish_near_bag",
        "vanish_near_waist",
        "vanish_unexplained_near_bag",
        "vanish_unexplained_near_waist",
        "container_start",
        "container_gain",
        "container_max",
        "bagging_gain",
        "unaccounted_items",
        "transfers",
        "bypass_transfers",
        "quick_pass_transfers",
        "scan_dwell_mean_s",
        "bag_present",
        "container_present",
    ),
    "pos": (
        "pos_total",
        "pos_events",
        "pos_voids",
        "unscanned_bagging",
        "flow_gap",
        "scans_per_transfer",
        "scan_match_min",
        "scan_match_mean",
        "price_ratio_min",
        "suspect_scans",
        "low_match_scans",
        "low_price_scans",
        "scan_before_bag",
        "txn_linked",
        "checkout_dwell_s",
        "eas_any",
    ),
    "context": (
        "scene_aisle",
        "scene_self_checkout",
        "scene_staffed_till",
        "scene_exit",
        "reached_exit",
    ),
}
FEATURE_NAMES: tuple[str, ...] = tuple(n for group in FEATURE_GROUPS.values() for n in group)


# --------------------------------------------------------------------------- #
# Small vectorised helpers
# --------------------------------------------------------------------------- #
def _fill(x: np.ndarray, valid: np.ndarray) -> np.ndarray:
    """Forward then backward fill along time. ``x`` is (N, T, D), ``valid`` is (N, T)."""
    n, t = valid.shape
    frames = np.arange(t)[None, :]
    fwd = np.maximum.accumulate(np.where(valid, frames, -1), axis=1)
    bwd = np.minimum.accumulate(np.where(valid, frames, t)[:, ::-1], axis=1)[:, ::-1]
    use = np.where(fwd >= 0, fwd, bwd)
    empty = use >= t
    out = np.take_along_axis(x, np.clip(use, 0, t - 1)[..., None], axis=1).astype(np.float32)
    out[empty] = np.nan
    return out


def _majority3(b: np.ndarray) -> np.ndarray:
    """Three frame majority vote along time."""
    p = np.pad(b.astype(np.int8), ((0, 0), (1, 1)), mode="edge")
    return (p[:, :-2] + p[:, 1:-1] + p[:, 2:]) >= 2


def _dilate(b: np.ndarray, before: int, after: int) -> np.ndarray:
    """True at t when ``b`` is True anywhere in ``[t - before, t + after]``."""
    t = b.shape[1]
    c = np.cumsum(np.pad(b.astype(np.int32), ((0, 0), (before + 1, after))), axis=1)
    return (c[:, before + after + 1 : before + after + 1 + t] - c[:, :t]) > 0


def _median5(x: np.ndarray) -> np.ndarray:
    p = np.pad(x, ((0, 0), (2, 2)), mode="edge")
    stack = np.stack([p[:, i : i + x.shape[1]] for i in range(5)], axis=0)
    return np.median(stack, axis=0)


def _rising(b: np.ndarray) -> np.ndarray:
    out = np.zeros_like(b, dtype=bool)
    out[:, 1:] = b[:, 1:] & ~b[:, :-1]
    return out


def _falling(b: np.ndarray) -> np.ndarray:
    """True on the first frame after a run of True ends."""
    out = np.zeros_like(b, dtype=bool)
    out[:, 1:] = ~b[:, 1:] & b[:, :-1]
    return out


def _nanmean(x: np.ndarray, axis: int = 1) -> np.ndarray:
    with np.errstate(invalid="ignore", divide="ignore"):
        cnt = np.sum(~np.isnan(x), axis=axis)
        return np.where(cnt > 0, np.nansum(x, axis=axis) / np.maximum(cnt, 1), 0.0)


# --------------------------------------------------------------------------- #
# Main entry point
# --------------------------------------------------------------------------- #
def _analyse(batch: ClipBatch) -> tuple[dict[str, np.ndarray], SimpleNamespace]:
    """Shared analysis: returns clip features and the frame level primitives behind them."""
    kp, sig, meta = batch.keypoints, batch.signals, batch.meta
    n, t = kp.shape[:2]
    fps = batch.fps
    dur = t / fps
    scene = meta["scene"].to_numpy().astype(int)
    f: dict[str, np.ndarray] = {}

    conf = kp[..., 2]
    ok = conf >= MIN_CONF

    def joint(name: str) -> tuple[np.ndarray, np.ndarray]:
        i = KP[name]
        return _fill(kp[:, :, i, :2], ok[:, :, i]), ok[:, :, i]

    l_sh, l_sh_ok = joint("left_shoulder")
    r_sh, r_sh_ok = joint("right_shoulder")
    l_hip, _ = joint("left_hip")
    r_hip, _ = joint("right_hip")
    l_wr, l_wr_ok = joint("left_wrist")
    r_wr, r_wr_ok = joint("right_wrist")
    nose, nose_ok = joint("nose")
    sh_mid = 0.5 * (l_sh + r_sh)
    hip_mid = 0.5 * (l_hip + r_hip)

    torso_len = np.linalg.norm(sh_mid - hip_mid, axis=-1)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        s_hat = np.nanmedian(torso_len, axis=1) / 0.29
    s_hat = np.where(np.isfinite(s_hat) & (s_hat > 0.05), s_hat, 0.5).astype(np.float32)
    s = s_hat[:, None]

    # ------------------------------ pose ------------------------------- #
    step = np.linalg.norm(np.diff(hip_mid, axis=1), axis=-1) / s  # body heights per frame
    step = np.nan_to_num(step)
    # A three frame box filter removes keypoint jitter before measuring speed.
    k = np.pad(step, ((0, 0), (1, 1)), mode="edge")
    step_s = (k[:, :-2] + k[:, 1:-1] + k[:, 2:]) / 3.0
    net = np.linalg.norm(np.nan_to_num(hip_mid[:, 5:] - hip_mid[:, :-5]), axis=-1) / s / 5.0 * fps
    f["speed_mean"] = net.mean(axis=1) if t > 5 else step_s.mean(axis=1) * fps
    f["speed_max"] = net.max(axis=1) if t > 5 else step_s.max(axis=1) * fps
    f["still_frac"] = (net < 0.12).mean(axis=1) if t > 5 else np.ones(n)
    f["path_length"] = np.linalg.norm(np.nan_to_num(hip_mid[:, -1] - hip_mid[:, 0]), axis=-1) / s_hat

    width = np.abs(l_sh[..., 0] - r_sh[..., 0]) / s
    width = np.where(l_sh_ok & r_sh_ok, width, np.nan)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        f["width_mean"] = _nanmean(width)
        w_min = np.nanpercentile(np.where(np.isnan(width).all(axis=1, keepdims=True), 0.0, width), 10, axis=1)
        w_max = np.nanpercentile(np.where(np.isnan(width).all(axis=1, keepdims=True), 0.0, width), 90, axis=1)
    f["width_min"] = w_min
    f["width_range"] = w_max - w_min

    yaw = np.nan_to_num(np.where(nose_ok, (nose[..., 0] - sh_mid[..., 0]) / s, np.nan))
    y_pad = np.pad(yaw, ((0, 0), (1, 1)), mode="edge")
    yaw_abs = np.abs((y_pad[:, :-2] + y_pad[:, 1:-1] + y_pad[:, 2:]) / 3.0)
    f["yaw_abs_p90"] = np.percentile(yaw_abs, 90, axis=1)
    looking = yaw_abs > 0.03
    f["glance_count"] = _rising(looking).sum(axis=1) + looking[:, 0]
    f["face_visible_frac"] = (conf[:, :, KP["nose"]] > 0.5).mean(axis=1)

    waist = hip_mid + 0.24 * (sh_mid - hip_mid)
    bag_xy = sig[:, :, [SIG["bag_x"], SIG["bag_y"]]]
    bag_ok = sig[:, :, SIG["bag_conf"]] >= MIN_CONF
    near_waist, near_bag, reach = [], [], []
    for wr, wr_ok, sh in ((l_wr, l_wr_ok, l_sh), (r_wr, r_wr_ok, r_sh)):
        with np.errstate(invalid="ignore"):
            near_waist.append((np.linalg.norm(wr - waist, axis=-1) / s < 0.075) & wr_ok)
            near_bag.append((np.linalg.norm(wr - bag_xy, axis=-1) / s < 0.09) & bag_ok & wr_ok)
            ext = np.linalg.norm(wr - sh, axis=-1) / s
            reach.append((ext > 0.27) & (wr[..., 1] < sh[..., 1] + 0.05 * s) & wr_ok)
    reach_any = _majority3(reach[0] | reach[1])
    f["reach_count"] = _rising(reach_any).sum(axis=1) + reach_any[:, 0]
    f["reach_frac"] = reach_any.mean(axis=1)
    f["near_waist_frac"] = np.maximum(near_waist[0].mean(axis=1), near_waist[1].mean(axis=1))
    f["near_bag_frac"] = np.maximum(near_bag[0].mean(axis=1), near_bag[1].mean(axis=1))
    with np.errstate(invalid="ignore"):
        together = (np.linalg.norm(l_wr - r_wr, axis=-1) / s < 0.10) & l_wr_ok & r_wr_ok
    f["hands_together_frac"] = together.mean(axis=1)
    upper = [KP[k] for k in ("left_shoulder", "right_shoulder", "left_elbow", "right_elbow", "left_wrist", "right_wrist")]
    f["kp_conf_mean"] = conf[:, :, upper].mean(axis=(1, 2))
    f["arm_occluded_frac"] = (~(l_wr_ok & r_wr_ok)).mean(axis=1)

    # ---------------------------- item flow ---------------------------- #
    held = [_majority3(sig[:, :, SIG["hand_item_l"]] > 0.5), _majority3(sig[:, :, SIG["hand_item_r"]] > 0.5)]
    held_any = held[0] | held[1]
    shelf = sig[:, :, SIG["shelf_delta"]]
    picks = np.maximum(shelf, 0).sum(axis=1)
    returns = np.maximum(-shelf, 0).sum(axis=1)
    f["shelf_picks"] = picks
    f["shelf_returns"] = returns

    is_pick = shelf > 0
    frames = np.arange(t)[None, :]
    last_pick = np.maximum.accumulate(np.where(is_pick, frames, -10_000), axis=1)
    prev_pick = np.concatenate([np.full((n, 1), -10_000), last_pick[:, :-1]], axis=1)
    gap = np.where(is_pick & (prev_pick >= 0), frames - prev_pick, 10_000)
    min_gap = gap.min(axis=1)
    f["min_pick_interval_s"] = np.where(min_gap < 10_000, min_gap / fps, dur)

    onsets = _rising(held[0]).sum(axis=1) + _rising(held[1]).sum(axis=1) + held[0][:, 0] + held[1][:, 0]
    f["item_frames_frac"] = held_any.mean(axis=1)
    f["item_onsets"] = onsets
    f["mean_hold_s"] = (held[0].sum(axis=1) + held[1].sum(axis=1)) / np.maximum(onsets, 1) / fps
    f["hand_item_end"] = held_any[:, -3:].any(axis=1).astype(np.float32)

    cont = _median5(sig[:, :, SIG["container_count"]])
    bagc = _median5(sig[:, :, SIG["bagging_count"]])
    edge = min(5, t)
    c_start, c_end = np.median(cont[:, :edge], axis=1), np.median(cont[:, -edge:], axis=1)
    b_start, b_end = np.median(bagc[:, :edge], axis=1), np.median(bagc[:, -edge:], axis=1)
    f["container_start"] = c_start
    f["container_gain"] = c_end - c_start
    f["container_max"] = cont.max(axis=1)
    f["bagging_gain"] = b_end - b_start

    pos = sig[:, :, SIG["pos_qty"]]
    cont_up = _dilate(np.diff(cont, axis=1, prepend=cont[:, :1]) > 0, 6, 3)
    bag_up = _dilate(np.diff(bagc, axis=1, prepend=bagc[:, :1]) > 0, 5, 3)
    shelf_back = _dilate(shelf < 0, 3, 3)
    scanned_recently = _dilate(pos > 0, 10, 3)

    vanish_total = np.zeros(n)
    vanish_unexpl = np.zeros(n)
    v_bag = np.zeros(n)
    v_waist = np.zeros(n)
    v_bag_u = np.zeros(n)
    v_waist_u = np.zeros(n)
    scan_before_bag = np.zeros(n)
    vanish_masks = []
    explained_mask = cont_up | bag_up | shelf_back | scanned_recently
    for h in (0, 1):
        other = 1 - h
        handover = _dilate(_rising(held[other]), 1, 1) & _dilate(together, 2, 2)
        # A vanish only counts when the item stays gone, which removes detector flicker.
        vanish = _falling(held[h]) & ~handover & ~_dilate(held[h], 0, VANISH_PERSIST_FRAMES - 1)
        vanish[:, t - 1 :] = False
        vanish_masks.append(vanish)
        explained = explained_mask
        at_bag = _dilate(near_bag[h], 2, 2)
        at_waist = _dilate(near_waist[h], 2, 2) & ~at_bag
        vanish_total += vanish.sum(axis=1)
        vanish_unexpl += (vanish & ~explained).sum(axis=1)
        v_bag += (vanish & at_bag).sum(axis=1)
        v_waist += (vanish & at_waist).sum(axis=1)
        v_bag_u += (vanish & at_bag & ~explained).sum(axis=1)
        v_waist_u += (vanish & at_waist & ~explained).sum(axis=1)
        scan_before_bag += (vanish & at_bag & scanned_recently).sum(axis=1)
    f["vanish_total"] = vanish_total
    f["vanish_unexplained"] = vanish_unexpl
    f["vanish_near_bag"] = v_bag
    f["vanish_near_waist"] = v_waist
    f["vanish_unexplained_near_bag"] = v_bag_u
    f["vanish_unexplained_near_waist"] = v_waist_u

    pos_total = pos.sum(axis=1)
    f["unaccounted_items"] = np.clip(
        picks - returns - np.maximum(f["container_gain"], 0) - f["hand_item_end"] - np.maximum(pos_total, 0), -6, 12
    )

    # Counter transfers: every run of "this hand holds an item" is checked against the zones.
    counter = np.isin(scene, (int(Scene.SELF_CHECKOUT), int(Scene.STAFFED_TILL)))
    transfers = np.zeros(n)
    bypass = np.zeros(n)
    quick = np.zeros(n)
    scan_frames = np.zeros(n)
    max_runs = 40
    size = n * (max_runs + 1)
    for h, wr in enumerate((l_wr, r_wr)):
        zone_hit = {"input": np.zeros((n, t), dtype=bool), "scanner": np.zeros((n, t), dtype=bool), "bagging": np.zeros((n, t), dtype=bool)}
        for sc_id in (Scene.SELF_CHECKOUT, Scene.STAFFED_TILL):
            rows = scene == int(sc_id)
            if not rows.any():
                continue
            zones = LAYOUTS[sc_id].zones
            zone_hit["input"][rows] = zones["input"].contains(wr[rows, :, 0], wr[rows, :, 1], margin=0.03)
            zone_hit["scanner"][rows] = zones["scanner"].contains(wr[rows, :, 0], wr[rows, :, 1], margin=0.015)
            zone_hit["bagging"][rows] = zones["bagging"].contains(wr[rows, :, 0], wr[rows, :, 1], margin=0.03)
        hold = held[h]
        first = _rising(hold) | (hold & (frames == 0))
        last = hold & ~np.concatenate([hold[:, 1:], np.zeros((n, 1), dtype=bool)], axis=1)
        run_id = np.minimum(np.cumsum(first, axis=1), max_runs) * hold
        flat = (np.arange(n)[:, None] * (max_runs + 1) + run_id).ravel()

        def per_run(values: np.ndarray, flat: np.ndarray = flat) -> np.ndarray:
            return np.bincount(flat, weights=values.ravel().astype(np.float64), minlength=size).reshape(n, -1)[:, 1:]

        run_len = per_run(hold)
        run_scan = per_run(hold & zone_hit["scanner"])
        run_to_bag = per_run(last & _dilate(zone_hit["bagging"], 1, 2)) > 0
        run_from_input = per_run(first & _dilate(zone_hit["input"], 2, 1)) > 0
        delivered = (run_len >= 2) & run_to_bag & counter[:, None]
        transfers += delivered.sum(axis=1)
        bypass += (delivered & (run_scan == 0)).sum(axis=1)
        quick += ((run_len >= 2) & run_from_input & (run_scan <= 1) & counter[:, None]).sum(axis=1)
        scan_frames += (hold & zone_hit["scanner"]).sum(axis=1) * counter
    f["transfers"] = transfers
    f["bypass_transfers"] = bypass
    f["quick_pass_transfers"] = quick
    f["scan_dwell_mean_s"] = scan_frames / np.maximum(transfers, 1) / fps
    f["bag_present"] = (sig[:, :, SIG["bag_conf"]] >= MIN_CONF).mean(axis=1)
    f["container_present"] = (sig[:, :, SIG["cont_conf"]] >= MIN_CONF).mean(axis=1)

    # ------------------------------- POS ------------------------------- #
    scans = pos > 0
    match = sig[:, :, SIG["scan_match"]]
    ratio = sig[:, :, SIG["price_ratio"]]
    n_scans = scans.sum(axis=1)
    f["pos_total"] = pos_total
    f["pos_events"] = n_scans
    f["pos_voids"] = (pos < 0).sum(axis=1)
    f["unscanned_bagging"] = f["bagging_gain"] - pos_total
    f["flow_gap"] = -f["container_gain"] - pos_total
    f["scans_per_transfer"] = n_scans / np.maximum(transfers, 1)
    f["scan_match_min"] = np.where(n_scans > 0, np.where(scans, match, 9.0).min(axis=1), 1.0)
    f["scan_match_mean"] = np.where(n_scans > 0, (match * scans).sum(axis=1) / np.maximum(n_scans, 1), 1.0)
    f["price_ratio_min"] = np.where(n_scans > 0, np.where(scans, ratio, 9.0).min(axis=1), 1.0)
    f["suspect_scans"] = (scans & (match < 0.5) & (ratio < 0.6)).sum(axis=1)
    f["low_match_scans"] = (scans & (match < 0.5)).sum(axis=1)
    f["low_price_scans"] = (scans & (ratio < 0.6)).sum(axis=1)
    f["scan_before_bag"] = scan_before_bag
    f["txn_linked"] = meta["txn_linked"].to_numpy().astype(np.float32)
    f["checkout_dwell_s"] = meta["checkout_dwell_s"].to_numpy().astype(np.float32)
    f["eas_any"] = (sig[:, :, SIG["eas_alarm"]] > 0.5).any(axis=1).astype(np.float32)

    # ----------------------------- context ----------------------------- #
    for sc_id, name in (
        (Scene.AISLE, "scene_aisle"),
        (Scene.SELF_CHECKOUT, "scene_self_checkout"),
        (Scene.STAFFED_TILL, "scene_staffed_till"),
        (Scene.EXIT, "scene_exit"),
    ):
        f[name] = (scene == int(sc_id)).astype(np.float32)
    exit_x0 = LAYOUTS[Scene.EXIT].zones["exit"].x0
    with np.errstate(invalid="ignore"):
        f["reached_exit"] = ((np.nanmax(np.where(np.isnan(hip_mid[..., 0]), -1.0, hip_mid[..., 0]), axis=1) > exit_x0) & (scene == int(Scene.EXIT))).astype(np.float32)

    prim = SimpleNamespace(
        held=held,
        near_bag=near_bag,
        near_waist=near_waist,
        vanish=vanish_masks,
        explained=explained_mask,
        looking=looking,
        container=cont,
        bagging=bagc,
        shelf=shelf,
        pos=pos,
        scan_match=match,
        price_ratio=ratio,
        eas=sig[:, :, SIG["eas_alarm"]] > 0.5,
        hip_x=hip_mid[..., 0],
        body_scale=s_hat,
    )
    return f, prim


def extract_features(batch: ClipBatch) -> pd.DataFrame:
    """Compute the feature table for a batch. One row per clip."""
    f, _ = _analyse(batch)
    table = pd.DataFrame({name: np.asarray(f[name], dtype=np.float32) for name in FEATURE_NAMES})
    return table.replace([np.inf, -np.inf], 0.0).fillna(0.0)


def frame_primitives(batch: ClipBatch) -> SimpleNamespace:
    """Frame level detections (item held, vanish events, glances) used for timelines."""
    return _analyse(batch)[1]
