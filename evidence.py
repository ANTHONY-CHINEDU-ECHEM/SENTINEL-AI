"""Evidence packets: the facts behind one alert.

The narrator (template or language model) is never shown raw video or free
text. It is given this packet: a short timeline of detected events, the
features that drove the score, and how each compares with normal behaviour on
that camera. If a fact is not in the packet, the narrator cannot claim it.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..features.catalog import FEATURE_LABELS
from ..features.extractor import _dilate, _rising, extract_features, frame_primitives
from ..models.sentinel import SentinelModel
from ..scenes import LAYOUTS
from ..schema import SCENARIO_TITLES, SCENE_TITLES, ClipBatch, Scenario, Scene


def build_timeline(batch: ClipBatch, index: int) -> list[dict]:
    """Detected events for one clip, in time order.

    Every event carries a ``weight`` from 1 (context) to 3 (decisive) that is
    used to choose storyboard frames.
    """
    one = batch.select([index])
    p = frame_primitives(one)
    fps = one.fps
    t = one.n_frames
    scene = Scene(int(one.meta["scene"].iloc[0]))
    events: list[dict] = []

    def add(frame: int, kind: str, text: str, weight: int) -> None:
        events.append({"t": round(frame / fps, 1), "frame": int(frame), "kind": kind, "text": text, "weight": weight})

    shelf = p.shelf[0]
    for f in np.flatnonzero(shelf > 0):
        add(f, "item_taken", "An item was taken from the shelf", 2)
    for f in np.flatnonzero(shelf < 0):
        add(f, "item_returned", "An item was put back on the shelf", 1)

    cont = p.container[0]
    step = np.diff(cont, prepend=cont[0])
    if scene in (Scene.AISLE,):
        for f in np.flatnonzero(step > 0):
            add(f, "basket_up", f"Basket count rose to {int(cont[f])}", 1)

    pos = p.pos[0]
    scan_near = _dilate(pos[None, :] > 0, 10, 3)[0]
    for f in np.flatnonzero(pos > 0):
        m, r = float(p.scan_match[0, f]), float(p.price_ratio[0, f])
        if m < 0.5 and r < 0.6:
            add(f, "suspect_scan", f"Scan registered, but the product match was {m:.2f} and the price was {r:.2f} of the expected price", 3)
        else:
            add(f, "scan", f"Scan registered (product match {m:.2f})", 1)
    for f in np.flatnonzero(pos < 0):
        add(f, "void", "A scanned item was voided", 1)

    if scene in (Scene.SELF_CHECKOUT, Scene.STAFFED_TILL):
        bag = p.bagging[0]
        for f in np.flatnonzero(np.diff(bag, prepend=bag[0]) > 0):
            if not scan_near[f]:
                add(f, "unscanned_bagging", f"Bagging count rose to {int(bag[f])} with no scan in the two seconds before", 3)

    for h, side in enumerate(("left", "right")):
        if scene == Scene.EXIT:
            break  # nothing is shelved, basketed or scanned at the doors, so releases mean little
        vanish = p.vanish[h][0]
        for f in np.flatnonzero(vanish):
            explained = bool(p.explained[0, f])
            lo, hi = max(f - 2, 0), min(f + 3, t)
            at_bag = bool(p.near_bag[h][0, lo:hi].any())
            at_waist = bool(p.near_waist[h][0, lo:hi].any()) and not at_bag
            where = "at a personal bag" if at_bag else ("at the waist" if at_waist else "away from the body")
            if explained:
                if at_bag and scan_near[f]:
                    add(f, "bagged_after_scan", f"Item placed in a personal bag by the {side} hand after a scan", 1)
                continue
            weight = 3 if (at_bag or at_waist) else 2
            add(f, "item_vanished", f"Item in the {side} hand was last seen {where}; no basket, shelf or scan event explains it", weight)

    for f in np.flatnonzero(_rising(p.looking)[0]):
        add(f, "head_turn", "Head turned to the side", 1)

    eas = p.eas[0]
    onset = np.flatnonzero(_rising(eas[None, :])[0] | (eas & (np.arange(t) == 0)))
    for f in onset:
        add(f, "tag_alarm", "Security tag alarm sounded at the doors", 3)

    if scene == Scene.EXIT:
        x0 = LAYOUTS[Scene.EXIT].zones["exit"].x0
        beyond = np.flatnonzero(np.nan_to_num(p.hip_x[0], nan=-1.0) > x0)
        if len(beyond):
            linked = int(one.meta["txn_linked"].iloc[0]) == 1
            text = "Reached the doors with a linked payment" if linked else "Reached the doors with no linked payment"
            add(int(beyond[0]), "exit_crossing", text, 1 if linked else 3)

    return sorted(events, key=lambda e: (e["frame"], -e["weight"]))


def local_drivers(model: SentinelModel, features: pd.Series, scene: int, target_class: int, top_k: int = 6) -> list[dict]:
    """Which features push this clip towards the flagged scenario.

    Each feature in turn is replaced by its typical value for normal clips on
    the same camera type; the drop in the log odds of the flagged scenario is
    that feature's evidence weight. Simple, faithful to the model and easy to
    explain to a reviewer.
    """
    ref = model.reference.get(int(scene))
    if not ref:
        return []
    names = list(model.feature_names)
    base_row = features[names].astype(float)
    grid = pd.DataFrame(np.repeat(base_row.to_numpy()[None, :], len(names) + 1, axis=0), columns=names)
    for i, name in enumerate(names):
        grid.iloc[i + 1, i] = ref["median"][name]
    proba = np.clip(model.predict_proba(grid)[:, int(target_class)], 1e-6, 1 - 1e-6)
    # Work in log odds so that features still register when the score is near one.
    logit = np.log(proba / (1 - proba))
    delta = logit[0] - logit[1:]
    drivers = []
    for i in np.argsort(-delta)[:top_k]:
        if delta[i] <= 0.1:
            break
        name = names[i]
        low, high, value = float(ref["p05"][name]), float(ref["p95"][name]), float(base_row[name])
        typical = float(ref["median"][name])
        margin = max(0.25 * (high - low), 0.05)  # ignore values that sit close to the typical one
        drivers.append(
            {
                "feature": name,
                "label": FEATURE_LABELS[name][0],
                "meaning": FEATURE_LABELS[name][1],
                "value": round(value, 2),
                "typical": round(typical, 2),
                "typical_low": round(low, 2),
                "typical_high": round(high, 2),
                "differs_from_typical": bool(abs(value - typical) > margin),
                "evidence_weight": round(float(delta[i]), 2),
            }
        )
    return drivers


def build_evidence(model: SentinelModel, batch: ClipBatch, index: int, top_k: int = 6) -> dict:
    """Assemble the evidence packet for one clip."""
    one = batch.select([index])
    feats = extract_features(one)
    assess = model.assess(feats).iloc[0]
    meta = one.meta.iloc[0]
    scene = int(meta["scene"])
    flag = int(assess["flag_class"])
    return {
        "clip_id": str(meta["clip_id"]),
        "camera": SCENE_TITLES[Scene(scene)],
        "duration_s": round(one.duration_s, 1),
        "assessment": {
            "tier": str(assess["tier"]),
            "scenario": SCENARIO_TITLES[Scenario(flag)],
            "scenario_id": flag,
            "risk": round(float(assess["risk"]), 3),
            "probability_normal": round(float(assess["p_normal"]), 3),
        },
        "drivers": local_drivers(model, feats.iloc[0], scene, flag, top_k),
        "timeline": [{k: e[k] for k in ("t", "kind", "text")} for e in build_timeline(batch, index)],
        "data_quality": {
            "pose_confidence": round(float(feats["kp_conf_mean"].iloc[0]), 2),
            "arm_hidden_share": round(float(feats["arm_occluded_frac"].iloc[0]), 2),
        },
    }
