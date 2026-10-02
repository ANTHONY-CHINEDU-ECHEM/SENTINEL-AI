"""Behaviour library: 25 benign subtypes and 7 theft scenarios.

Each behaviour is a plain function that scripts one :class:`ClipScript`. The
benign side is deliberately rich in *hard negatives* (behaviours that share
surface features with theft but are legitimate) because a loss prevention
model is only useful if it stays quiet when an honest shopper checks a phone,
uses scan and go, or carries a product in their hand.

Subtype weights are relative frequencies in the generated corpus. Theft is
enriched to roughly one clip in seven so that every scenario has enough
examples to learn from; evaluation later reweights to a realistic prevalence.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from ..schema import LEFT, RIGHT, Scenario, Scene
from .script import ClipScript


@dataclass(frozen=True)
class Subtype:
    name: str
    scene: Scene
    scenario: Scenario
    weight: float
    fn: Callable[[ClipScript], None]
    description: str


REGISTRY: dict[str, Subtype] = {}


def register(name: str, scene: Scene, scenario: Scenario, weight: float, description: str):
    def deco(fn: Callable[[ClipScript], None]):
        REGISTRY[name] = Subtype(name, scene, scenario, weight, fn, description)
        return fn

    return deco


def _value(rng: np.random.Generator, median: float, sigma: float = 0.55) -> float:
    """Lognormal value at risk in pounds."""
    return float(np.round(median * np.exp(rng.normal(0.0, sigma)), 2))


def _u(rng: np.random.Generator, lo: float, hi: float) -> float:
    return float(rng.uniform(lo, hi))


# =========================================================================== #
# Aisle building blocks
# =========================================================================== #
def _aisle_setup(
    sc: ClipScript, container: str | None, bag: str | None, walk_in: float = 0.4
) -> float:
    """Place the shopper in front of the shelf. Returns the time they are ready."""
    rng = sc.rng
    x0 = _u(rng, 0.25, 0.75)
    y0 = _u(rng, *sc.layout.typical_hip_y)
    if rng.random() < walk_in:
        side = rng.choice([-1, 1])
        xs = float(np.clip(x0 + side * _u(rng, 0.18, 0.30), 0.03, 0.97))
        t_arr = _u(rng, 1.2, 2.4)
        side_on = _u(rng, 0.40, 0.60)
        sc.body(0.0, xs, y0)
        sc.body(t_arr, x0, y0)
        sc.width(0.0, side_on)
        sc.width(t_arr - 0.3, side_on)
        sc.width(t_arr + 0.3, _u(rng, 0.88, 1.0))
        t = t_arr + _u(rng, 0.2, 0.6)
    else:
        sc.body(0.0, x0, y0)
        sc.width(0.0, _u(rng, 0.85, 1.0))
        t = _u(rng, 0.2, 1.0)

    free = [LEFT, RIGHT]
    cont_side = int(rng.choice([-1, 1]))
    if container == "basket":
        sc.cont_rel = (cont_side * 0.17, 0.12)
        sc.ctx["cont_hand"] = (cont_side * 0.17, 0.07)
        carrier = sc.image_hand(cont_side)
        free.remove(carrier)
        sc.hand_rel(carrier, 0.0, 0.15, 0.03, side=cont_side)
    elif container == "trolley":
        sc.cont_rel = (cont_side * 0.30, 0.16)
        sc.ctx["cont_hand"] = (cont_side * 0.25, 0.05)
    if bag is not None:
        bag_side = -cont_side if container == "basket" else int(rng.choice([-1, 1]))
        sc.bag_rel = (bag_side * 0.14, -0.03) if bag == "shoulder" else (bag_side * 0.15, 0.10)
        sc.ctx["bag_hand"] = sc.bag_rel
    for hand in free:
        sc.rest(hand, 0.0)
    sc.ctx["free"] = free
    sc.ctx["container"] = container
    return t


def _pick(sc: ClipScript, t: float, hand: int, quick: bool = False) -> float:
    """Reach to the shelf and take one item. Returns the grasp time."""
    rng = sc.rng
    t = max(t, sc.last_hand_time(hand))
    hx, _ = sc.hip_at(t)
    tx = hx + sc.hand_side(hand) * _u(rng, 0.02, 0.11)
    ty = _u(rng, 0.30, 0.50)
    reach = _u(rng, 0.30, 0.45) if quick else _u(rng, 0.45, 0.80)
    sc.hand_abs(hand, t + reach, tx, ty)
    tg = t + reach + (_u(rng, 0.05, 0.12) if quick else _u(rng, 0.10, 0.30))
    sc.hand_abs(hand, tg, tx, ty)
    sc.shelf_event(tg, +1)
    sc.item_on(hand, tg)
    sc.log(tg, "pick", hand=hand)
    return tg


def _touch(sc: ClipScript, t: float, hand: int) -> float:
    """Reach to the shelf without taking anything."""
    rng = sc.rng
    t = max(t, sc.last_hand_time(hand))
    hx, _ = sc.hip_at(t)
    tx = hx + sc.hand_side(hand) * _u(rng, 0.02, 0.11)
    ty = _u(rng, 0.30, 0.50)
    t1 = t + _u(rng, 0.5, 0.8)
    sc.hand_abs(hand, t1, tx, ty)
    t2 = t1 + _u(rng, 0.3, 1.2)
    sc.hand_abs(hand, t2, tx, ty)
    sc.rest(hand, t2 + 0.5)
    return t2 + 0.5


def _inspect(sc: ClipScript, t: float, hand: int, dur: float) -> float:
    sc.hand_rel(hand, t + 0.4, *sc.CARRY)
    sc.hand_rel(hand, t + 0.4 + dur, *sc.CARRY)
    return t + 0.4 + dur


def _to_container(sc: ClipScript, t: float, hand: int) -> float:
    rng = sc.rng
    cx, cy = sc.ctx["cont_hand"]
    t1 = t + _u(rng, 0.4, 0.7)
    sc.hand_rel(hand, t1, cx + _u(rng, -0.02, 0.02), cy + _u(rng, -0.02, 0.02), side=1)
    sc.item_off(hand, t1)
    sc.container_add(t1 + _u(rng, 0.0, 0.2), 1)
    sc.log(t1, "to_container", hand=hand)
    sc.hold(hand, t1 + 0.2)
    sc.rest(hand, t1 + 0.6)
    return t1 + 0.6


def _to_shelf(sc: ClipScript, t: float, hand: int) -> float:
    rng = sc.rng
    hx, _ = sc.hip_at(t)
    tx = hx + sc.hand_side(hand) * _u(rng, 0.02, 0.11)
    ty = _u(rng, 0.30, 0.50)
    t1 = t + _u(rng, 0.45, 0.8)
    sc.hand_abs(hand, t1, tx, ty)
    sc.shelf_event(t1, -1)
    sc.item_off(hand, t1)
    sc.log(t1, "return_to_shelf", hand=hand)
    sc.hold(hand, t1 + 0.15)
    sc.rest(hand, t1 + 0.65)
    return t1 + 0.65


def _to_bag(sc: ClipScript, t: float, hand: int, linger: float, quick: bool = False) -> float:
    """Move the held item into the personal bag. Returns the moment it disappears."""
    rng = sc.rng
    bx, by = sc.ctx["bag_hand"]
    t1 = t + (_u(rng, 0.28, 0.40) if quick else _u(rng, 0.35, 0.70))
    sc.hand_rel(hand, t1, bx + _u(rng, -0.015, 0.015), by + _u(rng, -0.015, 0.015), side=1)
    t_gone = t1 - _u(rng, 0.0, 0.08)
    sc.item_off(hand, t_gone)
    sc.log(t_gone, "to_bag", hand=hand)
    sc.hold(hand, t1 + linger)
    if not quick:
        sc.rest(hand, t1 + linger + 0.45)
    return t_gone


def _to_waist(sc: ClipScript, t: float, hand: int, linger: float) -> float:
    """Tuck the held item under clothing at the waistband."""
    rng = sc.rng
    t1 = t + _u(rng, 0.35, 0.65)
    sc.hand_rel(hand, t1, sc.WAIST[0] + _u(rng, -0.015, 0.02), sc.WAIST[1] + _u(rng, -0.02, 0.02))
    t_gone = t1 - _u(rng, 0.0, 0.12)
    sc.item_off(hand, t_gone)
    sc.log(t_gone, "to_clothing", hand=hand)
    sc.hold(hand, t1 + linger)
    sc.rest(hand, t1 + linger + 0.45)
    return t_gone


def _nervous_glances(sc: ClipScript, t0: float, t1: float, n: int) -> None:
    rng = sc.rng
    for _ in range(n):
        sc.glance(_u(rng, t0, max(t1, t0 + 0.2)), rng.choice([-1, 1]) * _u(rng, 0.5, 1.0), _u(rng, 0.5, 0.9))


def _turn_away(sc: ClipScript, t: float) -> None:
    rng = sc.rng
    sc.width(t - 0.2, 0.95)
    sc.width(t + 0.3, _u(rng, 0.45, 0.70))
    sc.width(t + 2.2, _u(rng, 0.45, 0.70))
    sc.width(t + 2.8, 0.95)


def _walk_off(sc: ClipScript, t: float) -> None:
    rng = sc.rng
    if t > sc.duration - 1.0:
        return
    hx, hy = sc.hip_at(t)
    dest = float(np.clip(hx + rng.choice([-1, 1]) * _u(rng, 0.2, 0.4), 0.02, 0.98))
    sc.body(t, hx, hy)
    sc.body(t + abs(dest - hx) / _u(rng, 0.10, 0.16), dest, hy)
    sc.width(t, 0.95)
    sc.width(t + 0.4, _u(rng, 0.40, 0.60))


def _pick_container(rng: np.random.Generator, options: dict[str | None, float]) -> str | None:
    keys = list(options)
    p = np.array([options[k] for k in keys], dtype=float)
    return keys[int(rng.choice(len(keys), p=p / p.sum()))]


# =========================================================================== #
# Aisle: benign behaviours
# =========================================================================== #
@register("browse", Scene.AISLE, Scenario.NORMAL, 9.0, "Looks along the shelf, may touch products")
def browse(sc: ClipScript) -> None:
    rng = sc.rng
    container = _pick_container(rng, {None: 0.3, "basket": 0.45, "trolley": 0.25})
    t = _aisle_setup(sc, container, "shoulder" if rng.random() < 0.35 else None)
    sc.head_wander = _u(rng, 0.20, 0.55)
    for _ in range(int(rng.integers(0, 3))):
        t = _touch(sc, t + _u(rng, 0.2, 1.5), int(rng.choice(sc.ctx["free"])))
    if rng.random() < 0.5:
        _walk_off(sc, max(t, _u(rng, 5.0, 10.0)))


@register("pick_to_basket", Scene.AISLE, Scenario.NORMAL, 16.0, "Takes items and puts them in a basket or trolley")
def pick_to_basket(sc: ClipScript) -> None:
    rng = sc.rng
    t = _aisle_setup(sc, _pick_container(rng, {"basket": 0.6, "trolley": 0.4}), "shoulder" if rng.random() < 0.4 else None)
    for _ in range(int(rng.integers(1, 5))):
        if t > sc.duration - 2.5:
            break
        hand = int(rng.choice(sc.ctx["free"]))
        tg = _pick(sc, t, hand)
        te = _inspect(sc, tg, hand, _u(rng, 0.3, 1.5))
        t = _to_container(sc, te, hand) + _u(rng, 0.2, 1.2)
    if rng.random() < 0.35:
        _walk_off(sc, t)


@register("pick_and_return", Scene.AISLE, Scenario.NORMAL, 6.0, "Inspects an item then puts it back")
def pick_and_return(sc: ClipScript) -> None:
    rng = sc.rng
    container = _pick_container(rng, {None: 0.3, "basket": 0.45, "trolley": 0.25})
    t = _aisle_setup(sc, container, "shoulder" if rng.random() < 0.4 else None)
    hand = int(rng.choice(sc.ctx["free"]))
    tg = _pick(sc, t, hand)
    te = _inspect(sc, tg, hand, _u(rng, 1.0, 3.0))
    t = _to_shelf(sc, te, hand) + _u(rng, 0.3, 1.0)
    if container and rng.random() < 0.5 and t < sc.duration - 3.0:
        tg = _pick(sc, t, hand)
        te = _inspect(sc, tg, hand, _u(rng, 0.3, 1.0))
        _to_container(sc, te, hand)


@register("phone_or_pocket", Scene.AISLE, Scenario.NORMAL, 5.0, "Takes a phone from a pocket and puts it back")
def phone_or_pocket(sc: ClipScript) -> None:
    rng = sc.rng
    container = _pick_container(rng, {None: 0.35, "basket": 0.4, "trolley": 0.25})
    t = _aisle_setup(sc, container, "shoulder" if rng.random() < 0.3 else None)
    hand = int(rng.choice(sc.ctx["free"]))
    if container and rng.random() < 0.5:
        tg = _pick(sc, t, hand)
        t = _to_container(sc, _inspect(sc, tg, hand, _u(rng, 0.3, 1.0)), hand) + _u(rng, 0.2, 0.8)
    sc.hand_rel(hand, t + 0.5, *sc.POCKET)
    sc.hold(hand, t + 0.8)
    seen_as_item = rng.random() < 0.35  # the detector sometimes labels a phone as merchandise
    if seen_as_item:
        sc.item_on(hand, t + 0.8)
    sc.hand_rel(hand, t + 1.3, *sc.PHONE)
    t_end = t + 1.3 + _u(rng, 2.0, 5.0)
    sc.hold(hand, t_end)
    sc.hand_rel(hand, t_end + 0.5, *sc.POCKET)
    if seen_as_item:
        sc.item_off(hand, t_end + 0.45)
    sc.rest(hand, t_end + 1.0)
    sc.head_wander = _u(rng, 0.02, 0.10)


@register("own_bag_adjust", Scene.AISLE, Scenario.NORMAL, 4.0, "Reaches into own bag for a list, wallet or phone")
def own_bag_adjust(sc: ClipScript) -> None:
    rng = sc.rng
    container = _pick_container(rng, {None: 0.3, "basket": 0.45, "trolley": 0.25})
    t = _aisle_setup(sc, container, str(rng.choice(["shoulder", "tote"])))
    hand = int(rng.choice(sc.ctx["free"]))
    if container and rng.random() < 0.4:
        tg = _pick(sc, t, hand)
        t = _to_container(sc, _inspect(sc, tg, hand, _u(rng, 0.3, 1.0)), hand) + _u(rng, 0.2, 0.8)
    bx, by = sc.ctx["bag_hand"]
    t1 = t + _u(rng, 0.4, 0.7)
    sc.hand_rel(hand, t1, bx, by, side=1)
    t2 = t1 + _u(rng, 0.6, 2.0)
    sc.hold(hand, t2)
    if rng.random() < 0.35:  # takes out a list or phone that the detector may see as an item
        sc.item_on(hand, t2)
        sc.hand_rel(hand, t2 + 0.5, *sc.PHONE)
        t3 = t2 + 0.5 + _u(rng, 1.5, 3.0)
        sc.hold(hand, t3)
        sc.hand_rel(hand, t3 + 0.5, bx, by, side=1)
        sc.item_off(hand, t3 + 0.45)
        t2 = t3 + 0.8
        sc.hold(hand, t2)
    sc.rest(hand, t2 + 0.5)


@register("scan_and_go", Scene.AISLE, Scenario.NORMAL, 3.0, "Scans items with a handset and packs them into own bag")
def scan_and_go(sc: ClipScript) -> None:
    rng = sc.rng
    container = _pick_container(rng, {None: 0.6, "trolley": 0.4})
    t = _aisle_setup(sc, container, "tote")
    for _ in range(int(rng.integers(1, 4))):
        if t > sc.duration - 3.0:
            break
        hand = int(rng.choice(sc.ctx["free"]))
        other = RIGHT if hand == LEFT else LEFT
        tg = _pick(sc, t, hand)
        te = _inspect(sc, tg, hand, _u(rng, 0.5, 1.2))
        if other in sc.ctx["free"]:
            sc.hand_rel(other, max(te - 0.4, sc.last_hand_time(other)), *sc.CARRY)
            sc.hold(other, te + 0.1)
            sc.rest(other, te + 0.6)
        sc.pos_scan(te - _u(rng, 0.05, 0.3), 1, _u(rng, 0.8, 0.98), rng.normal(1.0, 0.02))
        _to_bag(sc, te, hand, _u(rng, 0.2, 0.6))
        t = sc.last_hand_time(hand) + _u(rng, 0.2, 1.0)


@register("staff_restock", Scene.AISLE, Scenario.NORMAL, 3.0, "Colleague moves stock from a cage onto the shelf")
def staff_restock(sc: ClipScript) -> None:
    rng = sc.rng
    t = _aisle_setup(sc, "trolley", None, walk_in=0.1)
    sc.container_add(0.0, int(rng.integers(8, 15)))
    cx, cy = sc.ctx["cont_hand"]
    i = 0
    while t < sc.duration - 1.6:
        hand = sc.ctx["free"][i % 2]
        t0 = max(t, sc.last_hand_time(hand))
        t1 = t0 + _u(rng, 0.35, 0.55)
        sc.hand_rel(hand, t1, cx, cy, side=1)
        sc.item_on(hand, t1)
        sc.container_add(t1, -1)
        hx, _ = sc.hip_at(t1)
        t2 = t1 + _u(rng, 0.45, 0.75)
        sc.hand_abs(hand, t2, hx + sc.hand_side(hand) * _u(rng, 0.02, 0.10), _u(rng, 0.30, 0.50))
        sc.shelf_event(t2, -1)
        sc.item_off(hand, t2)
        sc.log(t2, "restock", hand=hand)
        sc.hold(hand, t2 + 0.15)
        t = t1 + _u(rng, 0.3, 0.7)
        i += 1


@register("bulk_buyer", Scene.AISLE, Scenario.NORMAL, 3.0, "Loads many items quickly into a trolley")
def bulk_buyer(sc: ClipScript) -> None:
    rng = sc.rng
    t = _aisle_setup(sc, "trolley", "shoulder" if rng.random() < 0.3 else None)
    for i in range(int(rng.integers(4, 9))):
        if t > sc.duration - 1.8:
            break
        hand = sc.ctx["free"][i % 2]
        tg = _pick(sc, t, hand, quick=True)
        _to_container(sc, tg + _u(rng, 0.0, 0.2), hand)
        t = tg + _u(rng, 0.25, 0.7)


@register("carry_in_hand", Scene.AISLE, Scenario.NORMAL, 4.0, "No basket: carries one or two items held against the body")
def carry_in_hand(sc: ClipScript) -> None:
    rng = sc.rng
    t = _aisle_setup(sc, None, "shoulder" if rng.random() < 0.35 else None)
    hands = list(sc.ctx["free"])
    rng.shuffle(hands)
    for hand in hands[: int(rng.integers(1, 3))]:
        tg = _pick(sc, t, hand)
        te = _inspect(sc, tg, hand, _u(rng, 0.3, 1.2))
        sc.hand_rel(hand, te + 0.4, 0.03 + _u(rng, 0.0, 0.03), _u(rng, -0.17, -0.10))
        sc.hold(hand, sc.duration + 1.0)
        t = te + _u(rng, 0.4, 1.2)
    if rng.random() < 0.6:
        _walk_off(sc, t + _u(rng, 0.3, 2.0))


# =========================================================================== #
# Aisle: theft scenarios
# =========================================================================== #
@register("concealment_bag", Scene.AISLE, Scenario.CONCEALMENT_BAG, 2.2, "Takes an item and hides it in a personal bag")
def concealment_bag(sc: ClipScript) -> None:
    rng = sc.rng
    container = _pick_container(rng, {None: 0.35, "basket": 0.5, "trolley": 0.15})
    t = _aisle_setup(sc, container, str(rng.choice(["shoulder", "tote"])))
    sc.head_wander = _u(rng, 0.08, 0.30)
    free = sc.ctx["free"]
    if container and rng.random() < 0.5:  # decoy purchase
        hand = int(rng.choice(free))
        tg = _pick(sc, t, hand)
        t = _to_container(sc, _inspect(sc, tg, hand, _u(rng, 0.3, 1.0)), hand) + _u(rng, 0.2, 0.8)
    t = max(t, _u(rng, 0.5, 4.5))
    first = None
    for _ in range(1 + int(rng.random() < 0.3)):
        if t > sc.duration - 3.0:
            break
        hand = int(rng.choice(free))
        tg = _pick(sc, t, hand)
        te = _inspect(sc, tg, hand, _u(rng, 0.3, 1.4))
        if rng.random() < 0.65:
            _nervous_glances(sc, tg, te + 0.3, int(rng.integers(1, 4)))
        if rng.random() < 0.4:
            _turn_away(sc, te)
        t_gone = _to_bag(sc, te, hand, _u(rng, 0.3, 0.9))
        first = t_gone if first is None else first
        t = sc.last_hand_time(hand) + _u(rng, 0.3, 1.0)
    sc.meta["event_time_s"] = first if first is not None else np.nan
    sc.meta["value_at_risk_gbp"] = _value(rng, 24.0)
    if rng.random() < 0.5:
        _walk_off(sc, t)


@register("concealment_clothing", Scene.AISLE, Scenario.CONCEALMENT_CLOTHING, 2.0, "Takes an item and tucks it under clothing")
def concealment_clothing(sc: ClipScript) -> None:
    rng = sc.rng
    container = _pick_container(rng, {None: 0.5, "basket": 0.4, "trolley": 0.1})
    t = _aisle_setup(sc, container, "shoulder" if rng.random() < 0.3 else None)
    sc.head_wander = _u(rng, 0.08, 0.30)
    free = sc.ctx["free"]
    if container and rng.random() < 0.4:
        hand = int(rng.choice(free))
        tg = _pick(sc, t, hand)
        t = _to_container(sc, _inspect(sc, tg, hand, _u(rng, 0.3, 1.0)), hand) + _u(rng, 0.2, 0.8)
    t = max(t, _u(rng, 0.5, 5.0))
    hand = int(rng.choice(free))
    tg = _pick(sc, t, hand)
    te = _inspect(sc, tg, hand, _u(rng, 0.3, 1.4))
    if rng.random() < 0.65:
        _nervous_glances(sc, tg, te + 0.3, int(rng.integers(1, 4)))
    if rng.random() < 0.55:
        _turn_away(sc, te)
    linger = _u(rng, 0.5, 1.2)
    t_gone = _to_waist(sc, te, hand, linger)
    other = RIGHT if hand == LEFT else LEFT
    if other in free and rng.random() < 0.4:  # second hand straightens the jacket
        t_o = max(t_gone, sc.last_hand_time(other))
        sc.hand_rel(other, t_o + 0.4, *sc.WAIST)
        sc.hold(other, t_o + 0.4 + linger * 0.7)
        sc.rest(other, t_o + 0.9 + linger * 0.7)
    sc.meta["event_time_s"] = t_gone
    sc.meta["value_at_risk_gbp"] = _value(rng, 19.0)
    if rng.random() < 0.5:
        _walk_off(sc, sc.last_hand_time(hand) + _u(rng, 0.2, 1.0))


@register("shelf_sweep", Scene.AISLE, Scenario.SHELF_SWEEP, 1.6, "Clears many items off a shelf in seconds")
def shelf_sweep(sc: ClipScript) -> None:
    rng = sc.rng
    into_bag = rng.random() < 0.8
    t = _aisle_setup(sc, None if into_bag else "trolley", "tote" if into_bag else None, walk_in=0.3)
    sc.head_wander = _u(rng, 0.05, 0.25)
    free = sc.ctx["free"]
    if rng.random() < 0.5:
        _nervous_glances(sc, t, t + 1.0, int(rng.integers(1, 3)))
    t = max(t, _u(rng, 0.4, 3.0))
    interval = _u(rng, 0.55, 0.95)
    k = int(rng.integers(5, 11))
    third = np.nan
    for i in range(k):
        hand = free[i % len(free)]
        start = max(t + i * interval, sc.last_hand_time(hand))
        if start > sc.duration - 1.2:
            k = i
            break
        tg = _pick(sc, start, hand, quick=True)
        t_gone = _to_bag(sc, tg, hand, 0.08, quick=True) if into_bag else _to_container(sc, tg, hand) - 0.6
        if i == 2:
            third = t_gone
    sc.meta["event_time_s"] = third
    sc.meta["value_at_risk_gbp"] = _value(rng, 34.0 * max(k, 3), 0.45)


# =========================================================================== #
# Counter scenes (self checkout and staffed till)
# =========================================================================== #
def _counter_setup(sc: ClipScript, basket_items: int) -> float:
    rng = sc.rng
    lay = sc.layout
    x0 = 0.5 + _u(rng, -0.025, 0.025)
    y0 = _u(rng, *lay.typical_hip_y)
    sc.body(0.0, x0, y0)
    sc.width(0.0, _u(rng, 0.90, 1.0))
    sc.cont_abs = lay.zones["input"].center
    sc.container_add(0.0, basket_items)
    for hand in (LEFT, RIGHT):
        sc.hand_rel(hand, 0.0, 0.12, 0.0)
    sc.ctx["a"] = sc.image_hand(-1)  # hand nearest the input side
    sc.ctx["b"] = sc.image_hand(+1)  # hand nearest the bagging side
    sc.head_wander = _u(rng, 0.03, 0.15)
    return _u(rng, 0.1, 0.8)


def _zone_point(sc: ClipScript, zone: str) -> tuple[float, float]:
    """A reachable point on the near edge of a counter zone."""
    rng = sc.rng
    r = sc.layout.zones[zone]
    if zone == "input":
        return (_u(rng, r.x1 - 0.075, r.x1 - 0.015), _u(rng, 0.615, 0.675))
    if zone == "bagging":
        return (_u(rng, r.x0 + 0.015, r.x0 + 0.075), _u(rng, 0.615, 0.675))
    return (_u(rng, r.x0 + 0.05, r.x1 - 0.05), _u(rng, 0.615, 0.675))


def _transfer(
    sc: ClipScript,
    t: float,
    mode: str = "normal",
    speed: float = 1.0,
    qty: int = 1,
) -> tuple[float, float]:
    """Move one item from the input side to the bagging side.

    Returns ``(next_start, bagging_time)``. ``mode`` controls what happens at
    the scanner: a normal scan, a slow rescan, loose produce, a marked down
    item, a switched ticket, or one of the skip variants (``cover``, ``slide``,
    ``bypass``, ``stack``, ``to_bag``).
    """
    rng = sc.rng
    a, b = sc.ctx["a"], sc.ctx["b"]
    t = max(t, sc.last_hand_time(a))
    px, py = _zone_point(sc, "input")
    t1 = t + _u(rng, 0.50, 0.80) * speed
    sc.hand_abs(a, t1, px, py)
    sc.item_on(a, t1)
    n_moved = 2 if mode == "stack" else 1
    sc.container_add(t1, -n_moved)

    if mode == "to_bag":  # item goes from the basket straight into a shoulder bag
        bx, by = sc.ctx["bag_hand"]
        t2 = t1 + _u(rng, 0.45, 0.70)
        sc.hand_rel(a, t2, bx, by, side=1)
        sc.item_off(a, t2 - 0.05)
        sc.log(t2, "to_bag", hand=a)
        sc.hold(a, t2 + _u(rng, 0.2, 0.5))
        return t2 + 0.3, t2

    sx, sy = _zone_point(sc, "scanner")
    if mode == "bypass":  # lifted across above the scanner glass
        sy = _u(rng, 0.47, 0.53)
    t2 = t1 + _u(rng, 0.50, 0.75) * speed
    sc.hand_abs(a, t2, sx - 0.02, sy)
    t_b = max(t2 - 0.25 * speed, sc.last_hand_time(b))
    sc.hand_abs(b, max(t_b, t2), sx + 0.03, sy)

    dwell = {
        "normal": _u(rng, 0.40, 0.90) * speed,
        "rescan": _u(rng, 1.2, 2.6),
        "produce": _u(rng, 1.4, 2.8),
        "markdown": _u(rng, 0.40, 0.90) * speed,
        "switch": _u(rng, 0.40, 0.90) * speed,
        "cover": _u(rng, 0.35, 0.80) * speed,
        "slide": _u(rng, 0.0, 0.08),
        "bypass": _u(rng, 0.0, 0.10),
        "stack": _u(rng, 0.40, 0.90) * speed,
        "keyed": _u(rng, 0.03, 0.15),
    }[mode]
    ts = t2 + dwell
    t_scan = t2 + dwell * _u(rng, 0.3, 0.9)
    if mode in ("normal", "stack"):
        sc.pos_scan(t_scan, qty, _u(rng, 0.78, 0.98), rng.normal(1.0, 0.03))
    elif mode == "rescan":
        sc.pos_scan(ts - 0.1, qty, _u(rng, 0.78, 0.98), rng.normal(1.0, 0.03))
    elif mode == "produce":
        sc.pos_scan(ts - 0.1, qty, _u(rng, 0.35, 0.72), _u(rng, 0.80, 1.25))
    elif mode == "markdown":
        sc.pos_scan(t_scan, qty, _u(rng, 0.75, 0.97), _u(rng, 0.25, 0.70))
    elif mode == "switch":
        sc.pos_scan(t_scan, qty, _u(rng, 0.08, 0.45), _u(rng, 0.04, 0.40))
    sc.hold(a, ts)
    sc.hold(b, ts)
    sc.item_off(a, ts)
    sc.item_on(b, ts)

    bx, by = _zone_point(sc, "bagging")
    t3 = ts + _u(rng, 0.45, 0.70)  # the carry to the bagging side does not speed up much
    sc.hand_abs(b, t3, bx, by)
    sc.item_off(b, t3)
    sc.bagging_add(t3, n_moved)
    sc.log(t3, "to_bagging", scanned=mode not in ("cover", "slide", "bypass"))
    sc.hold(b, t3 + 0.12)
    return ts + _u(rng, 0.0, 0.25) * speed, t3


def _run_counter(sc: ClipScript, t: float, plan: list[tuple[str, int]], speed: float) -> list[float]:
    """Run a list of ``(mode, qty)`` transfers until the clip runs out."""
    bag_times = []
    for mode, qty in plan:
        if t > sc.duration - 1.1 * max(speed, 0.6):
            break
        t, tb = _transfer(sc, t, mode, speed, qty)
        bag_times.append(tb if tb < sc.duration else np.nan)
    return bag_times


def _plan(n: int, special: dict[int, tuple[str, int]]) -> list[tuple[str, int]]:
    return [special.get(i, ("normal", 1)) for i in range(n)]


def _maybe_shoulder_bag(sc: ClipScript, p: float) -> None:
    if sc.rng.random() < p:
        side = -1
        sc.bag_rel = (side * 0.15, -0.05)
        sc.ctx["bag_hand"] = sc.bag_rel


@register("sco_normal", Scene.SELF_CHECKOUT, Scenario.NORMAL, 6.5, "Scans and bags every item")
def sco_normal(sc: ClipScript) -> None:
    rng = sc.rng
    t = _counter_setup(sc, int(rng.integers(6, 12)))
    _maybe_shoulder_bag(sc, 0.3)
    _run_counter(sc, t, _plan(9, {}), _u(rng, 0.85, 1.25))


@register("sco_rescan", Scene.SELF_CHECKOUT, Scenario.NORMAL, 2.0, "Barcode will not read, shopper retries")
def sco_rescan(sc: ClipScript) -> None:
    rng = sc.rng
    t = _counter_setup(sc, int(rng.integers(6, 12)))
    _maybe_shoulder_bag(sc, 0.3)
    special = {int(i): ("rescan", 1) for i in rng.choice(4, size=int(rng.integers(1, 3)), replace=False)}
    _run_counter(sc, t, _plan(9, special), _u(rng, 0.85, 1.25))


@register("sco_produce", Scene.SELF_CHECKOUT, Scenario.NORMAL, 2.0, "Weighs loose produce that has no barcode")
def sco_produce(sc: ClipScript) -> None:
    rng = sc.rng
    t = _counter_setup(sc, int(rng.integers(6, 12)))
    _maybe_shoulder_bag(sc, 0.3)
    special = {int(i): ("produce", 1) for i in rng.choice(4, size=int(rng.integers(1, 3)), replace=False)}
    _run_counter(sc, t, _plan(9, special), _u(rng, 0.85, 1.25))


@register("sco_own_bag", Scene.SELF_CHECKOUT, Scenario.NORMAL, 1.6, "Places own shopping bag on the bagging area")
def sco_own_bag(sc: ClipScript) -> None:
    rng = sc.rng
    t = _counter_setup(sc, int(rng.integers(6, 12)))
    sc.bag_abs = sc.layout.zones["bagging"].center
    if rng.random() < 0.6:  # bag placed during the clip: bagging count rises with no scan
        b = sc.ctx["b"]
        bx, by = _zone_point(sc, "bagging")
        sc.hand_abs(b, t + 0.5, bx, by)
        sc.bagging_add(t + 0.5, 1)
        sc.hold(b, t + 0.9)
        t += 1.0
    else:
        sc.bagging_add(0.0, 1)
    _run_counter(sc, t, _plan(9, {}), _u(rng, 0.85, 1.25))


@register("sco_void", Scene.SELF_CHECKOUT, Scenario.NORMAL, 1.0, "Changes mind and has a scanned item removed")
def sco_void(sc: ClipScript) -> None:
    rng = sc.rng
    t = _counter_setup(sc, int(rng.integers(6, 12)))
    _maybe_shoulder_bag(sc, 0.3)
    speed = _u(rng, 0.85, 1.25)
    for i in range(9):
        if t > sc.duration - 1.2:
            break
        t, tb = _transfer(sc, t, "normal", speed)
        if i == 1 and tb < sc.duration - 2.5:
            b = sc.ctx["b"]
            tv = tb + _u(rng, 0.5, 1.2)
            sc.hold(b, tv)
            sc.bagging_add(tv, -1)
            sc.pos_scan(tv + _u(rng, 0.2, 1.0), -1)
            sc.hand_rel(b, tv + 0.5, 0.20, -0.05)
            t = max(t, tv + 0.9)


@register("sco_multibuy", Scene.SELF_CHECKOUT, Scenario.NORMAL, 1.4, "Scans one item and keys the quantity for the rest")
def sco_multibuy(sc: ClipScript) -> None:
    rng = sc.rng
    t = _counter_setup(sc, int(rng.integers(6, 12)))
    _maybe_shoulder_bag(sc, 0.3)
    q = int(rng.integers(2, 5))
    start = int(rng.integers(0, 2))
    special = {start: ("normal", q)}
    special.update({start + j: ("keyed", 1) for j in range(1, q)})
    _run_counter(sc, t, _plan(9, special), _u(rng, 0.85, 1.25))


@register("sco_markdown", Scene.SELF_CHECKOUT, Scenario.NORMAL, 1.5, "Buys reduced to clear items with yellow stickers")
def sco_markdown(sc: ClipScript) -> None:
    rng = sc.rng
    t = _counter_setup(sc, int(rng.integers(6, 12)))
    _maybe_shoulder_bag(sc, 0.3)
    special = {int(i): ("markdown", 1) for i in rng.choice(4, size=int(rng.integers(1, 3)), replace=False)}
    _run_counter(sc, t, _plan(9, special), _u(rng, 0.85, 1.25))


@register("skip_scan", Scene.SELF_CHECKOUT, Scenario.SKIP_SCAN, 2.4, "Bags items without scanning them")
def skip_scan(sc: ClipScript) -> None:
    rng = sc.rng
    t = _counter_setup(sc, int(rng.integers(6, 12)))
    variant = str(rng.choice(["cover", "bypass", "stack", "to_bag"], p=[0.35, 0.30, 0.20, 0.15]))
    _maybe_shoulder_bag(sc, 1.0 if variant == "to_bag" else 0.3)
    k = int(rng.integers(1, 4))
    idx = sorted(int(i) for i in rng.choice(5, size=k, replace=False))
    special = {i: (variant, 1) for i in idx}
    if rng.random() < 0.5:
        sc.glance(t + idx[0] * 1.6 + _u(rng, 0.0, 0.8), rng.choice([-1, 1]) * _u(rng, 0.5, 1.0), _u(rng, 0.5, 0.9))
    times = _run_counter(sc, t, _plan(9, special), _u(rng, 0.85, 1.25))
    done = [times[i] for i in idx if i < len(times) and np.isfinite(times[i])]
    sc.meta["event_time_s"] = done[0] if done else np.nan
    sc.meta["value_at_risk_gbp"] = _value(rng, 7.5 * max(len(done), 1))
    sc.meta["variant"] = variant


@register("ticket_switch", Scene.SELF_CHECKOUT, Scenario.TICKET_SWITCH, 1.8, "Scans a cheap barcode on an expensive item")
def ticket_switch(sc: ClipScript) -> None:
    rng = sc.rng
    t = _counter_setup(sc, int(rng.integers(6, 12)))
    _maybe_shoulder_bag(sc, 0.3)
    k = int(rng.integers(1, 3))
    idx = sorted(int(i) for i in rng.choice(4, size=k, replace=False))
    special = {i: ("switch", 1) for i in idx}
    times = _run_counter(sc, t, _plan(9, special), _u(rng, 0.85, 1.25))
    done = [times[i] for i in idx if i < len(times) and np.isfinite(times[i])]
    sc.meta["event_time_s"] = done[0] if done else np.nan
    sc.meta["value_at_risk_gbp"] = _value(rng, 16.0 * max(len(done), 1))


# ----------------------------- staffed till -------------------------------- #
def _till_setup(sc: ClipScript) -> tuple[float, float]:
    rng = sc.rng
    t = _counter_setup(sc, int(rng.integers(8, 16)))
    return t, _u(rng, 0.55, 0.80)


@register("till_normal", Scene.STAFFED_TILL, Scenario.NORMAL, 5.0, "Cashier scans every item")
def till_normal(sc: ClipScript) -> None:
    t, speed = _till_setup(sc)
    _run_counter(sc, t, _plan(16, {}), speed)


@register("till_bulky", Scene.STAFFED_TILL, Scenario.NORMAL, 1.3, "Cashier scans bulky goods in the trolley with a handset")
def till_bulky(sc: ClipScript) -> None:
    rng = sc.rng
    t, speed = _till_setup(sc)
    b = sc.ctx["b"]
    for _ in range(int(rng.integers(1, 3))):
        t, _ = _transfer(sc, t, "normal", speed)
        t0 = max(t, sc.last_hand_time(b))
        sc.hand_abs(b, t0 + 0.5, 0.5 + _u(rng, 0.0, 0.12), 0.70)
        sc.pos_scan(t0 + 0.7, 1, _u(rng, 0.5, 0.9), rng.normal(1.0, 0.03))
        sc.hold(b, t0 + 0.9)
        t = t0 + 1.0
    _run_counter(sc, t, _plan(14, {}), speed)


@register("till_multibuy", Scene.STAFFED_TILL, Scenario.NORMAL, 1.2, "Cashier scans one and keys the quantity")
def till_multibuy(sc: ClipScript) -> None:
    rng = sc.rng
    t, speed = _till_setup(sc)
    q = int(rng.integers(2, 6))
    start = int(rng.integers(0, 3))
    special = {start: ("normal", q)}
    special.update({start + j: ("keyed", 1) for j in range(1, q)})
    _run_counter(sc, t, _plan(16, special), speed)


@register("till_rescan", Scene.STAFFED_TILL, Scenario.NORMAL, 1.0, "Cashier struggles with a damaged barcode")
def till_rescan(sc: ClipScript) -> None:
    rng = sc.rng
    t, speed = _till_setup(sc)
    special = {int(i): ("rescan", 1) for i in rng.choice(6, size=int(rng.integers(1, 3)), replace=False)}
    _run_counter(sc, t, _plan(16, special), speed)


@register("sweethearting", Scene.STAFFED_TILL, Scenario.SWEETHEARTING, 1.7, "Cashier passes items to an accomplice without scanning")
def sweethearting(sc: ClipScript) -> None:
    rng = sc.rng
    t, speed = _till_setup(sc)
    variant = str(rng.choice(["slide", "cover", "stack"], p=[0.45, 0.35, 0.20]))
    k = int(rng.integers(2, 5))
    idx = sorted(int(i) for i in rng.choice(8, size=k, replace=False))
    special = {i: (variant, 1) for i in idx}
    times = _run_counter(sc, t, _plan(16, special), speed)
    done = [times[i] for i in idx if i < len(times) and np.isfinite(times[i])]
    sc.meta["event_time_s"] = done[1] if len(done) > 1 else (done[0] if done else np.nan)
    sc.meta["value_at_risk_gbp"] = _value(rng, 13.0 * max(len(done), 1))
    sc.meta["variant"] = variant


# =========================================================================== #
# Store exit
# =========================================================================== #
def _exit_walk(sc: ClipScript, pace: tuple[float, float], carry: str, pause_p: float = 0.0) -> float:
    """Walk from the shop floor to the exit doors. Returns the door crossing time."""
    rng = sc.rng
    lay = sc.layout
    y0 = _u(rng, *lay.typical_hip_y)
    depth = (y0 - lay.typical_hip_y[0]) / (lay.typical_hip_y[1] - lay.typical_hip_y[0])
    sc.scale = lay.typical_scale[0] + (lay.typical_scale[1] - lay.typical_scale[0]) * depth
    door_x = lay.zones["exit"].x0 + 0.02
    x0, x1 = _u(rng, 0.02, 0.18), _u(rng, 0.93, 1.0)
    speed = _u(rng, *pace)
    travel = (x1 - x0) / speed
    pause = _u(rng, 1.0, 2.5) if rng.random() < pause_p else 0.0
    slack = sc.duration - travel - pause - 0.3
    t0 = _u(rng, 0.0, slack) if slack > 0 else 0.0
    sc.body(0.0, x0, y0)
    sc.body(t0, x0, y0)
    if pause > 0:
        xm = _u(rng, 0.55, 0.72)
        tm = t0 + (xm - x0) / speed
        sc.body(tm, xm, y0)
        sc.body(tm + pause, xm, y0)
        t_end = tm + pause + (x1 - xm) / speed
        t_door = tm + pause + (door_x - xm) / speed
    else:
        t_end = t0 + travel
        t_door = t0 + (door_x - x0) / speed
    sc.body(t_end, x1, y0 + _u(rng, -0.02, 0.02))
    sc.width(0.0, _u(rng, 0.35, 0.60))
    sc.head_wander = _u(rng, 0.03, 0.15)

    if carry == "trolley":
        sc.cont_rel = (0.25, 0.10)
        sc.hand_rel(LEFT, 0.0, 0.10, -0.06, side=1)
        sc.hand_rel(RIGHT, 0.0, 0.12, -0.05, side=1)
    elif carry == "basket":
        sc.cont_rel = (0.13, 0.10)
        sc.hand_rel(sc.image_hand(1), 0.0, 0.12, 0.03, side=1)
        sc.rest(sc.image_hand(-1), 0.0)
    elif carry == "bags":
        sc.bag_rel = (float(rng.choice([-1, 1])) * 0.12, 0.10)
        sc.rest(LEFT, 0.0)
        sc.rest(RIGHT, 0.0)
    elif carry == "arms":  # loose merchandise held against the chest
        for hand in (LEFT, RIGHT):
            sc.hand_rel(hand, 0.0, 0.04, -0.15)
            sc.item_on(hand, 0.0)
    else:
        sc.rest(LEFT, 0.0)
        sc.rest(RIGHT, 0.0)
    return t_door


def _dwell(rng: np.random.Generator, value: float) -> float:
    """Checkout dwell as measured by the multi camera tracker (noisy)."""
    return float(max(0.0, value * rng.normal(1.0, 0.15) + rng.normal(0.0, 1.5)))


@register("exit_paid_till", Scene.EXIT, Scenario.NORMAL, 4.2, "Leaves after paying at a staffed till")
def exit_paid_till(sc: ClipScript) -> None:
    rng = sc.rng
    carry = str(rng.choice(["trolley", "bags", "basket"], p=[0.6, 0.3, 0.1]))
    t_door = _exit_walk(sc, (0.075, 0.11), carry, pause_p=0.3)
    if carry != "bags":
        unbagged = rng.random() < 0.25
        sc.container_add(0.0, int(rng.integers(5, 13)) if unbagged else int(min(rng.poisson(1.5), 6)))
    sc.meta["txn_linked"] = int(rng.random() < 0.94)
    sc.meta["checkout_dwell_s"] = _dwell(rng, float(np.exp(rng.normal(np.log(110), 0.5))))
    if rng.random() < 0.015:
        sc.eas_alarm(t_door)


@register("exit_paid_sco", Scene.EXIT, Scenario.NORMAL, 2.6, "Leaves after paying at self checkout")
def exit_paid_sco(sc: ClipScript) -> None:
    rng = sc.rng
    carry = str(rng.choice(["bags", "basket", "trolley", "none"], p=[0.5, 0.2, 0.2, 0.1]))
    t_door = _exit_walk(sc, (0.075, 0.12), carry, pause_p=0.2)
    if carry in ("basket", "trolley"):
        sc.container_add(0.0, int(min(rng.poisson(1.2), 5)))
    sc.meta["txn_linked"] = int(rng.random() < 0.94)
    sc.meta["checkout_dwell_s"] = _dwell(rng, _u(rng, 35, 150))
    if rng.random() < 0.015:
        sc.eas_alarm(t_door)


@register("exit_no_purchase", Scene.EXIT, Scenario.NORMAL, 2.2, "Leaves without buying anything")
def exit_no_purchase(sc: ClipScript) -> None:
    rng = sc.rng
    carry = str(rng.choice(["none", "bags"], p=[0.75, 0.25]))  # bags from another shop
    _exit_walk(sc, (0.08, 0.14), carry)
    sc.meta["txn_linked"] = 0
    sc.meta["checkout_dwell_s"] = _dwell(rng, _u(rng, 0, 6) if rng.random() < 0.8 else _u(rng, 6, 40))


@register("exit_staff", Scene.EXIT, Scenario.NORMAL, 0.8, "Colleague takes trolleys or a cage outside")
def exit_staff(sc: ClipScript) -> None:
    rng = sc.rng
    _exit_walk(sc, (0.06, 0.10), "trolley")
    sc.container_add(0.0, int(rng.integers(0, 3)))
    sc.meta["txn_linked"] = 0
    sc.meta["checkout_dwell_s"] = _dwell(rng, _u(rng, 0, 10))


@register("exit_collection", Scene.EXIT, Scenario.NORMAL, 0.6, "Leaves with a click and collect or prepaid order")
def exit_collection(sc: ClipScript) -> None:
    rng = sc.rng
    carry = str(rng.choice(["trolley", "arms"], p=[0.6, 0.4]))
    _exit_walk(sc, (0.075, 0.12), carry)
    if carry == "trolley":
        sc.container_add(0.0, int(rng.integers(1, 6)))
    sc.meta["txn_linked"] = int(rng.random() < 0.3)  # collection desk sales rarely link to the track
    sc.meta["checkout_dwell_s"] = _dwell(rng, _u(rng, 0, 25))


@register("push_out", Scene.EXIT, Scenario.PUSH_OUT, 2.0, "Walks out with unpaid goods")
def push_out(sc: ClipScript) -> None:
    rng = sc.rng
    trolley = rng.random() < 0.8
    t_door = _exit_walk(sc, (0.11, 0.19) if rng.random() < 0.75 else (0.08, 0.12), "trolley" if trolley else "arms")
    if trolley:
        sc.container_add(0.0, int(rng.integers(5, 15)))
        sc.meta["value_at_risk_gbp"] = _value(rng, 170.0, 0.5)
    else:
        sc.meta["value_at_risk_gbp"] = _value(rng, 38.0, 0.5)
    sc.meta["txn_linked"] = int(rng.random() < 0.03)
    sc.meta["checkout_dwell_s"] = _dwell(rng, _u(rng, 0, 8) if rng.random() < 0.7 else _u(rng, 8, 45))
    if rng.random() < 0.5:
        sc.glance(max(t_door - _u(rng, 1.0, 3.0), 0.2), rng.choice([-1, 1]) * _u(rng, 0.5, 1.0), _u(rng, 0.5, 0.9))
    if rng.random() < 0.35:
        sc.eas_alarm(t_door)
    sc.meta["event_time_s"] = t_door
    sc.meta["variant"] = "trolley" if trolley else "walk_out"


# =========================================================================== #
# Registry helpers
# =========================================================================== #
def subtype_names() -> list[str]:
    return list(REGISTRY)


def subtype_weights() -> np.ndarray:
    w = np.array([REGISTRY[n].weight for n in REGISTRY], dtype=float)
    return w / w.sum()
