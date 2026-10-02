"""Storyboards and animated replays for single clips.

A storyboard is the picture a reviewer (or a README reader) needs: four key
frames chosen from the detected timeline, a caption under each, and the risk
curve from streaming replay with the alert threshold drawn on it.
"""

from __future__ import annotations

import textwrap
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from ..schema import ClipBatch
from .render import (
    AMBER,
    GREEN,
    GRID,
    INK,
    MINT,
    MUTED,
    PANEL,
    RED,
    TEXT,
    TIER_COLOURS,
    RenderStyle,
    font,
    render_frame,
)

PANEL_STYLE = RenderStyle(width=640, height=360, supersample=2)


def pick_keyframes(timeline: list[dict], n_frames: int, k: int = 4, min_gap: int = 4) -> list[dict]:
    """Choose ``k`` frames that tell the story: the heaviest events, spread out in time."""
    chosen: list[dict] = []
    for event in sorted(timeline, key=lambda e: (-e["weight"], e["frame"])):
        if all(abs(event["frame"] - c["frame"]) >= min_gap for c in chosen):
            chosen.append(event)
        if len(chosen) == k:
            break
    # Pad with evenly spaced context frames when the timeline is short.
    fillers = iter(np.linspace(3, n_frames - 4, 2 * k + 1).astype(int))
    while len(chosen) < k:
        try:
            f = int(next(fillers))
        except StopIteration:
            break
        if all(abs(f - c["frame"]) >= min_gap for c in chosen):
            chosen.append({"frame": f, "t": round(f / 5.0, 1), "kind": "context", "text": "Context frame", "weight": 0})
    return sorted(chosen, key=lambda e: e["frame"])[:k]


def _risk_at(counts: np.ndarray, values: np.ndarray, frame: int) -> float:
    """Risk known to the system once ``frame`` has been seen."""
    idx = np.searchsorted(counts, frame + 1, side="right") - 1
    return float(values[idx]) if idx >= 0 else 0.0  # nothing is scored during warm up


def _timeline_panel(
    size: tuple[int, int],
    counts: np.ndarray,
    risk: np.ndarray,
    fps: float,
    n_frames: int,
    review_thr: float,
    priority_thr: float,
    keyframes: list[dict],
    timeline: list[dict],
    colour: tuple[int, int, int],
) -> Image.Image:
    ss = 2
    w, h = size[0] * ss, size[1] * ss
    img = Image.new("RGB", (w, h), PANEL)
    d = ImageDraw.Draw(img, "RGBA")
    f_s, f_b = font("sans", 13 * ss), font("bold", 14 * ss)
    left, right, top, bottom = 62 * ss, w - 22 * ss, 56 * ss, h - 46 * ss
    dur = n_frames / fps

    def X(t: float) -> float:
        return left + (right - left) * t / dur

    def Y(v: float) -> float:
        return bottom - (bottom - top) * v

    d.text((left, 14 * ss), "Risk over time (streaming replay)", font=f_b, fill=TEXT)
    for v in (0.0, 0.5, 1.0):
        d.line([left, Y(v), right, Y(v)], fill=GRID, width=ss)
        d.text((left - 10 * ss, Y(v)), f"{v:.1f}", font=f_s, fill=MUTED, anchor="rm")
    for t in range(0, int(dur) + 1, 2):
        d.text((X(t), bottom + 22 * ss), f"{t} s", font=f_s, fill=MUTED, anchor="mm")
    for thr, name, col, offset in ((review_thr, "review threshold", AMBER, 10), (priority_thr, "priority threshold", RED, -10)):
        if thr <= 1.0:
            for x in np.arange(left, right, 14 * ss):
                d.line([x, Y(thr), min(x + 7 * ss, right), Y(thr)], fill=(*col, 190), width=ss)
            d.text((right - 4 * ss, Y(thr) + offset * ss), name, font=f_s, fill=col, anchor="rm")
    for e in timeline:  # event ticks along the base line
        if e["weight"] >= 2:
            x = X(e["frame"] / fps)
            d.line([x, bottom, x, bottom + 9 * ss], fill=(*TEXT, 200), width=ss)
    pts = [(X(c / fps), Y(float(np.clip(r, 0, 1)))) for c, r in zip(counts, risk, strict=True)]
    d.polygon([(pts[0][0], bottom), *pts, (pts[-1][0], bottom)], fill=(*colour, 46))
    d.line(pts, fill=colour, width=3 * ss, joint="curve")
    for i, e in enumerate(keyframes, start=1):
        x = X(e["frame"] / fps)
        y = Y(float(np.clip(_risk_at(counts, risk, e["frame"]), 0, 1)))
        d.line([x, y, x, bottom], fill=(*TEXT, 90), width=ss)
        r = 11 * ss
        d.ellipse([x - r, y - r, x + r, y + r], fill=INK, outline=TEXT, width=ss)
        d.text((x, y), str(i), font=f_s, fill=TEXT, anchor="mm")
    return img.resize(size, Image.LANCZOS)


def render_storyboard(
    batch: ClipBatch,
    index: int,
    assessment: dict,
    timeline: list[dict],
    counts: np.ndarray,
    risk: np.ndarray,
    tiers: np.ndarray,
    thresholds: tuple[float, float],
    title: str,
    subtitle: str,
    footer: str | None = None,
) -> Image.Image:
    """Compose the four frame storyboard for one clip.

    Parameters
    ----------
    assessment : dict with ``tier``, ``scenario`` and ``risk`` for the whole clip.
    counts, risk, tiers : streaming replay output for this clip.
    thresholds : ``(review, priority)`` thresholds of the flagged scenario.
    """
    keys = pick_keyframes(timeline, batch.n_frames)
    pw, ph = PANEL_STYLE.width, PANEL_STYLE.height
    gap, margin, head, cap_h, tl_h, foot = 20, 28, 96, 64, 210, 58 if footer else 0
    width = margin * 2 + pw * 2 + gap
    height = head + 2 * (ph + cap_h) + gap + tl_h + foot + margin
    img = Image.new("RGB", (width, height), INK)
    d = ImageDraw.Draw(img)
    colour = TIER_COLOURS.get(assessment["tier"], MINT)

    d.text((margin, 22), "SHELF SENTINEL AI", font=font("mono", 15), fill=MINT)
    d.text((margin, 44), title, font=font("bold", 30), fill=TEXT)
    d.text((margin + d.textlength(title, font=font("bold", 30)) + 16, 57), subtitle, font=font("sans", 17), fill=MUTED)
    chip = f"{assessment['tier'].upper()}   {assessment['scenario']}   {assessment['risk']:.2f}" if assessment["tier"] != "clear" else f"CLEAR   no alert   risk {assessment['risk']:.2f}"
    box = d.textbbox((width - margin - 16, 50), chip, font=font("bold", 17), anchor="rm")
    d.rounded_rectangle([box[0] - 16, box[1] - 10, box[2] + 16, box[3] + 10], radius=18, outline=colour, width=2, fill=PANEL)
    d.text((width - margin - 16, 50), chip, font=font("bold", 17), fill=colour, anchor="rm")

    for i, e in enumerate(keys):
        col, row = i % 2, i // 2
        x = margin + col * (pw + gap)
        y = head + row * (ph + cap_h + gap // 2)
        r = _risk_at(counts, risk, e["frame"])
        tier_id = int(_risk_at(counts, tiers, e["frame"]))
        tier = ("clear", "review", "priority")[tier_id]
        label = assessment["scenario"] if tier_id else None
        frame = render_frame(batch, index, e["frame"], risk=r, tier=tier, label=label, style=PANEL_STYLE)
        img.paste(frame, (x, y))
        d.rounded_rectangle([x, y, x + pw, y + ph], radius=4, outline=GRID, width=1)
        d.ellipse([x + 10, y + 10, x + 38, y + 38], fill=INK, outline=TEXT, width=2)
        d.text((x + 24, y + 24), str(i + 1), font=font("bold", 15), fill=TEXT, anchor="mm")
        caption = textwrap.wrap(e["text"], width=74)[:2]
        d.text((x + 2, y + ph + 10), f"{e['t']:.1f} s", font=font("mono", 14), fill=colour if e["weight"] >= 3 else MUTED)
        for j, line in enumerate(caption):
            d.text((x + 62, y + ph + 9 + j * 20), line, font=font("sans", 15), fill=TEXT if e["weight"] >= 2 else MUTED)

    y_tl = head + 2 * (ph + cap_h) + gap
    panel = _timeline_panel(
        (width - 2 * margin, tl_h), counts, risk, batch.fps, batch.n_frames, thresholds[0], thresholds[1], keys, timeline, colour
    )
    img.paste(panel, (margin, y_tl))
    d.rounded_rectangle([margin, y_tl, width - margin, y_tl + tl_h], radius=4, outline=GRID, width=1)
    if footer:
        for j, line in enumerate(textwrap.wrap(footer, width=150)[:2]):
            d.text((margin, y_tl + tl_h + 14 + j * 20), line, font=font("sans", 15), fill=MUTED)
    return img


def render_gif(
    batch: ClipBatch,
    index: int,
    path: str | Path,
    counts: np.ndarray,
    risk: np.ndarray,
    tiers: np.ndarray,
    scenario_title: str,
    size: tuple[int, int] = (560, 315),
) -> Path:
    """Animated replay of one clip with the live risk banner."""
    style = RenderStyle(width=size[0], height=size[1], supersample=2)
    frames = []
    for f in range(batch.n_frames):
        j = max(np.searchsorted(counts, f + 1, side="right") - 1, 0)
        tier_id = int(tiers[j]) if f + 1 >= counts[0] else 0
        r = float(risk[j]) if f + 1 >= counts[0] else 0.0
        frame = render_frame(
            batch, index, f, risk=r, tier=("clear", "review", "priority")[tier_id], label=scenario_title if tier_id else None, style=style
        )
        d = ImageDraw.Draw(frame)
        bar_w = int((size[0] - 24) * np.clip(r, 0, 1))
        d.rectangle([12, size[1] - 7, size[0] - 12, size[1] - 3], fill=PANEL)
        d.rectangle([12, size[1] - 7, 12 + bar_w, size[1] - 3], fill=(RED if tier_id == 2 else AMBER if tier_id == 1 else GREEN))
        frames.append(frame.quantize(colors=96, method=Image.MEDIANCUT, dither=Image.NONE))
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    frames[0].save(path, save_all=True, append_images=frames[1:], duration=int(1000 / batch.fps), loop=0, optimize=True)
    return path
