"""Gallery builder: picture examples for every scenario.

For each scenario the builder simulates a handful of candidate clips, scores
them with the trained detector and keeps one that tells a clear story. It then
writes a storyboard, an animated replay and the narrator's incident brief.

The examples are curated for clarity, so the gallery also includes two clips
the detector gets wrong. A portfolio that only shows wins is not an evaluation.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from ..evaluation.replay import risk_curves
from ..explain.evidence import build_evidence, build_timeline
from ..explain.narrator import Narrator, TemplateNarrator
from ..features.extractor import extract_features
from ..models.sentinel import SentinelModel
from ..schema import SCENARIO_TITLES, SCENE_TITLES, ClipBatch, Scene
from ..simulation.behaviours import REGISTRY
from ..simulation.generator import generate_batch
from .render import INK, MINT, MUTED, TEXT, RenderStyle, font, render_frame
from .storyboard import pick_keyframes, render_gif, render_storyboard


@dataclass(frozen=True)
class Showcase:
    subtype: str
    slug: str
    want: str  # "alert", "clear", "miss" or "false_alert"
    quality: str = "hd"
    crowding: int = 0
    variant: str | None = None  # restrict to one variant of the behaviour


SHOWCASES: tuple[Showcase, ...] = (
    Showcase("concealment_bag", "01_concealment_bag", "alert"),
    Showcase("concealment_clothing", "02_concealment_clothing", "alert"),
    Showcase("shelf_sweep", "03_shelf_sweep", "alert"),
    Showcase("skip_scan", "04_skip_scan", "alert"),
    Showcase("ticket_switch", "05_ticket_switch", "alert"),
    Showcase("sweethearting", "06_sweethearting", "alert"),
    Showcase("push_out", "07_push_out", "alert", variant="trolley"),
    Showcase("scan_and_go", "08_benign_scan_and_go", "clear"),
    Showcase("carry_in_hand", "09_benign_carry_in_hand", "clear"),
    Showcase("sco_own_bag", "10_benign_own_bag", "clear"),
    Showcase("concealment_bag", "11_missed_concealment", "miss", quality="low_light", crowding=2),
    Showcase("exit_collection", "12_false_alert_collection", "false_alert"),
)


def _story_score(timeline: list[dict], event_time: float, duration: float) -> float:
    """Prefer clips with a readable story: events spread out and the key moment mid clip."""
    heavy = sum(1 for e in timeline if e["weight"] >= 2)
    centred = 0.0 if not np.isfinite(event_time) else 1.0 - abs(event_time / duration - 0.55)
    return heavy + 2.0 * centred


def _select(model: SentinelModel, batch: ClipBatch, want: str, variant: str | None = None) -> int | None:
    out = model.assess(extract_features(batch))
    truth = batch.meta["scenario"].to_numpy()
    tier = out["tier_id"].to_numpy()
    if want == "alert":
        ok = (tier >= 1) & (out["flag_class"].to_numpy() == truth)
    elif want == "clear":
        ok = tier == 0
    elif want == "miss":
        ok = (tier == 0) & (truth != 0)
    else:
        ok = (tier >= 1) & (truth == 0)
    if variant is not None:
        ok &= batch.meta["variant"].to_numpy() == variant
    best, best_score = None, -np.inf
    for i in np.flatnonzero(ok):
        score = _story_score(build_timeline(batch, int(i)), float(batch.meta["event_time_s"].iloc[i]), batch.duration_s)
        if want == "alert":
            score += float(out["risk"].iloc[i])
        elif want == "miss":  # show a genuine miss, not a near miss at the threshold
            score -= 4.0 * float(out["risk"].iloc[i])
        if score > best_score:
            best, best_score = int(i), score
    return best


def _hero(panels: list[tuple[Image.Image, str]], path: Path) -> None:
    pw, ph = panels[0][0].size
    gap, margin, head = 16, 28, 132
    img = Image.new("RGB", (margin * 2 + pw * 2 + gap, head + ph * 2 + gap + margin), INK)
    d = ImageDraw.Draw(img)
    d.text((margin, 24), "SHELF SENTINEL AI", font=font("mono", 17), fill=MINT)
    d.text((margin, 48), "Seven shoplifting scenarios, one explainable alert", font=font("bold", 32), fill=TEXT)
    d.text((margin, 92), "Anonymous pose, item flow and point of sale signals. No faces, no video stored.", font=font("sans", 16), fill=MUTED)
    for i, (frame, _) in enumerate(panels[:4]):
        img.paste(frame, (margin + (i % 2) * (pw + gap), head + (i // 2) * (ph + gap)))
    img.save(path, optimize=True)


def build_gallery(
    model: SentinelModel,
    root: str | Path = ".",
    seed: int = 2026,
    narrator: Narrator | None = None,
    candidates: int = 36,
    gifs: bool = True,
    verbose: bool = True,
) -> list[dict]:
    """Render every showcase. Returns one summary record per example."""
    root = Path(root)
    img_dir = root / "docs" / "images" / "scenarios"
    inc_dir = root / "reports" / "incidents"
    img_dir.mkdir(parents=True, exist_ok=True)
    inc_dir.mkdir(parents=True, exist_ok=True)
    narrator = narrator or TemplateNarrator()
    records, hero_panels = [], []

    for k, show in enumerate(SHOWCASES):
        n = candidates * (8 if show.want in ("miss", "false_alert") else 1)
        batch = generate_batch(n, seed=seed + 17, shard=800 + k, subtypes=[show.subtype], quality=show.quality, crowding=show.crowding)
        idx = _select(model, batch, show.want, show.variant)
        if idx is None:
            if verbose:
                print(f"  {show.slug}: no candidate matched '{show.want}', skipped")
            continue
        one = batch.select([idx])
        evidence = build_evidence(model, one, 0)
        timeline = build_timeline(one, 0)
        counts, risk, tiers = risk_curves(model, one)
        a = evidence["assessment"]
        flag = int(a["scenario_id"])
        thresholds = (model.policy.review.get(flag, 1.01), model.policy.priority.get(flag, 1.01))
        spec = REGISTRY[show.subtype]
        truth_title = SCENARIO_TITLES[spec.scenario]
        keys = pick_keyframes(timeline, one.n_frames)
        keyframes = [render_frame(one, 0, e["frame"], style=RenderStyle(width=640, height=360)) for e in keys]
        brief = narrator.narrate(evidence, keyframes)

        title = {
            "alert": truth_title,
            "clear": f"Hard negative: {spec.description.lower()}",
            "miss": f"Missed: {truth_title.lower()}",
            "false_alert": f"False alert: {spec.description.lower()}",
        }[show.want]
        quality = {"hd": "HD", "sd": "SD", "low_light": "low light"}[show.quality]
        crowd = {0: "quiet", 1: "busy", 2: "crowded"}[show.crowding]
        subtitle = f"{SCENE_TITLES[Scene(int(spec.scene))]} camera, {quality}, {crowd} store"
        footer = f"Narrator: {brief.headline} {brief.recommended_action}"
        board = render_storyboard(one, 0, a, timeline, counts, risk[0], tiers[0], thresholds, title, subtitle, footer)
        board.save(img_dir / f"{show.slug}.png", optimize=True)
        if gifs and show.want == "alert":
            render_gif(one, 0, img_dir / f"{show.slug}.gif", counts, risk[0], tiers[0], a["scenario"])
        (inc_dir / f"{show.slug}.md").write_text(brief.to_markdown() + "\n")
        (inc_dir / f"{show.slug}.json").write_text(json.dumps({"brief": brief.to_dict(), "evidence": evidence}, indent=2))

        if show.want == "alert" and show.subtype in ("concealment_bag", "skip_scan", "sweethearting", "push_out"):
            # One illustrative frame carrying the verdict for the whole clip.
            if spec.scene == Scene.EXIT:
                hip_x = one.keypoints[0, :, 11:13, 0].mean(axis=1)
                f = int(np.argmin(np.abs(hip_x - 0.55)))
            else:
                fired = np.flatnonzero(tiers[0] >= 1)
                f = int(min(counts[fired[0]] + 3, one.n_frames - 1)) if len(fired) else one.n_frames // 2
            panel = render_frame(one, 0, f, risk=a["risk"], tier=a["tier"], label=a["scenario"], style=RenderStyle(width=640, height=360))
            hero_panels.append((panel, show.slug))
        records.append(
            {
                "slug": show.slug,
                "subtype": show.subtype,
                "kind": show.want,
                "truth": truth_title,
                "tier": a["tier"],
                "flagged": a["scenario"],
                "risk": a["risk"],
                "headline": brief.headline,
                "narrator": brief.provider,
            }
        )
        if verbose:
            print(f"  {show.slug}: {a['tier']} {a['scenario']} {a['risk']:.2f}", flush=True)

    if len(hero_panels) >= 4:
        _hero(hero_panels, root / "docs" / "images" / "hero.png")
    (inc_dir / "index.json").write_text(json.dumps(records, indent=2))
    return records
