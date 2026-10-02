"""Corpus generator: stores, shards and the streaming build of the dataset.

The full corpus is far too large to hold as raw keypoints in memory (sixteen
million frames), so it is produced shard by shard. Each shard is simulated,
passed through the sensor model, reduced to clip level features and written to
parquet. Only a small sample of raw clips is kept for rendering and replay.
Everything is seeded: the same config always yields the same bytes.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from ..config import DataConfig
from ..scenes import layout_for
from ..schema import CAMERA_QUALITIES, ClipBatch, Scenario
from .behaviours import REGISTRY, subtype_names, subtype_weights
from .kinematics import build_skeleton
from .perception import NoiseConfig, apply_sensor_noise
from .script import ClipScript, LatentClip

STORE_FORMATS = ("convenience", "supermarket", "superstore")


@dataclass(frozen=True)
class StoreProfile:
    store_id: int
    fmt: str
    quality_p: tuple[float, float, float]
    crowd_p: tuple[float, float, float]
    subtype_p: tuple[float, ...]
    traffic: float


def make_stores(n_stores: int, seed: int) -> list[StoreProfile]:
    """Create store profiles with different cameras, footfall and theft mix."""
    rng = np.random.default_rng([seed, 9001])
    names = subtype_names()
    base = subtype_weights()
    stores = []
    for sid in range(n_stores):
        fmt = STORE_FORMATS[int(rng.choice(3, p=[0.35, 0.45, 0.20]))]
        quality_alpha = {"convenience": (2, 4, 3), "supermarket": (5, 4, 1.5), "superstore": (7, 3, 1)}[fmt]
        crowd_alpha = {"convenience": (5, 3, 1), "supermarket": (4, 4, 2), "superstore": (3, 4, 3)}[fmt]
        mult = np.exp(rng.normal(0.0, 0.25, size=len(names)))
        for i, name in enumerate(names):
            # Small shops have no self checkout bank and see little trolley traffic.
            if fmt == "convenience" and name.startswith(("sco_", "skip_scan", "ticket_switch")):
                mult[i] *= 0.5
            if fmt == "convenience" and name in ("bulk_buyer", "push_out", "exit_staff"):
                mult[i] *= 0.6
            if fmt == "superstore" and name in ("push_out", "shelf_sweep"):
                mult[i] *= 1.3
        p = base * mult
        stores.append(
            StoreProfile(
                store_id=sid,
                fmt=fmt,
                quality_p=tuple(rng.dirichlet(quality_alpha)),
                crowd_p=tuple(rng.dirichlet(crowd_alpha)),
                subtype_p=tuple(p / p.sum()),
                traffic={"convenience": 0.6, "supermarket": 1.0, "superstore": 1.6}[fmt] * float(rng.uniform(0.7, 1.3)),
            )
        )
    return stores


def _hour(rng: np.random.Generator) -> int:
    hours = np.arange(7, 23)
    w = np.array([2, 3, 4, 5, 6, 7, 7, 6, 6, 7, 8, 9, 8, 6, 4, 2], dtype=float)
    return int(rng.choice(hours, p=w / w.sum()))


def simulate_latents(
    n: int,
    seed: int,
    shard: int,
    stores: list[StoreProfile],
    n_frames: int,
    fps: float,
    subtypes: list[str] | None = None,
    quality: str | None = None,
    crowding: int | None = None,
) -> tuple[list[LatentClip], pd.DataFrame]:
    """Script ``n`` clips. Returns latent states and the matching meta table."""
    names = subtype_names()
    traffic = np.array([s.traffic for s in stores])
    traffic = traffic / traffic.sum()
    latents: list[LatentClip] = []
    rows = []
    for i in range(n):
        rng = np.random.default_rng([seed, shard, i])
        store = stores[int(rng.choice(len(stores), p=traffic))]
        if subtypes is None:
            name = names[int(rng.choice(len(names), p=store.subtype_p))]
        else:
            name = subtypes[i % len(subtypes)]
        spec = REGISTRY[name]
        q = CAMERA_QUALITIES.index(quality) if quality else int(rng.choice(3, p=store.quality_p))
        c = int(crowding) if crowding is not None else int(rng.choice(3, p=store.crowd_p))
        script = ClipScript(rng, layout_for(spec.scene), n_frames=n_frames, fps=fps)
        spec.fn(script)
        latent = script.build()
        latents.append(latent)
        m = latent.meta
        rows.append(
            {
                "clip_id": f"S{shard:03d}C{i:05d}",
                "store_id": store.store_id,
                "store_format": store.fmt,
                "scene": int(spec.scene),
                "scenario": int(spec.scenario),
                "subtype": name,
                "variant": m.get("variant", ""),
                "camera_quality": CAMERA_QUALITIES[q],
                "crowding": c,
                "hour": _hour(rng),
                "body_scale": latent.scale,
                "txn_linked": int(m.get("txn_linked", 0)),
                "checkout_dwell_s": float(m.get("checkout_dwell_s", 0.0)),
                "event_time_s": float(m.get("event_time_s", np.nan)),
                "value_at_risk_gbp": float(m.get("value_at_risk_gbp", 0.0)),
                "events_json": json.dumps(latent.events),
            }
        )
    return latents, pd.DataFrame(rows)


def latents_to_batch(
    latents: list[LatentClip],
    meta: pd.DataFrame,
    seed: int,
    shard: int,
    fps: float,
    noise: NoiseConfig | None = None,
) -> ClipBatch:
    """Run kinematics and the sensor model over scripted clips."""
    stack = lambda attr: np.stack([getattr(c, attr) for c in latents])  # noqa: E731
    scale = np.array([c.scale for c in latents], dtype=np.float32)
    mirror = np.array([c.mirror for c in latents], dtype=np.float32)
    xy, vis = build_skeleton(
        hip=stack("hip"),
        scale=scale,
        width=stack("width"),
        left_hand=stack("left_hand"),
        right_hand=stack("right_hand"),
        yaw=stack("yaw"),
        gait_phase=stack("gait_phase"),
        walk=stack("walk"),
        mirror=mirror,
    )
    quality = meta["camera_quality"].map({q: i for i, q in enumerate(CAMERA_QUALITIES)}).to_numpy()
    lower_occ = np.array([layout_for(s).lower_body_occluded for s in meta["scene"]])
    keypoints, signals = apply_sensor_noise(
        xy=xy,
        visibility=vis,
        signals=stack("signals"),
        scale=scale,
        quality=quality,
        crowding=meta["crowding"].to_numpy(),
        lower_occluded=lower_occ,
        trolley_scene=~lower_occ,
        back_view=mirror > 0,
        rng=np.random.default_rng([seed, shard, 777]),
        cfg=noise,
    )
    return ClipBatch(keypoints=keypoints, signals=signals, meta=meta.reset_index(drop=True), fps=fps)


def generate_batch(
    n: int,
    seed: int = 0,
    shard: int = 0,
    stores: list[StoreProfile] | None = None,
    n_frames: int = 64,
    fps: float = 5.0,
    subtypes: list[str] | None = None,
    quality: str | None = None,
    crowding: int | None = None,
    noise: NoiseConfig | None = None,
) -> ClipBatch:
    """Generate one in memory batch of clips (the public entry point)."""
    stores = stores or make_stores(8, seed)
    latents, meta = simulate_latents(n, seed, shard, stores, n_frames, fps, subtypes, quality, crowding)
    return latents_to_batch(latents, meta, seed, shard, fps, noise)


def build_table(meta: pd.DataFrame, feats: pd.DataFrame) -> pd.DataFrame:
    """Join clip meta and features into the modelling table (no duplicate columns)."""
    drop = [c for c in meta.columns if c in feats.columns or c == "events_json"]
    return pd.concat([meta.drop(columns=drop).reset_index(drop=True), feats.reset_index(drop=True)], axis=1)


def generate_dataset(cfg: DataConfig, root: str | Path = ".", verbose: bool = True) -> dict:
    """Build the full corpus as parquet feature shards plus a raw sample."""
    from ..features.extractor import extract_features  # local import avoids a cycle

    out = Path(root) / cfg.out_dir
    (out / "features").mkdir(parents=True, exist_ok=True)
    (out / "raw").mkdir(parents=True, exist_ok=True)
    stores = make_stores(cfg.n_stores, cfg.seed)
    n_shards = int(np.ceil(cfg.n_clips / cfg.shard_size))
    done = 0
    started = time.time()
    for shard in range(n_shards):
        n = min(cfg.shard_size, cfg.n_clips - done)
        batch = generate_batch(n, cfg.seed, shard, stores, cfg.n_frames, cfg.fps)
        feats = extract_features(batch)
        table = build_table(batch.meta, feats)
        table.to_parquet(out / "features" / f"part_{shard:03d}.parquet", index=False)
        if shard < cfg.raw_sample_shards:
            batch.save(out / "raw" / f"clips_{shard:03d}")
        done += n
        if verbose:
            rate = done / max(time.time() - started, 1e-6)
            print(f"shard {shard + 1:>3}/{n_shards}  clips {done:>7,}  ({rate:,.0f} clips per second)", flush=True)
    manifest = {
        "n_clips": done,
        "n_frames_per_clip": cfg.n_frames,
        "fps": cfg.fps,
        "total_frames": done * cfg.n_frames,
        "hours_of_track_footage": round(done * cfg.n_frames / cfg.fps / 3600, 1),
        "n_stores": cfg.n_stores,
        "n_shards": n_shards,
        "seed": cfg.seed,
        "store_formats": {s.store_id: s.fmt for s in stores},
        "build_seconds": round(time.time() - started, 1),
    }
    if cfg.prefix_clips > 0:
        manifest["prefix_clips"] = cfg.prefix_clips
        manifest["prefix_rows"] = generate_prefix_features(cfg, root, verbose)
        manifest["build_seconds"] = round(time.time() - started, 1)
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2))
    return manifest


PREFIX_SHARD_OFFSET = 500  # prefix clips use their own shard numbers, so they never repeat corpus clips


def prefix_labels(scenario: np.ndarray, event_time_s: np.ndarray, seconds_seen: float) -> np.ndarray:
    """Label a partial clip by what has happened so far. Returns minus 1 to drop the row.

    A clip that will contain theft is still normal shopping until the act
    happens. Rows cut close to the act are ambiguous and are dropped.
    """
    scenario = np.asarray(scenario)
    e = np.asarray(event_time_s, dtype=float)
    out = np.full(len(scenario), -1, dtype=int)
    out[scenario == int(Scenario.NORMAL)] = int(Scenario.NORMAL)
    # (scenario, seconds before the act that are still clearly normal, seconds after it that are clearly theft)
    rules = (
        (Scenario.CONCEALMENT_BAG, 0.4, 0.8),
        (Scenario.CONCEALMENT_CLOTHING, 0.4, 0.8),
        (Scenario.SKIP_SCAN, 0.4, 0.8),
        (Scenario.TICKET_SWITCH, 2.0, 0.2),
        (Scenario.SHELF_SWEEP, None, 0.8),  # earlier removals already look like theft: never label them normal
        (Scenario.SWEETHEARTING, None, 0.8),
    )
    for scen, before, after in rules:
        m = (scenario == int(scen)) & np.isfinite(e)
        out[m & (seconds_seen >= e + after)] = int(scen)
        if before is not None:
            out[m & (seconds_seen <= e - before)] = int(Scenario.NORMAL)
    # At the doors the evidence (no payment, goods in view) exists from the first frame.
    out[scenario == int(Scenario.PUSH_OUT)] = int(Scenario.PUSH_OUT)
    return out


def generate_prefix_features(cfg: DataConfig, root: str | Path = ".", verbose: bool = True) -> int:
    """Build the streaming training set: features of partial clips with honest labels.

    A detector trained only on whole clips misreads partial ones (an item that
    has been picked up but not yet basketed looks unaccounted for). These rows
    teach it what "not finished yet" looks like.
    """
    from ..features.extractor import extract_features  # local import avoids a cycle

    out = Path(root) / cfg.out_dir / "prefix_features"
    out.mkdir(parents=True, exist_ok=True)
    stores = make_stores(cfg.n_stores, cfg.seed)
    n_shards = int(np.ceil(cfg.prefix_clips / cfg.shard_size))
    rows, done, chunk = 0, 0, 500
    for k in range(n_shards):
        n = min(cfg.shard_size, cfg.prefix_clips - done)
        shard = PREFIX_SHARD_OFFSET + k
        batch = generate_batch(n, cfg.seed, shard, stores, cfg.n_frames, cfg.fps)
        rng = np.random.default_rng([cfg.seed, shard, 555])
        parts = []
        for _ in range(cfg.prefixes_per_clip):
            for start in range(0, n, chunk):
                idx = np.arange(start, min(start + chunk, n))
                frames = int(rng.integers(10, cfg.n_frames - 3))
                sub = batch.select(idx).prefix(frames)
                table = build_table(sub.meta, extract_features(sub))
                table["prefix_frames"] = frames
                table["scenario_full_clip"] = table["scenario"]
                table["scenario"] = prefix_labels(table["scenario"].to_numpy(), table["event_time_s"].to_numpy(), frames / cfg.fps)
                parts.append(table[table["scenario"] >= 0])
        shard_table = pd.concat(parts, ignore_index=True)
        shard_table.to_parquet(out / f"part_{k:03d}.parquet", index=False)
        rows += len(shard_table)
        done += n
        if verbose:
            print(f"prefix shard {k + 1:>3}/{n_shards}  clips {done:>7,}  rows {rows:>8,}", flush=True)
    return rows


def load_prefix_features(root: str | Path, out_dir: str = "data/generated") -> pd.DataFrame | None:
    """Read the streaming training rows, or ``None`` when they were not built."""
    parts = sorted((Path(root) / out_dir / "prefix_features").glob("part_*.parquet"))
    return pd.concat([pd.read_parquet(p) for p in parts], ignore_index=True) if parts else None


def load_features(root: str | Path, out_dir: str = "data/generated") -> pd.DataFrame:
    """Read every feature shard into one table."""
    parts = sorted((Path(root) / out_dir / "features").glob("part_*.parquet"))
    if not parts:
        raise FileNotFoundError(f"no feature shards under {Path(root) / out_dir / 'features'}; run the data build first")
    return pd.concat([pd.read_parquet(p) for p in parts], ignore_index=True)
