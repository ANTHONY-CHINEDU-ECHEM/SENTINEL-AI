"""Command line interface.

Run ``shelfsentinel <command>`` or ``python -m shelfsentinel <command>``.

Commands
--------
data      build the synthetic corpus as parquet feature shards
train     fit, calibrate and evaluate the detector; write reports/metrics.json
charts    draw the report charts into docs/images
gallery   render scenario storyboards, replays and incident briefs
brief     print the incident brief for one clip of the raw sample
all       data, train, charts and gallery in sequence
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .config import load_config


def _model(root: Path, cfg):
    from .models.sentinel import SentinelModel

    path = root / cfg.model.out_dir / "sentinel.joblib"
    if not path.exists():
        sys.exit(f"no trained model at {path}; run the train command first")
    return SentinelModel.load(path)


def cmd_data(args, cfg) -> None:
    from .simulation.generator import generate_dataset

    manifest = generate_dataset(cfg.data, args.root)
    print(json.dumps({k: v for k, v in manifest.items() if k != "store_formats"}, indent=2))


def cmd_train(args, cfg) -> None:
    from .pipeline import run_training

    metrics = run_training(cfg, args.root, light=args.light)
    c, r = metrics["model"]["classification"], metrics["model"]["review_tier"]
    if r["precision"] is None:
        print(f"macro F1 {c['macro_f1']:.3f}   no review alerts: every scenario was switched off by the alert policy")
        print("the validation set is too small for the precision target; use more clips or lower policy.phantom_false_alerts")
    else:
        print(f"macro F1 {c['macro_f1']:.3f}   review tier precision {r['precision']:.3f}   recall {r['recall']:.3f}")


def cmd_charts(args, cfg) -> None:
    from .viz.charts import make_all_charts

    for path in make_all_charts(args.root):
        print(path)


def cmd_gallery(args, cfg) -> None:
    from .explain.narrator import get_narrator
    from .viz.gallery import build_gallery

    build_gallery(_model(Path(args.root), cfg), args.root, seed=cfg.data.seed, narrator=get_narrator(cfg.narrator), gifs=not args.no_gifs)


def cmd_brief(args, cfg) -> None:
    from .explain.evidence import build_evidence
    from .explain.narrator import get_narrator
    from .schema import ClipBatch

    root = Path(args.root)
    sample = Path(args.clips) if args.clips else root / "data" / "sample" / "clips_sample.npz"
    batch = ClipBatch.load(sample)
    evidence = build_evidence(_model(root, cfg), batch, args.index)
    brief = get_narrator(cfg.narrator).narrate(evidence)
    print(brief.to_markdown())


def cmd_all(args, cfg) -> None:
    args.light, args.no_gifs = False, False
    for step in (cmd_data, cmd_train, cmd_charts, cmd_gallery):
        step(args, cfg)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="shelfsentinel", description="Shelf Sentinel AI command line")
    parser.add_argument("--config", default=None, help="path to a YAML config (defaults are used when omitted)")
    parser.add_argument("--root", default=".", help="project root where data, models and reports live")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("data", help="build the synthetic corpus")
    p_train = sub.add_parser("train", help="train and evaluate")
    p_train.add_argument("--light", action="store_true", help="skip ablation, importance and replay")
    sub.add_parser("charts", help="draw report charts")
    p_gallery = sub.add_parser("gallery", help="render scenario storyboards")
    p_gallery.add_argument("--no-gifs", action="store_true", help="skip animated replays")
    p_brief = sub.add_parser("brief", help="incident brief for one sample clip")
    p_brief.add_argument("index", type=int, nargs="?", default=0)
    p_brief.add_argument("--clips", default=None, help="path to a clip batch saved with ClipBatch.save")
    sub.add_parser("all", help="run the whole pipeline")
    args = parser.parse_args(argv)
    cfg = load_config(args.config)
    {"data": cmd_data, "train": cmd_train, "charts": cmd_charts, "gallery": cmd_gallery, "brief": cmd_brief, "all": cmd_all}[args.command](args, cfg)


if __name__ == "__main__":
    main()
