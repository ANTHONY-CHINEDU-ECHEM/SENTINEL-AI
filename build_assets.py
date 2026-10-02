"""Rebuild the small committed assets that are derived from code.

* ``data/sample/clips_sample.npz``: 64 raw clips, two per behaviour, for the
  quick start, the ``brief`` command and anyone who wants to look at raw data.
* ``docs/feature_dictionary.md`` and ``docs/scenario_playbook.md``: generated
  from the feature catalog, the behaviour registry and the playbook so the
  documentation can never drift from the code.
* ``docs/images/architecture.png``: the system diagram.
* ``docs/model_card.md``: written from ``reports/metrics.json``.

Run from the project root: ``python scripts/build_assets.py``
"""

from __future__ import annotations

import json
from pathlib import Path

from PIL import Image, ImageDraw

from shelfsentinel.explain.playbook import PLAYBOOK
from shelfsentinel.features import FEATURE_NAMES
from shelfsentinel.features.catalog import feature_dictionary_markdown
from shelfsentinel.schema import SCENARIO_SCENE, SCENARIO_TITLES, SCENE_TITLES, Scenario
from shelfsentinel.simulation import REGISTRY, generate_batch
from shelfsentinel.viz.render import AMBER, GRID, INK, MINT, MUTED, PANEL, SKY, TEXT, VIOLET, font

ROOT = Path(__file__).resolve().parents[1]


def build_sample() -> None:
    names = list(REGISTRY)
    batch = generate_batch(len(names) * 2, seed=2026, shard=700, subtypes=names)
    batch.save(ROOT / "data" / "sample" / "clips_sample")
    print(f"sample: {len(batch)} clips, {batch.n_frames} frames each")


def build_playbook() -> None:
    lines = [
        "# Scenario playbook",
        "",
        "What each alert means, the innocent explanations a reviewer must rule out, and the proportionate response.",
        "The system never accuses anyone. An alert asks a trained colleague to look.",
        "",
    ]
    for scenario in Scenario:
        if scenario == Scenario.NORMAL:
            continue
        play = PLAYBOOK[scenario]
        lines += [
            f"## {SCENARIO_TITLES[scenario]}",
            "",
            f"**Camera context** {SCENE_TITLES[SCENARIO_SCENE[scenario]]}",
            "",
            play.summary,
            "",
            "**Innocent explanations to rule out**",
            "",
            *[f"* {b}" for b in play.benign_explanations],
            "",
            f"**Review tier response** {play.review_action}",
            "",
            f"**Priority tier response** {play.priority_action}",
            "",
            "**Checks for the reviewer**",
            "",
            *[f"* {v}" for v in play.verify],
            "",
        ]
    lines += ["## Benign behaviours the detector is trained to ignore", "", "| Camera | Behaviour | Why it is in the corpus |", "| :-- | :-- | :-- |"]
    for spec in REGISTRY.values():
        if spec.scenario == Scenario.NORMAL:
            lines.append(f"| {SCENE_TITLES[spec.scene]} | `{spec.name}` | {spec.description} |")
    (ROOT / "docs" / "scenario_playbook.md").write_text("\n".join(lines) + "\n")
    (ROOT / "docs" / "feature_dictionary.md").write_text(feature_dictionary_markdown() + "\n")
    print("docs: scenario_playbook.md, feature_dictionary.md")


def build_architecture() -> None:
    ss = 2
    w, h = 1500 * ss, 620 * ss
    img = Image.new("RGB", (w, h), INK)
    d = ImageDraw.Draw(img)
    f_t, f_b, f_s, f_m = font("bold", 17 * ss), font("bold", 15 * ss), font("sans", 13 * ss), font("mono", 13 * ss)
    d.text((36 * ss, 26 * ss), "SHELF SENTINEL AI", font=f_m, fill=MINT)
    d.text((36 * ss, 48 * ss), "From anonymous signals to a reviewed alert", font=font("bold", 26 * ss), fill=TEXT)

    def box(x, y, bw, bh, title, lines, colour):
        d.rounded_rectangle([x * ss, y * ss, (x + bw) * ss, (y + bh) * ss], radius=12 * ss, fill=PANEL, outline=colour, width=2 * ss)
        d.text(((x + 16) * ss, (y + 14) * ss), title, font=f_b, fill=colour)
        for i, line in enumerate(lines):
            d.text(((x + 16) * ss, (y + 44 + i * 21) * ss), line, font=f_s, fill=TEXT if i == 0 else MUTED)

    def arrow(x0, y0, x1, y1):
        d.line([x0 * ss, y0 * ss, x1 * ss, y1 * ss], fill=MUTED, width=2 * ss)
        if y0 == y1:
            d.polygon([(x1 * ss, y1 * ss), ((x1 - 9) * ss, (y1 - 6) * ss), ((x1 - 9) * ss, (y1 + 6) * ss)], fill=MUTED)
        else:
            d.polygon([(x1 * ss, y1 * ss), ((x1 - 6) * ss, (y1 - 9) * ss), ((x1 + 6) * ss, (y1 - 9) * ss)], fill=MUTED)

    top, bh, bw, gap = 120, 190, 256, 36
    xs = [36 + i * (bw + gap) for i in range(5)]
    box(xs[0], top, bw, bh, "1  Perception", ["Runs at the store edge", "pose keypoints (17 joints)", "item in hand, basket, bag", "shelf sensor events", "till and tag gate feed"], SKY)
    box(xs[1], top, bw, bh, "2  Clip schema", ["One tracked person, 12.8 s", "keypoints  (64, 17, 3)", "signals    (64, 15)", "journey context", "no pixels, no faces"], SKY)
    box(xs[2], top, bw, bh, "3  Features", ["62 readable measures", "items unaccounted for", "bagged but not scanned", "product and price match", "hand at bag or waist"], AMBER)
    box(xs[3], top, bw, bh, "4  Detector", ["Gradient boosted trees", "8 classes, 7 theft scenarios", "probabilities checked", "for calibration", "rule baseline as fallback"], AMBER)
    box(xs[4], top, bw, bh, "5  Alert policy", ["Precision first thresholds", "set at the real base rate", "clear, review or priority", "per scenario", "alert budget aware"], AMBER)
    for i in range(4):
        arrow(xs[i] + bw, top + bh / 2, xs[i + 1], top + bh / 2)

    low = 400
    box(xs[4], low, bw, bh - 20, "6  Evidence packet", ["Closed set of facts", "timeline of detected events", "top drivers vs normal range", "data quality flags"], MINT)
    box(xs[3] - 60, low, bw + 60, bh - 20, "7  LLM narrator", ["Claude writes a short brief", "from the packet only", "language guard and schema check", "template fallback, always on"], MINT)
    box(xs[1] + 40, low, bw + 150, bh - 20, "8  Human reviewer", ["Reads the brief, watches the storyboard", "rules out innocent explanations", "chooses a service led response", "decision is logged for retraining"], VIOLET)
    def arrow_left(x_from, x_to, y):
        d.line([x_from * ss, y * ss, x_to * ss, y * ss], fill=MUTED, width=2 * ss)
        d.polygon([(x_to * ss, y * ss), ((x_to + 9) * ss, (y - 6) * ss), ((x_to + 9) * ss, (y + 6) * ss)], fill=MUTED)

    arrow(xs[4] + bw / 2, top + bh, xs[4] + bw / 2, low)
    arrow_left(xs[4], xs[3] + bw, low + 85)
    arrow_left(xs[3] - 60, xs[1] + 40 + bw + 150, low + 85)
    d.line([36 * ss, (low + 85) * ss, (xs[1] + 40) * ss, (low + 85) * ss], fill=GRID, width=2 * ss)
    d.line([36 * ss, (low + 85) * ss, 36 * ss, (top + bh + 8) * ss], fill=GRID, width=2 * ss)
    d.text((48 * ss, (low + 58) * ss), "feedback: reviewed outcomes", font=f_s, fill=MUTED)
    d.text((48 * ss, (low + 96) * ss), "become training labels", font=f_s, fill=MUTED)
    d.text((36 * ss, (h / ss - 34) * ss), "Steps 1 and 2 are simulated in this repository; steps 3 to 8 are the code you can run.", font=f_t, fill=MUTED)
    img.resize((w // ss, h // ss), Image.LANCZOS).save(ROOT / "docs" / "images" / "architecture.png", optimize=True)
    print("docs: images/architecture.png")


def build_model_card() -> None:
    """Write docs/model_card.md from reports/metrics.json so the card always matches the run."""
    path = ROOT / "reports" / "metrics.json"
    if not path.exists():
        print("model card skipped: run the train command first")
        return
    m = json.loads(path.read_text())
    mod, rules, data = m["model"], m["rules"], m["dataset"]
    cls, rev, pri = mod["classification"], mod["review_tier"], mod["priority_tier"]
    cfg = m["config"]

    def pct(x: float | None) -> str:
        return "n/a" if x is None else f"{x * 100:.1f}%"

    lines = [
        "# Model card: Shelf Sentinel detector",
        "",
        "This card is generated from `reports/metrics.json` by `scripts/build_assets.py`, so it always describes the committed model.",
        "",
        "## Overview",
        "",
        "| Item | Detail |",
        "| :-- | :-- |",
        "| Task | Classify a 12.8 second single person track clip as normal shopping or one of seven theft scenarios |",
        "| Model | Histogram gradient boosted trees (scikit learn), eight classes |",
        f"| Inputs | {len(FEATURE_NAMES)} interpretable features from pose, item flow, point of sale and context |",
        "| Outputs | Class probabilities, the most likely theft scenario, its risk, and a tier: clear, review or priority |",
        f"| Training data | {data['clips_by_split']['train']:,} complete clips from {data['stores_by_split']['train']} simulated stores, plus partial clips for streaming aware training |",
        f"| Boosting iterations | {mod['n_iter']} (early stopping) |",
        f"| Calibration | Isotonic calibration tested on validation stores and {'kept' if mod['calibration']['isotonic_kept'] else 'rejected'}; test calibration error {mod['calibration']['test_ece_final']:.4f} |",
        f"| Alert policy | Per scenario thresholds for {cfg['policy']['review_precision_target']:.0%} (review) and {cfg['policy']['priority_precision_target']:.0%} (priority) precision at a theft rate of {cfg['policy']['deployment_theft_prevalence']:.1%}, with a safety margin of {cfg['policy']['phantom_false_alerts']:g} phantom false alert on the review tier |",
        "| Artefact | `models/sentinel.joblib` (estimator, policy and per camera reference statistics) |",
        "",
        "## Intended use",
        "",
        "Ranking clips for human review inside a loss prevention workflow, with the evidence packet and incident brief. The model must not be used to make automated decisions about a person. See [responsible use](responsible_use.md).",
        "",
        "## Evaluation data",
        "",
        f"{data['clips_by_split']['test']:,} clips from {data['stores_by_split']['test']} stores that were never used for training, calibration or thresholds. Alert metrics are reweighted so that theft is {cfg['policy']['deployment_theft_prevalence']:.1%} of clips.",
        "",
        "## Classification by most likely class",
        "",
        f"Accuracy {cls['accuracy']:.4f}, macro F1 {cls['macro_f1']:.4f}, macro F1 over the seven theft scenarios {cls['theft_macro_f1']:.4f}, theft against normal ROC AUC {mod['ranking']['theft_roc_auc']:.4f}, average precision at the deployment rate {mod['ranking']['theft_ap_deployment']:.4f}.",
        "",
        "| Scenario | Precision | Recall | F1 | Test clips |",
        "| :-- | --: | --: | --: | --: |",
    ]
    lines += [f"| {k} | {pct(v['precision'])} | {pct(v['recall'])} | {v['f1']:.3f} | {v['support']:,} |" for k, v in cls["per_class"].items()]
    lines += [
        "",
        "## Alert tiers at the deployment theft rate",
        "",
        "| Measure | Hand written rules | Review tier | Priority tier |",
        "| :-- | --: | --: | --: |",
        f"| Precision | {pct(rules['alerts']['precision'])} | {pct(rev['precision'])} | {pct(pri['precision'])} |",
        f"| Recall | {pct(rules['alerts']['recall'])} | {pct(rev['recall'])} | {pct(pri['recall'])} |",
        f"| Alerts per 10,000 clips | {rules['alerts']['alerts_per_10k_clips']} | {rev['alerts_per_10k_clips']} | {pri['alerts_per_10k_clips']} |",
        f"| False alerts per 10,000 honest clips | {rules['alerts']['false_alerts_per_10k_normal']} | {rev['false_alerts_per_10k_normal']} | {pri['false_alerts_per_10k_normal']} |",
        "",
        "| Scenario | Review recall | Review precision | Priority recall | Priority precision | Review threshold |",
        "| :-- | --: | --: | --: | --: | --: |",
    ]
    for k in rev["per_class"]:
        r, q = rev["per_class"][k], pri["per_class"][k]
        lines.append(f"| {k} | {pct(r['recall'])} | {pct(r['precision'])} | {pct(q['recall'])} | {pct(q['precision'])} | {mod['policy']['review'][k]:.2f} |")
    if "streaming" in m:
        a, b = m["streaming"]["streaming_aware_model"], m["streaming"].get("whole_clip_only_model")
        lines += [
            "",
            "## Scoring clips as they unfold",
            "",
            f"Measured on {a['clips']:,} fresh clips from the test stores, replayed frame by frame. A live alert must hold for {a['confirm_evaluations']} evaluations in a row.",
            "",
            "| Measure | Trained on whole clips only | Streaming aware (shipped) |",
            "| :-- | --: | --: |",
        ]
        if b:
            lines += [
                f"| False alerts per 1,000 honest clips, scored once complete | {b['whole_clip_false_alerts_per_1000']} | {a['whole_clip_false_alerts_per_1000']} |",
                f"| False alerts per 1,000 honest clips, live, unconfirmed | {b['streaming_false_alerts_unconfirmed_per_1000']} | {a['streaming_false_alerts_unconfirmed_per_1000']} |",
                f"| False alerts per 1,000 honest clips, live, confirmed | {b['streaming_false_alerts_per_1000']} | {a['streaming_false_alerts_per_1000']} |",
                f"| Theft recall, live, confirmed | {pct(b['streaming_theft_recall'])} | {pct(a['streaming_theft_recall'])} |",
            ]
    if "detection_latency" in m:
        lines += ["", "| Scenario | Caught inside the clip | Median seconds from act to alert | 90th percentile |", "| :-- | --: | --: | --: |"]
        def when(x: float) -> str:
            return f"{abs(x):.1f} {'before' if x < 0 else 'after'}"

        lines += [f"| {r['scenario']} | {pct(r['detected_in_clip'])} | {when(r['median_latency_s'])} | {when(r['p90_latency_s'])} |" for r in m["detection_latency"]]
        lines += ["", "Before means the alert fired ahead of the moment the act completed."]
    lines += [
        "",
        "## Factors that change performance",
        "",
        "See `reports/tables/robustness_slices.csv`. Low light cameras reduce recall and raise false alerts; crowding has a smaller effect; store format has little effect. Precision depends strongly on the true theft rate (`reports/tables/precision_vs_prevalence.csv`).",
        "",
        "## Limitations",
        "",
        "* Trained and tested on synthetic data. Real world accuracy is unknown.",
        "* Thresholds are fitted to rare events and will need refitting per store group on real data.",
        "* Features assume a correct single person track and calibrated camera zones.",
        "* Capture conditions (camera quality, crowding, hour) are deliberately not model inputs, so the model cannot compensate for them explicitly.",
        "* No demographic attributes exist in the data, so fairness across groups of people cannot be measured here and must be tested on real data before any deployment.",
        "",
        "## Reproduce",
        "",
        "```bash",
        "make train",
        "python scripts/build_assets.py",
        "```",
        "",
    ]
    (ROOT / "docs" / "model_card.md").write_text("\n".join(lines))
    print("docs: model_card.md")


if __name__ == "__main__":
    build_sample()
    build_playbook()
    build_architecture()
    build_model_card()
