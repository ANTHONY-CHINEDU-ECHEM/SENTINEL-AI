"""Score a handful of clips and print an incident brief.

Run from the project root after ``make install``::

    python examples/quickstart.py
"""

from pathlib import Path

from shelfsentinel.explain import TemplateNarrator, build_evidence
from shelfsentinel.features import extract_features
from shelfsentinel.models import SentinelModel
from shelfsentinel.schema import ClipBatch

ROOT = Path(__file__).resolve().parents[1]

model = SentinelModel.load(ROOT / "models" / "sentinel.joblib")
clips = ClipBatch.load(ROOT / "data" / "sample" / "clips_sample.npz")

# 1. Perception streams become interpretable features.
features = extract_features(clips)

# 2. Features become calibrated probabilities, a flagged scenario and a tier.
assessment = model.assess(features)
print(assessment[["flag_title", "risk", "tier"]].assign(truth=clips.meta["subtype"]).head(12).to_string())

# 3. Every alert comes with an evidence packet and a neutral incident brief.
alerts = assessment.index[assessment["tier"] != "clear"]
if len(alerts):
    evidence = build_evidence(model, clips, int(alerts[0]))
    print()
    print(TemplateNarrator().narrate(evidence).to_markdown())
