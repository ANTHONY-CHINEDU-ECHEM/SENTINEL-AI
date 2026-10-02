"""Ask Claude to write the incident brief for one alert.

Needs the optional dependency and an API key::

    pip install "shelfsentinel[llm]"
    export ANTHROPIC_API_KEY=...
    python examples/claude_narrator.py

The reply is parsed, checked against the brief schema and passed through the
language guard. If anything fails the deterministic template brief is returned
and ``fallback_reason`` says why.
"""

from pathlib import Path

from shelfsentinel.explain import AnthropicNarrator, build_evidence, build_timeline
from shelfsentinel.features import extract_features
from shelfsentinel.models import SentinelModel
from shelfsentinel.schema import ClipBatch
from shelfsentinel.viz.render import render_frame
from shelfsentinel.viz.storyboard import pick_keyframes

ROOT = Path(__file__).resolve().parents[1]

model = SentinelModel.load(ROOT / "models" / "sentinel.joblib")
clips = ClipBatch.load(ROOT / "data" / "sample" / "clips_sample.npz")
assessment = model.assess(extract_features(clips))
index = int(assessment.index[assessment["tier"] != "clear"][0])

evidence = build_evidence(model, clips, index)
# Key frames are skeleton overlays, never photographs, so no personal data leaves the store.
keyframes = [render_frame(clips, index, e["frame"]) for e in pick_keyframes(build_timeline(clips, index), clips.n_frames)]

brief = AnthropicNarrator().narrate(evidence, keyframes)
print(brief.to_markdown())
if brief.fallback_reason:
    print("\nFell back to the template narrator:", brief.fallback_reason)
