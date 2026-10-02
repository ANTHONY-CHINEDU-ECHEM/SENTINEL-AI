import json
from types import SimpleNamespace

import pytest

from shelfsentinel.explain import AnthropicNarrator, TemplateNarrator, build_evidence, build_timeline, language_guard
from shelfsentinel.explain.narrator import parse_brief_json
from shelfsentinel.explain.playbook import PLAYBOOK
from shelfsentinel.schema import THEFT_CLASSES
from shelfsentinel.simulation import generate_batch

GOOD = {
    "headline": "Review alert: an item taken from the shelf was not accounted for.",
    "observed": ["At 4.2 s an item left the shelf.", "At 6.0 s the item was last seen at a personal bag."],
    "benign_explanations": ["The shopper may be using scan and go."],
    "recommended_action": "Ask a colleague to offer help in the aisle.",
    "confidence_note": "Model risk is high and pose tracking was clear.",
}


class FakeClient:
    """Stands in for anthropic.Anthropic in tests."""

    def __init__(self, text: str | None = None, error: Exception | None = None):
        self.text, self.error, self.calls = text, error, []
        self.messages = SimpleNamespace(create=self._create)

    def _create(self, **kwargs):
        self.calls.append(kwargs)
        if self.error:
            raise self.error
        return SimpleNamespace(content=[SimpleNamespace(type="text", text=self.text)])


@pytest.fixture(scope="module")
def alert_evidence(tiny_project):
    model = tiny_project["model"]
    clips = generate_batch(60, seed=21, subtypes=["concealment_bag", "skip_scan", "push_out"], quality="hd", crowding=0)
    for i in range(len(clips)):
        ev = build_evidence(model, clips, i)
        if ev["assessment"]["tier"] != "clear":
            return ev
    pytest.fail("the tiny model raised no alert on 60 theft probe clips")


def test_language_guard():
    assert language_guard("The tracked person placed an item in a bag.") == []
    assert "thief" in language_guard("The thief ran.")
    assert "woman" in language_guard("A woman in a red coat.")
    assert language_guard("manual scan and human review") == []  # whole words only


def test_playbook_covers_every_theft_scenario_and_is_clean():
    assert set(PLAYBOOK) == set(THEFT_CLASSES)
    for play in PLAYBOOK.values():
        text = " ".join([play.summary, *play.benign_explanations, play.review_action, play.priority_action, *play.verify])
        assert language_guard(text) == []
        assert len(play.benign_explanations) >= 2


def test_timeline_is_ordered_and_typed(batch):
    events = build_timeline(batch, 0)
    assert [e["frame"] for e in events] == sorted(e["frame"] for e in events)
    assert all({"t", "frame", "kind", "text", "weight"} <= set(e) for e in events)


def test_template_brief(alert_evidence):
    brief = TemplateNarrator().narrate(alert_evidence)
    assert brief.provider == "template"
    assert language_guard(brief.text()) == []
    assert brief.benign_explanations and brief.observed
    assert "possible" in brief.headline.lower()
    md = brief.to_markdown()
    assert "Innocent explanations" in md and alert_evidence["clip_id"] in md


def test_evidence_drivers_are_explainable(alert_evidence):
    assert alert_evidence["drivers"], "an alert should come with at least one driver"
    for d in alert_evidence["drivers"]:
        assert d["evidence_weight"] > 0
        assert d["typical_low"] <= d["typical_high"]
    json.dumps(alert_evidence)  # the packet must be serialisable


def test_claude_narrator_accepts_a_valid_brief(alert_evidence):
    client = FakeClient("```json\n" + json.dumps(GOOD) + "\n```")
    brief = AnthropicNarrator(client=client, model="test-model").narrate(alert_evidence)
    assert brief.provider == "anthropic:test-model"
    assert brief.fallback_reason is None
    assert brief.headline == GOOD["headline"]
    call = client.calls[0]
    assert call["model"] == "test-model" and "evidence packet" in call["system"].lower()
    assert alert_evidence["clip_id"] in call["messages"][0]["content"][-1]["text"]


@pytest.mark.parametrize(
    "reply",
    [
        json.dumps({**GOOD, "headline": "The thief stole a bottle."}),
        json.dumps({**GOOD, "observed": ["A young man in a hoodie took an item."]}),
        json.dumps({k: v for k, v in GOOD.items() if k != "benign_explanations"}),
        json.dumps({**GOOD, "benign_explanations": []}),
        "Sorry, I cannot help with that.",
    ],
)
def test_claude_narrator_falls_back_on_bad_output(alert_evidence, reply):
    brief = AnthropicNarrator(client=FakeClient(reply), model="test-model").narrate(alert_evidence)
    assert brief.provider == "template"
    assert brief.fallback_reason
    assert language_guard(brief.text()) == []


def test_claude_narrator_falls_back_on_api_error(alert_evidence):
    brief = AnthropicNarrator(client=FakeClient(error=TimeoutError("slow")), model="test-model").narrate(alert_evidence)
    assert brief.provider == "template" and "TimeoutError" in brief.fallback_reason


def test_parse_brief_json_strips_fences():
    assert parse_brief_json("Here you go:\n```json\n" + json.dumps(GOOD) + "\n```")["headline"] == GOOD["headline"]
    with pytest.raises(ValueError):
        parse_brief_json("{}")
