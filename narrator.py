"""Incident narrators.

Two interchangeable narrators turn an evidence packet into an
:class:`IncidentBrief`:

* :class:`TemplateNarrator` is deterministic and runs offline. It is the
  default, the fallback, and what the test suite exercises.
* :class:`AnthropicNarrator` asks a Claude model to write the brief. Its output
  is parsed, validated against the schema and passed through the same language
  guard. Any failure falls back to the template, so a reviewer always gets a
  brief and never gets an unchecked one.
"""

from __future__ import annotations

import base64
import io
import json
import os
import re
from dataclasses import asdict, dataclass, field
from typing import Any, Protocol

from ..config import NarratorConfig
from ..schema import Scenario
from .playbook import PLAYBOOK
from .prompts import BRIEF_FIELDS, SYSTEM_PROMPT, build_user_prompt

# Words a brief must never contain: accusations and personal descriptors.
BANNED_TERMS: tuple[str, ...] = (
    "thief",
    "thieves",
    "stole",
    "stolen",
    "stealing",
    "shoplifter",
    "criminal",
    "offender",
    "culprit",
    "guilty",
    "perpetrator",
    "suspect is",
    "the suspect",
    "male",
    "female",
    "man",
    "woman",
    "boy",
    "girl",
    "elderly",
    "teenager",
    "youth",
    "ethnic",
    "race",
    "skin",
    "hoodie",
    "detain",
    "arrest",
    "search the",
    "confront",
)


def language_guard(text: str) -> list[str]:
    """Return the banned terms found in ``text`` (empty when the text is clean)."""
    lowered = text.lower()
    return [term for term in BANNED_TERMS if re.search(rf"\b{re.escape(term)}\b", lowered)]


@dataclass
class IncidentBrief:
    headline: str
    observed: list[str]
    benign_explanations: list[str]
    recommended_action: str
    confidence_note: str
    provider: str = "template"
    fallback_reason: str | None = None
    evidence: dict = field(default_factory=dict, repr=False)

    def text(self) -> str:
        return " ".join([self.headline, *self.observed, *self.benign_explanations, self.recommended_action, self.confidence_note])

    def to_dict(self, include_evidence: bool = False) -> dict:
        out = asdict(self)
        if not include_evidence:
            out.pop("evidence")
        return out

    def to_markdown(self) -> str:
        a = self.evidence.get("assessment", {})
        lines = [
            f"### {self.headline}",
            "",
            f"**Clip** {self.evidence.get('clip_id', 'n/a')}  |  **Camera** {self.evidence.get('camera', 'n/a')}  |  "
            f"**Tier** {a.get('tier', 'n/a')}  |  **Risk** {a.get('risk', 0):.2f}  |  **Narrator** {self.provider}",
            "",
            "**What was observed**",
            "",
            *[f"* {item}" for item in self.observed],
            "",
            "**Innocent explanations to rule out**",
            "",
            *[f"* {item}" for item in self.benign_explanations],
            "",
            f"**Recommended action** {self.recommended_action}",
            "",
            f"**Confidence** {self.confidence_note}",
        ]
        return "\n".join(lines)


class Narrator(Protocol):
    def narrate(self, evidence: dict, keyframes: list | None = None) -> IncidentBrief: ...


# --------------------------------------------------------------------------- #
# Template narrator
# --------------------------------------------------------------------------- #
def _confidence_note(evidence: dict) -> str:
    a = evidence["assessment"]
    q = evidence.get("data_quality", {})
    strength = "strong" if a["risk"] >= 0.9 else ("moderate" if a["risk"] >= 0.7 else "limited")
    note = f"Model risk is {a['risk']:.2f}, which is {strength} evidence for this scenario."
    if q.get("pose_confidence", 1.0) < 0.75 or q.get("arm_hidden_share", 0.0) > 0.25:
        note += " Pose tracking was partly obscured in this clip, so treat the evidence as weaker and watch the footage."
    else:
        note += " Pose tracking was clear throughout the clip."
    return note


class TemplateNarrator:
    """Deterministic narrator built from the evidence packet and the playbook."""

    provider = "template"

    def narrate(self, evidence: dict, keyframes: list | None = None) -> IncidentBrief:
        a = evidence["assessment"]
        scenario = Scenario(int(a["scenario_id"]))
        play = PLAYBOOK[scenario]
        if a["tier"] == "clear":
            return IncidentBrief(
                headline=f"No alert: behaviour on the {evidence['camera'].lower()} camera is consistent with normal shopping.",
                observed=[f"At {e['t']:.1f} s: {e['text']}." for e in evidence["timeline"][:3]] or ["No notable events were detected."],
                benign_explanations=[],
                recommended_action="No action needed.",
                confidence_note=f"Probability of normal behaviour is {a['probability_normal']:.2f}.",
                evidence=evidence,
            )
        timeline = evidence["timeline"]
        decisive = [e for e in timeline if e["kind"] in _DECISIVE]
        context = [e for e in timeline if e["kind"] == "item_taken"][:1] or [e for e in timeline if e["kind"] == "head_turn"][:1]
        chosen = sorted((context + decisive[:3]) or timeline[:3], key=lambda e: e["t"])
        observed = [f"At {e['t']:.1f} s: {e['text']}." for e in chosen]
        unusual = [d for d in evidence["drivers"] if d.get("differs_from_typical")]
        for d in unusual[:2]:
            observed.append(
                f"{d['label']} was {d['value']:g}; the typical value for normal clips on this camera is {d['typical']:g} "
                f"(range {d['typical_low']:g} to {d['typical_high']:g})."
            )
        action = play.priority_action if a["tier"] == "priority" else play.review_action
        return IncidentBrief(
            headline=f"{a['tier'].capitalize()} alert: possible {a['scenario'].lower()} on the {evidence['camera'].lower()} camera.",
            observed=observed[:5],
            benign_explanations=list(play.benign_explanations),
            recommended_action=action,
            confidence_note=_confidence_note(evidence),
            evidence=evidence,
        )


_DECISIVE = ("item_vanished", "unscanned_bagging", "suspect_scan", "tag_alarm", "exit_crossing")


# --------------------------------------------------------------------------- #
# Claude narrator
# --------------------------------------------------------------------------- #
def _encode_image(image: Any) -> dict:
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    return {
        "type": "image",
        "source": {"type": "base64", "media_type": "image/png", "data": base64.b64encode(buf.getvalue()).decode()},
    }


def parse_brief_json(text: str) -> dict:
    """Extract and validate the JSON object returned by the model."""
    cleaned = re.sub(r"```(?:json)?", "", text).strip()
    start, end = cleaned.find("{"), cleaned.rfind("}")
    if start < 0 or end <= start:
        raise ValueError("no JSON object in model reply")
    data = json.loads(cleaned[start : end + 1])
    missing = [k for k in BRIEF_FIELDS if k not in data]
    if missing:
        raise ValueError(f"brief is missing fields: {missing}")
    for key in ("observed", "benign_explanations"):
        if not isinstance(data[key], list) or not all(isinstance(x, str) for x in data[key]):
            raise ValueError(f"'{key}' must be a list of strings")
    for key in ("headline", "recommended_action", "confidence_note"):
        if not isinstance(data[key], str) or not data[key].strip():
            raise ValueError(f"'{key}' must be a non empty string")
    if not data["benign_explanations"]:
        raise ValueError("a brief must list at least one innocent explanation")
    return data


class AnthropicNarrator:
    """Narrator backed by the Anthropic Messages API.

    Parameters
    ----------
    client : optional preconfigured client. Anything with a compatible
        ``messages.create`` method works, which is how the tests inject a fake.
    model : model identifier. Defaults to ``SHELFSENTINEL_LLM_MODEL`` or the
        value in the config. Check the Anthropic models page for current names.
    """

    provider = "anthropic"

    def __init__(self, client: Any | None = None, model: str | None = None, max_tokens: int = 700, attach_keyframes: bool = True) -> None:
        if client is None:
            try:
                import anthropic
            except ImportError as exc:  # pragma: no cover - depends on the environment
                raise RuntimeError("install the optional dependency with: pip install 'shelfsentinel[llm]'") from exc
            client = anthropic.Anthropic()  # reads ANTHROPIC_API_KEY from the environment
        self.client = client
        self.model = model or os.environ.get("SHELFSENTINEL_LLM_MODEL", NarratorConfig().model)
        self.max_tokens = max_tokens
        self.attach_keyframes = attach_keyframes
        self._fallback = TemplateNarrator()

    def narrate(self, evidence: dict, keyframes: list | None = None) -> IncidentBrief:
        a = evidence["assessment"]
        if a["tier"] == "clear":
            return self._fallback.narrate(evidence)
        play = PLAYBOOK[Scenario(int(a["scenario_id"]))]
        action = play.priority_action if a["tier"] == "priority" else play.review_action
        content: list[dict] = []
        if self.attach_keyframes and keyframes:
            content.extend(_encode_image(img) for img in keyframes[:4])
        content.append({"type": "text", "text": build_user_prompt(evidence, action, play.benign_explanations)})
        try:
            reply = self.client.messages.create(
                model=self.model,
                max_tokens=self.max_tokens,
                system=SYSTEM_PROMPT,
                messages=[{"role": "user", "content": content}],
            )
            text = "".join(getattr(block, "text", "") for block in reply.content if getattr(block, "type", "") == "text")
            data = parse_brief_json(text)
            brief = IncidentBrief(
                headline=data["headline"].strip(),
                observed=[s.strip() for s in data["observed"]][:5],
                benign_explanations=[s.strip() for s in data["benign_explanations"]][:4],
                recommended_action=data["recommended_action"].strip(),
                confidence_note=data["confidence_note"].strip(),
                provider=f"anthropic:{self.model}",
                evidence=evidence,
            )
            violations = language_guard(brief.text())
            if violations:
                raise ValueError(f"language guard rejected the brief: {violations}")
            return brief
        except Exception as exc:  # any failure degrades to the deterministic brief
            brief = self._fallback.narrate(evidence)
            brief.fallback_reason = f"{type(exc).__name__}: {exc}"[:200]
            return brief


def get_narrator(cfg: NarratorConfig | None = None) -> Narrator:
    """Build the narrator named in the config, falling back to the template."""
    cfg = cfg or NarratorConfig()
    if cfg.provider == "anthropic" and os.environ.get("ANTHROPIC_API_KEY"):
        return AnthropicNarrator(model=cfg.model, max_tokens=cfg.max_tokens, attach_keyframes=cfg.attach_keyframes)
    return TemplateNarrator()
