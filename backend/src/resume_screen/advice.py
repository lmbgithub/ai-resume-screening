"""Strong points, weak points, and suggestions for improving the CV.

The split here is the point of the module:

* **Strong and weak points are derived, not generated.** They are a
  deterministic reading of the per-requirement verdicts that already produced
  the score. Asking the model to "list the strengths" would let it name a
  strength the scoring never credited, and a report that contradicts its own
  number is worse than no report.
* **Suggestions are generated**, because advice is genuinely open-ended. They
  are the one part of the output the model invents, they are labelled as such,
  and they can never move the score.

A failed suggestion call returns no suggestions and says so. It never returns
plausible filler, because filler is indistinguishable from advice.
"""

from __future__ import annotations

import json
from dataclasses import dataclass

from .judge import _extract_json_object
from .ollama import Client, OllamaError

MAX_SUGGESTIONS = 6
MAX_SUGGESTION_CHARS = 400

SUGGESTIONS_PROMPT = """A candidate is applying for this role. An automated screen has
already decided which requirements their CV does and does not evidence.

Requirements the CV does NOT evidence:
{gaps}

Requirements the CV DOES evidence:
{strengths}

Write specific, actionable suggestions for improving the CV for this role.

Rules:
- Only suggest things the candidate can actually do: rewording, adding
  measurable detail, surfacing experience already implied, or naming a skill
  gap worth closing.
- Never suggest claiming experience the CV gives no sign of.
- Be concrete. "Add metrics" is useless; "state the request volume and latency
  your service handled" is useful.
- At most {limit} suggestions, ordered by impact.

Answer with JSON only, no prose:
{{"suggestions": ["<suggestion>", "..."]}}"""


@dataclass(frozen=True)
class Point:
    """One strong or weak point, traced back to the requirement behind it."""

    requirement: str
    kind: str  # "required" | "preferred"
    verdict: str
    detail: str
    evidence: str | None

    def to_dict(self) -> dict:
        return {
            "requirement": self.requirement,
            "kind": self.kind,
            "verdict": self.verdict,
            "detail": self.detail,
            "evidence": self.evidence,
        }


@dataclass(frozen=True)
class Suggestions:
    """Model-written advice, or an explicit statement that there is none."""

    items: tuple[str, ...]
    error: str | None = None

    def to_dict(self) -> dict:
        return {"items": list(self.items), "error": self.error}


#: The default for a result that carries no advice. A module-level singleton
#: because a dataclass field default may not be a function call.
NO_SUGGESTIONS = Suggestions(())


def _point(item) -> Point:
    return Point(
        requirement=item.requirement.text,
        kind=item.requirement.kind,
        verdict=item.verdict.verdict,
        detail=item.verdict.reason,
        evidence=item.evidence[0].chunk.text if item.evidence else None,
    )


def strong_points(results) -> tuple[Point, ...]:
    """Requirements the evidence supported.

    Ordered required-before-preferred, then by how specific the supporting
    line was — the strongest claim a candidate can make about this job first.
    """
    # The sort key travels with the point rather than living in a dict keyed
    # by requirement text: a job description may repeat a bullet verbatim, and
    # two points sharing one key is a silent ordering bug.
    ranked = [
        (
            (
                0 if item.requirement.kind == "required" else 1,
                -(item.evidence[0].centered if item.evidence else 0.0),
                item.requirement.index,
            ),
            _point(item),
        )
        for item in results
        if item.verdict.verdict == "covered"
    ]
    return tuple(point for _, point in sorted(ranked, key=lambda entry: entry[0]))


def weak_points(results) -> tuple[Point, ...]:
    """Requirements the evidence did not support.

    `absent` before `partial`, required before preferred: the ordering a
    candidate should read top-down when deciding what to fix first. An
    `unknown` verdict is excluded — the model failed there, and presenting our
    parser failure as the candidate's weakness would be a lie.
    """
    rank = {"absent": 0, "partial": 1}
    ranked = [
        (
            (
                0 if item.requirement.kind == "required" else 1,
                rank[item.verdict.verdict],
                item.requirement.index,
            ),
            _point(item),
        )
        for item in results
        if item.verdict.verdict in rank
    ]
    return tuple(point for _, point in sorted(ranked, key=lambda entry: entry[0]))


def build_prompt(gaps: tuple[Point, ...], strengths: tuple[Point, ...]) -> str:
    def render(points: tuple[Point, ...]) -> str:
        if not points:
            return "- (none)"
        return "\n".join(f"- [{p.kind}] {p.requirement}" for p in points)

    return SUGGESTIONS_PROMPT.format(
        gaps=render(gaps), strengths=render(strengths), limit=MAX_SUGGESTIONS
    )


def parse_suggestions(response: str) -> Suggestions:
    """Parse the model's advice, never raising and never inventing.

    Anything unparseable becomes zero suggestions with the reason attached, so
    the UI can say "the model did not return usable advice" instead of showing
    an empty list that looks like "your CV is perfect".
    """
    blob = _extract_json_object(response)
    if blob is None:
        return Suggestions((), f"no JSON object in model output: {response.strip()[:120]!r}")
    try:
        parsed = json.loads(blob)
    except json.JSONDecodeError as exc:
        return Suggestions((), f"malformed JSON ({exc.msg})")
    if not isinstance(parsed, dict):
        return Suggestions((), f"expected a JSON object, got {type(parsed).__name__}")
    raw = parsed.get("suggestions")
    if not isinstance(raw, list):
        return Suggestions((), f"'suggestions' was {type(raw).__name__}, expected a list")

    items: list[str] = []
    for entry in raw:
        # Models sometimes return objects here instead of strings; take the
        # obvious field rather than rendering a dict into the UI.
        if isinstance(entry, dict):
            entry = entry.get("suggestion") or entry.get("text") or ""
        if not isinstance(entry, str):
            continue
        cleaned = " ".join(entry.split())[:MAX_SUGGESTION_CHARS]
        if cleaned:
            items.append(cleaned)
    if not items:
        return Suggestions((), "the model returned no usable suggestions")
    return Suggestions(tuple(items[:MAX_SUGGESTIONS]))


def suggest(client: Client, gaps: tuple[Point, ...], strengths: tuple[Point, ...]) -> Suggestions:
    """Ask for advice. A backend failure is reported, not raised."""
    if not gaps and not strengths:
        return Suggestions((), "there were no requirements to advise on")
    try:
        response = client.generate(build_prompt(gaps, strengths), json_mode=True)
    except OllamaError as exc:
        return Suggestions((), f"model call failed: {exc}")
    return parse_suggestions(response)
