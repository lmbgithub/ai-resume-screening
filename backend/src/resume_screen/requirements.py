"""Parsing a job description into individually scorable requirements.

A job description is not one thing to match against. It is a list of claims,
some of which are disqualifying and some of which are decoration.

Two paths, in this order:

1. **Structure, when the posting has it.** Bullets under a heading, with the
   heading deciding required vs preferred. Deterministic, free, and exactly
   reproducible — so it is tried first and used whenever it finds anything.
2. **The model, when it does not.** Real postings are prose pasted out of a
   job board with no Markdown in sight. Refusing those was a rule that only
   worked on job descriptions someone had already tidied up by hand.

The rule-based path stays free of any model: `parse_requirements` takes text
and nothing else. Only `find_requirements` knows a client exists, and the
result records which path produced it, because a model-extracted requirement
list is not reproducible the way a parsed one is.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass

from .documents import chunk_document
from .judge import _extract_json_object
from .ollama import Client, OllamaError

REQUIRED_WEIGHT = 1.0
PREFERRED_WEIGHT = 0.5

_PREFERRED_HINTS = (
    "preferred",
    "nice to have",
    "nice-to-have",
    "bonus",
    "plus",
    "desirable",
    "optional",
)


@dataclass(frozen=True)
class Requirement:
    """One line of a job description, with the weight its section implies."""

    text: str
    kind: str  # "required" | "preferred"
    section: str
    index: int

    @property
    def weight(self) -> float:
        return REQUIRED_WEIGHT if self.kind == "required" else PREFERRED_WEIGHT


def classify_section(title: str) -> str:
    """Map a job-description heading to a requirement kind.

    Only "preferred" has to be recognised. Everything else is required,
    including a heading nobody anticipated, because under-weighting a genuine
    requirement is the worse error. An earlier version also matched a list of
    "required" hints and then returned the same value as the fallback did,
    which read like a decision but changed nothing.
    """
    lowered = title.lower()
    if any(hint in lowered for hint in _PREFERRED_HINTS):
        return "preferred"
    return "required"


def parse_requirements(text: str) -> list[Requirement]:
    """Extract requirements from job-description text.

    A requirement is a *bullet under a heading*. Prose is dropped, headed or
    not: the "About us" paragraph every posting opens with is not a
    requirement, and scoring a candidate against a company's self-description
    adds a line every candidate fails identically.
    """
    requirements: list[Requirement] = []
    for chunk in chunk_document(text, default_section=""):
        if not chunk.section or not chunk.bullet:
            continue
        if not _looks_like_requirement(chunk.text):
            continue
        requirements.append(
            Requirement(
                text=chunk.text,
                kind=classify_section(chunk.section),
                section=chunk.section,
                index=len(requirements),
            )
        )
    return requirements


def _looks_like_requirement(text: str) -> bool:
    """Filter out boilerplate that survived chunking."""
    if len(text) < 12:
        return False
    words = re.findall(r"[A-Za-z]+", text)
    return len(words) >= 3


MAX_MODEL_REQUIREMENTS = 30

EXTRACTION_PROMPT = """Extract the individual requirements from this job posting.

Posting:
{posting}

Rules:
- One requirement per entry, in the posting's own words. Do not invent,
  merge, generalise, or rewrite them into something the posting does not say.
- Classify each as "preferred" ONLY where the posting marks it optional
  (preferred, nice to have, bonus, a plus, desirable). Everything else is
  "required".
- Ignore company description, culture, benefits, salary, location, equal
  opportunity statements and application instructions. Those are not
  requirements and scoring a candidate against them helps nobody.
- At most {limit} entries.

Answer with JSON only, no prose:
{{"requirements": [{{"text": "<requirement>", "kind": "required"}}]}}"""


@dataclass(frozen=True)
class RequirementSet:
    """Requirements, and how they were obtained.

    `source` travels with them so a caller can say whether the list is
    reproducible. `error` is set when the model path was tried and failed —
    never silently swallowed into an empty list.
    """

    requirements: tuple[Requirement, ...]
    source: str  # "headings" | "model"
    error: str | None = None

    def __bool__(self) -> bool:
        return bool(self.requirements)


def build_extraction_prompt(text: str) -> str:
    return EXTRACTION_PROMPT.format(posting=text.strip(), limit=MAX_MODEL_REQUIREMENTS)


def parse_extracted(response: str) -> tuple[tuple[Requirement, ...], str | None]:
    """Turn the model's extraction into Requirements, never raising.

    An unparseable answer becomes no requirements plus a reason, so the caller
    reports "the model could not read this posting" rather than the misleading
    "this posting contains no requirements".
    """
    blob = _extract_json_object(response)
    if blob is None:
        return (), f"no JSON object in model output: {response.strip()[:120]!r}"
    try:
        parsed = json.loads(blob)
    except json.JSONDecodeError as exc:
        return (), f"malformed JSON ({exc.msg})"
    if not isinstance(parsed, dict):
        return (), f"expected a JSON object, got {type(parsed).__name__}"
    raw = parsed.get("requirements")
    if not isinstance(raw, list):
        return (), f"'requirements' was {type(raw).__name__}, expected a list"

    found: list[Requirement] = []
    for entry in raw:
        # Models return bare strings here about as often as objects.
        if isinstance(entry, str):
            entry = {"text": entry}
        if not isinstance(entry, dict):
            continue
        text = entry.get("text") or entry.get("requirement")
        if not isinstance(text, str):
            continue
        cleaned = " ".join(text.split())
        if not _looks_like_requirement(cleaned):
            continue
        kind = entry.get("kind")
        # Anything unrecognised falls to "required": under-weighting a real
        # requirement is the worse error, exactly as for an unknown heading.
        kind = kind.strip().lower() if isinstance(kind, str) else ""
        found.append(
            Requirement(
                text=cleaned,
                kind="preferred" if kind == "preferred" else "required",
                section="extracted by model",
                index=len(found),
            )
        )
        if len(found) >= MAX_MODEL_REQUIREMENTS:
            break
    if not found:
        return (), "the model returned no usable requirements"
    return tuple(found), None


def find_requirements(client: Client, text: str) -> RequirementSet:
    """Structure first, the model only when structure finds nothing."""
    parsed = parse_requirements(text)
    if parsed:
        return RequirementSet(tuple(parsed), source="headings")
    try:
        response = client.generate(build_extraction_prompt(text), json_mode=True)
    except OllamaError as exc:
        return RequirementSet((), source="model", error=f"model call failed: {exc}")
    requirements, error = parse_extracted(response)
    return RequirementSet(requirements, source="model", error=error)
