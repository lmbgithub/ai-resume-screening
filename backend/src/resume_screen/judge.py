"""Asking the model one narrow question, and refusing to guess at its answer.

The model is never asked "score this candidate". It is asked, per requirement,
whether the supplied resume lines demonstrate it — a question with a
checkable answer, on evidence the caller has already selected. The arithmetic
stays in `scoring`, where it is deterministic and testable.
"""

from __future__ import annotations

import json
from dataclasses import dataclass

from .ollama import Client, OllamaError

VALID_VERDICTS = ("covered", "partial", "absent")
UNKNOWN = "unknown"

PROMPT = """You are screening one requirement against evidence from a resume.

Requirement:
{requirement}

Resume lines (the only evidence you may use):
{evidence}

Answer with JSON only, no prose, exactly these keys:
{{"verdict": "covered" | "partial" | "absent", "reason": "<one short sentence>"}}

"covered": the lines directly demonstrate the requirement.
"partial": related or adjacent experience, but not the requirement itself.
"absent": the lines do not support the requirement.
Judge only what the lines say. Do not infer from the candidate's job titles,
employers, or schools."""


@dataclass(frozen=True)
class Verdict:
    """One requirement's outcome. `verdict` is UNKNOWN when parsing failed."""

    verdict: str
    reason: str

    @property
    def is_known(self) -> bool:
        return self.verdict in VALID_VERDICTS


def _extract_json_object(text: str) -> str | None:
    """Find the first balanced top-level JSON object in a model response.

    Models wrap JSON in prose or fences even when told not to. Brace counting
    is used rather than a regex because a regex cannot match nested braces,
    and the reason strings do contain them.
    """
    depth = 0
    start = -1
    in_string = False
    escaped = False
    for i, ch in enumerate(text):
        if in_string:
            if escaped:
                escaped = False
            elif ch == "\\":
                escaped = True
            elif ch == '"':
                in_string = False
            continue
        if ch == '"':
            in_string = True
        elif ch == "{":
            if depth == 0:
                start = i
            depth += 1
        elif ch == "}":
            if depth == 0:
                continue
            depth -= 1
            if depth == 0:
                return text[start : i + 1]
    return None


def parse_verdict(response: str) -> Verdict:
    """Turn raw model output into a Verdict, never raising.

    Every failure lands on UNKNOWN with the reason recorded, because a
    screening run must be able to report "the model was unintelligible here"
    instead of silently producing a rejection.
    """
    blob = _extract_json_object(response)
    if blob is None:
        return Verdict(UNKNOWN, f"no JSON object in model output: {response.strip()[:120]!r}")
    try:
        parsed = json.loads(blob)
    except json.JSONDecodeError as exc:
        return Verdict(UNKNOWN, f"malformed JSON ({exc.msg})")
    if not isinstance(parsed, dict):
        return Verdict(UNKNOWN, f"expected a JSON object, got {type(parsed).__name__}")
    raw_verdict = parsed.get("verdict")
    # `isinstance(True, int)` is True in Python, and models do emit bare
    # booleans here; the type check must reject anything that is not a string.
    if not isinstance(raw_verdict, str):
        return Verdict(UNKNOWN, f"'verdict' was {type(raw_verdict).__name__}, expected string")
    normalised = raw_verdict.strip().lower()
    if normalised not in VALID_VERDICTS:
        return Verdict(UNKNOWN, f"unrecognised verdict {raw_verdict!r}")
    reason = parsed.get("reason")
    return Verdict(normalised, reason.strip() if isinstance(reason, str) else "")


def build_prompt(requirement: str, evidence: list[str]) -> str:
    lines = "\n".join(f"- {line}" for line in evidence) if evidence else "- (none found)"
    return PROMPT.format(requirement=requirement, evidence=lines)


def judge(client: Client, requirement: str, evidence: list[str]) -> Verdict:
    """Score one requirement. A backend failure is a verdict, not a crash."""
    try:
        response = client.generate(build_prompt(requirement, evidence), json_mode=True)
    except OllamaError as exc:
        return Verdict(UNKNOWN, f"model call failed: {exc}")
    return parse_verdict(response)
