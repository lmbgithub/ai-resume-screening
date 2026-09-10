"""Turning a ScreenResult into JSON-safe primitives for the HTTP API.

Kept out of `pipeline` so the dataclasses stay free of transport concerns, and
out of `api` so the shape can be tested without starting a server.
"""

from __future__ import annotations

from .pipeline import Evidence, RequirementResult, ScreenResult


def evidence_to_dict(evidence: Evidence) -> dict:
    return {
        "text": evidence.chunk.text,
        "section": evidence.chunk.section,
        "specificity": round(evidence.centered, 4),
        "cosine": round(evidence.raw, 4),
    }


def requirement_to_dict(item: RequirementResult) -> dict:
    return {
        "text": item.requirement.text,
        "kind": item.requirement.kind,
        "section": item.requirement.section,
        "weight": item.requirement.weight,
        "verdict": item.verdict.verdict,
        "reason": item.verdict.reason,
        "evidence": [evidence_to_dict(e) for e in item.evidence],
    }


def result_to_dict(result: ScreenResult) -> dict:
    """The API's response body.

    `score` is null rather than 0 when nothing could be scored, and
    `calibrated` travels with the score so a client cannot render an
    uncalibrated number as if it were comparable.

    `weaknesses` replaced an earlier `gaps` key that held the same content in
    a different shape. Two names for one concept is how a client ends up
    rendering both and double-counting a candidate's problems.
    """
    return {
        "score": result.summary.score,
        "percent": result.summary.percent,
        "calibrated": result.calibrated,
        "scored_weight": result.summary.scored_weight,
        "unscored": list(result.summary.unscored),
        "chunk_count": len(result.chunks),
        # "headings" means the requirement list is reproducible; "model" means
        # it was read out of prose and may differ between runs.
        "requirements_source": result.requirements_source,
        "requirements": [requirement_to_dict(r) for r in result.results],
        "strengths": [p.to_dict() for p in result.strengths],
        "weaknesses": [p.to_dict() for p in result.weaknesses],
        # `error` travels with the suggestions so the UI can distinguish "no
        # advice was produced" from "your CV needs no improvement".
        "suggestions": result.suggestions.to_dict(),
    }
