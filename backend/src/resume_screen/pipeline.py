"""The screening run: chunk, embed, retrieve, judge, aggregate."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from .advice import NO_SUGGESTIONS, Point, Suggestions, strong_points, suggest, weak_points
from .documents import Chunk, chunk_document
from .judge import Verdict, judge
from .ollama import Client
from .requirements import Requirement, find_requirements
from .scoring import (
    Aggregate,
    aggregate,
    double_center,
    is_degenerate,
    rank_chunks,
    similarity_matrix,
)

DEFAULT_TOP_K = 3


@dataclass(frozen=True)
class Evidence:
    """One resume line offered as support for one requirement."""

    chunk: Chunk
    centered: float
    raw: float


@dataclass(frozen=True)
class RequirementResult:
    requirement: Requirement
    verdict: Verdict
    evidence: tuple[Evidence, ...]


@dataclass(frozen=True)
class ScreenResult:
    results: tuple[RequirementResult, ...]
    summary: Aggregate
    chunks: tuple[Chunk, ...]
    calibrated: bool
    requirements_source: str = "headings"
    strengths: tuple[Point, ...] = ()
    weaknesses: tuple[Point, ...] = ()
    suggestions: Suggestions = NO_SUGGESTIONS


class EmptyDocumentError(ValueError):
    """A resume or job description produced nothing to work with."""


def _no_requirements_message(found) -> str:
    """Say which path failed, so the advice matches the actual problem."""
    if found.error:
        return (
            f"no requirements could be read from the job description: {found.error}. "
            "Try pasting the requirements as a bulleted list."
        )
    return (
        "the job description contains no requirements — it looks like company "
        "description or benefits rather than a list of what the role needs"
    )


def screen(
    client: Client,
    resume_text: str,
    job_text: str,
    *,
    top_k: int = DEFAULT_TOP_K,
    with_suggestions: bool = True,
    on_progress: Callable[[str], None] | None = None,
) -> ScreenResult:
    """Run one resume against one job description.

    Embeddings for every requirement and every chunk are requested in a single
    batch: the matrix needs all of them, and one round trip beats N.

    `on_progress` is called between steps. A screening run is one model call
    per requirement plus two, which on CPU-only hardware is minutes; without
    progress there is no way to tell a slow run from a hung one, and "it is
    stuck" is what everyone reasonably concludes.
    """
    say = on_progress or (lambda _message: None)
    chunks = chunk_document(resume_text, default_section="resume")
    if not chunks:
        raise EmptyDocumentError("the resume produced no chunks")
    say("reading requirements from the job description")
    found = find_requirements(client, job_text)
    if not found:
        raise EmptyDocumentError(_no_requirements_message(found))
    requirements = list(found.requirements)

    say(f"embedding {len(requirements)} requirements and {len(chunks)} CV lines")
    texts = [r.text for r in requirements] + [c.text for c in chunks]
    vectors = client.embed(texts)
    split = len(requirements)
    raw = similarity_matrix(vectors[:split], vectors[split:])

    degenerate = is_degenerate(raw)
    # With one requirement or one chunk every residual is 0, so ranking on the
    # centered score would be an arbitrary tie. Fall back to raw similarity
    # and say so in the report rather than emit a confident ordering.
    centered = raw if degenerate else double_center(raw)

    results: list[RequirementResult] = []
    # strict=True: the three sequences are built from one another, so a
    # length mismatch is a bug, not a case to silently truncate.
    for requirement, raw_row, centered_row in zip(requirements, raw, centered, strict=True):
        ranked = rank_chunks(raw_row, centered_row, top_k=top_k)
        evidence = tuple(Evidence(chunk=chunks[i], centered=c, raw=r) for i, c, r in ranked)
        say(f"judging {len(results) + 1}/{len(requirements)}: {requirement.text[:60]}")
        verdict = judge(client, requirement.text, [e.chunk.text for e in evidence])
        results.append(RequirementResult(requirement, verdict, evidence))

    summary = aggregate(
        [(r.verdict.verdict, r.requirement.weight, r.requirement.text) for r in results]
    )
    # Strengths and weaknesses are read off the verdicts that produced the
    # score. Only the suggestions cost another model call, and they are the
    # only part of the report the model authors rather than justifies.
    strengths = strong_points(results)
    weaknesses = weak_points(results)
    if with_suggestions:
        say("writing suggestions")
    suggestions = (
        suggest(client, weaknesses, strengths)
        if with_suggestions
        else Suggestions((), "suggestions were not requested")
    )
    say("done")
    return ScreenResult(
        results=tuple(results),
        summary=summary,
        chunks=tuple(chunks),
        calibrated=not degenerate,
        requirements_source=found.source,
        strengths=strengths,
        weaknesses=weaknesses,
        suggestions=suggestions,
    )
