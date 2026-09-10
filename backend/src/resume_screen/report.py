"""Rendering a screening result as text a human will actually read."""

from __future__ import annotations

from .pipeline import ScreenResult

_MARK = {"covered": "yes", "partial": "partial", "absent": "no", "unknown": "UNSCORED"}


def render(result: ScreenResult, *, show_evidence: bool = True) -> str:
    lines: list[str] = []
    summary = result.summary
    lines.append(
        f"Match: {summary.percent}  (weighted coverage over {len(result.results)} requirements)"
    )
    if not result.calibrated:
        lines.append(
            "  note: uncalibrated — one requirement or one resume line, "
            "ranking fell back to raw cosine"
        )
    if summary.unscored:
        lines.append(
            f"  note: {len(summary.unscored)} requirement(s) unscored "
            f"and excluded from the denominator"
        )
    lines.append("")

    for item in result.results:
        req = item.requirement
        mark = _MARK.get(item.verdict.verdict, item.verdict.verdict)
        lines.append(f"[{mark:>8}] ({req.kind}) {req.text}")
        if item.verdict.reason:
            lines.append(f"           reason: {item.verdict.reason}")
        if show_evidence and item.evidence:
            top = item.evidence[0]
            lines.append(
                f"           evidence: {top.chunk.text[:100]}"
                f"  [specificity {top.centered:+.3f}, cosine {top.raw:.3f}]"
            )
        lines.append("")

    lines.append("Strong points:" if result.strengths else "Strong points: none")
    for point in result.strengths:
        lines.append(f"  + ({point.kind}) {point.requirement}")
        if point.evidence:
            lines.append(f"      {point.evidence[:96]}")

    lines.append("")
    lines.append("Weak points (required first):" if result.weaknesses else "Weak points: none")
    for point in result.weaknesses:
        lines.append(f"  - ({point.kind}/{point.verdict}) {point.requirement}")
        if point.detail:
            lines.append(f"      {point.detail[:96]}")

    lines.append("")
    if result.suggestions.items:
        lines.append("Suggestions (written by the model, not scored):")
        for index, suggestion in enumerate(result.suggestions.items, start=1):
            lines.append(f"  {index}. {suggestion}")
    else:
        # Never print an empty list under a heading: it reads as "no
        # improvements needed", which is a different claim entirely.
        lines.append(f"Suggestions: none produced ({result.suggestions.error})")
    return "\n".join(lines).rstrip() + "\n"
