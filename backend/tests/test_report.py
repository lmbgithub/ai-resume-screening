"""Report rendering, built from hand-made results rather than a pipeline run.

Constructing the dataclasses directly keeps these tests about formatting: a
change in retrieval cannot make them fail, and a change in wording cannot hide
behind a passing pipeline test.
"""

import pytest

from resume_screen.documents import Chunk
from resume_screen.judge import Verdict
from resume_screen.pipeline import Evidence, RequirementResult, ScreenResult
from resume_screen.report import render
from resume_screen.requirements import Requirement
from resume_screen.scoring import Aggregate


def make_result(
    verdict="covered",
    *,
    kind="required",
    reason="because",
    evidence=True,
    score=1.0,
    unscored=(),
    calibrated=True,
    text="5+ years of production Python",
):
    item = RequirementResult(
        requirement=Requirement(text=text, kind=kind, section="Requirements", index=0),
        verdict=Verdict(verdict, reason),
        evidence=(
            (
                Evidence(
                    chunk=Chunk(
                        text="Built Python services", section="Experience", index=0, bullet=True
                    ),
                    centered=0.2468,
                    raw=0.8213,
                ),
            )
            if evidence
            else ()
        ),
    )
    return ScreenResult(
        results=(item,),
        summary=Aggregate(score=score, scored_weight=1.0, unscored=tuple(unscored)),
        chunks=(item.evidence[0].chunk,) if evidence else (),
        calibrated=calibrated,
    )


def test_score_appears_in_the_first_line():
    assert render(make_result(score=0.769)).splitlines()[0].startswith("Match: 76.9%")


def test_requirement_count_is_reported():
    assert "over 1 requirements" in render(make_result())


@pytest.mark.parametrize(
    "verdict,mark",
    [("covered", "yes"), ("partial", "partial"), ("absent", "no"), ("unknown", "UNSCORED")],
)
def test_each_verdict_gets_its_own_mark(verdict, mark):
    assert f"[{mark:>8}]" in render(make_result(verdict))


def test_unknown_is_shouted_so_it_cannot_be_skimmed_past():
    # It is the one verdict that means "we do not know", and it must not look
    # like a quiet third option next to yes and no.
    assert "UNSCORED" in render(make_result("unknown"))


def test_null_score_renders_as_not_available():
    text = render(make_result("unknown", score=None, unscored=("a requirement",)))
    assert "Match: n/a" in text
    assert "0.0%" not in text


def test_unscored_count_is_noted():
    text = render(make_result(score=0.5, unscored=("a", "b")))
    assert "2 requirement(s) unscored" in text
    assert "excluded from the denominator" in text


def test_no_unscored_note_when_everything_scored():
    assert "unscored" not in render(make_result())


def test_uncalibrated_run_is_flagged():
    assert "uncalibrated" in render(make_result(calibrated=False))


def test_calibrated_run_carries_no_warning():
    assert "uncalibrated" not in render(make_result(calibrated=True))


def test_kind_is_shown_next_to_each_requirement():
    assert "(preferred)" in render(make_result(kind="preferred"))


def test_reason_is_shown_when_present():
    assert "reason: because" in render(make_result(reason="because"))


def test_empty_reason_prints_no_reason_line():
    assert "reason:" not in render(make_result(reason=""))


def test_evidence_shows_both_scores_with_a_signed_specificity():
    text = render(make_result())
    assert "specificity +0.247" in text
    assert "cosine 0.821" in text


def test_negative_specificity_keeps_its_sign():
    result = make_result()
    patched = result.results[0].evidence[0]
    item = RequirementResult(
        requirement=result.results[0].requirement,
        verdict=result.results[0].verdict,
        evidence=(Evidence(chunk=patched.chunk, centered=-0.05, raw=0.4),),
    )
    text = render(ScreenResult((item,), result.summary, result.chunks, True))
    assert "specificity -0.050" in text


def test_long_evidence_is_truncated():
    long_chunk = Chunk(text="x" * 400, section="Experience", index=0, bullet=True)
    item = RequirementResult(
        requirement=Requirement("a requirement here", "required", "Requirements", 0),
        verdict=Verdict("covered", ""),
        evidence=(Evidence(chunk=long_chunk, centered=0.1, raw=0.2),),
    )
    text = render(ScreenResult((item,), Aggregate(1.0, 1.0, ()), (long_chunk,), True))
    assert "x" * 100 in text
    assert "x" * 101 not in text


def test_evidence_can_be_suppressed():
    assert "specificity" not in render(make_result(), show_evidence=False)


def test_requirement_with_no_evidence_still_renders():
    assert "5+ years" in render(make_result(evidence=False))


def test_report_ends_with_exactly_one_newline():
    text = render(make_result())
    assert text.endswith("\n") and not text.endswith("\n\n")


# --- strengths, weaknesses and suggestions -----------------------------------

from resume_screen.advice import Suggestions, strong_points, weak_points  # noqa: E402

SAMPLE_SUGGESTIONS = Suggestions(("state the latency you met",))


def full_result(verdict="covered", suggestions=SAMPLE_SUGGESTIONS):
    base = make_result(verdict, score=1.0 if verdict == "covered" else 0.0)
    return ScreenResult(
        results=base.results,
        summary=base.summary,
        chunks=base.chunks,
        calibrated=True,
        strengths=strong_points(base.results),
        weaknesses=weak_points(base.results),
        suggestions=suggestions,
    )


def test_strong_points_are_listed():
    text = render(full_result("covered"))
    assert "Strong points:" in text
    assert "+ (required) 5+ years of production Python" in text


def test_no_strong_points_says_none():
    assert "Strong points: none" in render(full_result("absent"))


def test_weak_points_are_listed_with_their_verdict():
    assert "- (required/absent) 5+ years" in render(full_result("absent"))


def test_no_weak_points_says_none():
    assert "Weak points: none" in render(full_result("covered"))


def test_suggestions_are_numbered_and_labelled_as_model_written():
    text = render(full_result())
    assert "Suggestions (written by the model, not scored):" in text
    assert "1. state the latency you met" in text


def test_absent_suggestions_print_the_reason_not_an_empty_heading():
    # An empty list under a "Suggestions" heading reads as "nothing to
    # improve", which is not what a failed model call means.
    text = render(full_result(suggestions=Suggestions((), "model call failed: refused")))
    assert "Suggestions: none produced" in text
    assert "model call failed: refused" in text
