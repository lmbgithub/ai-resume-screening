"""The JSON contract. `web/lib/api.ts` mirrors these keys by hand, so a change
here without a change there is exactly the bug these tests exist to catch."""

import json

import pytest

from resume_screen.advice import Suggestions, strong_points, weak_points
from resume_screen.documents import Chunk
from resume_screen.judge import Verdict
from resume_screen.pipeline import Evidence, RequirementResult, ScreenResult
from resume_screen.requirements import Requirement
from resume_screen.scoring import Aggregate
from resume_screen.serialize import evidence_to_dict, requirement_to_dict, result_to_dict

CHUNK = Chunk(text="Built Python services", section="Experience", index=0, bullet=True)
EVIDENCE = Evidence(chunk=CHUNK, centered=0.246812, raw=0.821345)
ITEM = RequirementResult(
    requirement=Requirement("5+ years of Python", "required", "Requirements", 0),
    verdict=Verdict("covered", "seven years shown"),
    evidence=(EVIDENCE,),
)


def make(score=1.0, unscored=(), calibrated=True, results=(ITEM,)):
    return ScreenResult(
        results=results,
        summary=Aggregate(score=score, scored_weight=1.0, unscored=tuple(unscored)),
        chunks=(CHUNK,),
        calibrated=calibrated,
        strengths=strong_points(results),
        weaknesses=weak_points(results),
        suggestions=Suggestions(("state the latency you met",)),
    )


def test_evidence_keys():
    assert set(evidence_to_dict(EVIDENCE)) == {"text", "section", "specificity", "cosine"}


def test_evidence_scores_are_rounded_to_four_places():
    body = evidence_to_dict(EVIDENCE)
    assert body["specificity"] == 0.2468
    assert body["cosine"] == 0.8213


def test_requirement_keys():
    assert set(requirement_to_dict(ITEM)) == {
        "text",
        "kind",
        "section",
        "weight",
        "verdict",
        "reason",
        "evidence",
    }


def test_preferred_weight_travels_with_the_requirement():
    item = RequirementResult(
        requirement=Requirement("nice to have", "preferred", "Preferred", 0),
        verdict=Verdict("absent", ""),
        evidence=(),
    )
    assert requirement_to_dict(item)["weight"] == 0.5


def test_result_keys():
    assert set(result_to_dict(make())) == {
        "score",
        "percent",
        "calibrated",
        "scored_weight",
        "unscored",
        "chunk_count",
        "requirements",
        "requirements_source",
        "strengths",
        "weaknesses",
        "suggestions",
    }


def test_suggestions_carry_items_and_error():
    body = result_to_dict(make())["suggestions"]
    assert set(body) == {"items", "error"}
    assert body["items"] == ["state the latency you met"]


def test_null_score_is_null_not_zero():
    # The single most important line in this module: a client that reads 0.0
    # here would display a rejection built entirely out of a parser failure.
    body = result_to_dict(make(score=None, unscored=("a",)))
    assert body["score"] is None
    assert body["percent"] == "n/a"


def test_zero_score_is_distinguishable_from_no_score():
    assert result_to_dict(make(score=0.0))["score"] == 0.0
    assert result_to_dict(make(score=0.0))["percent"] == "0.0%"


def test_calibration_flag_travels_beside_the_score():
    assert result_to_dict(make(calibrated=False))["calibrated"] is False


def test_unscored_is_a_list_not_a_tuple():
    body = result_to_dict(make(unscored=("a", "b")))
    assert body["unscored"] == ["a", "b"]
    assert isinstance(body["unscored"], list)


def test_requirements_source_is_reported():
    # A client needs to know whether the requirement list is reproducible.
    assert result_to_dict(make())["requirements_source"] == "headings"


def test_chunk_count_is_reported():
    assert result_to_dict(make())["chunk_count"] == 1


def test_weaknesses_are_serialised_as_points():
    absent = RequirementResult(
        requirement=Requirement("Kubernetes in production", "required", "Requirements", 0),
        verdict=Verdict("absent", "not shown"),
        evidence=(),
    )
    body = result_to_dict(make(score=0.0, results=(absent,)))
    assert set(body["weaknesses"][0]) == {"requirement", "kind", "verdict", "detail", "evidence"}


def test_covered_requirements_are_not_weaknesses():
    assert result_to_dict(make())["weaknesses"] == []
    assert len(result_to_dict(make())["strengths"]) == 1


def test_whole_body_survives_a_json_round_trip():
    body = result_to_dict(make())
    assert json.loads(json.dumps(body)) == body


@pytest.mark.parametrize("verdict", ["covered", "partial", "absent", "unknown"])
def test_every_verdict_serialises_as_its_own_string(verdict):
    item = RequirementResult(
        requirement=Requirement("a requirement", "required", "Requirements", 0),
        verdict=Verdict(verdict, ""),
        evidence=(),
    )
    assert requirement_to_dict(item)["verdict"] == verdict
