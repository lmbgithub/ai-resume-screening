import pytest

from resume_screen.fakes import FakeClient
from resume_screen.judge import UNKNOWN, build_prompt, judge, parse_verdict
from resume_screen.ollama import OllamaError


@pytest.mark.parametrize("verdict", ["covered", "partial", "absent"])
def test_clean_json_is_parsed(verdict):
    parsed = parse_verdict(f'{{"verdict": "{verdict}", "reason": "because"}}')
    assert (parsed.verdict, parsed.reason) == (verdict, "because")


def test_verdict_case_and_whitespace_are_normalised():
    assert parse_verdict('{"verdict": "  COVERED "}').verdict == "covered"


def test_json_inside_a_code_fence_is_recovered():
    raw = 'Sure!\n```json\n{"verdict": "covered", "reason": "ok"}\n```\nHope that helps.'
    assert parse_verdict(raw).verdict == "covered"


def test_json_with_leading_prose_is_recovered():
    assert parse_verdict('Here is my answer: {"verdict": "absent"}').verdict == "absent"


def test_nested_braces_in_the_reason_do_not_truncate_the_object():
    raw = '{"verdict": "covered", "reason": "matched {python} and {sql}"}'
    parsed = parse_verdict(raw)
    assert parsed.verdict == "covered"
    assert "{sql}" in parsed.reason


def test_braces_inside_a_string_do_not_confuse_the_scanner():
    assert parse_verdict('{"verdict": "absent", "reason": "a } brace"}').verdict == "absent"


def test_escaped_quote_in_the_reason():
    parsed = parse_verdict('{"verdict": "covered", "reason": "said \\"yes\\""}')
    assert parsed.verdict == "covered"


def test_prose_with_no_json_is_unknown():
    parsed = parse_verdict("I think the candidate is a great fit overall.")
    assert parsed.verdict == UNKNOWN
    assert "no JSON" in parsed.reason


def test_malformed_json_is_unknown():
    assert parse_verdict('{"verdict": "covered",}').verdict == UNKNOWN


def test_empty_response_is_unknown():
    assert parse_verdict("").verdict == UNKNOWN


def test_object_wrapped_in_an_array_is_still_recovered():
    # The scanner looks for the first balanced object, not for a document
    # root, so a model that answers with a one-element list still parses.
    assert parse_verdict('[{"verdict": "covered"}]').verdict == "covered"


def test_bare_json_array_is_unknown():
    assert parse_verdict('["covered", "partial"]').verdict == UNKNOWN


def test_missing_verdict_key_is_unknown():
    assert parse_verdict('{"reason": "looks good"}').verdict == UNKNOWN


def test_boolean_verdict_is_unknown_not_truthy():
    # `isinstance(True, int)` is True in Python; a loose check would let a
    # bare `true` through and silently score the requirement.
    parsed = parse_verdict('{"verdict": true}')
    assert parsed.verdict == UNKNOWN
    assert "bool" in parsed.reason


def test_numeric_verdict_is_unknown():
    assert parse_verdict('{"verdict": 1}').verdict == UNKNOWN


def test_invented_verdict_word_is_unknown():
    parsed = parse_verdict('{"verdict": "strong match"}')
    assert parsed.verdict == UNKNOWN
    assert "strong match" in parsed.reason


def test_missing_reason_gives_an_empty_string_not_a_crash():
    assert parse_verdict('{"verdict": "covered"}').reason == ""


def test_non_string_reason_is_dropped():
    assert parse_verdict('{"verdict": "covered", "reason": 7}').reason == ""


def test_unknown_verdict_is_not_known():
    assert not parse_verdict("nonsense").is_known
    assert parse_verdict('{"verdict": "absent"}').is_known


def test_parse_verdict_never_raises():
    for raw in ["", "{", "}", "{}", "null", "{'verdict': 'covered'}", "\x00"]:
        assert parse_verdict(raw).verdict == UNKNOWN


def test_prompt_contains_requirement_and_evidence():
    prompt = build_prompt("Kubernetes in production", ["ran k8s clusters"])
    assert "Kubernetes in production" in prompt
    assert "- ran k8s clusters" in prompt


def test_prompt_forbids_inferring_from_employer_or_school():
    prompt = build_prompt("anything", [])
    assert "employers, or schools" in prompt


def test_prompt_with_no_evidence_says_so_explicitly():
    assert "(none found)" in build_prompt("anything", [])


def test_judge_uses_json_mode():
    seen: list[bool] = []

    class Recorder(FakeClient):
        def generate(self, prompt, *, json_mode=False):
            seen.append(json_mode)
            return '{"verdict": "covered"}'

    judge(Recorder(), "req", ["evidence"])
    assert seen == [True]


def test_backend_failure_becomes_unknown_not_an_exception():
    class Broken(FakeClient):
        def generate(self, prompt, *, json_mode=False):
            raise OllamaError("connection refused")

    parsed = judge(Broken(), "req", [])
    assert parsed.verdict == UNKNOWN
    assert "connection refused" in parsed.reason
