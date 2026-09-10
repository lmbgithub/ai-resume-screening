from resume_screen.advice import (
    MAX_SUGGESTIONS,
    build_prompt,
    parse_suggestions,
    strong_points,
    suggest,
    weak_points,
)
from resume_screen.documents import Chunk
from resume_screen.fakes import FakeClient
from resume_screen.judge import Verdict
from resume_screen.ollama import OllamaError
from resume_screen.pipeline import Evidence, RequirementResult
from resume_screen.requirements import Requirement


def item(text, verdict, *, kind="required", index=0, reason="because", centered=0.2, evidence=True):
    chunk = Chunk(text=f"evidence for {text}", section="Experience", index=index, bullet=True)
    return RequirementResult(
        requirement=Requirement(text, kind, "Requirements", index),
        verdict=Verdict(verdict, reason),
        evidence=(Evidence(chunk=chunk, centered=centered, raw=0.5),) if evidence else (),
    )


# --- derived points ----------------------------------------------------------


def test_strong_points_are_the_covered_requirements():
    results = [item("python", "covered"), item("k8s", "absent", index=1)]
    assert [p.requirement for p in strong_points(results)] == ["python"]


def test_partial_is_not_a_strong_point():
    assert strong_points([item("python", "partial")]) == ()


def test_unknown_is_not_a_strong_point():
    # The model failed; that is not a strength.
    assert strong_points([item("python", "unknown")]) == ()


def test_strong_points_put_required_before_preferred():
    results = [
        item("open source", "covered", kind="preferred", index=0, centered=0.9),
        item("python", "covered", kind="required", index=1, centered=0.1),
    ]
    assert [p.kind for p in strong_points(results)] == ["required", "preferred"]


def test_strong_points_order_by_evidence_specificity_within_a_kind():
    results = [
        item("weakly evidenced", "covered", index=0, centered=0.1),
        item("strongly evidenced", "covered", index=1, centered=0.8),
    ]
    assert strong_points(results)[0].requirement == "strongly evidenced"


def test_weak_points_are_absent_and_partial():
    results = [
        item("python", "covered"),
        item("k8s", "absent", index=1),
        item("llm", "partial", index=2),
    ]
    assert {p.requirement for p in weak_points(results)} == {"k8s", "llm"}


def test_weak_points_exclude_unknown():
    # Presenting our parser failure as the candidate's weakness would be a lie.
    assert weak_points([item("python", "unknown")]) == ()


def test_weak_points_put_absent_before_partial():
    results = [item("partial one", "partial", index=0), item("absent one", "absent", index=1)]
    assert [p.verdict for p in weak_points(results)] == ["absent", "partial"]


def test_weak_points_put_required_before_preferred():
    results = [
        item("preferred absent", "absent", kind="preferred", index=0),
        item("required partial", "partial", kind="required", index=1),
    ]
    assert [p.kind for p in weak_points(results)] == ["required", "preferred"]


def test_points_carry_the_reason_and_the_evidence():
    point = strong_points([item("python", "covered", reason="seven years")])[0]
    assert point.detail == "seven years"
    assert point.evidence == "evidence for python"


def test_a_point_without_evidence_has_none_not_an_empty_string():
    point = weak_points([item("k8s", "absent", evidence=False)])[0]
    assert point.evidence is None


def test_point_to_dict_shape():
    body = strong_points([item("python", "covered")])[0].to_dict()
    assert set(body) == {"requirement", "kind", "verdict", "detail", "evidence"}


def test_points_are_derived_without_calling_the_model():
    client = FakeClient()
    strong_points([item("python", "covered")])
    weak_points([item("k8s", "absent")])
    assert client.prompts == []


# --- prompt ------------------------------------------------------------------


def test_prompt_lists_gaps_and_strengths():
    gaps = weak_points([item("Kubernetes", "absent")])
    strengths = strong_points([item("Python", "covered")])
    prompt = build_prompt(gaps, strengths)
    assert "Kubernetes" in prompt and "Python" in prompt


def test_prompt_forbids_inventing_experience():
    assert "Never suggest claiming experience" in build_prompt((), ())


def test_prompt_states_the_limit():
    assert str(MAX_SUGGESTIONS) in build_prompt((), ())


def test_prompt_renders_none_for_an_empty_section():
    assert "(none)" in build_prompt((), strong_points([item("Python", "covered")]))


# --- parsing -----------------------------------------------------------------


def test_clean_list_is_parsed():
    parsed = parse_suggestions('{"suggestions": ["state the latency you met", "name the volume"]}')
    assert parsed.items == ("state the latency you met", "name the volume")
    assert parsed.error is None


def test_fenced_json_is_recovered():
    raw = 'Sure!\n```json\n{"suggestions": ["add metrics to the gateway bullet"]}\n```'
    assert parse_suggestions(raw).items == ("add metrics to the gateway bullet",)


def test_objects_in_the_list_are_unwrapped():
    raw = (
        '{"suggestions": [{"suggestion": "quantify the cost saving"}, {"text": "name the stack"}]}'
    )
    assert parse_suggestions(raw).items == ("quantify the cost saving", "name the stack")


def test_whitespace_is_collapsed():
    assert parse_suggestions('{"suggestions": ["  add   metrics \\n here "]}').items == (
        "add metrics here",
    )


def test_suggestions_are_capped():
    many = ", ".join(f'"suggestion number {i}"' for i in range(20))
    parsed = parse_suggestions(f'{{"suggestions": [{many}]}}')
    assert len(parsed.items) == MAX_SUGGESTIONS


def test_overlong_suggestions_are_truncated():
    parsed = parse_suggestions('{"suggestions": ["' + "x" * 900 + '"]}')
    assert len(parsed.items[0]) == 400


def test_prose_with_no_json_yields_no_suggestions_and_an_error():
    parsed = parse_suggestions("I think you should add more detail to your CV.")
    assert parsed.items == ()
    assert "no JSON object" in parsed.error


def test_malformed_json_is_an_error():
    assert parse_suggestions('{"suggestions": [,]}').error is not None


def test_a_missing_key_is_an_error():
    parsed = parse_suggestions('{"advice": ["do a thing"]}')
    assert parsed.items == ()
    assert "expected a list" in parsed.error


def test_a_string_instead_of_a_list_is_an_error():
    assert "expected a list" in parse_suggestions('{"suggestions": "do a thing"}').error


def test_an_empty_list_is_an_error_not_silent_success():
    # An empty list rendered with no message reads as "your CV is perfect",
    # which is a very different claim from "the model said nothing usable".
    parsed = parse_suggestions('{"suggestions": []}')
    assert parsed.items == ()
    assert "no usable suggestions" in parsed.error


def test_non_string_entries_are_dropped_not_rendered():
    parsed = parse_suggestions('{"suggestions": ["a real suggestion", 7, null, {}]}')
    assert parsed.items == ("a real suggestion",)


def test_parse_never_raises():
    for raw in ["", "{", "}", "null", "[]", "\x00", '{"suggestions": {}}']:
        assert parse_suggestions(raw).items == ()


def test_to_dict_shape():
    body = parse_suggestions('{"suggestions": ["do a thing"]}').to_dict()
    assert set(body) == {"items", "error"}
    assert isinstance(body["items"], list)


# --- the call ----------------------------------------------------------------


def test_suggest_uses_json_mode():
    seen = []

    class Recorder(FakeClient):
        def generate(self, prompt, *, json_mode=False):
            seen.append(json_mode)
            return '{"suggestions": ["a suggestion"]}'

    suggest(Recorder(), weak_points([item("k8s", "absent")]), ())
    assert seen == [True]


def test_backend_failure_is_reported_not_raised():
    class Broken(FakeClient):
        def generate(self, prompt, *, json_mode=False):
            raise OllamaError("connection refused")

    parsed = suggest(Broken(), weak_points([item("k8s", "absent")]), ())
    assert parsed.items == ()
    assert "connection refused" in parsed.error


def test_no_requirements_means_no_call_and_a_clear_reason():
    client = FakeClient()
    parsed = suggest(client, (), ())
    assert client.prompts == []
    assert "no requirements to advise on" in parsed.error


def test_suggest_returns_the_parsed_items():
    client = FakeClient(default='{"suggestions": ["state the latency you met"]}')
    parsed = suggest(client, weak_points([item("k8s", "absent")]), ())
    assert parsed.items == ("state the latency you met",)


def test_duplicate_requirement_text_does_not_collapse_the_ordering():
    # A job description may repeat a bullet verbatim. Keying the sort order by
    # requirement text made two points share one key.
    results = [
        item("shipped systems", "covered", kind="preferred", index=0, centered=0.1),
        item("shipped systems", "covered", kind="required", index=1, centered=0.9),
    ]
    points = strong_points(results)
    assert len(points) == 2
    assert [p.kind for p in points] == ["required", "preferred"]


def test_duplicate_requirement_text_in_weak_points():
    results = [
        item("shipped systems", "partial", index=0),
        item("shipped systems", "absent", index=1),
    ]
    points = weak_points(results)
    assert len(points) == 2
    assert [p.verdict for p in points] == ["absent", "partial"]


def test_strong_points_are_stable_across_runs():
    results = [item(f"requirement {i}", "covered", index=i, centered=0.5) for i in range(6)]
    assert strong_points(results) == strong_points(results)
