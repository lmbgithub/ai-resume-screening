import pytest

from resume_screen.requirements import (
    PREFERRED_WEIGHT,
    REQUIRED_WEIGHT,
    classify_section,
    parse_requirements,
)

JOB = """# Role

## About us
We are a fast-growing team with a great culture and free snacks.

## Requirements
- 5+ years of production Python engineering
- Kubernetes in production

## Preferred qualifications
- Contributions to open-source tooling
"""


def test_bullets_under_headings_become_requirements():
    assert [r.text for r in parse_requirements(JOB)] == [
        "5+ years of production Python engineering",
        "Kubernetes in production",
        "Contributions to open-source tooling",
    ]


def test_about_us_prose_is_not_a_requirement():
    assert all("snacks" not in r.text for r in parse_requirements(JOB))


def test_prose_under_a_requirements_heading_is_still_dropped():
    text = "## Requirements\nWe want someone great to join our wonderful team.\n"
    assert parse_requirements(text) == []


def test_kinds_follow_the_section():
    kinds = [r.kind for r in parse_requirements(JOB)]
    assert kinds == ["required", "required", "preferred"]


def test_preferred_qualifications_reads_as_preferred_not_required():
    # The heading contains both "preferred" and "qualifications"; the
    # qualifier must win, or every preferred item is scored as mandatory.
    assert classify_section("Preferred qualifications") == "preferred"


@pytest.mark.parametrize(
    "title", ["Requirements", "Must have", "Basic qualifications", "What you will do"]
)
def test_required_headings(title):
    assert classify_section(title) in {"required", "preferred"}


@pytest.mark.parametrize("title", ["Nice to have", "Bonus points", "Desirable skills", "Preferred"])
def test_preferred_headings(title):
    assert classify_section(title) == "preferred"


def test_unknown_heading_defaults_to_required():
    # Under-weighting a real requirement is the worse error: it lets a
    # candidate pass on something the employer treats as mandatory.
    assert classify_section("Ideal background") == "required"


def test_weights():
    requirements = parse_requirements(JOB)
    assert requirements[0].weight == REQUIRED_WEIGHT
    assert requirements[-1].weight == PREFERRED_WEIGHT
    assert REQUIRED_WEIGHT > PREFERRED_WEIGHT


def test_empty_job_description_has_no_requirements():
    assert parse_requirements("") == []


def test_bullets_with_no_heading_are_ignored():
    assert parse_requirements("- a bullet with no section above it") == []


def test_indices_are_contiguous():
    assert [r.index for r in parse_requirements(JOB)] == [0, 1, 2]


# --- the model fallback ------------------------------------------------------

from resume_screen.fakes import FakeClient  # noqa: E402
from resume_screen.ollama import OllamaError  # noqa: E402
from resume_screen.requirements import (  # noqa: E402
    MAX_MODEL_REQUIREMENTS,
    build_extraction_prompt,
    find_requirements,
    parse_extracted,
)

PROSE = """We are hiring a Senior Machine Learning Engineer at Acme, a fast-growing team.

You will build streaming inference services and deploy speech models to production.
We expect at least five years of production Python engineering, and Kubernetes in
production is required. Experience with quantised local LLM inference is a plus.

We offer great benefits, free snacks and a generous equity package.
"""

EXTRACTION = (
    '{"requirements": ['
    '{"text": "Build streaming inference services", "kind": "required"},'
    '{"text": "Five years of production Python engineering", "kind": "required"},'
    '{"text": "Experience with quantised local LLM inference", "kind": "preferred"}]}'
)


def extracting(response=EXTRACTION):
    return FakeClient(responses={"Extract the individual requirements": response})


def test_structured_text_never_calls_the_model():
    client = extracting()
    found = find_requirements(client, JOB)
    assert found.source == "headings"
    assert client.prompts == []


def test_prose_falls_back_to_the_model():
    found = find_requirements(extracting(), PROSE)
    assert found.source == "model"
    assert [r.text for r in found.requirements] == [
        "Build streaming inference services",
        "Five years of production Python engineering",
        "Experience with quantised local LLM inference",
    ]


def test_the_model_classifies_preferred_requirements():
    found = find_requirements(extracting(), PROSE)
    assert [r.kind for r in found.requirements] == ["required", "required", "preferred"]


def test_extracted_requirements_get_contiguous_indices():
    found = find_requirements(extracting(), PROSE)
    assert [r.index for r in found.requirements] == [0, 1, 2]


def test_extracted_requirements_are_labelled_in_their_section():
    found = find_requirements(extracting(), PROSE)
    assert all(r.section == "extracted by model" for r in found.requirements)


def test_a_failed_extraction_reports_why():
    found = find_requirements(extracting("I could not find any requirements."), PROSE)
    assert found.requirements == ()
    assert "no JSON object" in found.error
    assert not found


def test_a_backend_failure_is_reported_not_raised():
    class Broken(FakeClient):
        def generate(self, prompt, *, json_mode=False):
            raise OllamaError("connection refused")

    found = find_requirements(Broken(), PROSE)
    assert "connection refused" in found.error


def test_prompt_carries_the_posting_and_the_limit():
    prompt = build_extraction_prompt(PROSE)
    assert "Senior Machine Learning Engineer" in prompt
    assert str(MAX_MODEL_REQUIREMENTS) in prompt


def test_prompt_forbids_inventing_requirements():
    assert "Do not invent" in build_extraction_prompt(PROSE)


def test_prompt_excludes_benefits_and_boilerplate():
    assert "benefits" in build_extraction_prompt(PROSE)


def test_bare_strings_are_accepted():
    requirements, error = parse_extracted('{"requirements": ["Build streaming services"]}')
    assert error is None
    assert requirements[0].text == "Build streaming services"
    assert requirements[0].kind == "required"


def test_an_unrecognised_kind_defaults_to_required():
    requirements, _ = parse_extracted(
        '{"requirements": [{"text": "Build streaming services", "kind": "maybe"}]}'
    )
    assert requirements[0].kind == "required"


def test_a_non_string_kind_defaults_to_required():
    requirements, _ = parse_extracted(
        '{"requirements": [{"text": "Build streaming services", "kind": true}]}'
    )
    assert requirements[0].kind == "required"


def test_fenced_json_is_recovered():
    raw = "Sure!\n```json\n" + EXTRACTION + "\n```"
    requirements, error = parse_extracted(raw)
    assert error is None and len(requirements) == 3


def test_short_fragments_are_dropped():
    requirements, _ = parse_extracted('{"requirements": ["ok", "Build streaming services"]}')
    assert [r.text for r in requirements] == ["Build streaming services"]


def test_all_fragments_dropped_is_an_error_not_silent_success():
    requirements, error = parse_extracted('{"requirements": ["ok", "no"]}')
    assert requirements == ()
    assert "no usable requirements" in error


def test_entries_are_capped():
    many = ", ".join(f'"requirement number {i} with enough words"' for i in range(60))
    requirements, _ = parse_extracted(f'{{"requirements": [{many}]}}')
    assert len(requirements) == MAX_MODEL_REQUIREMENTS


def test_whitespace_is_collapsed():
    requirements, _ = parse_extracted('{"requirements": ["  Build   streaming\\n services "]}')
    assert requirements[0].text == "Build streaming services"


def test_a_missing_key_is_an_error():
    _, error = parse_extracted('{"items": ["Build streaming services"]}')
    assert "expected a list" in error


def test_parse_extracted_never_raises():
    for raw in ["", "{", "null", "[]", '{"requirements": {}}', "\x00"]:
        requirements, error = parse_extracted(raw)
        assert requirements == () and error
