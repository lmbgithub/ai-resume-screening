import pytest

from resume_screen.fakes import FakeClient, hashed_embedding
from resume_screen.pipeline import EmptyDocumentError, screen
from resume_screen.report import render

RESUME = """## Experience
- Built a streaming ASR gateway handling 400 concurrent WebSocket sessions
- Wrote an evaluation harness with paired bootstrap confidence intervals
- Maintained ETL pipelines in Python and SQL over a warehouse
"""

JOB = """## Requirements
- Build streaming inference services with low latency
- Kubernetes and container orchestration in production

## Preferred qualifications
- Contributions to open-source tooling
"""


def scripted():
    return FakeClient(
        responses={
            "streaming inference": '{"verdict": "covered", "reason": "gateway"}',
            "Kubernetes": '{"verdict": "absent", "reason": "not shown"}',
            "open-source": '{"verdict": "partial", "reason": "unclear"}',
        }
    )


def test_screen_scores_every_requirement():
    result = screen(scripted(), RESUME, JOB)
    assert [r.verdict.verdict for r in result.results] == ["covered", "absent", "partial"]


def test_screen_score_is_the_weighted_mean():
    # covered*1.0 + absent*1.0 + partial*0.5 over weight 2.5
    assert screen(scripted(), RESUME, JOB).summary.score == pytest.approx(1.25 / 2.5)


def test_embeddings_are_requested_in_one_batch():
    client = scripted()
    result = screen(client, RESUME, JOB)
    assert len(client.embed_calls) == 1
    assert len(client.embed_calls[0]) == len(result.results) + len(result.chunks)


def test_one_generation_call_per_requirement_plus_one_for_suggestions():
    client = scripted()
    screen(client, RESUME, JOB)
    assert len(client.prompts) == 4
    assert sum("actionable suggestions" in p for p in client.prompts) == 1


def test_suggestions_can_be_skipped():
    client = scripted()
    result = screen(client, RESUME, JOB, with_suggestions=False)
    assert len(client.prompts) == 3
    assert result.suggestions.items == ()
    assert "not requested" in result.suggestions.error


def test_skipping_suggestions_does_not_change_the_score():
    with_advice = screen(scripted(), RESUME, JOB).summary.score
    without = screen(scripted(), RESUME, JOB, with_suggestions=False).summary.score
    assert with_advice == without


def test_evidence_is_capped_at_top_k():
    result = screen(scripted(), RESUME, JOB, top_k=2)
    assert all(len(r.evidence) == 2 for r in result.results)


def test_top_k_larger_than_the_resume_is_clamped():
    result = screen(scripted(), RESUME, JOB, top_k=50)
    assert all(len(r.evidence) == len(result.chunks) for r in result.results)


def test_only_selected_evidence_reaches_the_model():
    client = scripted()
    result = screen(client, RESUME, JOB, top_k=1)
    for item, prompt in zip(result.results, client.prompts, strict=False):
        assert item.evidence[0].chunk.text in prompt
        others = {c.text for c in result.chunks} - {item.evidence[0].chunk.text}
        assert not any(text in prompt for text in others)


def test_run_is_reproducible():
    first = screen(scripted(), RESUME, JOB)
    second = screen(scripted(), RESUME, JOB)
    assert first.summary.score == second.summary.score
    assert [e.centered for r in first.results for e in r.evidence] == [
        e.centered for r in second.results for e in r.evidence
    ]


def test_multi_requirement_run_is_calibrated():
    assert screen(scripted(), RESUME, JOB).calibrated


def test_single_requirement_run_is_flagged_uncalibrated():
    job = "## Requirements\n- Build streaming inference services with low latency\n"
    assert not screen(scripted(), RESUME, job).calibrated


def test_single_chunk_resume_is_flagged_uncalibrated():
    resume = "## Experience\n- Built a streaming ASR gateway for live sessions\n"
    assert not screen(scripted(), resume, JOB).calibrated


def test_uncalibrated_run_still_ranks_by_raw_similarity():
    job = "## Requirements\n- Build streaming inference services with low latency\n"
    result = screen(scripted(), RESUME, job)
    evidence = result.results[0].evidence
    assert [e.raw for e in evidence] == sorted((e.raw for e in evidence), reverse=True)


def test_empty_resume_is_rejected():
    with pytest.raises(EmptyDocumentError, match="resume"):
        screen(scripted(), "", JOB)


def test_job_description_with_no_requirements_is_rejected():
    with pytest.raises(EmptyDocumentError, match="job description"):
        screen(scripted(), RESUME, "We are hiring! Send us a note.")


def test_unparseable_verdicts_do_not_become_rejections():
    client = FakeClient(default="I would say the candidate is promising.")
    result = screen(client, RESUME, JOB)
    assert result.summary.score is None
    assert len(result.summary.unscored) == 3


def test_weaknesses_put_required_absent_first():
    weaknesses = screen(scripted(), RESUME, JOB).weaknesses
    assert weaknesses[0].kind == "required"
    assert weaknesses[0].verdict == "absent"


def test_weaknesses_exclude_covered_requirements():
    weaknesses = screen(scripted(), RESUME, JOB).weaknesses
    assert all("streaming inference" not in w.requirement for w in weaknesses)


def test_hashed_embedding_is_deterministic():
    assert hashed_embedding("python and sql") == hashed_embedding("python and sql")


def test_hashed_embedding_is_normalised():
    vector = hashed_embedding("python and sql")
    assert sum(x * x for x in vector) == pytest.approx(1.0)


def test_hashed_embedding_of_empty_text_is_the_zero_vector():
    assert set(hashed_embedding("")) == {0.0}


def test_shared_vocabulary_scores_higher_than_none():
    from resume_screen.scoring import cosine

    target = hashed_embedding("kubernetes container orchestration")
    near = hashed_embedding("kubernetes orchestration in production")
    far = hashed_embedding("bootstrap confidence intervals for word error rate")
    assert cosine(target, near) > cosine(target, far)


def test_report_mentions_the_score_and_the_weak_points():
    text = render(screen(scripted(), RESUME, JOB))
    assert "Match: 50.0%" in text
    assert "Kubernetes" in text.split("Weak points")[1]


def test_report_flags_unscored_requirements():
    client = FakeClient(default="no json at all here")
    text = render(screen(client, RESUME, JOB))
    assert "unscored" in text
    assert "n/a" in text


def test_report_flags_an_uncalibrated_run():
    job = "## Requirements\n- Build streaming inference services with low latency\n"
    assert "uncalibrated" in render(screen(scripted(), RESUME, job))


def test_report_says_none_when_there_are_no_weak_points():
    client = FakeClient(default='{"verdict": "covered", "reason": "ok"}')
    assert "Weak points: none" in render(screen(client, RESUME, JOB))


def test_progress_is_reported_for_every_step():
    # Without progress there is no way to tell a slow run from a hung one,
    # which is exactly the conclusion a silent five-minute request invites.
    messages = []
    screen(scripted(), RESUME, JOB, on_progress=messages.append)
    assert messages[0].startswith("reading requirements")
    assert any(m.startswith("embedding") for m in messages)
    assert sum(m.startswith("judging") for m in messages) == 3
    assert "writing suggestions" in messages
    assert messages[-1] == "done"


def test_progress_counts_requirements_as_it_goes():
    messages = []
    screen(scripted(), RESUME, JOB, on_progress=messages.append)
    judged = [m for m in messages if m.startswith("judging")]
    assert judged[0].startswith("judging 1/3")
    assert judged[-1].startswith("judging 3/3")


def test_no_suggestions_step_when_they_are_skipped():
    messages = []
    screen(scripted(), RESUME, JOB, with_suggestions=False, on_progress=messages.append)
    assert "writing suggestions" not in messages


def test_screen_works_without_a_progress_callback():
    assert screen(scripted(), RESUME, JOB).summary.score is not None
