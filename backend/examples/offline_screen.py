"""Run the full pipeline with no model, no network and no API key.

The fake backend returns hashed bag-of-words embeddings and scripted verdicts,
so this prints the same report on every machine. It exists to show the shape of
a run and to give the retrieval machinery a case with known-exact similarities.
"""

from pathlib import Path

from resume_screen import FakeClient, render, screen

DATA = Path(__file__).resolve().parent.parent / "data"

client = FakeClient(
    responses={
        # Routed first: the suggestions prompt quotes the requirement text, so
        # a requirement key would otherwise match it.
        "actionable suggestions": (
            '{"suggestions": ["State the request volume and latency your gateway handled",'
            ' "Name the orchestration tooling you have run in production"]}'
        ),
        "production Python": '{"verdict": "covered", "reason": "seven years of Python services"}',
        "speech recognition": '{"verdict": "covered", "reason": "deployed Whisper transcription"}',
        "streaming inference": '{"verdict": "covered", "reason": "400-session streaming gateway"}',
        "evaluation harnesses": '{"verdict": "covered", "reason": "paired bootstrap CIs"}',
        "Kubernetes": '{"verdict": "absent", "reason": "no orchestration experience shown"}',
        "quantised local LLM": '{"verdict": "partial", "reason": "benchmarked, did not deploy"}',
        "open-source": '{"verdict": "covered", "reason": "authored a benchmark tool"}',
        "Prometheus": 'Sure! ```json\n{"verdict": "covered", "reason": "built the dashboards"}\n```',
    },
)

result = screen(
    client,
    (DATA / "resume.md").read_text(encoding="utf-8"),
    (DATA / "job.md").read_text(encoding="utf-8"),
)
print(render(result))
