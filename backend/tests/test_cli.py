import pytest

from resume_screen import cli
from resume_screen.ollama import OllamaClient, OllamaError

RESUME = "## Experience\n- Built a streaming ASR gateway for live audio sessions\n- Wrote an evaluation harness with bootstrap intervals\n"
JOB = "## Requirements\n- Build streaming inference services\n- Kubernetes in production\n"


@pytest.fixture
def docs(tmp_path):
    resume = tmp_path / "resume.md"
    job = tmp_path / "job.md"
    resume.write_text(RESUME, encoding="utf-8")
    job.write_text(JOB, encoding="utf-8")
    return str(resume), str(job)


def test_offline_run_succeeds_without_a_server(docs, capsys):
    resume, job = docs
    code = cli.main(["--resume", resume, "--job", job, "--offline"])
    assert code == 0  # every requirement scored, even if all "absent"
    assert "Match:" in capsys.readouterr().out


def test_offline_run_prints_every_requirement(docs, capsys):
    resume, job = docs
    cli.main(["--resume", resume, "--job", job, "--offline"])
    out = capsys.readouterr().out
    assert "streaming inference" in out and "Kubernetes" in out


def test_missing_resume_file_is_exit_code_two(docs, capsys):
    _, job = docs
    assert cli.main(["--resume", "/nope/x.md", "--job", job, "--offline"]) == 2
    assert "error:" in capsys.readouterr().err


def test_job_with_no_requirements_is_exit_code_two(tmp_path, docs, capsys):
    resume, _ = docs
    empty = tmp_path / "empty.md"
    empty.write_text("We are hiring!", encoding="utf-8")
    assert cli.main(["--resume", resume, "--job", str(empty), "--offline"]) == 2
    assert "no requirements" in capsys.readouterr().err


def test_unreachable_server_is_exit_code_three(docs, capsys, monkeypatch):
    resume, job = docs

    def explode(self, texts):
        raise OllamaError("cannot reach Ollama")

    monkeypatch.setattr(OllamaClient, "embed", explode)
    assert cli.main(["--resume", resume, "--job", job]) == 3
    assert "cannot reach Ollama" in capsys.readouterr().err


def test_unscorable_run_exits_nonzero(docs, capsys, monkeypatch):
    # A pipeline step that could not score every requirement must not look
    # like a clean pass to whatever script is calling it.
    from resume_screen.fakes import FakeClient

    monkeypatch.setattr(cli, "FakeClient", lambda: FakeClient(default="not json"))
    resume, job = docs
    assert cli.main(["--resume", resume, "--job", job, "--offline"]) == 1
    assert "unscored" in capsys.readouterr().out


def test_fully_scored_run_exits_zero(docs, capsys, monkeypatch):
    from resume_screen.fakes import FakeClient

    monkeypatch.setattr(cli, "FakeClient", lambda: FakeClient(default='{"verdict": "covered"}'))
    resume, job = docs
    assert cli.main(["--resume", resume, "--job", job, "--offline"]) == 0


def test_non_positive_top_k_is_rejected(docs, capsys):
    resume, job = docs
    assert cli.main(["--resume", resume, "--job", job, "--offline", "--top-k", "0"]) == 2
    assert "--top-k" in capsys.readouterr().err


def test_resume_and_job_are_required():
    with pytest.raises(SystemExit):
        cli.build_parser().parse_args(["--resume", "only.md"])


def test_model_flags_reach_the_client(docs, monkeypatch):
    seen = {}

    class Spy(OllamaClient):
        def __init__(self, **kwargs):
            seen.update(kwargs)
            super().__init__(**kwargs)

        def embed(self, texts):
            raise OllamaError("stop here")

    monkeypatch.setattr(cli, "OllamaClient", Spy)
    resume, job = docs
    cli.main(
        [
            "--resume",
            resume,
            "--job",
            job,
            "--chat-model",
            "qwen3:4b",
            "--embed-model",
            "mxbai",
            "--host",
            "http://h:1",
            "--timeout",
            "5",
        ]
    )
    assert seen == {
        "host": "http://h:1",
        "chat_model": "qwen3:4b",
        "embed_model": "mxbai",
        "timeout": 5.0,
    }


def test_offline_flag_never_builds_a_network_client(docs, monkeypatch):
    def explode(**kwargs):
        raise AssertionError("--offline must not construct an OllamaClient")

    monkeypatch.setattr(cli, "OllamaClient", explode)
    resume, job = docs
    cli.main(["--resume", resume, "--job", job, "--offline"])
