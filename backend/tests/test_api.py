import base64
import json
import socket
import threading
import urllib.error
import urllib.parse
import urllib.request
from http.server import ThreadingHTTPServer

import pytest

from resume_screen.api import (
    UPLOAD_BODY_LIMIT,
    ApiError,
    Config,
    handle_extract,
    handle_screen,
    make_handler,
)
from resume_screen.fakes import FakeClient
from resume_screen.ollama import OllamaError
from resume_screen.server import build_config, build_parser

RESUME = "## Experience\n- Built a streaming ASR gateway for live audio sessions\n- Wrote an evaluation harness with bootstrap intervals\n"
JOB = "## Requirements\n- Build streaming inference services\n- Kubernetes in production\n\n## Preferred qualifications\n- Open-source contributions\n"


def scripted():
    return FakeClient(
        responses={
            "streaming inference": '{"verdict": "covered", "reason": "gateway"}',
            "Kubernetes": '{"verdict": "absent", "reason": "not shown"}',
            "Open-source": '{"verdict": "partial", "reason": "unclear"}',
        }
    )


@pytest.fixture
def config():
    return Config(client_factory=scripted)


def post(config, payload):
    return handle_screen(config, json.dumps(payload).encode("utf-8"))


def test_screen_returns_a_verdict_per_requirement(config):
    body = post(config, {"resume": RESUME, "job": JOB})
    assert [r["verdict"] for r in body["requirements"]] == ["covered", "absent", "partial"]


def test_response_is_json_serialisable(config):
    body = post(config, {"resume": RESUME, "job": JOB})
    assert json.loads(json.dumps(body)) == body


def test_response_carries_the_calibration_flag(config):
    assert post(config, {"resume": RESUME, "job": JOB})["calibrated"] is True


def test_weaknesses_are_included_and_ordered(config):
    weaknesses = post(config, {"resume": RESUME, "job": JOB})["weaknesses"]
    assert weaknesses[0]["kind"] == "required" and weaknesses[0]["verdict"] == "absent"


def test_evidence_carries_both_scores(config):
    evidence = post(config, {"resume": RESUME, "job": JOB})["requirements"][0]["evidence"]
    assert {"text", "section", "specificity", "cosine"} == set(evidence[0])


def test_top_k_is_honoured(config):
    body = post(config, {"resume": RESUME, "job": JOB, "top_k": 1})
    assert all(len(r["evidence"]) == 1 for r in body["requirements"])


def test_unscorable_run_reports_null_not_zero():
    config = Config(client_factory=lambda: FakeClient(default="not json"))
    body = post(config, {"resume": RESUME, "job": JOB})
    assert body["score"] is None
    assert body["percent"] == "n/a"
    assert len(body["unscored"]) == 3


@pytest.mark.parametrize("payload", [{}, {"resume": RESUME}, {"job": JOB}])
def test_missing_fields_are_400(config, payload):
    with pytest.raises(ApiError) as exc:
        post(config, payload)
    assert exc.value.status == 400


@pytest.mark.parametrize("value", ["", "   ", 7, None, ["text"]])
def test_bad_resume_types_are_400(config, value):
    with pytest.raises(ApiError) as exc:
        post(config, {"resume": value, "job": JOB})
    assert exc.value.status == 400


def test_malformed_json_is_400(config):
    with pytest.raises(ApiError) as exc:
        handle_screen(config, b"{not json")
    assert exc.value.status == 400


def test_json_array_body_is_400(config):
    with pytest.raises(ApiError) as exc:
        handle_screen(config, b"[1, 2]")
    assert exc.value.status == 400


def test_boolean_top_k_is_rejected(config):
    # `isinstance(True, int)` is True; without an explicit bool check this
    # would quietly become top_k=1 and change every result.
    with pytest.raises(ApiError) as exc:
        post(config, {"resume": RESUME, "job": JOB, "top_k": True})
    assert exc.value.status == 400


@pytest.mark.parametrize("value", [0, -1, 2.5, "3"])
def test_bad_top_k_values_are_400(config, value):
    with pytest.raises(ApiError) as exc:
        post(config, {"resume": RESUME, "job": JOB, "top_k": value})
    assert exc.value.status == 400


def test_oversized_body_is_413(config):
    with pytest.raises(ApiError) as exc:
        handle_screen(config, b"x" * 1_000_001)
    assert exc.value.status == 413


def test_job_without_requirements_is_422(config):
    with pytest.raises(ApiError) as exc:
        post(config, {"resume": RESUME, "job": "We are hiring, send a note!"})
    assert exc.value.status == 422


def test_unreachable_model_backend_is_503():
    class Broken(FakeClient):
        def embed(self, texts):
            raise OllamaError("cannot reach Ollama at http://ollama:11434")

    config = Config(client_factory=Broken)
    with pytest.raises(ApiError) as exc:
        post(config, {"resume": RESUME, "job": JOB})
    assert exc.value.status == 503
    assert "ollama" in exc.value.message


def test_result_to_dict_rounds_scores(config):
    body = post(config, {"resume": RESUME, "job": JOB})
    cosine = body["requirements"][0]["evidence"][0]["cosine"]
    assert cosine == round(cosine, 4)


# --- server wiring -----------------------------------------------------------


def test_offline_flag_selects_the_fake_backend():
    config = build_config(build_parser().parse_args(["--offline"]))
    assert isinstance(config.client_factory(), FakeClient)


def test_offline_env_var_selects_the_fake_backend(monkeypatch):
    monkeypatch.setenv("OFFLINE", "true")
    assert isinstance(build_config(build_parser().parse_args([])).client_factory(), FakeClient)


def test_ollama_host_env_var_reaches_the_client(monkeypatch):
    monkeypatch.setenv("OLLAMA_HOST", "http://ollama:11434")
    monkeypatch.setenv("CHAT_MODEL", "qwen3:4b")
    client = build_config(build_parser().parse_args([])).client_factory()
    assert client.host == "http://ollama:11434"
    assert client.chat_model == "qwen3:4b"


def test_each_request_gets_its_own_client():
    config = build_config(build_parser().parse_args([]))
    assert config.client_factory() is not config.client_factory()


def test_suggestions_are_on_by_default():
    assert build_config(build_parser().parse_args([])).with_suggestions is True


def test_no_suggestions_flag_turns_them_off():
    args = build_parser().parse_args(["--no-suggestions"])
    assert build_config(args).with_suggestions is False


# --- a real socket round trip ------------------------------------------------


@pytest.fixture
def live_server():
    handler = make_handler(Config(client_factory=scripted))
    server = ThreadingHTTPServer(("127.0.0.1", 0), handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{server.server_address[1]}"
    server.shutdown()
    server.server_close()


def request(url, data=None, method=None):
    body = json.dumps(data).encode("utf-8") if data is not None else None
    req = urllib.request.Request(
        url, data=body, headers={"Content-Type": "application/json"}, method=method
    )
    with urllib.request.urlopen(req, timeout=10) as response:
        return response.status, dict(response.headers), json.loads(response.read() or b"{}")


def test_health_endpoint(live_server):
    status, _, body = request(f"{live_server}/health")
    assert (status, body) == (200, {"ok": True})


def test_screen_over_http(live_server):
    status, _, body = request(f"{live_server}/screen", {"resume": RESUME, "job": JOB})
    assert status == 200
    assert len(body["requirements"]) == 3


def test_cors_header_is_open(live_server):
    _, headers, _ = request(f"{live_server}/health")
    assert headers["Access-Control-Allow-Origin"] == "*"


def test_preflight_is_answered(live_server):
    status, headers, _ = request(f"{live_server}/screen", method="OPTIONS")
    assert status == 204
    assert "POST" in headers["Access-Control-Allow-Methods"]


def test_bad_request_returns_json_error(live_server):
    with pytest.raises(urllib.error.HTTPError) as exc:
        request(f"{live_server}/screen", {"resume": ""})
    assert exc.value.code == 400
    assert "error" in json.loads(exc.value.read())


def test_unknown_route_is_404(live_server):
    with pytest.raises(urllib.error.HTTPError) as exc:
        request(f"{live_server}/nope")
    assert exc.value.code == 404


def test_post_to_the_wrong_path_is_404(live_server):
    with pytest.raises(urllib.error.HTTPError) as exc:
        request(f"{live_server}/nope", {"resume": RESUME, "job": JOB})
    assert exc.value.code == 404


# --- upload and extraction ---------------------------------------------------


def upload(filename, data):
    return json.dumps(
        {"filename": filename, "content_base64": base64.b64encode(data).decode("ascii")}
    ).encode("utf-8")


CV = "\n".join(
    ["# Alex Rivera", "## Experience"]
    + [f"- shipped production system {i} with measurable impact" for i in range(12)]
)


def test_extract_returns_text_and_confidence():
    body = handle_extract(upload("cv.md", CV.encode("utf-8")))
    assert body["text"] == CV
    assert (body["format"], body["confidence"]) == ("text", "exact")


def test_extract_reports_the_word_count():
    assert handle_extract(upload("cv.md", CV.encode()))["words"] == len(CV.split())


def test_extract_response_shape():
    assert set(handle_extract(upload("cv.md", CV.encode()))) == {
        "text",
        "format",
        "confidence",
        "warnings",
        "words",
    }


def test_pdf_is_marked_approximate_with_a_warning():
    pdf = (
        b"%PDF-1.4\n1 0 obj\nstream\nBT /F1 12 Tf\n72 700 Td ("
        + CV.splitlines()[2].encode()
        + b") Tj\nET\nendstream\n%%EOF"
    )
    body = handle_extract(upload("cv.pdf", pdf))
    assert body["confidence"] == "approximate"
    assert any("approximate" in w for w in body["warnings"])


@pytest.mark.parametrize(
    "payload",
    [
        {},
        {"filename": "cv.md"},
        {"content_base64": "aGVsbG8="},
        {"filename": "", "content_base64": "aGVsbG8="},
        {"filename": "cv.md", "content_base64": ""},
        {"filename": 7, "content_base64": "aGVsbG8="},
        {"filename": "cv.md", "content_base64": 7},
    ],
)
def test_invalid_upload_payloads_are_400(payload):
    with pytest.raises(ApiError) as exc:
        handle_extract(json.dumps(payload).encode("utf-8"))
    assert exc.value.status == 400


def test_invalid_base64_is_400():
    body = json.dumps({"filename": "cv.md", "content_base64": "not base64!!"}).encode()
    with pytest.raises(ApiError) as exc:
        handle_extract(body)
    assert exc.value.status == 400
    assert "base64" in exc.value.message


def test_an_unusable_file_is_422_not_400():
    # The request was well formed; the file was not usable. A client can tell
    # "you sent it wrong" from "your file has no text in it".
    with pytest.raises(ApiError) as exc:
        handle_extract(upload("cv.pdf", b"%PDF-1.4\ntrailer\n%%EOF"))
    assert exc.value.status == 422


def test_a_legacy_doc_is_422_with_advice():
    with pytest.raises(ApiError) as exc:
        handle_extract(upload("cv.doc", b"\xd0\xcf\x11\xe0 old word"))
    assert exc.value.status == 422
    assert ".docx" in exc.value.message


def test_the_upload_route_has_a_larger_body_limit_than_screen():
    # Base64 inflates by 4/3, so the screen limit would reject a legal upload.
    from resume_screen.api import MAX_BODY_BYTES, UPLOAD_BODY_LIMIT

    assert UPLOAD_BODY_LIMIT > MAX_BODY_BYTES


def test_an_oversized_upload_is_413():
    body = json.dumps({"filename": "cv.md", "content_base64": "A" * (7_000_000)}).encode("utf-8")
    with pytest.raises(ApiError) as exc:
        handle_extract(body)
    assert exc.value.status == 413


def test_extract_over_http(live_server):
    status, _, body = request(
        f"{live_server}/extract",
        {"filename": "cv.md", "content_base64": base64.b64encode(CV.encode()).decode()},
    )
    assert status == 200 and body["text"] == CV


def test_extract_error_over_http_is_json(live_server):
    with pytest.raises(urllib.error.HTTPError) as exc:
        request(f"{live_server}/extract", {"filename": "cv.md", "content_base64": ""})
    assert exc.value.code == 400
    assert "error" in json.loads(exc.value.read())


# --- the analysis payload ----------------------------------------------------


def advising():
    return FakeClient(
        responses={
            "actionable suggestions": '{"suggestions": ["state the request volume you handled"]}',
            "streaming inference": '{"verdict": "covered", "reason": "gateway"}',
            "Kubernetes": '{"verdict": "absent", "reason": "not shown"}',
            "Open-source": '{"verdict": "partial", "reason": "unclear"}',
        }
    )


def test_response_carries_strengths_weaknesses_and_suggestions():
    body = post(Config(client_factory=advising), {"resume": RESUME, "job": JOB})
    assert [s["requirement"] for s in body["strengths"]] == ["Build streaming inference services"]
    assert {w["requirement"] for w in body["weaknesses"]} == {
        "Kubernetes in production",
        "Open-source contributions",
    }
    assert body["suggestions"]["items"] == ["state the request volume you handled"]


def test_suggestions_error_travels_with_the_payload():
    config = Config(client_factory=lambda: FakeClient(default="no json here"))
    body = post(config, {"resume": RESUME, "job": JOB})
    assert body["suggestions"]["items"] == []
    assert body["suggestions"]["error"]


def test_suggestions_can_be_disabled_without_changing_the_score():
    with_advice = post(Config(client_factory=advising), {"resume": RESUME, "job": JOB})
    without = post(
        Config(client_factory=advising, with_suggestions=False), {"resume": RESUME, "job": JOB}
    )
    assert with_advice["score"] == without["score"]
    assert without["suggestions"]["items"] == []


def test_a_query_string_does_not_break_routing(live_server):
    # `self.path == "/health"` would 404 here; urlparse drops the query.
    status, _, body = request(f"{live_server}/health?probe=1")
    assert (status, body) == (200, {"ok": True})


def test_a_path_that_merely_starts_with_a_route_name_is_404(live_server):
    with pytest.raises(urllib.error.HTTPError) as exc:
        request(f"{live_server}/healthcheck")
    assert exc.value.code == 404


def test_a_trailing_slash_still_routes(live_server):
    status, _, body = request(f"{live_server}/health/")
    assert (status, body) == (200, {"ok": True})


# --- request framing ---------------------------------------------------------


def read_http_response(sock) -> tuple[bytes, bytes]:
    """Read exactly one response: (status line, body).

    A single `recv` is not one response — it can stop after the headers, which
    leaves the body in the socket and makes the *next* read on a keep-alive
    connection return the tail of the previous message. Framing has to be
    honoured here or the test measures its own buffering.
    """
    buffer = b""
    while b"\r\n\r\n" not in buffer:
        chunk = sock.recv(65536)
        if not chunk:
            return buffer.split(b"\r\n")[0], b""
        buffer += chunk
    head, _, body = buffer.partition(b"\r\n\r\n")
    lengths = [
        line.split(b":", 1)[1].strip()
        for line in head.split(b"\r\n")
        if line.lower().startswith(b"content-length:")
    ]
    expected = int(lengths[0]) if lengths else 0
    while len(body) < expected:
        chunk = sock.recv(65536)
        if not chunk:
            break
        body += chunk
    return head.split(b"\r\n")[0], body


def raw_request(url, head, payload=b""):
    """Send a hand-built request, so malformed framing can be tested."""
    parsed = urllib.parse.urlparse(url)
    with socket.create_connection((parsed.hostname, parsed.port), timeout=10) as sock:
        sock.sendall(head + payload)
        status, body = read_http_response(sock)
    return status + b"\r\n\r\n" + body


def test_a_malformed_content_length_is_a_400_not_a_dropped_connection(live_server):
    # `int()` on the header used to raise inside do_POST, where nothing caught
    # it: the client got a closed socket and no way to tell what went wrong.
    head = (
        b"POST /screen HTTP/1.1\r\nHost: x\r\n"
        b"Content-Type: application/json\r\nContent-Length: abc\r\n\r\n"
    )
    response = raw_request(live_server, head)
    assert response.startswith(b"HTTP/1.1 400")
    assert b"Content-Length is not a number" in response


def test_a_negative_content_length_is_a_400(live_server):
    head = (
        b"POST /screen HTTP/1.1\r\nHost: x\r\n"
        b"Content-Type: application/json\r\nContent-Length: -5\r\n\r\n"
    )
    assert raw_request(live_server, head).startswith(b"HTTP/1.1 400")


def test_an_enormous_declared_body_is_refused_before_it_is_read(live_server):
    # The ceiling has to apply before `read(length)`, not after: a header
    # claiming 500 MB was previously honoured in full and then rejected.
    head = (
        b"POST /screen HTTP/1.1\r\nHost: x\r\n"
        b"Content-Type: application/json\r\nContent-Length: 500000000\r\n\r\n"
    )
    response = raw_request(live_server, head)
    assert response.startswith(b"HTTP/1.1 413")


def test_the_upload_route_accepts_a_body_larger_than_the_screen_limit(live_server):
    # base64 inflates by 4/3, so /extract must not inherit /screen's ceiling.
    payload = json.dumps(
        {"filename": "cv.md", "content_base64": base64.b64encode(CV.encode()).decode()}
    ).encode()
    assert len(payload) < UPLOAD_BODY_LIMIT
    head = (
        b"POST /extract HTTP/1.1\r\nHost: x\r\nContent-Type: application/json\r\n"
        b"Content-Length: %d\r\n\r\n" % len(payload)
    )
    assert raw_request(live_server, head, payload).startswith(b"HTTP/1.1 200")


def test_a_404_still_consumes_the_body_so_the_connection_survives(live_server):
    # Replying without draining the body desynchronises a keep-alive
    # connection, and the *next* request on it fails instead of this one.
    parsed = urllib.parse.urlparse(live_server)
    payload = b'{"resume": "x", "job": "y"}'
    with socket.create_connection((parsed.hostname, parsed.port), timeout=10) as sock:
        sock.sendall(
            b"POST /nope HTTP/1.1\r\nHost: x\r\nContent-Type: application/json\r\n"
            b"Content-Length: %d\r\n\r\n" % len(payload) + payload
        )
        status, _ = read_http_response(sock)
        assert status.startswith(b"HTTP/1.1 404")

        sock.sendall(b"GET /health HTTP/1.1\r\nHost: x\r\n\r\n")
        status, body = read_http_response(sock)
    assert status.startswith(b"HTTP/1.1 200")
    assert body == b'{"ok": true}'
