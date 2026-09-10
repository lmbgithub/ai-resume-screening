import json
import urllib.error

import pytest

from resume_screen import ollama
from resume_screen.ollama import OllamaClient, OllamaError


class FakeResponse:
    def __init__(self, payload: bytes):
        self.payload = payload

    def read(self):
        return self.payload

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def patch_urlopen(monkeypatch, payload, capture=None):
    def fake(request, timeout=None):
        if capture is not None:
            capture["url"] = request.full_url
            capture["body"] = json.loads(request.data)
            capture["timeout"] = timeout
        if isinstance(payload, Exception):
            raise payload
        return FakeResponse(payload)

    monkeypatch.setattr(ollama.urllib.request, "urlopen", fake)


def test_generate_returns_the_response_field(monkeypatch):
    patch_urlopen(monkeypatch, b'{"response": "hello"}')
    assert OllamaClient().generate("prompt") == "hello"


def test_generate_defaults_to_temperature_zero(monkeypatch):
    seen = {}
    patch_urlopen(monkeypatch, b'{"response": "x"}', seen)
    OllamaClient().generate("prompt")
    assert seen["body"]["options"]["temperature"] == 0.0
    assert seen["body"]["stream"] is False


def test_json_mode_sets_the_format_field(monkeypatch):
    seen = {}
    patch_urlopen(monkeypatch, b'{"response": "x"}', seen)
    OllamaClient().generate("prompt", json_mode=True)
    assert seen["body"]["format"] == "json"


def test_json_mode_is_off_by_default(monkeypatch):
    seen = {}
    patch_urlopen(monkeypatch, b'{"response": "x"}', seen)
    OllamaClient().generate("prompt")
    assert "format" not in seen["body"]


def test_generate_without_a_response_field_raises(monkeypatch):
    patch_urlopen(monkeypatch, b'{"error": "model not found"}')
    with pytest.raises(OllamaError, match="no 'response'"):
        OllamaClient().generate("prompt")


def test_embed_returns_one_vector_per_input(monkeypatch):
    patch_urlopen(monkeypatch, b'{"embeddings": [[1, 2], [3, 4]]}')
    assert OllamaClient().embed(["a", "b"]) == [[1.0, 2.0], [3.0, 4.0]]


def test_embed_coerces_ints_to_floats(monkeypatch):
    patch_urlopen(monkeypatch, b'{"embeddings": [[1, 2]]}')
    assert OllamaClient().embed(["a"]) == [[1.0, 2.0]]


def test_embed_of_nothing_makes_no_request(monkeypatch):
    def explode(*args, **kwargs):
        raise AssertionError("should not have called the server")

    monkeypatch.setattr(ollama.urllib.request, "urlopen", explode)
    assert OllamaClient().embed([]) == []


def test_embed_detects_a_count_mismatch(monkeypatch):
    # Silently accepting fewer vectors than inputs would misalign the whole
    # similarity matrix and produce a plausible, wrong report.
    patch_urlopen(monkeypatch, b'{"embeddings": [[1, 2]]}')
    with pytest.raises(OllamaError, match="1 vectors for 2 inputs"):
        OllamaClient().embed(["a", "b"])


def test_missing_embeddings_field_raises(monkeypatch):
    patch_urlopen(monkeypatch, b'{"nope": true}')
    with pytest.raises(OllamaError):
        OllamaClient().embed(["a"])


def test_connection_refused_names_the_host(monkeypatch):
    patch_urlopen(monkeypatch, urllib.error.URLError("refused"))
    with pytest.raises(OllamaError, match="ollama serve"):
        OllamaClient(host="http://localhost:9999").generate("p")


def test_http_error_includes_the_status(monkeypatch):
    error = urllib.error.HTTPError("u", 404, "Not Found", {}, None)
    monkeypatch.setattr(error, "read", lambda: b'{"error":"model not found"}')
    patch_urlopen(monkeypatch, error)
    with pytest.raises(OllamaError, match="HTTP 404"):
        OllamaClient().generate("p")


def test_timeout_raises_ollama_error(monkeypatch):
    patch_urlopen(monkeypatch, TimeoutError("slow"))
    with pytest.raises(OllamaError):
        OllamaClient().generate("p")


def test_non_json_body_raises(monkeypatch):
    patch_urlopen(monkeypatch, b"<html>proxy error</html>")
    with pytest.raises(OllamaError, match="non-JSON"):
        OllamaClient().generate("p")


def test_json_list_body_raises(monkeypatch):
    patch_urlopen(monkeypatch, b"[1, 2]")
    with pytest.raises(OllamaError, match="expected object"):
        OllamaClient().generate("p")


def test_trailing_slash_in_the_host_does_not_double_up(monkeypatch):
    seen = {}
    patch_urlopen(monkeypatch, b'{"response": "x"}', seen)
    OllamaClient(host="http://localhost:11434/").generate("p")
    assert seen["url"] == "http://localhost:11434/api/generate"


def test_timeout_is_passed_to_urlopen(monkeypatch):
    seen = {}
    patch_urlopen(monkeypatch, b'{"response": "x"}', seen)
    OllamaClient(timeout=7.5).generate("p")
    assert seen["timeout"] == 7.5
