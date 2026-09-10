"""A JSON HTTP API over the screening pipeline, on `http.server`.

FastAPI would be the reflex here. It was rejected for the same reason the rest
of the project has no dependencies: this API is three routes with no auth, no
sessions and no schema evolution, and `ThreadingHTTPServer` already handles
concurrent requests. The cost of the reflex is a container that has to install
a framework, a validation library and an ASGI server to serve a POST.

No authentication, no rate limiting, and CORS is open to every origin. Nothing
is stored: an uploaded CV lives in memory for the length of the request and is
never written to disk. That is deliberate for a local demo, stated here so it
cannot be mistaken for an oversight — see "Not included" in the README.
"""

from __future__ import annotations

import base64
import binascii
import json
from collections.abc import Callable
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse

from .extract import MAX_UPLOAD_BYTES, ExtractionError, extract
from .ollama import Client, OllamaError
from .pipeline import DEFAULT_TOP_K, EmptyDocumentError, screen
from .serialize import result_to_dict

MAX_BODY_BYTES = 1_000_000  # a resume is kilobytes; anything larger is a mistake
# Base64 inflates by 4/3, so the upload route needs its own, larger ceiling.
UPLOAD_BODY_LIMIT = MAX_UPLOAD_BYTES * 4 // 3 + 1024


@dataclass(frozen=True)
class Config:
    """Everything the handler needs, injected so tests can supply a fake."""

    client_factory: Callable[[], Client]
    top_k: int = DEFAULT_TOP_K
    with_suggestions: bool = True
    on_progress: Callable[[str], None] | None = None


class ApiError(Exception):
    """A failure with an HTTP status already decided."""

    def __init__(self, status: int, message: str):
        super().__init__(message)
        self.status = status
        self.message = message


def handle_screen(config: Config, body: bytes) -> dict:
    """The POST /screen route as a pure function of its input.

    Separated from the handler class so every branch is testable without
    binding a socket.
    """
    payload = _decode(body)

    resume = _require_text(payload, "resume")
    job = _require_text(payload, "job")

    top_k = payload.get("top_k", config.top_k)
    # `isinstance(True, int)` is True in Python, so `bool` must be excluded
    # explicitly or `{"top_k": true}` would silently mean top_k=1.
    if isinstance(top_k, bool) or not isinstance(top_k, int) or top_k < 1:
        raise ApiError(400, "'top_k' must be a positive integer")

    try:
        result = screen(
            config.client_factory(),
            resume,
            job,
            top_k=top_k,
            with_suggestions=config.with_suggestions,
            on_progress=config.on_progress,
        )
    except EmptyDocumentError as exc:
        raise ApiError(422, str(exc)) from exc
    except OllamaError as exc:
        # 503: the request was fine, the model backend was not.
        raise ApiError(503, str(exc)) from exc
    return result_to_dict(result)


def _require_text(payload: dict, field: str) -> str:
    value = payload.get(field)
    if not isinstance(value, str) or not value.strip():
        raise ApiError(400, f"'{field}' must be a non-empty string")
    return value


def _decode(body: bytes, *, limit: int = MAX_BODY_BYTES) -> dict:
    if len(body) > limit:
        raise ApiError(413, f"request body exceeds {limit} bytes")
    try:
        payload = json.loads(body or b"{}")
    except json.JSONDecodeError as exc:
        raise ApiError(400, f"body is not valid JSON: {exc.msg}") from exc
    if not isinstance(payload, dict):
        raise ApiError(400, "body must be a JSON object")
    return payload


def handle_extract(body: bytes) -> dict:
    """POST /extract — a base64 file in, text plus a confidence rating out.

    The upload is base64 inside JSON rather than multipart/form-data. That
    costs 33% in transfer for a file measured in kilobytes, and buys an API
    with exactly one content type, no multipart parser to get wrong, and a
    request that is inspectable in a terminal. Multipart would be the right
    call for large or many files; this is one CV.

    Extraction is deliberately a separate round trip from screening: the
    caller has to see what came out of a PDF before anything is scored on it.
    """
    payload = _decode(body, limit=UPLOAD_BODY_LIMIT)
    filename = payload.get("filename")
    encoded = payload.get("content_base64")
    if not isinstance(filename, str) or not filename.strip():
        raise ApiError(400, "'filename' must be a non-empty string")
    if not isinstance(encoded, str) or not encoded:
        raise ApiError(400, "'content_base64' must be a non-empty base64 string")
    try:
        data = base64.b64decode(encoded, validate=True)
    except (binascii.Error, ValueError) as exc:
        raise ApiError(400, f"'content_base64' is not valid base64: {exc}") from exc

    try:
        return extract(filename, data).to_dict()
    except ExtractionError as exc:
        # 422: the request was well formed, the file was not usable.
        raise ApiError(422, str(exc)) from exc


def make_handler(config: Config) -> type[BaseHTTPRequestHandler]:
    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"
        server_version = "resume-screen"

        def _send(self, status: int, payload: dict) -> None:
            body = json.dumps(payload).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Access-Control-Allow-Origin", "*")
            self.send_header("Access-Control-Allow-Headers", "Content-Type")
            self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
            self.end_headers()
            self.wfile.write(body)

        def _route(self, status: int, work: Callable[[], dict]) -> None:
            try:
                self._send(status, work())
            except ApiError as exc:
                self._send(exc.status, {"error": exc.message})
            except Exception as exc:  # last resort: never leak a stack trace
                self._send(500, {"error": f"unexpected {type(exc).__name__}: {exc}"})

        def _body(self) -> bytes:
            length = int(self.headers.get("Content-Length") or 0)
            return self.rfile.read(length) if length else b""

        def _segments(self) -> list[str]:
            """Path segments, parsed rather than string-matched.

            `urlparse` drops any query string, so `/health?probe=1` still
            routes; `self.path == "/health"` would not, and `startswith`
            would also match `/healthcheck`.
            """
            return [p for p in urlparse(self.path).path.split("/") if p]

        def do_OPTIONS(self) -> None:
            self._send(204, {})

        def do_GET(self) -> None:
            parts = self._segments()
            if parts == ["health"]:
                self._send(200, {"ok": True})
            else:
                self._send(404, {"error": f"no route for GET {self.path}"})

        def do_POST(self) -> None:
            parts = self._segments()
            body = self._body()
            if parts == ["screen"]:
                self._route(200, lambda: handle_screen(config, body))
            elif parts == ["extract"]:
                self._route(200, lambda: handle_extract(body))
            else:
                self._send(404, {"error": f"no route for POST {self.path}"})

        def log_message(self, fmt: str, *args) -> None:
            print(
                f"{self.command} {self.path} -> {args[1] if len(args) > 1 else '?'}",
                flush=True,
            )

    return Handler


def serve(
    host: str = "0.0.0.0",
    port: int = 8000,
    *,
    config: Config,
) -> None:  # pragma: no cover - exercised by the container, not the suite
    server = ThreadingHTTPServer((host, port), make_handler(config))
    print(f"resume-screen api on http://{host}:{port}")
    server.serve_forever()
