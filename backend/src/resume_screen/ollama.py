"""HTTP client for a local Ollama server, built on urllib.

The only network dependency in the project. Everything above this module talks
to the `Client` protocol, so the whole pipeline runs offline against a fake.
"""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol

DEFAULT_HOST = "http://localhost:11434"
DEFAULT_CHAT_MODEL = "gemma4:e2b"
DEFAULT_EMBED_MODEL = "nomic-embed-text"


class OllamaError(RuntimeError):
    """The server was unreachable, slow, or answered with something unusable."""


class Client(Protocol):
    """What the pipeline needs from a model backend. Nothing more."""

    def generate(self, prompt: str, *, json_mode: bool = False) -> str: ...

    def embed(self, texts: Sequence[str]) -> list[list[float]]: ...


@dataclass(frozen=True)
class OllamaClient:
    """Talks to `ollama serve` on localhost.

    `temperature` defaults to 0. Extraction is not a creative task, and a
    screening decision that changes between runs is not a decision.
    """

    host: str = DEFAULT_HOST
    chat_model: str = DEFAULT_CHAT_MODEL
    embed_model: str = DEFAULT_EMBED_MODEL
    timeout: float = 120.0
    temperature: float = 0.0

    def generate(self, prompt: str, *, json_mode: bool = False) -> str:
        payload: dict[str, object] = {
            "model": self.chat_model,
            "prompt": prompt,
            "stream": False,
            "options": {"temperature": self.temperature},
        }
        if json_mode:
            payload["format"] = "json"
        body = self._post("/api/generate", payload)
        response = body.get("response")
        if not isinstance(response, str):
            raise OllamaError(f"/api/generate returned no 'response' field: {body!r}")
        return response

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        if not texts:
            return []
        body = self._post("/api/embed", {"model": self.embed_model, "input": list(texts)})
        vectors = body.get("embeddings")
        if not isinstance(vectors, list) or len(vectors) != len(texts):
            raise OllamaError(
                f"/api/embed returned {len(vectors) if isinstance(vectors, list) else '?'} "
                f"vectors for {len(texts)} inputs"
            )
        return [[float(x) for x in vector] for vector in vectors]

    def _post(self, path: str, payload: dict[str, object]) -> dict:
        request = urllib.request.Request(
            self.host.rstrip("/") + path,
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                raw = response.read()
        except urllib.error.HTTPError as exc:  # 404 on an un-pulled model, mostly
            detail = exc.read().decode("utf-8", "replace")[:200]
            raise OllamaError(f"{path} failed with HTTP {exc.code}: {detail}") from exc
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            raise OllamaError(
                f"cannot reach Ollama at {self.host} ({exc}). Is `ollama serve` running?"
            ) from exc
        try:
            body = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise OllamaError(f"{path} returned non-JSON: {raw[:200]!r}") from exc
        if not isinstance(body, dict):
            raise OllamaError(f"{path} returned {type(body).__name__}, expected object")
        return body
