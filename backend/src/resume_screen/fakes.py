"""A deterministic backend, so the whole pipeline is testable without a model.

Embeddings are a hashed bag of words: two texts that share vocabulary land
close together, and the same text always produces the same vector. That is
enough structure for the retrieval and calibration machinery to be checked
against similarities computed by hand.
"""

from __future__ import annotations

import hashlib
import re
from collections.abc import Sequence
from dataclasses import dataclass, field

DIMENSIONS = 64
_WORD = re.compile(r"[A-Za-z0-9+#.]+")


def hashed_embedding(text: str, dimensions: int = DIMENSIONS) -> list[float]:
    """Bag-of-words hashing, L2-normalised. Deterministic across processes."""
    vector = [0.0] * dimensions
    for word in _WORD.findall(text.lower()):
        digest = hashlib.sha256(word.encode("utf-8")).digest()
        bucket = int.from_bytes(digest[:4], "big") % dimensions
        sign = 1.0 if digest[4] % 2 == 0 else -1.0
        vector[bucket] += sign
    norm = sum(x * x for x in vector) ** 0.5
    return vector if norm == 0.0 else [x / norm for x in vector]


@dataclass
class FakeClient:
    """Scripted generations plus hashed embeddings.

    `responses` maps a substring of the prompt to the reply. The first match
    wins; anything unmatched gets `default`. Every call is recorded, so a test
    can assert on what the pipeline actually sent the model.
    """

    responses: dict[str, str] = field(default_factory=dict)
    default: str = '{"verdict": "absent", "reason": "no evidence"}'
    dimensions: int = DIMENSIONS
    prompts: list[str] = field(default_factory=list)
    embed_calls: list[list[str]] = field(default_factory=list)

    def generate(self, prompt: str, *, json_mode: bool = False) -> str:
        self.prompts.append(prompt)
        for needle, reply in self.responses.items():
            if needle in prompt:
                return reply
        return self.default

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        self.embed_calls.append(list(texts))
        return [hashed_embedding(text, self.dimensions) for text in texts]
