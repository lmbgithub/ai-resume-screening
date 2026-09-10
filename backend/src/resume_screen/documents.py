"""Turning a plain-text resume into the units that get compared.

The comparison unit is a *chunk*: one bullet, or one short paragraph. Nothing
here is model-aware — chunking is a structural problem and using an LLM for it
would make the pipeline slower, non-reproducible, and no more accurate.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

# The en dash in the character class is deliberate: Word and Pages
# autocorrect a leading hyphen-space into an en dash.
BULLET = re.compile(r"^\s*(?:[-*•–]|\d+[.)])\s+")  # noqa: RUF001
HEADING = re.compile(r"^\s*#{1,6}\s+(?P<title>.+?)\s*#*\s*$")
# An ALL-CAPS line with no sentence punctuation, e.g. "WORK EXPERIENCE".
CAPS_HEADING = re.compile(r"^\s*(?P<title>[A-Z][A-Z0-9 &/'()+.-]{2,60})\s*:?\s*$")

MIN_CHUNK_CHARS = 12


@dataclass(frozen=True)
class Chunk:
    """One comparable unit of a document."""

    text: str
    section: str
    index: int
    bullet: bool


def normalise(text: str) -> str:
    """Collapse whitespace inside a chunk without joining separate chunks."""
    return re.sub(r"[ \t]+", " ", text).strip()


def _is_heading(line: str) -> str | None:
    match = HEADING.match(line)
    if match:
        return normalise(match.group("title"))
    match = CAPS_HEADING.match(line)
    if match and not line.strip().endswith("."):
        title = match.group("title").strip().rstrip(":")
        # Require at least one lowercase-free word of substance, not "I" or "AWS."
        if len(title) >= 3:
            return normalise(title)
    return None


def chunk_document(text: str, *, default_section: str = "body") -> list[Chunk]:
    """Split a document into bullets and paragraphs, tagged with their section.

    Consecutive non-bullet lines join into one paragraph; a bullet always
    starts a new chunk, because two bullets under one heading are two separate
    claims and averaging them is how a strong line gets buried by a weak one.
    """
    chunks: list[Chunk] = []
    section = default_section
    pending: list[str] = []

    def flush(*, bullet: bool = False) -> None:
        if not pending:
            return
        body = normalise(" ".join(pending))
        pending.clear()
        if len(body) >= MIN_CHUNK_CHARS:
            chunks.append(Chunk(text=body, section=section, index=len(chunks), bullet=bullet))

    for raw_line in text.splitlines():
        line = raw_line.rstrip()
        if not line.strip():
            flush()
            continue
        heading = _is_heading(line)
        if heading is not None:
            flush()
            section = heading
            continue
        if BULLET.match(line):
            flush()
            pending.append(BULLET.sub("", line))
            flush(bullet=True)
            continue
        pending.append(line.strip())
    flush()
    return chunks


def load_document(path: str) -> str:
    """Read a UTF-8 text document.

    Deliberately text-only: see "Not included" in the README for why PDF
    extraction is out of scope rather than half-done.
    """
    with open(path, encoding="utf-8") as handle:
        return handle.read()
