"""Similarity, calibration and aggregation. No model calls happen here.

Two ideas carry this module:

1. A raw cosine is not a match percentage. Embedding models put unrelated
   English text at 0.5-0.7, so "78% match" is a number with an arbitrary
   origin. Calibrating removes it.
2. A resume line that resembles every requirement equally is evidence for
   none of them. Double-centering the similarity matrix subtracts both the
   requirement's own generic level and the line's, leaving only the part that
   is specific to that pair.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

Vector = Sequence[float]
Matrix = list[list[float]]


class ScoringError(ValueError):
    """The inputs cannot produce a meaningful score."""


def cosine(a: Vector, b: Vector) -> float:
    """Cosine similarity, clamped to [-1, 1] against float error."""
    if len(a) != len(b):
        raise ScoringError(f"dimension mismatch: {len(a)} vs {len(b)}")
    if not a:
        raise ScoringError("cannot take the cosine of empty vectors")
    dot = sum(x * y for x, y in zip(a, b, strict=True))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(y * y for y in b))
    if norm_a == 0.0 or norm_b == 0.0:
        # A zero vector has no direction; 0.0 is the only defensible answer,
        # and it must not be confused with "orthogonal, therefore measured".
        return 0.0
    return max(-1.0, min(1.0, dot / (norm_a * norm_b)))


def similarity_matrix(rows: Sequence[Vector], cols: Sequence[Vector]) -> Matrix:
    """Raw cosine of every (requirement, chunk) pair."""
    return [[cosine(row, col) for col in cols] for row in rows]


def double_center(matrix: Matrix) -> Matrix:
    """Subtract row and column means, add back the grand mean.

    residual[r][c] = S[r][c] - mean(row r) - mean(col c) + mean(S)

    With a single row or a single column every residual is 0 by construction:
    there is nothing to compare a pair against. The caller must fall back to
    raw similarity there rather than pretend the zeros are informative.
    """
    if not matrix or not matrix[0]:
        return []
    width = len(matrix[0])
    if any(len(row) != width for row in matrix):
        raise ScoringError("similarity matrix is ragged")
    row_means = [sum(row) / width for row in matrix]
    col_means = [sum(row[c] for row in matrix) / len(matrix) for c in range(width)]
    grand = sum(row_means) / len(row_means)
    return [
        [value - row_means[r] - col_means[c] + grand for c, value in enumerate(row)]
        for r, row in enumerate(matrix)
    ]


def is_degenerate(matrix: Matrix) -> bool:
    """True when double-centering cannot say anything (one row or one column)."""
    return len(matrix) < 2 or (bool(matrix) and len(matrix[0]) < 2)


def rank_chunks(
    raw_row: Sequence[float], centered_row: Sequence[float], *, top_k: int
) -> list[tuple[int, float, float]]:
    """Best chunks for one requirement as (index, centered, raw), best first.

    Ranking is by the centered score. Ties break on the raw score and then on
    the chunk index, so the output is stable across runs and platforms.
    """
    if top_k <= 0:
        raise ScoringError("top_k must be positive")
    if len(raw_row) != len(centered_row):
        raise ScoringError("raw and centered rows have different lengths")
    ordered = sorted(
        ((i, centered_row[i], raw_row[i]) for i in range(len(raw_row))),
        key=lambda item: (-item[1], -item[2], item[0]),
    )
    return ordered[:top_k]


VERDICT_CREDIT = {"covered": 1.0, "partial": 0.5, "absent": 0.0}


@dataclass(frozen=True)
class Aggregate:
    """The final score, and the accounting behind it."""

    score: float | None
    scored_weight: float
    unscored: tuple[str, ...]

    @property
    def percent(self) -> str:
        return "n/a" if self.score is None else f"{self.score * 100:.1f}%"


def aggregate(verdicts: Sequence[tuple[str, float, str]]) -> Aggregate:
    """Weighted coverage over requirements, as (verdict, weight, label) triples.

    An `unknown` verdict — the model's answer could not be parsed — is removed
    from *both* the numerator and the denominator, and named in the result.
    Scoring it as 0 would charge the candidate for our parser's failure, which
    is the single most common way a screening pipeline invents a rejection.
    """
    numerator = 0.0
    denominator = 0.0
    unscored: list[str] = []
    for verdict, weight, label in verdicts:
        if weight <= 0:
            raise ScoringError(f"requirement {label!r} has non-positive weight {weight}")
        if verdict not in VERDICT_CREDIT:
            unscored.append(label)
            continue
        numerator += VERDICT_CREDIT[verdict] * weight
        denominator += weight
    if denominator == 0.0:
        return Aggregate(score=None, scored_weight=0.0, unscored=tuple(unscored))
    return Aggregate(
        score=numerator / denominator,
        scored_weight=denominator,
        unscored=tuple(unscored),
    )
