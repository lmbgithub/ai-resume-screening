import math

import pytest

from resume_screen.scoring import (
    Aggregate,
    ScoringError,
    aggregate,
    cosine,
    double_center,
    is_degenerate,
    rank_chunks,
    similarity_matrix,
)


def test_cosine_of_identical_vectors_is_one():
    assert cosine([1.0, 2.0, 3.0], [1.0, 2.0, 3.0]) == pytest.approx(1.0)


def test_cosine_of_orthogonal_vectors_is_zero():
    assert cosine([1.0, 0.0], [0.0, 1.0]) == pytest.approx(0.0)


def test_cosine_of_opposite_vectors_is_minus_one():
    assert cosine([1.0, 0.0], [-1.0, 0.0]) == pytest.approx(-1.0)


def test_cosine_is_scale_invariant():
    assert cosine([1.0, 2.0], [10.0, 20.0]) == pytest.approx(1.0)


def test_cosine_known_value():
    assert cosine([1.0, 1.0], [1.0, 0.0]) == pytest.approx(1 / math.sqrt(2))


def test_cosine_with_a_zero_vector_is_zero_not_a_crash():
    assert cosine([0.0, 0.0], [1.0, 2.0]) == 0.0


def test_cosine_rejects_dimension_mismatch():
    with pytest.raises(ScoringError):
        cosine([1.0], [1.0, 2.0])


def test_cosine_rejects_empty_vectors():
    with pytest.raises(ScoringError):
        cosine([], [])


def test_cosine_stays_within_bounds_under_float_error():
    vector = [0.1] * 300
    assert -1.0 <= cosine(vector, vector) <= 1.0


def test_similarity_matrix_shape():
    matrix = similarity_matrix([[1.0, 0.0], [0.0, 1.0]], [[1.0, 0.0]])
    assert len(matrix) == 2 and len(matrix[0]) == 1


def test_double_center_rows_and_columns_sum_to_zero():
    matrix = [[0.9, 0.4, 0.3], [0.5, 0.8, 0.2], [0.1, 0.2, 0.7]]
    centered = double_center(matrix)
    for row in centered:
        assert sum(row) == pytest.approx(0.0, abs=1e-12)
    for c in range(3):
        assert sum(row[c] for row in centered) == pytest.approx(0.0, abs=1e-12)


def test_double_center_known_value():
    # row means 1.5 / 3.5, col means 2.0 / 3.0, grand mean 2.5
    # residual[0][0] = 1 - 1.5 - 2.0 + 2.5 = 0.0
    assert double_center([[1.0, 2.0], [3.0, 4.0]]) == [
        [pytest.approx(0.0)] * 2,
        [pytest.approx(0.0)] * 2,
    ]


def test_double_center_removes_a_constant_row_offset():
    # A requirement whose wording is similar to everything gains nothing.
    base = [[0.5, 0.2], [0.4, 0.9]]
    lifted = [[v + 0.3 for v in base[0]], list(base[1])]
    assert double_center(lifted)[0] == [pytest.approx(v) for v in double_center(base)[0]]


def test_double_center_removes_a_generic_column():
    # A resume line similar to every requirement stops dominating retrieval.
    # Column 0 is the resume's summary line: high similarity to everything.
    matrix = [[0.9, 0.8, 0.2], [0.9, 0.2, 0.85]]
    centered = double_center(matrix)
    assert matrix[1][0] > matrix[1][2]  # raw cosine prefers the generic line
    assert centered[1][0] < centered[1][2]  # centering prefers the specific one


def test_double_center_of_empty_matrix():
    assert double_center([]) == []
    assert double_center([[]]) == []


def test_double_center_rejects_ragged_matrix():
    with pytest.raises(ScoringError):
        double_center([[1.0, 2.0], [3.0]])


def test_single_row_matrix_is_degenerate():
    assert is_degenerate([[0.1, 0.2, 0.3]])


def test_single_column_matrix_is_degenerate():
    assert is_degenerate([[0.1], [0.2]])


def test_two_by_two_matrix_is_not_degenerate():
    assert not is_degenerate([[0.1, 0.2], [0.3, 0.4]])


def test_degenerate_matrix_centres_to_all_zeros():
    # Which is why the pipeline must fall back rather than rank on it.
    assert double_center([[0.9, 0.1, 0.5]]) == [[pytest.approx(0.0)] * 3]


def test_rank_chunks_orders_by_centered_score():
    ranked = rank_chunks([0.9, 0.1, 0.5], [0.1, 0.7, 0.3], top_k=3)
    assert [i for i, _, _ in ranked] == [1, 2, 0]


def test_rank_chunks_returns_both_scores():
    ranked = rank_chunks([0.9], [0.4], top_k=1)
    assert ranked == [(0, 0.4, 0.9)]


def test_rank_chunks_truncates_to_top_k():
    assert len(rank_chunks([0.1] * 5, [0.5, 0.4, 0.3, 0.2, 0.1], top_k=2)) == 2


def test_rank_chunks_top_k_larger_than_input_is_fine():
    assert len(rank_chunks([0.1, 0.2], [0.1, 0.2], top_k=99)) == 2


def test_rank_chunks_breaks_ties_on_raw_then_index():
    ranked = rank_chunks([0.2, 0.8, 0.8], [0.5, 0.5, 0.5], top_k=3)
    assert [i for i, _, _ in ranked] == [1, 2, 0]


def test_rank_chunks_rejects_non_positive_top_k():
    with pytest.raises(ScoringError):
        rank_chunks([0.1], [0.1], top_k=0)


def test_rank_chunks_rejects_mismatched_rows():
    with pytest.raises(ScoringError):
        rank_chunks([0.1, 0.2], [0.1], top_k=1)


def test_aggregate_all_covered_is_one():
    assert aggregate([("covered", 1.0, "a"), ("covered", 0.5, "b")]).score == 1.0


def test_aggregate_all_absent_is_zero():
    assert aggregate([("absent", 1.0, "a")]).score == 0.0


def test_aggregate_partial_counts_half():
    assert aggregate([("partial", 1.0, "a")]).score == pytest.approx(0.5)


def test_aggregate_weights_required_above_preferred():
    weighted = aggregate([("covered", 1.0, "req"), ("absent", 0.5, "pref")])
    assert weighted.score == pytest.approx(1.0 / 1.5)


def test_unknown_is_excluded_from_the_denominator():
    # The candidate must not be charged for our parser failing.
    result = aggregate([("covered", 1.0, "a"), ("unknown", 1.0, "b")])
    assert result.score == 1.0
    assert result.scored_weight == 1.0


def test_unknown_requirements_are_named():
    result = aggregate([("covered", 1.0, "a"), ("unknown", 1.0, "b is broken")])
    assert result.unscored == ("b is broken",)


def test_all_unknown_yields_no_score_rather_than_zero():
    result = aggregate([("unknown", 1.0, "a")])
    assert result.score is None
    assert result.percent == "n/a"


def test_empty_input_yields_no_score():
    assert aggregate([]).score is None


def test_aggregate_rejects_non_positive_weight():
    with pytest.raises(ScoringError):
        aggregate([("covered", 0.0, "a")])


def test_percent_formatting():
    assert Aggregate(score=0.8075, scored_weight=1.0, unscored=()).percent == "80.8%"
