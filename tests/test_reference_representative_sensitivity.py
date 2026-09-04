import numpy as np

from src.reference_representative_sensitivity import (
    EXPECTED_ARCHIVE_SHA256,
    PINNED_COMMIT_ARCHIVE_SHA256,
    compare_reference_matrices,
    normalize_reference_values,
    validate_source_archive_hash,
)


def test_both_byte_verified_archives_are_accepted():
    assert validate_source_archive_hash(EXPECTED_ARCHIVE_SHA256) == EXPECTED_ARCHIVE_SHA256
    assert (
        validate_source_archive_hash(PINNED_COMMIT_ARCHIVE_SHA256)
        == PINNED_COMMIT_ARCHIVE_SHA256
    )


def test_normalize_reference_values_uses_off_diagonal_maximum():
    values = np.array([[0.0, 2.0], [2.0, 0.0]])
    normalized, maximum = normalize_reference_values(values)
    assert maximum == 2.0
    assert normalized.tolist() == [[0.0, 1.0], [1.0, 0.0]]


def test_compare_reference_matrices_reports_36_relations_for_nine_labels():
    labels = [str(index) for index in range(9)]
    base = np.abs(np.subtract.outer(np.arange(9), np.arange(9))).astype(float)
    report = compare_reference_matrices(
        {"left": base, "right": base * 2.0}, labels
    )
    pair = report["pairs"][0]
    assert report["relation_count"] == 36
    assert pair["pearson"] == 1.0
    assert pair["spearman"] == 1.0
    assert pair["rank_order_reversals"] == 0
