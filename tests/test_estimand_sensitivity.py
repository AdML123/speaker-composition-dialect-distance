import pytest

from src.estimand_sensitivity import (
    dyadic_weighting_bootstrap,
    relation_key,
    weighted_mae,
)


ROWS = [
    {
        "absolute_error": 0.0,
        "speaker_ids": ["a"],
        "dialect_labels": ["B"],
        "matched_stratum": "s1",
    },
    {
        "absolute_error": 1.0,
        "speaker_ids": ["a", "b"],
        "dialect_labels": ["B", "J"],
        "matched_stratum": "s1",
    },
    {
        "absolute_error": 1.0,
        "speaker_ids": ["c", "d"],
        "dialect_labels": ["J", "Z"],
        "matched_stratum": "s2",
    },
]


def test_pair_weighting_is_the_row_mean():
    assert weighted_mae(ROWS, "pair") == pytest.approx(2 / 3)


def test_same_speaker_pair_is_counted_once_in_endpoint_weighting():
    assert weighted_mae(ROWS[:1], "endpoint_speaker") == 0.0


def test_endpoint_weighting_averages_incident_errors_per_speaker():
    assert weighted_mae(ROWS, "endpoint_speaker") == pytest.approx(0.875)


def test_relation_key_is_unordered():
    reversed_row = dict(ROWS[1], dialect_labels=["J", "B"])
    assert relation_key(ROWS[1]["dialect_labels"]) == ("B", "J")
    assert weighted_mae([ROWS[1], reversed_row], "dialect_relation") == 1.0


def test_matched_stratum_weighting_averages_strata_equally():
    assert weighted_mae(ROWS, "matched_stratum") == pytest.approx(0.75)


def test_unknown_weighting_is_rejected():
    with pytest.raises(ValueError, match="unknown weighting"):
        weighted_mae(ROWS, "unknown")


def test_stratum_weighting_rejects_rows_without_defined_strata():
    rows = [dict(ROWS[0], matched_stratum=None)]
    with pytest.raises(ValueError, match="defined matched_stratum"):
        weighted_mae(rows, "matched_stratum")


def test_dyadic_weighting_bootstrap_reports_all_estimands():
    baseline = [
        {
            "pair_id": "p1",
            "absolute_error": 0.5,
            "speaker_ids": ["a"],
            "dialect_labels": ["B"],
            "matched_stratum": "s1",
        },
        {
            "pair_id": "p2",
            "absolute_error": 0.7,
            "speaker_ids": ["b", "c"],
            "dialect_labels": ["B", "J"],
            "matched_stratum": "s1",
        },
        {
            "pair_id": "p3",
            "absolute_error": 0.6,
            "speaker_ids": ["d", "e"],
            "dialect_labels": ["J", "Z"],
            "matched_stratum": "s2",
        },
    ]
    method = [dict(row, absolute_error=row["absolute_error"] - 0.1) for row in baseline]
    result = dyadic_weighting_bootstrap(
        baseline, method, seed=7, replicates=200
    )
    assert set(result) == set(("pair", "endpoint_speaker", "dialect_relation", "matched_stratum"))
    for summary in result.values():
        assert summary["gain"] > 0
        assert summary["ci"]["lower"] > 0
        assert summary["bootstrap_replicates"] == 200
