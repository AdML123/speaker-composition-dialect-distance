import pytest

from src.estimand_sensitivity import (
    aggregate_seed_bootstraps,
    attach_design_strata,
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
        assert summary["gain_unit"] == "fraction"
        assert summary["gain_orientation"] == "(baseline_mae-method_mae)/baseline_mae"
        assert summary["resampling_unit"] == "global_endpoint_speaker"
        assert len(summary["bootstrap_gain"]) == 200


def test_pairing_rejects_endpoint_mismatch_even_when_pair_ids_match():
    baseline = [
        {
            "pair_id": "p1",
            "absolute_error": 0.5,
            "speaker_ids": ["a", "b"],
            "dialect_labels": ["B", "J"],
            "utterance_ids": ["u1", "u2"],
            "matched_stratum": "s1",
        }
    ]
    method = [dict(baseline[0], utterance_ids=["u1", "u3"], absolute_error=0.4)]
    with pytest.raises(ValueError, match="pair identity mismatch"):
        dyadic_weighting_bootstrap(baseline, method, seed=7, replicates=20)


def test_design_strata_join_uses_pair_and_endpoint_identity():
    rows = [
        {
            "pair_id": "p1",
            "utterance_ids": ["u2", "u1"],
            "speaker_ids": ["b", "a"],
            "dialect_labels": ["J", "B"],
            "absolute_error": 0.2,
        }
    ]
    manifest = [
        {
            "pair_id": "p1",
            "source_utterance_ids": ["u1", "u2"],
            "matched_stratum": "symmetric",
        }
    ]
    joined = attach_design_strata(rows, manifest)
    assert joined[0]["matched_stratum"] == "symmetric"
    with pytest.raises(ValueError, match="pair identity mismatch"):
        attach_design_strata(
            [dict(rows[0], utterance_ids=["u2", "other"])], manifest
        )


def test_seed_aggregation_uses_paired_replicate_medians():
    seed_results = [
        {
            "seed": 1,
            "estimands": {
                "pair": {
                    "gain": 0.1,
                    "baseline_mae": 0.5,
                    "method_mae": 0.45,
                    "bootstrap_gain": [0.0, 0.1, 0.2, 0.3],
                }
            },
        },
        {
            "seed": 2,
            "estimands": {
                "pair": {
                    "gain": 0.2,
                    "baseline_mae": 0.5,
                    "method_mae": 0.4,
                    "bootstrap_gain": [0.1, 0.2, 0.3, 0.4],
                }
            },
        },
    ]
    summary = aggregate_seed_bootstraps(seed_results, "pair")
    assert summary["gain"] == pytest.approx(0.15)
    assert summary["seed_values"] == [
        {"seed": 1, "gain": 0.1},
        {"seed": 2, "gain": 0.2},
    ]
    assert summary["ci"]["lower"] == pytest.approx(0.0575)
    assert summary["ci"]["upper"] == pytest.approx(0.3425)
    assert summary["gain_unit"] == "fraction"
