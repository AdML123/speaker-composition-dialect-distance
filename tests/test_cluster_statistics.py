from __future__ import annotations

import pytest

from src.cluster_statistics import (
    ClusterStatisticsError,
    clustered_ab_effect,
    clustered_bootstrap,
    clustered_sign_flip_test,
)


def _row(
    pair_id: str,
    group: str,
    dialect: str,
    distance: float,
    speakers: list[str],
    utterances: list[str],
    stratum: str = "d1|c1|clean",
) -> dict[str, object]:
    return {
        "pair_id": pair_id,
        "group": group,
        "dialect": dialect,
        "distance": distance,
        "speaker_ids": speakers,
        "utterance_ids": utterances,
        "matched_stratum": stratum,
    }


def _fixture() -> list[dict[str, object]]:
    return [
        _row("a1", "A", "d1", 0.10, ["s1"], ["u1", "u2"]),
        _row("a2", "A", "d1", 0.12, ["s1"], ["u3", "u4"]),
        _row("a3", "A", "d1", 0.11, ["s2"], ["u5", "u6"]),
        _row("b1", "B", "d1", 0.40, ["s1", "s3"], ["u7", "u8"]),
        _row("b2", "B", "d1", 0.42, ["s1", "s3"], ["u9", "u10"]),
        _row("b3", "B", "d1", 0.39, ["s2", "s4"], ["u11", "u12"]),
    ]


def test_clustered_effect_reports_b_minus_a_and_cluster_counts():
    report = clustered_ab_effect(_fixture())

    assert report["effect"] == pytest.approx(0.29)
    assert report["matched_stratum_count"] == 1
    assert report["speaker_cluster_count"] == 4
    assert report["utterance_count"] == 12
    assert report["resampling_unit"] == "speaker_cluster_within_matched_stratum"


def test_clustered_bootstrap_is_reproducible_and_nested():
    first = clustered_bootstrap(_fixture(), seed=7, replicates=1000)
    second = clustered_bootstrap(_fixture(), seed=7, replicates=1000)

    assert first == second
    assert first["ci"]["lower"] < first["effect"] < first["ci"]["upper"]
    assert first["resampling"]["utterance_within_speaker"] is True


def test_clustered_sign_flip_reports_a_permutation_null_and_p_value():
    report = clustered_sign_flip_test(_fixture(), seed=7, permutations=1000)

    assert report["null_type"] == "cluster_sign_flip"
    assert report["exchangeability_unit"] == "speaker_cluster_within_matched_stratum"
    assert report["permutations"] == 1000
    assert 0.0 <= report["raw_p"] <= 1.0
    assert report["observed_effect"] > 0.0


def test_clustered_inference_rejects_pair_only_rows():
    rows = [{"pair_id": "p1", "group": "A", "dialect": "d1", "distance": 0.1}]

    with pytest.raises(ClusterStatisticsError, match="speaker_ids"):
        clustered_bootstrap(rows, seed=1, replicates=1000)


def test_clustered_inference_rejects_too_few_resamples():
    with pytest.raises(ClusterStatisticsError, match="1000"):
        clustered_bootstrap(_fixture(), seed=1, replicates=999)
