import pytest

from src.paired_randomness import clustered_paired_bootstrap, make_paired_schedule


def test_schedule_is_identical_for_shared_seed_and_changes_with_seed():
    first = make_paired_schedule(seed=7, pair_count=10, cross_count=8, epochs=3, batch_size=4)
    second = make_paired_schedule(seed=7, pair_count=10, cross_count=8, epochs=3, batch_size=4)
    third = make_paired_schedule(seed=8, pair_count=10, cross_count=8, epochs=3, batch_size=4)
    assert first.to_dict() == second.to_dict()
    assert first.schedule_id != third.schedule_id
    assert first.initialization_seed == first.batch_seed == first.fold_seed == 7


def test_clustered_paired_bootstrap_reports_cluster_unit():
    rows = [
        {"delta": 0.2, "matched_stratum": "d1", "speaker_ids": ["s1", "s2"]},
        {"delta": 0.3, "matched_stratum": "d1", "speaker_ids": ["s1", "s3"]},
        {"delta": 0.4, "matched_stratum": "d2", "speaker_ids": ["s4", "s5"]},
    ]
    report = clustered_paired_bootstrap(rows, seed=7, replicates=1000)
    assert report["resampling_unit"] == "speaker_cluster_within_matched_stratum"
    assert report["nested_utterance_sampling"] is True
    assert 0.0 < report["bootstrap_tail_p_nonpositive"] <= 1.0
    assert report["ci"]["lower"] - 1e-12 <= report["observed_delta"] <= report["ci"]["upper"] + 1e-12


def test_clustered_bootstrap_uses_same_cluster_estimand_for_point_and_interval():
    rows = [
        {"matched_stratum": "A", "speaker_ids": ["large"], "delta": 1.0},
        {"matched_stratum": "A", "speaker_ids": ["large"], "delta": 1.0},
        {"matched_stratum": "A", "speaker_ids": ["large"], "delta": 1.0},
        {"matched_stratum": "A", "speaker_ids": ["small"], "delta": -1.0},
    ]
    report = clustered_paired_bootstrap(rows, seed=3, replicates=1000)
    assert report["observed_delta"] == pytest.approx(0.0)
    assert report["estimand"] == "equal speaker-cluster means within stratum, then equal stratum means"
