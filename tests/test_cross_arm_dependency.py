import pytest

from src.cross_arm_dependency import (
    audit_cross_arm_dependencies,
    utterance_block_bootstrap,
)
from src.dyadic_bootstrap import global_utterance_multiplicities


def _record(utterance, speaker, dialect="d"):
    return {
        "utterance_id": utterance,
        "speaker_id": speaker,
        "dialect_label": dialect,
    }


def _pair(pair_id, group, utterances, speakers, stratum="s"):
    return {
        "pair_id": pair_id,
        "group": group,
        "utterance_ids": utterances,
        "speaker_ids": speakers,
        "dialect_labels": ["d"],
        "matched_stratum": stratum,
    }


def _distance(group, distance, utterances, speakers, stratum):
    return {
        "group": group,
        "distance": distance,
        "utterance_ids": utterances,
        "speaker_ids": speakers,
        "matched_stratum": stratum,
    }


def test_dependency_audit_tracks_shared_identity_and_nesting():
    records = [
        _record("u1", "sp1"),
        _record("u2", "sp1"),
        _record("u3", "sp2"),
        _record("u4", "sp3"),
    ]
    pairs = [
        _pair("A-1", "A", ["u1", "u2"], ["sp1"], "s1"),
        _pair("B-1", "B", ["u1", "u3"], ["sp1", "sp2"], "s1"),
        _pair("B-2", "B", ["u2", "u4"], ["sp1", "sp3"], "s2"),
    ]
    report = audit_cross_arm_dependencies(pairs, records, expected_shared=2)
    assert report["cross_arm_shared_utterance_count"] == 2
    assert report["shared_utterances_uniquely_nested_in_speaker"] is True
    assert report["shared_utterances_uniquely_nested_in_dialect"] is True
    assert report["within_arm_utterance_reuse"] == {"A": 0, "B": 0}
    assert report["speakers_in_multiple_arms"] == ["sp1"]
    assert report["speakers_in_multiple_strata"] == ["sp1"]
    assert report["cluster_identity"]["speaker"] == "global speaker_id"
    assert report["cluster_identity"]["utterance"] == "global utterance_id"


def test_dependency_audit_fails_closed_on_shared_count_drift():
    records = [_record("u1", "sp1"), _record("u2", "sp1"), _record("u3", "sp2")]
    pairs = [
        _pair("A-1", "A", ["u1", "u2"], ["sp1"]),
        _pair("B-1", "B", ["u1", "u3"], ["sp1", "sp2"]),
    ]
    with pytest.raises(ValueError, match="shared utterance count drift"):
        audit_cross_arm_dependencies(pairs, records, expected_shared=6244)


def test_one_utterance_multiplier_is_reused_across_arms():
    rows = [
        _distance("A", 0.0, ["shared", "a"], ["sp", "a-sp"], "s"),
        _distance("B", 1.0, ["shared", "b"], ["sp", "b-sp"], "s"),
    ]
    counts = global_utterance_multiplicities(rows, seed=19)
    assert counts["shared"] == counts["shared"]
    assert sum(counts.values()) == 3


def test_cross_arm_blocking_recovers_wider_correlated_uncertainty():
    rows = [
        _distance("A", 0.0, ["shared", "a1"], ["sp", "a1-sp"], "s1"),
        _distance("A", 10.0, ["a2", "a3"], ["a2-sp", "a3-sp"], "s1"),
        _distance("B", 20.0, ["b1", "b2"], ["b1-sp", "b2-sp"], "s1"),
        _distance("B", 20.0, ["b3", "b4"], ["b3-sp", "b4-sp"], "s1"),
        _distance("A", 0.0, ["c1", "c2"], ["c1-sp", "c2-sp"], "s2"),
        _distance("A", 0.0, ["c3", "c4"], ["c3-sp", "c4-sp"], "s2"),
        _distance("B", 20.0, ["shared", "d1"], ["sp", "d1-sp"], "s2"),
        _distance("B", 10.0, ["d2", "d3"], ["d2-sp", "d3-sp"], "s2"),
    ]
    blocked = utterance_block_bootstrap(
        rows, seed=23, replicates=4000, share_cross_arm_clusters=True
    )
    independent = utterance_block_bootstrap(
        rows, seed=23, replicates=4000, share_cross_arm_clusters=False
    )
    assert blocked["cross_arm_cluster_identity"] == "global utterance_id"
    assert blocked["bootstrap_estimate_standard_deviation"] > independent[
        "bootstrap_estimate_standard_deviation"
    ]
