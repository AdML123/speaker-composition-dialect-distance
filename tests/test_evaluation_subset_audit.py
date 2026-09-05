import pytest

from src.evaluation_subset_audit import (
    audit_annotated_projection_manifest,
    audit_projection_manifest,
    build_symmetric_projection_annotation,
    detect_pair_id_collisions,
    downstream_manifest_status,
)


def _pair(pair_id, group, utterances, speakers, dialects, conditions):
    return {
        "pair_id": pair_id,
        "group": group,
        "split": "evaluation",
        "source_utterance_ids": utterances,
        "speaker_ids": speakers,
        "dialect_labels": dialects,
        "recording_conditions": conditions,
    }


def _fixture():
    evaluation = {
        "schema": "pair-sampling-v1",
        "seed": 20260829,
        "pairs": [
            _pair("A-1", "A", ["u1", "u2"], ["e1"], ["d1"], ["c1", "c1"]),
            _pair("B-1", "B", ["u1", "u3"], ["e1", "e2"], ["d1"], ["c1", "c1"]),
            _pair("C-1", "C", ["u4", "u5"], ["e3"], ["d1", "d2"], ["c1", "c2"]),
            _pair("D-1", "D", ["u6", "u7"], ["e4", "e5"], ["d1", "d2"], ["c1", "c2"]),
        ],
    }
    calibration = {
        "schema": "pair-sampling-v1",
        "seed": 20260829,
        "pairs": [
            {
                **_pair("A-1", "A", ["c1", "c2"], ["cal"], ["d1"], ["c1", "c1"]),
                "split": "calibration",
            }
        ],
    }
    phenomenon = {
        "schema": "matched-pair-audit-v1",
        "pairs": [
            {
                "pair_id": "A-1",
                "group": "A",
                "utterance_ids": ["other-1", "other-2"],
            }
        ],
    }
    return evaluation, calibration, phenomenon


def test_projection_manifest_reports_locked_design_and_cross_group_reuse():
    evaluation, calibration, phenomenon = _fixture()
    report = audit_projection_manifest(
        evaluation,
        calibration,
        phenomenon,
        expected_group_counts={"A": 1, "B": 1, "C": 1, "D": 1},
        expected_seed=20260829,
        projection_sha256="projection-hash",
        phenomenon_sha256="phenomenon-hash",
    )
    assert report["distinct_from_phenomenon_manifest"] is True
    assert report["group_pair_counts"] == {"A": 1, "B": 1, "C": 1, "D": 1}
    assert report["within_group_utterance_reuse"] == {"A": 0, "B": 0, "C": 0, "D": 0}
    assert report["cross_group_utterance_reuse_count"] == 1
    assert report["evaluation_calibration_speaker_overlap"] == []
    assert report["sampling"]["seed"] == 20260829
    assert report["sampling"]["implementation"] == "src.pair_sampling.sample_pairs"
    assert report["groups"]["D"]["dialect_relation_count"] == 1
    assert report["groups"]["C"]["metadata_condition_relation_count"] == 1


def test_projection_manifest_rejects_speaker_leakage_and_within_group_reuse():
    evaluation, calibration, phenomenon = _fixture()
    calibration["pairs"][0]["speaker_ids"] = ["e1"]
    with pytest.raises(ValueError, match="speaker-disjoint"):
        audit_projection_manifest(
            evaluation,
            calibration,
            phenomenon,
            expected_group_counts={"A": 1, "B": 1, "C": 1, "D": 1},
            expected_seed=20260829,
            projection_sha256="projection-hash",
            phenomenon_sha256="phenomenon-hash",
        )


def test_pair_id_collisions_compare_utterance_identity_not_local_id_only():
    evaluation, _, phenomenon = _fixture()
    report = detect_pair_id_collisions(evaluation["pairs"], phenomenon["pairs"])
    assert report["shared_pair_id_count"] == 1
    assert report["different_utterance_identity_count"] == 1
    assert report["safe_to_join_on_pair_id_alone"] is False


def test_downstream_manifest_status_invalidates_unproven_stratum_annotation():
    expected = "a" * 64
    valid = {"source_hashes": {"evaluation_pairs": expected}}
    current_valid = {"source_hashes": {"projection_manifest": expected}}
    wrong_hash = {"source_hashes": {"evaluation_pairs": "b" * 64}}
    unproven_weighting = {
        "bootstrap": {"resampling_unit": "endpoint_speaker_within_matched_stratum"},
        "runs": [{"counts": {"pairs": 4, "strata": 2}}],
    }
    assert downstream_manifest_status(valid, expected)["status"] == "verified"
    assert downstream_manifest_status(current_valid, expected)["status"] == "verified"
    assert downstream_manifest_status(wrong_hash, expected)["status"] == "invalidated"
    report = downstream_manifest_status(unproven_weighting, expected)
    assert report["status"] == "invalidated"
    assert report["reason"] == "matched_stratum_provenance_missing"


def test_annotated_manifest_requires_exact_identity_and_symmetric_relation_strata():
    evaluation, _, _ = _fixture()
    annotated = {"pairs": []}
    for row in evaluation["pairs"]:
        annotated["pairs"].append(
            {
                **row,
                "utterance_ids": list(row["source_utterance_ids"]),
                "matched_stratum": "|".join(
                    [
                        row["group"],
                        "::".join(sorted(row["dialect_labels"])),
                        "::".join(sorted(row["recording_conditions"])),
                    ]
                ),
            }
        )
    valid = audit_annotated_projection_manifest(evaluation, annotated)
    assert valid["base_pair_identity_match"] is True
    assert valid["symmetric_relation_strata"] is True

    annotated["pairs"][2]["matched_stratum"] = "C|d1|c1"
    invalid = audit_annotated_projection_manifest(evaluation, annotated)
    assert invalid["base_pair_identity_match"] is True
    assert invalid["symmetric_relation_strata"] is False
    assert invalid["invalid_stratum_pair_ids"] == ["C-1"]


def test_symmetric_projection_annotation_is_endpoint_order_invariant():
    evaluation, _, _ = _fixture()
    built = build_symmetric_projection_annotation(
        evaluation, base_manifest_sha256="a" * 64
    )
    c_row = next(row for row in built["pairs"] if row["pair_id"] == "C-1")
    assert c_row["matched_stratum"] == "C|d1::d2|c1::c2"
    assert c_row["matched_fields"] == [
        "group",
        "unordered_dialect_relation",
        "unordered_metadata_condition_relation",
    ]
    assert built["base_manifest_sha256"] == "a" * 64
    assert audit_annotated_projection_manifest(evaluation, built)[
        "symmetric_relation_strata"
    ] is True
