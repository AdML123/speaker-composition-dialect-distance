import pytest

from src.calibration_leakage_audit import audit_projection_sources


def pair(pair_id, speaker, left="u1", right="u2"):
    return {
        "pair_id": pair_id,
        "speaker_ids": [speaker, speaker],
        "source_utterance_ids": [left, right],
    }


def test_audit_accepts_disjoint_calibration_sources():
    report = audit_projection_sources(
        calibration_pairs=[pair("c1", "cal", "c-u1", "c-u2")],
        cross_examples=[pair("x1", "cal", "c-u3", "c-u4")],
        evaluation_pairs=[pair("e1", "eval", "e-u1", "e-u2")],
        fitted_sources={"projection_calibration": [pair("c1", "cal", "c-u1", "c-u2")]},
    )
    assert report["status"] == "passed"
    assert report["evaluation_rows_used_for_fitting"] is False
    assert all(row["intersections"]["pair_ids"] == [] for row in report["sources"])


def test_local_pair_number_reuse_is_not_identity_collision():
    report = audit_projection_sources(
        calibration_pairs=[pair("A-000001", "cal", "c-u1", "c-u2")],
        cross_examples=[],
        evaluation_pairs=[pair("A-000001", "eval", "e-u1", "e-u2")],
        fitted_sources={"projection_calibration": [pair("A-000001", "cal", "c-u1", "c-u2")]},
    )
    row = report["sources"][0]
    assert row["intersections"]["pair_ids"] == ["A-000001"]
    assert row["intersections"]["pair_identities"] == []
    assert row["pair_id_namespace_overlap_only"] is True


@pytest.mark.parametrize("collision", ["pair_id", "speaker_id", "utterance_id"])
def test_audit_rejects_evaluation_collision(collision):
    evaluation = pair("e1", "eval", "e-u1", "e-u2")
    source = pair("c1", "cal", "c-u1", "c-u2")
    if collision == "pair_id":
        source["pair_id"] = "e1"
        source["source_utterance_ids"] = ["e-u1", "e-u2"]
    elif collision == "speaker_id":
        source["speaker_ids"] = ["eval", "eval"]
    else:
        source["source_utterance_ids"] = ["e-u1", "c-u2"]
    with pytest.raises(ValueError, match="evaluation collision"):
        audit_projection_sources(
            calibration_pairs=[source],
            cross_examples=[],
            evaluation_pairs=[evaluation],
            fitted_sources={"projection_calibration": [source]},
        )


def test_audit_requires_explicit_fit_role_and_hash_fields():
    report = audit_projection_sources(
        calibration_pairs=[pair("c1", "cal")],
        cross_examples=[pair("x1", "cal", "u3", "u4")],
        evaluation_pairs=[pair("e1", "eval", "e-u1", "e-u2")],
        fitted_sources={
            "projection_calibration": [pair("c1", "cal")],
            "calibration_auxiliary_cross_pair": [pair("x1", "cal", "u3", "u4")],
        },
    )
    assert {row["role"] for row in report["sources"]} == {
        "projection_calibration",
        "calibration_auxiliary_cross_pair",
    }
    assert report["evaluation_manifest_role"] == "projection_evaluation"
