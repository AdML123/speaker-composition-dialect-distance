import pytest

from src.frozen_correction_scope import (
    build_scope_contract,
    strict_inductive_row,
    validate_scope_contract,
)


def test_scope_contract_classifies_current_pair_and_evaluation_pool():
    current = build_scope_contract(
        fit_scope="calibration_speakers",
        evaluation_feature_scope="current_pair_only",
        fallback_count=0,
    )
    pooled = build_scope_contract(
        fit_scope="calibration_speakers",
        evaluation_feature_scope="evaluation_speaker_pool",
        fallback_count=0,
    )
    leave_pair_out = build_scope_contract(
        fit_scope="calibration_speakers",
        evaluation_feature_scope="leave_pair_out_pool",
        fallback_count=3,
    )
    assert current["inference_class"] == "inductive"
    assert pooled["inference_class"] == "label_free_transductive"
    assert leave_pair_out["inference_class"] == "leave_pair_out_transductive"
    assert leave_pair_out["fallback_count"] == 3


def test_scope_contract_rejects_calibration_only_label_for_evaluation_pool():
    row = {
        "fit_scope": "calibration_speakers",
        "evaluation_feature_scope": "evaluation_speaker_pool",
        "inference_class": "inductive",
        "fallback_count": 0,
    }
    with pytest.raises(ValueError, match="transductive"):
        validate_scope_contract(row)


def test_strict_inductive_without_enrollment_is_not_applicable():
    row = strict_inductive_row(
        method="speaker_mean_normalization",
        enrollment_available=False,
    )
    assert row["status"] == "not_applicable"
    assert row["improvement_ratio"] is None
    assert row["evaluation_feature_scope"] == "current_pair_only"
    assert row["inference_class"] == "inductive"


def test_strict_inductive_cannot_recycle_a_transductive_score():
    with pytest.raises(ValueError, match="cannot carry a score"):
        strict_inductive_row(
            method="ecapa_regression",
            enrollment_available=False,
            improvement_ratio=0.02,
        )
