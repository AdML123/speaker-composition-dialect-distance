from __future__ import annotations

import numpy as np

import src.speaker_regression_gate as gate
from src.speaker_regression_gate import (
    _shuffled_centroids,
    _wrong_centroids,
    apply_speaker_regression,
    evaluate_ecapa_regression_gate,
    fit_speaker_regression,
    leave_pair_out_ecapa_centroids,
    speaker_centroids,
)


def test_speaker_centroids_are_deterministic():
    embeddings = {
        "u2": [3.0, 5.0],
        "u1": [1.0, 3.0],
        "u3": [10.0, 20.0],
    }
    metadata = {
        "u1": {"speaker_id": "s1", "dialect_label": "Mandarin"},
        "u2": {"speaker_id": "s1", "dialect_label": "Southwestern"},
        "u3": {"speaker_id": "s2", "dialect_label": "Mandarin"},
    }

    report = speaker_centroids(embeddings, metadata)

    assert report["speaker_ids"] == ["s1", "s2"]
    assert report["speaker_counts"] == {"s1": 2, "s2": 1}
    np.testing.assert_allclose(report["speaker_centroids"]["s1"], [2.0, 4.0])
    np.testing.assert_allclose(report["global_mean"], [14.0 / 3.0, 28.0 / 3.0])


def test_ridge_regression_recovers_known_speaker_component():
    ecapa = {
        "u1": [1.0, 0.0],
        "u2": [1.0, 0.1],
        "u3": [1.0, 2.0],
        "u4": [1.0, 2.1],
        "u5": [-1.0, 0.0],
        "u6": [-1.0, 0.1],
        "u7": [-1.0, 2.0],
        "u8": [-1.0, 2.1],
    }
    frozen = {
        "u1": [2.0, 1.0],
        "u2": [2.0, 1.1],
        "u3": [2.0, 3.0],
        "u4": [2.0, 3.1],
        "u5": [-2.0, 1.0],
        "u6": [-2.0, 1.1],
        "u7": [-2.0, 3.0],
        "u8": [-2.0, 3.1],
    }
    metadata = {
        "u1": {"speaker_id": "s1", "dialect_label": "Mandarin"},
        "u2": {"speaker_id": "s1", "dialect_label": "Mandarin"},
        "u3": {"speaker_id": "s1", "dialect_label": "Southwestern"},
        "u4": {"speaker_id": "s1", "dialect_label": "Southwestern"},
        "u5": {"speaker_id": "s2", "dialect_label": "Mandarin"},
        "u6": {"speaker_id": "s2", "dialect_label": "Mandarin"},
        "u7": {"speaker_id": "s2", "dialect_label": "Southwestern"},
        "u8": {"speaker_id": "s2", "dialect_label": "Southwestern"},
    }

    fitted = fit_speaker_regression(ecapa, frozen, metadata, alpha=0.1)
    cleaned = apply_speaker_regression(frozen, ecapa, metadata, fitted)

    assert cleaned["u1"] != frozen["u1"]
    assert fitted["speaker_centroid_r2"] > 0.8
    assert fitted["target_source"] == "cell_offset_minus_dialect_main_effect"


def test_leave_pair_out_reports_fallback_counts():
    ecapa = {"u1": [1.0, 0.0], "u2": [1.0, 0.1]}
    frozen = {"u1": [2.0, 1.0], "u2": [2.0, 1.1]}
    metadata = {
        "u1": {"speaker_id": "s1", "dialect_label": "d1"},
        "u2": {"speaker_id": "s1", "dialect_label": "d1"},
    }
    pairs = [{"pair_id": "A-1", "source_utterance_ids": ["u1", "u2"], "group": "A", "dialect_labels": ["d1"]}]

    report = leave_pair_out_ecapa_centroids(ecapa, frozen, metadata, pairs)

    assert report["fallback_endpoint_count"] == 2
    assert report["fallback_pair_count"] == 1


def test_shuffle_and_wrong_centroid_ablations_have_distinct_mapping_rules():
    centroids = {
        "s1": np.asarray([1.0]),
        "s2": np.asarray([2.0]),
        "s3": np.asarray([3.0]),
    }

    shuffled = _shuffled_centroids(centroids, seed=0)
    wrong = _wrong_centroids(centroids, seed=0)

    assert sorted(float(value[0]) for value in shuffled.values()) == [1.0, 2.0, 3.0]
    assert sorted(float(value[0]) for value in wrong.values()) == [1.0, 2.0, 3.0]
    assert any(np.array_equal(shuffled[speaker], centroids[speaker]) for speaker in centroids)
    assert all(not np.array_equal(wrong[speaker], centroids[speaker]) for speaker in centroids)


def test_dialect_bias_swap_is_deterministic_and_changes_assignments():
    bias = {"Mandarin": np.asarray([1.0]), "Southwestern": np.asarray([2.0]), "Ji-Lu": np.asarray([3.0])}
    from src.speaker_regression_gate import _swapped_dialect_bias

    swapped = _swapped_dialect_bias(bias)
    assert sorted(float(value[0]) for value in swapped.values()) == [1.0, 2.0, 3.0]
    assert all(not np.array_equal(swapped[dialect], bias[dialect]) for dialect in bias)


def test_ecapa_regression_gate_report_schema():
    ecapa = {
        "u1": [1.0, 0.0],
        "u2": [1.0, 0.1],
        "u3": [-1.0, 0.0],
        "u4": [-1.0, 0.1],
    }
    frozen = {
        "u1": [2.0, 1.0],
        "u2": [2.0, 1.1],
        "u3": [-2.0, 1.0],
        "u4": [-2.0, 1.1],
    }
    metadata = {
        "u1": {"speaker_id": "s1", "dialect_label": "d1"},
        "u2": {"speaker_id": "s1", "dialect_label": "d1"},
        "u3": {"speaker_id": "s2", "dialect_label": "d2"},
        "u4": {"speaker_id": "s2", "dialect_label": "d2"},
    }
    pairs = [
        {"pair_id": "A-1", "source_utterance_ids": ["u1", "u2"], "group": "A", "dialect_labels": ["d1"]},
        {"pair_id": "D-1", "source_utterance_ids": ["u1", "u3"], "group": "D", "dialect_labels": ["d1", "d2"]},
    ]

    report = evaluate_ecapa_regression_gate(
        calibration_frozen_embeddings={"m": frozen},
        calibration_ecapa_embeddings=ecapa,
        evaluation_frozen_embeddings={"m": frozen},
        evaluation_ecapa_embeddings=ecapa,
        calibration_metadata=metadata,
        evaluation_metadata=metadata,
        calibration_pairs=pairs,
        evaluation_pairs=pairs,
        references=[{"name": "taxonomy", "matrix": {"d1": {"d1": 0.0, "d2": 1.0}, "d2": {"d1": 1.0, "d2": 0.0}}}],
        bootstrap_replicates=1000,
        seed=20260829,
        global_regression_reference={
            "models": [
                {
                    "model_name": "m",
                    "references": [
                        {"reference_name": "taxonomy", "improvement_ratio": -0.01}
                    ],
                }
            ]
        },
    )

    assert report["schema"] == "ecapa-regression-gate-v1"
    assert report["evaluation_ecapa_scope"] == "full"
    assert report["decision"] in {"continue_to_review", "stop_before_manuscript_and_release"}
    assert report["method"]["target_source"] == "cell_offset_minus_dialect_main_effect"
    assert report["models"][0]["references"][0]["baseline_mae"] >= 0.0
    assert "leave_pair_out" in report["models"][0]["references"][0]
    assert "global_regression_comparison" in report["models"][0]["references"][0]
    assert "dialect_bias_swap" in report["models"][0]["references"][0]["ablation"]
    assert report["models"][0]["references"][0]["global_regression_comparison"]["global_improvement_ratio"] == -0.01


def test_rank1_dialect_modulation_recovers_known_speaker_dialect_component():
    ecapa = {
        "u1": [1.0, 0.0],
        "u2": [1.0, 0.1],
        "u3": [1.0, 2.0],
        "u4": [1.0, 2.1],
        "u5": [-1.0, 0.0],
        "u6": [-1.0, 0.1],
        "u7": [-1.0, 2.0],
        "u8": [-1.0, 2.1],
    }
    frozen = {
        "u1": [2.0, 1.0],
        "u2": [2.0, 1.1],
        "u3": [2.0, 3.0],
        "u4": [2.0, 3.1],
        "u5": [-2.0, 1.0],
        "u6": [-2.0, 1.1],
        "u7": [-2.0, 3.0],
        "u8": [-2.0, 3.1],
    }
    metadata = {
        "u1": {"speaker_id": "s1", "dialect_label": "Mandarin"},
        "u2": {"speaker_id": "s1", "dialect_label": "Mandarin"},
        "u3": {"speaker_id": "s1", "dialect_label": "Southwestern"},
        "u4": {"speaker_id": "s1", "dialect_label": "Southwestern"},
        "u5": {"speaker_id": "s2", "dialect_label": "Mandarin"},
        "u6": {"speaker_id": "s2", "dialect_label": "Mandarin"},
        "u7": {"speaker_id": "s2", "dialect_label": "Southwestern"},
        "u8": {"speaker_id": "s2", "dialect_label": "Southwestern"},
    }

    fitted = gate.fit_rank1_dialect_modulation(ecapa, frozen, metadata, alpha=1.0, rank=1)
    cleaned = gate.apply_rank1_dialect_modulation(frozen, ecapa, metadata, fitted)

    assert fitted["rank"] == 1
    assert fitted["parameterization"] == "shared_ridge_plus_rank1_dialect_modulation"
    assert fitted["target_source"] == "cell_offset_minus_dialect_main_effect"
    assert fitted["target_r2"] > 0.8
    assert cleaned["u1"] != frozen["u1"]


def test_rank1_dialect_modulation_ablation_changes_gain():
    ecapa = {
        "u1": [1.0, 0.0],
        "u2": [1.0, 0.1],
        "u3": [1.0, 2.0],
        "u4": [1.0, 2.1],
        "u5": [-1.0, 0.0],
        "u6": [-1.0, 0.1],
        "u7": [-1.0, 2.0],
        "u8": [-1.0, 2.1],
    }
    frozen = {
        "u1": [2.0, 1.0],
        "u2": [2.0, 1.1],
        "u3": [2.0, 3.0],
        "u4": [2.0, 3.1],
        "u5": [-2.0, 1.0],
        "u6": [-2.0, 1.1],
        "u7": [-2.0, 3.0],
        "u8": [-2.0, 3.1],
    }
    metadata = {
        "u1": {"speaker_id": "s1", "dialect_label": "Mandarin"},
        "u2": {"speaker_id": "s1", "dialect_label": "Mandarin"},
        "u3": {"speaker_id": "s1", "dialect_label": "Southwestern"},
        "u4": {"speaker_id": "s1", "dialect_label": "Southwestern"},
        "u5": {"speaker_id": "s2", "dialect_label": "Mandarin"},
        "u6": {"speaker_id": "s2", "dialect_label": "Mandarin"},
        "u7": {"speaker_id": "s2", "dialect_label": "Southwestern"},
        "u8": {"speaker_id": "s2", "dialect_label": "Southwestern"},
    }
    pairs = [
        {"pair_id": "A-1", "source_utterance_ids": ["u1", "u2"], "group": "A", "dialect_labels": ["Mandarin"]},
        {"pair_id": "D-1", "source_utterance_ids": ["u1", "u3"], "group": "D", "dialect_labels": ["Mandarin", "Southwestern"]},
    ]
    references = [{"name": "taxonomy", "matrix": {"Mandarin": {"Mandarin": 0.0, "Southwestern": 1.0}, "Southwestern": {"Mandarin": 1.0, "Southwestern": 0.0}}}]

    full = gate.evaluate_rank1_dialect_perturbation_gate(
        calibration_frozen_embeddings={"m": frozen},
        calibration_ecapa_embeddings=ecapa,
        evaluation_frozen_embeddings={"m": frozen},
        evaluation_ecapa_embeddings=ecapa,
        calibration_metadata=metadata,
        evaluation_metadata=metadata,
        calibration_pairs=pairs,
        evaluation_pairs=pairs,
        references=references,
        bootstrap_replicates=1000,
        seed=20260829,
    )
    swapped = full["models"][0]["references"][0]["ablation"]["v_d_swap"]["max_improvement_ratio"]
    disabled = full["models"][0]["references"][0]["ablation"]["interaction_disabled"]["max_improvement_ratio"]
    baseline = full["models"][0]["references"][0]["improvement_ratio"]

    assert full["schema"] == "low-rank-dialect-perturbation-r1-gate-v1"
    assert np.isfinite(baseline)
    assert swapped <= baseline
    assert disabled <= baseline


def test_block_regularized_rank1_recovers_known_speaker_dialect_component():
    ecapa = {
        "u1": [1.0, 0.0],
        "u2": [1.0, 0.1],
        "u3": [1.0, 2.0],
        "u4": [1.0, 2.1],
        "u5": [-1.0, 0.0],
        "u6": [-1.0, 0.1],
        "u7": [-1.0, 2.0],
        "u8": [-1.0, 2.1],
    }
    frozen = {
        "u1": [2.0, 1.0],
        "u2": [2.0, 1.1],
        "u3": [2.0, 3.0],
        "u4": [2.0, 3.1],
        "u5": [-2.0, 1.0],
        "u6": [-2.0, 1.1],
        "u7": [-2.0, 3.0],
        "u8": [-2.0, 3.1],
    }
    metadata = {
        "u1": {"speaker_id": "s1", "dialect_label": "Mandarin"},
        "u2": {"speaker_id": "s1", "dialect_label": "Mandarin"},
        "u3": {"speaker_id": "s1", "dialect_label": "Southwestern"},
        "u4": {"speaker_id": "s1", "dialect_label": "Southwestern"},
        "u5": {"speaker_id": "s2", "dialect_label": "Mandarin"},
        "u6": {"speaker_id": "s2", "dialect_label": "Mandarin"},
        "u7": {"speaker_id": "s2", "dialect_label": "Southwestern"},
        "u8": {"speaker_id": "s2", "dialect_label": "Southwestern"},
    }

    fitted = gate.fit_block_regularized_rank1_dialect_modulation(
        ecapa,
        frozen,
        metadata,
        base_alpha=1.0,
        rank=1,
        w_penalty_multiplier=100.0,
        bias_penalty_multiplier=10.0,
        low_rank_penalty_multiplier=1.0,
    )
    cleaned = gate.apply_block_regularized_rank1_dialect_modulation(frozen, ecapa, metadata, fitted)

    assert fitted["rank"] == 1
    assert fitted["parameterization"] == "block_regularized_shared_ridge_plus_rank1_dialect_modulation"
    assert fitted["target_source"] == "cell_offset_minus_dialect_main_effect"
    assert fitted["target_r2"] > 0.8
    assert cleaned["u1"] != frozen["u1"]


def test_block_regularized_gate_report_schema():
    ecapa = {
        "u1": [1.0, 0.0],
        "u2": [1.0, 0.1],
        "u3": [-1.0, 0.0],
        "u4": [-1.0, 0.1],
    }
    frozen = {
        "u1": [2.0, 1.0],
        "u2": [2.0, 1.1],
        "u3": [-2.0, 1.0],
        "u4": [-2.0, 1.1],
    }
    metadata = {
        "u1": {"speaker_id": "s1", "dialect_label": "d1"},
        "u2": {"speaker_id": "s1", "dialect_label": "d1"},
        "u3": {"speaker_id": "s2", "dialect_label": "d2"},
        "u4": {"speaker_id": "s2", "dialect_label": "d2"},
    }
    pairs = [
        {"pair_id": "A-1", "source_utterance_ids": ["u1", "u2"], "group": "A", "dialect_labels": ["d1"]}
    ]

    report = gate.evaluate_block_regularized_low_rank_gate(
        calibration_frozen_embeddings={"m": frozen},
        calibration_ecapa_embeddings=ecapa,
        evaluation_frozen_embeddings={"m": frozen},
        evaluation_ecapa_embeddings=ecapa,
        calibration_metadata=metadata,
        evaluation_metadata=metadata,
        calibration_pairs=pairs,
        evaluation_pairs=pairs,
        references=[{"name": "taxonomy", "matrix": {"d1": {"d1": 0.0}}}],
        bootstrap_replicates=1000,
        seed=20260829,
    )

    assert report["schema"] == "block-regularized-low-rank-dialect-perturbation-r1-gate-v1"
    assert report["decision"] in {"continue_to_review", "stop_before_manuscript_and_release"}
    assert report["models"][0]["references"][0]["baseline_mae"] >= 0.0
    assert "previous_branch_comparison" in report["models"][0]["references"][0]
    assert "global_regression_comparison" in report["models"][0]["references"][0]
    assert "uniform_control_comparison" in report["models"][0]["references"][0]
