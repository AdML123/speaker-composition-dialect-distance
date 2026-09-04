import numpy as np
import pytest

from src.pca_projection_gate import (
    apply_projection,
    evaluate_pca_projection_gate,
    fit_projection_model,
    select_k_by_calibration,
)


def _metadata(speaker, dialect="d1", condition="clean"):
    return {"speaker_id": speaker, "dialect_label": dialect, "recording_condition": condition}


def _row(model, pair_id, group, labels, distance):
    return {
        "model_name": model,
        "pair_id": pair_id,
        "group": group,
        "dialect_labels": labels,
        "distance": distance,
    }


def _reference():
    return {
        "name": "synthetic",
        "matrix": {
            "d1": {"d1": 0.0, "d2": 1.0},
            "d2": {"d1": 1.0, "d2": 0.0},
        },
    }


def test_projection_k_zero_returns_original_embeddings():
    embeddings = {"u1": [1.0, 0.0], "u2": [3.0, 0.0]}
    metadata = {"u1": _metadata("s1"), "u2": _metadata("s2")}

    model = fit_projection_model(embeddings, metadata, max_components=1)
    cleaned = apply_projection(embeddings, model, k=0)

    assert cleaned == {"u1": [1.0, 0.0], "u2": [3.0, 0.0]}


def test_projection_removes_ranked_speaker_component():
    embeddings = {
        "u1": [1.0, 0.0],
        "u2": [1.0, 0.0],
        "u3": [-1.0, 0.0],
        "u4": [-1.0, 0.0],
    }
    metadata = {
        "u1": _metadata("s1"),
        "u2": _metadata("s1"),
        "u3": _metadata("s2"),
        "u4": _metadata("s2"),
    }

    model = fit_projection_model(embeddings, metadata, max_components=1)
    cleaned = apply_projection(embeddings, model, k=1)

    assert all(abs(vector[0]) < 1e-10 for vector in cleaned.values())


def test_speaker_ranking_residualizes_dialect_before_anova():
    embeddings = {
        "u1": [10.0, 1.0],
        "u2": [10.0, 1.2],
        "u3": [10.0, -1.0],
        "u4": [10.0, -1.2],
        "u5": [-10.0, 1.0],
        "u6": [-10.0, 1.2],
        "u7": [-10.0, -1.0],
        "u8": [-10.0, -1.2],
    }
    metadata = {
        "u1": _metadata("s1", "d1"),
        "u2": _metadata("s1", "d1"),
        "u3": _metadata("s2", "d1"),
        "u4": _metadata("s2", "d1"),
        "u5": _metadata("s3", "d2"),
        "u6": _metadata("s3", "d2"),
        "u7": _metadata("s4", "d2"),
        "u8": _metadata("s4", "d2"),
    }

    model = fit_projection_model(embeddings, metadata, max_components=2)

    assert model["ranked_components"][0] == 1


def test_select_k_by_calibration_uses_smallest_k_on_tie():
    result = select_k_by_calibration(
        candidates={
            0: [{"target_distance": 0.0, "distance": 0.1}],
            1: [{"target_distance": 0.0, "distance": 0.1}],
        },
        k_grid=[0, 1],
    )

    assert result["selected_k"] == 0


def test_pca_projection_gate_improves_synthetic_speaker_axis():
    def build_split(prefix):
        embeddings = {}
        metadata = {}
        pairs = []
        dialect_vectors = {"d1": [0.0, 1.0, 0.0], "d2": [0.0, 0.0, 1.0]}
        speaker_values = {"s1": 10.0, "s2": -10.0, "s3": 8.0, "s4": -8.0}
        for dialect, base in dialect_vectors.items():
            for speaker, value in speaker_values.items():
                utterance_id = f"{prefix}-{dialect}-{speaker}"
                embeddings[utterance_id] = [value, base[1], base[2]]
                metadata[utterance_id] = _metadata(speaker, dialect)
        for index, left_speaker in enumerate(["s1", "s3"]):
            right_speaker = "s2" if left_speaker == "s1" else "s4"
            pairs.append(
                {
                    "pair_id": f"{prefix}-a-{index}",
                    "group": "A",
                    "dialect_labels": ["d1"],
                    "source_utterance_ids": [f"{prefix}-d1-{left_speaker}", f"{prefix}-d1-{right_speaker}"],
                }
            )
            pairs.append(
                {
                    "pair_id": f"{prefix}-d-{index}",
                    "group": "D",
                    "dialect_labels": ["d1", "d2"],
                    "source_utterance_ids": [f"{prefix}-d1-{left_speaker}", f"{prefix}-d2-{right_speaker}"],
                }
            )
        return embeddings, metadata, pairs

    calibration_embeddings, calibration_metadata, calibration_pairs = build_split("cal")
    evaluation_embeddings, evaluation_metadata, evaluation_pairs = build_split("eval")

    report = evaluate_pca_projection_gate(
        {"m1": calibration_embeddings},
        {"m1": evaluation_embeddings},
        calibration_metadata,
        evaluation_metadata,
        calibration_pairs,
        evaluation_pairs,
        references=[_reference()],
        k_grid=[0, 1],
        bootstrap_replicates=1000,
        random_seeds=[1, 2],
    )

    reference_report = report["models"][0]["references"][0]
    assert report["status"] == "passed"
    assert reference_report["selected_k"] == 1
    assert reference_report["improvement_ratio"] >= 0.05
    assert reference_report["ci"]["lower"] > 0.0
    assert reference_report["random_component_scale_policy"] == "calibration_refit_per_random_draw"


def test_pca_projection_gate_rejects_too_few_bootstrap_replicates():
    with pytest.raises(ValueError, match="1000"):
        evaluate_pca_projection_gate({}, {}, {}, {}, [], [], references=[_reference()], bootstrap_replicates=999)
