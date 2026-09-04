import pytest

from src.correction_gate import evaluate_correction_gate, threshold_sensitivity


def _row(model, pair_id, group, labels, speech_distance):
    return {
        "model_name": model,
        "pair_id": pair_id,
        "group": group,
        "dialect_labels": labels,
        "distance": speech_distance,
    }


def _speaker(pair_id, group, labels, distance):
    return {
        "model_name": "ecapa",
        "pair_id": pair_id,
        "group": group,
        "dialect_labels": labels,
        "distance": distance,
    }


def _reference_matrix():
    return {
        "name": "synthetic",
        "matrix": {
            "d1": {"d1": 0.0, "d2": 1.0},
            "d2": {"d1": 1.0, "d2": 0.0},
        },
    }


def test_correction_gate_selects_lambda_on_calibration_and_improves_evaluation():
    calibration_rows = []
    evaluation_rows = []
    calibration_speaker = []
    evaluation_speaker = []
    for split_rows, split_speaker, offset in [
        (calibration_rows, calibration_speaker, 0),
        (evaluation_rows, evaluation_speaker, 100),
    ]:
        for index in range(20):
            pair_id = f"same-{offset + index}"
            speaker_distance = 0.2 + (index % 5) * 0.2
            split_rows.append(_row("m1", pair_id, "A", ["d1"], 0.5 * speaker_distance))
            split_speaker.append(_speaker(pair_id, "A", ["d1"], speaker_distance))
        for index in range(20):
            pair_id = f"diff-{offset + index}"
            speaker_distance = 0.2 + (index % 5) * 0.2
            split_rows.append(_row("m1", pair_id, "D", ["d1", "d2"], 1.0 + 0.5 * speaker_distance))
            split_speaker.append(_speaker(pair_id, "D", ["d1", "d2"], speaker_distance))

    report = evaluate_correction_gate(
        calibration_rows,
        evaluation_rows,
        calibration_speaker,
        evaluation_speaker,
        references=[_reference_matrix()],
        lambdas=[0.0, 0.5, 1.0],
        seed=20260829,
        bootstrap_replicates=1000,
    )

    model_report = report["models"][0]["references"][0]
    assert report["status"] == "passed"
    assert model_report["selected_lambda"] == 0.5
    assert model_report["improvement_ratio"] >= 0.05
    assert model_report["ci"]["lower"] > 0.0


def test_correction_gate_uses_calibration_not_evaluation_to_select_lambda():
    calibration_rows = []
    calibration_speaker = []
    evaluation_rows = []
    evaluation_speaker = []
    for index, speaker_distance in enumerate([0.0, 0.5, 1.0, 1.5]):
        calibration_rows.append(_row("m1", f"cal-a-{index}", "A", ["d1"], 0.5 * speaker_distance))
        calibration_speaker.append(_speaker(f"cal-a-{index}", "A", ["d1"], speaker_distance))
        calibration_rows.append(_row("m1", f"cal-d-{index}", "D", ["d1", "d2"], 1.0 + 0.5 * speaker_distance))
        calibration_speaker.append(_speaker(f"cal-d-{index}", "D", ["d1", "d2"], speaker_distance))
        evaluation_rows.append(_row("m1", f"eval-a-{index}", "A", ["d1"], 1.0 * speaker_distance))
        evaluation_speaker.append(_speaker(f"eval-a-{index}", "A", ["d1"], speaker_distance))
        evaluation_rows.append(_row("m1", f"eval-d-{index}", "D", ["d1", "d2"], 1.0 + 1.0 * speaker_distance))
        evaluation_speaker.append(_speaker(f"eval-d-{index}", "D", ["d1", "d2"], speaker_distance))

    report = evaluate_correction_gate(
        calibration_rows,
        evaluation_rows,
        calibration_speaker,
        evaluation_speaker,
        references=[_reference_matrix()],
        lambdas=[0.0, 0.5, 1.0],
        seed=20260829,
        bootstrap_replicates=1000,
    )

    assert report["models"][0]["references"][0]["selected_lambda"] == 0.5


def test_correction_gate_rejects_too_few_bootstrap_replicates():
    with pytest.raises(ValueError, match="1000"):
        evaluate_correction_gate([], [], [], [], references=[_reference_matrix()], bootstrap_replicates=999)


def test_threshold_sensitivity_is_not_a_significance_test():
    report = threshold_sensitivity([
        {
            "method": "pca",
            "model_name": "m1",
            "reference_name": "binary",
            "improvement_ratio": 0.04,
            "ci": {"lower": 0.01, "upper": 0.06},
        }
    ])
    assert report["rows"][0]["thresholds"][0]["passed"] is True
    assert report["rows"][0]["thresholds"][1]["passed"] is False
    assert "not inferential" in report["operational_rationale"]
