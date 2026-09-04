import inspect

import pytest

from src.cross_dialect_gradient_isolation import _run_training_condition
from src.projection_seed_sweep import build_arg_parser, run_seed_sweep, summarize_seed_distribution


def test_seed_distribution_reports_locked_summary_statistics():
    runs = [
        {"seed": 1, "b3_mae": 0.50, "b4_mae": 0.45, "b3_gain": 0.00, "b4_gain": 0.10},
        {"seed": 2, "b3_mae": 0.48, "b4_mae": 0.44, "b3_gain": 0.04, "b4_gain": 0.12},
        {"seed": 3, "b3_mae": 0.49, "b4_mae": 0.43, "b3_gain": 0.02, "b4_gain": 0.14},
        {"seed": 4, "b3_mae": 0.47, "b4_mae": 0.42, "b3_gain": 0.06, "b4_gain": 0.16},
        {"seed": 5, "b3_mae": 0.46, "b4_mae": 0.41, "b3_gain": 0.08, "b4_gain": 0.18},
    ]
    report = summarize_seed_distribution(runs)
    assert report["seed_count"] == 5
    assert report["b4_gain"]["median"] == pytest.approx(0.14)
    assert report["b4_gain"]["minimum"] == pytest.approx(0.10)
    assert report["b4_gain"]["maximum"] == pytest.approx(0.18)
    assert report["b3_minus_b4_mae"]["median"] == pytest.approx(0.05)
    assert report["all_b4_better_than_b3"] is True


def test_seed_distribution_rejects_duplicate_seeds():
    with pytest.raises(ValueError, match="unique"):
        summarize_seed_distribution([
            {"seed": 1, "b3_mae": 0.5, "b4_mae": 0.4, "b3_gain": 0.0, "b4_gain": 0.1},
            {"seed": 1, "b3_mae": 0.5, "b4_mae": 0.4, "b3_gain": 0.0, "b4_gain": 0.1},
        ])


def test_seed_sweep_exposes_and_forwards_head_kind():
    assert "head_kind" in inspect.signature(_run_training_condition).parameters
    assert "head_kind" in inspect.signature(run_seed_sweep).parameters
    args = build_arg_parser().parse_args([
        "--config", "config.yaml",
        "--calibration-embedding", "calibration.json",
        "--evaluation-embedding", "evaluation.json",
        "--records", "records.json",
        "--calibration-pairs", "calibration-pairs.json",
        "--evaluation-pairs", "evaluation-pairs.json",
        "--reference", "reference.json",
        "--model-name", "model",
        "--reference-name", "reference",
        "--output", "output.json",
        "--gate-output", "gate.json",
        "--lambda-cross", "0.0",
        "--lambda-dialect", "0.1",
        "--learning-rate", "0.0003",
        "--weight-decay", "0.001",
        "--fixed-epochs", "28",
        "--head-kind", "linear",
    ])
    assert args.head_kind == "linear"
