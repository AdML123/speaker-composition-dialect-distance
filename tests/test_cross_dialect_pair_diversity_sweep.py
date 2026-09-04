import pytest
import torch

from src.cross_dialect_pair_diversity_sweep import (
    build_arg_parser,
    apply_normalized_reference_targets,
    run_trainable_head_architecture_control,
    run_pair_diversity_sweep_training,
)
from src.cross_dialect_projection_head import (
    LinearProjectionHead,
    build_pair_diversity_sweep_conditions,
    evaluate_projection_head_gate,
    summarize_cross_pool_diversity,
    validate_pair_count_matched_conditions,
)


def test_linear_head_control_uses_a_trainable_single_layer_projection():
    head = LinearProjectionHead(input_dim=4, output_dim=2)
    output = head(torch.tensor([[1.0, 0.0, 0.0, 1.0]]))
    assert tuple(output.shape) == (1, 2)
    assert torch.linalg.vector_norm(output, dim=1).item() == pytest.approx(1.0)
    modules = list(head.modules())
    linear_layers = [module for module in modules if isinstance(module, torch.nn.Linear)]
    assert len(linear_layers) == 1
    assert not any(isinstance(module, torch.nn.GELU) for module in modules)
    assert not any(isinstance(module, torch.nn.Dropout) for module in modules)


def test_pair_diversity_sweep_reports_pair_count_matched_conditions():
    records = [
        {"utterance_id": "s1m", "speaker_id": "s1", "dialect_label": "Mandarin", "split": "calibration"},
        {"utterance_id": "s1r", "speaker_id": "s1", "dialect_label": "Southwestern", "split": "calibration"},
        {"utterance_id": "s2m", "speaker_id": "s2", "dialect_label": "Mandarin", "split": "calibration"},
        {"utterance_id": "s2r", "speaker_id": "s2", "dialect_label": "Southwestern", "split": "calibration"},
        {"utterance_id": "s3m", "speaker_id": "s3", "dialect_label": "Mandarin", "split": "calibration"},
        {"utterance_id": "s3e", "speaker_id": "s3", "dialect_label": "Eastern", "split": "calibration"},
        {"utterance_id": "s4m", "speaker_id": "s4", "dialect_label": "Mandarin", "split": "calibration"},
        {"utterance_id": "s4e", "speaker_id": "s4", "dialect_label": "Eastern", "split": "calibration"},
    ]
    base_examples = [
        {"pair_id": "b1", "utterance_ids": ["s1m", "s1r"], "speaker_ids": ["s1", "s1"], "dialect_labels": ["Mandarin", "Southwestern"], "target": 1.0},
        {"pair_id": "b2", "utterance_ids": ["s2m", "s2r"], "speaker_ids": ["s2", "s2"], "dialect_labels": ["Mandarin", "Southwestern"], "target": 1.0},
    ]
    conditions = build_pair_diversity_sweep_conditions(records, base_examples, seed=20260829)
    assert set(conditions) == {"same_speaker", "coverage_matched_shuffled", "speaker_broadened_shuffled", "speaker_and_dialect_broadened_shuffled"}
    validate_pair_count_matched_conditions(conditions)
    summaries = {name: summarize_cross_pool_diversity(examples) for name, examples in conditions.items() if examples}
    assert summaries["coverage_matched_shuffled"]["pair_count"] == 2
    assert summaries["coverage_matched_shuffled"]["unique_speaker_count"] == summaries["same_speaker"]["unique_speaker_count"]
    assert summaries["speaker_broadened_shuffled"]["unique_speaker_count"] >= summaries["coverage_matched_shuffled"]["unique_speaker_count"]
    assert summaries["speaker_and_dialect_broadened_shuffled"]["unique_dialect_pair_count"] >= summaries["coverage_matched_shuffled"]["unique_dialect_pair_count"]


def test_pair_diversity_sweep_rejects_unequal_pair_counts():
    conditions = {
        "a": [{"pair_id": "p1", "speaker_ids": ["s1"], "dialect_labels": ["Mandarin", "Southwestern"], "target": 1.0}],
        "b": [
            {"pair_id": "p2", "speaker_ids": ["s2"], "dialect_labels": ["Mandarin", "Southwestern"], "target": 1.0},
            {"pair_id": "p3", "speaker_ids": ["s3"], "dialect_labels": ["Mandarin", "Eastern"], "target": 0.8},
        ],
    }
    with pytest.raises(ValueError, match="pair count"):
        validate_pair_count_matched_conditions(conditions)


def test_pair_diversity_sweep_parser_exposes_help():
    parser = build_arg_parser()
    help_text = parser.format_help()
    assert "--calibration-embedding" in help_text
    assert "--analysis-output" in help_text


def test_diversity_sweep_recomputes_targets_after_dialect_pair_changes():
    conditions = {
        "speaker_and_dialect_broadened_shuffled": [
            {
                "pair_id": "p1",
                "dialect_labels": ["Mandarin", "Eastern"],
                "target": 0.0,
            }
        ]
    }
    reference = {
        "Mandarin": {"Mandarin": 0.0, "Eastern": 40.0},
        "Eastern": {"Mandarin": 40.0, "Eastern": 0.0},
    }
    updated = apply_normalized_reference_targets(conditions, reference)
    assert updated["speaker_and_dialect_broadened_shuffled"][0]["target"] == pytest.approx(1.0)


def test_diversity_sweep_trains_pair_count_matched_conditions():
    calibration_records = [
        {"utterance_id": f"c{speaker}{suffix}", "speaker_id": f"s{speaker}", "dialect_label": dialect, "split": "calibration"}
        for speaker, regional in enumerate(("Southwestern", "Southwestern", "Eastern", "Eastern"), start=1)
        for suffix, dialect in (("m", "Mandarin"), ("r", regional))
    ]
    evaluation_records = [
        {"utterance_id": f"e{speaker}{suffix}", "speaker_id": f"t{speaker}", "dialect_label": dialect, "split": "evaluation"}
        for speaker, regional in enumerate(("Southwestern", "Eastern"), start=1)
        for suffix, dialect in (("m", "Mandarin"), ("r", regional))
    ]
    calibration_embeddings = {
        record["utterance_id"]: [1.0, float(index), float(index % 2), 0.0]
        for index, record in enumerate(calibration_records)
    }
    evaluation_embeddings = {
        record["utterance_id"]: [1.0, float(index), float(index % 2), 0.0]
        for index, record in enumerate(evaluation_records)
    }
    calibration_pairs = [
        {"pair_id": f"c{speaker}", "source_utterance_ids": [f"c{speaker}m", f"c{speaker}r"], "group": "C"}
        for speaker in range(1, 5)
    ]
    evaluation_pairs = [
        {"pair_id": f"e{speaker}", "source_utterance_ids": [f"e{speaker}m", f"e{speaker}r"], "group": "C"}
        for speaker in range(1, 3)
    ]
    reference = {
        "Mandarin": {"Mandarin": 0.0, "Southwestern": 80.0, "Eastern": 40.0},
        "Southwestern": {"Mandarin": 80.0, "Southwestern": 0.0, "Eastern": 60.0},
        "Eastern": {"Mandarin": 40.0, "Southwestern": 60.0, "Eastern": 0.0},
    }
    config = {
        "projection_head": {
            "input_dim": 4,
            "hidden_dim": 3,
            "output_dim": 2,
            "dropout": 0.0,
            "max_epochs": 1,
            "early_stopping_patience": 1,
            "batch_size": 4,
            "lambda_cross_grid": [1.0],
            "lambda_dialect_grid": [0.05],
            "learning_rate_grid": [0.0003],
            "weight_decay_grid": [0.0001],
            "bootstrap_replicates": 10,
        }
    }
    report = run_pair_diversity_sweep_training(
        calibration_embeddings=calibration_embeddings,
        evaluation_embeddings=evaluation_embeddings,
        records=calibration_records + evaluation_records,
        calibration_pairs=calibration_pairs,
        evaluation_pairs=evaluation_pairs,
        reference=reference,
        config=config,
        seeds=[20260829],
    )
    assert report["conditions"]["same_speaker"]["diversity_metrics"]["pair_count"] == 4
    assert "per_pair" not in report["conditions"]["speaker_and_dialect_broadened_shuffled"]["seed_results"][0]
    assert "broadest_vs_coverage_matched" in report["comparison"]


def test_linear_control_gate_accepts_single_targeted_model():
    records = [
        {"utterance_id": f"c{index}{suffix}", "speaker_id": f"s{index}", "dialect_label": dialect, "split": "calibration"}
        for index in range(3)
        for suffix, dialect in (("m", "Mandarin"), ("r", "Southwestern"))
    ]
    evaluation_records = [
        {"utterance_id": f"e{index}{suffix}", "speaker_id": f"t{index}", "dialect_label": dialect, "split": "evaluation"}
        for index in range(2)
        for suffix, dialect in (("m", "Mandarin"), ("r", "Southwestern"))
    ]
    calibration_pairs = [
        {"pair_id": f"c{index}", "source_utterance_ids": [f"c{index}m", f"c{index}r"], "group": "C"}
        for index in range(3)
    ]
    evaluation_pairs = [
        {"pair_id": f"e{index}", "source_utterance_ids": [f"e{index}m", f"e{index}r"], "group": "C"}
        for index in range(2)
    ]
    calibration_embeddings = {
        record["utterance_id"]: [1.0, float(index), 0.0, 0.0]
        for index, record in enumerate(records)
    }
    evaluation_embeddings = {
        record["utterance_id"]: [1.0, float(index), 0.0, 0.0]
        for index, record in enumerate(evaluation_records)
    }
    config = {
        "projection_head": {
            "input_dim": 4,
            "hidden_dim": 3,
            "output_dim": 2,
            "dropout": 0.0,
            "max_epochs": 1,
            "early_stopping_patience": 1,
            "batch_size": 2,
            "inner_cv_folds": 2,
            "lambda_cross_grid": [0.25],
            "lambda_dialect_grid": [0.05],
            "learning_rate_grid": [0.0003],
            "weight_decay_grid": [0.0001],
            "bootstrap_replicates": 10,
        }
    }
    reference = {
        "Mandarin": {"Mandarin": 0.0, "Southwestern": 80.0},
        "Southwestern": {"Mandarin": 80.0, "Southwestern": 0.0},
    }
    report = evaluate_projection_head_gate(
        calibration_embeddings_by_model={"chinese_hubert_large": calibration_embeddings},
        evaluation_embeddings_by_model={"chinese_hubert_large": evaluation_embeddings},
        records=records + evaluation_records,
        calibration_pairs=calibration_pairs,
        evaluation_pairs=evaluation_pairs,
        references=[{"name": "taxonomy", "matrix": reference}],
        config=config,
        seed=20260829,
        head_kind="linear",
    )
    assert [row["model_name"] for row in report["models"]] == ["chinese_hubert_large"]
    assert report["models"][0]["references"][0]["training"]["head_kind"] == "linear"


def test_trainable_architecture_control_compares_matched_head_kinds():
    records = [
        {"utterance_id": f"c{index}{suffix}", "speaker_id": f"s{index}", "dialect_label": dialect, "split": "calibration"}
        for index in range(4)
        for suffix, dialect in (("m", "Mandarin"), ("r", "Southwestern"))
    ] + [
        {"utterance_id": f"e{index}{suffix}", "speaker_id": f"t{index}", "dialect_label": dialect, "split": "evaluation"}
        for index in range(2)
        for suffix, dialect in (("m", "Mandarin"), ("r", "Southwestern"))
    ]
    calibration_embeddings = {
        record["utterance_id"]: [1.0, float(index), float(index % 2), 0.0]
        for index, record in enumerate(record for record in records if record["split"] == "calibration")
    }
    evaluation_embeddings = {
        record["utterance_id"]: [1.0, float(index), float(index % 2), 0.0]
        for index, record in enumerate(record for record in records if record["split"] == "evaluation")
    }
    calibration_pairs = [
        {"pair_id": f"c{index}", "source_utterance_ids": [f"c{index}m", f"c{index}r"], "group": "C"}
        for index in range(4)
    ]
    evaluation_pairs = [
        {"pair_id": f"e{index}", "source_utterance_ids": [f"e{index}m", f"e{index}r"], "group": "C"}
        for index in range(2)
    ]
    reference = {
        "Mandarin": {"Mandarin": 0.0, "Southwestern": 80.0},
        "Southwestern": {"Mandarin": 80.0, "Southwestern": 0.0},
    }
    config = {
        "projection_head": {
            "input_dim": 4,
            "hidden_dim": 3,
            "output_dim": 2,
            "dropout": 0.0,
            "max_epochs": 1,
            "early_stopping_patience": 1,
            "batch_size": 4,
            "lambda_cross_grid": [1.0],
            "lambda_dialect_grid": [0.05],
            "learning_rate_grid": [0.0003],
            "weight_decay_grid": [0.0001],
            "bootstrap_replicates": 10,
        }
    }
    report = run_trainable_head_architecture_control(
        calibration_embeddings=calibration_embeddings,
        evaluation_embeddings=evaluation_embeddings,
        records=records,
        calibration_pairs=calibration_pairs,
        evaluation_pairs=evaluation_pairs,
        reference=reference,
        config=config,
        seeds=[20260829],
    )
    assert report["schema"] == "trainable-head-architecture-control-v1"
    assert len(report["seed_results"]) == 1
    assert report["comparison"]["pair_count"] == 2
    assert {"mlp", "linear"} <= set(report["seed_results"][0]["heads"])
