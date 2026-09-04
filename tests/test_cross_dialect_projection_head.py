import pytest
import torch

from src.cross_dialect_projection_head import (
    LinearProjectionHead,
    build_same_count_no_cross_control,
    build_shuffled_cross_dialect_control,
    build_pair_diversity_sweep_conditions,
    build_training_examples,
    build_permuted_target_cross_examples,
    compute_projection_loss,
    fit_standardizer,
    make_grouped_calibration_folds,
    make_synthetic_fitted_projection_head,
    parameter_count,
    fit_projection_head_cv,
    evaluate_projection_head_gate,
    paired_bootstrap_b4_minus_b3,
    fit_affine_from_training_pairs,
    _raw_pair_rows,
    score_pair_distances,
    ProjectionHead,
    transform_embeddings,
    train_projection_head,
    summarize_cross_pool_diversity,
    validate_pair_count_matched_conditions,
    _reference_gate_decisions,
    SupportError,
)


def test_parameter_matched_mlp_is_within_one_percent_of_linear():
    linear = LinearProjectionHead(1024, 256)
    matched = ProjectionHead(1024, 205, 256, dropout=0.0)
    assert parameter_count(linear) == 262400
    assert parameter_count(matched) == 262861
    assert abs(parameter_count(matched) / parameter_count(linear) - 1.0) < 0.01


def test_wide_mlp_remains_a_separate_capacity_control():
    wide = ProjectionHead(1024, 512, 256, dropout=0.2)
    assert parameter_count(wide) == 656128


def test_target_permutation_changes_pair_targets_not_pair_identity():
    examples = [
        {"pair_id": "x1", "utterance_ids": ["a1", "b1"], "target": 0.2},
        {"pair_id": "x2", "utterance_ids": ["a2", "b2"], "target": 0.8},
        {"pair_id": "x3", "utterance_ids": ["a3", "b3"], "target": 0.5},
    ]
    permuted = build_permuted_target_cross_examples(examples, seed=11)
    assert [item["pair_id"] for item in permuted] == ["x1", "x2", "x3"]
    assert [item["utterance_ids"] for item in permuted] == [
        ["a1", "b1"],
        ["a2", "b2"],
        ["a3", "b3"],
    ]
    assert sorted(item["target"] for item in permuted) == [0.2, 0.5, 0.8]
    assert [item["target"] for item in permuted] != [0.2, 0.8, 0.5]
    assert all(item["target_permutation_source"] == "cross_loss_pair_distance_target" for item in permuted)


def test_fit_standardizer_uses_only_calibration_rows():
    calibration = {"u1": [1.0, 2.0], "u2": [3.0, 4.0]}
    standardizer = fit_standardizer(calibration)
    transformed = standardizer.transform({"u1": [1.0, 2.0]})
    assert transformed["u1"][0] == pytest.approx(-1.0)
    assert transformed["u1"][1] == pytest.approx(-1.0)
    assert standardizer.dimension == 2


def test_affine_calibration_from_training_pairs_is_fit_without_validation_rows():
    result = fit_affine_from_training_pairs([0.2, 0.8], [0.0, 1.0])

    assert result["slope"] == pytest.approx(1.6666666667)
    assert result["intercept"] == pytest.approx(-0.3333333333)


def test_fit_standardizer_handles_zero_variance_features():
    calibration = {"u1": [1.0, 2.0], "u2": [1.0, 4.0]}
    standardizer = fit_standardizer(calibration)
    transformed = standardizer.transform({"u1": [1.0, 2.0]})
    assert transformed["u1"][0] == pytest.approx(0.0)
    assert all(value == value for value in transformed["u1"])


def test_build_training_examples_discovers_same_speaker_cross_dialect_records():
    records = [
        {"utterance_id": "u1", "speaker_id": "s1", "dialect_label": "Mandarin", "split": "calibration"},
        {"utterance_id": "u2", "speaker_id": "s1", "dialect_label": "Southwestern", "split": "calibration"},
        {"utterance_id": "u3", "speaker_id": "s2", "dialect_label": "Mandarin", "split": "calibration"},
    ]
    pair_manifest = [
        {
            "pair_id": "B-1",
            "source_utterance_ids": ["u1", "u3"],
            "group": "B",
            "dialect_labels": ["Mandarin"],
            "speaker_ids": ["s1", "s2"],
        }
    ]
    reference = {
        "Mandarin": {"Mandarin": 0.0, "Southwestern": 80.0},
        "Southwestern": {"Mandarin": 80.0, "Southwestern": 0.0},
    }
    examples = build_training_examples(records, pair_manifest, reference)
    assert examples["same_speaker_cross_dialect_count"] == 1
    assert examples["pair_examples"][0]["target"] == pytest.approx(0.0)
    assert examples["cross_dialect_examples"][0]["speaker_id"] == "s1"
    assert examples["cross_dialect_examples"][0]["target"] == pytest.approx(1.0)
    assert [item["pair_id"] for item in examples["pair_examples"]] == ["B-1"]


def test_projection_head_outputs_unit_norm_256_vectors():
    torch.manual_seed(20260829)
    model = ProjectionHead(input_dim=4, hidden_dim=3, output_dim=2, dropout=0.0)
    output = model(torch.tensor([[1.0, 2.0, 3.0, 4.0]]))
    assert tuple(output.shape) == (1, 2)
    assert torch.linalg.vector_norm(output, dim=1).item() == pytest.approx(1.0)


def test_loss_has_explicit_cross_dialect_component():
    loss = compute_projection_loss(
        pair_distances=torch.tensor([0.1, 0.8]),
        pair_targets=torch.tensor([0.0, 1.0]),
        cross_distances=torch.tensor([0.8]),
        cross_targets=torch.tensor([1.0]),
        dialect_logits=torch.tensor([[3.0, 0.0], [0.0, 3.0]]),
        dialect_targets=torch.tensor([0, 1]),
        lambda_cross=0.5,
        lambda_dialect=0.1,
    )
    assert torch.isfinite(loss["total"])
    assert loss["cross"].item() > 0.0
    assert loss["dialect"].item() > 0.0


def test_zero_cross_weight_keeps_diagnostic_but_removes_cross_contribution():
    kwargs = dict(
        pair_distances=torch.tensor([0.1]),
        pair_targets=torch.tensor([0.0]),
        cross_distances=torch.tensor([0.8]),
        cross_targets=torch.tensor([1.0]),
        dialect_logits=torch.tensor([[3.0, 0.0]]),
        dialect_targets=torch.tensor([0]),
        lambda_dialect=0.1,
    )
    weighted = compute_projection_loss(**kwargs, lambda_cross=0.5)
    unweighted = compute_projection_loss(**kwargs, lambda_cross=0.0)
    assert weighted["cross"].item() > 0.0
    assert unweighted["cross"].item() > 0.0
    assert unweighted["total"].item() == pytest.approx(
        unweighted["pair"].item() + 0.1 * unweighted["dialect"].item()
    )


def test_projection_head_same_seed_is_deterministic_in_eval_mode():
    inputs = torch.tensor([[1.0, 2.0, 3.0, 4.0]])
    torch.manual_seed(20260829)
    first = ProjectionHead(input_dim=4, hidden_dim=3, output_dim=2, dropout=0.2)
    torch.manual_seed(20260829)
    second = ProjectionHead(input_dim=4, hidden_dim=3, output_dim=2, dropout=0.2)
    first.eval()
    second.eval()
    assert torch.equal(first(inputs), second(inputs))


def test_grouped_folds_keep_cross_dialect_support():
    records = []
    for speaker in ("s1", "s2", "s3", "s4", "s5"):
        records.extend(
            [
                {"utterance_id": f"{speaker}-m", "speaker_id": speaker, "dialect_label": "Mandarin", "split": "calibration"},
                {
                    "utterance_id": f"{speaker}-r",
                    "speaker_id": speaker,
                    "dialect_label": "Southwestern",
                    "split": "calibration",
                },
            ]
        )
    folds = make_grouped_calibration_folds(records, n_splits=5, seed=20260829)
    assert len(folds) == 5
    assert all(fold["validation_speakers"] for fold in folds)
    assert all(fold["cross_dialect_validation_count"] >= 1 for fold in folds)


def test_grouped_folds_raise_when_cross_dialect_support_is_missing():
    records = [
        {"utterance_id": "u1", "speaker_id": "s1", "dialect_label": "Mandarin", "split": "calibration"},
        {"utterance_id": "u2", "speaker_id": "s2", "dialect_label": "Mandarin", "split": "calibration"},
    ]
    with pytest.raises(SupportError):
        make_grouped_calibration_folds(records, n_splits=2, seed=20260829)


def test_transform_does_not_require_evaluation_dialect_labels():
    fitted_model = make_synthetic_fitted_projection_head(input_dim=4, output_dim=2, seed=20260829)
    transformed = transform_embeddings(
        embeddings={"u1": [1.0, 0.0, 0.0, 0.0]},
        fitted_model=fitted_model,
    )
    assert set(transformed) == {"u1"}
    assert len(transformed["u1"]) == 2


def test_shuffled_cross_dialect_control_is_deterministic():
    records = [
        {"utterance_id": "u1", "speaker_id": "s1", "dialect_label": "Mandarin", "split": "calibration"},
        {"utterance_id": "u2", "speaker_id": "s1", "dialect_label": "Southwestern", "split": "calibration"},
        {"utterance_id": "u3", "speaker_id": "s2", "dialect_label": "Mandarin", "split": "calibration"},
        {"utterance_id": "u4", "speaker_id": "s2", "dialect_label": "Southwestern", "split": "calibration"},
    ]
    first = build_shuffled_cross_dialect_control(records, seed=20260829)
    second = build_shuffled_cross_dialect_control(records, seed=20260829)
    assert first == second
    assert any(
        tuple(item["utterance_ids"]) != expected
        for item, expected in zip(first, [("u1", "u2"), ("u3", "u4")])
    )


def test_same_count_no_cross_control_adds_pairs_without_cross_weighting():
    examples = {
        "pair_examples": [{"pair_id": "B-1", "target": 0.0}],
        "cross_dialect_examples": [{"pair_id": "C-extra", "target": 1.0}],
    }
    control = build_same_count_no_cross_control(examples)
    assert len(control["pair_examples"]) == 2
    assert control["lambda_cross"] == 0.0
    assert control["pair_examples"][1]["pair_id"] == "C-extra"


def test_projection_head_cv_selects_locked_grid_and_uses_grouped_folds():
    records = []
    embeddings = {}
    pairs = []
    for index in range(5):
        speaker = f"s{index}"
        left_id = f"{speaker}-m"
        right_id = f"{speaker}-r"
        records.extend(
            [
                {"utterance_id": left_id, "speaker_id": speaker, "dialect_label": "Mandarin", "split": "calibration"},
                {"utterance_id": right_id, "speaker_id": speaker, "dialect_label": "Southwestern", "split": "calibration"},
            ]
        )
        embeddings[left_id] = [1.0, float(index), 0.0, 0.0]
        embeddings[right_id] = [0.0, float(index), 1.0, 0.0]
        pairs.append(
            {
                "pair_id": f"C-{index}",
                "source_utterance_ids": [left_id, right_id],
                "group": "C",
                "dialect_labels": ["Mandarin", "Southwestern"],
                "speaker_ids": [speaker],
            }
        )
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
            "max_epochs": 2,
            "early_stopping_patience": 1,
            "batch_size": 4,
            "inner_cv_folds": 5,
            "lambda_cross_grid": [0.25],
            "lambda_dialect_grid": [0.05],
            "learning_rate_grid": [0.0003],
            "weight_decay_grid": [0.0001],
        }
    }
    fitted = fit_projection_head_cv(
        embeddings=embeddings,
        records=records,
        pair_manifest=pairs,
        reference=reference,
        config=config,
        seed=20260829,
    )
    assert fitted["selected"]["lambda_cross"] == 0.25
    assert fitted["selected"]["lambda_dialect"] == 0.05
    assert fitted["folds_used"] == 5
    assert fitted["best_validation_mae"] >= 0.0


def test_train_records_fixed_batch_composition():
    records = [
        {"utterance_id": "u1", "speaker_id": "s1", "dialect_label": "Mandarin", "split": "calibration"},
        {"utterance_id": "u2", "speaker_id": "s1", "dialect_label": "Southwestern", "split": "calibration"},
        {"utterance_id": "u3", "speaker_id": "s2", "dialect_label": "Mandarin", "split": "calibration"},
        {"utterance_id": "u4", "speaker_id": "s2", "dialect_label": "Southwestern", "split": "calibration"},
    ]
    embeddings = {
        "u1": [1.0, 0.0, 0.0, 0.0],
        "u2": [0.0, 1.0, 0.0, 0.0],
        "u3": [1.0, 0.1, 0.0, 0.0],
        "u4": [0.1, 1.0, 0.0, 0.0],
    }
    pairs = [
        {"pair_id": "p1", "source_utterance_ids": ["u1", "u3"], "group": "A"},
        {"pair_id": "p2", "source_utterance_ids": ["u2", "u4"], "group": "A"},
    ]
    reference = {
        "Mandarin": {"Mandarin": 0.0, "Southwestern": 1.0},
        "Southwestern": {"Mandarin": 1.0, "Southwestern": 0.0},
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
        }
    }
    fitted = train_projection_head(
        embeddings,
        records,
        pairs,
        reference,
        lambda_cross=0.25,
        lambda_dialect=0.05,
        learning_rate=0.0003,
        weight_decay=0.0001,
        config=config,
        seed=20260829,
        validation_embeddings=embeddings,
        validation_records=records,
        validation_pairs=pairs,
    )
    assert fitted["batch_composition"]["batch_size"] == 2
    assert fitted["batch_composition"]["n_pair"] + fitted["batch_composition"]["n_cross"] == 2
    assert fitted["batch_composition"]["n_cross"] == 1


def test_prevalence_balanced_cross_loss_uses_equal_stratum_means():
    pair_distances = torch.tensor([0.0])
    pair_targets = torch.tensor([0.0])
    cross_distances = torch.tensor([0.5, 0.5, 0.0])
    cross_targets = torch.tensor([0.0, 0.0, 1.0])
    dialect_logits = torch.empty((0, 2))
    dialect_targets = torch.empty((0,), dtype=torch.long)
    ordinary = compute_projection_loss(
        pair_distances, pair_targets, cross_distances, cross_targets,
        dialect_logits, dialect_targets, 1.0, 0.0, "ordinary"
    )
    balanced = compute_projection_loss(
        pair_distances, pair_targets, cross_distances, cross_targets,
        dialect_logits, dialect_targets, 1.0, 0.0, "prevalence_balanced"
    )
    assert ordinary["cross"].item() == pytest.approx(0.25)
    assert balanced["cross"].item() == pytest.approx(0.3125)


def test_mixed_mean_aggregation_uses_pool_counts():
    empty_logits = torch.empty((0, 2))
    empty_targets = torch.empty((0,), dtype=torch.long)
    result = compute_projection_loss(
        torch.tensor([0.0, 1.0]), torch.tensor([0.0, 0.0]),
        torch.tensor([0.0]), torch.tensor([1.0]),
        empty_logits, empty_targets, 2.0, 0.0, "ordinary", "mixed_mean"
    )
    assert result["total"].item() == pytest.approx(0.5)


def test_projection_head_gate_report_schema():
    calibration_records = [
        {"utterance_id": f"c{index}{suffix}", "speaker_id": f"s{index}", "dialect_label": dialect, "split": "calibration"}
        for index in range(3)
        for suffix, dialect in (("m", "Mandarin"), ("r", "Southwestern"))
    ]
    calibration_records.append(
        {
            "utterance_id": "c-missing",
            "speaker_id": "s-missing",
            "dialect_label": "Mandarin",
            "split": "calibration",
        }
    )
    evaluation_records = [
        {"utterance_id": f"e{index}{suffix}", "speaker_id": f"s{index+3}", "dialect_label": dialect, "split": "evaluation"}
        for index in range(2)
        for suffix, dialect in (("m", "Mandarin"), ("r", "Southwestern"))
    ]
    calibration_pairs = [
        {
            "pair_id": f"C-{index}",
            "source_utterance_ids": [f"c{index}m", f"c{index}r"],
            "group": "C",
            "dialect_labels": ["Mandarin", "Southwestern"],
            "speaker_ids": [f"s{index}"],
        }
        for index in range(3)
    ]
    evaluation_pairs = [
        {
            "pair_id": f"C-{index}",
            "source_utterance_ids": [f"e{index}m", f"e{index}r"],
            "group": "C",
            "dialect_labels": ["Mandarin", "Southwestern"],
            "speaker_ids": [f"s{index+3}"],
        }
        for index in range(2)
    ]
    calibration_embeddings = {
        record["utterance_id"]: [1.0, float(index), 0.0, 0.0]
        for index, record in enumerate(calibration_records)
        if record["utterance_id"] != "c-missing"
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
        calibration_embeddings_by_model={"m": calibration_embeddings},
        evaluation_embeddings_by_model={"m": evaluation_embeddings},
        records=calibration_records + evaluation_records,
        calibration_pairs=calibration_pairs,
        evaluation_pairs=evaluation_pairs,
        references=[{"name": "taxonomy", "matrix": reference}],
        config=config,
        seed=20260829,
    )
    assert report["schema"] == "cross-dialect-projection-head-gate-v1"
    assert report["baseline_registry"]["B0"]["name"] == "frozen_ssl_affine"
    assert report["baseline_registry"]["B1"]["role"] == "post_processing_boundary"
    assert report["baseline_registry"]["B2"]["role"] == "post_processing_boundary"
    assert report["baseline_registry"]["B3"]["name"] == "projection_head_pair_only"
    assert report["baseline_registry"]["B4"]["name"] == "projection_head_pair_plus_same_speaker_cross"
    assert report["training"]["backbone_frozen"] is True
    assert report["training"]["evaluation_labels_used"] is False
    assert report["training"]["same_speaker_cross_dialect_count"] >= 1
    assert report["comparisons"]["lambda_cross_zero"] is not None
    assert report["comparisons"]["same_count_no_cross"] is not None
    assert report["ablations"]["shuffled_cross_dialect"] is not None
    reference_report = report["models"][0]["references"][0]
    assert reference_report["ablations"]["shuffled_cross_dialect"]["status"] == "evaluated"
    assert reference_report["ablations"]["permuted_dialect"]["status"] == "evaluated"
    assert reference_report["ablations"]["same_count_no_cross"]["status"] == "evaluated"
    assert reference_report["ablations"]["identity_head"]["status"] == "evaluated"
    assert reference_report["per_pair"][0]["absolute_error"] >= 0.0
    assert reference_report["comparisons"]["lambda_cross_zero"]["per_pair"][0]["absolute_error"] >= 0.0
    assert reference_report["comparisons"]["same_count_no_cross"]["per_pair"][0]["absolute_error"] >= 0.0
    assert reference_report["ablations"]["shuffled_cross_dialect"]["improvement_ratio"] != pytest.approx(
        reference_report["improvement_ratio"]
    )
    assert report["models"][0]["references"][0]["improvement_ratio"] >= -1.0
    assert "group_a" in report["models"][0]["references"][0]
    assert "group_c" in report["models"][0]["references"][0]


def test_linear_head_control_uses_a_trainable_single_layer_projection():
    torch.manual_seed(20260829)
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
        {
            "pair_id": "b1",
            "utterance_ids": ["s1m", "s1r"],
            "speaker_ids": ["s1", "s1"],
            "dialect_labels": ["Mandarin", "Southwestern"],
            "target": 1.0,
        },
        {
            "pair_id": "b2",
            "utterance_ids": ["s2m", "s2r"],
            "speaker_ids": ["s2", "s2"],
            "dialect_labels": ["Mandarin", "Southwestern"],
            "target": 1.0,
        },
    ]
    conditions = build_pair_diversity_sweep_conditions(records, base_examples, seed=20260829)
    assert set(conditions) == {
        "same_speaker",
        "coverage_matched_shuffled",
        "speaker_broadened_shuffled",
        "speaker_and_dialect_broadened_shuffled",
    }
    validate_pair_count_matched_conditions(conditions)
    summaries = {
        name: summarize_cross_pool_diversity(examples)
        for name, examples in conditions.items()
        if examples
    }
    assert summaries["coverage_matched_shuffled"]["pair_count"] == 2
    assert summaries["speaker_broadened_shuffled"]["unique_speaker_count"] >= summaries["coverage_matched_shuffled"]["unique_speaker_count"]
    assert summaries["speaker_and_dialect_broadened_shuffled"]["unique_dialect_pair_count"] >= summaries["coverage_matched_shuffled"]["unique_dialect_pair_count"]


def test_pair_diversity_sweep_rejects_unequal_pair_counts():
    conditions = {
        "a": [
            {
                "pair_id": "p1",
                "speaker_ids": ["s1"],
                "dialect_labels": ["Mandarin", "Southwestern"],
                "target": 1.0,
            }
        ],
        "b": [
            {
                "pair_id": "p2",
                "speaker_ids": ["s2"],
                "dialect_labels": ["Mandarin", "Southwestern"],
                "target": 1.0,
            },
            {
                "pair_id": "p3",
                "speaker_ids": ["s3"],
                "dialect_labels": ["Mandarin", "Eastern"],
                "target": 0.8,
            },
        ],
    }
    with pytest.raises(ValueError, match="pair count"):
        validate_pair_count_matched_conditions(conditions)


def test_paired_bootstrap_contrast_requires_matched_pair_ids():
    b4 = [
        {"pair_id": "p1", "absolute_error": 0.10},
        {"pair_id": "p2", "absolute_error": 0.20},
        {"pair_id": "p3", "absolute_error": 0.10},
    ]
    b3 = [
        {"pair_id": "p1", "absolute_error": 0.20},
        {"pair_id": "p2", "absolute_error": 0.30},
        {"pair_id": "p3", "absolute_error": 0.20},
    ]
    report = paired_bootstrap_b4_minus_b3(b4, b3, seed=20260829, replicates=1000)
    assert report["pair_count"] == 3
    assert report["mae_delta_b3_minus_b4"] == pytest.approx(0.10)
    assert report["ci"]["lower"] > 0
    assert report["passed"] is True


def test_paired_bootstrap_contrast_rejects_unmatched_pair_ids():
    with pytest.raises(ValueError, match="matched pair_id"):
        paired_bootstrap_b4_minus_b3(
            [{"pair_id": "p1", "absolute_error": 0.1}],
            [{"pair_id": "p2", "absolute_error": 0.1}],
            seed=20260829,
            replicates=100,
        )


def test_reference_gate_decisions_require_mechanism_controls():
    reference_report = {
        "improvement_ratio": 0.06,
        "ci": {"lower": 0.02, "upper": 0.08},
        "matched_speaker": {"relative_change": 0.0},
        "comparisons": {
            "lambda_cross_zero": {"improvement_ratio": 0.07},
            "same_count_no_cross": {"improvement_ratio": 0.02},
        },
        "ablations": {
            "shuffled_cross_dialect": {"improvement_ratio": 0.01},
            "permuted_dialect": {"improvement_ratio": 0.01},
            "identity_head": {"improvement_ratio": 0.0},
        },
        "paired_contrast_b4_vs_b3": {"passed": True, "ci": {"lower": 0.01}},
    }
    failed = _reference_gate_decisions(reference_report)
    assert failed["gate_2_efficacy"]["passed"] is True
    assert failed["gate_5_mechanism_specificity"]["passed"] is False
    assert failed["passed"] is False

    reference_report["comparisons"]["lambda_cross_zero"]["improvement_ratio"] = 0.03
    passed = _reference_gate_decisions(reference_report)
    assert passed["gate_5_mechanism_specificity"]["passed"] is True
    assert passed["passed"] is True

    reference_report["paired_contrast_b4_vs_b3"] = {"passed": False, "ci": {"lower": -0.01}}
    failed_paired = _reference_gate_decisions(reference_report)
    assert failed_paired["gate_5_mechanism_specificity"]["passed"] is False
    assert failed_paired["passed"] is False


def test_raw_pair_rows_uses_records_not_pair_manifest_labels():
    embeddings = {"u1": [1.0, 0.0], "u2": [0.0, 1.0]}
    reference = {
        "Mandarin": {"Mandarin": 0.0, "Southwestern": 80.0},
        "Southwestern": {"Mandarin": 80.0, "Southwestern": 0.0},
    }
    pairs = [
        {
            "pair_id": "A-1",
            "group": "A",
            "source_utterance_ids": ["u1", "u2"],
            "dialect_labels": ["Mandarin"],
            "speaker_ids": ["s1"],
        }
    ]
    records = [
        {"utterance_id": "u1", "speaker_id": "s1", "dialect_label": "Mandarin", "split": "calibration"},
        {"utterance_id": "u2", "speaker_id": "s1", "dialect_label": "Southwestern", "split": "calibration"},
    ]
    record_index = {record["utterance_id"]: record for record in records}
    rows = _raw_pair_rows(pairs, embeddings, reference, record_index)
    assert rows[0]["target"] == pytest.approx(1.0)


def test_score_pair_distances_uses_records_not_pair_manifest_labels():
    transformed = {"u1": [1.0, 0.0], "u2": [0.0, 1.0]}
    reference = {
        "Mandarin": {"Mandarin": 0.0, "Southwestern": 80.0},
        "Southwestern": {"Mandarin": 80.0, "Southwestern": 0.0},
    }
    pairs = [
        {
            "pair_id": "A-1",
            "group": "A",
            "source_utterance_ids": ["u1", "u2"],
            "dialect_labels": ["Mandarin"],
            "speaker_ids": ["s1"],
        }
    ]
    record_index = {
        "u1": {"utterance_id": "u1", "speaker_id": "s1", "dialect_label": "Mandarin", "split": "evaluation"},
        "u2": {
            "utterance_id": "u2",
            "speaker_id": "s1",
            "dialect_label": "Southwestern",
            "split": "evaluation",
        },
    }
    rows = score_pair_distances(pairs, transformed, reference, {"slope": 1.0, "intercept": 0.0}, record_index)
    assert rows[0]["target"] == pytest.approx(1.0)
