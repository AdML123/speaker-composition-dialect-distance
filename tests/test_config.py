from pathlib import Path

import pytest

from src.config import ConfigError, load_config


ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "configs" / "experiment.yaml"


def test_locked_protocol_values_are_present():
    config = load_config(CONFIG_PATH)

    assert {
        "protocol",
        "extractors",
        "embedding",
        "distance",
        "statistics",
        "gates",
        "speaker_proxy",
        "speaker_regression",
        "references",
        "corpora",
    } <= config.keys()
    assert config["protocol"]["seed"] == 20260829
    assert config["extractors"]["count"] == 3
    assert len(config["extractors"]["models"]) == 3
    assert all(model["dimension"] == 1024 for model in config["extractors"]["models"])
    assert config["embedding"]["pooling"] == "mean"
    assert config["distance"]["metric"] == "cosine"
    assert config["statistics"]["repeats"] == 5
    assert config["statistics"]["multiple_testing"] == "holm"
    assert config["gates"]["group_c_min_pairs"] == 200
    assert config["gates"]["correction_improvement"] == 0.05
    assert config["gates"]["matched_speaker_tolerance"] == 0.01


def test_locked_model_provenance_values_are_present():
    config = load_config(CONFIG_PATH)

    assert [model["source"] for model in config["extractors"]["models"]] == [
        "MODEL_CACHE/wavlm-large",
        "MODEL_CACHE/chinese-hubert-large",
        "MODEL_CACHE/chinese-wav2vec2-large",
    ]
    assert [model["revision"] for model in config["extractors"]["models"]] == [
        "07d9d3d8576fd3d718ee7b16b2b6242e9610d9af",
        "90cb660492214f687e60f5ca509b20edae6e75bd",
        "e3aa671ca0cfe7978171a8ebbcc4bdfecdfb2f95",
    ]
    assert [model["dimension"] for model in config["extractors"]["models"]] == [1024, 1024, 1024]
    assert [model["license"] for model in config["extractors"]["models"]] == [
        "MIT",
        "mit",
        "mit",
    ]
    assert config["speaker_proxy"]["status"] == "locked"
    assert config["speaker_proxy"]["source"] == "speechbrain/spkrec-ecapa-voxceleb"
    assert config["speaker_proxy"]["checkpoint"] == "embedding_model.ckpt"
    assert config["speaker_proxy"]["revision"] == "0f99f2d"
    assert config["speaker_proxy"]["checkpoint_hash"] == "0575cb64845e6b9a10db9bcb74d5ac32b326b8dc90352671d345e2ee3d0126a2"
    assert config["speaker_proxy"]["dimension"] == 192
    assert config["speaker_proxy"]["license"] == "apache-2.0"
    assert config["references"]["taxonomy"]["status"] == "locked"
    assert config["references"]["taxonomy"]["matrix"] == "results/references/taxonomy_matrix.json"
    assert config["references"]["taxonomy"]["provenance_record"] == "results/provenance/reference_matrices.yaml"
    assert config["references"]["sincomp"]["status"] == "locked"
    assert config["references"]["sincomp"]["matrix"] == "results/references/sinitic_data4_overall_matrix.json"
    assert config["references"]["sincomp"]["provenance_record"] == "results/provenance/reference_matrices.yaml"


def test_missing_required_section_is_rejected(tmp_path):
    path = tmp_path / "missing.yaml"
    path.write_text("protocol:\n  seed: 20260829\n", encoding="utf-8")

    with pytest.raises(ConfigError, match="missing required section"):
        load_config(path)


def test_malformed_yaml_is_rejected(tmp_path):
    path = tmp_path / "malformed.yaml"
    path.write_text("protocol: [unterminated\n", encoding="utf-8")

    with pytest.raises(ConfigError, match="malformed YAML"):
        load_config(path)


def test_non_positive_threshold_is_rejected(tmp_path):
    text = CONFIG_PATH.read_text(encoding="utf-8").replace("group_c_min_pairs: 200", "group_c_min_pairs: 0")
    path = tmp_path / "invalid-threshold.yaml"
    path.write_text(text, encoding="utf-8")

    with pytest.raises(ConfigError, match="positive"):
        load_config(path)


def test_revision_requires_provenance_record(tmp_path):
    text = CONFIG_PATH.read_text(encoding="utf-8").replace(
        "  checkpoint: embedding_model.ckpt\n  revision: 0f99f2d\n  checkpoint_hash: 0575cb64845e6b9a10db9bcb74d5ac32b326b8dc90352671d345e2ee3d0126a2\n  provenance_record: results/provenance/model_inventory.yaml",
        "  checkpoint: model.ckpt\n  revision: abc123\n  checkpoint_hash: 0575cb64845e6b9a10db9bcb74d5ac32b326b8dc90352671d345e2ee3d0126a2\n  provenance_record: pending_provenance",
    )
    path = tmp_path / "unprovenanced.yaml"
    path.write_text(text, encoding="utf-8")

    with pytest.raises(ConfigError, match="provenance"):
        load_config(path)


@pytest.mark.parametrize("pending_value", ["pending_provenance", "pending_audit"])
def test_concrete_extractor_revision_requires_provenance_record(tmp_path, pending_value):
    text = CONFIG_PATH.read_text(encoding="utf-8").replace(
        "      revision: 07d9d3d8576fd3d718ee7b16b2b6242e9610d9af",
        "      revision: abc123",
        1,
    ).replace(
        "      provenance_record: results/provenance/model_inventory.yaml",
        f"      provenance_record: {pending_value}",
        1,
    )
    path = tmp_path / "extractor-unprovenanced.yaml"
    path.write_text(text, encoding="utf-8")

    with pytest.raises(ConfigError, match="provenance"):
        load_config(path)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("group_c_min_pairs", "199"),
        ("correction_improvement", "0.04"),
        ("matched_speaker_tolerance", "0.02"),
    ],
)
def test_locked_gate_values_cannot_be_changed(tmp_path, field, value):
    text = CONFIG_PATH.read_text(encoding="utf-8").replace(
        f"{field}: { {'group_c_min_pairs': '200', 'correction_improvement': '0.05', 'matched_speaker_tolerance': '0.01'}[field] }",
        f"{field}: {value}",
    )
    path = tmp_path / f"invalid-{field}.yaml"
    path.write_text(text, encoding="utf-8")

    with pytest.raises(ConfigError, match="locked"):
        load_config(path)


def test_speaker_proxy_status_matches_pending_or_locked_fields(tmp_path):
    pending_with_checkpoint = CONFIG_PATH.read_text(encoding="utf-8").replace(
        "speaker_proxy:\n  name: ecapa\n  source: speechbrain/spkrec-ecapa-voxceleb\n  status: locked\n  checkpoint: embedding_model.ckpt\n  revision: 0f99f2d\n  checkpoint_hash: 0575cb64845e6b9a10db9bcb74d5ac32b326b8dc90352671d345e2ee3d0126a2\n  provenance_record: results/provenance/model_inventory.yaml\n  dimension: 192\n  license: apache-2.0\n  cache_root: MODEL_CACHE",
        "speaker_proxy:\n  name: ecapa\n  source: speechbrain/spkrec-ecapa-voxceleb\n  status: pending_provenance\n  checkpoint: model.ckpt\n  revision: 0f99f2d\n  checkpoint_hash: 0575cb64845e6b9a10db9bcb74d5ac32b326b8dc90352671d345e2ee3d0126a2\n  provenance_record: results/provenance/model_inventory.yaml\n  dimension: 192\n  license: apache-2.0\n  cache_root: MODEL_CACHE",
    )
    path = tmp_path / "pending-mismatch.yaml"
    path.write_text(pending_with_checkpoint, encoding="utf-8")
    with pytest.raises(ConfigError, match="speaker_proxy"):
        load_config(path)


def test_locked_references_require_concrete_matrix_and_provenance(tmp_path):
    text = CONFIG_PATH.read_text(encoding="utf-8").replace(
        "  taxonomy:\n    status: locked\n    source: Language Atlas of China 2012 Mandarin subgroup taxonomy\n    matrix: results/references/taxonomy_matrix.json\n    provenance_record: results/provenance/reference_matrices.yaml",
        "  taxonomy:\n    status: locked\n    source: pending_audit\n    matrix: pending_audit\n    provenance_record: pending_audit",
    )
    path = tmp_path / "reference-missing-provenance.yaml"
    path.write_text(text, encoding="utf-8")

    with pytest.raises(ConfigError, match="references.taxonomy"):
        load_config(path)

    locked_with_audit_pending = CONFIG_PATH.read_text(encoding="utf-8").replace(
        "speaker_proxy:\n  name: ecapa\n  source: speechbrain/spkrec-ecapa-voxceleb\n  status: locked\n  checkpoint: embedding_model.ckpt\n  revision: 0f99f2d\n  checkpoint_hash: 0575cb64845e6b9a10db9bcb74d5ac32b326b8dc90352671d345e2ee3d0126a2\n  provenance_record: results/provenance/model_inventory.yaml\n  dimension: 192\n  license: apache-2.0\n  cache_root: MODEL_CACHE",
        "speaker_proxy:\n  name: ecapa\n  source: speechbrain/spkrec-ecapa-voxceleb\n  status: locked\n  checkpoint: model.ckpt\n  revision: abc123\n  checkpoint_hash: 0575cb64845e6b9a10db9bcb74d5ac32b326b8dc90352671d345e2ee3d0126a2\n  provenance_record: pending_audit\n  dimension: 192\n  license: apache-2.0\n  cache_root: MODEL_CACHE",
    )
    path.write_text(locked_with_audit_pending, encoding="utf-8")
    with pytest.raises(ConfigError, match="speaker_proxy"):
        load_config(path)


def test_locked_block_regularized_rank1_dialect_perturbation_contract_is_present():
    config = load_config(CONFIG_PATH)

    regression = config["speaker_regression"]
    assert regression["status"] == "locked"
    assert regression["feature_source"] == "ecapa"
    assert regression["target_source"] == "cell_offset_minus_dialect_main_effect"
    assert regression["model"] == "block_regularized_rank1_dialect_modulation"
    assert regression["parameterization"] == "block_regularized_shared_ridge_plus_rank1_dialect_modulation"
    assert regression["regularization_family"] == "block_diagonal_ridge"
    assert regression["rank"] == 1
    assert regression["base_alpha_grid"] == [1.0, 3.0, 10.0, 30.0]
    assert regression["uniform_control_alpha_grid"] == [1.0, 10.0, 30.0, 100.0, 300.0, 1000.0]
    assert regression["w_penalty_multiplier"] == 100.0
    assert regression["bias_penalty_multiplier"] == 10.0
    assert regression["low_rank_penalty_multiplier"] == 1.0
    assert regression["als_max_iter"] == 100
    assert regression["als_n_restarts"] == 5
    assert regression["als_convergence_tol"] == 1e-06
    assert regression["als_convergence_patience"] == 20
    assert regression["inner_cv_folds"] == 5
    assert regression["leave_pair_out"] is True


def test_projection_head_contract_is_locked():
    config = load_config(CONFIG_PATH)
    head = config["projection_head"]
    assert head["status"] == "proposed"
    assert head["input_dim"] == 1024
    assert head["hidden_dim"] == 512
    assert head["output_dim"] == 256
    assert head["dropout"] == 0.20
    assert head["max_epochs"] == 100
    assert head["early_stopping_patience"] == 10
    assert head["inner_cv_folds"] == 5
    assert head["lambda_cross_grid"] == [0.25, 0.5, 1.0]
    assert head["lambda_dialect_grid"] == [0.05, 0.1]
    assert head["learning_rate_grid"] == [0.0001, 0.0003]
    assert head["weight_decay_grid"] == [0.0001, 0.001]


def test_projection_head_seed_sweep_is_locked():
    config = load_config(CONFIG_PATH)
    seeds = config["projection_head"]["seed_sweep"]
    assert seeds == [20260829, 20260830, 20260831, 20260901, 20260902]
    assert config["protocol"]["seed"] == 20260829


def test_gradient_isolation_contract_is_locked():
    config = load_config(CONFIG_PATH)
    isolation = config["gradient_isolation"]
    assert isolation["lambda_cross_values"] == [0.0, 0.05, 0.1, 0.25, 0.5, 1.0, 2.0, 5.0]
    assert isolation["seeds"] == [20260829, 20260830, 20260831, 20260901, 20260902]
    assert isolation["same_dialect_pair_count_match"] is True
    assert isolation["gradient_probe_seed"] == 20260829
    assert isolation["normalized_gradient_threshold"] == 0.3
    assert isolation["gradient_cosine_threshold"] == 0.3


def test_locked_block_regularized_dialect_perturbation_base_alpha_grid_is_positive(tmp_path):
    text = CONFIG_PATH.read_text(encoding="utf-8").replace(
        "  base_alpha_grid: [1.0, 3.0, 10.0, 30.0]",
        "  base_alpha_grid: [1.0, 0.0, 10.0]",
    )
    path = tmp_path / "invalid-block-regularized-base-alpha.yaml"
    path.write_text(text, encoding="utf-8")

    with pytest.raises(ConfigError, match="positive"):
        load_config(path)

    locked_with_pending = CONFIG_PATH.read_text(encoding="utf-8").replace(
        "speaker_proxy:\n  name: ecapa\n  source: speechbrain/spkrec-ecapa-voxceleb\n  status: locked\n  checkpoint: embedding_model.ckpt\n  revision: 0f99f2d\n  checkpoint_hash: 0575cb64845e6b9a10db9bcb74d5ac32b326b8dc90352671d345e2ee3d0126a2\n  provenance_record: results/provenance/model_inventory.yaml\n  dimension: 192\n  license: apache-2.0\n  cache_root: MODEL_CACHE",
        "speaker_proxy:\n  name: ecapa\n  source: speechbrain/spkrec-ecapa-voxceleb\n  status: locked\n  checkpoint: pending_provenance\n  revision: pending_provenance\n  checkpoint_hash: pending_provenance\n  provenance_record: pending_provenance\n  dimension: 192\n  license: apache-2.0\n  cache_root: MODEL_CACHE",
    )
    path.write_text(locked_with_pending, encoding="utf-8")
    with pytest.raises(ConfigError, match="speaker_proxy"):
        load_config(path)


def test_locked_block_regularized_dialect_perturbation_parameterization_is_fixed(tmp_path):
    text = CONFIG_PATH.read_text(encoding="utf-8").replace(
        "  parameterization: block_regularized_shared_ridge_plus_rank1_dialect_modulation",
        "  parameterization: independent_ridge_per_dialect",
    )
    path = tmp_path / "invalid-block-regularized-parameterization.yaml"
    path.write_text(text, encoding="utf-8")

    with pytest.raises(ConfigError, match="parameterization"):
        load_config(path)


def test_locked_block_regularized_dialect_perturbation_rank_cannot_change(tmp_path):
    text = CONFIG_PATH.read_text(encoding="utf-8").replace(
        "  rank: 1",
        "  rank: 2",
    )
    path = tmp_path / "invalid-block-regularized-rank.yaml"
    path.write_text(text, encoding="utf-8")

    with pytest.raises(ConfigError, match="rank"):
        load_config(path)


def test_locked_block_regularized_dialect_perturbation_als_max_iter_must_be_positive(tmp_path):
    text = CONFIG_PATH.read_text(encoding="utf-8").replace(
        "  als_max_iter: 100",
        "  als_max_iter: 0",
    )
    path = tmp_path / "invalid-block-regularized-als-max-iter.yaml"
    path.write_text(text, encoding="utf-8")

    with pytest.raises(ConfigError, match="als_max_iter"):
        load_config(path)


def test_locked_block_regularized_dialect_perturbation_multiplier_must_be_positive(tmp_path):
    text = CONFIG_PATH.read_text(encoding="utf-8").replace(
        "  low_rank_penalty_multiplier: 1.0",
        "  low_rank_penalty_multiplier: 0.0",
    )
    path = tmp_path / "invalid-block-regularized-multiplier.yaml"
    path.write_text(text, encoding="utf-8")

    with pytest.raises(ConfigError, match="positive"):
        load_config(path)


def test_locked_block_regularized_dialect_perturbation_status_cannot_be_unlocked(tmp_path):
    text = CONFIG_PATH.read_text(encoding="utf-8").replace(
        "speaker_regression:\n  status: locked",
        "speaker_regression:\n  status: pending_audit",
    )
    path = tmp_path / "invalid-block-regularized-status.yaml"
    path.write_text(text, encoding="utf-8")

    with pytest.raises(ConfigError, match="speaker_regression status"):
        load_config(path)
