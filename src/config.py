"""Load and validate the locked experiment protocol."""

from pathlib import Path
from typing import Any, Mapping

import yaml


class ConfigError(ValueError):
    """Raised when the experiment configuration is malformed or unsafe."""


REQUIRED_SECTIONS = {
    "protocol",
    "extractors",
    "embedding",
    "distance",
    "statistics",
    "gates",
    "speaker_proxy",
    "speaker_regression",
    "projection_head",
    "references",
    "corpora",
}
PENDING_PROVENANCE = "pending_provenance"
PENDING_VALUES = {"pending_provenance", "pending_audit"}
LOCKED_GATE_VALUES = {
    "group_c_min_pairs": 200,
    "correction_improvement": 0.05,
    "matched_speaker_tolerance": 0.01,
}


def _require_mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ConfigError(f"{label} must be a mapping")
    return value


def load_config(path: str | Path) -> dict[str, Any]:
    """Read a YAML protocol and reject incomplete or unverifiable settings."""
    try:
        with Path(path).open("r", encoding="utf-8") as handle:
            data = yaml.safe_load(handle)
    except (OSError, yaml.YAMLError) as exc:
        if isinstance(exc, yaml.YAMLError):
            raise ConfigError("malformed YAML") from exc
        raise ConfigError(f"unable to read configuration: {exc}") from exc
    if not isinstance(data, Mapping):
        raise ConfigError("malformed YAML: top-level document must be a mapping")

    missing = REQUIRED_SECTIONS.difference(data)
    if missing:
        raise ConfigError(f"missing required section: {sorted(missing)[0]}")

    protocol = _require_mapping(data["protocol"], "protocol")
    extractors = _require_mapping(data["extractors"], "extractors")
    embedding = _require_mapping(data["embedding"], "embedding")
    distance = _require_mapping(data["distance"], "distance")
    statistics = _require_mapping(data["statistics"], "statistics")
    gates = _require_mapping(data["gates"], "gates")
    speaker_proxy = _require_mapping(data["speaker_proxy"], "speaker_proxy")
    speaker_regression = _require_mapping(data["speaker_regression"], "speaker_regression")
    projection_head = _require_mapping(data["projection_head"], "projection_head")
    references = _require_mapping(data["references"], "references")

    if protocol.get("seed") != 20260829:
        raise ConfigError("protocol seed must be 20260829")
    models = extractors.get("models")
    if extractors.get("count") != 3 or not isinstance(models, list) or len(models) != 3:
        raise ConfigError("exactly three extractors are required")
    for model in models:
        model_map = _require_mapping(model, "extractor")
        if model_map.get("dimension") != 1024:
            raise ConfigError("extractor dimension must be 1024")
        revision = model_map.get("revision")
        provenance = model_map.get("provenance_record")
        if revision not in (None, *PENDING_VALUES) and provenance in (None, "", *PENDING_VALUES):
            raise ConfigError("extractor revision requires a provenance record")
    if embedding.get("pooling") != "mean":
        raise ConfigError("pooling must be mean")
    if distance.get("metric") != "cosine":
        raise ConfigError("distance metric must be cosine")
    if statistics.get("repeats") != 5 or statistics.get("multiple_testing") != "holm":
        raise ConfigError("statistics contract is not locked")
    for key, expected in LOCKED_GATE_VALUES.items():
        value = gates.get(key)
        if not isinstance(value, (int, float)) or value <= 0:
            raise ConfigError(f"{key} must be positive")
        if value != expected:
            raise ConfigError(f"{key} is locked at {expected}")

    checkpoint = speaker_proxy.get("checkpoint")
    revision = speaker_proxy.get("revision")
    provenance = speaker_proxy.get("provenance_record")
    status = speaker_proxy.get("status")
    fields = (checkpoint, revision, provenance)
    if status == PENDING_PROVENANCE:
        if any(field != PENDING_PROVENANCE for field in fields):
            raise ConfigError("speaker_proxy pending_provenance status requires pending fields")
    elif status == "locked":
        if any(field in (None, "", *PENDING_VALUES) for field in fields):
            raise ConfigError("speaker_proxy locked status requires concrete checkpoint, revision, and provenance")
    else:
        raise ConfigError("speaker_proxy status must be pending_provenance or locked")
    if speaker_regression.get("status") != "locked":
        raise ConfigError("speaker_regression status must be locked")
    if speaker_regression.get("feature_source") != "ecapa":
        raise ConfigError("speaker_regression feature_source must be ecapa")
    if speaker_regression.get("target_source") != "cell_offset_minus_dialect_main_effect":
        raise ConfigError("speaker_regression target_source must be cell_offset_minus_dialect_main_effect")
    if speaker_regression.get("model") != "block_regularized_rank1_dialect_modulation":
        raise ConfigError("speaker_regression model must be block_regularized_rank1_dialect_modulation")
    if speaker_regression.get("parameterization") != "block_regularized_shared_ridge_plus_rank1_dialect_modulation":
        raise ConfigError(
            "speaker_regression parameterization must be block_regularized_shared_ridge_plus_rank1_dialect_modulation"
        )
    if speaker_regression.get("regularization_family") != "block_diagonal_ridge":
        raise ConfigError("speaker_regression regularization_family must be block_diagonal_ridge")
    if speaker_regression.get("rank") != 1:
        raise ConfigError("speaker_regression rank must be 1")
    base_alpha_grid = speaker_regression.get("base_alpha_grid")
    uniform_control_alpha_grid = speaker_regression.get("uniform_control_alpha_grid")
    if not isinstance(base_alpha_grid, list) or not base_alpha_grid:
        raise ConfigError("speaker_regression base_alpha_grid must be a non-empty list")
    if any(not isinstance(alpha, (int, float)) or alpha <= 0 for alpha in base_alpha_grid):
        raise ConfigError("speaker_regression base_alpha_grid must contain positive values")
    if [float(alpha) for alpha in base_alpha_grid] != [1.0, 3.0, 10.0, 30.0]:
        raise ConfigError("speaker_regression base_alpha_grid is locked at [1.0, 3.0, 10.0, 30.0]")
    if not isinstance(uniform_control_alpha_grid, list) or not uniform_control_alpha_grid:
        raise ConfigError("speaker_regression uniform_control_alpha_grid must be a non-empty list")
    if any(not isinstance(alpha, (int, float)) or alpha <= 0 for alpha in uniform_control_alpha_grid):
        raise ConfigError("speaker_regression uniform_control_alpha_grid must contain positive values")
    if [float(alpha) for alpha in uniform_control_alpha_grid] != [1.0, 10.0, 30.0, 100.0, 300.0, 1000.0]:
        raise ConfigError(
            "speaker_regression uniform_control_alpha_grid is locked at [1.0, 10.0, 30.0, 100.0, 300.0, 1000.0]"
        )
    for key, expected in {
        "w_penalty_multiplier": 100.0,
        "bias_penalty_multiplier": 10.0,
        "low_rank_penalty_multiplier": 1.0,
    }.items():
        value = speaker_regression.get(key)
        if not isinstance(value, (int, float)) or value <= 0:
            raise ConfigError(f"speaker_regression {key} must be positive")
        if float(value) != expected:
            raise ConfigError(f"speaker_regression {key} must be {expected}")
    if speaker_regression.get("als_max_iter") != 100:
        raise ConfigError("speaker_regression als_max_iter must be 100")
    if speaker_regression.get("als_n_restarts") != 5:
        raise ConfigError("speaker_regression als_n_restarts must be 5")
    if speaker_regression.get("als_convergence_tol") != 1e-6:
        raise ConfigError("speaker_regression als_convergence_tol must be 1e-6")
    if speaker_regression.get("als_convergence_patience") != 20:
        raise ConfigError("speaker_regression als_convergence_patience must be 20")
    if speaker_regression.get("inner_cv_folds") != 5:
        raise ConfigError("speaker_regression inner_cv_folds must be 5")
    if not isinstance(speaker_regression.get("leave_pair_out"), bool) or speaker_regression.get("leave_pair_out") is not True:
        raise ConfigError("speaker_regression leave_pair_out must be true")
    if projection_head.get("status") not in {"proposed", "locked"}:
        raise ConfigError("projection_head status must be proposed or locked")
    for key in ("input_dim", "hidden_dim", "output_dim"):
        if not isinstance(projection_head.get(key), int) or projection_head[key] <= 0:
            raise ConfigError(f"projection_head {key} must be a positive integer")
    dropout = projection_head.get("dropout")
    if not isinstance(dropout, (int, float)) or not 0 <= dropout < 1:
        raise ConfigError("projection_head dropout must be in [0, 1)")
    for key in ("max_epochs", "early_stopping_patience", "batch_size", "inner_cv_folds", "bootstrap_replicates"):
        if not isinstance(projection_head.get(key), int) or projection_head[key] <= 0:
            raise ConfigError(f"projection_head {key} must be a positive integer")
    if projection_head["inner_cv_folds"] < 2:
        raise ConfigError("projection_head inner_cv_folds must be at least 2")
    if projection_head.get("activation") != "gelu":
        raise ConfigError("projection_head activation must be gelu")
    if projection_head.get("optimizer") != "adamw":
        raise ConfigError("projection_head optimizer must be adamw")
    for key in ("lambda_cross_grid", "lambda_dialect_grid", "learning_rate_grid", "weight_decay_grid"):
        values = projection_head.get(key)
        if not isinstance(values, list) or not values:
            raise ConfigError(f"projection_head {key} must be a non-empty list")
        if any(not isinstance(value, (int, float)) or value <= 0 for value in values):
            raise ConfigError(f"projection_head {key} must contain positive values")
    if projection_head.get("seed") != 20260829:
        raise ConfigError("projection_head seed must be 20260829")
    seed_sweep = projection_head.get("seed_sweep")
    if not isinstance(seed_sweep, list) or not seed_sweep:
        raise ConfigError("projection_head seed_sweep must be a non-empty list")
    if any(not isinstance(value, int) or value <= 0 for value in seed_sweep):
        raise ConfigError("projection_head seed_sweep must contain positive integer seeds")
    if 20260829 not in seed_sweep:
        raise ConfigError("projection_head seed_sweep must include 20260829")
    for name in ("taxonomy", "sincomp"):
        ref = _require_mapping(references.get(name), f"references.{name}")
        ref_status = ref.get("status")
        if ref_status not in {"pending_audit", "locked"}:
            raise ConfigError(f"references.{name} must be pending_audit or locked")
        if ref_status == "locked":
            for field in ("source", "matrix", "provenance_record"):
                if ref.get(field) in (None, "", *PENDING_VALUES):
                    raise ConfigError(f"references.{name} locked status requires concrete {field}")
    return dict(data)
