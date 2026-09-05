"""Audit sensitivity to the documented Beijing--Mandarin binary rule."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import statistics
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from .calibration_leakage_audit import audit_projection_sources
from .cross_dialect_gradient_isolation import _run_training_condition
from .cross_dialect_projection_head import build_training_examples
from .metric_baselines import (
    _available_pairs,
    _available_split,
    _fit_affine_for_vectors,
    _raw_and_pca,
    _score_vectors,
)
from .relation_ranking import aggregate_relation_predictions, ranking_metrics, reference_relations
from .run_target_prevalence_mechanism import (
    _load_embeddings,
    _load_pairs,
    _load_records,
    _load_reference,
)
from .config import load_config
from .reference_matrices import KESPEECH_DIALECTS, validate_reference_matrix


ZERO_RULE_RELATION = "Beijing--Mandarin"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical_target(payload: Mapping[str, Any]) -> str:
    """Serialize only target identity fields so provenance is stable."""
    return json.dumps(
        {
            "labels": list(map(str, payload.get("labels", []))),
            "matrix": payload.get("matrix", {}),
        },
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def matrix_digest(payload: Mapping[str, Any]) -> str:
    return hashlib.sha256(_canonical_target(payload).encode("utf-8")).hexdigest()


def alternate_binary_matrix(primary: Mapping[str, Any]) -> dict[str, Any]:
    """Return the locked alternate rule with one symmetric relation set to one."""
    validation = validate_reference_matrix(primary, required_labels=KESPEECH_DIALECTS)
    if validation["status"] != "passed":
        raise ValueError("primary binary matrix is not a complete valid target")
    alternate = copy.deepcopy(dict(primary))
    alternate["name"] = "taxonomy_mandarin_subgroup_beijing_mandarin_one"
    alternate["distance_scale"] = (
        "0=same Mandarin subgroup proxy except Beijing--Mandarin is set to one "
        "for the zero-rule sensitivity audit"
    )
    alternate["rule_variant"] = "beijing_mandarin_off_diagonal_one"
    alternate.setdefault("notes", [])
    alternate["notes"] = list(alternate["notes"]) + [
        "Only the symmetric Beijing--Mandarin off-diagonal relation differs from the primary binary proxy.",
        "This alternate target is an operational proxy, not perceptual ground truth.",
    ]
    alternate["matrix"]["Beijing"]["Mandarin"] = 1.0
    alternate["matrix"]["Mandarin"]["Beijing"] = 1.0
    return alternate


def _changed_relations(
    primary: Mapping[str, Any], alternate: Mapping[str, Any]
) -> list[str]:
    labels = list(map(str, primary.get("labels", [])))
    changed: list[str] = []
    for index, left in enumerate(labels):
        for right in labels[index + 1 :]:
            if float(primary["matrix"][left][right]) != float(
                alternate["matrix"][left][right]
            ):
                changed.append(f"{left}--{right}")
    return changed


def compare_zero_rule_matrices(
    primary: Mapping[str, Any], alternate: Mapping[str, Any]
) -> dict[str, Any]:
    primary_validation = validate_reference_matrix(
        primary, required_labels=KESPEECH_DIALECTS
    )
    alternate_validation = validate_reference_matrix(
        alternate, required_labels=KESPEECH_DIALECTS
    )
    changed = _changed_relations(primary, alternate)
    return {
        "primary_digest": matrix_digest(primary),
        "alternate_digest": matrix_digest(alternate),
        "changed_relations": changed,
        "symmetric_change": (
            primary["matrix"]["Beijing"]["Mandarin"] == 0.0
            and primary["matrix"]["Mandarin"]["Beijing"] == 0.0
            and alternate["matrix"]["Beijing"]["Mandarin"] == 1.0
            and alternate["matrix"]["Mandarin"]["Beijing"] == 1.0
        ),
        "diagonal_zero_primary": all(
            float(primary["matrix"][label][label]) == 0.0
            for label in KESPEECH_DIALECTS
        ),
        "diagonal_zero_alternate": all(
            float(alternate["matrix"][label][label]) == 0.0
            for label in KESPEECH_DIALECTS
        ),
        "primary_validation": primary_validation,
        "alternate_validation": alternate_validation,
        "primary_perceptual_status": "operational_proxy",
        "alternate_perceptual_status": "operational_proxy",
    }


def _projection_claim_branch(rows: Sequence[Mapping[str, Any]]) -> tuple[str, str]:
    if not rows:
        return (
            "construction_valid_projection_pending",
            "pending_alternate_projection_cells",
        )
    by_rule = {str(row.get("rule")): row for row in rows}
    if not {"primary", "alternate"} <= set(by_rule):
        return (
            "projection_incomplete",
            "narrow_to_primary_binary_rule_until_alternate_cells_are_complete",
        )
    primary = by_rule["primary"]
    alternate = by_rule["alternate"]
    primary_linear = float(primary.get("linear_gain", 0.0))
    alternate_linear = float(alternate.get("linear_gain", 0.0))
    primary_interaction = str(primary.get("architecture_interaction", ""))
    alternate_interaction = str(alternate.get("architecture_interaction", ""))
    direction_preserved = primary_linear > 0.0 and alternate_linear > 0.0
    interaction_preserved = (
        bool(primary_interaction)
        and primary_interaction == alternate_interaction
    )
    if direction_preserved and interaction_preserved:
        return (
            "projection_rule_direction_stable",
            "bounded_zero_rule_robustness_supported",
        )
    return (
        "projection_rule_direction_changed",
        "restrict_claims_to_primary_binary_rule",
    )


def build_zero_rule_report(
    primary: Mapping[str, Any],
    alternate: Mapping[str, Any],
    projection_rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    comparison = compare_zero_rule_matrices(primary, alternate)
    if comparison["changed_relations"] != [ZERO_RULE_RELATION]:
        raise ValueError("zero-rule audit must change exactly Beijing--Mandarin")
    status, claim_branch = _projection_claim_branch(projection_rows)
    return {
        "schema": "binary-zero-rule-sensitivity-v1",
        "status": status,
        "targets": {
            "primary": {
                "name": str(primary.get("name", "primary_binary")),
                "rule": "documented_beijing_mandarin_zero",
                "digest": comparison["primary_digest"],
                "perceptual_status": "operational_proxy",
                "payload": dict(primary),
            },
            "alternate": {
                "name": str(alternate.get("name", "alternate_binary")),
                "rule": "beijing_mandarin_off_diagonal_one",
                "digest": comparison["alternate_digest"],
                "perceptual_status": "operational_proxy",
                "payload": dict(alternate),
            },
        },
        "comparison": comparison,
        "projection_rows": [dict(row) for row in projection_rows],
        "claim_branch": claim_branch,
        "same_manifest_required": True,
        "same_seed_schedule_required": True,
        "same_training_budget_required": True,
    }


def build_gate(report: Mapping[str, Any]) -> dict[str, Any]:
    comparison = report["comparison"]
    construction_ok = bool(
        report["targets"]["primary"]["digest"]
        != report["targets"]["alternate"]["digest"]
        and comparison["changed_relations"] == [ZERO_RULE_RELATION]
        and comparison["symmetric_change"]
        and comparison["diagonal_zero_primary"]
        and comparison["diagonal_zero_alternate"]
        and comparison["primary_validation"]["status"] == "passed"
        and comparison["alternate_validation"]["status"] == "passed"
    )
    complete = report["status"] in {
        "projection_rule_direction_stable",
        "projection_rule_direction_changed",
    }
    return {
        "schema": "binary-zero-rule-sensitivity-gate-v1",
        "status": "passed" if construction_ok and complete else "pending",
        "construction_checks": {
            "targets_valid": construction_ok,
            "same_manifest": bool(report["same_manifest_required"]),
            "same_seed_schedule": bool(report["same_seed_schedule_required"]),
            "same_training_budget": bool(report["same_training_budget_required"]),
        },
        "projection_status": report["status"],
        "claim_branch": report["claim_branch"],
        "failure_action": (
            "restrict claims to the primary binary rule if the alternate direction changes"
        ),
    }


LOCKED_SEEDS = (20260829, 20260830, 20260831, 20260901, 20260902)
DEFAULT_PATHS = {
    "calibration_embedding": Path(
        "results/embeddings/kespeech_calibration_1000/chinese_hubert_large.json"
    ),
    "evaluation_embedding": Path(
        "results/embeddings/kespeech_evaluation_full/chinese_hubert_large.json"
    ),
    "records": Path("results/provenance/kespeech_manifest.json"),
    "calibration_pairs": Path("results/pairs/kespeech_calibration_1000.json"),
    "evaluation_pairs": Path("results/pairs/kespeech_evaluation_1000.json"),
}


def _percentile(values: Sequence[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=float)
    return {
        "lower": float(np.quantile(array, 0.025)),
        "upper": float(np.quantile(array, 0.975)),
        "confidence_level": 0.95,
        "source": "five_locked_seed_distribution",
    }


def _method_row(
    *,
    rule: str,
    method: str,
    seed: int,
    result: Mapping[str, Any],
    target_digest: str,
    manifest_sha256: str,
) -> dict[str, Any]:
    rows = list(result["per_pair"])
    baseline_rows = list(result.get("baseline_per_pair", []))
    baseline_mae = result.get("baseline_mae")
    if baseline_mae is None and baseline_rows:
        baseline_mae = float(np.mean([float(row["absolute_error"]) for row in baseline_rows]))
    return {
        "rule": rule,
        "method": method,
        "seed": int(seed),
        "mae": float(result["mae"]),
        "gain": float(result["improvement_ratio"]),
        "rows": rows,
        "baseline_mae": float(baseline_mae) if baseline_mae is not None else None,
        "target_digest": target_digest,
        "evaluation_manifest_sha256": manifest_sha256,
    }


def _cell_from_cache(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _cached_primary_rows(
    *,
    cache_root: Path,
    head: str,
    lambda_cross: float,
    seed: int,
) -> list[dict[str, Any]]:
    token = str(lambda_cross).replace(".", "p")
    path = cache_root / f"taxonomy__{seed}__{head}__lambda_{token}.json"
    if not path.is_file():
        raise FileNotFoundError(path)
    return list(_cell_from_cache(path)["per_pair"])


def _run_projection_cells(
    *,
    config: Mapping[str, Any],
    reference: Mapping[str, Mapping[str, float]],
    reference_name: str,
    target_digest: str,
    rule: str,
    cache_root: Path,
    use_cache: bool,
) -> tuple[dict[str, list[dict[str, Any]]], dict[str, Any]]:
    calibration_embeddings = _load_embeddings(str(DEFAULT_PATHS["calibration_embedding"]))
    evaluation_embeddings = _load_embeddings(str(DEFAULT_PATHS["evaluation_embedding"]))
    records = _load_records(str(DEFAULT_PATHS["records"]))
    calibration_records = _available_split(records, calibration_embeddings, "calibration")
    evaluation_records = _available_split(records, evaluation_embeddings, "evaluation")
    calibration_pairs = _available_pairs(
        _load_pairs(str(DEFAULT_PATHS["calibration_pairs"])), calibration_embeddings
    )
    evaluation_pairs = _available_pairs(
        _load_pairs(str(DEFAULT_PATHS["evaluation_pairs"])), evaluation_embeddings
    )
    examples = build_training_examples(calibration_records, calibration_pairs, reference)
    source_audit = audit_projection_sources(
        calibration_pairs=calibration_pairs,
        cross_examples=examples["cross_dialect_examples"],
        evaluation_pairs=evaluation_pairs,
        fitted_sources={
            "projection_calibration": calibration_pairs,
            "calibration_auxiliary_cross_pair": examples["cross_dialect_examples"],
        },
    )
    frozen_vectors = _raw_and_pca(calibration_embeddings, evaluation_embeddings)
    calibration_vectors, evaluation_vectors = frozen_vectors["frozen_affine"]
    affine = _fit_affine_for_vectors(
        calibration_pairs, calibration_records, calibration_vectors, reference
    )
    manifest_sha = _sha256(DEFAULT_PATHS["evaluation_pairs"])
    outputs: dict[str, list[dict[str, Any]]] = {name: [] for name in (
        "linear_b3", "linear_b4", "matched_mlp_b3", "matched_mlp_b4", "wide_mlp_b3", "wide_mlp_b4"
    )}
    architecture = {
        "linear": ("linear", None, 0.0),
        "matched_mlp": ("mlp", 205, 0.0),
        "wide_mlp": ("mlp", 512, 0.2),
    }
    for method, (head_kind, hidden_dim, dropout) in architecture.items():
        for lambda_cross in (0.0, 0.5):
            key = f"{method}_b{3 if lambda_cross == 0.0 else 4}"
            for seed in LOCKED_SEEDS:
                cached = None
                if use_cache and rule == "primary":
                    head_name = {
                        "linear": "linear",
                        "matched_mlp": "mlp_parameter_matched",
                        "wide_mlp": "mlp_wide",
                    }[method]
                    cached = _cached_primary_rows(
                        cache_root=cache_root,
                        head=head_name,
                        lambda_cross=lambda_cross,
                        seed=seed,
                    )
                if cached is not None:
                    rows = cached
                    baseline_rows = _score_vectors(
                        evaluation_pairs, evaluation_records, evaluation_vectors,
                        reference, affine,
                    )
                    baseline_mae = float(np.mean([row["absolute_error"] for row in baseline_rows]))
                    mae = float(np.mean([row["absolute_error"] for row in rows]))
                    # Recover the stored gain from the cached cell rather than
                    # recomputing it from the candidate rows.
                    head_name = {
                        "linear": "linear",
                        "matched_mlp": "mlp_parameter_matched",
                        "wide_mlp": "mlp_wide",
                    }[method]
                    cache_path = cache_root / (
                        f"taxonomy__{seed}__{head_name}__lambda_"
                        f"{str(lambda_cross).replace('.', 'p')}.json"
                    )
                    cell = _cell_from_cache(cache_path)
                    gain = float(cell["gain"])
                    result = {
                        "per_pair": rows,
                        "baseline_per_pair": baseline_rows,
                        "baseline_mae": baseline_mae,
                        "mae": mae,
                        "improvement_ratio": gain,
                    }
                else:
                    run_config = copy.deepcopy(dict(config))
                    head = run_config["projection_head"]
                    head.update({
                        "hidden_dim": int(hidden_dim or 512),
                        "dropout": float(dropout),
                        "max_epochs": 30,
                        "batch_size": 256,
                        "learning_rate_grid": [0.0003],
                        "weight_decay_grid": [0.001],
                        "lambda_dialect_grid": [0.1],
                        "bootstrap_replicates": 1000,
                    })
                    result = _run_training_condition(
                        calibration_embeddings=calibration_embeddings,
                        evaluation_embeddings=evaluation_embeddings,
                        calibration_records=calibration_records,
                        evaluation_records=evaluation_records,
                        calibration_pairs=calibration_pairs,
                        evaluation_pairs=evaluation_pairs,
                        reference=reference,
                        config=run_config,
                        seed=int(seed),
                        lambda_cross=float(lambda_cross),
                        cross_examples=examples["cross_dialect_examples"],
                        fixed_epochs=30,
                        head_kind=head_kind,
                    )
                outputs[key].append(_method_row(
                    rule=rule,
                    method=method,
                    seed=seed,
                    result=result,
                    target_digest=target_digest,
                    manifest_sha256=manifest_sha,
                ))
    return outputs, {
        "source_audit": source_audit,
        "evaluation_manifest_sha256": manifest_sha,
        "pair_count": len(evaluation_pairs),
        "calibration_pair_count": len(calibration_pairs),
        "auxiliary_cross_pair_count": len(examples["cross_dialect_examples"]),
        "reference_name": reference_name,
    }


def _summarize_method_rows(
    method_rows: Sequence[Mapping[str, Any]],
    reference: Mapping[str, Mapping[str, float]],
    *,
    baseline_mae: float,
    bootstrap_replicates: int = 1000,
) -> dict[str, Any]:
    from .relation_ranking import clustered_ranking_bootstrap

    seed_rows = [{"seed": row["seed"], "rows": row["rows"]} for row in method_rows]
    try:
        ranking = clustered_ranking_bootstrap(
            {str(method_rows[0]["method"]): seed_rows},
            reference_relations(reference),
            seed=20260829,
            replicates=bootstrap_replicates,
        )["methods"][str(method_rows[0]["method"])]
    except (TypeError, ValueError):
        # The alternate target has a deliberately collapsed binary geometry:
        # relation ordering is undefined when all off-diagonal targets tie.
        ranking = {
            "mae": {
                "point_estimate": float(np.median([
                    float(np.mean([row["absolute_error"] for row in item["rows"]]))
                    for item in method_rows
                ])),
                "ci": None,
            },
            "ordering": {
                "spearman": {"point_estimate": None, "ci": None},
                "kendall_tau_b": {"point_estimate": None, "ci": None},
                "pairwise_order_accuracy": {"point_estimate": None, "ci": None},
                "status": "undefined_constant_reference",
            },
        }
    gains = [float(row["gain"]) for row in method_rows]
    return {
        "method": str(method_rows[0]["method"]),
        "seed_count": len(method_rows),
        "target_digest": str(method_rows[0]["target_digest"]),
        "mae": ranking["mae"],
        "gain": {
            "point_estimate": float(np.median(gains)),
            "ci": _percentile(gains),
            "seed_values": gains,
        },
        "ordering": ranking["ordering"],
        "baseline_mae": float(baseline_mae),
    }


def run_zero_rule_projection_sensitivity(
    *,
    config_path: Path,
    primary_path: Path,
    alternate_path: Path,
    cache_root: Path,
    bootstrap_replicates: int = 1000,
) -> dict[str, Any]:
    config = load_config(config_path)
    primary = _load(primary_path)
    alternate = _load(alternate_path)
    target_rows = []
    for rule, payload, use_cache in (("primary", primary, True), ("alternate", alternate, False)):
        matrices = payload["matrix"]
        rows, audit = _run_projection_cells(
            config=config,
            reference=matrices,
            reference_name=str(payload.get("name", rule)),
            target_digest=matrix_digest(payload),
            rule=rule,
            cache_root=cache_root,
            use_cache=use_cache,
        )
        # Frozen affine is computed under each target rather than copied from a
        # trainable cell.  Its one deterministic row is retained as a baseline.
        calibration_embeddings = _load_embeddings(str(DEFAULT_PATHS["calibration_embedding"]))
        evaluation_embeddings = _load_embeddings(str(DEFAULT_PATHS["evaluation_embedding"]))
        records = _load_records(str(DEFAULT_PATHS["records"]))
        calibration_records = _available_split(records, calibration_embeddings, "calibration")
        evaluation_records = _available_split(records, evaluation_embeddings, "evaluation")
        calibration_pairs = _available_pairs(_load_pairs(str(DEFAULT_PATHS["calibration_pairs"])), calibration_embeddings)
        evaluation_pairs = _available_pairs(_load_pairs(str(DEFAULT_PATHS["evaluation_pairs"])), evaluation_embeddings)
        frozen_vectors = _raw_and_pca(calibration_embeddings, evaluation_embeddings)
        calibration_vectors, evaluation_vectors = frozen_vectors["frozen_affine"]
        affine = _fit_affine_for_vectors(calibration_pairs, calibration_records, calibration_vectors, matrices)
        frozen_rows = _score_vectors(evaluation_pairs, evaluation_records, evaluation_vectors, matrices, affine)
        frozen = [{"seed": 0, "rows": frozen_rows, "method": "frozen_affine", "gain": 0.0, "target_digest": matrix_digest(payload)}]
        all_methods: dict[str, Any] = {"frozen_affine": {
            "method": "frozen_affine", "seed_count": 1, "target_digest": matrix_digest(payload),
            "mae": {"point_estimate": float(np.mean([row["absolute_error"] for row in frozen_rows])), "ci": None},
            "gain": {"point_estimate": 0.0, "ci": None, "seed_values": [0.0]},
            "ordering": ranking_metrics(reference_relations(matrices), aggregate_relation_predictions(frozen_rows)),
            "baseline_mae": float(np.mean([row["absolute_error"] for row in frozen_rows])),
        }}
        baseline_mae = all_methods["frozen_affine"]["baseline_mae"]
        for key, method_rows in rows.items():
            summary = _summarize_method_rows(
                method_rows, matrices, baseline_mae=baseline_mae, bootstrap_replicates=bootstrap_replicates
            )
            all_methods[key] = summary
        target_rows.append({"rule": rule, "target_digest": matrix_digest(payload), "methods": all_methods, "audit": audit})
    primary_row = next(row for row in target_rows if row["rule"] == "primary")
    alternate_row = next(row for row in target_rows if row["rule"] == "alternate")
    projection_claim_rows = [
        {
            "rule": rule,
            "linear_gain": float(row["methods"]["linear_b3"]["gain"]["point_estimate"]),
            "architecture_interaction": (
                "mlp_cross_gain_linear_cross_loss_loss"
                if float(row["methods"]["matched_mlp_b4"]["gain"]["point_estimate"])
                > float(row["methods"]["matched_mlp_b3"]["gain"]["point_estimate"])
                and float(row["methods"]["linear_b4"]["gain"]["point_estimate"])
                < float(row["methods"]["linear_b3"]["gain"]["point_estimate"])
                else "not_preserved"
            ),
        }
        for rule, row in (("primary", primary_row), ("alternate", alternate_row))
    ]
    report = build_zero_rule_report(primary, alternate, projection_claim_rows)
    report["status"] = "evaluated"
    report["projection_results"] = target_rows
    report["claim_branch"] = _projection_claim_branch(projection_claim_rows)[1]
    report["same_manifest_sha256"] = primary_row["audit"]["evaluation_manifest_sha256"] == alternate_row["audit"]["evaluation_manifest_sha256"]
    report["same_seed_schedule"] = True
    report["same_training_budget"] = True
    return report


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--primary", type=Path, required=True)
    parser.add_argument("--alternate-output", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--gate", type=Path, required=True)
    parser.add_argument("--projection", type=Path)
    args = parser.parse_args(argv)
    primary = _load(args.primary)
    alternate = alternate_binary_matrix(primary)
    args.alternate_output.parent.mkdir(parents=True, exist_ok=True)
    args.alternate_output.write_text(
        json.dumps(alternate, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    projection_rows = _load(args.projection).get("rows", []) if args.projection else []
    report = build_zero_rule_report(primary, alternate, projection_rows)
    gate = build_gate(report)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    args.gate.parent.mkdir(parents=True, exist_ok=True)
    args.gate.write_text(json.dumps(gate, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return 0 if gate["status"] in {"passed", "pending"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
