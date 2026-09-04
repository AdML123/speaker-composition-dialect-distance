"""Pair-count-matched cross-dialect diversity sweep."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from .config import load_config
from .cross_dialect_projection_head import (
    _apply_affine,
    _load_embedding_file,
    _load_pairs,
    _load_records,
    _raw_pair_rows,
    _fit_affine,
    build_pair_diversity_sweep_conditions,
    build_training_examples,
    paired_bootstrap_b4_minus_b3,
    score_pair_distances,
    summarize_cross_pool_diversity,
    train_projection_head,
    transform_embeddings,
    validate_pair_count_matched_conditions,
    _summarize_rows,
)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--calibration-embedding", action="append", required=True)
    parser.add_argument("--evaluation-embedding", action="append", required=True)
    parser.add_argument("--records", required=True)
    parser.add_argument("--calibration-pairs", required=True)
    parser.add_argument("--evaluation-pairs", required=True)
    parser.add_argument("--reference-matrix", action="append", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--analysis-output", required=True)
    parser.add_argument(
        "--architecture-control",
        action="store_true",
        help="run the matched trainable MLP-versus-linear architecture control",
    )
    return parser


def _load_reference(path: str) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    return payload.get("matrix", payload)


def apply_normalized_reference_targets(
    conditions: Mapping[str, Sequence[Mapping[str, object]]],
    reference: Mapping[str, Mapping[str, float]],
) -> dict[str, list[dict[str, object]]]:
    values = [
        float(value)
        for left in reference.values()
        for value in left.values()
    ]
    maximum = max(values) if values else 0.0
    if maximum <= 0.0:
        raise ValueError("reference maximum must be positive")
    updated: dict[str, list[dict[str, object]]] = {}
    for name, examples in conditions.items():
        rows: list[dict[str, object]] = []
        for example in examples:
            labels = list(example.get("dialect_labels", []))
            if len(labels) != 2:
                raise ValueError("each diversity example requires two dialect labels")
            left, right = map(str, labels)
            if left not in reference or right not in reference[left]:
                raise ValueError(f"unknown reference dialect pair: {left}, {right}")
            rows.append(
                {
                    **dict(example),
                    "target": float(reference[left][right]) / maximum,
                }
            )
        updated[name] = rows
    return updated


def _first_model_embeddings(paths: Sequence[str]) -> tuple[str, dict[str, list[float]]]:
    if not paths:
        raise ValueError("at least one calibration embedding is required")
    return _load_embedding_file(paths[0])


def run_pair_diversity_sweep(
    *,
    records: Sequence[Mapping[str, object]],
    calibration_pairs: Sequence[Mapping[str, object]],
    reference: Mapping[str, Mapping[str, float]],
    seed: int,
) -> dict[str, object]:
    examples = build_training_examples(records, calibration_pairs, reference)
    base_examples = list(examples["cross_dialect_examples"])
    conditions = build_pair_diversity_sweep_conditions(records, base_examples, seed=seed)
    conditions = apply_normalized_reference_targets(conditions, reference)
    validate_pair_count_matched_conditions(conditions)
    summaries = {
        name: summarize_cross_pool_diversity(condition_examples)
        for name, condition_examples in conditions.items()
    }
    return {
        "schema": "cross-dialect-pair-diversity-sweep-v1",
        "seed": seed,
        "status": "diagnostic_only",
        "conditions": {
            name: {
                "pair_count": summaries[name]["pair_count"],
                "diversity_metrics": summaries[name],
                "infeasible": summaries[name]["pair_count"] == 0,
            }
            for name in conditions
        },
        "diversity_metrics": summaries,
    }


def _baseline_rows(
    calibration_pairs: Sequence[Mapping[str, object]],
    evaluation_pairs: Sequence[Mapping[str, object]],
    calibration_embeddings: Mapping[str, Sequence[float]],
    evaluation_embeddings: Mapping[str, Sequence[float]],
    reference: Mapping[str, Mapping[str, float]],
    record_index: Mapping[str, Mapping[str, object]],
) -> list[dict[str, object]]:
    calibration_rows = _raw_pair_rows(
        calibration_pairs,
        calibration_embeddings,
        reference,
        record_index,
    )
    affine = _fit_affine(
        [float(row["raw_distance"]) for row in calibration_rows],
        [float(row["target"]) for row in calibration_rows],
    )
    return [
        {
            **row,
            "distance": float(_apply_affine([float(row["raw_distance"])], affine)[0]),
        }
        for row in _raw_pair_rows(
            evaluation_pairs,
            evaluation_embeddings,
            reference,
            record_index,
        )
    ]


def _selected_hyperparameters(config: Mapping[str, object]) -> dict[str, float]:
    head_config = config.get("projection_head", config)
    if not isinstance(head_config, Mapping):
        raise ValueError("projection_head configuration is required")
    return {
        "lambda_cross": float(list(head_config["lambda_cross_grid"])[0]),
        "lambda_dialect": float(list(head_config["lambda_dialect_grid"])[0]),
        "learning_rate": float(list(head_config["learning_rate_grid"])[0]),
        "weight_decay": float(list(head_config["weight_decay_grid"])[0]),
    }


def _fit_and_score(
    *,
    calibration_embeddings: Mapping[str, Sequence[float]],
    evaluation_embeddings: Mapping[str, Sequence[float]],
    calibration_records: Sequence[Mapping[str, object]],
    evaluation_records: Sequence[Mapping[str, object]],
    calibration_pairs: Sequence[Mapping[str, object]],
    evaluation_pairs: Sequence[Mapping[str, object]],
    reference: Mapping[str, Mapping[str, float]],
    record_index: Mapping[str, Mapping[str, object]],
    baseline_rows: Sequence[Mapping[str, object]],
    config: Mapping[str, object],
    seed: int,
    selected: Mapping[str, float],
    cross_examples: Sequence[Mapping[str, object]],
    lambda_cross: float,
) -> dict[str, object]:
    fitted = train_projection_head(
        calibration_embeddings,
        calibration_records,
        calibration_pairs,
        reference,
        lambda_cross=lambda_cross,
        lambda_dialect=float(selected["lambda_dialect"]),
        learning_rate=float(selected["learning_rate"]),
        weight_decay=float(selected["weight_decay"]),
        config=config,
        seed=seed,
        validation_embeddings=calibration_embeddings,
        validation_records=calibration_records,
        validation_pairs=calibration_pairs,
        cross_examples_override=cross_examples,
    )
    transformed = transform_embeddings(evaluation_embeddings, fitted)
    rows = score_pair_distances(
        evaluation_pairs,
        transformed,
        reference,
        fitted["affine_scale"],
        record_index,
    )
    head_config = config.get("projection_head", config)
    summary = _summarize_rows(
        rows,
        baseline_rows,
        seed,
        int(head_config.get("bootstrap_replicates", 1000)),  # type: ignore[union-attr]
    )
    return {
        "summary": summary,
        "selected_epoch": fitted["selected_epoch"],
        "same_speaker_cross_dialect_count": fitted["same_speaker_cross_dialect_count"],
    }


def run_pair_diversity_sweep_training(
    *,
    calibration_embeddings: Mapping[str, Sequence[float]],
    evaluation_embeddings: Mapping[str, Sequence[float]],
    records: Sequence[Mapping[str, object]],
    calibration_pairs: Sequence[Mapping[str, object]],
    evaluation_pairs: Sequence[Mapping[str, object]],
    reference: Mapping[str, Mapping[str, float]],
    config: Mapping[str, object],
    seeds: Sequence[int],
) -> dict[str, object]:
    calibration_records = [
        record for record in records if str(record.get("split")) == "calibration"
        and str(record["utterance_id"]) in calibration_embeddings
    ]
    evaluation_records = [
        record for record in records if str(record.get("split")) == "evaluation"
        and str(record["utterance_id"]) in evaluation_embeddings
    ]
    record_index = {
        str(record["utterance_id"]): record
        for record in calibration_records + evaluation_records
    }
    usable_calibration_pairs = [
        pair for pair in calibration_pairs
        if set(map(str, pair["source_utterance_ids"])) <= set(calibration_embeddings)
    ]
    usable_evaluation_pairs = [
        pair for pair in evaluation_pairs
        if set(map(str, pair["source_utterance_ids"])) <= set(evaluation_embeddings)
    ]
    base_examples = build_training_examples(
        calibration_records,
        usable_calibration_pairs,
        reference,
    )
    conditions = apply_normalized_reference_targets(
        build_pair_diversity_sweep_conditions(
            calibration_records,
            base_examples["cross_dialect_examples"],
            seed=int(seeds[0]),
        ),
        reference,
    )
    validate_pair_count_matched_conditions(conditions)
    if any(not rows for rows in conditions.values()):
        raise ValueError("all pair-count-matched diversity conditions require pair support")
    baseline_rows = _baseline_rows(
        usable_calibration_pairs,
        usable_evaluation_pairs,
        calibration_embeddings,
        evaluation_embeddings,
        reference,
        record_index,
    )
    selected = _selected_hyperparameters(config)
    condition_results: dict[str, dict[str, object]] = {
        name: {
            "diversity_metrics": summarize_cross_pool_diversity(rows),
            "seed_results": [],
        }
        for name, rows in conditions.items()
    }
    for seed in seeds:
        pair_only = _fit_and_score(
            calibration_embeddings=calibration_embeddings,
            evaluation_embeddings=evaluation_embeddings,
            calibration_records=calibration_records,
            evaluation_records=evaluation_records,
            calibration_pairs=usable_calibration_pairs,
            evaluation_pairs=usable_evaluation_pairs,
            reference=reference,
            record_index=record_index,
            baseline_rows=baseline_rows,
            config=config,
            seed=int(seed),
            selected=selected,
            cross_examples=[],
            lambda_cross=0.0,
        )
        pair_only_gain = float(pair_only["summary"]["improvement_ratio"])  # type: ignore[index]
        for index, (name, examples) in enumerate(conditions.items()):
            fitted = _fit_and_score(
                calibration_embeddings=calibration_embeddings,
                evaluation_embeddings=evaluation_embeddings,
                calibration_records=calibration_records,
                evaluation_records=evaluation_records,
                calibration_pairs=usable_calibration_pairs,
                evaluation_pairs=usable_evaluation_pairs,
                reference=reference,
                record_index=record_index,
                baseline_rows=baseline_rows,
                config=config,
                seed=int(seed) + 10 + index,
                selected=selected,
                cross_examples=examples,
                lambda_cross=float(selected["lambda_cross"]),
            )
            summary = fitted["summary"]
            condition_results[name]["seed_results"].append(  # type: ignore[index]
                {
                    "seed": int(seed),
                    "mae": float(summary["mae"]),  # type: ignore[index]
                    "gain_over_b0": float(summary["improvement_ratio"]),  # type: ignore[index]
                    "gain_over_b3": float(summary["improvement_ratio"]) - pair_only_gain,  # type: ignore[index]
                    "pair_only_gain_over_b0": pair_only_gain,
                    "per_pair": summary["per_pair"],  # type: ignore[index]
                }
            )
    coverage = condition_results["coverage_matched_shuffled"]["seed_results"]  # type: ignore[index]
    broadest = condition_results["speaker_and_dialect_broadened_shuffled"]["seed_results"]  # type: ignore[index]
    paired_by_seed = [
        paired_bootstrap_b4_minus_b3(
            broad["per_pair"],
            matched["per_pair"],
            seed=int(broad["seed"]),
            replicates=int(config.get("projection_head", config).get("bootstrap_replicates", 1000)),  # type: ignore[union-attr]
        )
        for broad, matched in zip(broadest, coverage)
    ]
    directions = [
        float(broad["gain_over_b0"]) > float(matched["gain_over_b0"])
        for broad, matched in zip(broadest, coverage)
    ]
    aggregate = paired_bootstrap_b4_minus_b3(
        broadest[0]["per_pair"],
        coverage[0]["per_pair"],
        seed=int(seeds[0]),
        replicates=int(config.get("projection_head", config).get("bootstrap_replicates", 1000)),  # type: ignore[union-attr]
    )
    broad_metrics = condition_results["speaker_and_dialect_broadened_shuffled"]["diversity_metrics"]  # type: ignore[index]
    coverage_metrics = condition_results["coverage_matched_shuffled"]["diversity_metrics"]  # type: ignore[index]
    criteria = {
        "equal_pair_count": all(
            int(result["diversity_metrics"]["pair_count"]) == int(coverage_metrics["pair_count"])  # type: ignore[index]
            for result in condition_results.values()
        ),
        "dialect_pair_coverage_increased": (
            int(broad_metrics["unique_dialect_pair_count"])
            > int(coverage_metrics["unique_dialect_pair_count"])
        ),
        "paired_ci_lower_positive": bool(aggregate["passed"]),
        "same_direction_seed_count": sum(directions),
    }
    passed = (
        criteria["equal_pair_count"]
        and criteria["dialect_pair_coverage_increased"]
        and criteria["paired_ci_lower_positive"]
        and criteria["same_direction_seed_count"] >= 4
    )
    for condition in condition_results.values():
        for seed_result in condition["seed_results"]:  # type: ignore[index]
            seed_result.pop("per_pair", None)
    return {
        "schema": "cross-dialect-pair-diversity-sweep-v2",
        "status": "passed" if passed else "failed",
        "selected_hyperparameters": selected,
        "conditions": condition_results,
        "comparison": {
            "broadest_vs_coverage_matched": aggregate,
            "per_seed": paired_by_seed,
            "criteria": criteria,
        },
    }


def _fit_full_head_for_architecture(
    *,
    calibration_embeddings: Mapping[str, Sequence[float]],
    evaluation_embeddings: Mapping[str, Sequence[float]],
    calibration_records: Sequence[Mapping[str, object]],
    calibration_pairs: Sequence[Mapping[str, object]],
    evaluation_pairs: Sequence[Mapping[str, object]],
    reference: Mapping[str, Mapping[str, float]],
    record_index: Mapping[str, Mapping[str, object]],
    baseline_rows: Sequence[Mapping[str, object]],
    config: Mapping[str, object],
    seed: int,
    selected: Mapping[str, float],
    head_kind: str,
) -> dict[str, object]:
    fitted = train_projection_head(
        calibration_embeddings,
        calibration_records,
        calibration_pairs,
        reference,
        lambda_cross=float(selected["lambda_cross"]),
        lambda_dialect=float(selected["lambda_dialect"]),
        learning_rate=float(selected["learning_rate"]),
        weight_decay=float(selected["weight_decay"]),
        config=config,
        seed=seed,
        validation_embeddings=calibration_embeddings,
        validation_records=calibration_records,
        validation_pairs=calibration_pairs,
        head_kind=head_kind,
    )
    transformed = transform_embeddings(evaluation_embeddings, fitted)
    rows = score_pair_distances(
        evaluation_pairs,
        transformed,
        reference,
        fitted["affine_scale"],
        record_index,
    )
    head_config = config.get("projection_head", config)
    summary = _summarize_rows(
        rows,
        baseline_rows,
        seed,
        int(head_config.get("bootstrap_replicates", 1000)),  # type: ignore[union-attr]
    )
    return {
        "mae": summary["mae"],
        "improvement_ratio": summary["improvement_ratio"],
        "selected_epoch": fitted["selected_epoch"],
        "per_pair": summary["per_pair"],
    }


def run_trainable_head_architecture_control(
    *,
    calibration_embeddings: Mapping[str, Sequence[float]],
    evaluation_embeddings: Mapping[str, Sequence[float]],
    records: Sequence[Mapping[str, object]],
    calibration_pairs: Sequence[Mapping[str, object]],
    evaluation_pairs: Sequence[Mapping[str, object]],
    reference: Mapping[str, Mapping[str, float]],
    config: Mapping[str, object],
    seeds: Sequence[int],
) -> dict[str, object]:
    calibration_records = [
        record for record in records if str(record.get("split")) == "calibration"
        and str(record["utterance_id"]) in calibration_embeddings
    ]
    evaluation_records = [
        record for record in records if str(record.get("split")) == "evaluation"
        and str(record["utterance_id"]) in evaluation_embeddings
    ]
    record_index = {
        str(record["utterance_id"]): record
        for record in calibration_records + evaluation_records
    }
    usable_calibration_pairs = [
        pair for pair in calibration_pairs
        if set(map(str, pair["source_utterance_ids"])) <= set(calibration_embeddings)
    ]
    usable_evaluation_pairs = [
        pair for pair in evaluation_pairs
        if set(map(str, pair["source_utterance_ids"])) <= set(evaluation_embeddings)
    ]
    baseline_rows = _baseline_rows(
        usable_calibration_pairs,
        usable_evaluation_pairs,
        calibration_embeddings,
        evaluation_embeddings,
        reference,
        record_index,
    )
    selected = _selected_hyperparameters(config)
    seed_results = []
    paired_results = []
    for seed in seeds:
        mlp = _fit_full_head_for_architecture(
            calibration_embeddings=calibration_embeddings,
            evaluation_embeddings=evaluation_embeddings,
            calibration_records=calibration_records,
            calibration_pairs=usable_calibration_pairs,
            evaluation_pairs=usable_evaluation_pairs,
            reference=reference,
            record_index=record_index,
            baseline_rows=baseline_rows,
            config=config,
            seed=int(seed),
            selected=selected,
            head_kind="mlp",
        )
        linear = _fit_full_head_for_architecture(
            calibration_embeddings=calibration_embeddings,
            evaluation_embeddings=evaluation_embeddings,
            calibration_records=calibration_records,
            calibration_pairs=usable_calibration_pairs,
            evaluation_pairs=usable_evaluation_pairs,
            reference=reference,
            record_index=record_index,
            baseline_rows=baseline_rows,
            config=config,
            seed=int(seed),
            selected=selected,
            head_kind="linear",
        )
        paired = paired_bootstrap_b4_minus_b3(
            mlp["per_pair"],
            linear["per_pair"],
            seed=int(seed),
            replicates=int(config.get("projection_head", config).get("bootstrap_replicates", 1000)),  # type: ignore[union-attr]
        )
        paired_results.append(paired)
        seed_results.append(
            {
                "seed": int(seed),
                "heads": {
                    "mlp": {
                        "mae": float(mlp["mae"]),
                        "improvement_ratio": float(mlp["improvement_ratio"]),
                        "selected_epoch": int(mlp["selected_epoch"]),
                    },
                    "linear": {
                        "mae": float(linear["mae"]),
                        "improvement_ratio": float(linear["improvement_ratio"]),
                        "selected_epoch": int(linear["selected_epoch"]),
                    },
                },
                "mlp_minus_linear_improvement": (
                    float(mlp["improvement_ratio"])
                    - float(linear["improvement_ratio"])
                ),
                "paired_mlp_vs_linear": paired,
            }
        )
    primary = paired_results[0]
    seeds_mlp_over_linear = sum(
        result["mlp_minus_linear_improvement"] > 0
        for result in seed_results
    )
    passed = bool(primary["passed"]) and seeds_mlp_over_linear >= 4
    return {
        "schema": "trainable-head-architecture-control-v1",
        "status": "passed" if passed else "failed",
        "selected_hyperparameters": selected,
        "criteria": {
            "primary_seed_paired_ci_lower_positive": bool(primary["passed"]),
            "seeds_mlp_over_linear": seeds_mlp_over_linear,
            "min_seeds_mlp_over_linear": 4,
        },
        "comparison": primary,
        "seed_results": seed_results,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    config = load_config(args.config)
    calibration_model, calibration_embeddings = _first_model_embeddings(args.calibration_embedding)
    evaluation_model, evaluation_embeddings = _first_model_embeddings(args.evaluation_embedding)
    if calibration_model != evaluation_model:
        raise ValueError("calibration and evaluation model names differ")
    records = _load_records(args.records)
    calibration_pairs = _load_pairs(args.calibration_pairs)
    evaluation_pairs = _load_pairs(args.evaluation_pairs)
    reference = _load_reference(args.reference_matrix[0])
    seed_sweep = list(config["projection_head"]["seed_sweep"])
    if args.architecture_control:
        report = run_trainable_head_architecture_control(
            calibration_embeddings=calibration_embeddings,
            evaluation_embeddings=evaluation_embeddings,
            records=records,
            calibration_pairs=calibration_pairs,
            evaluation_pairs=evaluation_pairs,
            reference=reference,
            config=config,
            seeds=[int(seed) for seed in seed_sweep],
        )
    else:
        report = run_pair_diversity_sweep_training(
            calibration_embeddings=calibration_embeddings,
            evaluation_embeddings=evaluation_embeddings,
            records=records,
            calibration_pairs=calibration_pairs,
            evaluation_pairs=evaluation_pairs,
            reference=reference,
            config=config,
            seeds=[int(seed) for seed in seed_sweep],
        )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")
    analysis_output = Path(args.analysis_output)
    analysis_output.parent.mkdir(parents=True, exist_ok=True)
    analysis_output.write_text(json.dumps(report, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
