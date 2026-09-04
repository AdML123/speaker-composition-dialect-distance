"""Diagnostics for the nonzero-target gradient-isolation hypothesis."""

from __future__ import annotations

from dataclasses import dataclass
import argparse
import json
from pathlib import Path
import random
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from torch import nn


LAMBDA_CROSS_DOSE_RESPONSE = [0.0, 0.05, 0.1, 0.25, 0.5, 1.0, 2.0, 5.0]


@dataclass(frozen=True)
class IndependentLossPool:
    """Deterministically constructed pool and its sampling audit."""

    examples: tuple[dict[str, object], ...]
    requested_count: int
    achieved_count: int
    target_histogram: dict[float, int]
    feasible: bool
    seed: int


def build_lambda_dose_response_grid() -> list[float]:
    return list(LAMBDA_CROSS_DOSE_RESPONSE)


def _sorted_examples(examples: Sequence[Mapping[str, object]]) -> list[dict[str, object]]:
    return sorted(
        [dict(example) for example in examples],
        key=lambda item: str(item.get("pair_id", "")),
    )


def _sample_examples(
    candidates: Sequence[Mapping[str, object]],
    requested_count: int,
    seed: int,
) -> tuple[dict[str, object], ...]:
    if requested_count < 0:
        raise ValueError("requested_count must be non-negative")
    ordered = _sorted_examples(candidates)
    count = min(requested_count, len(ordered))
    indices = sorted(random.Random(seed).sample(range(len(ordered)), count))
    return tuple(ordered[index] for index in indices)


def _histogram(examples: Sequence[Mapping[str, object]]) -> dict[float, int]:
    counts: dict[float, int] = {}
    for example in examples:
        target = float(example["target"])
        counts[target] = counts.get(target, 0) + 1
    return {key: counts[key] for key in sorted(counts)}


def build_same_dialect_independent_pool(
    generic_examples: Sequence[Mapping[str, object]],
    *,
    requested_count: int,
    seed: int,
) -> IndependentLossPool:
    candidates = [
        example
        for example in generic_examples
        if float(example["target"]) == 0.0
        and len(example.get("dialect_labels", [])) == 2
        and str(example["dialect_labels"][0]) == str(example["dialect_labels"][1])
    ]
    sampled = _sample_examples(candidates, requested_count, seed)
    return IndependentLossPool(
        examples=sampled,
        requested_count=requested_count,
        achieved_count=len(sampled),
        target_histogram=_histogram(sampled),
        feasible=len(sampled) == requested_count,
        seed=seed,
    )


def build_target_matched_random_pool(
    generic_examples: Sequence[Mapping[str, object]],
    *,
    requested_count: int,
    seed: int,
) -> IndependentLossPool:
    """Sample a deterministic pool while preserving the generic target histogram."""
    if requested_count < 0:
        raise ValueError("requested_count must be non-negative")
    ordered = _sorted_examples(generic_examples)
    achieved = min(requested_count, len(ordered))
    if achieved == 0:
        sampled = tuple()
    else:
        # Allocate the requested count across target strata by largest
        # remainder, then sample within each stratum with deterministic
        # per-stratum streams. This preserves the empirical histogram as
        # closely as the available pool permits.
        by_target: dict[float, list[dict[str, object]]] = {}
        for example in ordered:
            by_target.setdefault(float(example["target"]), []).append(example)
        total = len(ordered)
        allocations = {
            target: min(len(items), int(np.floor(achieved * len(items) / total)))
            for target, items in by_target.items()
        }
        remaining = achieved - sum(allocations.values())
        ranked = sorted(
            by_target,
            key=lambda target: (
                -(achieved * len(by_target[target]) / total - allocations[target]),
                target,
            ),
        )
        for target in ranked:
            if remaining <= 0:
                break
            if allocations[target] < len(by_target[target]):
                allocations[target] += 1
                remaining -= 1
        selected: list[dict[str, object]] = []
        for index, target in enumerate(sorted(by_target)):
            items = by_target[target]
            selected.extend(
                _sample_examples(items, allocations[target], seed + index)
            )
        sampled = tuple(sorted(selected, key=lambda item: str(item.get("pair_id", ""))))
    return IndependentLossPool(
        examples=sampled,
        requested_count=requested_count,
        achieved_count=len(sampled),
        target_histogram=_histogram(sampled),
        feasible=len(sampled) == requested_count,
        seed=seed,
    )


def _flatten_gradients(model: nn.Module) -> torch.Tensor:
    values = []
    for parameter in model.parameters():
        if parameter.grad is None:
            values.append(torch.zeros_like(parameter).reshape(-1))
        else:
            values.append(parameter.grad.detach().reshape(-1))
    return torch.cat(values) if values else torch.empty(0)


def _loss_gradient(
    model: nn.Module,
    inputs: torch.Tensor,
    targets: torch.Tensor,
) -> torch.Tensor:
    model.zero_grad(set_to_none=True)
    if inputs.numel() == 0:
        return torch.zeros(sum(parameter.numel() for parameter in model.parameters()))
    outputs = model(inputs)
    if outputs.ndim > 1:
        outputs = outputs[:, 0]
    outputs = outputs.reshape(-1)
    loss = nn.SmoothL1Loss()(outputs, targets.reshape(-1))
    loss.backward()
    return _flatten_gradients(model).cpu()


def _cosine(left: torch.Tensor, right: torch.Tensor) -> float:
    if left.numel() == 0 or right.numel() == 0:
        return float("nan")
    left_norm = torch.linalg.vector_norm(left)
    right_norm = torch.linalg.vector_norm(right)
    if float(left_norm) == 0.0 or float(right_norm) == 0.0:
        return 0.0
    return float(torch.dot(left, right) / (left_norm * right_norm))


def compute_gradient_probe_metrics(
    model: nn.Module,
    pair_inputs: torch.Tensor,
    pair_targets: torch.Tensor,
    cross_inputs: torch.Tensor,
    cross_targets: torch.Tensor,
    *,
    mixed_inputs: torch.Tensor | None = None,
    mixed_targets: torch.Tensor | None = None,
) -> dict[str, float | int]:
    """Compute raw and per-example normalized stratum gradients.

    The mixed nonzero stratum is evaluated separately, so its norm is not
    confounded with the number of zero-target examples in the mixed pool.
    """
    if mixed_inputs is None:
        mixed_inputs = cross_inputs
    if mixed_targets is None:
        mixed_targets = cross_targets
    g_pair = _loss_gradient(model, pair_inputs, pair_targets)
    g_cross = _loss_gradient(model, cross_inputs, cross_targets)
    nonzero = mixed_targets.reshape(-1) != 0
    g_mixed_nonzero = _loss_gradient(model, mixed_inputs[nonzero], mixed_targets[nonzero])
    g_mixed_total = _loss_gradient(model, mixed_inputs, mixed_targets)
    pair_count = int(pair_targets.numel())
    cross_count = int(cross_targets.numel())
    mixed_nonzero_count = int(nonzero.sum().item())
    return {
        "pair_count": pair_count,
        "cross_count": cross_count,
        "mixed_nonzero_count": mixed_nonzero_count,
        "pair_norm": float(torch.linalg.vector_norm(g_pair)),
        "cross_norm": float(torch.linalg.vector_norm(g_cross)),
        "mixed_nonzero_norm": float(torch.linalg.vector_norm(g_mixed_nonzero)),
        "mixed_total_norm": float(torch.linalg.vector_norm(g_mixed_total)),
        "pair_norm_per_example": float(torch.linalg.vector_norm(g_pair) / max(pair_count, 1)),
        "cross_norm_per_example": float(torch.linalg.vector_norm(g_cross) / max(cross_count, 1)),
        "mixed_nonzero_norm_per_example": float(
            torch.linalg.vector_norm(g_mixed_nonzero) / max(mixed_nonzero_count, 1)
        ),
        "mixed_total_norm_per_example": float(
            torch.linalg.vector_norm(g_mixed_total) / max(int(mixed_targets.numel()), 1)
        ),
        "mixed_to_cross_normalized_ratio": float(
            (torch.linalg.vector_norm(g_mixed_nonzero) / max(mixed_nonzero_count, 1))
            / max(float(torch.linalg.vector_norm(g_cross) / max(cross_count, 1)), 1e-12)
        ),
        "pair_cross_cosine": _cosine(g_pair, g_cross),
        "pair_mixed_nonzero_cosine": _cosine(g_pair, g_mixed_nonzero),
    }


def _distance_gradient(
    model: nn.Module,
    left: torch.Tensor,
    right: torch.Tensor,
    targets: torch.Tensor,
) -> torch.Tensor:
    model.zero_grad(set_to_none=True)
    if targets.numel() == 0:
        return torch.zeros(sum(parameter.numel() for parameter in model.parameters()))
    z_left = nn.functional.normalize(model(left), p=2, dim=1)
    z_right = nn.functional.normalize(model(right), p=2, dim=1)
    distances = 1.0 - nn.functional.cosine_similarity(z_left, z_right, dim=1)
    loss = nn.SmoothL1Loss()(distances, targets.reshape(-1))
    loss.backward()
    return _flatten_gradients(model).cpu()


def compute_distance_gradient_probe_metrics(
    model: nn.Module,
    pair_left: torch.Tensor,
    pair_right: torch.Tensor,
    pair_targets: torch.Tensor,
    cross_left: torch.Tensor,
    cross_right: torch.Tensor,
    cross_targets: torch.Tensor,
    *,
    mixed_left: torch.Tensor | None = None,
    mixed_right: torch.Tensor | None = None,
    mixed_targets: torch.Tensor | None = None,
) -> dict[str, float | int]:
    """Probe gradients of the actual cosine-distance objective."""
    if mixed_left is None:
        mixed_left = cross_left
    if mixed_right is None:
        mixed_right = cross_right
    if mixed_targets is None:
        mixed_targets = cross_targets
    g_pair = _distance_gradient(model, pair_left, pair_right, pair_targets)
    g_cross = _distance_gradient(model, cross_left, cross_right, cross_targets)
    nonzero = mixed_targets.reshape(-1) != 0
    g_mixed_nonzero = _distance_gradient(
        model,
        mixed_left[nonzero],
        mixed_right[nonzero],
        mixed_targets[nonzero],
    )
    g_mixed_total = _distance_gradient(model, mixed_left, mixed_right, mixed_targets)
    pair_count = int(pair_targets.numel())
    cross_count = int(cross_targets.numel())
    mixed_nonzero_count = int(nonzero.sum().item())
    pair_norm = torch.linalg.vector_norm(g_pair)
    cross_norm = torch.linalg.vector_norm(g_cross)
    mixed_norm = torch.linalg.vector_norm(g_mixed_nonzero)
    pair_per = pair_norm / max(pair_count, 1)
    cross_per = cross_norm / max(cross_count, 1)
    mixed_per = mixed_norm / max(mixed_nonzero_count, 1)
    return {
        "pair_count": pair_count,
        "cross_count": cross_count,
        "mixed_nonzero_count": mixed_nonzero_count,
        "pair_norm": float(pair_norm),
        "cross_norm": float(cross_norm),
        "mixed_nonzero_norm": float(mixed_norm),
        "mixed_total_norm": float(torch.linalg.vector_norm(g_mixed_total)),
        "pair_norm_per_example": float(pair_per),
        "cross_norm_per_example": float(cross_per),
        "mixed_nonzero_norm_per_example": float(mixed_per),
        "mixed_total_norm_per_example": float(
            torch.linalg.vector_norm(g_mixed_total) / max(int(mixed_targets.numel()), 1)
        ),
        "mixed_to_cross_normalized_ratio": float(mixed_per / max(float(cross_per), 1e-12)),
        "pair_cross_cosine": _cosine(g_pair, g_cross),
        "pair_mixed_nonzero_cosine": _cosine(g_pair, g_mixed_nonzero),
    }


def aggregate_gradient_isolation_gate(
    *,
    dose_response: Mapping[str, object],
    independent_loss_control: Mapping[str, object],
    gradient_probe: Mapping[str, object],
) -> dict[str, object]:
    def _passed(report: Mapping[str, object]) -> bool:
        if "passed" in report:
            return bool(report["passed"])
        return str(report.get("status", "")).lower() == "passed"

    parts = {
        "dose_response": _passed(dose_response),
        "independent_loss_control": _passed(independent_loss_control),
        "gradient_probe": _passed(gradient_probe),
    }
    return {
        "schema": "cross-dialect-gradient-isolation-gate-v1",
        "status": "passed" if all(parts.values()) else "failed",
        "criteria": parts,
        "dose_response": dict(dose_response),
        "independent_loss_control": dict(independent_loss_control),
        "gradient_probe": dict(gradient_probe),
    }


def _load_json(path: str) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _load_embedding(path: str) -> dict[str, list[float]]:
    payload = _load_json(path)
    return payload.get("embeddings", payload)


def _load_records(path: str) -> list[dict[str, object]]:
    payload = _load_json(path)
    return list(payload.get("records", payload))


def _load_pairs(path: str) -> list[dict[str, object]]:
    payload = _load_json(path)
    return list(payload.get("pairs", payload))


def _load_reference(path: str) -> dict[str, dict[str, float]]:
    payload = _load_json(path)
    return payload.get("matrix", payload)


def _run_training_condition(
    *,
    calibration_embeddings: Mapping[str, Sequence[float]],
    evaluation_embeddings: Mapping[str, Sequence[float]],
    calibration_records: Sequence[Mapping[str, object]],
    evaluation_records: Sequence[Mapping[str, object]],
    calibration_pairs: Sequence[Mapping[str, object]],
    evaluation_pairs: Sequence[Mapping[str, object]],
    reference: Mapping[str, Mapping[str, float]],
    config: Mapping[str, object],
    seed: int,
    lambda_cross: float,
    cross_examples: Sequence[Mapping[str, object]],
    pair_examples: Sequence[Mapping[str, object]] | None = None,
    cross_loss_mode: str = "ordinary",
    fixed_epochs: int | None = None,
    aggregation_mode: str = "separate",
    exposure_ratio: tuple[int, int] | None = None,
    regularization_strength: float = 0.0,
    replace_cross_with_regularization: bool = False,
    record_gradient_budget: bool = False,
    head_kind: str = "mlp",
) -> dict[str, object]:
    from src.cross_dialect_pair_diversity_sweep import _baseline_rows, _summarize_rows
    from src.cross_dialect_projection_head import (
        _fit_affine,
        _apply_affine,
        _raw_pair_rows,
        train_projection_head,
        transform_embeddings,
        score_pair_distances,
    )

    record_index = {
        str(record["utterance_id"]): record
        for record in list(calibration_records) + list(evaluation_records)
    }
    baseline_rows = _baseline_rows(
        calibration_pairs,
        evaluation_pairs,
        calibration_embeddings,
        evaluation_embeddings,
        reference,
        record_index,
    )
    head_config = config.get("projection_head", config)
    fitted = train_projection_head(
        calibration_embeddings,
        calibration_records,
        calibration_pairs,
        reference,
        lambda_cross=lambda_cross,
        lambda_dialect=float(head_config.get("lambda_dialect_grid", [0.05])[0]),
        learning_rate=float(head_config.get("learning_rate_grid", [0.0003])[0]),
        weight_decay=float(head_config.get("weight_decay_grid", [0.0001])[0]),
        config=config,
        seed=seed,
        validation_embeddings=calibration_embeddings,
        validation_records=calibration_records,
        validation_pairs=calibration_pairs,
        pair_examples_override=pair_examples,
        cross_examples_override=cross_examples,
        cross_loss_mode=cross_loss_mode,
        fixed_epochs=fixed_epochs,
        aggregation_mode=aggregation_mode,
        exposure_ratio=exposure_ratio,
        regularization_strength=regularization_strength,
        replace_cross_with_regularization=replace_cross_with_regularization,
        record_gradient_budget=record_gradient_budget,
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
    summary = _summarize_rows(
        rows,
        baseline_rows,
        seed,
        int(head_config.get("bootstrap_replicates", 1000)),
    )
    return {
        "mae": float(summary["mae"]),
        "calibration_mae": float(fitted["loss_history"][-1]["validation_mae"]),
        "improvement_ratio": float(summary["improvement_ratio"]),
        "per_pair": summary["per_pair"],
        "baseline_per_pair": [
            {
                **dict(row),
                "absolute_error": abs(float(row["distance"]) - float(row["target"])),
            }
            for row in baseline_rows
        ],
        "fitted": fitted,
        "cross_loss_mode": cross_loss_mode,
        "aggregation_mode": aggregation_mode,
    }


def _paired_improvement(
    candidate: Mapping[str, object],
    baseline: Mapping[str, object],
    *,
    seed: int,
    replicates: int,
) -> dict[str, object]:
    from src.cross_dialect_projection_head import paired_bootstrap_b4_minus_b3

    return paired_bootstrap_b4_minus_b3(
        candidate["per_pair"],  # type: ignore[arg-type]
        baseline["per_pair"],  # type: ignore[arg-type]
        seed=seed,
        replicates=replicates,
    )


def run_gradient_isolation_experiments(
    *,
    calibration_embeddings: Mapping[str, Sequence[float]],
    evaluation_embeddings: Mapping[str, Sequence[float]],
    records: Sequence[Mapping[str, object]],
    calibration_pairs: Sequence[Mapping[str, object]],
    evaluation_pairs: Sequence[Mapping[str, object]],
    reference: Mapping[str, Mapping[str, float]],
    config: Mapping[str, object],
) -> dict[str, object]:
    from src.cross_dialect_projection_head import (
        _make_projection_model,
        _pair_tensors,
        build_training_examples,
        build_same_count_no_cross_control,
        fit_standardizer,
    )

    calibration_records = [
        record for record in records
        if str(record.get("split")) == "calibration"
        and str(record["utterance_id"]) in calibration_embeddings
    ]
    evaluation_records = [
        record for record in records
        if str(record.get("split")) == "evaluation"
        and str(record["utterance_id"]) in evaluation_embeddings
    ]
    calibration_pairs = [
        pair for pair in calibration_pairs
        if set(map(str, pair["source_utterance_ids"])) <= set(calibration_embeddings)
    ]
    evaluation_pairs = [
        pair for pair in evaluation_pairs
        if set(map(str, pair["source_utterance_ids"])) <= set(evaluation_embeddings)
    ]
    base = build_training_examples(calibration_records, calibration_pairs, reference)
    generic = list(base["pair_examples"])
    cross = list(base["cross_dialect_examples"])
    isolation = config.get("gradient_isolation", {})
    seeds = [int(seed) for seed in isolation.get(
        "seeds", config.get("projection_head", {}).get("seed_sweep", [20260829])
    )]
    replicates = int(config.get("projection_head", {}).get("bootstrap_replicates", 1000))
    dose_rows: dict[str, dict[str, object]] = {}
    for lambda_cross in build_lambda_dose_response_grid():
        rows = []
        for seed in seeds:
            result = _run_training_condition(
                calibration_embeddings=calibration_embeddings,
                evaluation_embeddings=evaluation_embeddings,
                calibration_records=calibration_records,
                evaluation_records=evaluation_records,
                calibration_pairs=calibration_pairs,
                evaluation_pairs=evaluation_pairs,
                reference=reference,
                config=config,
                seed=seed,
                lambda_cross=lambda_cross,
                cross_examples=cross,
            )
            rows.append({"seed": seed, "gain_over_b0": result["improvement_ratio"], "result": result})
        dose_rows[str(lambda_cross)] = {
            "lambda_cross": lambda_cross,
            "seed_results": [
                {"seed": row["seed"], "gain_over_b0": row["gain_over_b0"]}
                for row in rows
            ],
            "median_gain_over_b0": float(np.median([row["gain_over_b0"] for row in rows])),
            "_results": rows,
        }
    zero_results = dose_rows["0.0"]["_results"]
    interior_keys = ["0.05", "0.1", "0.25", "0.5", "1.0"]
    best_key = max(interior_keys, key=lambda key: dose_rows[key]["median_gain_over_b0"])
    best_results = dose_rows[best_key]["_results"]
    primary_dose = _paired_improvement(
        best_results[0]["result"],
        zero_results[0]["result"],
        seed=seeds[0],
        replicates=replicates,
    )
    directions = [
        float(candidate["gain_over_b0"]) > float(base_row["gain_over_b0"])
        for candidate, base_row in zip(best_results, zero_results)
    ]
    high_keys = ["2.0", "5.0"]
    high_decline = any(
        dose_rows[key]["median_gain_over_b0"] < dose_rows[best_key]["median_gain_over_b0"]
        for key in high_keys
    )
    dose_pass = bool(primary_dose["passed"]) and sum(directions) >= 4 and high_decline
    dose_report = {
        "schema": "cross-dialect-lambda-cross-dose-response-v1",
        "status": "passed" if dose_pass else "failed",
        "conditions": {
            key: {field: value for field, value in row.items() if field != "_results"}
            for key, row in dose_rows.items()
        },
        "best_interior_lambda": float(dose_rows[best_key]["lambda_cross"]),
        "comparison_to_lambda_zero": primary_dose,
        "criteria": {
            "positive_interior_beats_zero": bool(primary_dose["passed"]),
            "same_direction_seed_count": sum(directions),
            "min_same_direction_seed_count": 4,
            "high_lambda_decline": high_decline,
        },
    }

    cross_count = len(cross)
    same_pool = build_same_dialect_independent_pool(
        generic,
        requested_count=cross_count,
        seed=seeds[0],
    )
    common_count = same_pool.achieved_count
    cross_common = _sample_examples(cross, common_count, seeds[0] + 100)
    random_pool = build_target_matched_random_pool(
        generic,
        requested_count=common_count,
        seed=seeds[0] + 200,
    )
    control_pools = {
        "A_cross": list(cross_common),
        "B_same_dialect": list(same_pool.examples),
        "C_target_matched_random": list(random_pool.examples),
    }
    control_rows: dict[str, list[dict[str, object]]] = {key: [] for key in control_pools}
    b3_rows: list[dict[str, object]] = []
    for seed in seeds:
        b3 = _run_training_condition(
            calibration_embeddings=calibration_embeddings,
            evaluation_embeddings=evaluation_embeddings,
            calibration_records=calibration_records,
            evaluation_records=evaluation_records,
            calibration_pairs=calibration_pairs,
            evaluation_pairs=evaluation_pairs,
            reference=reference,
            config=config,
            seed=seed,
            lambda_cross=0.0,
            cross_examples=[],
        )
        b3_rows.append(b3)
        for name, pool in control_pools.items():
            result = _run_training_condition(
                calibration_embeddings=calibration_embeddings,
                evaluation_embeddings=evaluation_embeddings,
                calibration_records=calibration_records,
                evaluation_records=evaluation_records,
                calibration_pairs=calibration_pairs,
                evaluation_pairs=evaluation_pairs,
                reference=reference,
                config=config,
                seed=seed,
                lambda_cross=1.0,
                cross_examples=pool,
            )
            control_rows[name].append(result)
    a_b = _paired_improvement(control_rows["A_cross"][0], control_rows["B_same_dialect"][0], seed=seeds[0], replicates=replicates)
    a_c = _paired_improvement(control_rows["A_cross"][0], control_rows["C_target_matched_random"][0], seed=seeds[0], replicates=replicates)
    control_pass = bool(a_b["passed"]) and bool(a_c["passed"])
    independent_report = {
        "schema": "cross-dialect-same-dialect-independent-loss-v1",
        "status": "passed" if control_pass else "failed",
        "requested_count": cross_count,
        "achieved_common_count": common_count,
        "pools": {
            name: {
                "pair_count": len(pool),
                "target_histogram": _histogram(pool),
                "seed_results": [
                    {"seed": seed, "gain_over_b0": float(result["improvement_ratio"])}
                    for seed, result in zip(seeds, control_rows[name])
                ],
            }
            for name, pool in control_pools.items()
        },
        "comparisons": {"A_vs_B": a_b, "A_vs_C": a_c},
        "b3_seed_results": [
            {"seed": seed, "gain_over_b0": float(result["improvement_ratio"])}
            for seed, result in zip(seeds, b3_rows)
        ],
        "criteria": {
            "A_beats_B": bool(a_b["passed"]),
            "A_beats_C": bool(a_c["passed"]),
            "count_fallback_used": common_count < cross_count,
        },
    }

    # Train B4 and same-count once for the fixed-probe gradient report.
    same_count = build_same_count_no_cross_control(base)
    b4_fit = _run_training_condition(
        calibration_embeddings=calibration_embeddings,
        evaluation_embeddings=evaluation_embeddings,
        calibration_records=calibration_records,
        evaluation_records=evaluation_records,
        calibration_pairs=calibration_pairs,
        evaluation_pairs=evaluation_pairs,
        reference=reference,
        config=config,
        seed=seeds[0],
        lambda_cross=1.0,
        cross_examples=cross,
    )["fitted"]
    mixed_fit = _run_training_condition(
        calibration_embeddings=calibration_embeddings,
        evaluation_embeddings=evaluation_embeddings,
        calibration_records=calibration_records,
        evaluation_records=evaluation_records,
        calibration_pairs=calibration_pairs,
        evaluation_pairs=evaluation_pairs,
        reference=reference,
        config=config,
        seed=seeds[0],
        lambda_cross=0.0,
        cross_examples=[],
        pair_examples=same_count["pair_examples"],
    )["fitted"]
    standardizer = fit_standardizer(calibration_embeddings)
    standardized = standardizer.transform(calibration_embeddings)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    head_config = config.get("projection_head", config)
    model_b4 = _make_projection_model(head_config, b4_fit["head_kind"]).to(device)
    model_b4.load_state_dict(b4_fit["model_state"])
    model_mixed = _make_projection_model(head_config, mixed_fit["head_kind"]).to(device)
    model_mixed.load_state_dict(mixed_fit["model_state"])
    pair_left, pair_right, pair_targets = _pair_tensors(generic, standardized, device)
    cross_left, cross_right, cross_targets = _pair_tensors(cross, standardized, device)
    mixed_examples = list(same_count["pair_examples"])
    mixed_left, mixed_right, mixed_targets = _pair_tensors(mixed_examples, standardized, device)
    probe_b4 = compute_distance_gradient_probe_metrics(
        model_b4, pair_left, pair_right, pair_targets, cross_left, cross_right, cross_targets,
        mixed_left=mixed_left, mixed_right=mixed_right, mixed_targets=mixed_targets,
    )
    probe_mixed = compute_distance_gradient_probe_metrics(
        model_mixed, pair_left, pair_right, pair_targets, cross_left, cross_right, cross_targets,
        mixed_left=mixed_left, mixed_right=mixed_right, mixed_targets=mixed_targets,
    )
    threshold = float(isolation.get("normalized_gradient_threshold", 0.3))
    cosine_threshold = float(isolation.get("gradient_cosine_threshold", 0.3))
    ratio = float(probe_mixed["mixed_to_cross_normalized_ratio"])
    cosine = abs(float(probe_mixed["pair_cross_cosine"]))
    gradient_pass = ratio < threshold and cosine < cosine_threshold
    gradient_report = {
        "schema": "cross-dialect-gradient-norm-log-v1",
        "status": "passed" if gradient_pass else "failed",
        "seed": seeds[0],
        "b4": probe_b4,
        "same_count_no_cross": probe_mixed,
        "criteria": {
            "normalized_mixed_to_cross_ratio": ratio,
            "normalized_ratio_threshold": threshold,
            "pair_cross_cosine_absolute": cosine,
            "cosine_threshold": cosine_threshold,
        },
    }
    gate = aggregate_gradient_isolation_gate(
        dose_response=dose_report,
        independent_loss_control=independent_report,
        gradient_probe=gradient_report,
    )
    return {
        "dose_response": dose_report,
        "independent_loss_control": independent_report,
        "gradient_probe": gradient_report,
        "gate": gate,
    }


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--calibration-embedding", required=True)
    parser.add_argument("--evaluation-embedding", required=True)
    parser.add_argument("--records", required=True)
    parser.add_argument("--calibration-pairs", required=True)
    parser.add_argument("--evaluation-pairs", required=True)
    parser.add_argument("--reference-matrix", required=True)
    parser.add_argument("--dose-response-output", required=True)
    parser.add_argument("--independent-loss-output", required=True)
    parser.add_argument("--gradient-log-output", required=True)
    parser.add_argument("--gate-output", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    from src.config import load_config

    config = load_config(args.config)
    report = run_gradient_isolation_experiments(
        calibration_embeddings=_load_embedding(args.calibration_embedding),
        evaluation_embeddings=_load_embedding(args.evaluation_embedding),
        records=_load_records(args.records),
        calibration_pairs=_load_pairs(args.calibration_pairs),
        evaluation_pairs=_load_pairs(args.evaluation_pairs),
        reference=_load_reference(args.reference_matrix),
        config=config,
    )
    outputs = {
        args.dose_response_output: report["dose_response"],
        args.independent_loss_output: report["independent_loss_control"],
        args.gradient_log_output: report["gradient_probe"],
        args.gate_output: report["gate"],
    }
    for path, payload in outputs.items():
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2, default=str) + "\n",
            encoding="utf-8",
        )
    return 0 if report["gate"]["status"] == "passed" else 2


if __name__ == "__main__":
    main()
