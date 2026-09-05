"""Calibration-isolated speaker-distance correction gate."""

from __future__ import annotations

import argparse
import json
import math
import random
from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Any, Iterable, Mapping, Sequence


DEFAULT_LAMBDAS = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.5, 2.0]


def scalar_subtraction_scope_contract() -> dict[str, Any]:
    """Declare the evaluation information used by scalar subtraction."""
    return {
        "fit_scope": "calibration_speakers",
        "evaluation_feature_scope": "current_pair_only",
        "inference_class": "inductive",
        "evaluation_statistic": "current pair endpoint speaker-embedding distance",
        "fallback_count": 0,
    }


def threshold_sensitivity(
    method_rows: Sequence[Mapping[str, Any]],
    *,
    thresholds: Sequence[float] = (0.03, 0.05, 0.10),
) -> dict[str, Any]:
    """Evaluate unchanged correction estimates against operational thresholds."""
    if not method_rows:
        raise ValueError("at least one correction estimate is required")
    rows = []
    for method in method_rows:
        estimate = float(method["improvement_ratio"])
        ci = dict(method.get("ci", {}))
        rows.append({
            "method": str(method["method"]),
            "model_name": str(method.get("model_name", "")),
            "reference_name": str(method.get("reference_name", "")),
            "improvement_ratio": estimate,
            "ci": ci,
            "thresholds": [
                {
                    "threshold": float(threshold),
                    "passed": estimate >= float(threshold) and float(ci.get("lower", float("-inf"))) > 0.0,
                }
                for threshold in thresholds
            ],
        })
    return {
        "schema": "correction-threshold-sensitivity-v1",
        "operational_rationale": "3%, 5%, and 10% are reported as sensitivity thresholds, not inferential cutoffs or preregistered criteria",
        "thresholds": [float(value) for value in thresholds],
        "rows": rows,
        "all_below_three_percent": all(float(row["improvement_ratio"]) < 0.03 for row in rows),
    }


def _target_distance(row: Mapping[str, Any], reference: Mapping[str, Any]) -> float | None:
    labels = row.get("dialect_labels")
    if not isinstance(labels, list) or not labels:
        raise ValueError("distance row missing dialect_labels")
    if len(labels) == 1:
        return 0.0
    if len(labels) != 2:
        return None
    matrix = reference["matrix"]
    try:
        return float(matrix[labels[0]][labels[1]])
    except KeyError:
        return None


def _by_model(rows: Iterable[Mapping[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    models: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        model = str(row["model_name"])
        models[model].append(dict(row))
    return dict(sorted(models.items()))


def _by_pair(rows: Iterable[Mapping[str, Any]]) -> dict[str, dict[str, Any]]:
    return {str(row["pair_id"]): dict(row) for row in rows}


def _joined_rows(
    speech_rows: Iterable[Mapping[str, Any]],
    speaker_rows: Iterable[Mapping[str, Any]],
    reference: Mapping[str, Any],
) -> list[dict[str, Any]]:
    speaker_by_pair = _by_pair(speaker_rows)
    joined: list[dict[str, Any]] = []
    for row in speech_rows:
        pair_id = str(row["pair_id"])
        speaker = speaker_by_pair.get(pair_id)
        if speaker is None:
            continue
        target = _target_distance(row, reference)
        if target is None:
            continue
        speech_distance = float(row["distance"])
        speaker_distance = float(speaker["distance"])
        if not all(math.isfinite(value) for value in (speech_distance, speaker_distance, target)):
            raise ValueError("correction row has non-finite distance")
        joined.append(
            {
                "pair_id": pair_id,
                "group": row.get("group"),
                "speech_distance": speech_distance,
                "speaker_distance": speaker_distance,
                "target_distance": target,
            }
        )
    if not joined:
        raise ValueError("no rows join between speech, speaker, and reference distances")
    return joined


def _corrected_distance(row: Mapping[str, float], lambda_value: float) -> float:
    return max(0.0, float(row["speech_distance"]) - lambda_value * float(row["speaker_distance"]))


def _fit_affine(xs: Sequence[float], ys: Sequence[float]) -> tuple[float, float]:
    if len(xs) != len(ys) or not xs:
        raise ValueError("cannot fit affine scale without paired values")
    x_mean = mean(xs)
    y_mean = mean(ys)
    variance = sum((x - x_mean) ** 2 for x in xs)
    if variance <= 0:
        return y_mean, 0.0
    covariance = sum((x - x_mean) * (y - y_mean) for x, y in zip(xs, ys))
    slope = max(0.0, covariance / variance)
    intercept = y_mean - slope * x_mean
    return intercept, slope


def _mae(rows: Sequence[Mapping[str, Any]], lambda_value: float, intercept: float, slope: float) -> float:
    errors = [
        abs(intercept + slope * _corrected_distance(row, lambda_value) - float(row["target_distance"]))
        for row in rows
    ]
    return mean(errors) if errors else 0.0


def _select_lambda(
    calibration_rows: Sequence[Mapping[str, Any]],
    lambdas: Sequence[float],
) -> tuple[float, float, float]:
    best: tuple[float, float, float, float] | None = None
    for lambda_value in lambdas:
        xs = [_corrected_distance(row, lambda_value) for row in calibration_rows]
        ys = [float(row["target_distance"]) for row in calibration_rows]
        intercept, slope = _fit_affine(xs, ys)
        mae = _mae(calibration_rows, lambda_value, intercept, slope)
        candidate = (mae, lambda_value, intercept, slope)
        if best is None or candidate < best:
            best = candidate
    assert best is not None
    _, selected_lambda, intercept, slope = best
    return selected_lambda, intercept, slope


def _improvement_ratio(baseline_mae: float, corrected_mae: float) -> float:
    if baseline_mae <= 0:
        return 0.0
    return (baseline_mae - corrected_mae) / baseline_mae


def _quantile(values: Sequence[float], probability: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * probability
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def _bootstrap_metric(
    rows: Sequence[Mapping[str, Any]],
    *,
    lambda_value: float,
    baseline_intercept: float,
    baseline_slope: float,
    corrected_intercept: float,
    corrected_slope: float,
    seed: int,
    replicates: int,
    groups: set[str] | None = None,
) -> list[float]:
    rng = random.Random(seed)
    pool = [row for row in rows if groups is None or str(row.get("group")) in groups]
    if not pool:
        return [0.0 for _ in range(replicates)]
    estimates: list[float] = []
    for _ in range(replicates):
        sample = [rng.choice(pool) for _ in pool]
        baseline = _mae(sample, 0.0, baseline_intercept, baseline_slope)
        corrected = _mae(sample, lambda_value, corrected_intercept, corrected_slope)
        estimates.append(_improvement_ratio(baseline, corrected))
    return estimates


def _shuffle_speaker_distances(rows: Sequence[Mapping[str, Any]], seed: int) -> list[dict[str, Any]]:
    rng = random.Random(seed)
    values = [float(row["speaker_distance"]) for row in rows]
    rng.shuffle(values)
    shuffled: list[dict[str, Any]] = []
    for row, value in zip(rows, values):
        item = dict(row)
        item["speaker_distance"] = value
        shuffled.append(item)
    return shuffled


def _reference_report(
    *,
    model_name: str,
    reference: Mapping[str, Any],
    calibration_rows: Sequence[Mapping[str, Any]],
    evaluation_rows: Sequence[Mapping[str, Any]],
    lambdas: Sequence[float],
    seed: int,
    bootstrap_replicates: int,
    improvement_threshold: float,
    matched_tolerance: float,
) -> dict[str, Any]:
    baseline_lambda, baseline_intercept, baseline_slope = _select_lambda(calibration_rows, [0.0])
    selected_lambda, corrected_intercept, corrected_slope = _select_lambda(calibration_rows, lambdas)
    baseline_mae = _mae(evaluation_rows, baseline_lambda, baseline_intercept, baseline_slope)
    corrected_mae = _mae(evaluation_rows, selected_lambda, corrected_intercept, corrected_slope)
    improvement = _improvement_ratio(baseline_mae, corrected_mae)
    estimates = _bootstrap_metric(
        evaluation_rows,
        lambda_value=selected_lambda,
        baseline_intercept=baseline_intercept,
        baseline_slope=baseline_slope,
        corrected_intercept=corrected_intercept,
        corrected_slope=corrected_slope,
        seed=seed,
        replicates=bootstrap_replicates,
    )
    matched_estimates = _bootstrap_metric(
        evaluation_rows,
        lambda_value=selected_lambda,
        baseline_intercept=baseline_intercept,
        baseline_slope=baseline_slope,
        corrected_intercept=corrected_intercept,
        corrected_slope=corrected_slope,
        seed=seed + 17,
        replicates=bootstrap_replicates,
        groups={"A", "C"},
    )
    matched_increase = -_improvement_ratio(
        _mae([row for row in evaluation_rows if row.get("group") in {"A", "C"}], baseline_lambda, baseline_intercept, baseline_slope),
        _mae(
            [row for row in evaluation_rows if row.get("group") in {"A", "C"}],
            selected_lambda,
            corrected_intercept,
            corrected_slope,
        ),
    )
    shuffled_improvements: list[float] = []
    for offset in range(5):
        shuffled_calibration = _shuffle_speaker_distances(calibration_rows, seed + 100 + offset)
        shuffled_evaluation = _shuffle_speaker_distances(evaluation_rows, seed + 200 + offset)
        shuffled_lambda, shuffled_intercept, shuffled_slope = _select_lambda(shuffled_calibration, lambdas)
        shuffled_corrected = _mae(shuffled_evaluation, shuffled_lambda, shuffled_intercept, shuffled_slope)
        shuffled_improvements.append(_improvement_ratio(baseline_mae, shuffled_corrected))

    passed = (
        improvement >= improvement_threshold
        and _quantile(estimates, 0.025) > 0.0
        and matched_increase <= matched_tolerance
        and -_quantile(matched_estimates, 0.025) <= matched_tolerance
        and max(shuffled_improvements) < improvement_threshold
    )
    return {
        "reference_name": reference.get("name", "reference"),
        "calibration_count": len(calibration_rows),
        "evaluation_count": len(evaluation_rows),
        "selected_lambda": selected_lambda,
        "baseline_scale": {"intercept": baseline_intercept, "slope": baseline_slope},
        "corrected_scale": {"intercept": corrected_intercept, "slope": corrected_slope},
        "baseline_mae": baseline_mae,
        "corrected_mae": corrected_mae,
        "improvement_ratio": improvement,
        "ci": {"lower": _quantile(estimates, 0.025), "upper": _quantile(estimates, 0.975), "confidence_level": 0.95},
        "matched_speaker_mae_increase_ratio": matched_increase,
        "matched_speaker_ci": {
            "increase_upper": -_quantile(matched_estimates, 0.025),
            "confidence_level": 0.95,
        },
        "shuffled_speaker_improvement_ratios": shuffled_improvements,
        "status": "passed" if passed else "failed",
        "model_name": model_name,
    }


def evaluate_correction_gate(
    calibration_rows: Iterable[Mapping[str, Any]],
    evaluation_rows: Iterable[Mapping[str, Any]],
    calibration_speaker_rows: Iterable[Mapping[str, Any]],
    evaluation_speaker_rows: Iterable[Mapping[str, Any]],
    *,
    references: Sequence[Mapping[str, Any]],
    lambdas: Sequence[float] = DEFAULT_LAMBDAS,
    seed: int = 20260829,
    bootstrap_replicates: int = 1000,
    improvement_threshold: float = 0.05,
    matched_tolerance: float = 0.01,
) -> dict[str, Any]:
    if bootstrap_replicates < 1000:
        raise ValueError("at least 1000 bootstrap replicates are required")
    calibration_by_model = _by_model(calibration_rows)
    evaluation_by_model = _by_model(evaluation_rows)
    calibration_speaker = list(calibration_speaker_rows)
    evaluation_speaker = list(evaluation_speaker_rows)
    reports: list[dict[str, Any]] = []
    for model_name, model_calibration_rows in calibration_by_model.items():
        if model_name not in evaluation_by_model:
            continue
        reference_reports: list[dict[str, Any]] = []
        for reference in references:
            joined_calibration = _joined_rows(model_calibration_rows, calibration_speaker, reference)
            joined_evaluation = _joined_rows(evaluation_by_model[model_name], evaluation_speaker, reference)
            reference_reports.append(
                _reference_report(
                    model_name=model_name,
                    reference=reference,
                    calibration_rows=joined_calibration,
                    evaluation_rows=joined_evaluation,
                    lambdas=lambdas,
                    seed=seed,
                    bootstrap_replicates=bootstrap_replicates,
                    improvement_threshold=improvement_threshold,
                    matched_tolerance=matched_tolerance,
                )
            )
        reports.append(
            {
                "model_name": model_name,
                "status": "passed" if reference_reports and all(report["status"] == "passed" for report in reference_reports) else "failed",
                "references": reference_reports,
            }
        )
    status = "passed" if reports and all(report["status"] == "passed" for report in reports) else "failed"
    return {
        "schema": "correction-gate-v1",
        "seed": seed,
        "lambda_grid": list(lambdas),
        "thresholds": {
            "correction_improvement": improvement_threshold,
            "matched_speaker_tolerance": matched_tolerance,
        },
        "status": status,
        "decision": "continue_to_manuscript" if status == "passed" else "stop_before_manuscript_and_release",
        "models": reports,
    }


def _load_distance_rows(paths: Iterable[str | Path]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in paths:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        model = payload["model_name"]
        for row in payload["distances"]:
            item = dict(row)
            item["model_name"] = model
            rows.append(item)
    return rows


def _load_references(paths: Iterable[str | Path]) -> list[dict[str, Any]]:
    return [json.loads(Path(path).read_text(encoding="utf-8")) for path in paths]


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--calibration-distances", nargs="+", action="append", required=True)
    parser.add_argument("--evaluation-distances", nargs="+", action="append", required=True)
    parser.add_argument("--calibration-speaker-distances", required=True)
    parser.add_argument("--evaluation-speaker-distances", required=True)
    parser.add_argument("--reference-matrix", nargs="+", action="append", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--seed", type=int, default=20260829)
    parser.add_argument("--bootstrap-replicates", type=int, default=1000)
    args = parser.parse_args(argv)
    args.calibration_distances = [path for group in args.calibration_distances for path in group]
    args.evaluation_distances = [path for group in args.evaluation_distances for path in group]
    args.reference_matrix = [path for group in args.reference_matrix for path in group]
    return args


def main() -> int:
    args = _parse_args()
    report = evaluate_correction_gate(
        _load_distance_rows(args.calibration_distances),
        _load_distance_rows(args.evaluation_distances),
        _load_distance_rows([args.calibration_speaker_distances]),
        _load_distance_rows([args.evaluation_speaker_distances]),
        references=_load_references(args.reference_matrix),
        seed=args.seed,
        bootstrap_replicates=args.bootstrap_replicates,
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return 0 if report["status"] == "passed" else 2


if __name__ == "__main__":
    raise SystemExit(main())
