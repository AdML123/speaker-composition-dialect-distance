"""Alpha-sweep diagnostic for speaker-mean normalization trade-offs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import mean
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from .speaker_mean_normalization_gate import (
    _fit_affine,
    _improvement_ratio,
    _load_embeddings,
    _load_pairs,
    _mae,
    _metadata_for_embedding_sets,
    _pair_distances,
    fit_split_speaker_mean_model,
)


DEFAULT_ALPHA_GRID = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.2, 1.5]


def _matrix(embeddings: Mapping[str, Sequence[float]], ids: Sequence[str]) -> np.ndarray:
    values = np.asarray([embeddings[utterance_id] for utterance_id in ids], dtype=np.float64)
    if values.ndim != 2 or values.shape[0] != len(ids) or not np.isfinite(values).all():
        raise ValueError("embeddings must be a finite 2D matrix")
    return values


def _alpha_normalize_embeddings(
    embeddings: Mapping[str, Sequence[float]],
    metadata: Mapping[str, Mapping[str, Any]],
    *,
    alpha: float,
) -> dict[str, list[float]]:
    if alpha < 0:
        raise ValueError("alpha must be non-negative")
    model = fit_split_speaker_mean_model(embeddings, metadata)
    ids = model["utterance_ids"]
    x = _matrix(embeddings, ids)
    global_mean = np.asarray(model["global_mean"], dtype=np.float64)
    speaker_means = model["speaker_means"]
    speakers = model["speaker_ids"]
    cleaned = np.asarray(
        [
            x[index] - alpha * (np.asarray(speaker_means[speakers[utterance_id]], dtype=np.float64) - global_mean)
            for index, utterance_id in enumerate(ids)
        ],
        dtype=np.float64,
    )
    return {utterance_id: cleaned[index].astype(float).tolist() for index, utterance_id in enumerate(ids)}


def _change_ratio(baseline: float, current: float) -> float:
    if baseline <= 0:
        return 0.0
    return (current - baseline) / baseline


def _pareto_frontier(points: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    frontier: list[dict[str, Any]] = []
    for candidate in points:
        dominated = False
        for other in points:
            if other is candidate:
                continue
            if (
                other["group_a_harm_ratio"] <= candidate["group_a_harm_ratio"]
                and other["group_c_gain_ratio"] >= candidate["group_c_gain_ratio"]
                and (
                    other["group_a_harm_ratio"] < candidate["group_a_harm_ratio"]
                    or other["group_c_gain_ratio"] > candidate["group_c_gain_ratio"]
                )
            ):
                dominated = True
                break
        if not dominated:
            frontier.append(
                {
                    "alpha": candidate["alpha"],
                    "group_a_harm_ratio": candidate["group_a_harm_ratio"],
                    "group_c_gain_ratio": candidate["group_c_gain_ratio"],
                }
            )
    return sorted(frontier, key=lambda item: (item["group_a_harm_ratio"], -item["group_c_gain_ratio"], item["alpha"]))


def _curve_for_reference(
    *,
    model_name: str,
    calibration_embeddings: Mapping[str, Sequence[float]],
    evaluation_embeddings: Mapping[str, Sequence[float]],
    calibration_metadata: Mapping[str, Mapping[str, Any]],
    evaluation_metadata: Mapping[str, Mapping[str, Any]],
    calibration_pairs: Sequence[Mapping[str, Any]],
    evaluation_pairs: Sequence[Mapping[str, Any]],
    reference: Mapping[str, Any],
    alpha_grid: Sequence[float],
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    baseline_group_a_mae = baseline_group_c_mae = None
    baseline_eval_mae = None
    for alpha in alpha_grid:
        calibrated = _alpha_normalize_embeddings(calibration_embeddings, calibration_metadata, alpha=alpha)
        evaluated = _alpha_normalize_embeddings(evaluation_embeddings, evaluation_metadata, alpha=alpha)
        calibration_rows = _pair_distances(calibration_pairs, calibrated, reference)
        evaluation_rows = _pair_distances(evaluation_pairs, evaluated, reference)
        intercept, slope = _fit_affine(calibration_rows)
        calibration_mae = _mae(calibration_rows, intercept, slope)
        evaluation_mae = _mae(evaluation_rows, intercept, slope)
        group_a_rows = [row for row in evaluation_rows if row.get("group") == "A"]
        group_c_rows = [row for row in evaluation_rows if row.get("group") == "C"]
        group_a_mae = _mae(group_a_rows, intercept, slope)
        group_c_mae = _mae(group_c_rows, intercept, slope)
        if baseline_eval_mae is None and alpha == alpha_grid[0]:
            baseline_eval_mae = evaluation_mae
            baseline_group_a_mae = group_a_mae
            baseline_group_c_mae = group_c_mae
        rows.append(
            {
                "alpha": float(alpha),
                "calibration_mae": calibration_mae,
                "evaluation_mae": evaluation_mae,
                "improvement_ratio": _improvement_ratio(baseline_eval_mae, evaluation_mae) if baseline_eval_mae is not None else 0.0,
                "group_a_mae": group_a_mae,
                "group_c_mae": group_c_mae,
                "group_a_mae_change_ratio": _change_ratio(baseline_group_a_mae, group_a_mae) if baseline_group_a_mae is not None else 0.0,
                "group_c_mae_change_ratio": _change_ratio(baseline_group_c_mae, group_c_mae) if baseline_group_c_mae is not None else 0.0,
                "group_a_harm_ratio": max(0.0, _change_ratio(baseline_group_a_mae, group_a_mae) if baseline_group_a_mae is not None else 0.0),
                "group_c_gain_ratio": max(0.0, -(_change_ratio(baseline_group_c_mae, group_c_mae) if baseline_group_c_mae is not None else 0.0)),
                "affine_scale": {"intercept": float(intercept), "slope": float(slope)},
            }
        )
    return {
        "model_name": model_name,
        "reference_name": reference.get("name", "reference"),
        "baseline": rows[0],
        "curve": rows,
        "pareto_frontier": _pareto_frontier(rows),
    }


def run_alpha_sweep(
    calibration_embeddings: Mapping[str, Mapping[str, Sequence[float]]] | None = None,
    evaluation_embeddings: Mapping[str, Mapping[str, Sequence[float]]] | None = None,
    calibration_metadata: Mapping[str, Mapping[str, Any]] | None = None,
    evaluation_metadata: Mapping[str, Mapping[str, Any]] | None = None,
    calibration_pairs: Sequence[Mapping[str, Any]] | None = None,
    evaluation_pairs: Sequence[Mapping[str, Any]] | None = None,
    *,
    references: Sequence[Mapping[str, Any]],
    alpha_grid: Sequence[float] = DEFAULT_ALPHA_GRID,
    bootstrap_replicates: int | None = None,
    calibration_embeddings_by_model: Mapping[str, Mapping[str, Sequence[float]]] | None = None,
    evaluation_embeddings_by_model: Mapping[str, Mapping[str, Sequence[float]]] | None = None,
) -> dict[str, Any]:
    del bootstrap_replicates
    if calibration_embeddings_by_model is None:
        calibration_embeddings_by_model = calibration_embeddings
    if evaluation_embeddings_by_model is None:
        evaluation_embeddings_by_model = evaluation_embeddings
    if calibration_embeddings_by_model is None or evaluation_embeddings_by_model is None:
        raise ValueError("calibration and evaluation embeddings are required")
    if calibration_metadata is None or evaluation_metadata is None:
        raise ValueError("calibration and evaluation metadata are required")
    if calibration_pairs is None or evaluation_pairs is None:
        raise ValueError("calibration and evaluation pairs are required")
    alpha_grid = sorted({float(alpha) for alpha in alpha_grid} | {0.0})
    reports = []
    for model_name, calibration_embeddings in sorted(calibration_embeddings_by_model.items()):
        if model_name not in evaluation_embeddings_by_model:
            continue
        reference_reports = [
            _curve_for_reference(
                model_name=model_name,
                calibration_embeddings=calibration_embeddings,
                evaluation_embeddings=evaluation_embeddings_by_model[model_name],
                calibration_metadata=calibration_metadata,
                evaluation_metadata=evaluation_metadata,
                calibration_pairs=calibration_pairs,
                evaluation_pairs=evaluation_pairs,
                reference=reference,
                alpha_grid=alpha_grid,
            )
            for reference in references
        ]
        reports.append({"model_name": model_name, "references": reference_reports})
    return {
        "schema": "speaker-offset-alpha-sweep-v1",
        "alphas": alpha_grid,
        "models": reports,
    }


def _load_references(paths: Iterable[str | Path]) -> list[dict[str, Any]]:
    return [json.loads(Path(path).read_text(encoding="utf-8")) for path in paths]


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--calibration-embedding", nargs="+", action="append", required=True)
    parser.add_argument("--evaluation-embedding", nargs="+", action="append", required=True)
    parser.add_argument("--calibration-pairs", required=True)
    parser.add_argument("--evaluation-pairs", required=True)
    parser.add_argument("--records", required=True)
    parser.add_argument("--reference-matrix", nargs="+", action="append", required=True)
    parser.add_argument("--alpha-grid", nargs="+", type=float, required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    args.calibration_embedding = [path for group in args.calibration_embedding for path in group]
    args.evaluation_embedding = [path for group in args.evaluation_embedding for path in group]
    args.reference_matrix = [path for group in args.reference_matrix for path in group]
    return args


def main() -> int:
    args = _parse_args()
    calibration_embeddings = _load_embeddings(args.calibration_embedding)
    evaluation_embeddings = _load_embeddings(args.evaluation_embedding)
    report = run_alpha_sweep(
        calibration_embeddings,
        evaluation_embeddings,
        _metadata_for_embedding_sets(args.records, calibration_embeddings),
        _metadata_for_embedding_sets(args.records, evaluation_embeddings),
        _load_pairs(args.calibration_pairs),
        _load_pairs(args.evaluation_pairs),
        references=_load_references(args.reference_matrix),
        alpha_grid=args.alpha_grid,
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
