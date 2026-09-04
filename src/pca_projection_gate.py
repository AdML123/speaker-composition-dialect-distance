"""PCA speaker-subspace projection gate."""

from __future__ import annotations

import argparse
import json
import math
import random
from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Any, Iterable, Mapping, Sequence

import numpy as np


DEFAULT_K_GRID = [0, 1, 2, 5, 10, 20, 50, 100]
DEFAULT_RANDOM_SEEDS = [20260829, 20260830, 20260831, 20260901, 20260902]


def _ordered_ids(embeddings: Mapping[str, Sequence[float]], metadata: Mapping[str, Mapping[str, Any]]) -> list[str]:
    ids = sorted(embeddings)
    missing = [utterance_id for utterance_id in ids if utterance_id not in metadata]
    if missing:
        raise ValueError(f"missing metadata for utterance: {missing[0]}")
    return ids


def _matrix(embeddings: Mapping[str, Sequence[float]], ids: Sequence[str]) -> np.ndarray:
    values = np.asarray([embeddings[utterance_id] for utterance_id in ids], dtype=np.float64)
    if values.ndim != 2 or values.shape[0] != len(ids) or not np.isfinite(values).all():
        raise ValueError("embeddings must be a finite 2D matrix")
    return values


def _one_hot(values: Sequence[str]) -> np.ndarray:
    levels = sorted(set(values))
    if len(levels) <= 1:
        return np.zeros((len(values), 0), dtype=np.float64)
    columns = []
    for level in levels[1:]:
        columns.append([1.0 if value == level else 0.0 for value in values])
    return np.asarray(columns, dtype=np.float64).T


def _residualize(values: np.ndarray, metadata_rows: Sequence[Mapping[str, Any]]) -> np.ndarray:
    nuisance = np.column_stack(
        [
            np.ones(len(metadata_rows), dtype=np.float64),
            _one_hot([str(row["dialect_label"]) for row in metadata_rows]),
            _one_hot([str(row["recording_condition"]) for row in metadata_rows]),
        ]
    )
    q, _ = np.linalg.qr(nuisance, mode="reduced")
    return values - q @ (q.T @ values)


def _speaker_f_statistics(scores: np.ndarray, metadata_rows: Sequence[Mapping[str, Any]]) -> np.ndarray:
    residual_scores = _residualize(scores, metadata_rows)
    speakers = [str(row["speaker_id"]) for row in metadata_rows]
    levels = sorted(set(speakers))
    if len(levels) <= 1:
        return np.zeros(scores.shape[1], dtype=np.float64)
    grand_mean = residual_scores.mean(axis=0)
    between = np.zeros(scores.shape[1], dtype=np.float64)
    within = np.zeros(scores.shape[1], dtype=np.float64)
    for speaker in levels:
        mask = np.asarray([value == speaker for value in speakers])
        group = residual_scores[mask]
        if group.size == 0:
            continue
        group_mean = group.mean(axis=0)
        between += group.shape[0] * (group_mean - grand_mean) ** 2
        within += ((group - group_mean) ** 2).sum(axis=0)
    df_between = len(levels) - 1
    df_within = max(1, len(speakers) - len(levels))
    denominator = within / df_within
    with np.errstate(divide="ignore", invalid="ignore"):
        f_values = (between / df_between) / denominator
    return np.nan_to_num(f_values, nan=0.0, posinf=np.finfo(np.float64).max)


def fit_projection_model(
    embeddings: Mapping[str, Sequence[float]],
    metadata: Mapping[str, Mapping[str, Any]],
    *,
    max_components: int = 100,
) -> dict[str, Any]:
    """Fit calibration-only PCA and rank components by residual speaker ANOVA."""
    ids = _ordered_ids(embeddings, metadata)
    x = _matrix(embeddings, ids)
    mean_vector = x.mean(axis=0)
    centered = x - mean_vector
    _, _, vh = np.linalg.svd(centered, full_matrices=False)
    component_count = min(max_components, vh.shape[0])
    components = vh[:component_count]
    scores = centered @ components.T
    metadata_rows = [metadata[utterance_id] for utterance_id in ids]
    f_values = _speaker_f_statistics(scores, metadata_rows)
    ranked = sorted(range(component_count), key=lambda index: (-float(f_values[index]), index))
    return {
        "utterance_ids": ids,
        "mean": mean_vector.tolist(),
        "components": components.tolist(),
        "speaker_f_statistics": [float(value) for value in f_values],
        "ranked_components": ranked,
    }


def apply_projection(
    embeddings: Mapping[str, Sequence[float]],
    projection_model: Mapping[str, Any],
    *,
    k: int,
    component_indices: Sequence[int] | None = None,
) -> dict[str, list[float]]:
    if k < 0:
        raise ValueError("k must be non-negative")
    if k == 0:
        return {utterance_id: [float(value) for value in vector] for utterance_id, vector in sorted(embeddings.items())}
    mean_vector = np.asarray(projection_model["mean"], dtype=np.float64)
    components = np.asarray(projection_model["components"], dtype=np.float64)
    selected = list(component_indices if component_indices is not None else projection_model["ranked_components"][:k])
    if len(selected) < k:
        raise ValueError("not enough PCA components for requested k")
    basis = components[np.asarray(selected[:k], dtype=int)]
    ids = sorted(embeddings)
    x = _matrix(embeddings, ids)
    centered = x - mean_vector
    cleaned = x - (centered @ basis.T) @ basis
    return {utterance_id: cleaned[index].astype(float).tolist() for index, utterance_id in enumerate(ids)}


def _cosine_distance(a: Sequence[float], b: Sequence[float]) -> float:
    left = np.asarray(a, dtype=np.float64)
    right = np.asarray(b, dtype=np.float64)
    left_norm = float(np.linalg.norm(left))
    right_norm = float(np.linalg.norm(right))
    if left_norm == 0.0 and right_norm == 0.0:
        return 0.0
    if left_norm == 0.0 or right_norm == 0.0:
        return 1.0
    return float(min(2.0, max(0.0, 1.0 - float(np.dot(left, right) / (left_norm * right_norm)))))


def _pair_distances(
    pairs: Sequence[Mapping[str, Any]],
    embeddings: Mapping[str, Sequence[float]],
    reference: Mapping[str, Any],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for pair in pairs:
        utterance_a, utterance_b = pair["source_utterance_ids"]
        labels = pair["dialect_labels"]
        target = _target_distance(labels, reference)
        if target is None:
            continue
        rows.append(
            {
                "pair_id": pair["pair_id"],
                "group": pair.get("group"),
                "dialect_labels": labels,
                "distance": _cosine_distance(embeddings[utterance_a], embeddings[utterance_b]),
                "target_distance": target,
            }
        )
    return rows


def _target_distance(labels: Sequence[str], reference: Mapping[str, Any]) -> float | None:
    if len(labels) == 1:
        return 0.0
    if len(labels) != 2:
        return None
    matrix = reference["matrix"]
    try:
        return float(matrix[labels[0]][labels[1]])
    except KeyError:
        return None


def _fit_affine(rows: Sequence[Mapping[str, Any]]) -> tuple[float, float]:
    xs = [float(row["distance"]) for row in rows]
    ys = [float(row["target_distance"]) for row in rows]
    x_mean = mean(xs)
    y_mean = mean(ys)
    variance = sum((x - x_mean) ** 2 for x in xs)
    if variance <= 0:
        return y_mean, 0.0
    covariance = sum((x - x_mean) * (y - y_mean) for x, y in zip(xs, ys))
    slope = max(0.0, covariance / variance)
    return y_mean - slope * x_mean, slope


def _mae(rows: Sequence[Mapping[str, Any]], intercept: float, slope: float) -> float:
    if not rows:
        return 0.0
    return mean(abs(intercept + slope * float(row["distance"]) - float(row["target_distance"])) for row in rows)


def select_k_by_calibration(
    *,
    candidates: Mapping[int, Sequence[Mapping[str, Any]]],
    k_grid: Sequence[int],
) -> dict[str, Any]:
    best: tuple[float, int, float, float] | None = None
    for k in k_grid:
        rows = list(candidates[k])
        intercept, slope = _fit_affine(rows)
        mae = _mae(rows, intercept, slope)
        candidate = (mae, k, intercept, slope)
        if best is None or candidate < best:
            best = candidate
    assert best is not None
    mae, selected_k, intercept, slope = best
    return {"selected_k": selected_k, "calibration_mae": mae, "intercept": intercept, "slope": slope}


def _improvement_ratio(baseline: float, corrected: float) -> float:
    if baseline <= 0:
        return 0.0
    return (baseline - corrected) / baseline


def _quantile(values: Sequence[float], probability: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * probability
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def _bootstrap_improvements(
    baseline_rows: Sequence[Mapping[str, Any]],
    corrected_rows: Sequence[Mapping[str, Any]],
    baseline_scale: tuple[float, float],
    corrected_scale: tuple[float, float],
    *,
    seed: int,
    replicates: int,
    groups: set[str] | None = None,
) -> list[float]:
    baseline_by_pair = {row["pair_id"]: row for row in baseline_rows if groups is None or row.get("group") in groups}
    corrected_by_pair = {row["pair_id"]: row for row in corrected_rows if row["pair_id"] in baseline_by_pair}
    pair_ids = sorted(corrected_by_pair)
    if not pair_ids:
        return [0.0 for _ in range(replicates)]
    rng = random.Random(seed)
    estimates: list[float] = []
    for _ in range(replicates):
        sample_ids = [rng.choice(pair_ids) for _ in pair_ids]
        baseline = _mae([baseline_by_pair[pair_id] for pair_id in sample_ids], *baseline_scale)
        corrected = _mae([corrected_by_pair[pair_id] for pair_id in sample_ids], *corrected_scale)
        estimates.append(_improvement_ratio(baseline, corrected))
    return estimates


def _random_component_indices(
    ranked_components: Sequence[int],
    component_count: int,
    k: int,
    seed: int,
) -> list[int]:
    if k == 0:
        return []
    ranked_set = set(ranked_components[:k])
    pool = [index for index in range(component_count) if index not in ranked_set]
    if len(pool) < k:
        pool = [index for index in range(component_count) if index not in ranked_components[:1]]
    rng = random.Random(seed)
    return rng.sample(pool, k)


def _reference_report(
    *,
    model_name: str,
    projection_model: Mapping[str, Any],
    calibration_embeddings: Mapping[str, Sequence[float]],
    evaluation_embeddings: Mapping[str, Sequence[float]],
    calibration_pairs: Sequence[Mapping[str, Any]],
    evaluation_pairs: Sequence[Mapping[str, Any]],
    reference: Mapping[str, Any],
    k_grid: Sequence[int],
    bootstrap_replicates: int,
    random_seeds: Sequence[int],
    seed: int,
    improvement_threshold: float,
    matched_tolerance: float,
) -> dict[str, Any]:
    calibration_candidates: dict[int, list[dict[str, Any]]] = {}
    evaluation_candidates: dict[int, list[dict[str, Any]]] = {}
    for k in k_grid:
        cleaned_calibration = apply_projection(calibration_embeddings, projection_model, k=k)
        cleaned_evaluation = apply_projection(evaluation_embeddings, projection_model, k=k)
        calibration_candidates[k] = _pair_distances(calibration_pairs, cleaned_calibration, reference)
        evaluation_candidates[k] = _pair_distances(evaluation_pairs, cleaned_evaluation, reference)
    selection = select_k_by_calibration(candidates=calibration_candidates, k_grid=k_grid)
    baseline_selection = select_k_by_calibration(candidates={0: calibration_candidates[0]}, k_grid=[0])
    selected_k = int(selection["selected_k"])
    baseline_scale = (float(baseline_selection["intercept"]), float(baseline_selection["slope"]))
    corrected_scale = (float(selection["intercept"]), float(selection["slope"]))
    baseline_rows = evaluation_candidates[0]
    corrected_rows = evaluation_candidates[selected_k]
    baseline_mae = _mae(baseline_rows, *baseline_scale)
    corrected_mae = _mae(corrected_rows, *corrected_scale)
    improvement = _improvement_ratio(baseline_mae, corrected_mae)
    estimates = _bootstrap_improvements(
        baseline_rows,
        corrected_rows,
        baseline_scale,
        corrected_scale,
        seed=seed,
        replicates=bootstrap_replicates,
    )
    matched_estimates = _bootstrap_improvements(
        baseline_rows,
        corrected_rows,
        baseline_scale,
        corrected_scale,
        seed=seed + 31,
        replicates=bootstrap_replicates,
        groups={"A", "C"},
    )
    matched_baseline = _mae([row for row in baseline_rows if row.get("group") in {"A", "C"}], *baseline_scale)
    matched_corrected = _mae([row for row in corrected_rows if row.get("group") in {"A", "C"}], *corrected_scale)
    matched_increase = -_improvement_ratio(matched_baseline, matched_corrected)
    group_reports = {}
    for group in ("A", "C"):
        group_baseline = _mae([row for row in baseline_rows if row.get("group") == group], *baseline_scale)
        group_corrected = _mae([row for row in corrected_rows if row.get("group") == group], *corrected_scale)
        group_reports[group] = {"mae_increase_ratio": -_improvement_ratio(group_baseline, group_corrected)}

    random_improvements = []
    component_count = len(projection_model["components"])
    for random_seed in random_seeds:
        random_components = _random_component_indices(
            projection_model["ranked_components"],
            component_count,
            selected_k,
            random_seed,
        )
        random_calibration_embeddings = apply_projection(
            calibration_embeddings,
            projection_model,
            k=selected_k,
            component_indices=random_components,
        )
        random_evaluation_embeddings = apply_projection(
            evaluation_embeddings,
            projection_model,
            k=selected_k,
            component_indices=random_components,
        )
        random_calibration_rows = _pair_distances(calibration_pairs, random_calibration_embeddings, reference)
        random_evaluation_rows = _pair_distances(evaluation_pairs, random_evaluation_embeddings, reference)
        random_scale = _fit_affine(random_calibration_rows)
        random_mae = _mae(random_evaluation_rows, *random_scale)
        random_improvements.append(_improvement_ratio(baseline_mae, random_mae))

    passed = (
        improvement >= improvement_threshold
        and _quantile(estimates, 0.025) > 0.0
        and matched_increase <= matched_tolerance
        and -_quantile(matched_estimates, 0.025) <= matched_tolerance
        and max(random_improvements) < improvement_threshold
    )
    return {
        "model_name": model_name,
        "reference_name": reference.get("name", "reference"),
        "selected_k": selected_k,
        "calibration_mae": selection["calibration_mae"],
        "baseline_mae": baseline_mae,
        "corrected_mae": corrected_mae,
        "improvement_ratio": improvement,
        "ci": {"lower": _quantile(estimates, 0.025), "upper": _quantile(estimates, 0.975), "confidence_level": 0.95},
        "matched_speaker_mae_increase_ratio": matched_increase,
        "matched_speaker_ci": {
            "increase_upper": -_quantile(matched_estimates, 0.025),
            "confidence_level": 0.95,
        },
        "matched_speaker_groups": group_reports,
        "random_component_improvement_ratios": random_improvements,
        "random_component_scale_policy": "calibration_refit_per_random_draw",
        "baseline_scale": {"intercept": baseline_scale[0], "slope": baseline_scale[1]},
        "corrected_scale": {"intercept": corrected_scale[0], "slope": corrected_scale[1]},
        "calibration_pair_count": len(calibration_candidates[selected_k]),
        "evaluation_pair_count": len(corrected_rows),
        "status": "passed" if passed else "failed",
    }


def evaluate_pca_projection_gate(
    calibration_embeddings_by_model: Mapping[str, Mapping[str, Sequence[float]]],
    evaluation_embeddings_by_model: Mapping[str, Mapping[str, Sequence[float]]],
    calibration_metadata: Mapping[str, Mapping[str, Any]],
    evaluation_metadata: Mapping[str, Mapping[str, Any]],
    calibration_pairs: Sequence[Mapping[str, Any]],
    evaluation_pairs: Sequence[Mapping[str, Any]],
    *,
    references: Sequence[Mapping[str, Any]],
    k_grid: Sequence[int] = DEFAULT_K_GRID,
    bootstrap_replicates: int = 1000,
    random_seeds: Sequence[int] = DEFAULT_RANDOM_SEEDS,
    seed: int = 20260829,
    improvement_threshold: float = 0.05,
    matched_tolerance: float = 0.01,
) -> dict[str, Any]:
    if bootstrap_replicates < 1000:
        raise ValueError("at least 1000 bootstrap replicates are required")
    reports = []
    for model_name, calibration_embeddings in sorted(calibration_embeddings_by_model.items()):
        if model_name not in evaluation_embeddings_by_model:
            continue
        first_vector = next(iter(calibration_embeddings.values()))
        projection_model = fit_projection_model(calibration_embeddings, calibration_metadata, max_components=len(first_vector))
        reference_reports = [
            _reference_report(
                model_name=model_name,
                projection_model=projection_model,
                calibration_embeddings=calibration_embeddings,
                evaluation_embeddings=evaluation_embeddings_by_model[model_name],
                calibration_pairs=calibration_pairs,
                evaluation_pairs=evaluation_pairs,
                reference=reference,
                k_grid=k_grid,
                bootstrap_replicates=bootstrap_replicates,
                random_seeds=random_seeds,
                seed=seed,
                improvement_threshold=improvement_threshold,
                matched_tolerance=matched_tolerance,
            )
            for reference in references
        ]
        reports.append(
            {
                "model_name": model_name,
                "projection": {
                    "component_count": len(projection_model["components"]),
                    "top_ranked_components": projection_model["ranked_components"][:10],
                    "top_speaker_f_statistics": [
                        projection_model["speaker_f_statistics"][index]
                        for index in projection_model["ranked_components"][:10]
                    ],
                },
                "status": "passed" if reference_reports and all(report["status"] == "passed" for report in reference_reports) else "failed",
                "references": reference_reports,
            }
        )
    status = "passed" if reports and all(report["status"] == "passed" for report in reports) else "failed"
    return {
        "schema": "pca-projection-gate-v1",
        "seed": seed,
        "k_grid": list(k_grid),
        "random_component_seeds": list(random_seeds),
        "thresholds": {
            "correction_improvement": improvement_threshold,
            "matched_speaker_tolerance": matched_tolerance,
        },
        "status": status,
        "decision": "continue_to_manuscript" if status == "passed" else "stop_before_manuscript_and_release",
        "models": reports,
    }


def _load_embedding_file(path: str | Path) -> tuple[str, dict[str, list[float]]]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    return str(payload["model_name"]), {key: [float(value) for value in values] for key, values in payload["embeddings"].items()}


def _load_embeddings(paths: Iterable[str | Path]) -> dict[str, dict[str, list[float]]]:
    loaded = {}
    for path in paths:
        model_name, embeddings = _load_embedding_file(path)
        loaded[model_name] = embeddings
    return loaded


def _load_pairs(path: str | Path) -> list[dict[str, Any]]:
    return json.loads(Path(path).read_text(encoding="utf-8"))["pairs"]


def _load_references(paths: Iterable[str | Path]) -> list[dict[str, Any]]:
    return [json.loads(Path(path).read_text(encoding="utf-8")) for path in paths]


def _metadata_for_pair_manifest(record_manifest_path: str | Path, pair_manifest_path: str | Path) -> dict[str, dict[str, Any]]:
    records = json.loads(Path(record_manifest_path).read_text(encoding="utf-8"))["records"]
    pairs = _load_pairs(pair_manifest_path)
    wanted = {
        utterance_id
        for pair in pairs
        for utterance_id in pair["source_utterance_ids"]
    }
    metadata = {}
    for record in records:
        utterance_id = str(record["utterance_id"])
        if utterance_id in wanted:
            metadata[utterance_id] = {
                "speaker_id": str(record["speaker_id"]),
                "dialect_label": str(record["dialect_label"]),
                "recording_condition": str(record["recording_condition"]),
            }
    missing = sorted(wanted.difference(metadata))
    if missing:
        raise ValueError(f"missing record metadata for utterance: {missing[0]}")
    return metadata


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--calibration-embedding", nargs="+", action="append", required=True)
    parser.add_argument("--evaluation-embedding", nargs="+", action="append", required=True)
    parser.add_argument("--calibration-pairs", required=True)
    parser.add_argument("--evaluation-pairs", required=True)
    parser.add_argument("--records", required=True)
    parser.add_argument("--reference-matrix", nargs="+", action="append", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--seed", type=int, default=20260829)
    parser.add_argument("--bootstrap-replicates", type=int, default=1000)
    args = parser.parse_args(argv)
    args.calibration_embedding = [path for group in args.calibration_embedding for path in group]
    args.evaluation_embedding = [path for group in args.evaluation_embedding for path in group]
    args.reference_matrix = [path for group in args.reference_matrix for path in group]
    return args


def main() -> int:
    args = _parse_args()
    report = evaluate_pca_projection_gate(
        _load_embeddings(args.calibration_embedding),
        _load_embeddings(args.evaluation_embedding),
        _metadata_for_pair_manifest(args.records, args.calibration_pairs),
        _metadata_for_pair_manifest(args.records, args.evaluation_pairs),
        _load_pairs(args.calibration_pairs),
        _load_pairs(args.evaluation_pairs),
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
