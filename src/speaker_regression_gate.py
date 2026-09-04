"""ECAPA-anchored speaker regression correction gate."""

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
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler

from .config import load_config


DEFAULT_ALPHA_GRID = [0.1, 1.0, 10.0, 100.0, 1000.0]
DEFAULT_ABLATION_SEEDS = [20260829, 20260830, 20260831, 20260901, 20260902]
DEFAULT_RANK1_ALS_MAX_ITER = 100
DEFAULT_RANK1_ALS_N_RESTARTS = 5
DEFAULT_RANK1_ALS_CONVERGENCE_TOL = 1e-6
DEFAULT_RANK1_ALS_CONVERGENCE_PATIENCE = 20
DEFAULT_RANK1_SEED = 20260903


def _as_matrix(embeddings: Mapping[str, Sequence[float]], ids: Sequence[str]) -> np.ndarray:
    x = np.asarray([embeddings[utterance_id] for utterance_id in ids], dtype=np.float64)
    if x.ndim != 2 or x.shape[0] != len(ids) or not np.isfinite(x).all():
        raise ValueError("embeddings must be a finite 2D matrix")
    return x


def _speaker_for(utterance_id: str, metadata: Mapping[str, Mapping[str, Any]]) -> str:
    try:
        return str(metadata[utterance_id]["speaker_id"])
    except KeyError as exc:
        raise ValueError(f"missing metadata for utterance: {utterance_id}") from exc


def speaker_centroids(
    embeddings: Mapping[str, Sequence[float]],
    metadata: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    ids = sorted(embeddings)
    x = _as_matrix(embeddings, ids)
    by_speaker: dict[str, list[int]] = defaultdict(list)
    by_cell: dict[tuple[str, str], list[int]] = defaultdict(list)
    for index, utterance_id in enumerate(ids):
        speaker = _speaker_for(utterance_id, metadata)
        dialect = str(metadata[utterance_id]["dialect_label"])
        by_speaker[speaker].append(index)
        by_cell[(speaker, dialect)].append(index)
    centroids = {
        speaker: x[indices].mean(axis=0)
        for speaker, indices in sorted(by_speaker.items())
    }
    cell_means = {
        (speaker, dialect): x[indices].mean(axis=0)
        for (speaker, dialect), indices in sorted(by_cell.items())
    }
    return {
        "speaker_ids": sorted(centroids),
        "speaker_centroids": centroids,
        "speaker_counts": {speaker: len(by_speaker[speaker]) for speaker in sorted(by_speaker)},
        "cell_means": cell_means,
        "cell_counts": {(speaker, dialect): len(indices) for (speaker, dialect), indices in sorted(by_cell.items())},
        "dialects": sorted({dialect for _, dialect in by_cell}),
        "global_mean": x.mean(axis=0),
    }


def _design_matrix(
    x_scaled: np.ndarray,
    dialects: Sequence[str],
    row_dialects: Sequence[str],
    *,
    ecapa_scale: float = 1.0,
    bias_scale: float = 1.0,
) -> np.ndarray:
    one_hot = np.zeros((len(row_dialects), len(dialects)), dtype=np.float64)
    index = {dialect: i for i, dialect in enumerate(dialects)}
    for row, dialect in enumerate(row_dialects):
        if dialect not in index:
            raise ValueError(f"unknown dialect label: {dialect}")
        one_hot[row, index[dialect]] = 1.0
    return np.hstack([x_scaled * float(ecapa_scale), one_hot * float(bias_scale)])


def _fit_with_alpha(
    x: np.ndarray,
    y: np.ndarray,
    alpha: float,
    dialects: Sequence[str],
    row_dialects: Sequence[str],
    *,
    ecapa_scale: float = 1.0,
    bias_scale: float = 1.0,
) -> tuple[StandardScaler, Ridge]:
    scaler = StandardScaler()
    x_scaled = scaler.fit_transform(x)
    model = Ridge(alpha=float(alpha), fit_intercept=False)
    model.fit(
        _design_matrix(
            x_scaled,
            dialects,
            row_dialects,
            ecapa_scale=ecapa_scale,
            bias_scale=bias_scale,
        ),
        y,
    )
    return scaler, model


def _regression_arrays(
    ecapa: Mapping[str, Sequence[float]],
    frozen: Mapping[str, Sequence[float]],
    metadata: Mapping[str, Mapping[str, Any]],
    *,
    ecapa_centroid_override: Mapping[str, np.ndarray] | None = None,
) -> tuple[list[tuple[str, str]], np.ndarray, np.ndarray, np.ndarray, list[str]]:
    ecapa_summary = speaker_centroids(ecapa, metadata)
    frozen_summary = speaker_centroids(frozen, metadata)
    ecapa_centroids = ecapa_centroid_override or ecapa_summary["speaker_centroids"]
    cell_means = frozen_summary["cell_means"]
    dialects = sorted(frozen_summary["dialects"])
    cells = sorted(
        cell_means.keys(),
        key=lambda item: (item[0], item[1]),
    )
    speakers = sorted({speaker for speaker, _ in cells if speaker in ecapa_centroids})
    if len(speakers) < 2:
        raise ValueError("at least two shared speakers are required for ECAPA regression")
    cells = [(speaker, dialect) for speaker, dialect in cells if speaker in ecapa_centroids]
    if len(cells) < 2:
        raise ValueError("at least two shared speaker-dialect cells are required for ECAPA regression")
    x = np.asarray([ecapa_centroids[speaker] for speaker, _ in cells], dtype=np.float64)
    y_centroids = np.asarray([cell_means[(speaker, dialect)] for speaker, dialect in cells], dtype=np.float64)
    global_mean = np.asarray(frozen_summary["global_mean"], dtype=np.float64)
    dialect_main_effect = {
        dialect: np.asarray(
            np.mean([cell_means[cell] for cell in cells if cell[1] == dialect], axis=0),
            dtype=np.float64,
        )
        - global_mean
        for dialect in dialects
    }
    y = np.asarray(
        [cell_mean - global_mean - dialect_main_effect[dialect] for (speaker, dialect), cell_mean in zip(cells, y_centroids)],
        dtype=np.float64,
    )
    return cells, x, y, global_mean, dialects


def _select_alpha(
    x: np.ndarray,
    y: np.ndarray,
    dialects: Sequence[str],
    row_dialects: Sequence[str],
    alpha_grid: Sequence[float],
    inner_cv_folds: int,
    *,
    ecapa_scale: float = 1.0,
    bias_scale: float = 1.0,
) -> float:
    if not alpha_grid or any(alpha <= 0 for alpha in alpha_grid):
        raise ValueError("alpha_grid must contain positive values")
    if len(x) < 3:
        return float(alpha_grid[0])
    folds = min(int(inner_cv_folds), len(x))
    if folds < 2:
        return float(alpha_grid[0])
    splitter = KFold(n_splits=folds, shuffle=True, random_state=20260829)
    best: tuple[float, float] | None = None
    for alpha in alpha_grid:
        losses = []
        for train_index, test_index in splitter.split(x):
            train_dialects = [row_dialects[i] for i in train_index]
            test_dialects = [row_dialects[i] for i in test_index]
            scaler, model = _fit_with_alpha(
                x[train_index],
                y[train_index],
                float(alpha),
                dialects,
                train_dialects,
                ecapa_scale=ecapa_scale,
                bias_scale=bias_scale,
            )
            prediction = model.predict(
                _design_matrix(
                    scaler.transform(x[test_index]),
                    dialects,
                    test_dialects,
                    ecapa_scale=ecapa_scale,
                    bias_scale=bias_scale,
                )
            )
            losses.append(float(np.mean((prediction - y[test_index]) ** 2)))
        candidate = (mean(losses), float(alpha))
        if best is None or candidate < best:
            best = candidate
    assert best is not None
    return best[1]


def fit_speaker_regression(
    ecapa: Mapping[str, Sequence[float]],
    frozen: Mapping[str, Sequence[float]],
    metadata: Mapping[str, Mapping[str, Any]],
    *,
    alpha: float | None = None,
    alpha_grid: Sequence[float] = DEFAULT_ALPHA_GRID,
    inner_cv_folds: int = 5,
    ecapa_centroid_override: Mapping[str, np.ndarray] | None = None,
) -> dict[str, Any]:
    cells, x, y, global_mean, dialects = _regression_arrays(
        ecapa,
        frozen,
        metadata,
        ecapa_centroid_override=ecapa_centroid_override,
    )
    row_dialects = [dialect for _, dialect in cells]
    selected_alpha = float(alpha) if alpha is not None else _select_alpha(
        x, y, dialects, row_dialects, alpha_grid, inner_cv_folds
    )
    scaler, model = _fit_with_alpha(x, y, selected_alpha, dialects, row_dialects)
    prediction = model.predict(_design_matrix(scaler.transform(x), dialects, row_dialects))
    score = float(r2_score(y, prediction, multioutput="variance_weighted"))
    if not math.isfinite(score):
        score = 0.0
    return {
        "model": model,
        "scaler": scaler,
        "selected_alpha": selected_alpha,
        "speaker_ids": sorted({speaker for speaker, _ in cells}),
        "speaker_count": len({speaker for speaker, _ in cells}),
        "cell_keys": cells,
        "cell_count": len(cells),
        "dialects": list(dialects),
        "parameterization": "shared_ridge_per_dialect_bias",
        "target_source": "cell_offset_minus_dialect_main_effect",
        "shared_weight_shape": tuple(model.coef_[:, : x.shape[1]].T.shape),
        "dialect_bias": {
            dialect: model.coef_[:, x.shape[1] + index].copy()
            for index, dialect in enumerate(dialects)
        },
        "frozen_global_mean": global_mean,
        "predicted_component_mean": prediction.mean(axis=0),
        "target_r2": score,
        "speaker_centroid_r2": score,
        "target_mse": float(np.mean((prediction - y) ** 2)),
        "speaker_centroid_mse": float(np.mean((prediction - y) ** 2)),
    }


def _predict_component(
    fitted: Mapping[str, Any],
    centroid: np.ndarray,
    dialect: str,
    *,
    dialect_bias_override: Mapping[str, np.ndarray] | None = None,
) -> np.ndarray:
    x = np.asarray([centroid], dtype=np.float64)
    bias = fitted["dialect_bias"] if dialect_bias_override is None else dialect_bias_override
    if dialect not in bias:
        raise ValueError(f"unknown dialect label: {dialect}")
    base = np.asarray(
        fitted["model"].coef_[:, : x.shape[1]] @ fitted["scaler"].transform(x)[0] * float(fitted.get("feature_scales", {}).get("ecapa", 1.0)),
        dtype=np.float64,
    )
    return base + np.asarray(bias[dialect], dtype=np.float64) * float(fitted.get("feature_scales", {}).get("bias", 1.0))


def apply_speaker_regression(
    frozen_embeddings: Mapping[str, Sequence[float]],
    ecapa_embeddings: Mapping[str, Sequence[float]],
    metadata: Mapping[str, Mapping[str, Any]],
    fitted_model: Mapping[str, Any],
    *,
    ecapa_centroid_override: Mapping[str, np.ndarray] | None = None,
    dialect_bias_override: Mapping[str, np.ndarray] | None = None,
) -> dict[str, list[float]]:
    centroids = ecapa_centroid_override or speaker_centroids(ecapa_embeddings, metadata)["speaker_centroids"]
    anchor = np.asarray(fitted_model["predicted_component_mean"], dtype=np.float64)
    cleaned: dict[str, list[float]] = {}
    for utterance_id in sorted(frozen_embeddings):
        speaker = _speaker_for(utterance_id, metadata)
        dialect = str(metadata[utterance_id]["dialect_label"])
        if speaker not in centroids:
            raise ValueError(f"missing ECAPA centroid for speaker: {speaker}")
        vector = np.asarray(frozen_embeddings[utterance_id], dtype=np.float64)
        component = _predict_component(
            fitted_model,
            np.asarray(centroids[speaker], dtype=np.float64),
            dialect,
            dialect_bias_override=dialect_bias_override,
        )
        cleaned[utterance_id] = (vector - component + anchor).astype(float).tolist()
    return cleaned


def leave_pair_out_ecapa_centroids(
    ecapa_embeddings: Mapping[str, Sequence[float]],
    frozen_embeddings: Mapping[str, Sequence[float]],
    metadata: Mapping[str, Mapping[str, Any]],
    pairs: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    del frozen_embeddings
    ids = sorted(ecapa_embeddings)
    x = _as_matrix(ecapa_embeddings, ids)
    id_to_index = {utterance_id: index for index, utterance_id in enumerate(ids)}
    speaker_sums: dict[str, np.ndarray] = {}
    speaker_counts: dict[str, int] = defaultdict(int)
    for index, utterance_id in enumerate(ids):
        speaker = _speaker_for(utterance_id, metadata)
        speaker_sums.setdefault(speaker, np.zeros(x.shape[1], dtype=np.float64))
        speaker_sums[speaker] += x[index]
        speaker_counts[speaker] += 1
    fallback_endpoint_count = 0
    fallback_pair_count = 0
    endpoint_count = 0
    for pair in pairs:
        pair_ids = [str(value) for value in pair["source_utterance_ids"]]
        excluded_sums: dict[str, np.ndarray] = {}
        excluded_counts: dict[str, int] = defaultdict(int)
        for utterance_id in set(pair_ids):
            if utterance_id not in id_to_index:
                raise ValueError(f"missing ECAPA embedding for utterance: {utterance_id}")
            speaker = _speaker_for(utterance_id, metadata)
            excluded_sums.setdefault(speaker, np.zeros(x.shape[1], dtype=np.float64))
            excluded_sums[speaker] += x[id_to_index[utterance_id]]
            excluded_counts[speaker] += 1
        pair_fallbacks = 0
        for utterance_id in pair_ids:
            speaker = _speaker_for(utterance_id, metadata)
            if speaker_counts[speaker] - excluded_counts[speaker] <= 0:
                pair_fallbacks += 1
        endpoint_count += len(pair_ids)
        fallback_endpoint_count += pair_fallbacks
        fallback_pair_count += 1 if pair_fallbacks else 0
    return {
        "fallback_endpoint_count": fallback_endpoint_count,
        "fallback_pair_count": fallback_pair_count,
        "endpoint_count": endpoint_count,
        "fallback_endpoint_ratio": fallback_endpoint_count / endpoint_count if endpoint_count else 0.0,
        "fallback_pair_ratio": fallback_pair_count / len(pairs) if pairs else 0.0,
    }


def _cosine_distance(a: Sequence[float], b: Sequence[float]) -> float:
    left = np.asarray(a, dtype=np.float64)
    right = np.asarray(b, dtype=np.float64)
    left_norm = float(np.linalg.norm(left))
    right_norm = float(np.linalg.norm(right))
    if left_norm == 0.0 or right_norm == 0.0:
        raise ValueError("cosine distance is undefined for zero vectors")
    value = 1.0 - float(np.dot(left, right) / (left_norm * right_norm))
    return float(min(2.0, max(0.0, value)))


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


def pair_distances(
    pairs: Sequence[Mapping[str, Any]],
    embeddings: Mapping[str, Sequence[float]],
    reference: Mapping[str, Any],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for pair in pairs:
        left_id, right_id = [str(value) for value in pair["source_utterance_ids"]]
        if left_id not in embeddings or right_id not in embeddings:
            raise ValueError(f"missing embedding for pair: {pair.get('pair_id')}")
        labels = [str(value) for value in pair["dialect_labels"]]
        target = _target_distance(labels, reference)
        if target is None:
            continue
        rows.append(
            {
                "pair_id": str(pair["pair_id"]),
                "group": pair.get("group"),
                "dialect_labels": labels,
                "distance": _cosine_distance(embeddings[left_id], embeddings[right_id]),
                "target_distance": target,
            }
        )
    return rows


def _fit_affine(rows: Sequence[Mapping[str, Any]]) -> tuple[float, float]:
    xs = [float(row["distance"]) for row in rows]
    ys = [float(row["target_distance"]) for row in rows]
    if not xs:
        raise ValueError("cannot fit affine scale without rows")
    x_mean = mean(xs)
    y_mean = mean(ys)
    variance = sum((x - x_mean) ** 2 for x in xs)
    if variance <= 0.0:
        return y_mean, 0.0
    covariance = sum((x - x_mean) * (y - y_mean) for x, y in zip(xs, ys))
    slope = max(0.0, covariance / variance)
    return y_mean - slope * x_mean, slope


def _mae(rows: Sequence[Mapping[str, Any]], intercept: float, slope: float) -> float:
    if not rows:
        return 0.0
    return mean(abs(intercept + slope * float(row["distance"]) - float(row["target_distance"])) for row in rows)


def _improvement_ratio(baseline_mae: float, corrected_mae: float) -> float:
    if baseline_mae <= 1e-12:
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


def _group_report(
    group: str,
    baseline_rows: Sequence[Mapping[str, Any]],
    corrected_rows: Sequence[Mapping[str, Any]],
    baseline_scale: tuple[float, float],
    corrected_scale: tuple[float, float],
) -> dict[str, float]:
    baseline_group = [row for row in baseline_rows if row.get("group") == group]
    corrected_group = [row for row in corrected_rows if row.get("group") == group]
    baseline_mae = _mae(baseline_group, *baseline_scale)
    corrected_mae = _mae(corrected_group, *corrected_scale)
    improvement = _improvement_ratio(baseline_mae, corrected_mae)
    return {
        "baseline_mae": baseline_mae,
        "corrected_mae": corrected_mae,
        "improvement_ratio": improvement,
        "mae_increase_ratio": -improvement,
    }


def _wrong_centroids(centroids: Mapping[str, np.ndarray], seed: int) -> dict[str, np.ndarray]:
    speakers = sorted(centroids)
    if len(speakers) <= 1:
        return {speaker: np.asarray(value, dtype=np.float64) for speaker, value in centroids.items()}
    shuffled = speakers[:]
    rng = random.Random(seed)
    for _ in range(32):
        rng.shuffle(shuffled)
        if all(left != right for left, right in zip(speakers, shuffled)):
            break
    if any(left == right for left, right in zip(speakers, shuffled)):
        shuffled = speakers[1:] + speakers[:1]
    return {speaker: np.asarray(centroids[other], dtype=np.float64) for speaker, other in zip(speakers, shuffled)}


def _shuffled_centroids(centroids: Mapping[str, np.ndarray], seed: int) -> dict[str, np.ndarray]:
    speakers = sorted(centroids)
    shuffled = speakers[:]
    random.Random(seed).shuffle(shuffled)
    return {speaker: np.asarray(centroids[other], dtype=np.float64) for speaker, other in zip(speakers, shuffled)}


def _swapped_dialect_bias(bias: Mapping[str, np.ndarray]) -> dict[str, np.ndarray]:
    dialects = sorted(bias)
    if len(dialects) <= 1:
        return {dialect: np.asarray(value, dtype=np.float64) for dialect, value in bias.items()}
    rotated = dialects[1:] + dialects[:1]
    return {
        dialect: np.asarray(bias[other], dtype=np.float64)
        for dialect, other in zip(dialects, rotated)
    }


def _swapped_rank1_scale(scale: Mapping[str, float]) -> dict[str, float]:
    dialects = sorted(scale)
    if len(dialects) <= 1:
        return {dialect: float(value) for dialect, value in scale.items()}
    rotated = dialects[1:] + dialects[:1]
    return {dialect: float(scale[other]) for dialect, other in zip(dialects, rotated)}


def _fit_rank1_residual_factorization(
    x_scaled: np.ndarray,
    residual: np.ndarray,
    row_dialects: Sequence[str],
    dialects: Sequence[str],
    *,
    alpha: float,
    max_iter: int,
    n_restarts: int,
    convergence_tol: float,
    convergence_patience: int,
    seed: int,
) -> dict[str, Any]:
    if residual.ndim != 2 or x_scaled.ndim != 2:
        raise ValueError("rank-1 factorization requires 2D arrays")
    if not dialects:
        raise ValueError("at least one dialect is required")
    if len(row_dialects) != len(x_scaled) or len(row_dialects) != len(residual):
        raise ValueError("row_dialects must align with x_scaled and residual")
    residual_variance = float(np.var(residual, axis=0, dtype=np.float64).sum())
    if residual_variance <= 1e-12:
        zero_output = np.zeros(residual.shape[1], dtype=np.float64)
        zero_input = np.zeros(x_scaled.shape[1], dtype=np.float64)
        return {
            "rank1_output_direction": zero_output,
            "rank1_input_direction": zero_input,
            "rank1_scale": {dialect: 1.0 for dialect in dialects},
            "rank1_score": 0.0,
            "rank1_iterations": 0,
            "rank1_restart": 0,
            "predicted_residual": np.zeros_like(residual, dtype=np.float64),
            "score": 0.0,
        }
    dialect_index = {dialect: index for index, dialect in enumerate(dialects)}
    row_indices = np.asarray([dialect_index[dialect] for dialect in row_dialects], dtype=np.int64)
    rng = np.random.default_rng(seed)
    best_state: dict[str, Any] | None = None

    def _normalized(vector: np.ndarray) -> np.ndarray:
        norm = float(np.linalg.norm(vector))
        if norm <= 0.0:
            vector = vector.copy()
            vector[0] = 1.0
            norm = float(np.linalg.norm(vector))
        return vector / norm

    for restart in range(max(1, int(n_restarts))):
        output_direction = _normalized(rng.normal(size=residual.shape[1]).astype(np.float64))
        input_direction = _normalized(rng.normal(size=x_scaled.shape[1]).astype(np.float64))
        dialect_scale = {dialect: 1.0 for dialect in dialects}
        best_local: dict[str, Any] | None = None
        stale = 0
        previous_score: float | None = None
        for iteration in range(max(1, int(max_iter))):
            scale_row = np.asarray([dialect_scale[dialect] for dialect in row_dialects], dtype=np.float64)
            latent = (x_scaled @ input_direction) * scale_row
            latent_norm = float(np.dot(latent, latent))
            if latent_norm > 0.0:
                output_direction = (latent[:, None] * residual).sum(axis=0) / latent_norm
                output_direction = _normalized(output_direction.astype(np.float64))
            target_projection = residual @ output_direction
            latent_scale = scale_row[:, None]
            model = Ridge(alpha=float(alpha), fit_intercept=False)
            model.fit(x_scaled * latent_scale, target_projection)
            input_direction = _normalized(np.asarray(model.coef_, dtype=np.float64))
            projected = x_scaled @ input_direction
            for dialect in dialects:
                mask = row_indices == dialect_index[dialect]
                if not np.any(mask):
                    continue
                denom = float(np.dot(projected[mask], projected[mask]))
                dialect_scale[dialect] = (
                    float(np.dot(target_projection[mask], projected[mask]) / denom)
                    if denom > 0.0
                    else 0.0
                )
            scale_row = np.asarray([dialect_scale[dialect] for dialect in row_dialects], dtype=np.float64)
            latent = projected * scale_row
            predicted = latent[:, None] * output_direction[None, :]
            score = (
                float(r2_score(residual, predicted, multioutput="variance_weighted"))
                if residual_variance > 0.0
                else 0.0
            )
            if not math.isfinite(score):
                score = 0.0
            if best_local is None or score > best_local["score"] + convergence_tol:
                best_local = {
                    "rank1_output_direction": output_direction.copy(),
                    "rank1_input_direction": input_direction.copy(),
                    "rank1_scale": {dialect: float(value) for dialect, value in dialect_scale.items()},
                    "rank1_score": score,
                    "rank1_iterations": iteration + 1,
                    "score": score,
                    "predicted_residual": predicted.copy(),
                }
            if previous_score is not None and abs(score - previous_score) < convergence_tol:
                stale += 1
                if stale >= convergence_patience:
                    break
            else:
                stale = 0
            previous_score = score
        if best_local is not None and (best_state is None or best_local["score"] > best_state["score"]):
            best_state = {
                "rank1_output_direction": best_local["rank1_output_direction"],
                "rank1_input_direction": best_local["rank1_input_direction"],
                "rank1_scale": best_local["rank1_scale"],
                "rank1_score": best_local["rank1_score"],
                "rank1_iterations": best_local["rank1_iterations"],
                "rank1_restart": restart,
                "predicted_residual": best_local["predicted_residual"],
                "score": best_local["score"],
            }
    if best_state is None:
        raise RuntimeError("rank-1 ALS did not converge")
    return best_state


def fit_rank1_dialect_modulation(
    ecapa: Mapping[str, Sequence[float]],
    frozen: Mapping[str, Sequence[float]],
    metadata: Mapping[str, Mapping[str, Any]],
    *,
    alpha: float | None = None,
    alpha_grid: Sequence[float] = DEFAULT_ALPHA_GRID,
    inner_cv_folds: int = 5,
    rank: int = 1,
    als_max_iter: int = DEFAULT_RANK1_ALS_MAX_ITER,
    als_n_restarts: int = DEFAULT_RANK1_ALS_N_RESTARTS,
    als_convergence_tol: float = DEFAULT_RANK1_ALS_CONVERGENCE_TOL,
    als_convergence_patience: int = DEFAULT_RANK1_ALS_CONVERGENCE_PATIENCE,
    seed: int = DEFAULT_RANK1_SEED,
    ecapa_centroid_override: Mapping[str, np.ndarray] | None = None,
    parameterization: str = "shared_ridge_plus_rank1_dialect_modulation",
    regularization_family: str | None = None,
    w_penalty_multiplier: float = 1.0,
    bias_penalty_multiplier: float = 1.0,
    low_rank_penalty_multiplier: float = 1.0,
) -> dict[str, Any]:
    if rank != 1:
        raise ValueError("rank-1 dialect modulation requires rank=1")
    cells, x, y, global_mean, dialects = _regression_arrays(
        ecapa,
        frozen,
        metadata,
        ecapa_centroid_override=ecapa_centroid_override,
    )
    row_dialects = [dialect for _, dialect in cells]
    ecapa_scale = math.sqrt(1.0 / float(w_penalty_multiplier))
    bias_scale = math.sqrt(1.0 / float(bias_penalty_multiplier))
    selected_alpha = float(alpha) if alpha is not None else _select_alpha(
        x,
        y,
        dialects,
        row_dialects,
        alpha_grid,
        inner_cv_folds,
        ecapa_scale=ecapa_scale,
        bias_scale=bias_scale,
    )
    base_scaler, base_model = _fit_with_alpha(
        x,
        y,
        selected_alpha,
        dialects,
        row_dialects,
        ecapa_scale=ecapa_scale,
        bias_scale=bias_scale,
    )
    x_scaled = base_scaler.transform(x)
    base_prediction = base_model.predict(
        _design_matrix(
            x_scaled,
            dialects,
            row_dialects,
            ecapa_scale=ecapa_scale,
            bias_scale=bias_scale,
        )
    )
    residual = y - base_prediction
    low_rank = _fit_rank1_residual_factorization(
        x_scaled,
        residual,
        row_dialects,
        dialects,
        alpha=selected_alpha * float(low_rank_penalty_multiplier),
        max_iter=als_max_iter,
        n_restarts=als_n_restarts,
        convergence_tol=als_convergence_tol,
        convergence_patience=als_convergence_patience,
        seed=seed,
    )
    combined_prediction = base_prediction + low_rank["predicted_residual"]
    score = float(r2_score(y, combined_prediction, multioutput="variance_weighted"))
    if not math.isfinite(score):
        score = 0.0
    return {
        "model": base_model,
        "scaler": base_scaler,
        "selected_alpha": selected_alpha,
        "base_alpha": selected_alpha,
        "feature_scales": {"ecapa": ecapa_scale, "bias": bias_scale, "low_rank": float(low_rank_penalty_multiplier)},
        "speaker_ids": sorted({speaker for speaker, _ in cells}),
        "speaker_count": len({speaker for speaker, _ in cells}),
        "cell_keys": cells,
        "cell_count": len(cells),
        "dialects": list(dialects),
        "parameterization": parameterization,
        "regularization_family": regularization_family,
        "w_penalty_multiplier": float(w_penalty_multiplier),
        "bias_penalty_multiplier": float(bias_penalty_multiplier),
        "low_rank_penalty_multiplier": float(low_rank_penalty_multiplier),
        "target_source": "cell_offset_minus_dialect_main_effect",
        "rank": 1,
        "als_max_iter": int(als_max_iter),
        "als_n_restarts": int(als_n_restarts),
        "als_convergence_tol": float(als_convergence_tol),
        "als_convergence_patience": int(als_convergence_patience),
        "shared_weight_shape": tuple(base_model.coef_[:, : x.shape[1]].T.shape),
        "dialect_bias": {
            dialect: base_model.coef_[:, x.shape[1] + index].copy()
            for index, dialect in enumerate(dialects)
        },
        "rank1_output_direction": low_rank["rank1_output_direction"].copy(),
        "rank1_input_direction": low_rank["rank1_input_direction"].copy(),
        "rank1_scale": {dialect: float(value) for dialect, value in low_rank["rank1_scale"].items()},
        "frozen_global_mean": global_mean,
        "predicted_component_mean": combined_prediction.mean(axis=0),
        "base_component_mean": base_prediction.mean(axis=0),
        "rank1_component_mean": low_rank["predicted_residual"].mean(axis=0),
        "target_r2": score,
        "speaker_centroid_r2": score,
        "target_mse": float(np.mean((combined_prediction - y) ** 2)),
        "speaker_centroid_mse": float(np.mean((combined_prediction - y) ** 2)),
        "rank1_r2": float(low_rank["rank1_score"]),
        "rank1_iterations": int(low_rank["rank1_iterations"]),
        "rank1_restart": int(low_rank["rank1_restart"]),
    }


def _predict_rank1_component(
    fitted: Mapping[str, Any],
    centroid: np.ndarray,
    dialect: str,
    *,
    rank1_scale_override: Mapping[str, float] | None = None,
) -> np.ndarray:
    scale = fitted["rank1_scale"] if rank1_scale_override is None else rank1_scale_override
    if dialect not in scale:
        raise ValueError(f"unknown dialect label: {dialect}")
    x = np.asarray([centroid], dtype=np.float64)
    latent = float(np.dot(np.asarray(fitted["rank1_input_direction"], dtype=np.float64), fitted["scaler"].transform(x)[0]))
    return np.asarray(fitted["rank1_output_direction"], dtype=np.float64) * (latent * float(scale[dialect]))


def apply_rank1_dialect_modulation(
    frozen_embeddings: Mapping[str, Sequence[float]],
    ecapa_embeddings: Mapping[str, Sequence[float]],
    metadata: Mapping[str, Mapping[str, Any]],
    fitted_model: Mapping[str, Any],
    *,
    ecapa_centroid_override: Mapping[str, np.ndarray] | None = None,
    dialect_bias_override: Mapping[str, np.ndarray] | None = None,
    rank1_scale_override: Mapping[str, float] | None = None,
    interaction_disabled: bool = False,
) -> dict[str, list[float]]:
    centroids = ecapa_centroid_override or speaker_centroids(ecapa_embeddings, metadata)["speaker_centroids"]
    anchor = np.asarray(fitted_model["predicted_component_mean"], dtype=np.float64)
    cleaned: dict[str, list[float]] = {}
    for utterance_id in sorted(frozen_embeddings):
        speaker = _speaker_for(utterance_id, metadata)
        dialect = str(metadata[utterance_id]["dialect_label"])
        if speaker not in centroids:
            raise ValueError(f"missing ECAPA centroid for speaker: {speaker}")
        vector = np.asarray(frozen_embeddings[utterance_id], dtype=np.float64)
        base_component = _predict_component(
            fitted_model,
            np.asarray(centroids[speaker], dtype=np.float64),
            dialect,
            dialect_bias_override=dialect_bias_override,
        )
        low_rank_component = (
            np.zeros_like(base_component)
            if interaction_disabled
            else _predict_rank1_component(
                fitted_model,
                np.asarray(centroids[speaker], dtype=np.float64),
                dialect,
                rank1_scale_override=rank1_scale_override,
            )
        )
        cleaned[utterance_id] = (vector - base_component - low_rank_component + anchor).astype(float).tolist()
    return cleaned


def fit_block_regularized_rank1_dialect_modulation(
    ecapa: Mapping[str, Sequence[float]],
    frozen: Mapping[str, Sequence[float]],
    metadata: Mapping[str, Mapping[str, Any]],
    *,
    base_alpha: float | None = None,
    alpha_grid: Sequence[float] = (1.0, 3.0, 10.0, 30.0),
    inner_cv_folds: int = 5,
    rank: int = 1,
    als_max_iter: int = DEFAULT_RANK1_ALS_MAX_ITER,
    als_n_restarts: int = DEFAULT_RANK1_ALS_N_RESTARTS,
    als_convergence_tol: float = DEFAULT_RANK1_ALS_CONVERGENCE_TOL,
    als_convergence_patience: int = DEFAULT_RANK1_ALS_CONVERGENCE_PATIENCE,
    seed: int = DEFAULT_RANK1_SEED,
    ecapa_centroid_override: Mapping[str, np.ndarray] | None = None,
    w_penalty_multiplier: float = 100.0,
    bias_penalty_multiplier: float = 10.0,
    low_rank_penalty_multiplier: float = 1.0,
) -> dict[str, Any]:
    return fit_rank1_dialect_modulation(
        ecapa,
        frozen,
        metadata,
        alpha=base_alpha,
        alpha_grid=alpha_grid,
        inner_cv_folds=inner_cv_folds,
        rank=rank,
        als_max_iter=als_max_iter,
        als_n_restarts=als_n_restarts,
        als_convergence_tol=als_convergence_tol,
        als_convergence_patience=als_convergence_patience,
        seed=seed,
        ecapa_centroid_override=ecapa_centroid_override,
        parameterization="block_regularized_shared_ridge_plus_rank1_dialect_modulation",
        regularization_family="block_diagonal_ridge",
        w_penalty_multiplier=w_penalty_multiplier,
        bias_penalty_multiplier=bias_penalty_multiplier,
        low_rank_penalty_multiplier=low_rank_penalty_multiplier,
    )


def apply_block_regularized_rank1_dialect_modulation(
    frozen_embeddings: Mapping[str, Sequence[float]],
    ecapa_embeddings: Mapping[str, Sequence[float]],
    metadata: Mapping[str, Mapping[str, Any]],
    fitted_model: Mapping[str, Any],
    *,
    ecapa_centroid_override: Mapping[str, np.ndarray] | None = None,
    dialect_bias_override: Mapping[str, np.ndarray] | None = None,
    rank1_scale_override: Mapping[str, float] | None = None,
    interaction_disabled: bool = False,
) -> dict[str, list[float]]:
    return apply_rank1_dialect_modulation(
        frozen_embeddings,
        ecapa_embeddings,
        metadata,
        fitted_model,
        ecapa_centroid_override=ecapa_centroid_override,
        dialect_bias_override=dialect_bias_override,
        rank1_scale_override=rank1_scale_override,
        interaction_disabled=interaction_disabled,
    )


def _ablation_improvements(
    *,
    calibration_frozen: Mapping[str, Sequence[float]],
    evaluation_frozen: Mapping[str, Sequence[float]],
    calibration_ecapa: Mapping[str, Sequence[float]],
    evaluation_ecapa: Mapping[str, Sequence[float]],
    calibration_metadata: Mapping[str, Mapping[str, Any]],
    evaluation_metadata: Mapping[str, Mapping[str, Any]],
    calibration_pairs: Sequence[Mapping[str, Any]],
    evaluation_pairs: Sequence[Mapping[str, Any]],
    reference: Mapping[str, Any],
    baseline_mae: float,
    alpha_grid: Sequence[float],
    inner_cv_folds: int,
    seeds: Sequence[int],
    mode: str,
) -> list[float]:
    improvements: list[float] = []
    for seed in seeds:
        calibration_centroids = speaker_centroids(calibration_ecapa, calibration_metadata)["speaker_centroids"]
        evaluation_centroids = speaker_centroids(evaluation_ecapa, evaluation_metadata)["speaker_centroids"]
        if mode == "wrong_centroid":
            calibration_override = _wrong_centroids(calibration_centroids, seed)
            evaluation_override = _wrong_centroids(evaluation_centroids, seed + 1000)
        elif mode == "dialect_bias_swap":
            calibration_override = calibration_centroids
            evaluation_override = evaluation_centroids
        else:
            calibration_override = _shuffled_centroids(calibration_centroids, seed)
            evaluation_override = _shuffled_centroids(evaluation_centroids, seed + 1000)
        fitted = fit_speaker_regression(
            calibration_ecapa,
            calibration_frozen,
            calibration_metadata,
            alpha_grid=alpha_grid,
            inner_cv_folds=inner_cv_folds,
            ecapa_centroid_override=calibration_override,
        )
        bias_override = _swapped_dialect_bias(fitted["dialect_bias"]) if mode == "dialect_bias_swap" else None
        cleaned_calibration = apply_speaker_regression(
            calibration_frozen,
            calibration_ecapa,
            calibration_metadata,
            fitted,
            ecapa_centroid_override=calibration_override,
            dialect_bias_override=bias_override,
        )
        cleaned_evaluation = apply_speaker_regression(
            evaluation_frozen,
            evaluation_ecapa,
            evaluation_metadata,
            fitted,
            ecapa_centroid_override=evaluation_override,
            dialect_bias_override=bias_override,
        )
        ablation_calibration_rows = pair_distances(calibration_pairs, cleaned_calibration, reference)
        ablation_evaluation_rows = pair_distances(evaluation_pairs, cleaned_evaluation, reference)
        ablation_scale = _fit_affine(ablation_calibration_rows)
        improvements.append(_improvement_ratio(baseline_mae, _mae(ablation_evaluation_rows, *ablation_scale)))
    return improvements


def _leave_pair_out_rows(
    pairs: Sequence[Mapping[str, Any]],
    frozen_embeddings: Mapping[str, Sequence[float]],
    ecapa_embeddings: Mapping[str, Sequence[float]],
    metadata: Mapping[str, Mapping[str, Any]],
    fitted: Mapping[str, Any],
    reference: Mapping[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    ecapa_ids = sorted(ecapa_embeddings)
    ecapa_x = _as_matrix(ecapa_embeddings, ecapa_ids)
    id_to_index = {utterance_id: index for index, utterance_id in enumerate(ecapa_ids)}
    speaker_sums: dict[str, np.ndarray] = {}
    speaker_counts: dict[str, int] = defaultdict(int)
    for index, utterance_id in enumerate(ecapa_ids):
        speaker = _speaker_for(utterance_id, metadata)
        speaker_sums.setdefault(speaker, np.zeros(ecapa_x.shape[1], dtype=np.float64))
        speaker_sums[speaker] += ecapa_x[index]
        speaker_counts[speaker] += 1
    anchor = np.asarray(fitted["predicted_component_mean"], dtype=np.float64)
    rows: list[dict[str, Any]] = []
    fallback_endpoint_count = 0
    fallback_pair_count = 0
    endpoint_count = 0
    for pair in pairs:
        labels = [str(value) for value in pair["dialect_labels"]]
        target = _target_distance(labels, reference)
        if target is None:
            continue
        pair_ids = [str(value) for value in pair["source_utterance_ids"]]
        excluded_sums: dict[str, np.ndarray] = {}
        excluded_counts: dict[str, int] = defaultdict(int)
        for utterance_id in set(pair_ids):
            if utterance_id not in id_to_index:
                raise ValueError(f"missing ECAPA embedding for utterance: {utterance_id}")
            speaker = _speaker_for(utterance_id, metadata)
            excluded_sums.setdefault(speaker, np.zeros(ecapa_x.shape[1], dtype=np.float64))
            excluded_sums[speaker] += ecapa_x[id_to_index[utterance_id]]
            excluded_counts[speaker] += 1
        cleaned_vectors = []
        pair_fallback = 0
        for utterance_id in pair_ids:
            speaker = _speaker_for(utterance_id, metadata)
            dialect = str(metadata[utterance_id]["dialect_label"])
            support_count = speaker_counts[speaker] - excluded_counts[speaker]
            if support_count > 0:
                centroid = (speaker_sums[speaker] - excluded_sums[speaker]) / support_count
            else:
                centroid = speaker_sums[speaker] / speaker_counts[speaker]
                pair_fallback += 1
            frozen_vector = np.asarray(frozen_embeddings[utterance_id], dtype=np.float64)
            cleaned_vectors.append(
                frozen_vector
                - _predict_component(fitted, centroid, dialect)
                + anchor
            )
        endpoint_count += len(pair_ids)
        fallback_endpoint_count += pair_fallback
        fallback_pair_count += 1 if pair_fallback else 0
        rows.append(
            {
                "pair_id": str(pair["pair_id"]),
                "group": pair.get("group"),
                "dialect_labels": labels,
                "distance": _cosine_distance(cleaned_vectors[0], cleaned_vectors[1]),
                "target_distance": target,
            }
        )
    return rows, {
        "fallback_endpoint_count": fallback_endpoint_count,
        "fallback_pair_count": fallback_pair_count,
        "endpoint_count": endpoint_count,
        "fallback_endpoint_ratio": fallback_endpoint_count / endpoint_count if endpoint_count else 0.0,
        "fallback_pair_ratio": fallback_pair_count / len(rows) if rows else 0.0,
    }


def _rank1_leave_pair_out_rows(
    pairs: Sequence[Mapping[str, Any]],
    frozen_embeddings: Mapping[str, Sequence[float]],
    ecapa_embeddings: Mapping[str, Sequence[float]],
    metadata: Mapping[str, Mapping[str, Any]],
    fitted: Mapping[str, Any],
    reference: Mapping[str, Any],
    *,
    dialect_bias_override: Mapping[str, np.ndarray] | None = None,
    rank1_scale_override: Mapping[str, float] | None = None,
    interaction_disabled: bool = False,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    ecapa_ids = sorted(ecapa_embeddings)
    ecapa_x = _as_matrix(ecapa_embeddings, ecapa_ids)
    id_to_index = {utterance_id: index for index, utterance_id in enumerate(ecapa_ids)}
    speaker_sums: dict[str, np.ndarray] = {}
    speaker_counts: dict[str, int] = defaultdict(int)
    for index, utterance_id in enumerate(ecapa_ids):
        speaker = _speaker_for(utterance_id, metadata)
        speaker_sums.setdefault(speaker, np.zeros(ecapa_x.shape[1], dtype=np.float64))
        speaker_sums[speaker] += ecapa_x[index]
        speaker_counts[speaker] += 1
    anchor = np.asarray(fitted["predicted_component_mean"], dtype=np.float64)
    rows: list[dict[str, Any]] = []
    fallback_endpoint_count = 0
    fallback_pair_count = 0
    endpoint_count = 0
    for pair in pairs:
        labels = [str(value) for value in pair["dialect_labels"]]
        target = _target_distance(labels, reference)
        if target is None:
            continue
        pair_ids = [str(value) for value in pair["source_utterance_ids"]]
        excluded_sums: dict[str, np.ndarray] = {}
        excluded_counts: dict[str, int] = defaultdict(int)
        for utterance_id in set(pair_ids):
            if utterance_id not in id_to_index:
                raise ValueError(f"missing ECAPA embedding for utterance: {utterance_id}")
            speaker = _speaker_for(utterance_id, metadata)
            excluded_sums.setdefault(speaker, np.zeros(ecapa_x.shape[1], dtype=np.float64))
            excluded_sums[speaker] += ecapa_x[id_to_index[utterance_id]]
            excluded_counts[speaker] += 1
        cleaned_vectors = []
        pair_fallback = 0
        for utterance_id in pair_ids:
            speaker = _speaker_for(utterance_id, metadata)
            dialect = str(metadata[utterance_id]["dialect_label"])
            support_count = speaker_counts[speaker] - excluded_counts[speaker]
            if support_count > 0:
                centroid = (speaker_sums[speaker] - excluded_sums[speaker]) / support_count
            else:
                centroid = speaker_sums[speaker] / speaker_counts[speaker]
                pair_fallback += 1
            frozen_vector = np.asarray(frozen_embeddings[utterance_id], dtype=np.float64)
            base_component = _predict_component(
                fitted,
                centroid,
                dialect,
                dialect_bias_override=dialect_bias_override,
            )
            low_rank_component = (
                np.zeros_like(base_component)
                if interaction_disabled
                else _predict_rank1_component(
                    fitted,
                    centroid,
                    dialect,
                    rank1_scale_override=rank1_scale_override,
                )
            )
            cleaned_vectors.append(frozen_vector - base_component - low_rank_component + anchor)
        endpoint_count += len(pair_ids)
        fallback_endpoint_count += pair_fallback
        fallback_pair_count += 1 if pair_fallback else 0
        rows.append(
            {
                "pair_id": str(pair["pair_id"]),
                "group": pair.get("group"),
                "dialect_labels": labels,
                "distance": _cosine_distance(cleaned_vectors[0], cleaned_vectors[1]),
                "target_distance": target,
            }
        )
    return rows, {
        "fallback_endpoint_count": fallback_endpoint_count,
        "fallback_pair_count": fallback_pair_count,
        "endpoint_count": endpoint_count,
        "fallback_endpoint_ratio": fallback_endpoint_count / endpoint_count if endpoint_count else 0.0,
        "fallback_pair_ratio": fallback_pair_count / len(rows) if rows else 0.0,
    }


def _rank1_ablation_improvements(
    *,
    calibration_frozen: Mapping[str, Sequence[float]],
    evaluation_frozen: Mapping[str, Sequence[float]],
    calibration_ecapa: Mapping[str, Sequence[float]],
    evaluation_ecapa: Mapping[str, Sequence[float]],
    calibration_metadata: Mapping[str, Mapping[str, Any]],
    evaluation_metadata: Mapping[str, Mapping[str, Any]],
    calibration_pairs: Sequence[Mapping[str, Any]],
    evaluation_pairs: Sequence[Mapping[str, Any]],
    reference: Mapping[str, Any],
    baseline_mae: float,
    alpha_grid: Sequence[float],
    inner_cv_folds: int,
    als_max_iter: int,
    als_n_restarts: int,
    als_convergence_tol: float,
    als_convergence_patience: int,
    seeds: Sequence[int],
    mode: str,
    fit_model_fn=fit_rank1_dialect_modulation,
    apply_model_fn=apply_rank1_dialect_modulation,
    fit_kwargs: Mapping[str, Any] | None = None,
    apply_kwargs: Mapping[str, Any] | None = None,
) -> list[float]:
    improvements: list[float] = []
    fit_kwargs = dict(fit_kwargs or {})
    apply_kwargs = dict(apply_kwargs or {})
    for seed in seeds:
        calibration_centroids = speaker_centroids(calibration_ecapa, calibration_metadata)["speaker_centroids"]
        evaluation_centroids = speaker_centroids(evaluation_ecapa, evaluation_metadata)["speaker_centroids"]
        if mode == "wrong_centroid":
            calibration_override = _wrong_centroids(calibration_centroids, seed)
            evaluation_override = _wrong_centroids(evaluation_centroids, seed + 1000)
        elif mode == "shuffled_ecapa":
            calibration_override = _shuffled_centroids(calibration_centroids, seed)
            evaluation_override = _shuffled_centroids(evaluation_centroids, seed + 1000)
        else:
            calibration_override = calibration_centroids
            evaluation_override = evaluation_centroids
        fitted = fit_model_fn(
            calibration_ecapa,
            calibration_frozen,
            calibration_metadata,
            alpha_grid=alpha_grid,
            inner_cv_folds=inner_cv_folds,
            als_max_iter=als_max_iter,
            als_n_restarts=als_n_restarts,
            als_convergence_tol=als_convergence_tol,
            als_convergence_patience=als_convergence_patience,
            ecapa_centroid_override=calibration_override,
            **fit_kwargs,
        )
        bias_override = _swapped_dialect_bias(fitted["dialect_bias"]) if mode == "dialect_bias_swap" else None
        rank1_scale_override = _swapped_rank1_scale(fitted["rank1_scale"]) if mode == "v_d_swap" else None
        interaction_disabled = mode == "interaction_disabled"
        cleaned_calibration = apply_model_fn(
            calibration_frozen,
            calibration_ecapa,
            calibration_metadata,
            fitted,
            ecapa_centroid_override=calibration_override,
            dialect_bias_override=bias_override,
            rank1_scale_override=rank1_scale_override,
            interaction_disabled=interaction_disabled,
            **apply_kwargs,
        )
        cleaned_evaluation = apply_model_fn(
            evaluation_frozen,
            evaluation_ecapa,
            evaluation_metadata,
            fitted,
            ecapa_centroid_override=evaluation_override,
            dialect_bias_override=bias_override,
            rank1_scale_override=rank1_scale_override,
            interaction_disabled=interaction_disabled,
            **apply_kwargs,
        )
        ablation_calibration_rows = pair_distances(calibration_pairs, cleaned_calibration, reference)
        ablation_evaluation_rows = pair_distances(evaluation_pairs, cleaned_evaluation, reference)
        ablation_scale = _fit_affine(ablation_calibration_rows)
        improvements.append(_improvement_ratio(baseline_mae, _mae(ablation_evaluation_rows, *ablation_scale)))
    return improvements


def _rank1_reference_report(
    *,
    model_name: str,
    calibration_frozen: Mapping[str, Sequence[float]],
    evaluation_frozen: Mapping[str, Sequence[float]],
    calibration_ecapa: Mapping[str, Sequence[float]],
    evaluation_ecapa: Mapping[str, Sequence[float]],
    calibration_metadata: Mapping[str, Mapping[str, Any]],
    evaluation_metadata: Mapping[str, Mapping[str, Any]],
    calibration_pairs: Sequence[Mapping[str, Any]],
    evaluation_pairs: Sequence[Mapping[str, Any]],
    reference: Mapping[str, Any],
    alpha_grid: Sequence[float],
    inner_cv_folds: int,
    bootstrap_replicates: int,
    ablation_seeds: Sequence[int],
    seed: int,
    improvement_threshold: float,
    matched_tolerance: float,
    als_max_iter: int,
    als_n_restarts: int,
    als_convergence_tol: float,
    als_convergence_patience: int,
    global_regression_reference: Mapping[str, Any] | None = None,
    fit_model_fn=fit_rank1_dialect_modulation,
    apply_model_fn=apply_rank1_dialect_modulation,
    fit_kwargs: Mapping[str, Any] | None = None,
    apply_kwargs: Mapping[str, Any] | None = None,
    report_schema: str = "low-rank-dialect-perturbation-r1-gate-v1",
    method_model: str = "rank1_low_rank_modulation",
    parameterization_label: str = "shared_ridge_plus_rank1_dialect_modulation",
) -> dict[str, Any]:
    fit_kwargs = dict(fit_kwargs or {})
    apply_kwargs = dict(apply_kwargs or {})
    fitted = fit_model_fn(
        calibration_ecapa,
        calibration_frozen,
        calibration_metadata,
        alpha_grid=alpha_grid,
        inner_cv_folds=inner_cv_folds,
        als_max_iter=als_max_iter,
        als_n_restarts=als_n_restarts,
        als_convergence_tol=als_convergence_tol,
        als_convergence_patience=als_convergence_patience,
        **fit_kwargs,
    )
    cleaned_calibration = apply_model_fn(
        calibration_frozen,
        calibration_ecapa,
        calibration_metadata,
        fitted,
        **apply_kwargs,
    )
    cleaned_evaluation = apply_model_fn(
        evaluation_frozen,
        evaluation_ecapa,
        evaluation_metadata,
        fitted,
        **apply_kwargs,
    )
    baseline_calibration_rows = pair_distances(calibration_pairs, calibration_frozen, reference)
    baseline_evaluation_rows = pair_distances(evaluation_pairs, evaluation_frozen, reference)
    corrected_calibration_rows = pair_distances(calibration_pairs, cleaned_calibration, reference)
    corrected_evaluation_rows = pair_distances(evaluation_pairs, cleaned_evaluation, reference)
    leave_pair_out_rows, leave_pair_out_fallback = _rank1_leave_pair_out_rows(
        evaluation_pairs,
        evaluation_frozen,
        evaluation_ecapa,
        evaluation_metadata,
        fitted,
        reference,
    )
    baseline_scale = _fit_affine(baseline_calibration_rows)
    corrected_scale = _fit_affine(corrected_calibration_rows)
    baseline_mae = _mae(baseline_evaluation_rows, *baseline_scale)
    corrected_mae = _mae(corrected_evaluation_rows, *corrected_scale)
    leave_pair_out_mae = _mae(leave_pair_out_rows, *corrected_scale)
    improvement = _improvement_ratio(baseline_mae, corrected_mae)
    estimates = _bootstrap_improvements(
        baseline_evaluation_rows,
        corrected_evaluation_rows,
        baseline_scale,
        corrected_scale,
        seed=seed,
        replicates=bootstrap_replicates,
    )
    matched_estimates = _bootstrap_improvements(
        baseline_evaluation_rows,
        corrected_evaluation_rows,
        baseline_scale,
        corrected_scale,
        seed=seed + 17,
        replicates=bootstrap_replicates,
        groups={"A", "C"},
    )
    matched_baseline = _mae([row for row in baseline_evaluation_rows if row.get("group") in {"A", "C"}], *baseline_scale)
    matched_corrected = _mae([row for row in corrected_evaluation_rows if row.get("group") in {"A", "C"}], *corrected_scale)
    matched_increase = -_improvement_ratio(matched_baseline, matched_corrected)
    shuffled_improvements = _rank1_ablation_improvements(
        calibration_frozen=calibration_frozen,
        evaluation_frozen=evaluation_frozen,
        calibration_ecapa=calibration_ecapa,
        evaluation_ecapa=evaluation_ecapa,
        calibration_metadata=calibration_metadata,
        evaluation_metadata=evaluation_metadata,
        calibration_pairs=calibration_pairs,
        evaluation_pairs=evaluation_pairs,
        reference=reference,
        baseline_mae=baseline_mae,
        alpha_grid=alpha_grid,
        inner_cv_folds=inner_cv_folds,
        als_max_iter=als_max_iter,
        als_n_restarts=als_n_restarts,
        als_convergence_tol=als_convergence_tol,
        als_convergence_patience=als_convergence_patience,
        seeds=ablation_seeds,
        mode="shuffled_ecapa",
    )
    wrong_improvements = _rank1_ablation_improvements(
        calibration_frozen=calibration_frozen,
        evaluation_frozen=evaluation_frozen,
        calibration_ecapa=calibration_ecapa,
        evaluation_ecapa=evaluation_ecapa,
        calibration_metadata=calibration_metadata,
        evaluation_metadata=evaluation_metadata,
        calibration_pairs=calibration_pairs,
        evaluation_pairs=evaluation_pairs,
        reference=reference,
        baseline_mae=baseline_mae,
        alpha_grid=alpha_grid,
        inner_cv_folds=inner_cv_folds,
        als_max_iter=als_max_iter,
        als_n_restarts=als_n_restarts,
        als_convergence_tol=als_convergence_tol,
        als_convergence_patience=als_convergence_patience,
        seeds=ablation_seeds,
        mode="wrong_centroid",
    )
    dialect_bias_swap_improvements = _rank1_ablation_improvements(
        calibration_frozen=calibration_frozen,
        evaluation_frozen=evaluation_frozen,
        calibration_ecapa=calibration_ecapa,
        evaluation_ecapa=evaluation_ecapa,
        calibration_metadata=calibration_metadata,
        evaluation_metadata=evaluation_metadata,
        calibration_pairs=calibration_pairs,
        evaluation_pairs=evaluation_pairs,
        reference=reference,
        baseline_mae=baseline_mae,
        alpha_grid=alpha_grid,
        inner_cv_folds=inner_cv_folds,
        als_max_iter=als_max_iter,
        als_n_restarts=als_n_restarts,
        als_convergence_tol=als_convergence_tol,
        als_convergence_patience=als_convergence_patience,
        seeds=ablation_seeds,
        mode="dialect_bias_swap",
    )
    interaction_disabled_improvements = _rank1_ablation_improvements(
        calibration_frozen=calibration_frozen,
        evaluation_frozen=evaluation_frozen,
        calibration_ecapa=calibration_ecapa,
        evaluation_ecapa=evaluation_ecapa,
        calibration_metadata=calibration_metadata,
        evaluation_metadata=evaluation_metadata,
        calibration_pairs=calibration_pairs,
        evaluation_pairs=evaluation_pairs,
        reference=reference,
        baseline_mae=baseline_mae,
        alpha_grid=alpha_grid,
        inner_cv_folds=inner_cv_folds,
        als_max_iter=als_max_iter,
        als_n_restarts=als_n_restarts,
        als_convergence_tol=als_convergence_tol,
        als_convergence_patience=als_convergence_patience,
        seeds=ablation_seeds,
        mode="interaction_disabled",
    )
    v_d_swap_improvements = _rank1_ablation_improvements(
        calibration_frozen=calibration_frozen,
        evaluation_frozen=evaluation_frozen,
        calibration_ecapa=calibration_ecapa,
        evaluation_ecapa=evaluation_ecapa,
        calibration_metadata=calibration_metadata,
        evaluation_metadata=evaluation_metadata,
        calibration_pairs=calibration_pairs,
        evaluation_pairs=evaluation_pairs,
        reference=reference,
        baseline_mae=baseline_mae,
        alpha_grid=alpha_grid,
        inner_cv_folds=inner_cv_folds,
        als_max_iter=als_max_iter,
        als_n_restarts=als_n_restarts,
        als_convergence_tol=als_convergence_tol,
        als_convergence_patience=als_convergence_patience,
        seeds=ablation_seeds,
        mode="v_d_swap",
    )
    ci_lower = _quantile(estimates, 0.025)
    matched_upper = -_quantile(matched_estimates, 0.025)
    global_improvement = (
        float(global_regression_reference["improvement_ratio"])
        if global_regression_reference and "improvement_ratio" in global_regression_reference
        else None
    )
    outperforms_global = global_improvement is None or improvement > global_improvement
    passed = (
        improvement >= improvement_threshold
        and ci_lower > 0.0
        and matched_increase <= matched_tolerance
        and matched_upper <= matched_tolerance
        and max(shuffled_improvements) < improvement_threshold
        and max(wrong_improvements) < improvement_threshold
        and max(dialect_bias_swap_improvements) < improvement_threshold
        and max(v_d_swap_improvements) < max(improvement * 0.5, 0.0)
        and outperforms_global
    )
    return {
        "model_name": model_name,
        "reference_name": reference.get("name", "reference"),
        "calibration_pair_count": len(corrected_calibration_rows),
        "evaluation_pair_count": len(corrected_evaluation_rows),
        "fit": {
            "model": method_model,
            "selected_alpha": fitted["selected_alpha"],
            "speaker_count": fitted["speaker_count"],
            "cell_count": fitted["cell_count"],
            "parameterization": parameterization_label,
            "target_source": fitted["target_source"],
            "target_r2": fitted["target_r2"],
            "target_mse": fitted["target_mse"],
            "shared_weight_shape": list(fitted["shared_weight_shape"]),
            "dialect_count": len(fitted["dialects"]),
            "rank": fitted["rank"],
            "als_max_iter": fitted["als_max_iter"],
            "als_n_restarts": fitted["als_n_restarts"],
            "als_convergence_tol": fitted["als_convergence_tol"],
            "als_convergence_patience": fitted["als_convergence_patience"],
            "rank1_r2": fitted["rank1_r2"],
            "regularization_family": fitted.get("regularization_family"),
            "feature_scales": {
                key: float(value) for key, value in fitted.get("feature_scales", {}).items()
            } if fitted.get("feature_scales") is not None else None,
            "w_penalty_multiplier": float(fitted.get("w_penalty_multiplier", 1.0)),
            "bias_penalty_multiplier": float(fitted.get("bias_penalty_multiplier", 1.0)),
            "low_rank_penalty_multiplier": float(fitted.get("low_rank_penalty_multiplier", 1.0)),
        },
        "baseline_scale": {"intercept": baseline_scale[0], "slope": baseline_scale[1]},
        "corrected_scale": {"intercept": corrected_scale[0], "slope": corrected_scale[1]},
        "baseline_mae": baseline_mae,
        "corrected_mae": corrected_mae,
        "improvement_ratio": improvement,
        "ci": {"lower": ci_lower, "upper": _quantile(estimates, 0.975), "confidence_level": 0.95},
        "matched_speaker_mae_increase_ratio": matched_increase,
        "matched_speaker_ci": {"increase_upper": matched_upper, "confidence_level": 0.95},
        "matched_speaker_groups": {
            "A": _group_report("A", baseline_evaluation_rows, corrected_evaluation_rows, baseline_scale, corrected_scale),
            "C": _group_report("C", baseline_evaluation_rows, corrected_evaluation_rows, baseline_scale, corrected_scale),
        },
        "leave_pair_out": {
            "mae": leave_pair_out_mae,
            "improvement_ratio": _improvement_ratio(baseline_mae, leave_pair_out_mae),
            **leave_pair_out_fallback,
        },
        "ablation": {
            "shuffled_ecapa": {
                "improvement_ratios": shuffled_improvements,
                "max_improvement_ratio": max(shuffled_improvements),
            },
            "wrong_centroid": {
                "improvement_ratios": wrong_improvements,
                "max_improvement_ratio": max(wrong_improvements),
            },
            "dialect_bias_swap": {
                "improvement_ratios": dialect_bias_swap_improvements,
                "max_improvement_ratio": max(dialect_bias_swap_improvements),
            },
            "interaction_disabled": {
                "improvement_ratios": interaction_disabled_improvements,
                "max_improvement_ratio": max(interaction_disabled_improvements),
            },
            "v_d_swap": {
                "improvement_ratios": v_d_swap_improvements,
                "max_improvement_ratio": max(v_d_swap_improvements),
            },
        },
        "global_regression_comparison": {
            "global_improvement_ratio": global_improvement,
            "conditional_improvement_ratio": improvement,
            "outperforms_global": outperforms_global,
        },
        "status": "passed" if passed else "failed",
    }


def evaluate_rank1_dialect_perturbation_gate(
    calibration_frozen_embeddings: Mapping[str, Mapping[str, Sequence[float]]],
    calibration_ecapa_embeddings: Mapping[str, Sequence[float]],
    evaluation_frozen_embeddings: Mapping[str, Mapping[str, Sequence[float]]],
    evaluation_ecapa_embeddings: Mapping[str, Sequence[float]],
    calibration_metadata: Mapping[str, Mapping[str, Any]],
    evaluation_metadata: Mapping[str, Mapping[str, Any]],
    calibration_pairs: Sequence[Mapping[str, Any]],
    evaluation_pairs: Sequence[Mapping[str, Any]],
    *,
    references: Sequence[Mapping[str, Any]],
    alpha_grid: Sequence[float] = DEFAULT_ALPHA_GRID,
    inner_cv_folds: int = 5,
    bootstrap_replicates: int = 1000,
    ablation_seeds: Sequence[int] = DEFAULT_ABLATION_SEEDS,
    seed: int = 20260829,
    improvement_threshold: float = 0.05,
    matched_tolerance: float = 0.01,
    als_max_iter: int = DEFAULT_RANK1_ALS_MAX_ITER,
    als_n_restarts: int = DEFAULT_RANK1_ALS_N_RESTARTS,
    als_convergence_tol: float = DEFAULT_RANK1_ALS_CONVERGENCE_TOL,
    als_convergence_patience: int = DEFAULT_RANK1_ALS_CONVERGENCE_PATIENCE,
    global_regression_reference: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    if bootstrap_replicates < 1000:
        raise ValueError("at least 1000 bootstrap replicates are required")
    reports = []
    for model_name, calibration_frozen in sorted(calibration_frozen_embeddings.items()):
        if model_name not in evaluation_frozen_embeddings:
            continue
        reference_reports = [
            _rank1_reference_report(
                model_name=model_name,
                calibration_frozen=calibration_frozen,
                evaluation_frozen=evaluation_frozen_embeddings[model_name],
                calibration_ecapa=calibration_ecapa_embeddings,
                evaluation_ecapa=evaluation_ecapa_embeddings,
                calibration_metadata=calibration_metadata,
                evaluation_metadata=evaluation_metadata,
                calibration_pairs=calibration_pairs,
                evaluation_pairs=evaluation_pairs,
                reference=reference,
                alpha_grid=alpha_grid,
                inner_cv_folds=inner_cv_folds,
                bootstrap_replicates=bootstrap_replicates,
                ablation_seeds=ablation_seeds,
                seed=seed,
                improvement_threshold=improvement_threshold,
                matched_tolerance=matched_tolerance,
                als_max_iter=als_max_iter,
                als_n_restarts=als_n_restarts,
                als_convergence_tol=als_convergence_tol,
                als_convergence_patience=als_convergence_patience,
                global_regression_reference=_find_reference_report(
                    global_regression_reference,
                    model_name,
                    str(reference.get("name", "reference")),
                ),
            )
            for reference in references
        ]
        reports.append(
            {
                "model_name": model_name,
                "status": "passed" if reference_reports and all(report["status"] == "passed" for report in reference_reports) else "failed",
                "references": reference_reports,
            }
        )
    status = "passed" if reports and all(report["status"] == "passed" for report in reports) else "failed"
    return {
        "schema": "low-rank-dialect-perturbation-r1-gate-v1",
        "seed": seed,
        "alpha_grid": list(alpha_grid),
        "inner_cv_folds": inner_cv_folds,
        "evaluation_ecapa_scope": "full",
        "thresholds": {
            "correction_improvement": improvement_threshold,
            "matched_speaker_tolerance": matched_tolerance,
        },
        "method": {
            "feature_source": "ecapa",
            "target_source": "cell_offset_minus_dialect_main_effect",
            "parameterization": "shared_ridge_plus_rank1_dialect_modulation",
            "regression": "ridge+als",
            "rank": 1,
            "formula": "e_clean = e - q_rank1(ecapa_speaker_centroid, dialect_label) + mean_calibration_q",
            "affine_scaling": "fit_on_calibration_and_freeze_to_evaluation",
        },
        "status": status,
        "decision": "continue_to_review" if status == "passed" else "stop_before_manuscript_and_release",
        "models": reports,
    }


def evaluate_block_regularized_low_rank_gate(
    calibration_frozen_embeddings: Mapping[str, Mapping[str, Sequence[float]]],
    calibration_ecapa_embeddings: Mapping[str, Sequence[float]],
    evaluation_frozen_embeddings: Mapping[str, Mapping[str, Sequence[float]]],
    evaluation_ecapa_embeddings: Mapping[str, Sequence[float]],
    calibration_metadata: Mapping[str, Mapping[str, Any]],
    evaluation_metadata: Mapping[str, Mapping[str, Any]],
    calibration_pairs: Sequence[Mapping[str, Any]],
    evaluation_pairs: Sequence[Mapping[str, Any]],
    *,
    references: Sequence[Mapping[str, Any]],
    base_alpha_grid: Sequence[float] = (1.0, 3.0, 10.0, 30.0),
    uniform_control_alpha_grid: Sequence[float] = (1.0, 10.0, 30.0, 100.0, 300.0, 1000.0),
    inner_cv_folds: int = 5,
    bootstrap_replicates: int = 1000,
    ablation_seeds: Sequence[int] = DEFAULT_ABLATION_SEEDS,
    seed: int = 20260829,
    improvement_threshold: float = 0.05,
    matched_tolerance: float = 0.01,
    als_max_iter: int = DEFAULT_RANK1_ALS_MAX_ITER,
    als_n_restarts: int = DEFAULT_RANK1_ALS_N_RESTARTS,
    als_convergence_tol: float = DEFAULT_RANK1_ALS_CONVERGENCE_TOL,
    als_convergence_patience: int = DEFAULT_RANK1_ALS_CONVERGENCE_PATIENCE,
    w_penalty_multiplier: float = 100.0,
    bias_penalty_multiplier: float = 10.0,
    low_rank_penalty_multiplier: float = 1.0,
    previous_branch_reference: Mapping[str, Any] | None = None,
    global_regression_reference: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    if bootstrap_replicates < 1000:
        raise ValueError("at least 1000 bootstrap replicates are required")
    reports = []
    for model_name, calibration_frozen in sorted(calibration_frozen_embeddings.items()):
        if model_name not in evaluation_frozen_embeddings:
            continue
        reference_reports = []
        for reference in references:
            block_report = _rank1_reference_report(
                model_name=model_name,
                calibration_frozen=calibration_frozen,
                evaluation_frozen=evaluation_frozen_embeddings[model_name],
                calibration_ecapa=calibration_ecapa_embeddings,
                evaluation_ecapa=evaluation_ecapa_embeddings,
                calibration_metadata=calibration_metadata,
                evaluation_metadata=evaluation_metadata,
                calibration_pairs=calibration_pairs,
                evaluation_pairs=evaluation_pairs,
                reference=reference,
                alpha_grid=base_alpha_grid,
                inner_cv_folds=inner_cv_folds,
                bootstrap_replicates=bootstrap_replicates,
                ablation_seeds=ablation_seeds,
                seed=seed,
                improvement_threshold=improvement_threshold,
                matched_tolerance=matched_tolerance,
                als_max_iter=als_max_iter,
                als_n_restarts=als_n_restarts,
                als_convergence_tol=als_convergence_tol,
                als_convergence_patience=als_convergence_patience,
                global_regression_reference=_find_reference_report(
                    global_regression_reference,
                    model_name,
                    str(reference.get("name", "reference")),
                ),
                fit_model_fn=fit_block_regularized_rank1_dialect_modulation,
                apply_model_fn=apply_block_regularized_rank1_dialect_modulation,
                fit_kwargs={
                    "w_penalty_multiplier": w_penalty_multiplier,
                    "bias_penalty_multiplier": bias_penalty_multiplier,
                    "low_rank_penalty_multiplier": low_rank_penalty_multiplier,
                },
                report_schema="block-regularized-low-rank-dialect-perturbation-r1-gate-v1",
                method_model="block_regularized_rank1_dialect_modulation",
                parameterization_label="block_regularized_shared_ridge_plus_rank1_dialect_modulation",
            )
            uniform_report = _rank1_reference_report(
                model_name=model_name,
                calibration_frozen=calibration_frozen,
                evaluation_frozen=evaluation_frozen_embeddings[model_name],
                calibration_ecapa=calibration_ecapa_embeddings,
                evaluation_ecapa=evaluation_ecapa_embeddings,
                calibration_metadata=calibration_metadata,
                evaluation_metadata=evaluation_metadata,
                calibration_pairs=calibration_pairs,
                evaluation_pairs=evaluation_pairs,
                reference=reference,
                alpha_grid=uniform_control_alpha_grid,
                inner_cv_folds=inner_cv_folds,
                bootstrap_replicates=bootstrap_replicates,
                ablation_seeds=ablation_seeds,
                seed=seed,
                improvement_threshold=improvement_threshold,
                matched_tolerance=matched_tolerance,
                als_max_iter=als_max_iter,
                als_n_restarts=als_n_restarts,
                als_convergence_tol=als_convergence_tol,
                als_convergence_patience=als_convergence_patience,
                global_regression_reference=_find_reference_report(
                    global_regression_reference,
                    model_name,
                    str(reference.get("name", "reference")),
                ),
                fit_model_fn=fit_block_regularized_rank1_dialect_modulation,
                apply_model_fn=apply_block_regularized_rank1_dialect_modulation,
                fit_kwargs={
                    "w_penalty_multiplier": 1.0,
                    "bias_penalty_multiplier": 1.0,
                    "low_rank_penalty_multiplier": 1.0,
                },
                report_schema="block-regularized-low-rank-dialect-perturbation-r1-gate-v1",
                method_model="uniform_control_rank1_dialect_modulation",
                parameterization_label="uniform_shared_ridge_plus_rank1_dialect_modulation",
            )
            block_report["uniform_control_comparison"] = {
                "uniform_improvement_ratio": uniform_report["improvement_ratio"],
                "block_improvement_ratio": block_report["improvement_ratio"],
                "block_better_than_uniform": block_report["improvement_ratio"] > uniform_report["improvement_ratio"],
            }
            previous_reference_report = _find_reference_report(
                previous_branch_reference,
                model_name,
                str(reference.get("name", "reference")),
            )
            block_report["previous_branch_comparison"] = {
                "previous_improvement_ratio": previous_reference_report.get("improvement_ratio") if previous_reference_report else None,
                "current_improvement_ratio": block_report["improvement_ratio"],
                "outperforms_previous_branch": (
                    block_report["improvement_ratio"]
                    > (previous_reference_report.get("improvement_ratio") if previous_reference_report else float("-inf"))
                ),
            }
            reference_reports.append(block_report)
        reports.append(
            {
                "model_name": model_name,
                "status": "passed" if reference_reports and all(report["status"] == "passed" for report in reference_reports) else "failed",
                "references": reference_reports,
            }
        )
    status = "passed" if reports and all(report["status"] == "passed" for report in reports) else "failed"
    return {
        "schema": "block-regularized-low-rank-dialect-perturbation-r1-gate-v1",
        "seed": seed,
        "base_alpha_grid": list(base_alpha_grid),
        "uniform_control_alpha_grid": list(uniform_control_alpha_grid),
        "inner_cv_folds": inner_cv_folds,
        "evaluation_ecapa_scope": "full",
        "thresholds": {
            "correction_improvement": improvement_threshold,
            "matched_speaker_tolerance": matched_tolerance,
        },
        "method": {
            "feature_source": "ecapa",
            "target_source": "cell_offset_minus_dialect_main_effect",
            "parameterization": "block_regularized_shared_ridge_plus_rank1_dialect_modulation",
            "regularization_family": "block_diagonal_ridge",
            "regression": "ridge+als",
            "rank": 1,
            "formula": "e_clean = e - q_rank1(ecapa_speaker_centroid, dialect_label) + mean_calibration_q",
            "affine_scaling": "fit_on_calibration_and_freeze_to_evaluation",
        },
        "status": status,
        "decision": "continue_to_review" if status == "passed" else "stop_before_manuscript_and_release",
        "models": reports,
    }


def _reference_report(
    *,
    model_name: str,
    calibration_frozen: Mapping[str, Sequence[float]],
    evaluation_frozen: Mapping[str, Sequence[float]],
    calibration_ecapa: Mapping[str, Sequence[float]],
    evaluation_ecapa: Mapping[str, Sequence[float]],
    calibration_metadata: Mapping[str, Mapping[str, Any]],
    evaluation_metadata: Mapping[str, Mapping[str, Any]],
    calibration_pairs: Sequence[Mapping[str, Any]],
    evaluation_pairs: Sequence[Mapping[str, Any]],
    reference: Mapping[str, Any],
    alpha_grid: Sequence[float],
    inner_cv_folds: int,
    bootstrap_replicates: int,
    ablation_seeds: Sequence[int],
    seed: int,
    improvement_threshold: float,
    matched_tolerance: float,
    global_regression_reference: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    fitted = fit_speaker_regression(
        calibration_ecapa,
        calibration_frozen,
        calibration_metadata,
        alpha_grid=alpha_grid,
        inner_cv_folds=inner_cv_folds,
    )
    cleaned_calibration = apply_speaker_regression(calibration_frozen, calibration_ecapa, calibration_metadata, fitted)
    cleaned_evaluation = apply_speaker_regression(evaluation_frozen, evaluation_ecapa, evaluation_metadata, fitted)
    baseline_calibration_rows = pair_distances(calibration_pairs, calibration_frozen, reference)
    baseline_evaluation_rows = pair_distances(evaluation_pairs, evaluation_frozen, reference)
    corrected_calibration_rows = pair_distances(calibration_pairs, cleaned_calibration, reference)
    corrected_evaluation_rows = pair_distances(evaluation_pairs, cleaned_evaluation, reference)
    leave_pair_out_rows, leave_pair_out_fallback = _leave_pair_out_rows(
        evaluation_pairs,
        evaluation_frozen,
        evaluation_ecapa,
        evaluation_metadata,
        fitted,
        reference,
    )
    baseline_scale = _fit_affine(baseline_calibration_rows)
    corrected_scale = _fit_affine(corrected_calibration_rows)
    baseline_mae = _mae(baseline_evaluation_rows, *baseline_scale)
    corrected_mae = _mae(corrected_evaluation_rows, *corrected_scale)
    leave_pair_out_mae = _mae(leave_pair_out_rows, *corrected_scale)
    improvement = _improvement_ratio(baseline_mae, corrected_mae)
    estimates = _bootstrap_improvements(
        baseline_evaluation_rows,
        corrected_evaluation_rows,
        baseline_scale,
        corrected_scale,
        seed=seed,
        replicates=bootstrap_replicates,
    )
    matched_estimates = _bootstrap_improvements(
        baseline_evaluation_rows,
        corrected_evaluation_rows,
        baseline_scale,
        corrected_scale,
        seed=seed + 17,
        replicates=bootstrap_replicates,
        groups={"A", "C"},
    )
    matched_baseline = _mae([row for row in baseline_evaluation_rows if row.get("group") in {"A", "C"}], *baseline_scale)
    matched_corrected = _mae([row for row in corrected_evaluation_rows if row.get("group") in {"A", "C"}], *corrected_scale)
    matched_increase = -_improvement_ratio(matched_baseline, matched_corrected)
    shuffled_improvements = _ablation_improvements(
        calibration_frozen=calibration_frozen,
        evaluation_frozen=evaluation_frozen,
        calibration_ecapa=calibration_ecapa,
        evaluation_ecapa=evaluation_ecapa,
        calibration_metadata=calibration_metadata,
        evaluation_metadata=evaluation_metadata,
        calibration_pairs=calibration_pairs,
        evaluation_pairs=evaluation_pairs,
        reference=reference,
        baseline_mae=baseline_mae,
        alpha_grid=alpha_grid,
        inner_cv_folds=inner_cv_folds,
        seeds=ablation_seeds,
        mode="shuffled_ecapa",
    )
    wrong_improvements = _ablation_improvements(
        calibration_frozen=calibration_frozen,
        evaluation_frozen=evaluation_frozen,
        calibration_ecapa=calibration_ecapa,
        evaluation_ecapa=evaluation_ecapa,
        calibration_metadata=calibration_metadata,
        evaluation_metadata=evaluation_metadata,
        calibration_pairs=calibration_pairs,
        evaluation_pairs=evaluation_pairs,
        reference=reference,
        baseline_mae=baseline_mae,
        alpha_grid=alpha_grid,
        inner_cv_folds=inner_cv_folds,
        seeds=ablation_seeds,
        mode="wrong_centroid",
    )
    dialect_bias_swap_improvements = _ablation_improvements(
        calibration_frozen=calibration_frozen,
        evaluation_frozen=evaluation_frozen,
        calibration_ecapa=calibration_ecapa,
        evaluation_ecapa=evaluation_ecapa,
        calibration_metadata=calibration_metadata,
        evaluation_metadata=evaluation_metadata,
        calibration_pairs=calibration_pairs,
        evaluation_pairs=evaluation_pairs,
        reference=reference,
        baseline_mae=baseline_mae,
        alpha_grid=alpha_grid,
        inner_cv_folds=inner_cv_folds,
        seeds=ablation_seeds,
        mode="dialect_bias_swap",
    )
    ci_lower = _quantile(estimates, 0.025)
    matched_upper = -_quantile(matched_estimates, 0.025)
    global_improvement = (
        float(global_regression_reference["improvement_ratio"])
        if global_regression_reference and "improvement_ratio" in global_regression_reference
        else None
    )
    outperforms_global = global_improvement is None or improvement > global_improvement
    passed = (
        improvement >= improvement_threshold
        and ci_lower > 0.0
        and matched_increase <= matched_tolerance
        and matched_upper <= matched_tolerance
        and max(shuffled_improvements) < improvement_threshold
        and max(wrong_improvements) < improvement_threshold
        and max(dialect_bias_swap_improvements) < improvement_threshold
        and outperforms_global
    )
    return {
        "model_name": model_name,
        "reference_name": reference.get("name", "reference"),
        "calibration_pair_count": len(corrected_calibration_rows),
        "evaluation_pair_count": len(corrected_evaluation_rows),
        "fit": {
            "model": "ridge",
            "selected_alpha": fitted["selected_alpha"],
            "speaker_count": fitted["speaker_count"],
            "cell_count": fitted["cell_count"],
            "parameterization": fitted["parameterization"],
            "target_source": fitted["target_source"],
            "target_r2": fitted["target_r2"],
            "target_mse": fitted["target_mse"],
            "shared_weight_shape": list(fitted["shared_weight_shape"]),
            "dialect_count": len(fitted["dialects"]),
        },
        "baseline_scale": {"intercept": baseline_scale[0], "slope": baseline_scale[1]},
        "corrected_scale": {"intercept": corrected_scale[0], "slope": corrected_scale[1]},
        "baseline_mae": baseline_mae,
        "corrected_mae": corrected_mae,
        "improvement_ratio": improvement,
        "ci": {"lower": ci_lower, "upper": _quantile(estimates, 0.975), "confidence_level": 0.95},
        "matched_speaker_mae_increase_ratio": matched_increase,
        "matched_speaker_ci": {"increase_upper": matched_upper, "confidence_level": 0.95},
        "matched_speaker_groups": {
            "A": _group_report("A", baseline_evaluation_rows, corrected_evaluation_rows, baseline_scale, corrected_scale),
            "C": _group_report("C", baseline_evaluation_rows, corrected_evaluation_rows, baseline_scale, corrected_scale),
        },
        "leave_pair_out": {
            "mae": leave_pair_out_mae,
            "improvement_ratio": _improvement_ratio(baseline_mae, leave_pair_out_mae),
            **leave_pair_out_fallback,
        },
        "ablation": {
            "shuffled_ecapa": {
                "improvement_ratios": shuffled_improvements,
                "max_improvement_ratio": max(shuffled_improvements),
            },
            "wrong_centroid": {
                "improvement_ratios": wrong_improvements,
                "max_improvement_ratio": max(wrong_improvements),
            },
            "dialect_bias_swap": {
                "improvement_ratios": dialect_bias_swap_improvements,
                "max_improvement_ratio": max(dialect_bias_swap_improvements),
            },
        },
        "global_regression_comparison": {
            "global_improvement_ratio": global_improvement,
            "conditional_improvement_ratio": improvement,
            "outperforms_global": outperforms_global,
        },
        "status": "passed" if passed else "failed",
    }


def evaluate_ecapa_regression_gate(
    calibration_frozen_embeddings: Mapping[str, Mapping[str, Sequence[float]]],
    calibration_ecapa_embeddings: Mapping[str, Sequence[float]],
    evaluation_frozen_embeddings: Mapping[str, Mapping[str, Sequence[float]]],
    evaluation_ecapa_embeddings: Mapping[str, Sequence[float]],
    calibration_metadata: Mapping[str, Mapping[str, Any]],
    evaluation_metadata: Mapping[str, Mapping[str, Any]],
    calibration_pairs: Sequence[Mapping[str, Any]],
    evaluation_pairs: Sequence[Mapping[str, Any]],
    *,
    references: Sequence[Mapping[str, Any]],
    alpha_grid: Sequence[float] = DEFAULT_ALPHA_GRID,
    inner_cv_folds: int = 5,
    bootstrap_replicates: int = 1000,
    ablation_seeds: Sequence[int] = DEFAULT_ABLATION_SEEDS,
    seed: int = 20260829,
    improvement_threshold: float = 0.05,
    matched_tolerance: float = 0.01,
    global_regression_reference: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    if bootstrap_replicates < 1000:
        raise ValueError("at least 1000 bootstrap replicates are required")
    reports = []
    for model_name, calibration_frozen in sorted(calibration_frozen_embeddings.items()):
        if model_name not in evaluation_frozen_embeddings:
            continue
        reference_reports = [
            _reference_report(
                model_name=model_name,
                calibration_frozen=calibration_frozen,
                evaluation_frozen=evaluation_frozen_embeddings[model_name],
                calibration_ecapa=calibration_ecapa_embeddings,
                evaluation_ecapa=evaluation_ecapa_embeddings,
                calibration_metadata=calibration_metadata,
                evaluation_metadata=evaluation_metadata,
                calibration_pairs=calibration_pairs,
                evaluation_pairs=evaluation_pairs,
                reference=reference,
                alpha_grid=alpha_grid,
                inner_cv_folds=inner_cv_folds,
                bootstrap_replicates=bootstrap_replicates,
                ablation_seeds=ablation_seeds,
                seed=seed,
                improvement_threshold=improvement_threshold,
                matched_tolerance=matched_tolerance,
                global_regression_reference=_find_reference_report(
                    global_regression_reference,
                    model_name,
                    str(reference.get("name", "reference")),
                ),
            )
            for reference in references
        ]
        reports.append(
            {
                "model_name": model_name,
                "status": "passed" if reference_reports and all(report["status"] == "passed" for report in reference_reports) else "failed",
                "references": reference_reports,
            }
        )
    status = "passed" if reports and all(report["status"] == "passed" for report in reports) else "failed"
    return {
        "schema": "ecapa-regression-gate-v1",
        "seed": seed,
        "alpha_grid": list(alpha_grid),
        "inner_cv_folds": inner_cv_folds,
        "evaluation_ecapa_scope": "full",
        "thresholds": {
            "correction_improvement": improvement_threshold,
            "matched_speaker_tolerance": matched_tolerance,
        },
        "method": {
            "feature_source": "ecapa",
            "target_source": "cell_offset_minus_dialect_main_effect",
            "parameterization": "shared_ridge_per_dialect_bias",
            "regression": "ridge",
            "formula": "e_clean = e - q(ecapa_speaker_centroid, dialect_label) + mean_calibration_q",
            "affine_scaling": "fit_on_calibration_and_freeze_to_evaluation",
        },
        "status": status,
        "decision": "continue_to_review" if status == "passed" else "stop_before_manuscript_and_release",
        "models": reports,
    }


def _load_embedding_file(path: str | Path) -> tuple[str, dict[str, list[float]]]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    return str(payload["model_name"]), {key: [float(value) for value in values] for key, values in payload["embeddings"].items()}


def _load_model_embeddings(paths: Iterable[str | Path]) -> dict[str, dict[str, list[float]]]:
    loaded: dict[str, dict[str, list[float]]] = {}
    for path in paths:
        model_name, embeddings = _load_embedding_file(path)
        loaded[model_name] = embeddings
    return loaded


def _load_ecapa(path: str | Path) -> dict[str, list[float]]:
    _, embeddings = _load_embedding_file(path)
    return embeddings


def _load_pairs(path: str | Path) -> list[dict[str, Any]]:
    return json.loads(Path(path).read_text(encoding="utf-8"))["pairs"]


def _load_references(paths: Iterable[str | Path]) -> list[dict[str, Any]]:
    return [json.loads(Path(path).read_text(encoding="utf-8")) for path in paths]


def _find_reference_report(
    report: Mapping[str, Any] | None,
    model_name: str,
    reference_name: str,
) -> Mapping[str, Any] | None:
    if not report:
        return None
    if report.get("model_name") == model_name and report.get("reference_name") == reference_name:
        return report
    for model_report in report.get("models", []):
        if model_report.get("model_name") != model_name:
            continue
        for reference_report in model_report.get("references", []):
            if reference_report.get("reference_name") == reference_name:
                return reference_report
    return None


def _metadata_for_ids(record_manifest_path: str | Path, wanted: set[str]) -> dict[str, dict[str, Any]]:
    records = json.loads(Path(record_manifest_path).read_text(encoding="utf-8"))["records"]
    metadata: dict[str, dict[str, Any]] = {}
    for record in records:
        utterance_id = str(record["utterance_id"])
        if utterance_id in wanted:
            metadata[utterance_id] = {
                "speaker_id": str(record["speaker_id"]),
                "dialect_label": str(record["dialect_label"]),
                "recording_condition": str(record.get("recording_condition", "")),
            }
    missing = sorted(wanted.difference(metadata))
    if missing:
        raise ValueError(f"missing record metadata for utterance: {missing[0]}")
    return metadata


def _wanted_ids(*embedding_sets: Mapping[str, Sequence[float]], pairs: Sequence[Mapping[str, Any]] = ()) -> set[str]:
    wanted = {utterance_id for embeddings in embedding_sets for utterance_id in embeddings}
    wanted.update(str(utterance_id) for pair in pairs for utterance_id in pair["source_utterance_ids"])
    return wanted


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--calibration-embedding", nargs="+", action="append", required=True)
    parser.add_argument("--evaluation-embedding", nargs="+", action="append", required=True)
    parser.add_argument("--calibration-ecapa", required=True)
    parser.add_argument("--evaluation-ecapa", required=True)
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
    config = load_config(args.config)
    regression = config["speaker_regression"]
    gates = config["gates"]
    statistics = config["statistics"]
    calibration_frozen = _load_model_embeddings(args.calibration_embedding)
    evaluation_frozen = _load_model_embeddings(args.evaluation_embedding)
    calibration_ecapa = _load_ecapa(args.calibration_ecapa)
    evaluation_ecapa = _load_ecapa(args.evaluation_ecapa)
    calibration_pairs = _load_pairs(args.calibration_pairs)
    evaluation_pairs = _load_pairs(args.evaluation_pairs)
    calibration_metadata = _metadata_for_ids(
        args.records,
        _wanted_ids(*calibration_frozen.values(), calibration_ecapa, pairs=calibration_pairs),
    )
    evaluation_metadata = _metadata_for_ids(
        args.records,
        _wanted_ids(*evaluation_frozen.values(), evaluation_ecapa, pairs=evaluation_pairs),
    )
    output = Path(args.output)
    global_regression_reference: Mapping[str, Any] | None = None
    baseline_report_path = Path("results/gates/ecapa_regression_gate.json")
    if baseline_report_path.exists():
        try:
            loaded = json.loads(baseline_report_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            loaded = None
        if isinstance(loaded, Mapping):
            global_regression_reference = loaded
    previous_branch_reference: Mapping[str, Any] | None = None
    previous_branch_path = Path("results/gates/low_rank_dialect_perturbation_r1_gate.json")
    if previous_branch_path.exists():
        try:
            loaded = json.loads(previous_branch_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            loaded = None
        if isinstance(loaded, Mapping):
            previous_branch_reference = loaded
    if regression["model"] == "block_regularized_rank1_dialect_modulation":
        evaluator = evaluate_block_regularized_low_rank_gate
    elif regression["model"] == "rank1_low_rank_modulation":
        evaluator = evaluate_rank1_dialect_perturbation_gate
    else:
        evaluator = evaluate_ecapa_regression_gate
    common_kwargs = dict(
        calibration_frozen_embeddings=calibration_frozen,
        calibration_ecapa_embeddings=calibration_ecapa,
        evaluation_frozen_embeddings=evaluation_frozen,
        evaluation_ecapa_embeddings=evaluation_ecapa,
        calibration_metadata=calibration_metadata,
        evaluation_metadata=evaluation_metadata,
        calibration_pairs=calibration_pairs,
        evaluation_pairs=evaluation_pairs,
        references=_load_references(args.reference_matrix),
        inner_cv_folds=int(regression["inner_cv_folds"]),
        seed=args.seed,
        bootstrap_replicates=args.bootstrap_replicates or int(statistics.get("bootstrap_replicates", 1000)),
        improvement_threshold=float(gates["correction_improvement"]),
        matched_tolerance=float(gates["matched_speaker_tolerance"]),
    )
    if regression["model"] == "block_regularized_rank1_dialect_modulation":
        report = evaluator(
            **common_kwargs,
            base_alpha_grid=[float(alpha) for alpha in regression["base_alpha_grid"]],
            uniform_control_alpha_grid=[float(alpha) for alpha in regression["uniform_control_alpha_grid"]],
            als_max_iter=int(regression["als_max_iter"]),
            als_n_restarts=int(regression["als_n_restarts"]),
            als_convergence_tol=float(regression["als_convergence_tol"]),
            als_convergence_patience=int(regression["als_convergence_patience"]),
            w_penalty_multiplier=float(regression["w_penalty_multiplier"]),
            bias_penalty_multiplier=float(regression["bias_penalty_multiplier"]),
            low_rank_penalty_multiplier=float(regression["low_rank_penalty_multiplier"]),
            previous_branch_reference=previous_branch_reference,
            global_regression_reference=global_regression_reference,
        )
    else:
        report = evaluator(
            **common_kwargs,
            alpha_grid=[float(alpha) for alpha in regression.get("base_alpha_grid", regression.get("alpha_grid", []))],
            als_max_iter=int(regression["als_max_iter"]),
            als_n_restarts=int(regression["als_n_restarts"]),
            als_convergence_tol=float(regression["als_convergence_tol"]),
            als_convergence_patience=int(regression["als_convergence_patience"]),
            global_regression_reference=global_regression_reference,
        )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return 0 if report["status"] == "passed" else 2


if __name__ == "__main__":
    raise SystemExit(main())
