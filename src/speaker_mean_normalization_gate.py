"""Speaker-mean normalization correction gate."""

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


DEFAULT_ABLATION_SEEDS = [20260829, 20260830, 20260831, 20260901, 20260902]


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


def _speaker_ids(ids: Sequence[str], metadata: Mapping[str, Mapping[str, Any]]) -> list[str]:
    return [str(metadata[utterance_id]["speaker_id"]) for utterance_id in ids]


def fit_split_speaker_mean_model(
    embeddings: Mapping[str, Sequence[float]],
    metadata: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    """Fit split-local speaker means and global mean without dialect/reference labels."""
    ids = _ordered_ids(embeddings, metadata)
    x = _matrix(embeddings, ids)
    speakers = _speaker_ids(ids, metadata)
    global_mean = x.mean(axis=0)
    by_speaker: dict[str, list[int]] = defaultdict(list)
    for index, speaker in enumerate(speakers):
        by_speaker[speaker].append(index)
    speaker_means = {
        speaker: x[indices].mean(axis=0)
        for speaker, indices in sorted(by_speaker.items())
    }
    return {
        "utterance_ids": ids,
        "global_mean": global_mean,
        "speaker_means": speaker_means,
        "speaker_ids": dict(zip(ids, speakers)),
        "speaker_counts": {speaker: len(indices) for speaker, indices in sorted(by_speaker.items())},
    }


def _shuffled_labels(labels: Sequence[str], seed: int) -> list[str]:
    shuffled = list(labels)
    random.Random(seed).shuffle(shuffled)
    return shuffled


def _wrong_mean_map(speaker_means: Mapping[str, np.ndarray], seed: int) -> dict[str, np.ndarray]:
    speakers = sorted(speaker_means)
    if len(speakers) <= 1:
        return {speaker: np.asarray(mean_vector, dtype=np.float64) for speaker, mean_vector in speaker_means.items()}
    shuffled = speakers[:]
    rng = random.Random(seed)
    for _ in range(32):
        rng.shuffle(shuffled)
        if all(left != right for left, right in zip(speakers, shuffled)):
            break
    if any(left == right for left, right in zip(speakers, shuffled)):
        shuffled = speakers[1:] + speakers[:1]
    return {
        speaker: np.asarray(speaker_means[wrong_speaker], dtype=np.float64)
        for speaker, wrong_speaker in zip(speakers, shuffled)
    }


def mean_normalize_embeddings(
    embeddings: Mapping[str, Sequence[float]],
    metadata: Mapping[str, Mapping[str, Any]],
    *,
    shuffle_speaker_ids: bool = False,
    wrong_mean: bool = False,
    seed: int = 20260829,
) -> tuple[dict[str, list[float]], dict[str, Any]]:
    if shuffle_speaker_ids and wrong_mean:
        raise ValueError("shuffle_speaker_ids and wrong_mean are mutually exclusive")
    ids = _ordered_ids(embeddings, metadata)
    x = _matrix(embeddings, ids)
    model = fit_split_speaker_mean_model(embeddings, metadata)
    true_speakers = _speaker_ids(ids, metadata)
    global_mean = np.asarray(model["global_mean"], dtype=np.float64)
    if shuffle_speaker_ids:
        labels = _shuffled_labels(true_speakers, seed)
        by_label: dict[str, list[int]] = defaultdict(list)
        for index, label in enumerate(labels):
            by_label[label].append(index)
        means = {label: x[indices].mean(axis=0) for label, indices in by_label.items()}
        cleaned = np.asarray([x[index] - means[label] + global_mean for index, label in enumerate(labels)])
        mode = "shuffled_speaker"
        fallback_speaker_count = 0
    elif wrong_mean:
        means = _wrong_mean_map(model["speaker_means"], seed)
        cleaned = np.asarray([x[index] - means[speaker] + global_mean for index, speaker in enumerate(true_speakers)])
        mode = "wrong_mean"
        fallback_speaker_count = 1 if len(means) <= 1 else 0
    else:
        means = model["speaker_means"]
        cleaned = np.asarray([x[index] - means[speaker] + global_mean for index, speaker in enumerate(true_speakers)])
        mode = "speaker_mean"
        fallback_speaker_count = 0
    return (
        {utterance_id: cleaned[index].astype(float).tolist() for index, utterance_id in enumerate(ids)},
        {
            "mode": mode,
            "utterance_count": len(ids),
            "speaker_count": len(set(true_speakers)),
            "fallback_speaker_count": fallback_speaker_count,
        },
    )


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


def pairwise_mean_normalized_distance(
    pair: Mapping[str, Any],
    embeddings: Mapping[str, Sequence[float]],
    metadata: Mapping[str, Mapping[str, Any]],
    *,
    leave_pair_out: bool,
    return_report: bool = False,
) -> float | tuple[float, dict[str, int]]:
    if not leave_pair_out:
        cleaned, _ = mean_normalize_embeddings(embeddings, metadata)
        left_id, right_id = pair["source_utterance_ids"]
        distance = _cosine_distance(cleaned[left_id], cleaned[right_id])
        return (distance, {"fallback_endpoint_count": 0, "fallback_pair_count": 0}) if return_report else distance
    ids = _ordered_ids(embeddings, metadata)
    x = _matrix(embeddings, ids)
    id_to_index = {utterance_id: index for index, utterance_id in enumerate(ids)}
    global_mean = x.mean(axis=0)
    pair_ids = [str(value) for value in pair["source_utterance_ids"]]
    excluded = set(pair_ids)
    fallback_endpoint_count = 0
    cleaned_vectors = []
    for utterance_id in pair_ids:
        speaker = str(metadata[utterance_id]["speaker_id"])
        support_ids = [
            candidate_id
            for candidate_id in ids
            if str(metadata[candidate_id]["speaker_id"]) == speaker and candidate_id not in excluded
        ]
        if support_ids:
            speaker_mean = x[[id_to_index[candidate_id] for candidate_id in support_ids]].mean(axis=0)
        else:
            support_ids = [
                candidate_id
                for candidate_id in ids
                if str(metadata[candidate_id]["speaker_id"]) == speaker
            ]
            speaker_mean = x[[id_to_index[candidate_id] for candidate_id in support_ids]].mean(axis=0)
            fallback_endpoint_count += 1
        cleaned_vectors.append(x[id_to_index[utterance_id]] - speaker_mean + global_mean)
    distance = _cosine_distance(cleaned_vectors[0], cleaned_vectors[1])
    report = {
        "fallback_endpoint_count": fallback_endpoint_count,
        "fallback_pair_count": 1 if fallback_endpoint_count else 0,
    }
    return (distance, report) if return_report else distance


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
                "pair_id": str(pair["pair_id"]),
                "group": pair.get("group"),
                "dialect_labels": labels,
                "distance": _cosine_distance(embeddings[utterance_a], embeddings[utterance_b]),
                "target_distance": target,
            }
        )
    return rows


def _leave_pair_out_rows(
    pairs: Sequence[Mapping[str, Any]],
    embeddings: Mapping[str, Sequence[float]],
    metadata: Mapping[str, Mapping[str, Any]],
    reference: Mapping[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    ids = _ordered_ids(embeddings, metadata)
    x = _matrix(embeddings, ids)
    id_to_index = {utterance_id: index for index, utterance_id in enumerate(ids)}
    global_mean = x.mean(axis=0)
    speaker_sums: dict[str, np.ndarray] = {}
    speaker_counts: dict[str, int] = defaultdict(int)
    for index, utterance_id in enumerate(ids):
        speaker = str(metadata[utterance_id]["speaker_id"])
        if speaker not in speaker_sums:
            speaker_sums[speaker] = np.zeros(x.shape[1], dtype=np.float64)
        speaker_sums[speaker] += x[index]
        speaker_counts[speaker] += 1
    rows: list[dict[str, Any]] = []
    fallback_endpoint_count = 0
    fallback_pair_count = 0
    endpoint_count = 0
    for pair in pairs:
        labels = pair["dialect_labels"]
        target = _target_distance(labels, reference)
        if target is None:
            continue
        pair_ids = [str(value) for value in pair["source_utterance_ids"]]
        excluded_counts: dict[str, int] = defaultdict(int)
        excluded_sums: dict[str, np.ndarray] = {}
        for utterance_id in set(pair_ids):
            speaker = str(metadata[utterance_id]["speaker_id"])
            if speaker not in excluded_sums:
                excluded_sums[speaker] = np.zeros(x.shape[1], dtype=np.float64)
            excluded_sums[speaker] += x[id_to_index[utterance_id]]
            excluded_counts[speaker] += 1
        cleaned_vectors = []
        pair_fallback_endpoint_count = 0
        for utterance_id in pair_ids:
            speaker = str(metadata[utterance_id]["speaker_id"])
            support_count = speaker_counts[speaker] - excluded_counts[speaker]
            if support_count > 0:
                speaker_mean = (speaker_sums[speaker] - excluded_sums[speaker]) / support_count
            else:
                speaker_mean = speaker_sums[speaker] / speaker_counts[speaker]
                pair_fallback_endpoint_count += 1
            cleaned_vectors.append(x[id_to_index[utterance_id]] - speaker_mean + global_mean)
        distance = _cosine_distance(cleaned_vectors[0], cleaned_vectors[1])
        endpoint_count += 2
        fallback_endpoint_count += pair_fallback_endpoint_count
        fallback_pair_count += 1 if pair_fallback_endpoint_count else 0
        rows.append(
            {
                "pair_id": str(pair["pair_id"]),
                "group": pair.get("group"),
                "dialect_labels": labels,
                "distance": distance,
                "target_distance": target,
            }
        )
    return rows, {
        "fallback_endpoint_count": fallback_endpoint_count,
        "fallback_pair_count": fallback_pair_count,
        "fallback_endpoint_ratio": fallback_endpoint_count / endpoint_count if endpoint_count else 0.0,
        "fallback_pair_ratio": fallback_pair_count / len(rows) if rows else 0.0,
    }


def _fit_affine(rows: Sequence[Mapping[str, Any]]) -> tuple[float, float]:
    xs = [float(row["distance"]) for row in rows]
    ys = [float(row["target_distance"]) for row in rows]
    if not xs:
        raise ValueError("cannot fit affine scale without rows")
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


def _ablation_improvements(
    *,
    calibration_embeddings: Mapping[str, Sequence[float]],
    evaluation_embeddings: Mapping[str, Sequence[float]],
    calibration_metadata: Mapping[str, Mapping[str, Any]],
    evaluation_metadata: Mapping[str, Mapping[str, Any]],
    calibration_pairs: Sequence[Mapping[str, Any]],
    evaluation_pairs: Sequence[Mapping[str, Any]],
    reference: Mapping[str, Any],
    baseline_mae: float,
    seeds: Sequence[int],
    mode: str,
) -> list[float]:
    improvements = []
    for seed in seeds:
        ablation_calibration, _ = mean_normalize_embeddings(
            calibration_embeddings,
            calibration_metadata,
            shuffle_speaker_ids=mode == "shuffled_speaker",
            wrong_mean=mode == "wrong_mean",
            seed=seed,
        )
        ablation_evaluation, _ = mean_normalize_embeddings(
            evaluation_embeddings,
            evaluation_metadata,
            shuffle_speaker_ids=mode == "shuffled_speaker",
            wrong_mean=mode == "wrong_mean",
            seed=seed + 1000,
        )
        ablation_calibration_rows = _pair_distances(calibration_pairs, ablation_calibration, reference)
        ablation_evaluation_rows = _pair_distances(evaluation_pairs, ablation_evaluation, reference)
        ablation_scale = _fit_affine(ablation_calibration_rows)
        improvements.append(_improvement_ratio(baseline_mae, _mae(ablation_evaluation_rows, *ablation_scale)))
    return improvements


def _reference_report(
    *,
    model_name: str,
    calibration_embeddings: Mapping[str, Sequence[float]],
    evaluation_embeddings: Mapping[str, Sequence[float]],
    calibration_metadata: Mapping[str, Mapping[str, Any]],
    evaluation_metadata: Mapping[str, Mapping[str, Any]],
    calibration_pairs: Sequence[Mapping[str, Any]],
    evaluation_pairs: Sequence[Mapping[str, Any]],
    reference: Mapping[str, Any],
    bootstrap_replicates: int,
    ablation_seeds: Sequence[int],
    seed: int,
    improvement_threshold: float,
    matched_tolerance: float,
) -> dict[str, Any]:
    corrected_calibration, calibration_summary = mean_normalize_embeddings(calibration_embeddings, calibration_metadata)
    corrected_evaluation, evaluation_summary = mean_normalize_embeddings(evaluation_embeddings, evaluation_metadata)
    baseline_calibration_rows = _pair_distances(calibration_pairs, calibration_embeddings, reference)
    baseline_evaluation_rows = _pair_distances(evaluation_pairs, evaluation_embeddings, reference)
    corrected_calibration_rows = _pair_distances(calibration_pairs, corrected_calibration, reference)
    corrected_evaluation_rows = _pair_distances(evaluation_pairs, corrected_evaluation, reference)
    leave_pair_out_rows, leave_pair_out_fallback = _leave_pair_out_rows(
        evaluation_pairs,
        evaluation_embeddings,
        evaluation_metadata,
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
        calibration_embeddings=calibration_embeddings,
        evaluation_embeddings=evaluation_embeddings,
        calibration_metadata=calibration_metadata,
        evaluation_metadata=evaluation_metadata,
        calibration_pairs=calibration_pairs,
        evaluation_pairs=evaluation_pairs,
        reference=reference,
        baseline_mae=baseline_mae,
        seeds=ablation_seeds,
        mode="shuffled_speaker",
    )
    wrong_improvements = _ablation_improvements(
        calibration_embeddings=calibration_embeddings,
        evaluation_embeddings=evaluation_embeddings,
        calibration_metadata=calibration_metadata,
        evaluation_metadata=evaluation_metadata,
        calibration_pairs=calibration_pairs,
        evaluation_pairs=evaluation_pairs,
        reference=reference,
        baseline_mae=baseline_mae,
        seeds=ablation_seeds,
        mode="wrong_mean",
    )
    matched_upper = -_quantile(matched_estimates, 0.025)
    passed = (
        improvement >= improvement_threshold
        and _quantile(estimates, 0.025) > 0.0
        and matched_increase <= matched_tolerance
        and matched_upper <= matched_tolerance
        and max(shuffled_improvements) < improvement_threshold
        and max(wrong_improvements) < improvement_threshold
    )
    return {
        "model_name": model_name,
        "reference_name": reference.get("name", "reference"),
        "calibration_pair_count": len(corrected_calibration_rows),
        "evaluation_pair_count": len(corrected_evaluation_rows),
        "normalization": {
            "formula": "e_clean = e - mu_speaker + mu_global",
            "calibration_summary": calibration_summary,
            "evaluation_summary": evaluation_summary,
            "uses_evaluation_labels_or_errors": False,
            "scalar_parameter": None,
        },
        "baseline_scale": {"intercept": baseline_scale[0], "slope": baseline_scale[1]},
        "corrected_scale": {"intercept": corrected_scale[0], "slope": corrected_scale[1]},
        "baseline_mae": baseline_mae,
        "corrected_mae": corrected_mae,
        "improvement_ratio": improvement,
        "ci": {"lower": _quantile(estimates, 0.025), "upper": _quantile(estimates, 0.975), "confidence_level": 0.95},
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
            "shuffled_speaker": {
                "improvement_ratios": shuffled_improvements,
                "max_improvement_ratio": max(shuffled_improvements),
            },
            "wrong_mean": {
                "improvement_ratios": wrong_improvements,
                "max_improvement_ratio": max(wrong_improvements),
            },
        },
        "status": "passed" if passed else "failed",
    }


def evaluate_speaker_mean_normalization_gate(
    calibration_embeddings_by_model: Mapping[str, Mapping[str, Sequence[float]]],
    evaluation_embeddings_by_model: Mapping[str, Mapping[str, Sequence[float]]],
    calibration_metadata: Mapping[str, Mapping[str, Any]],
    evaluation_metadata: Mapping[str, Mapping[str, Any]],
    calibration_pairs: Sequence[Mapping[str, Any]],
    evaluation_pairs: Sequence[Mapping[str, Any]],
    *,
    references: Sequence[Mapping[str, Any]],
    bootstrap_replicates: int = 1000,
    ablation_seeds: Sequence[int] = DEFAULT_ABLATION_SEEDS,
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
        reference_reports = [
            _reference_report(
                model_name=model_name,
                calibration_embeddings=calibration_embeddings,
                evaluation_embeddings=evaluation_embeddings_by_model[model_name],
                calibration_metadata=calibration_metadata,
                evaluation_metadata=evaluation_metadata,
                calibration_pairs=calibration_pairs,
                evaluation_pairs=evaluation_pairs,
                reference=reference,
                bootstrap_replicates=bootstrap_replicates,
                ablation_seeds=ablation_seeds,
                seed=seed,
                improvement_threshold=improvement_threshold,
                matched_tolerance=matched_tolerance,
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
        "schema": "speaker-mean-normalization-gate-v1",
        "seed": seed,
        "ablation_seeds": list(ablation_seeds),
        "thresholds": {
            "correction_improvement": improvement_threshold,
            "matched_speaker_tolerance": matched_tolerance,
        },
        "method": {
            "normalization": "per_split_label_free_speaker_mean",
            "formula": "e_clean = e - mu_speaker + mu_global",
            "affine_scaling": "fit_on_calibration_and_freeze_to_evaluation",
            "evaluation_preprocessing_note": (
                "Evaluation embeddings are used only to estimate label-free speaker means; "
                "dialect labels, reference distances, and evaluation errors are not used for normalization or selection."
            ),
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


def _metadata_for_embedding_sets(
    record_manifest_path: str | Path,
    embeddings_by_model: Mapping[str, Mapping[str, Sequence[float]]],
) -> dict[str, dict[str, Any]]:
    records = json.loads(Path(record_manifest_path).read_text(encoding="utf-8"))["records"]
    wanted = {
        utterance_id
        for embeddings in embeddings_by_model.values()
        for utterance_id in embeddings
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
    calibration_embeddings = _load_embeddings(args.calibration_embedding)
    evaluation_embeddings = _load_embeddings(args.evaluation_embedding)
    report = evaluate_speaker_mean_normalization_gate(
        calibration_embeddings,
        evaluation_embeddings,
        _metadata_for_embedding_sets(args.records, calibration_embeddings),
        _metadata_for_embedding_sets(args.records, evaluation_embeddings),
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
