"""Cross-dialect speaker-mean decomposition diagnostics."""

from __future__ import annotations

import argparse
import json
from itertools import combinations
from pathlib import Path
from statistics import mean, median
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from .speaker_mean_normalization_gate import (
    _fit_affine,
    _improvement_ratio,
    _load_embeddings,
    _load_pairs,
    _load_references,
    _mae,
    _metadata_for_embedding_sets,
    _pair_distances,
)


def _as_matrix(embeddings: Mapping[str, Sequence[float]], ids: Sequence[str]) -> np.ndarray:
    matrix = np.asarray([embeddings[utterance_id] for utterance_id in ids], dtype=np.float64)
    if matrix.ndim != 2 or matrix.shape[0] != len(ids) or not np.isfinite(matrix).all():
        raise ValueError("embeddings must be a finite 2D matrix")
    return matrix


def _cosine(left: np.ndarray, right: np.ndarray) -> float:
    denominator = float(np.linalg.norm(left) * np.linalg.norm(right))
    if denominator <= 0:
        return 0.0
    return float(np.dot(left, right) / denominator)


def _safe_ratio(numerator: float, denominator: float) -> float:
    if denominator <= 0:
        return 0.0
    return float(numerator / denominator)


def _summary(values: Sequence[float]) -> dict[str, float | int | None]:
    if not values:
        return {"count": 0, "mean": None, "median": None, "min": None, "max": None}
    values = [float(value) for value in values]
    return {
        "count": len(values),
        "mean": float(mean(values)),
        "median": float(median(values)),
        "min": float(min(values)),
        "max": float(max(values)),
    }


def _speaker_buckets(
    embeddings: Mapping[str, Sequence[float]],
    metadata: Mapping[str, Mapping[str, Any]],
) -> dict[str, dict[str, list[str]]]:
    buckets: dict[str, dict[str, list[str]]] = {}
    for utterance_id in sorted(embeddings):
        if utterance_id not in metadata:
            continue
        speaker = str(metadata[utterance_id]["speaker_id"])
        dialect = str(metadata[utterance_id]["dialect_label"])
        buckets.setdefault(speaker, {}).setdefault(dialect, []).append(utterance_id)
    return buckets


def _cross_dialect_normalize(
    embeddings: Mapping[str, Sequence[float]],
    metadata: Mapping[str, Mapping[str, Any]],
    eligible_speakers: set[str],
) -> dict[str, list[float]]:
    ids = sorted(utterance_id for utterance_id in embeddings if utterance_id in metadata)
    matrix = _as_matrix(embeddings, ids)
    global_mean = matrix.mean(axis=0)
    by_speaker: dict[str, list[int]] = {}
    for index, utterance_id in enumerate(ids):
        speaker = str(metadata[utterance_id]["speaker_id"])
        by_speaker.setdefault(speaker, []).append(index)
    speaker_means = {
        speaker: matrix[indices].mean(axis=0)
        for speaker, indices in by_speaker.items()
    }
    cleaned: dict[str, list[float]] = {}
    for index, utterance_id in enumerate(ids):
        speaker = str(metadata[utterance_id]["speaker_id"])
        vector = matrix[index]
        if speaker in eligible_speakers:
            vector = vector - speaker_means[speaker] + global_mean
        cleaned[utterance_id] = vector.astype(float).tolist()
    return cleaned


def _reference_name(reference: Mapping[str, Any]) -> str:
    return str(reference.get("name") or reference.get("source") or "reference")


def _group_mae_changes(
    *,
    embeddings: Mapping[str, Sequence[float]],
    metadata: Mapping[str, Mapping[str, Any]],
    eligible_speakers: set[str],
    pairs: Sequence[Mapping[str, Any]] | None,
    references: Sequence[Mapping[str, Any]] | None,
) -> dict[str, Any]:
    empty = {
        "group_a_mae_change": None,
        "group_c_mae_change": None,
        "group_ac_combined_mae_change": None,
        "reference_summaries": [],
    }
    if not pairs or not references:
        return empty

    eligible_pairs = [
        pair for pair in pairs
        if all(
            utterance_id in embeddings
            and utterance_id in metadata
            and str(metadata[utterance_id]["speaker_id"]) in eligible_speakers
            for utterance_id in pair["source_utterance_ids"]
        )
    ]
    if not eligible_pairs:
        return empty

    corrected_embeddings = _cross_dialect_normalize(embeddings, metadata, eligible_speakers)
    reference_summaries = []
    group_a_changes = []
    group_c_changes = []
    group_ac_changes = []
    group_a_not_worse_flags = []
    for reference in references:
        baseline_rows = _pair_distances(eligible_pairs, embeddings, reference)
        corrected_rows = _pair_distances(eligible_pairs, corrected_embeddings, reference)
        if not baseline_rows or not corrected_rows:
            continue
        baseline_scale = _fit_affine(baseline_rows)
        corrected_scale = _fit_affine(corrected_rows)
        group_changes: dict[str, float | None] = {}
        for group, target in [("A", group_a_changes), ("C", group_c_changes)]:
            baseline_group = [row for row in baseline_rows if row.get("group") == group]
            corrected_group = [row for row in corrected_rows if row.get("group") == group]
            if not baseline_group or not corrected_group:
                group_changes[group] = None
                continue
            baseline_mae = _mae(baseline_group, *baseline_scale)
            corrected_mae = _mae(corrected_group, *corrected_scale)
            change = _safe_ratio(corrected_mae - baseline_mae, baseline_mae)
            group_changes[group] = change
            target.append(change)
            if group == "A":
                group_a_not_worse_flags.append(change <= 0.01)
        baseline_ac = [row for row in baseline_rows if row.get("group") in {"A", "C"}]
        corrected_ac = [row for row in corrected_rows if row.get("group") in {"A", "C"}]
        combined_change = None
        if baseline_ac and corrected_ac:
            combined_change = _safe_ratio(
                _mae(corrected_ac, *corrected_scale) - _mae(baseline_ac, *baseline_scale),
                _mae(baseline_ac, *baseline_scale),
            )
            group_ac_changes.append(combined_change)
        reference_summaries.append(
            {
                "reference_name": _reference_name(reference),
                "eligible_pair_count": len(baseline_rows),
                "group_a_mae_change": group_changes.get("A"),
                "group_c_mae_change": group_changes.get("C"),
                "group_ac_combined_mae_change": combined_change,
            }
        )

    return {
        "group_a_mae_change": float(mean(group_a_changes)) if group_a_changes else None,
        "group_c_mae_change": float(mean(group_c_changes)) if group_c_changes else None,
        "group_ac_combined_mae_change": float(mean(group_ac_changes)) if group_ac_changes else None,
        "eligible_speaker_group_a_not_worse_fraction": (
            float(mean(group_a_not_worse_flags)) if group_a_not_worse_flags else None
        ),
        "reference_summaries": reference_summaries,
    }


def _model_decomposition(
    *,
    model_name: str,
    embeddings: Mapping[str, Sequence[float]],
    metadata: Mapping[str, Mapping[str, Any]],
    pairs: Sequence[Mapping[str, Any]] | None,
    references: Sequence[Mapping[str, Any]] | None,
) -> dict[str, Any]:
    buckets = _speaker_buckets(embeddings, metadata)
    eligible = {
        speaker for speaker, dialects in buckets.items()
        if len(dialects) >= 2
    }
    speaker_means: dict[str, Any] = {}
    residuals: list[dict[str, Any]] = []
    residual_cosines: list[dict[str, Any]] = []
    ratio_values: list[float] = []
    cosine_values: list[float] = []

    for speaker in sorted(eligible):
        dialects = buckets[speaker]
        speaker_ids = [utterance_id for ids in dialects.values() for utterance_id in ids]
        speaker_matrix = _as_matrix(embeddings, sorted(speaker_ids))
        speaker_mean = speaker_matrix.mean(axis=0)
        speaker_norm = float(np.linalg.norm(speaker_mean))
        dialect_vectors: dict[str, np.ndarray] = {}
        dialect_entries = []
        for dialect in sorted(dialects):
            dialect_matrix = _as_matrix(embeddings, sorted(dialects[dialect]))
            dialect_mean = dialect_matrix.mean(axis=0)
            delta = dialect_mean - speaker_mean
            delta_norm = float(np.linalg.norm(delta))
            ratio = _safe_ratio(delta_norm, speaker_norm)
            ratio_values.append(ratio)
            dialect_vectors[dialect] = delta
            entry = {
                "speaker_id": speaker,
                "dialect_label": dialect,
                "utterance_count": len(dialects[dialect]),
                "delta_norm": delta_norm,
                "speaker_mean_norm": speaker_norm,
                "delta_norm_ratio": ratio,
            }
            residuals.append(entry)
            dialect_entries.append(entry)
        for left, right in combinations(sorted(dialect_vectors), 2):
            value = _cosine(dialect_vectors[left], dialect_vectors[right])
            cosine_values.append(value)
            residual_cosines.append(
                {
                    "speaker_id": speaker,
                    "dialect_pair": [left, right],
                    "cosine_similarity": value,
                }
            )
        speaker_means[speaker] = {
            "utterance_count": len(speaker_ids),
            "dialect_count": len(dialects),
            "speaker_mean_norm": speaker_norm,
            "delta_norm_ratio": float(mean([entry["delta_norm_ratio"] for entry in dialect_entries])),
            "dialects": dialect_entries,
        }

    change_report = _group_mae_changes(
        embeddings=embeddings,
        metadata=metadata,
        eligible_speakers=eligible,
        pairs=pairs,
        references=references,
    )
    return {
        "model_name": model_name,
        "eligible_speaker_count": len(eligible),
        "cross_dialect_speaker_count": len(eligible),
        "speaker_means": speaker_means,
        "dialect_residuals": residuals,
        "residual_cosines": residual_cosines,
        "speaker_mean_ratio_summary": _summary(ratio_values),
        "residual_cosine_summary": _summary(cosine_values),
        **change_report,
    }


def run_cross_dialect_decomposition(
    *,
    embeddings: Mapping[str, Mapping[str, Sequence[float]]],
    metadata: Mapping[str, Mapping[str, Any]],
    pairs: Sequence[Mapping[str, Any]] | None = None,
    references: Sequence[Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    return {
        "schema": "speaker-offset-cross-dialect-decomposition-v1",
        "models": [
            _model_decomposition(
                model_name=model_name,
                embeddings=model_embeddings,
                metadata=metadata,
                pairs=pairs,
                references=references,
            )
            for model_name, model_embeddings in sorted(embeddings.items())
        ],
    }


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--embeddings", nargs="+", action="append", required=True)
    parser.add_argument("--records", required=True)
    parser.add_argument("--evaluation-pairs")
    parser.add_argument("--reference-matrix", nargs="+", action="append")
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    args.embeddings = [path for group in args.embeddings for path in group]
    args.reference_matrix = [path for group in args.reference_matrix or [] for path in group]
    return args


def main() -> int:
    args = _parse_args()
    embeddings = _load_embeddings(args.embeddings)
    metadata = _metadata_for_embedding_sets(args.records, embeddings)
    pairs = _load_pairs(args.evaluation_pairs) if args.evaluation_pairs else None
    references = _load_references(args.reference_matrix) if args.reference_matrix else None
    report = run_cross_dialect_decomposition(
        embeddings=embeddings,
        metadata=metadata,
        pairs=pairs,
        references=references,
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
