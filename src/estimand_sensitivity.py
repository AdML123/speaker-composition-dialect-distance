"""Compute declared pair-error estimands from locked per-pair predictions."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable, Hashable, Iterable, Mapping, Sequence

import numpy as np
from scipy import sparse


WEIGHTINGS = ("pair", "endpoint_speaker", "dialect_relation", "matched_stratum")


def relation_key(labels: Iterable[Any]) -> tuple[str, ...]:
    """Return an order-invariant dialect-relation key."""
    return tuple(sorted(dict.fromkeys(map(str, labels))))


def equal_group_mean(
    rows: Sequence[Mapping[str, Any]],
    memberships: Callable[[Mapping[str, Any]], Iterable[Hashable]],
) -> float:
    grouped: dict[Hashable, list[float]] = defaultdict(list)
    for row in rows:
        for key in dict.fromkeys(memberships(row)):
            grouped[key].append(float(row["absolute_error"]))
    if not grouped:
        raise ValueError("weighting produced no groups")
    return float(np.mean([np.mean(values) for values in grouped.values()]))


def weighted_mae(rows: Iterable[Mapping[str, Any]], weighting: str) -> float:
    """Compute pair- or group-balanced mean absolute error."""
    checked = [dict(row) for row in rows]
    if not checked:
        raise ValueError("at least one row is required")
    if weighting == "pair":
        return float(np.mean([float(row["absolute_error"]) for row in checked]))
    if weighting == "endpoint_speaker":
        return equal_group_mean(
            checked, lambda row: tuple(dict.fromkeys(map(str, row["speaker_ids"])))
        )
    if weighting == "dialect_relation":
        return equal_group_mean(
            checked, lambda row: (relation_key(row["dialect_labels"]),)
        )
    if weighting == "matched_stratum":
        if any(not row.get("matched_stratum") for row in checked):
            raise ValueError("all rows require a defined matched_stratum")
        return equal_group_mean(
            checked, lambda row: (str(row["matched_stratum"]),)
        )
    raise ValueError(f"unknown weighting: {weighting}")


def _grouped_bootstrap_mean(
    row_weights: np.ndarray,
    errors: np.ndarray,
    group_ids: np.ndarray,
) -> np.ndarray:
    values = []
    for group in np.unique(group_ids):
        mask = group_ids == group
        weights = row_weights[:, mask]
        denominator = weights.sum(axis=1)
        numerator = weights @ errors[mask]
        values.append(
            np.divide(
                numerator,
                denominator,
                out=np.full_like(numerator, np.nan, dtype=float),
                where=denominator > 0,
            )
        )
    matrix = np.column_stack(values)
    finite = np.isfinite(matrix)
    return np.divide(
        np.nansum(matrix, axis=1),
        finite.sum(axis=1),
        out=np.full(matrix.shape[0], np.nan),
        where=finite.sum(axis=1) > 0,
    )


def _bootstrap_estimands(
    row_weights: np.ndarray,
    errors: np.ndarray,
    speaker_incidence: sparse.csr_matrix,
    sampled_speaker_counts: np.ndarray,
    relation_ids: np.ndarray,
    stratum_ids: np.ndarray,
) -> dict[str, np.ndarray]:
    denominator = row_weights.sum(axis=1)
    pair = np.divide(
        row_weights @ errors,
        denominator,
        out=np.full(row_weights.shape[0], np.nan),
        where=denominator > 0,
    )
    speaker_numerator = np.asarray((row_weights * errors) @ speaker_incidence)
    speaker_denominator = np.asarray(row_weights @ speaker_incidence)
    speaker_means = np.divide(
        speaker_numerator,
        speaker_denominator,
        out=np.full_like(speaker_numerator, np.nan, dtype=float),
        where=speaker_denominator > 0,
    )
    active_counts = np.where(np.isfinite(speaker_means), sampled_speaker_counts, 0.0)
    speaker = np.divide(
        np.nansum(speaker_means * active_counts, axis=1),
        active_counts.sum(axis=1),
        out=np.full(row_weights.shape[0], np.nan),
        where=active_counts.sum(axis=1) > 0,
    )
    return {
        "pair": pair,
        "endpoint_speaker": speaker,
        "dialect_relation": _grouped_bootstrap_mean(
            row_weights, errors, relation_ids
        ),
        "matched_stratum": _grouped_bootstrap_mean(
            row_weights, errors, stratum_ids
        ),
    }


def dyadic_weighting_bootstrap(
    baseline_rows: Sequence[Mapping[str, Any]],
    method_rows: Sequence[Mapping[str, Any]],
    *,
    seed: int,
    replicates: int,
) -> dict[str, Any]:
    """Bootstrap four MAE estimands by endpoint speaker within stratum."""
    if replicates <= 0:
        raise ValueError("replicates must be positive")
    baseline_by_id = {str(row["pair_id"]): dict(row) for row in baseline_rows}
    method_by_id = {str(row["pair_id"]): dict(row) for row in method_rows}
    if set(baseline_by_id) != set(method_by_id):
        raise ValueError("baseline and method require identical pair IDs")
    pair_ids = sorted(baseline_by_id)
    baseline = [baseline_by_id[pair_id] for pair_id in pair_ids]
    method = [method_by_id[pair_id] for pair_id in pair_ids]
    if any(not row.get("matched_stratum") for row in baseline):
        raise ValueError("dyadic bootstrap requires matched_stratum on every row")

    cluster_keys = sorted(
        {
            (str(row["matched_stratum"]), speaker)
            for row in baseline
            for speaker in dict.fromkeys(map(str, row["speaker_ids"]))
        }
    )
    cluster_index = {key: index for index, key in enumerate(cluster_keys)}
    global_speakers = sorted(
        {speaker for _, speaker in cluster_keys}
    )
    speaker_index = {speaker: index for index, speaker in enumerate(global_speakers)}
    clusters_by_stratum: dict[str, list[int]] = defaultdict(list)
    for index, (stratum, _) in enumerate(cluster_keys):
        clusters_by_stratum[stratum].append(index)

    rng = np.random.default_rng(seed)
    cluster_counts = np.zeros((replicates, len(cluster_keys)), dtype=np.float64)
    for indices in clusters_by_stratum.values():
        draws = rng.multinomial(
            len(indices), np.full(len(indices), 1.0 / len(indices)), size=replicates
        )
        cluster_counts[:, indices] = draws

    left_cluster = []
    right_cluster = []
    relation_labels = []
    stratum_labels = []
    incidence_rows = []
    incidence_columns = []
    for row_number, row in enumerate(baseline):
        stratum = str(row["matched_stratum"])
        speakers = list(dict.fromkeys(map(str, row["speaker_ids"])))
        left_cluster.append(cluster_index[(stratum, speakers[0])])
        right_cluster.append(cluster_index[(stratum, speakers[-1])])
        relation_labels.append("|".join(relation_key(row["dialect_labels"])))
        stratum_labels.append(stratum)
        for speaker in speakers:
            incidence_rows.append(row_number)
            incidence_columns.append(speaker_index[speaker])
    left_cluster_array = np.asarray(left_cluster, dtype=int)
    right_cluster_array = np.asarray(right_cluster, dtype=int)
    left_counts = cluster_counts[:, left_cluster_array]
    right_counts = cluster_counts[:, right_cluster_array]
    same_endpoint = left_cluster_array == right_cluster_array
    row_weights = left_counts * right_counts
    row_weights[:, same_endpoint] = left_counts[:, same_endpoint]

    speaker_incidence = sparse.csr_matrix(
        (
            np.ones(len(incidence_rows)),
            (incidence_rows, incidence_columns),
        ),
        shape=(len(pair_ids), len(global_speakers)),
    )
    cluster_to_speaker = sparse.csr_matrix(
        (
            np.ones(len(cluster_keys)),
            (
                np.arange(len(cluster_keys)),
                [speaker_index[speaker] for _, speaker in cluster_keys],
            ),
        ),
        shape=(len(cluster_keys), len(global_speakers)),
    )
    sampled_speaker_counts = np.asarray(cluster_counts @ cluster_to_speaker)
    baseline_errors = np.asarray(
        [float(row["absolute_error"]) for row in baseline], dtype=float
    )
    method_errors = np.asarray(
        [float(row["absolute_error"]) for row in method], dtype=float
    )
    relation_ids = np.asarray(relation_labels)
    stratum_ids = np.asarray(stratum_labels)
    baseline_bootstrap = _bootstrap_estimands(
        row_weights,
        baseline_errors,
        speaker_incidence,
        sampled_speaker_counts,
        relation_ids,
        stratum_ids,
    )
    method_bootstrap = _bootstrap_estimands(
        row_weights,
        method_errors,
        speaker_incidence,
        sampled_speaker_counts,
        relation_ids,
        stratum_ids,
    )

    output = {}
    for weighting in WEIGHTINGS:
        baseline_value = weighted_mae(baseline, weighting)
        method_value = weighted_mae(method, weighting)
        observed_gain = (baseline_value - method_value) / baseline_value
        bootstrap_gain = np.divide(
            baseline_bootstrap[weighting] - method_bootstrap[weighting],
            baseline_bootstrap[weighting],
            out=np.full(replicates, np.nan),
            where=baseline_bootstrap[weighting] != 0,
        )
        finite = bootstrap_gain[np.isfinite(bootstrap_gain)]
        if not len(finite):
            raise ValueError(f"no finite bootstrap estimates for {weighting}")
        output[weighting] = {
            "baseline_mae": baseline_value,
            "method_mae": method_value,
            "gain": observed_gain,
            "ci": {
                "lower": float(np.quantile(finite, 0.025)),
                "upper": float(np.quantile(finite, 0.975)),
                "confidence_level": 0.95,
            },
            "bootstrap_replicates": replicates,
            "finite_replicates": int(len(finite)),
            "resampling_unit": "endpoint_speaker_within_matched_stratum",
        }
    return output


def summarize_method(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    estimands: dict[str, float | None] = {}
    for weighting in WEIGHTINGS:
        try:
            estimands[weighting] = weighted_mae(rows, weighting)
        except ValueError as error:
            if weighting != "matched_stratum":
                raise
            estimands[weighting] = None
    return {
        "pair_count": len(rows),
        "estimands": estimands,
    }


def _method_rows(reference: Mapping[str, Any]) -> dict[str, Sequence[Mapping[str, Any]]]:
    methods: dict[str, Sequence[Mapping[str, Any]]] = {}
    if isinstance(reference.get("per_pair"), list):
        methods["linear_with_cross_loss"] = reference["per_pair"]
    comparison = reference.get("comparisons", {}).get("lambda_cross_zero", {})
    if isinstance(comparison.get("per_pair"), list):
        methods["linear_pair_only"] = comparison["per_pair"]
    return methods


def build_report(
    speaker_effect_path: Path, projection_report_path: Path
) -> tuple[dict[str, Any], dict[str, Any]]:
    speaker_effect = json.loads(speaker_effect_path.read_text(encoding="utf-8"))
    projection = json.loads(projection_report_path.read_text(encoding="utf-8"))
    references: dict[str, Any] = {}
    for model in projection.get("models", []):
        if model.get("model_name") != "chinese_hubert_large":
            continue
        for reference in model.get("references", []):
            name = str(reference["reference"])
            references[name] = {
                method: summarize_method(rows)
                for method, rows in _method_rows(reference).items()
            }

    report = {
        "schema": "estimand-weighting-sensitivity-v1",
        "status": "awaiting_architecture_factorial",
        "headline_weighting": "pair",
        "sensitivity_weightings": list(WEIGHTINGS[1:]),
        "weighting_definitions": {
            "pair": "mean absolute error over pair rows",
            "endpoint_speaker": "mean incident-pair error within each endpoint speaker, then equal mean over speakers",
            "dialect_relation": "mean error within each unordered dialect relation, then equal mean over relations",
            "matched_stratum": "mean error within each matched stratum, then equal mean over strata",
        },
        "primary_phenomenon": next(
            (
                row
                for row in speaker_effect.get("models", [])
                if row.get("model_name") == "chinese_hubert_large"
            ),
            None,
        ),
        "available_projection_methods": references,
        "pending_methods": [
            "frozen_affine",
            "mlp_parameter_matched",
            "mlp_wide",
        ],
        "source_files": [str(speaker_effect_path), str(projection_report_path)],
    }
    gate = {
        "schema": "estimand-weighting-gate-v1",
        "status": "awaiting_architecture_factorial",
        "required_weightings": list(WEIGHTINGS),
        "failure_action": "report_pair_weighted_result_as_weighting_dependent",
    }
    return report, gate


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--speaker-effect", type=Path, required=True)
    parser.add_argument("--projection-report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--gate", type=Path, default=Path("results/gates/estimand_weighting_gate.json")
    )
    args = parser.parse_args()
    report, gate = build_report(args.speaker_effect, args.projection_report)
    for path, payload in ((args.output, report), (args.gate, gate)):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
