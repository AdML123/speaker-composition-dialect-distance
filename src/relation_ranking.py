"""Relation-level summaries for operational dialect-distance references."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from itertools import combinations
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence
import warnings

import numpy as np
from scipy.stats import kendalltau, spearmanr


Relation = tuple[str, str]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_relation(labels: Iterable[Any]) -> Relation:
    """Return one unordered, off-diagonal dialect relation."""
    unique = sorted(dict.fromkeys(map(str, labels)))
    if len(unique) != 2:
        raise ValueError("relation must contain two distinct off-diagonal labels")
    return unique[0], unique[1]


def aggregate_relation_predictions(
    rows: Sequence[Mapping[str, Any]],
) -> dict[Relation, float]:
    """Average predicted distance for each unordered off-diagonal relation."""
    grouped: dict[Relation, list[float]] = defaultdict(list)
    for row in rows:
        labels = list(map(str, row["dialect_labels"]))
        if len(set(labels)) != 2:
            continue
        value = row.get("predicted_distance", row.get("distance"))
        if value is None:
            raise ValueError("row lacks predicted distance")
        grouped[canonical_relation(labels)].append(float(value))
    return {
        relation: float(np.mean(values))
        for relation, values in sorted(grouped.items())
    }


def reference_relations(
    reference: Mapping[str, Mapping[str, float]],
) -> dict[Relation, float]:
    labels = sorted(reference)
    return {
        (left, right): float(reference[left][right])
        for left, right in combinations(labels, 2)
    }


def ranking_metrics(
    reference: Mapping[Relation, float],
    prediction: Mapping[Relation, float],
) -> dict[str, Any]:
    """Compare relation ordering with a declared operational reference."""
    if any(left == right for left, right in reference):
        raise ValueError("ranking requires off-diagonal relations")
    keys = sorted(set(reference) & set(prediction))
    if len(keys) < 2:
        raise ValueError("ranking requires at least two shared relations")
    expected = np.asarray([float(reference[key]) for key in keys], dtype=float)
    observed = np.asarray([float(prediction[key]) for key in keys], dtype=float)
    if not np.isfinite(expected).all() or not np.isfinite(observed).all():
        raise ValueError("ranking inputs must be finite")

    correct = 0.0
    reversals = 0
    reference_ties = 0
    predicted_ties = 0
    comparable = 0
    for first, second in combinations(range(len(keys)), 2):
        expected_sign = np.sign(expected[first] - expected[second])
        observed_sign = np.sign(observed[first] - observed[second])
        if expected_sign == 0:
            reference_ties += 1
            continue
        comparable += 1
        if observed_sign == 0:
            predicted_ties += 1
            correct += 0.5
            continue
        if expected_sign == observed_sign:
            correct += 1
        else:
            reversals += 1

    spearman = spearmanr(expected, observed).statistic
    kendall = kendalltau(expected, observed, variant="b").statistic
    return {
        "relation_count": len(keys),
        "spearman": float(spearman),
        "kendall_tau_b": float(kendall),
        "pairwise_order_accuracy": (
            float(correct / comparable) if comparable else None
        ),
        "order_reversals": reversals,
        "ties": reference_ties + predicted_ties,
        "reference_ties_excluded": reference_ties,
        "predicted_ties_half_credit": predicted_ties,
        "predicted_tie_score": 0.5,
        "comparable_relation_pairs": comparable,
    }


def _identity(row: Mapping[str, Any]) -> tuple[Any, ...]:
    return (
        str(row["pair_id"]),
        tuple(sorted(map(str, row.get("utterance_ids", [])))),
        tuple(sorted(dict.fromkeys(map(str, row.get("speaker_ids", []))))),
        tuple(sorted(dict.fromkeys(map(str, row.get("dialect_labels", []))))),
    )


def _percentile(values: np.ndarray) -> dict[str, float]:
    finite = values[np.isfinite(values)]
    if not len(finite):
        raise ValueError("no finite bootstrap values")
    return {
        "lower": float(np.quantile(finite, 0.025)),
        "upper": float(np.quantile(finite, 0.975)),
        "confidence_level": 0.95,
    }


def _serial_values(values: np.ndarray) -> list[float | None]:
    return [float(value) if np.isfinite(value) else None for value in values]


def _relation_predictions_with_weights(
    rows: Sequence[Mapping[str, Any]], row_weights: np.ndarray
) -> dict[Relation, float]:
    grouped: dict[Relation, list[int]] = defaultdict(list)
    for index, row in enumerate(rows):
        if len(set(map(str, row["dialect_labels"]))) == 2:
            grouped[canonical_relation(row["dialect_labels"])].append(index)
    output = {}
    for relation, indices in grouped.items():
        weights = row_weights[indices]
        denominator = float(np.sum(weights))
        if denominator <= 0:
            continue
        values = np.asarray(
            [float(rows[index]["predicted_distance"]) for index in indices]
        )
        output[relation] = float(np.dot(weights, values) / denominator)
    return output


def clustered_ranking_bootstrap(
    method_seed_rows: Mapping[str, Sequence[Mapping[str, Any]]],
    reference: Mapping[Relation, float],
    *,
    seed: int,
    replicates: int,
    contrasts: Sequence[tuple[str, str]] = (),
    required_relation_count: int = 36,
) -> dict[str, Any]:
    """Bootstrap MAE and relation ordering with shared global speaker draws."""
    if replicates <= 0:
        raise ValueError("replicates must be positive")
    if len(reference) != required_relation_count:
        raise ValueError(
            f"expected {required_relation_count} reference relations, got {len(reference)}"
        )
    if not method_seed_rows:
        raise ValueError("at least one method is required")
    anchor = next(iter(method_seed_rows.values()))[0]["rows"]
    anchor_by_id = {str(row["pair_id"]): row for row in anchor}
    pair_ids = sorted(anchor_by_id)
    anchor_rows = [anchor_by_id[pair_id] for pair_id in pair_ids]
    for method, seed_rows in method_seed_rows.items():
        if not seed_rows:
            raise ValueError(f"{method} has no seed rows")
        for seed_row in seed_rows:
            indexed = {str(row["pair_id"]): row for row in seed_row["rows"]}
            if set(indexed) != set(pair_ids):
                raise ValueError(f"pair identity mismatch for {method}")
            for pair_id in pair_ids:
                if _identity(indexed[pair_id]) != _identity(anchor_by_id[pair_id]):
                    raise ValueError(f"pair identity mismatch for {method}:{pair_id}")

    speakers = sorted({
        speaker
        for row in anchor_rows
        for speaker in dict.fromkeys(map(str, row["speaker_ids"]))
    })
    speaker_index = {speaker: index for index, speaker in enumerate(speakers)}
    left = []
    right = []
    for row in anchor_rows:
        endpoints = list(dict.fromkeys(map(str, row["speaker_ids"])))
        left.append(speaker_index[endpoints[0]])
        right.append(speaker_index[endpoints[-1]])
    rng = np.random.default_rng(seed)
    counts = rng.multinomial(
        len(speakers), np.full(len(speakers), 1.0 / len(speakers)), size=replicates
    ).astype(float)
    left = np.asarray(left, dtype=int)
    right = np.asarray(right, dtype=int)
    row_weights = counts[:, left] * counts[:, right]
    same = left == right
    row_weights[:, same] = counts[:, left[same]]

    metric_names = ("spearman", "kendall_tau_b", "pairwise_order_accuracy")
    methods = {}
    aggregate_arrays = {}
    for method, seed_rows in method_seed_rows.items():
        seed_summaries = []
        seed_arrays: dict[str, list[np.ndarray]] = {
            "mae": [],
            **{name: [] for name in metric_names},
        }
        for seed_row in sorted(seed_rows, key=lambda row: int(row["seed"])):
            indexed = {str(row["pair_id"]): row for row in seed_row["rows"]}
            rows = [indexed[pair_id] for pair_id in pair_ids]
            point_ranking = ranking_metrics(
                reference, aggregate_relation_predictions(rows)
            )
            errors = np.asarray([float(row["absolute_error"]) for row in rows])
            point_mae = float(np.mean(errors))
            denominator = row_weights.sum(axis=1)
            mae_bootstrap = np.divide(
                row_weights @ errors,
                denominator,
                out=np.full(replicates, np.nan),
                where=denominator > 0,
            )
            ranking_bootstrap = {
                name: np.full(replicates, np.nan) for name in metric_names
            }
            for replicate in range(replicates):
                prediction = _relation_predictions_with_weights(
                    rows, row_weights[replicate]
                )
                if len(prediction) < 2:
                    continue
                metrics = ranking_metrics(reference, prediction)
                for name in metric_names:
                    value = metrics[name]
                    if value is not None:
                        ranking_bootstrap[name][replicate] = float(value)
            seed_summaries.append({
                "seed": int(seed_row["seed"]),
                "mae": point_mae,
                "ordering": {
                    name: float(point_ranking[name]) for name in metric_names
                },
            })
            seed_arrays["mae"].append(mae_bootstrap)
            for name in metric_names:
                seed_arrays[name].append(ranking_bootstrap[name])
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            aggregated = {
                name: np.nanmedian(np.asarray(values), axis=0)
                for name, values in seed_arrays.items()
            }
        aggregate_arrays[method] = aggregated
        methods[method] = {
            "seed_count": len(seed_summaries),
            "seed_values": seed_summaries,
            "mae": {
                "point_estimate": float(np.median([row["mae"] for row in seed_summaries])),
                "ci": _percentile(aggregated["mae"]),
                "bootstrap_values": _serial_values(aggregated["mae"]),
            },
            "ordering": {
                name: {
                    "point_estimate": float(np.median([
                        row["ordering"][name] for row in seed_summaries
                    ])),
                    "ci": _percentile(aggregated[name]),
                    "bootstrap_values": _serial_values(aggregated[name]),
                }
                for name in metric_names
            },
        }

    contrast_output = {}
    for method, comparator in contrasts:
        key = f"{method}_minus_{comparator}"
        contrast_output[key] = {}
        for metric in (*metric_names, "mae"):
            values = aggregate_arrays[method][metric] - aggregate_arrays[comparator][metric]
            if metric == "mae":
                method_point = methods[method]["mae"]["point_estimate"]
                comparator_point = methods[comparator]["mae"]["point_estimate"]
            else:
                method_point = methods[method]["ordering"][metric]["point_estimate"]
                comparator_point = methods[comparator]["ordering"][metric]["point_estimate"]
            contrast_output[key][metric] = {
                "point_estimate": float(method_point - comparator_point),
                "ci": _percentile(values),
                "bootstrap_values": _serial_values(values),
            }
    return {
        "relation_count": len(reference),
        "pair_count": len(anchor_rows),
        "speaker_count": len(speakers),
        "bootstrap_replicates": replicates,
        "bootstrap_seed": seed,
        "resampling_unit": "global_endpoint_speaker",
        "shared_resamples_across_methods": True,
        "seed_aggregation": "median within paired speaker-bootstrap replicate",
        "tie_policy": {
            "zero_reference_difference": "excluded",
            "predicted_tie": "half_credit",
        },
        "methods": methods,
        "contrasts": contrast_output,
    }


def _load_matrix(path: Path) -> dict[str, dict[str, float]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload.get("matrix", payload)


def _attach_manifest_identity(
    rows: Sequence[Mapping[str, Any]],
    manifest_rows: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    indexed = {str(row["pair_id"]): row for row in manifest_rows}
    output = []
    for row in rows:
        pair_id = str(row["pair_id"])
        source = indexed.get(pair_id)
        if source is None:
            raise ValueError(f"pair identity mismatch: {pair_id} missing")
        actual = tuple(sorted(map(str, row.get("utterance_ids", []))))
        expected = tuple(sorted(map(str, source["source_utterance_ids"])))
        if actual != expected:
            raise ValueError(f"pair identity mismatch: {pair_id} endpoints")
        merged = dict(row)
        merged["speaker_ids"] = list(map(str, source["speaker_ids"]))
        merged["dialect_labels"] = list(map(str, source["dialect_labels"]))
        output.append(merged)
    if len(output) != len(manifest_rows):
        raise ValueError("pair identity mismatch: row count")
    return output


def _primary_methods(
    baselines: Mapping[str, Any],
    architecture: Mapping[str, Any],
    reference: str,
    manifest_rows: Sequence[Mapping[str, Any]],
) -> dict[str, list[dict[str, Any]]]:
    methods = baselines["references"][reference]["methods"]
    output = {
        "frozen_affine": [{
            "seed": 0,
            "rows": _attach_manifest_identity(
                methods["frozen_affine"]["per_pair"], manifest_rows
            ),
        }],
        "principal_component": [{
            "seed": 0,
            "rows": _attach_manifest_identity(
                methods["frozen_pca256_affine"]["per_pair"], manifest_rows
            ),
        }],
        "diagonal_metric": [
            {
                "seed": int(row["seed"]),
                "rows": _attach_manifest_identity(row["per_pair"], manifest_rows),
            }
            for row in methods["diagonal_metric"]["seed_results"]
        ],
    }
    for head, name in (
        ("linear", "linear"),
        ("mlp_parameter_matched", "matched_mlp"),
        ("mlp_wide", "wide_mlp"),
    ):
        cells = [
            row for row in architecture["cells"]
            if str(row["reference"]) == reference
            and str(row["head"]) == head
            and float(row["lambda_cross"]) == 0.0
        ]
        output[name] = [
            {
                "seed": int(row["seed"]),
                "rows": _attach_manifest_identity(row["per_pair"], manifest_rows),
            }
            for row in cells
        ]
    return output


def _pareto_flags(methods: Mapping[str, Any], ordering_metric: str) -> dict[str, bool]:
    points = {
        name: (
            float(row["mae"]["point_estimate"]),
            float(row["ordering"][ordering_metric]["point_estimate"]),
        )
        for name, row in methods.items()
    }
    return {
        name: not any(
            other != name
            and other_point[0] <= point[0]
            and other_point[1] >= point[1]
            and (other_point[0] < point[0] or other_point[1] > point[1])
            for other, other_point in points.items()
        )
        for name, point in points.items()
    }


REQUIRED_REPORT_REFERENCES = {
    "taxonomy", "city_nearest", "subgroup_medoid", "subgroup_aggregate"
}
REQUIRED_REPORT_METHODS = {
    "frozen_affine", "principal_component", "diagonal_metric",
    "linear", "matched_mlp", "wide_mlp",
}


def validate_complete_clustered_report(report: Mapping[str, Any]) -> None:
    references = report.get("references", {})
    if set(references) != REQUIRED_REPORT_REFERENCES:
        raise ValueError("clustered report reference set is incomplete")
    required_metrics = {"spearman", "kendall_tau_b", "pairwise_order_accuracy"}
    for reference, row in references.items():
        methods = row.get("methods", {})
        if set(methods) != REQUIRED_REPORT_METHODS:
            raise ValueError(f"clustered report method set is incomplete for {reference}")
        for method, method_row in methods.items():
            if "point_estimate" not in method_row.get("mae", {}) or "ci" not in method_row.get("mae", {}):
                raise ValueError(f"MAE interval missing for {reference}:{method}")
            if set(method_row.get("ordering", {})) != required_metrics:
                raise ValueError(f"ordering interval missing for {reference}:{method}")
        contrasts = row.get("contrasts", {})
        ordering_contrast = contrasts.get("matched_mlp_minus_linear")
        mae_contrast = contrasts.get("linear_minus_matched_mlp")
        if not ordering_contrast or not required_metrics <= set(ordering_contrast):
            raise ValueError(f"ordering contrast missing for {reference}")
        if not mae_contrast or set(mae_contrast) != {"mae"}:
            raise ValueError(f"MAE contrast missing for {reference}")


def build_practical_gate(
    ordering_support: Mapping[str, Mapping[str, bool]],
    mae_support: Mapping[str, bool],
) -> dict[str, Any]:
    all_ordering_supported = all(
        all(checks.values()) for checks in ordering_support.values()
    )
    return {
        "schema": "practical-consequence-gate-v2",
        "status": "passed" if all_ordering_supported else "narrowed",
        "paired_contrast": "parameter_matched_mlp_minus_linear",
        "contrast_support": dict(ordering_support),
        "ordering_contrast_support": dict(ordering_support),
        "mae_contrast_support": dict(mae_support),
        "better_ordering_wording_allowed": all_ordering_supported,
        "selected_wording": (
            "The parameter-matched MLP has better operational relation ordering under both primary references. The linear map has lower MAE point estimates under both, with paired-interval support under the binary reference."
            if all_ordering_supported
            else "The parameter-matched MLP has higher operational ordering point estimates; unsupported contrasts are described without superiority language, and MAE support is reported separately."
        ),
        "failure_wording": (
            "If a paired interval includes zero, report only the higher point "
            "estimate for that metric and reference."
        ),
    }


def _separate_endpoint_contrasts(clustered: dict[str, Any]) -> None:
    for key in ("matched_mlp_minus_linear", "wide_mlp_minus_linear"):
        clustered["contrasts"][key].pop("mae", None)
    reverse = clustered["contrasts"]["linear_minus_matched_mlp"]
    clustered["contrasts"]["linear_minus_matched_mlp"] = {
        "mae": reverse["mae"]
    }


def build_clustered_report(
    *,
    architecture_path: Path,
    baselines_path: Path,
    reference_sweep_path: Path,
    manifest_path: Path,
    reference_paths: Mapping[str, Path],
    seed: int,
    replicates: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Build dual-endpoint primary and reference-construction reports."""
    architecture = json.loads(architecture_path.read_text(encoding="utf-8"))
    baselines = json.loads(baselines_path.read_text(encoding="utf-8"))
    sweep = json.loads(reference_sweep_path.read_text(encoding="utf-8"))
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))["pairs"]
    references = {}
    if "references" in sweep:
        for reference, reference_row in sweep["references"].items():
            method_rows = {
                method: [
                    {
                        "seed": int(seed_row["seed"]),
                        "rows": _attach_manifest_identity(
                            seed_row["per_pair"], manifest
                        ),
                    }
                    for seed_row in method_row["seed_results"]
                ]
                for method, method_row in reference_row["methods"].items()
            }
            clustered = clustered_ranking_bootstrap(
                method_rows,
                reference_relations(_load_matrix(reference_paths[reference])),
                seed=seed,
                replicates=replicates,
                contrasts=[
                    ("matched_mlp", "linear"), ("wide_mlp", "linear"),
                    ("linear", "matched_mlp"),
                ],
            )
            _separate_endpoint_contrasts(clustered)
            clustered["pareto_metric"] = "spearman"
            clustered["pareto_flags"] = _pareto_flags(
                clustered["methods"], "spearman"
            )
            references[reference] = clustered
    else:
        for reference in ("taxonomy", "city_nearest"):
            matrix = _load_matrix(reference_paths[reference])
            clustered = clustered_ranking_bootstrap(
                _primary_methods(baselines, architecture, reference, manifest),
                reference_relations(matrix),
                seed=seed,
                replicates=replicates,
                contrasts=[
                    ("matched_mlp", "linear"), ("wide_mlp", "linear"),
                    ("linear", "matched_mlp"),
                ],
            )
            _separate_endpoint_contrasts(clustered)
            clustered["pareto_metric"] = "spearman"
            clustered["pareto_flags"] = _pareto_flags(
                clustered["methods"], "spearman"
            )
            references[reference] = clustered
        for variant in sweep["projection_sweep"]["variants"]:
            name = str(variant["variant"])
            if name not in {"subgroup_medoid", "subgroup_aggregate"}:
                continue
            method_rows = {
                "frozen_affine": [{
                    "seed": 0,
                    "rows": _attach_manifest_identity(
                        variant["seed_results"][0]["baseline_per_pair"], manifest
                    ),
                }],
                "linear": [
                    {
                        "seed": int(row["seed"]),
                        "rows": _attach_manifest_identity(row["per_pair"], manifest),
                    }
                    for row in variant["seed_results"]
                ],
            }
            references[name] = clustered_ranking_bootstrap(
                method_rows,
                reference_relations(_load_matrix(reference_paths[name])),
                seed=seed,
                replicates=replicates,
            )

    primary = {name: references[name] for name in ("taxonomy", "city_nearest")}
    variants = {
        name: references[name]
        for name in ("subgroup_medoid", "subgroup_aggregate")
        if name in references
    }

    contrast_rows = {
        reference: row["contrasts"]["matched_mlp_minus_linear"]
        for reference, row in primary.items()
    }
    inferential_support = {
        reference: {
            metric: float(contrast[metric]["ci"]["lower"]) > 0.0
            for metric in ("spearman", "kendall_tau_b", "pairwise_order_accuracy")
        }
        for reference, contrast in contrast_rows.items()
    }
    mae_support = {
        reference: float(
            row["contrasts"]["linear_minus_matched_mlp"]["mae"]["ci"]["upper"]
        ) < 0.0
        for reference, row in primary.items()
    }
    report = {
        "schema": "relation-ranking-clustered-v1",
        "status": "evaluated",
        "model": "chinese_hubert_large",
        "source_hashes": {
            "projection_manifest": _sha256(manifest_path),
        },
        "primary_endpoints": {
            "calibration": "pair-weighted mean absolute error against the named operational reference",
            "ordering": (
                "relation-level Spearman, Kendall tau-b, and pairwise ordering "
                "accuracy against the named operational reference"
            ),
        },
        "global_best_model_defined": False,
        "primary_references": primary,
        "reference_construction_sensitivity": variants,
        "references": references,
        "matched_mlp_minus_linear_support": inferential_support,
        "linear_minus_matched_mlp_mae_support": mae_support,
        "boundary": (
            "Ordering measures agreement with operational references, not "
            "perceptual ground truth or downstream task utility."
        ),
    }
    gate = build_practical_gate(inferential_support, mae_support)
    if "references" in sweep:
        validate_complete_clustered_report(report)
    return report, gate


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--architecture", type=Path, required=True)
    parser.add_argument("--baselines", type=Path, required=True)
    parser.add_argument("--reference-sweep", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--references", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--gate", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260905)
    parser.add_argument("--replicates", type=int, default=1000)
    args = parser.parse_args()
    reference_paths = json.loads(args.references.read_text(encoding="utf-8"))
    report, gate = build_clustered_report(
        architecture_path=args.architecture,
        baselines_path=args.baselines,
        reference_sweep_path=args.reference_sweep,
        manifest_path=args.manifest,
        reference_paths={key: Path(value) for key, value in reference_paths.items()},
        seed=args.seed,
        replicates=args.replicates,
    )
    for path, payload in ((args.output, report), (args.gate, gate)):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


def ordering_changes(
    baseline: Mapping[Relation, float], method: Mapping[Relation, float]
) -> dict[str, int]:
    """Count pairwise relation-order reversals against a baseline ordering."""
    keys = sorted(set(baseline) & set(method))
    if len(keys) < 2:
        raise ValueError("ordering comparison requires at least two shared relations")
    reversals = 0
    ties = 0
    comparable = 0
    for first, second in combinations(keys, 2):
        baseline_sign = np.sign(float(baseline[first]) - float(baseline[second]))
        method_sign = np.sign(float(method[first]) - float(method[second]))
        if baseline_sign == 0 or method_sign == 0:
            ties += 1
            continue
        comparable += 1
        reversals += int(baseline_sign != method_sign)
    return {
        "reversals_vs_baseline": reversals,
        "ties_in_either_ordering": ties,
        "comparable_relation_pairs": comparable,
    }
