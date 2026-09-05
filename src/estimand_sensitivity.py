"""Compute declared pair-error estimands from locked per-pair predictions."""

from __future__ import annotations

import argparse
import hashlib
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
    """Bootstrap four paired MAE estimands by global endpoint speaker."""
    if replicates <= 0:
        raise ValueError("replicates must be positive")
    baseline_by_id = {str(row["pair_id"]): dict(row) for row in baseline_rows}
    method_by_id = {str(row["pair_id"]): dict(row) for row in method_rows}
    if set(baseline_by_id) != set(method_by_id):
        raise ValueError("baseline and method require identical pair IDs")
    pair_ids = sorted(baseline_by_id)
    baseline = [baseline_by_id[pair_id] for pair_id in pair_ids]
    method = [method_by_id[pair_id] for pair_id in pair_ids]
    identity_fields = ("utterance_ids", "speaker_ids", "dialect_labels", "matched_stratum")
    for pair_id, baseline_row, method_row in zip(pair_ids, baseline, method):
        for field in identity_fields:
            left = baseline_row.get(field)
            right = method_row.get(field)
            if field in {"speaker_ids", "dialect_labels"}:
                left = tuple(sorted(dict.fromkeys(map(str, left or []))))
                right = tuple(sorted(dict.fromkeys(map(str, right or []))))
            elif field == "utterance_ids":
                left = tuple(sorted(map(str, left or [])))
                right = tuple(sorted(map(str, right or [])))
            if left != right:
                raise ValueError(f"pair identity mismatch for {pair_id}: {field}")
    if any(not row.get("matched_stratum") for row in baseline):
        raise ValueError("dyadic bootstrap requires matched_stratum on every row")

    global_speakers = sorted({
        speaker
        for row in baseline
        for speaker in dict.fromkeys(map(str, row["speaker_ids"]))
    })
    speaker_index = {speaker: index for index, speaker in enumerate(global_speakers)}

    rng = np.random.default_rng(seed)
    speaker_counts = rng.multinomial(
        len(global_speakers),
        np.full(len(global_speakers), 1.0 / len(global_speakers)),
        size=replicates,
    ).astype(np.float64)

    left_cluster = []
    right_cluster = []
    relation_labels = []
    stratum_labels = []
    incidence_rows = []
    incidence_columns = []
    for row_number, row in enumerate(baseline):
        speakers = list(dict.fromkeys(map(str, row["speaker_ids"])))
        left_cluster.append(speaker_index[speakers[0]])
        right_cluster.append(speaker_index[speakers[-1]])
        relation_labels.append("|".join(relation_key(row["dialect_labels"])))
        stratum_labels.append(str(row["matched_stratum"]))
        for speaker in speakers:
            incidence_rows.append(row_number)
            incidence_columns.append(speaker_index[speaker])
    left_cluster_array = np.asarray(left_cluster, dtype=int)
    right_cluster_array = np.asarray(right_cluster, dtype=int)
    left_counts = speaker_counts[:, left_cluster_array]
    right_counts = speaker_counts[:, right_cluster_array]
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
    sampled_speaker_counts = speaker_counts
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
            "resampling_unit": "global_endpoint_speaker",
            "gain_unit": "fraction",
            "gain_orientation": "(baseline_mae-method_mae)/baseline_mae",
            "bootstrap_gain": list(map(float, bootstrap_gain)),
        }
    return output


def attach_design_strata(
    rows: Sequence[Mapping[str, Any]],
    manifest_rows: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Attach a symmetric design stratum after verifying pair endpoints."""
    manifest_by_id = {str(row["pair_id"]): row for row in manifest_rows}
    if len(manifest_by_id) != len(manifest_rows):
        raise ValueError("design manifest contains duplicate pair IDs")
    output = []
    for row in rows:
        pair_id = str(row["pair_id"])
        if pair_id not in manifest_by_id:
            raise ValueError(f"pair identity mismatch for {pair_id}: missing manifest row")
        manifest = manifest_by_id[pair_id]
        actual = tuple(sorted(map(str, row.get("utterance_ids", []))))
        expected = tuple(sorted(map(str, manifest.get("source_utterance_ids", []))))
        if actual != expected:
            raise ValueError(f"pair identity mismatch for {pair_id}: utterance_ids")
        checked = dict(row)
        checked["matched_stratum"] = str(manifest["matched_stratum"])
        output.append(checked)
    if len(output) != len(manifest_rows):
        raise ValueError("pair identity mismatch: row count differs from manifest")
    return output


def aggregate_seed_bootstraps(
    seed_results: Sequence[Mapping[str, Any]], weighting: str
) -> dict[str, Any]:
    """Summarize a five-seed endpoint using paired replicate medians."""
    if not seed_results:
        raise ValueError("at least one seed result is required")
    seed_values = [
        {
            "seed": int(row["seed"]),
            "gain": float(row["estimands"][weighting]["gain"]),
        }
        for row in seed_results
    ]
    bootstrap = np.asarray(
        [row["estimands"][weighting]["bootstrap_gain"] for row in seed_results],
        dtype=float,
    )
    if bootstrap.ndim != 2 or len({len(row) for row in bootstrap}) != 1:
        raise ValueError("seed bootstraps require the same paired replicate count")
    replicate_medians = np.nanmedian(bootstrap, axis=0)
    finite = replicate_medians[np.isfinite(replicate_medians)]
    if not len(finite):
        raise ValueError("no finite paired seed-median replicates")
    first = seed_results[0]["estimands"][weighting]
    return {
        "gain": float(np.median([row["gain"] for row in seed_values])),
        "baseline_mae": float(first["baseline_mae"]),
        "method_mae_seed_median": float(
            np.median([
                row["estimands"][weighting]["method_mae"] for row in seed_results
            ])
        ),
        "ci": {
            "lower": float(np.quantile(finite, 0.025)),
            "upper": float(np.quantile(finite, 0.975)),
            "confidence_level": 0.95,
        },
        "seed_values": seed_values,
        "bootstrap_replicates": int(bootstrap.shape[1]),
        "finite_replicates": int(len(finite)),
        "independent_unit": "global_endpoint_speaker",
        "gain_unit": "fraction",
        "gain_orientation": "(baseline_mae-method_mae)/baseline_mae",
    }


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


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _sample_size(rows: Sequence[Mapping[str, Any]]) -> dict[str, int]:
    return {
        "pairs": len(rows),
        "speakers": len({
            str(speaker) for row in rows for speaker in row["speaker_ids"]
        }),
        "dialect_relations": len({
            relation_key(row["dialect_labels"]) for row in rows
        }),
        "design_strata": len({str(row["matched_stratum"]) for row in rows}),
    }


def _audit_old_report(payload: Mapping[str, Any]) -> dict[str, Any]:
    inconsistent = []
    checked = 0
    for run in payload.get("runs", []):
        for weighting, result in run.get("estimands", {}).items():
            baseline = float(result["baseline_mae"])
            method = float(result["method_mae"])
            expected = (baseline - method) / baseline
            stored = float(result["gain"])
            checked += 1
            if not np.isclose(expected, stored, atol=1e-12, rtol=1e-10):
                inconsistent.append({
                    "reference": run.get("reference"),
                    "method": run.get("method"),
                    "seed": run.get("seed"),
                    "weighting": weighting,
                    "stored_gain": stored,
                    "recalculated_gain": expected,
                })
    return {
        "source_schema": payload.get("schema"),
        "checked_estimand_rows": checked,
        "gain_orientation": "(baseline_mae-method_mae)/baseline_mae",
        "gain_unit": "fraction",
        "orientation_mismatches": inconsistent,
        "orientation_verified": not inconsistent,
        "invalidated_dimension": "matched_stratum",
        "invalidation_reason": (
            "The superseded annotation used one ordered endpoint for cross-relation "
            "strata; all intervals are regenerated with the symmetric 126-stratum key."
        ),
    }


def build_interval_report(
    *,
    architecture_path: Path,
    baselines_path: Path,
    design_manifest_path: Path,
    old_report_path: Path,
    seed: int,
    replicates: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Build complete primary HuBERT MAE intervals from locked pair rows."""
    architecture = json.loads(architecture_path.read_text(encoding="utf-8"))
    baselines = json.loads(baselines_path.read_text(encoding="utf-8"))
    design_manifest = json.loads(design_manifest_path.read_text(encoding="utf-8"))
    old_report = json.loads(old_report_path.read_text(encoding="utf-8"))
    manifest_rows = design_manifest["pairs"]
    references: dict[str, Any] = {}
    all_branch_points = []
    for reference_name in ("taxonomy", "city_nearest"):
        baseline_source = baselines["references"][reference_name]["methods"]
        baseline_rows = attach_design_strata(
            baseline_source["frozen_affine"]["per_pair"], manifest_rows
        )
        primary_cells = [
            cell
            for cell in architecture["cells"]
            if str(cell["reference"]) == reference_name
            and str(cell["head"]) == "linear"
            and float(cell["lambda_cross"]) == 0.0
        ]
        if len(primary_cells) != 5:
            raise ValueError(
                f"{reference_name} requires five linear pair-only seed cells"
            )
        seed_results = []
        for cell in sorted(primary_cells, key=lambda row: int(row["seed"])):
            method_rows = attach_design_strata(cell["per_pair"], manifest_rows)
            estimands = dyadic_weighting_bootstrap(
                baseline_rows,
                method_rows,
                seed=seed,
                replicates=replicates,
            )
            seed_results.append({
                "seed": int(cell["seed"]),
                "method": "linear",
                "lambda_cross": 0.0,
                "estimands": estimands,
            })
        sample_size = _sample_size(baseline_rows)
        aggregate = {}
        for weighting in WEIGHTINGS:
            row = aggregate_seed_bootstraps(seed_results, weighting)
            row["sample_size"] = sample_size
            row["estimand_definition"] = {
                "pair": "equal mean over pair rows",
                "endpoint_speaker": (
                    "equal mean over speakers after averaging incident-pair errors"
                ),
                "dialect_relation": (
                    "equal mean over unordered dialect relations after within-relation averaging"
                ),
                "matched_stratum": (
                    "equal mean over nonempty symmetric design strata after within-stratum averaging"
                ),
            }[weighting]
            aggregate[weighting] = row
        references[reference_name] = {
            "baseline": "frozen_affine",
            "sample_size": sample_size,
            "baseline_estimands": {
                weighting: weighted_mae(baseline_rows, weighting)
                for weighting in WEIGHTINGS
            },
            "seed_results": seed_results,
            "estimands": aggregate,
        }

        for cell in architecture["cells"]:
            if str(cell["reference"]) != reference_name:
                continue
            method_rows = attach_design_strata(cell["per_pair"], manifest_rows)
            all_branch_points.append({
                "reference": reference_name,
                "head": str(cell["head"]),
                "lambda_cross": float(cell["lambda_cross"]),
                "seed": int(cell["seed"]),
                "estimands": {
                    weighting: weighted_mae(method_rows, weighting)
                    for weighting in WEIGHTINGS
                },
            })

    lower_bounds = [
        float(row["ci"]["lower"])
        for reference in references.values()
        for row in reference["estimands"].values()
    ]
    report = {
        "schema": "estimand-weighting-intervals-v1",
        "status": "evaluated",
        "primary_method": "hubert_linear_pair_only",
        "model": "chinese_hubert_large",
        "seed_count": 5,
        "weightings": list(WEIGHTINGS),
        "manuscript_weighting_name": {
            "matched_stratum": "design_stratum"
        },
        "bootstrap": {
            "replicates": replicates,
            "random_seed": seed,
            "resampling_unit": "global_endpoint_speaker",
            "paired_across_methods_references_and_seeds": True,
            "same_speaker_pair_multiplicity": "one speaker multiplicity",
            "different_speaker_pair_multiplicity": "product of endpoint multiplicities",
            "seed_aggregation": "median gain within each paired replicate",
        },
        "old_report_audit": _audit_old_report(old_report),
        "source_files": {
            str(path): _sha256(path)
            for path in (
                architecture_path,
                baselines_path,
                design_manifest_path,
                old_report_path,
            )
        },
        "source_hashes": {
            "projection_manifest": str(design_manifest["base_manifest_sha256"]),
            "design_strata_manifest": _sha256(design_manifest_path),
        },
        "references": references,
        "trainable_branch_point_estimands": all_branch_points,
        "trainable_branch_point_count": len(all_branch_points),
        "boundary": (
            "Intervals quantify agreement with two operational references for "
            "HuBERT on one locked KeSpeech pair manifest; they do not validate "
            "perceptual dialect geometry."
        ),
    }
    status = "passed" if all(value > 0 for value in lower_bounds) else "narrowed"
    gate = {
        "schema": "estimand-weighting-gate-v3",
        "status": status,
        "primary_method": "hubert_linear_pair_only",
        "required_weightings": list(WEIGHTINGS),
        "required_reference_count": 2,
        "all_rows_complete": True,
        "all_paired_cluster_lower_bounds_positive": all(
            value > 0 for value in lower_bounds
        ),
        "gain_unit": "fraction",
        "gain_orientation": "(baseline_mae-method_mae)/baseline_mae",
        "selected_wording": (
            "The HuBERT linear pair-only gain retains its direction under all four declared MAE estimands."
            if status == "passed"
            else "The HuBERT linear pair-only result is limited to the named MAE estimands whose paired intervals remain positive."
        ),
        "failure_wording": (
            "A reversed direction limits efficacy to the pair-weighted endpoint; "
            "an interval crossing zero is reported as imprecise."
        ),
    }
    return report, gate


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
    parser.add_argument("--architecture", type=Path, required=True)
    parser.add_argument("--baselines", type=Path, required=True)
    parser.add_argument("--design-manifest", type=Path, required=True)
    parser.add_argument("--old-report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--gate", type=Path, default=Path("results/gates/estimand_weighting_gate.json")
    )
    parser.add_argument("--seed", type=int, default=20260905)
    parser.add_argument("--replicates", type=int, default=1000)
    args = parser.parse_args()
    report, gate = build_interval_report(
        architecture_path=args.architecture,
        baselines_path=args.baselines,
        design_manifest_path=args.design_manifest,
        old_report_path=args.old_report,
        seed=args.seed,
        replicates=args.replicates,
    )
    for path, payload in ((args.output, report), (args.gate, gate)):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
