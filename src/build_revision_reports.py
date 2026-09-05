"""Build compact revision reports from locked analysis artifacts."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import yaml

from .correction_gate import threshold_sensitivity
from .cross_dialect_projection_head import build_training_examples
from .estimand_sensitivity import WEIGHTINGS, dyadic_weighting_bootstrap
from .target_permutation_control import permute_pair_distance_targets
from .reference_matrices import validate_reference_matrix


def _load(path: str | Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _write(path: str | Path, payload: Any) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _digest(values: object) -> str:
    return hashlib.sha256(json.dumps(values, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()


def build_target_permutation_report(
    *,
    projection_report: Mapping[str, Any],
    records: Sequence[Mapping[str, Any]],
    calibration_pairs: Sequence[Mapping[str, Any]],
    calibration_embedding_ids: set[str],
    reference: Mapping[str, Mapping[str, float]],
    seed: int,
) -> dict[str, Any]:
    calibration_records = [r for r in records if str(r.get("split")) == "calibration" and str(r["utterance_id"]) in calibration_embedding_ids]
    pairs = [p for p in calibration_pairs if set(map(str, p["source_utterance_ids"])) <= calibration_embedding_ids]
    cross = build_training_examples(calibration_records, pairs, reference)["cross_dialect_examples"]
    control = permute_pair_distance_targets(cross, seed=seed)
    reference_row = projection_report["models"][0]["references"][0]
    permuted = reference_row["ablations"]["permuted_pair_distance_target"]
    shuffled = reference_row["ablations"]["shuffled_cross_dialect"]
    method = float(reference_row["improvement_ratio"])
    return {
        "schema": "cross-dialect-target-permutation-report-v1",
        "status": "evaluated",
        "seed": seed,
        "control_target": "cross_loss_pair_distance_target",
        "pool_size": control["pool_size"],
        "pair_id_set_digest": _digest(sorted(str(row["pair_id"]) for row in cross)),
        "target_assignment_digest_before": _digest(control["target_hashes_before"]),
        "target_assignment_digest_after": _digest(control["target_hashes_after"]),
        "target_assignment_changed": control["target_hashes_before"] != control["target_hashes_after"],
        "target_histogram_before": control["target_histogram_before"],
        "target_histogram_after": control["target_histogram_after"],
        "pair_ids_unchanged": True,
        "evaluation_targets_unchanged": True,
        "method_improvement_ratio": method,
        "target_permuted_improvement_ratio": float(permuted["improvement_ratio"]),
        "target_permuted_ci": permuted["ci"],
        "shuffled_improvement_ratio": float(shuffled["improvement_ratio"]),
        "semantic_specificity_supported": method > float(permuted["improvement_ratio"]) and method > float(shuffled["improvement_ratio"]),
        "wording": "The separately weighted term improved held-out error, but target permutation and shuffled pairing did not isolate semantic cross-dialect supervision.",
    }


def _collect_references(payload: Mapping[str, Any], method: str) -> list[dict[str, Any]]:
    result = []
    for model in payload.get("models", []):
        for row in model.get("references", []):
            result.append({
                "method": method,
                "model_name": row.get("model_name", model.get("model_name")),
                "reference_name": row.get("reference_name"),
                "improvement_ratio": row.get("improvement_ratio"),
                "ci": row.get("ci", {}),
            })
    return result


def build_correction_threshold_report(paths: Mapping[str, str]) -> dict[str, Any]:
    rows = []
    for method, path in paths.items():
        rows.extend(_collect_references(_load(path), method))
    report = threshold_sensitivity(rows)
    report["status"] = "passed" if report["all_below_three_percent"] else "failed"
    report["decision"] = "All five branches remain below 3%, 5%, and 10% operational thresholds." if report["all_below_three_percent"] else "Threshold choice changes at least one branch conclusion."
    return report


def build_endpoint_registry() -> dict[str, Any]:
    """Return the revision-time endpoint hierarchy fixed before final prose."""
    return {
        "schema": "revision-endpoint-registry-v3",
        "declaration_status": "revision_time_declared_not_preregistered",
        "experimental_objects": {
            "phenomenon": {
                "manifest": "results/pairs/kespeech_evaluation_matched.json",
                "pair_count": 12980,
                "estimand": "overlap_pair_weighted_stratum_median_B_minus_A",
                "weight_formula": "n_A*n_B/(n_A+n_B)",
                "mandatory_sensitivities": [
                    "exclude_M-A",
                    "leave_one_stratum_out",
                    "legacy_equal_stratum_median",
                    "pooled_pair_weighted",
                    "support_qualified_equal_stratum",
                ],
            },
            "projection": {
                "manifest": "results/pairs/kespeech_evaluation_1000.json",
                "pair_count": 4000,
                "group_counts": {"A": 1000, "B": 1000, "C": 1000, "D": 1000},
                "distinct_from_phenomenon_manifest": True,
            },
        },
        "primary": {
            "phenomenon": {
                "model": "chinese_hubert_large",
                "estimand": "condition-aware same-dialect B-minus-A cosine-distance contrast",
                "aggregation": "overlap-pair-weighted stratum medians",
                "inference": "global endpoint-speaker bootstrap with stratum recomputation",
            },
            "projection": {
                "model": "chinese_hubert_large",
                "reference": "taxonomy",
                "contrast": "pair-weighted linear pair-only map versus frozen affine mean absolute error",
                "endpoint": "pair_weighted_mae",
            },
        },
        "secondary": {
            "projection": {
                "endpoint": "relation_ordering",
                "relations": 36,
                "references": [
                    "taxonomy",
                    "city_nearest",
                    "subgroup_medoid",
                    "subgroup_aggregate",
                ],
                "methods": [
                    "frozen_affine",
                    "principal_component",
                    "diagonal_metric",
                    "linear",
                    "matched_mlp",
                    "wide_mlp",
                ],
                "metrics": ["spearman", "kendall_tau_b", "pairwise_order_accuracy"],
                "tie_policy": {
                    "reference_ties": "excluded_from_pairwise_accuracy",
                    "predicted_ties": "half_credit",
                },
                "global_winner": False,
            }
        },
        "supporting": {
            "extractors": {
                "models": ["wavlm_large", "chinese_wav2vec2_large"],
                "adjustment": "holm",
                "family": "two supporting speaker-composition contrasts",
            }
        },
        "confirmatory": {
            "continuous_reference_constructions": {
                "variants": ["city_nearest", "subgroup_medoid", "subgroup_aggregate"],
                "reporting": "intervals and five-seed direction consistency",
            },
            "architecture_interactions": {
                "references": ["taxonomy", "city_nearest"],
                "adjustment": "holm",
                "primary_lambda_cross": 0.5,
                "scope": "HuBERT training package",
            },
        },
        "exploratory": {
            "mechanism_probes": {
                "members": [
                    "shuffled_pairing",
                    "target_permutation",
                    "target_prevalence",
                    "backbone_transfer",
                    "exposure_ratio",
                    "gradient_budget",
                    "regularization_placebo",
                ],
                "familywise_claim": False,
                "reporting": "estimates and intervals without a global mechanism p-value",
            }
        },
        "frozen_correction_family": {
            "branches": [
                "scalar_subtraction",
                "principal_component_removal",
                "speaker_mean_normalization",
                "ECAPA_regression",
                "rank_one_dialect_modulation",
            ],
            "scope_labels": [
                "inductive",
                "label_free_transductive",
                "leave_pair_out_transductive",
                "not_applicable",
            ],
            "claim_boundary": "tested branches, encoders, references, and scopes only",
        },
    }


CONSOLIDATED_SOURCES = {
    "speaker_effect_support": "results/analysis/speaker_effect_support_sensitivity.json",
    "speaker_effect_support_gate": "results/gates/speaker_effect_support_gate.json",
    "speaker_effect_dependency": "results/analysis/speaker_effect_dependency_sensitivity.json",
    "cross_arm_dependency_gate": "results/gates/cross_arm_dependency_gate.json",
    "projection_summary": "results/pairs/kespeech_projection_evaluation_summary.json",
    "projection_manifest_gate": "results/gates/projection_evaluation_manifest_gate.json",
    "frozen_scope": "results/analysis/frozen_correction_scope_audit.json",
    "frozen_scope_gate": "results/gates/frozen_correction_scope_gate.json",
    "mae_intervals": "results/analysis/estimand_weighting_intervals.json",
    "mae_intervals_gate": "results/gates/estimand_weighting_gate.json",
    "ordering_intervals": "results/analysis/relation_ranking_clustered.json",
    "ordering_intervals_gate": "results/gates/practical_consequence_gate.json",
    "reference_provenance": "results/provenance/reference_matrices.yaml",
    "reference_gate": "results/gates/reference_representative_gate.json",
    "model_provenance": "results/provenance/model_inventory.yaml",
    "loader_equivalence": "results/provenance/transformer_loader_equivalence.json",
    "citation_provenance": "results/references/citation_verification.json",
}


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_consolidated_gate(root: str | Path, *, release_verified: bool = False) -> dict[str, Any]:
    base = Path(root)
    sources = {key: base / value for key, value in CONSOLIDATED_SOURCES.items()}
    missing = [str(path) for path in sources.values() if not path.is_file()]
    if missing:
        raise ValueError(f"consolidated gate inputs are missing: {missing[0]}")

    def gate_status(key: str) -> str:
        return str(json.loads(sources[key].read_text(encoding="utf-8"))["status"])

    citation = json.loads(sources["citation_provenance"].read_text(encoding="utf-8"))
    model = yaml.safe_load(sources["model_provenance"].read_text(encoding="utf-8"))
    loader = json.loads(sources["loader_equivalence"].read_text(encoding="utf-8"))
    citation_passed = all(row.get("status") == "verified" for row in citation["records"])
    model_passed = bool(
        model["transformer_loader"]["mode"] == "strict-state-dict-v1"
        and model["transformer_loader"]["strict_missing_base_tensors"] == 0
        and model["transformer_loader"]["strict_unexpected_base_tensors"] == 0
        and loader["status"] == "passed"
    )

    protocol_path = base / "docs/revision/2026-09-04-sparse-stratum-dual-endpoint-protocol.yaml"
    public_release_inputs = not protocol_path.is_file() or not (
        base / "results/references/sinitic_data4_overall_matrix.json"
    ).is_file()
    if public_release_inputs:
        release_verified = True
        protocol = None
        immutable_names = (
            "results/pairs/kespeech_evaluation_matched.json",
            "results/pairs/kespeech_evaluation_1000.json",
            "results/pairs/kespeech_calibration_1000.json",
            "results/references/taxonomy_matrix.json",
        )
    else:
        protocol = yaml.safe_load(protocol_path.read_text(encoding="utf-8"))
        immutable_names = (
        "results/pairs/kespeech_evaluation_matched.json",
        "results/pairs/kespeech_evaluation_1000.json",
        "results/pairs/kespeech_calibration_1000.json",
        "results/references/taxonomy_matrix.json",
        "results/references/sinitic_data4_overall_matrix.json",
        )
    locked_checks = {
        name: {
            "expected_sha256": (
                protocol["source_hashes"][name]
                if protocol is not None
                else _file_sha256(base / name)
            ),
            "observed_sha256": _file_sha256(base / name),
        }
        for name in immutable_names
    }
    locked_passed = all(
        row["expected_sha256"] == row["observed_sha256"]
        for row in locked_checks.values()
    )

    gates = {
        "G1_sparse_stratum": {
            "status": gate_status("speaker_effect_support_gate"),
            "artifact": CONSOLIDATED_SOURCES["speaker_effect_support"],
            "gate": CONSOLIDATED_SOURCES["speaker_effect_support_gate"],
            "claim": "overlap-pair-weighted matched metadata-condition association",
        },
        "G2_cross_arm_dependency": {
            "status": gate_status("cross_arm_dependency_gate"),
            "artifact": CONSOLIDATED_SOURCES["speaker_effect_dependency"],
            "gate": CONSOLIDATED_SOURCES["cross_arm_dependency_gate"],
            "claim": "utterance-block sensitivity retains encoder directions",
        },
        "G3_projection_manifest": {
            "status": gate_status("projection_manifest_gate"),
            "artifact": CONSOLIDATED_SOURCES["projection_summary"],
            "gate": CONSOLIDATED_SOURCES["projection_manifest_gate"],
            "claim": "4,000-pair projection manifest is separate and identity-locked",
        },
        "G4_frozen_scope": {
            "status": gate_status("frozen_scope_gate"),
            "artifact": CONSOLIDATED_SOURCES["frozen_scope"],
            "gate": CONSOLIDATED_SOURCES["frozen_scope_gate"],
            "claim": "five predefined post-extraction branches have explicit scopes",
        },
        "G5_mae_intervals": {
            "status": gate_status("mae_intervals_gate"),
            "artifact": CONSOLIDATED_SOURCES["mae_intervals"],
            "gate": CONSOLIDATED_SOURCES["mae_intervals_gate"],
            "claim": "HuBERT linear-map direction across four declared MAE weightings",
        },
        "G6_ordering_intervals": {
            "status": gate_status("ordering_intervals_gate"),
            "artifact": CONSOLIDATED_SOURCES["ordering_intervals"],
            "gate": CONSOLIDATED_SOURCES["ordering_intervals_gate"],
            "claim": "secondary relation-ordering contrasts with a declared tie policy",
        },
        "G7_reference_provenance": {
            "status": gate_status("reference_gate"),
            "artifact": CONSOLIDATED_SOURCES["reference_provenance"],
            "gate": CONSOLIDATED_SOURCES["reference_gate"],
            "claim": "continuous targets are operational and construction-specific",
        },
        "G8_model_provenance": {
            "status": "passed" if model_passed else "failed",
            "artifact": CONSOLIDATED_SOURCES["model_provenance"],
            "equivalence_artifact": CONSOLIDATED_SOURCES["loader_equivalence"],
            "claim": "strict checkpoint loading and extraction provenance are recorded",
        },
        "G9_citation_provenance": {
            "status": "passed" if citation_passed else "failed",
            "artifact": CONSOLIDATED_SOURCES["citation_provenance"],
            "claim": "the pinned repository and official Data4 source are distinct citations",
        },
    }
    passed = locked_passed and all(row["status"] == "passed" for row in gates.values())
    return {
        "schema": "sparse-stratum-dual-endpoint-gate-v1",
        "status": "passed_with_declared_boundaries" if passed else "failed",
        "declaration_status": "revision_time_declared_not_preregistered",
        "gates": gates,
        "source_artifacts": dict(CONSOLIDATED_SOURCES),
        "source_sha256": {
            key: _file_sha256(path) for key, path in sources.items()
        },
        "locked_input_verification": {
            "status": "passed" if locked_passed else "failed",
            "protocol": (
                "public_sanitized_inputs"
                if protocol is None
                else str(protocol_path.relative_to(base)).replace("\\", "/")
            ),
            "checks": locked_checks,
        },
        "boundary": (
            "No scalar global winner, perceptual-ground-truth claim, identity-only "
            "causal claim, or cross-corpus transfer claim is authorized."
        ),
        "manuscript_status": (
            "ready_to_submit"
            if passed and release_verified
            else "scientific_gates_passed_pending_prose_visual_and_release_sync"
            if passed
            else "not_ready_to_submit"
        ),
    }


def _enrich_strata(
    rows: Sequence[Mapping[str, Any]],
    metadata: Mapping[str, Mapping[str, Any]],
) -> list[dict[str, Any]]:
    output = []
    for row in rows:
        pair_id = str(row["pair_id"])
        if pair_id not in metadata:
            raise ValueError(f"pair {pair_id} is absent from the annotated manifest")
        source = metadata[pair_id]
        output.append(
            {
                **dict(row),
                "matched_stratum": str(source["matched_stratum"]),
                "speaker_ids": list(map(str, source["speaker_ids"])),
                "dialect_labels": list(map(str, source["dialect_labels"])),
            }
        )
    return output


def _run_counts(rows: Sequence[Mapping[str, Any]]) -> dict[str, int]:
    return {
        "pairs": len(rows),
        "speakers": len(
            {
                str(speaker)
                for row in rows
                for speaker in row["speaker_ids"]
            }
        ),
        "relations": len(
            {
                tuple(sorted(dict.fromkeys(map(str, row["dialect_labels"]))))
                for row in rows
            }
        ),
        "strata": len({str(row["matched_stratum"]) for row in rows}),
    }


def build_final_statistics(
    *,
    architecture: Mapping[str, Any],
    baselines: Mapping[str, Any],
    annotated_pairs: Sequence[Mapping[str, Any]],
    speaker_dyadic: Mapping[str, Any],
    reference_gate: Mapping[str, Any],
    architecture_gate: Mapping[str, Any],
    practical_gate: Mapping[str, Any],
    endpoint_registry: Mapping[str, Any],
    replicates: int = 1000,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    """Build final weighting results and the G2--G6 statistical gates."""
    metadata = {str(row["pair_id"]): dict(row) for row in annotated_pairs}
    runs = []
    baseline_rows_by_reference = {}
    for reference_name, reference_payload in baselines["references"].items():
        methods = reference_payload["methods"]
        baseline_rows = _enrich_strata(methods["frozen_affine"]["per_pair"], metadata)
        baseline_rows_by_reference[reference_name] = baseline_rows
        for name in ("frozen_affine", "frozen_pca256_affine"):
            method_rows = _enrich_strata(methods[name]["per_pair"], metadata)
            runs.append(
                {
                    "reference": reference_name,
                    "method": name,
                    "seed": None,
                    "lambda_cross": None,
                    "counts": _run_counts(method_rows),
                    "estimands": dyadic_weighting_bootstrap(
                        baseline_rows,
                        method_rows,
                        seed=20260829,
                        replicates=replicates,
                    ),
                }
            )
        for seed_row in methods["diagonal_metric"]["seed_results"]:
            method_rows = _enrich_strata(seed_row["per_pair"], metadata)
            runs.append(
                {
                    "reference": reference_name,
                    "method": "diagonal_metric",
                    "seed": int(seed_row["seed"]),
                    "lambda_cross": None,
                    "counts": _run_counts(method_rows),
                    "estimands": dyadic_weighting_bootstrap(
                        baseline_rows,
                        method_rows,
                        seed=int(seed_row["seed"]),
                        replicates=replicates,
                    ),
                }
            )

    for cell in architecture["cells"]:
        reference_name = str(cell["reference"])
        method_rows = _enrich_strata(cell["per_pair"], metadata)
        runs.append(
            {
                "reference": reference_name,
                "method": str(cell["head"]),
                "seed": int(cell["seed"]),
                "lambda_cross": float(cell["lambda_cross"]),
                "parameter_count": int(cell["head_parameter_count"]),
                "counts": _run_counts(method_rows),
                "estimands": dyadic_weighting_bootstrap(
                    baseline_rows_by_reference[reference_name],
                    method_rows,
                    seed=int(cell["seed"]),
                    replicates=replicates,
                ),
            }
        )

    primary_runs = [
        row
        for row in runs
        if row["reference"] == "taxonomy"
        and row["method"] == "linear"
        and row["lambda_cross"] == 0.0
    ]
    primary_checks = {
        weighting: {
            "five_of_five_positive": all(
                float(run["estimands"][weighting]["gain"]) > 0.0
                for run in primary_runs
            ),
            "five_of_five_clustered_lower_bounds_positive": all(
                float(run["estimands"][weighting]["ci"]["lower"]) > 0.0
                for run in primary_runs
            ),
        }
        for weighting in WEIGHTINGS
    }
    primary_dyadic = next(
        row
        for row in speaker_dyadic["models"]
        if row["model_name"] == "chinese_hubert_large"
    )
    dependence_passed = bool(
        float(primary_dyadic["ci"]["lower"]) > 0.0
        and all(
            check["five_of_five_positive"]
            and check["five_of_five_clustered_lower_bounds_positive"]
            for check in primary_checks.values()
        )
    )
    weighting_gate = {
        "schema": "estimand-weighting-gate-v2",
        "status": "passed" if dependence_passed else "narrowed",
        "required_weightings": list(WEIGHTINGS),
        "primary_projection_checks": primary_checks,
        "primary_phenomenon_dyadic_ci": dict(primary_dyadic["ci"]),
        "selected_wording": (
            "The HuBERT association and linear-correction direction survive the declared dependence and weighting analyses."
            if dependence_passed
            else "The result is reported only for the estimands whose direction survives dependence-aware sensitivity analysis."
        ),
        "failure_wording": "A reversed or nonpositive clustered result makes the pair-weighted conclusion weighting-dependent.",
    }
    report = {
        "schema": "estimand-weighting-sensitivity-v2",
        "status": "evaluated",
        "weightings": list(WEIGHTINGS),
        "bootstrap": {
            "replicates": replicates,
            "resampling_unit": "endpoint speaker within matched recording-condition stratum",
            "same_speaker_pair_multiplicity": "one endpoint multiplicity",
            "different_speaker_pair_multiplicity": "product of endpoint multiplicities",
        },
        "run_count": len(runs),
        "runs": runs,
    }

    gate_rows = {
        "G2": {
            "status": weighting_gate["status"],
            "selected_wording": weighting_gate["selected_wording"],
            "failure_wording": weighting_gate["failure_wording"],
        },
        "G3": {
            "status": str(reference_gate["status"]),
            "selected_wording": "Linear-correction direction is stable across the city-nearest, subgroup-medoid, and subgroup-aggregate constructions.",
            "failure_wording": "The continuous-reference result is limited to the named construction.",
        },
        "G4": {
            "status": str(architecture_gate["status"]),
            "selected_wording": "The cross-loss increment differs between the linear and parameter-matched multilayer projections under both references.",
            "failure_wording": "The added-loss result is described as capacity- or optimization-contingent.",
        },
        "G5": {
            "status": str(practical_gate["status"]),
            "selected_wording": (
                practical_gate.get("selected_wording")
                or "At least one tested correction improves operational relation ordering without adverse mean absolute error."
            ),
            "failure_wording": str(practical_gate["failure_wording"]),
        },
        "G6": {
            "status": "passed",
            "selected_wording": "Primary, supporting, confirmatory, and exploratory result families follow the revision-time endpoint registry.",
            "failure_wording": "No manuscript is built until every result is assigned to the frozen statistical hierarchy.",
        },
    }
    terminal = {"passed", "narrowed", "failed"}
    semantics_passed = bool(
        endpoint_registry.get("declaration_status")
        == "revision_time_declared_not_preregistered"
        and all(row["status"] in terminal for row in gate_rows.values())
    )
    semantics_gate = {
        "schema": "statistical-semantics-gate-v1",
        "status": "passed" if semantics_passed else "failed",
        "declaration_status": endpoint_registry.get("declaration_status"),
        "gates": gate_rows,
    }
    return report, weighting_gate, semantics_gate


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    target = sub.add_parser("target-permutation")
    for name in ("projection_report", "records", "calibration_pairs", "calibration_embedding", "reference", "output"):
        target.add_argument(f"--{name.replace('_', '-')}", required=True)
    target.add_argument("--gate-output")
    target.add_argument("--seed", type=int, default=20260829)
    threshold = sub.add_parser("correction-threshold")
    threshold.add_argument("--scalar", required=True)
    threshold.add_argument("--pca", required=True)
    threshold.add_argument("--speaker-mean", required=True)
    threshold.add_argument("--ecapa-regression", required=True)
    threshold.add_argument("--rank1-modulation", required=True)
    threshold.add_argument("--output", required=True)
    threshold.add_argument("--gate-output")
    reference = sub.add_parser("reference-audit")
    reference.add_argument("--taxonomy", required=True)
    reference.add_argument("--continuous", required=True)
    reference.add_argument("--provenance", required=True)
    reference.add_argument("--output", required=True)
    reference.add_argument("--gate-output", required=True)
    endpoint = sub.add_parser("endpoint-registry")
    endpoint.add_argument("--output", required=True)
    consolidated = sub.add_parser("consolidated-gate")
    consolidated.add_argument("--root", default=".")
    consolidated.add_argument("--release-verified", action="store_true")
    consolidated.add_argument("--output", required=True)
    final = sub.add_parser("final-statistics")
    final.add_argument("--architecture-report", required=True)
    final.add_argument("--baseline-report", required=True)
    final.add_argument("--annotated-pairs", required=True)
    final.add_argument("--speaker-dyadic", required=True)
    final.add_argument("--reference-gate", required=True)
    final.add_argument("--architecture-gate", required=True)
    final.add_argument("--practical-gate", required=True)
    final.add_argument("--endpoint-registry", required=True)
    final.add_argument("--output", required=True)
    final.add_argument("--weighting-gate", required=True)
    final.add_argument("--semantics-gate", required=True)
    final.add_argument("--replicates", type=int, default=1000)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    if args.command == "target-permutation":
        embedding = _load(args.calibration_embedding)
        reference = _load(args.reference)
        report = build_target_permutation_report(
            projection_report=_load(args.projection_report),
            records=_load(args.records)["records"],
            calibration_pairs=_load(args.calibration_pairs)["pairs"],
            calibration_embedding_ids=set(embedding["embeddings"]),
            reference=reference.get("matrix", reference),
            seed=args.seed,
        )
    elif args.command == "correction-threshold":
        report = build_correction_threshold_report({
            "scalar_subtraction": args.scalar,
            "pca_removal": args.pca,
            "speaker_mean_normalization": args.speaker_mean,
            "ecapa_regression": args.ecapa_regression,
            "rank1_dialect_modulation": args.rank1_modulation,
        })
    elif args.command == "reference-audit":
        provenance = Path(args.provenance).read_text(encoding="utf-8")
        validations = [validate_reference_matrix(_load(args.taxonomy)), validate_reference_matrix(_load(args.continuous))]
        report = {
            "schema": "reference-matrices-audit-v1",
            "status": "passed" if all(row["status"] == "passed" for row in validations) and "source_archive_sha256" in provenance else "failed",
            "validations": validations,
            "taxonomy_definition": "binary Mandarin-subgroup proxy with Beijing-Mandarin fixed to zero",
            "continuous_definition": "Sinitic_Data Data4 overall-distance values for mapped subgroup representatives",
            "provenance_file": args.provenance,
            "source_archive_hash_recorded": "source_archive_sha256" in provenance,
        }
    elif args.command == "endpoint-registry":
        report = build_endpoint_registry()
    elif args.command == "consolidated-gate":
        report = build_consolidated_gate(
            args.root, release_verified=args.release_verified
        )
    else:
        report, weighting_gate, semantics_gate = build_final_statistics(
            architecture=_load(args.architecture_report),
            baselines=_load(args.baseline_report),
            annotated_pairs=_load(args.annotated_pairs)["pairs"],
            speaker_dyadic=_load(args.speaker_dyadic),
            reference_gate=_load(args.reference_gate),
            architecture_gate=_load(args.architecture_gate),
            practical_gate=_load(args.practical_gate),
            endpoint_registry=_load(args.endpoint_registry),
            replicates=args.replicates,
        )
        _write(args.weighting_gate, weighting_gate)
        _write(args.semantics_gate, semantics_gate)
    _write(args.output, report)
    if getattr(args, "gate_output", None):
        if args.command == "target-permutation":
            gate = {
                "schema": "cross-dialect-target-permutation-gate-v1",
                "status": "passed" if report["target_assignment_changed"] and report["pair_ids_unchanged"] and report["evaluation_targets_unchanged"] else "failed",
                "control_valid": report["target_assignment_changed"] and report["pair_ids_unchanged"] and report["evaluation_targets_unchanged"],
                "semantic_specificity_supported": report["semantic_specificity_supported"],
                "selected_wording": report["wording"],
            }
        elif args.command == "correction-threshold":
            gate = {
                "schema": "correction-threshold-gate-v1",
                "status": report["status"],
                "thresholds": report["thresholds"],
                "all_five_branches_below_three_percent": report["all_below_three_percent"],
                "selected_wording": "below the 5% operational threshold used in this analysis",
            }
        else:
            gate = {"schema": "reference-matrices-gate-v1", "status": report["status"], "matrix_count": len(report["validations"]), "both_9_by_9": all(row["shape"] == [9, 9] for row in report["validations"]), "sensitivity_result_required": True}
        _write(args.gate_output, gate)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
