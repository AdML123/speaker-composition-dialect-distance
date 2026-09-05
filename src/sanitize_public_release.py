"""Remove fields that would reconstruct an unlicensed continuous reference."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any


REDACTION = {
    "schema": "public-release-redaction-v1",
    "reason": "upstream_license_not_specified",
    "scope": "fields_that_would_reconstruct_the_continuous_reference",
    "reconstruction_route": "obtain_pinned_upstream_commit_and_rebuild_locally",
}

_LOCAL_PATH_REPLACEMENTS = (
    ("C:" + "/" + "Users/yuefe", "LOCAL_USER_PATH_REDACTED"),
    ("C:" + "\\" + "Users\\yuefe", "LOCAL_USER_PATH_REDACTED"),
    ("D:" + "/" + "paper48", "WORKSPACE_PATH_REDACTED"),
    ("D:" + "\\" + "paper48", "WORKSPACE_PATH_REDACTED"),
    ("E:" + "/" + "paper48-datasets", "DATASET_PATH_REDACTED"),
    ("E:" + "\\" + "paper48-datasets", "DATASET_PATH_REDACTED"),
)

_ANALYSIS_KEEP = {
    "binary_zero_rule_sensitivity.json",
    "calibration_leakage_audit.json",
    "architecture_cross_loss_factorial.json",
    "correction_threshold_sensitivity.json",
    "cross_dialect_gradient_norm_log.json",
    "cross_dialect_target_permutation.json",
    "embedding_diagnostics.json",
    "estimand_weighting_intervals.json",
    "estimand_weighting_sensitivity.json",
    "frozen_correction_scope_audit.json",
    "loss_structure_placebo.json",
    "metric_baseline_and_ranking.json",
    "pool_ratio_gradient_budget.json",
    "reference_matrices_audit.json",
    "reference_representative_sensitivity.json",
    "reference_sensitivity_binary_linear_seed_distribution.json",
    "reference_sensitivity_clustered.json",
    "reference_sensitivity_continuous_linear_seed_distribution.json",
    "reference_sensitivity_continuous_seed_distribution.json",
    "relation_ranking_clustered.json",
    "revision_endpoint_registry.json",
    "speaker_dialect_variance_components.json",
    "speaker_effect_clustered.json",
    "speaker_effect_dependency_sensitivity.json",
    "speaker_effect_support_sensitivity.json",
    "speaker_support_sensitivity.json",
    "target_prevalence_mechanism.json",
    "target_prevalence_process_traces.json",
    "target_prevalence_rule_transfer.json",
    "title_branch.json",
}

_GATE_KEEP = {
    "binary_zero_rule_sensitivity_gate.json",
    "calibration_manifest_role_gate.json",
    "architecture_factorial_gate.json",
    "correction_threshold_gate.json",
    "cross_arm_dependency_gate.json",
    "embedding_diagnostics_gate.json",
    "estimand_weighting_gate.json",
    "frozen_correction_scope_gate.json",
    "gradient_budget_mechanism_gate.json",
    "matched_design_disclosure_gate.json",
    "practical_consequence_gate.json",
    "projection_evaluation_manifest_gate.json",
    "reference_matrices_gate.json",
    "reference_representative_gate.json",
    "reference_sensitivity_binary_linear_seed_gate.json",
    "reference_sensitivity_continuous_linear_seed_gate.json",
    "reference_sensitivity_continuous_seed_gate.json",
    "reference_sensitivity_gate.json",
    "sparse_stratum_dual_endpoint_gate.json",
    "speaker_dialect_variance_gate.json",
    "speaker_effect_clustered_gate.json",
    "speaker_effect_support_gate.json",
    "speaker_mean_normalization_gate.json",
    "sign_flip_finite_strata_gate.json",
    "statistical_semantics_gate.json",
    "target_prevalence_mechanism_gate.json",
    "target_prevalence_rule_transfer_gate.json",
}

_PAIR_KEEP = {
    "calibration_manifest_roles.json",
    "kespeech_calibration_1000.json",
    "kespeech_calibration_matched.json",
    "kespeech_calibration_matched_exact_content.json",
    "kespeech_evaluation_1000.json",
    "kespeech_evaluation_1000_annotated.json",
    "kespeech_evaluation_1000_audit.json",
    "kespeech_evaluation_1000_design_strata.json",
    "kespeech_evaluation_matched.json",
    "kespeech_evaluation_matched_audit.json",
    "kespeech_evaluation_matched_exact_content.json",
    "kespeech_matched_strata_summary.json",
    "kespeech_projection_evaluation_summary.json",
}


def _without_key(value: Any, key: str) -> Any:
    if isinstance(value, dict):
        return {
            item_key: _without_key(item_value, key)
            for item_key, item_value in value.items()
            if item_key != key
        }
    if isinstance(value, list):
        return [_without_key(item, key) for item in value]
    return value


def _without_public_payloads(value: Any) -> Any:
    """Keep aggregate evidence while removing pair-level or trace-level payloads."""
    if isinstance(value, dict):
        output = {}
        for item_key, item_value in value.items():
            lowered = item_key.lower()
            if lowered in {
                "per_pair", "baseline_per_pair", "bootstrap_values", "trace",
                "traces", "process_traces", "gradient_trace", "pair_rows",
            }:
                continue
            output[item_key] = _without_public_payloads(item_value)
        return output
    if isinstance(value, list):
        return [_without_public_payloads(item) for item in value]
    return value


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _redact_local_paths(root: Path) -> int:
    """Replace known machine-local roots in text artifacts before publication."""
    changed = 0
    for path in root.rglob("*"):
        if not path.is_file() or path.name == "MANIFEST.sha256":
            continue
        if any(part in {"__pycache__", ".pytest_cache"} for part in path.parts):
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        redacted = text
        for source, replacement in _LOCAL_PATH_REPLACEMENTS:
            redacted = redacted.replace(source, replacement)
        if redacted != text:
            path.write_text(redacted, encoding="utf-8")
            changed += 1
    return changed


def _prune_nonrelease_reports(root: Path) -> int:
    removed = 0
    for directory, keep in (
        (root / "results" / "analysis", _ANALYSIS_KEEP),
        (root / "results" / "gates", _GATE_KEEP),
        (root / "results" / "pairs", _PAIR_KEEP),
    ):
        if not directory.is_dir():
            continue
        for path in directory.glob("*.json"):
            if path.name not in keep:
                path.unlink()
                removed += 1
    analysis = root / "results" / "analysis"
    for name in _ANALYSIS_KEEP:
        path = analysis / name
        if not path.is_file():
            continue
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        sanitized = _without_public_payloads(payload)
        _write(path, sanitized)
    return removed


def sanitize_release_reports(release_root: str | Path) -> dict[str, int]:
    root = Path(release_root)
    analysis = root / "results" / "analysis"

    architecture_path = analysis / "architecture_cross_loss_factorial.json"
    architecture = _without_key(_load(architecture_path), "per_pair")
    architecture["public_release_redaction"] = {
        **REDACTION,
        "removed": ["cells[*].per_pair"],
    }
    _write(architecture_path, architecture)

    ranking_path = analysis / "metric_baseline_and_ranking.json"
    ranking = _without_key(_load(ranking_path), "per_pair")
    ranking["public_release_redaction"] = {
        **REDACTION,
        "removed": ["references.*.methods.*.per_pair", "seed_results[*].per_pair"],
    }
    _write(ranking_path, ranking)

    sensitivity_path = analysis / "reference_sensitivity_clustered.json"
    sensitivity = _load(sensitivity_path)
    continuous = sensitivity["references"]["continuous_sinitic"]["target_distribution"]
    continuous.pop("target_histogram", None)
    sensitivity["public_release_redaction"] = {
        **REDACTION,
        "removed": ["references.continuous_sinitic.target_distribution.target_histogram"],
    }
    _write(sensitivity_path, sensitivity)

    model_path = root / "results" / "provenance" / "model_inventory.yaml"
    if model_path.is_file():
        import yaml

        model_inventory = yaml.safe_load(model_path.read_text(encoding="utf-8"))
        model_inventory = _without_key(model_inventory, "cache_root")
        temporary = model_path.with_name(f".{model_path.name}.tmp")
        temporary.write_text(
            yaml.safe_dump(model_inventory, sort_keys=False, allow_unicode=True),
            encoding="utf-8",
        )
        temporary.replace(model_path)

    reference_path = root / "results" / "provenance" / "reference_matrices.yaml"
    if reference_path.is_file():
        import yaml

        reference_payload = yaml.safe_load(reference_path.read_text(encoding="utf-8"))
        sincomp = reference_payload.get("sincomp", {})
        sincomp.pop("matrix", None)
        for mapping in sincomp.get("mappings", {}).values():
            mapping.pop("private_artifact", None)
        sincomp["redistribution"] = {
            "status": "not_redistributed",
            "reason": "upstream_license_not_specified",
            "public_route": "obtain_pinned_upstream_commit_and_rebuild_locally",
        }
        temporary = reference_path.with_name(f".{reference_path.name}.tmp")
        temporary.write_text(
            yaml.safe_dump(reference_payload, sort_keys=False, allow_unicode=True),
            encoding="utf-8",
        )
        temporary.replace(reference_path)

    return {
        "architecture_cells": len(architecture["cells"]),
        "ranking_references": len(ranking["references"]),
        "sensitivity_references": len(sensitivity["references"]),
        "model_cache_paths_removed": int(model_path.is_file()),
    }


def validate_public_tree(release_root: str | Path) -> dict[str, int]:
    """Fail closed if a staged tree contains manuscript, secret, or data payloads."""
    root = Path(release_root)
    forbidden_suffixes = {
        ".tex", ".aux", ".bbl", ".blg", ".log", ".pdf", ".npz", ".npy",
        ".pkl", ".wav", ".flac", ".zip", ".tar", ".gz", ".tgz", ".pt", ".pth",
    }
    forbidden_names = {"key" + ".txt", "access" + "token.txt"}
    forbidden_tokens = (
        "c:" + "/" + "users" + "/", "d:" + "/" + "paper48",
        "e:" + "/" + "paper48", "access" + "token.txt",
        "key" + ".txt", "-----" + "begin", "github" + "_pat_",
        "bearer" + " ",
    )
    files = 0
    for path in root.rglob("*"):
        if not path.is_file() or path.name == "MANIFEST.sha256":
            continue
        if any(part in {"__pycache__", ".pytest_cache"} for part in path.parts):
            raise ValueError(f"cache directory in public tree: {path}")
        if path.suffix.lower() in forbidden_suffixes or path.name.lower() in forbidden_names:
            raise ValueError(f"forbidden public artifact: {path}")
        files += 1
        try:
            content = path.read_text(encoding="utf-8").lower()
        except UnicodeDecodeError:
            continue
        if any(token in content for token in forbidden_tokens):
            raise ValueError(f"forbidden path or secret token in: {path}")
        if re.search(r"sinitic_data4_(?:overall|city_nearest|subgroup_|matrix)", path.name.lower()):
            raise ValueError(f"derived continuous matrix payload: {path}")
    return {"files_checked": files}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--release-root", required=True)
    args = parser.parse_args()
    root = Path(args.release_root)
    report = {"nonrelease_reports_removed": _prune_nonrelease_reports(root)}
    report.update(sanitize_release_reports(args.release_root))
    report["local_path_files_redacted"] = _redact_local_paths(root)
    report.update(validate_public_tree(args.release_root))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
