"""Classify evaluation information used by frozen correction branches."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable, Mapping

from .correction_gate import scalar_subtraction_scope_contract
from .speaker_mean_normalization_gate import speaker_mean_scope_contracts
from .speaker_regression_gate import speaker_regression_scope_contracts


SCOPE_TO_CLASS = {
    "current_pair_only": "inductive",
    "evaluation_speaker_pool": "label_free_transductive",
    "leave_pair_out_pool": "leave_pair_out_transductive",
}


def build_scope_contract(
    *, fit_scope: str, evaluation_feature_scope: str, fallback_count: int
) -> dict[str, Any]:
    if evaluation_feature_scope not in SCOPE_TO_CLASS:
        raise ValueError("unknown evaluation feature scope")
    if fallback_count < 0:
        raise ValueError("fallback_count cannot be negative")
    return {
        "fit_scope": fit_scope,
        "evaluation_feature_scope": evaluation_feature_scope,
        "inference_class": SCOPE_TO_CLASS[evaluation_feature_scope],
        "fallback_count": int(fallback_count),
    }


def validate_scope_contract(row: Mapping[str, Any]) -> None:
    required = {
        "fit_scope",
        "evaluation_feature_scope",
        "inference_class",
        "fallback_count",
    }
    missing = required - set(row)
    if missing:
        raise ValueError(f"scope contract missing {sorted(missing)[0]}")
    feature_scope = str(row["evaluation_feature_scope"])
    expected = SCOPE_TO_CLASS.get(feature_scope)
    if expected is None:
        raise ValueError("unknown evaluation feature scope")
    if row["inference_class"] != expected:
        if feature_scope != "current_pair_only":
            raise ValueError("evaluation-pool statistics must be labelled transductive")
        raise ValueError("scope and inference class disagree")
    if int(row["fallback_count"]) < 0:
        raise ValueError("fallback_count cannot be negative")


def strict_inductive_row(
    *,
    method: str,
    enrollment_available: bool,
    improvement_ratio: float | None = None,
) -> dict[str, Any]:
    if not enrollment_available and improvement_ratio is not None:
        raise ValueError("strict inductive row cannot carry a score without enrollment")
    row = {
        "method": method,
        **build_scope_contract(
            fit_scope="calibration_speakers",
            evaluation_feature_scope="current_pair_only",
            fallback_count=0,
        ),
        "status": "evaluated" if enrollment_available else "not_applicable",
        "improvement_ratio": improvement_ratio if enrollment_available else None,
        "reason": (
            "evaluation-speaker enrollment was supplied"
            if enrollment_available
            else "the correction requires a speaker aggregate unavailable from the current pair alone"
        ),
    }
    return row


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _reference_rows(payload: Mapping[str, Any]) -> Iterable[tuple[str, Mapping[str, Any]]]:
    for model in payload.get("models", []):
        for reference in model.get("references", []):
            yield str(model["model_name"]), reference


def _base_row(
    method: str,
    model_name: str,
    reference: Mapping[str, Any],
    contract: Mapping[str, Any],
) -> dict[str, Any]:
    row = {
        "method": method,
        "model_name": model_name,
        "reference_name": str(reference["reference_name"]),
        **dict(contract),
    }
    validate_scope_contract(row)
    return row


def _current_pair_rows(
    method: str,
    payload: Mapping[str, Any],
    *,
    evaluation_statistic: str,
) -> list[dict[str, Any]]:
    contract = (
        scalar_subtraction_scope_contract()
        if method == "scalar_subtraction"
        else {
            **build_scope_contract(
                fit_scope="calibration_speakers",
                evaluation_feature_scope="current_pair_only",
                fallback_count=0,
            ),
            "evaluation_statistic": evaluation_statistic,
        }
    )
    rows = []
    for model_name, reference in _reference_rows(payload):
        rows.append(
            {
                **_base_row(method, model_name, reference, contract),
                "status": "evaluated",
                "improvement_ratio": float(reference["improvement_ratio"]),
                "ci": dict(reference.get("ci", {})),
                "evaluation_contributors": "two current pair endpoints",
                "evaluation_pool_uses_dialect_or_reference_labels": False,
            }
        )
    return rows


def _speaker_pool_rows(
    method: str,
    payload: Mapping[str, Any],
    *,
    evaluation_utterance_count: int,
    uses_dialect_at_application: bool,
) -> list[dict[str, Any]]:
    declared = (
        speaker_mean_scope_contracts()
        if method == "speaker_mean_normalization"
        else speaker_regression_scope_contracts()
    )
    if len(declared) != 3:
        raise ValueError("speaker-pool correction must declare three scope rows")
    rows = []
    for model_name, reference in _reference_rows(payload):
        full_contract = build_scope_contract(
            fit_scope="calibration_speakers",
            evaluation_feature_scope="evaluation_speaker_pool",
            fallback_count=int(
                reference.get("normalization", {})
                .get("evaluation_summary", {})
                .get("fallback_speaker_count", 0)
            ),
        )
        rows.append(
            {
                **_base_row(method, model_name, reference, full_contract),
                "scope_variant": "full_evaluation_speaker_pool",
                "status": "evaluated",
                "improvement_ratio": float(reference["improvement_ratio"]),
                "ci": dict(reference.get("ci", {})),
                "evaluation_contributors": (
                    f"all {evaluation_utterance_count} evaluation utterances, grouped by speaker"
                ),
                "current_pair_excluded_from_aggregate": False,
                "evaluation_pool_uses_dialect_or_reference_labels": False,
                "correction_application_uses_dialect_label": uses_dialect_at_application,
            }
        )
        leave_pair_out = reference["leave_pair_out"]
        lpo_contract = build_scope_contract(
            fit_scope="calibration_speakers",
            evaluation_feature_scope="leave_pair_out_pool",
            fallback_count=int(leave_pair_out.get("fallback_pair_count", 0)),
        )
        rows.append(
            {
                **_base_row(method, model_name, reference, lpo_contract),
                "scope_variant": "leave_current_pair_out",
                "status": "evaluated",
                "improvement_ratio": float(leave_pair_out["improvement_ratio"]),
                "ci": None,
                "fallback_unit": "pair",
                "evaluation_contributors": (
                    "all other available evaluation utterances for each endpoint speaker"
                ),
                "current_pair_excluded_from_aggregate": True,
                "evaluation_pool_uses_dialect_or_reference_labels": False,
                "correction_application_uses_dialect_label": uses_dialect_at_application,
            }
        )
        strict = strict_inductive_row(method=method, enrollment_available=False)
        rows.append(
            {
                **strict,
                "model_name": model_name,
                "reference_name": str(reference["reference_name"]),
                "scope_variant": "strict_inductive_without_enrollment",
                "evaluation_contributors": "current pair only",
                "correction_application_uses_dialect_label": uses_dialect_at_application,
            }
        )
    return rows


def build_report(paths: Mapping[str, Path]) -> tuple[dict[str, Any], dict[str, Any]]:
    payloads = {
        name: json.loads(path.read_text(encoding="utf-8"))
        for name, path in paths.items()
    }
    speaker_mean_payload = payloads["speaker_mean_normalization"]
    first_reference = next(_reference_rows(speaker_mean_payload))[1]
    evaluation_count = int(
        first_reference["normalization"]["evaluation_summary"]["utterance_count"]
    )
    rows = []
    rows.extend(
        _current_pair_rows(
            "scalar_subtraction",
            payloads["scalar_subtraction"],
            evaluation_statistic="current pair endpoint speaker-embedding distance",
        )
    )
    rows.extend(
        _current_pair_rows(
            "pca_removal",
            payloads["pca_removal"],
            evaluation_statistic="calibration-fitted component removal applied to current pair embeddings",
        )
    )
    rows.extend(
        _speaker_pool_rows(
            "speaker_mean_normalization",
            speaker_mean_payload,
            evaluation_utterance_count=evaluation_count,
            uses_dialect_at_application=False,
        )
    )
    rows.extend(
        _speaker_pool_rows(
            "ecapa_regression",
            payloads["ecapa_regression"],
            evaluation_utterance_count=evaluation_count,
            uses_dialect_at_application=True,
        )
    )
    rows.extend(
        _speaker_pool_rows(
            "rank1_dialect_modulation",
            payloads["rank1_dialect_modulation"],
            evaluation_utterance_count=evaluation_count,
            uses_dialect_at_application=True,
        )
    )
    for row in rows:
        validate_scope_contract(row)
        if row["status"] == "not_applicable" and row["improvement_ratio"] is not None:
            raise ValueError("not-applicable strict inductive row carries a score")

    evaluated = [row for row in rows if row["status"] == "evaluated"]
    methods = sorted({str(row["method"]) for row in rows})
    report = {
        "schema": "frozen-correction-scope-audit-v1",
        "status": "evaluated",
        "family_label": "predefined post-extraction branches",
        "family_scope": {
            "branch_count": 5,
            "encoder_count": 3,
            "operational_reference_count": 2,
        },
        "source_files": {
            name: {"path": str(path), "sha256": _sha256(path)}
            for name, path in paths.items()
        },
        "rows": rows,
        "method_summary": {
            method: {
                "evaluated_scope_rows": sum(
                    row["method"] == method and row["status"] == "evaluated"
                    for row in rows
                ),
                "not_applicable_scope_rows": sum(
                    row["method"] == method and row["status"] == "not_applicable"
                    for row in rows
                ),
                "observed_gain_min": min(
                    float(row["improvement_ratio"])
                    for row in evaluated
                    if row["method"] == method
                ),
                "observed_gain_max": max(
                    float(row["improvement_ratio"])
                    for row in evaluated
                    if row["method"] == method
                ),
            }
            for method in methods
        },
        "boundary": (
            "Results apply only to the five tested branches, three encoders, two "
            "operational references, and the explicitly named inference scopes."
        ),
    }
    gate_pass = (
        len(methods) == 5
        and all(
            any(row["method"] == method and row["status"] == "evaluated" for row in rows)
            for method in methods
        )
        and all(
            row["improvement_ratio"] is None
            for row in rows
            if row["status"] == "not_applicable"
        )
    )
    gate = {
        "schema": "frozen-correction-scope-gate-v1",
        "status": "passed" if gate_pass else "failed",
        "method_count": len(methods),
        "all_rows_scope_validated": gate_pass,
        "strict_inductive_scores_recycled": False,
        "table_family_wording": "predefined post-extraction branches",
    }
    return report, gate


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scalar", type=Path, required=True)
    parser.add_argument("--pca", type=Path, required=True)
    parser.add_argument("--speaker-mean", type=Path, required=True)
    parser.add_argument("--ecapa", type=Path, required=True)
    parser.add_argument("--rank-one", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--gate-output", type=Path, required=True)
    args = parser.parse_args()
    report, gate = build_report(
        {
            "scalar_subtraction": args.scalar,
            "pca_removal": args.pca,
            "speaker_mean_normalization": args.speaker_mean,
            "ecapa_regression": args.ecapa,
            "rank1_dialect_modulation": args.rank_one,
        }
    )
    for path, payload in ((args.output, report), (args.gate_output, gate)):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
