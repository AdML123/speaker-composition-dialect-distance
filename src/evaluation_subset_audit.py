"""Audit the separate 4,000-pair projection evaluation manifest."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping

import yaml

from .pair_sampling import _canonical_record_hash


GROUPS = ("A", "B", "C", "D")
HASH_KEYS = {
    "evaluation_pairs",
    "pair_manifest_sha256",
    "pair_manifest_hash",
    "projection_manifest",
}
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _utterances(pair: Mapping[str, Any]) -> list[str]:
    values = pair.get("source_utterance_ids", pair.get("utterance_ids"))
    if not isinstance(values, list) or len(values) != 2:
        raise ValueError("every pair must expose two source utterance IDs")
    return list(map(str, values))


def _speakers(pairs: Iterable[Mapping[str, Any]]) -> set[str]:
    return {
        str(speaker)
        for pair in pairs
        for speaker in pair.get("speaker_ids", [])
    }


def detect_pair_id_collisions(
    projection_pairs: Iterable[Mapping[str, Any]],
    phenomenon_pairs: Iterable[Mapping[str, Any]],
) -> dict[str, Any]:
    """Detect local IDs that refer to different endpoint identities."""
    projection = {str(row["pair_id"]): tuple(sorted(_utterances(row))) for row in projection_pairs}
    phenomenon = {str(row["pair_id"]): tuple(sorted(_utterances(row))) for row in phenomenon_pairs}
    shared = sorted(set(projection).intersection(phenomenon))
    different = [pair_id for pair_id in shared if projection[pair_id] != phenomenon[pair_id]]
    return {
        "shared_pair_id_count": len(shared),
        "different_utterance_identity_count": len(different),
        "different_utterance_identity_pair_id_sha256": hashlib.sha256(
            "\n".join(different).encode("utf-8")
        ).hexdigest(),
        "safe_to_join_on_pair_id_alone": not different,
        "required_join_identity": "pair_id plus sorted source_utterance_ids",
    }


def audit_annotated_projection_manifest(
    base: Mapping[str, Any], annotated: Mapping[str, Any]
) -> dict[str, Any]:
    """Verify endpoint identity and symmetric relation coverage in annotations."""
    base_rows = {str(row["pair_id"]): dict(row) for row in base.get("pairs", [])}
    annotated_rows = {
        str(row["pair_id"]): dict(row) for row in annotated.get("pairs", [])
    }
    mismatched = []
    invalid_strata = []
    for pair_id, base_row in base_rows.items():
        candidate = annotated_rows.get(pair_id)
        if candidate is None or tuple(sorted(_utterances(base_row))) != tuple(
            sorted(_utterances(candidate))
        ):
            mismatched.append(pair_id)
            continue
        stratum = str(candidate.get("matched_stratum", ""))
        required_tokens = {
            *map(str, base_row.get("dialect_labels", [])),
            *map(str, base_row.get("recording_conditions", [])),
        }
        if not stratum or any(token not in stratum for token in required_tokens):
            invalid_strata.append(pair_id)
    extra = sorted(set(annotated_rows) - set(base_rows))
    mismatched.extend(extra)
    return {
        "base_pair_identity_match": not mismatched,
        "identity_mismatch_pair_ids": sorted(mismatched),
        "symmetric_relation_strata": not invalid_strata,
        "invalid_stratum_pair_ids": sorted(invalid_strata),
        "required_identity": "pair_id plus sorted source_utterance_ids",
        "required_stratum_property": (
            "contains every distinct dialect and metadata-condition endpoint token"
        ),
    }


def build_symmetric_projection_annotation(
    base: Mapping[str, Any], *, base_manifest_sha256: str
) -> dict[str, Any]:
    """Attach an unordered projection design-stratum key to every pair."""
    output = {key: value for key, value in base.items() if key != "pairs"}
    output.update(
        {
            "schema": "projection-design-strata-v1",
            "base_manifest_sha256": base_manifest_sha256,
            "stratum_definition": (
                "group|unordered dialect relation|unordered metadata-condition relation"
            ),
            "pairs": [],
        }
    )
    for raw_row in base.get("pairs", []):
        row = dict(raw_row)
        dialect_relation = "::".join(
            sorted(set(map(str, row.get("dialect_labels", []))))
        )
        condition_relation = "::".join(
            sorted(set(map(str, row.get("recording_conditions", []))))
        )
        if not dialect_relation or not condition_relation:
            raise ValueError("projection design strata require dialect and condition labels")
        output["pairs"].append(
            {
                **row,
                "utterance_ids": _utterances(row),
                "matched_stratum": (
                    f"{row['group']}|{dialect_relation}|{condition_relation}"
                ),
                "matched_fields": [
                    "group",
                    "unordered_dialect_relation",
                    "unordered_metadata_condition_relation",
                ],
            }
        )
    return output


def audit_projection_manifest(
    evaluation: Mapping[str, Any],
    calibration: Mapping[str, Any],
    phenomenon: Mapping[str, Any],
    *,
    expected_group_counts: Mapping[str, int],
    expected_seed: int,
    projection_sha256: str,
    phenomenon_sha256: str,
) -> dict[str, Any]:
    """Summarize the locked projection design and reject identity leakage."""
    evaluation_pairs = [dict(row) for row in evaluation.get("pairs", [])]
    calibration_pairs = [dict(row) for row in calibration.get("pairs", [])]
    if int(evaluation.get("seed", -1)) != expected_seed:
        raise ValueError("projection sampling seed drift")
    group_counts = Counter(str(row.get("group")) for row in evaluation_pairs)
    observed_counts = {group: int(group_counts[group]) for group in GROUPS}
    if observed_counts != dict(expected_group_counts):
        raise ValueError(
            f"projection group count drift: expected {dict(expected_group_counts)}, got {observed_counts}"
        )

    utterances_by_group: dict[str, list[str]] = {group: [] for group in GROUPS}
    group_reports = {}
    for group in GROUPS:
        rows = [row for row in evaluation_pairs if row["group"] == group]
        for row in rows:
            utterances_by_group[group].extend(_utterances(row))
        group_reports[group] = {
            "pair_count": len(rows),
            "speaker_count": len(_speakers(rows)),
            "utterance_count": len(set(utterances_by_group[group])),
            "dialect_label_count": len(
                {str(label) for row in rows for label in row.get("dialect_labels", [])}
            ),
            "dialect_relation_count": len(
                {tuple(sorted(map(str, row.get("dialect_labels", [])))) for row in rows}
            ),
            "metadata_condition_count": len(
                {str(value) for row in rows for value in row.get("recording_conditions", [])}
            ),
            "metadata_condition_relation_count": len(
                {
                    tuple(sorted(map(str, row.get("recording_conditions", []))))
                    for row in rows
                }
            ),
        }
    within_reuse = {
        group: len(values) - len(set(values))
        for group, values in utterances_by_group.items()
    }
    if any(within_reuse.values()):
        raise ValueError("projection manifest reuses an utterance within a group")
    utterance_groups: dict[str, set[str]] = defaultdict(set)
    for group, values in utterances_by_group.items():
        for utterance in values:
            utterance_groups[utterance].add(group)
    cross_group = sorted(
        utterance for utterance, groups in utterance_groups.items() if len(groups) > 1
    )
    speaker_overlap = sorted(
        _speakers(evaluation_pairs).intersection(_speakers(calibration_pairs))
    )
    if speaker_overlap:
        raise ValueError("projection evaluation must be speaker-disjoint from calibration")
    collision = detect_pair_id_collisions(
        evaluation_pairs, phenomenon.get("pairs", [])
    )
    return {
        "schema": "projection-evaluation-manifest-summary-v1",
        "projection_manifest_sha256": projection_sha256,
        "phenomenon_manifest_sha256": phenomenon_sha256,
        "distinct_from_phenomenon_manifest": projection_sha256 != phenomenon_sha256,
        "pair_count": len(evaluation_pairs),
        "group_pair_counts": observed_counts,
        "groups": group_reports,
        "within_group_utterance_reuse": within_reuse,
        "cross_group_utterance_reuse_count": len(cross_group),
        "cross_group_utterance_id_sha256": hashlib.sha256(
            "\n".join(cross_group).encode("utf-8")
        ).hexdigest(),
        "evaluation_speaker_count": len(_speakers(evaluation_pairs)),
        "calibration_speaker_count_in_pairs": len(_speakers(calibration_pairs)),
        "evaluation_calibration_speaker_overlap": speaker_overlap,
        "sampling": {
            "seed": expected_seed,
            "implementation": "src.pair_sampling.sample_pairs",
            "algorithm": "deterministic groupwise round-robin without within-group utterance reuse",
            "ordering": "recording_condition then utterance_id within sorted group strata",
            "seed_role": "recorded protocol identifier; current deterministic selector does not randomize order",
        },
        "pair_id_collision_audit": collision,
    }


def _manifest_hash_values(payload: Any) -> list[tuple[str, str]]:
    found = []

    def walk(value: Any, path: str) -> None:
        if isinstance(value, Mapping):
            for key, child in value.items():
                child_path = f"{path}/{key}"
                if key in HASH_KEYS and isinstance(child, str):
                    found.append((child_path, child.lower()))
                walk(child, child_path)
        elif isinstance(value, list):
            for index, child in enumerate(value):
                walk(child, f"{path}/{index}")

    walk(payload, "")
    return found


def downstream_manifest_status(
    payload: Mapping[str, Any], expected_hash: str
) -> dict[str, Any]:
    """Classify whether a downstream report proves projection manifest identity."""
    values = _manifest_hash_values(payload)
    recognized = [(path, value) for path, value in values if SHA256_RE.match(value)]
    if any(value == expected_hash for _, value in recognized):
        return {
            "status": "verified",
            "reason": "expected_projection_manifest_hash_recorded",
            "hash_locations": [path for path, value in recognized if value == expected_hash],
        }
    if recognized:
        return {
            "status": "invalidated",
            "reason": "projection_manifest_hash_mismatch",
            "recorded_hashes": sorted({value for _, value in recognized}),
        }
    bootstrap = payload.get("bootstrap", {})
    if "matched_stratum" in str(bootstrap.get("resampling_unit", "")):
        return {
            "status": "invalidated",
            "reason": "matched_stratum_provenance_missing",
        }
    return {"status": "invalidated", "reason": "projection_manifest_hash_missing"}


def _verify_source_records(
    pairs: Iterable[Mapping[str, Any]], records: Iterable[Mapping[str, Any]]
) -> dict[str, Any]:
    record_index = {str(row["utterance_id"]): row for row in records}
    checked = 0
    for pair in pairs:
        utterances = _utterances(pair)
        expected_hashes = list(map(str, pair.get("source_record_hashes", [])))
        if len(expected_hashes) != 2:
            raise ValueError("pair source_record_hashes are missing")
        observed_hashes = []
        for utterance in utterances:
            if utterance not in record_index:
                raise ValueError(f"projection utterance missing from source records: {utterance}")
            observed_hashes.append(_canonical_record_hash(record_index[utterance]))
        if observed_hashes != expected_hashes:
            raise ValueError(f"source record identity drift for pair {pair['pair_id']}")
        checked += 1
    return {
        "pair_source_identity_checked": checked,
        "identity_fields": [
            "utterance_id",
            "speaker_id",
            "dialect_label",
            "split",
            "recording_condition",
        ],
        "status": "passed",
    }


def build_report(
    protocol_path: Path,
    pair_path: Path,
    calibration_path: Path,
    phenomenon_path: Path,
    annotated_path: Path,
    symmetric_annotated_output: Path,
    records_path: Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    protocol = yaml.safe_load(protocol_path.read_text(encoding="utf-8"))
    evaluation = json.loads(pair_path.read_text(encoding="utf-8"))
    calibration = json.loads(calibration_path.read_text(encoding="utf-8"))
    phenomenon = json.loads(phenomenon_path.read_text(encoding="utf-8"))
    annotated = json.loads(annotated_path.read_text(encoding="utf-8"))
    records = json.loads(records_path.read_text(encoding="utf-8"))
    projection_hash = _sha256(pair_path)
    phenomenon_hash = _sha256(phenomenon_path)
    expected_hash = str(protocol["source_hashes"][str(pair_path).replace("\\", "/")])
    if projection_hash != expected_hash:
        raise ValueError("projection manifest hash drift")
    summary = audit_projection_manifest(
        evaluation,
        calibration,
        phenomenon,
        expected_group_counts=protocol["projection_evaluation"]["group_counts"],
        expected_seed=int(protocol["projection_evaluation"]["seed"]),
        projection_sha256=projection_hash,
        phenomenon_sha256=phenomenon_hash,
    )
    source_verification = _verify_source_records(
        evaluation["pairs"], records["records"]
    )
    annotated_audit = audit_annotated_projection_manifest(evaluation, annotated)
    if not annotated_audit["base_pair_identity_match"]:
        raise ValueError("annotated projection manifest has endpoint identity drift")
    symmetric_annotation = build_symmetric_projection_annotation(
        evaluation, base_manifest_sha256=projection_hash
    )
    symmetric_annotated_output.parent.mkdir(parents=True, exist_ok=True)
    symmetric_annotated_output.write_text(
        json.dumps(symmetric_annotation, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    symmetric_audit = audit_annotated_projection_manifest(
        evaluation, symmetric_annotation
    )
    if not symmetric_audit["base_pair_identity_match"] or not symmetric_audit[
        "symmetric_relation_strata"
    ]:
        raise ValueError("generated symmetric projection annotation failed validation")

    report_paths = [
        Path("results/analysis/architecture_cross_loss_factorial.json"),
        Path("results/analysis/reference_representative_sensitivity.json"),
        Path("results/analysis/estimand_weighting_intervals.json"),
        Path("results/analysis/relation_ranking_clustered.json"),
    ]
    downstream = []
    for path in report_paths:
        if not path.exists():
            downstream.append(
                {"path": str(path), "status": "invalidated", "reason": "report_missing"}
            )
            continue
        status = downstream_manifest_status(
            json.loads(path.read_text(encoding="utf-8")), projection_hash
        )
        if path.name == "estimand_weighting_intervals.json" and not annotated_audit[
            "symmetric_relation_strata"
        ]:
            status = {
                "status": "invalidated",
                "reason": "projection_stratum_definition_not_endpoint_symmetric",
                "annotated_manifest": str(annotated_path),
                "annotated_manifest_sha256": _sha256(annotated_path),
            }
        downstream.append({"path": str(path), **status})
    invalidated = [row for row in downstream if row["status"] != "verified"]
    report = {
        **summary,
        "status": "passed",
        "protocol": str(protocol_path),
        "protocol_sha256": _sha256(protocol_path),
        "projection_manifest": str(pair_path),
        "calibration_manifest": str(calibration_path),
        "phenomenon_manifest": str(phenomenon_path),
        "source_records": str(records_path),
        "source_record_verification": source_verification,
        "annotated_projection_manifest": str(annotated_path),
        "annotated_projection_manifest_sha256": _sha256(annotated_path),
        "annotated_projection_audit": annotated_audit,
        "symmetric_projection_manifest": str(symmetric_annotated_output),
        "symmetric_projection_manifest_sha256": _sha256(symmetric_annotated_output),
        "symmetric_projection_audit": symmetric_audit,
        "downstream_report_audit": downstream,
        "invalidated_reports": invalidated,
        "invalidation_note": (
            "Current downstream reports must record the locked projection-manifest "
            "hash. Weighting reports additionally require a symmetric design-stratum "
            "annotation. Superseded reports are not used for manuscript claims."
        ),
    }
    gate = {
        "schema": "projection-evaluation-manifest-gate-v1",
        "status": "passed",
        "manifest_identity_valid": True,
        "group_counts_valid": summary["group_pair_counts"]
        == protocol["projection_evaluation"]["group_counts"],
        "speaker_disjoint": not summary["evaluation_calibration_speaker_overlap"],
        "within_group_utterance_reuse_absent": not any(
            summary["within_group_utterance_reuse"].values()
        ),
        "pair_id_only_join_forbidden": not summary["pair_id_collision_audit"][
            "safe_to_join_on_pair_id_alone"
        ],
        "downstream_status": (
            "invalidated_pending_rerun" if invalidated else "verified"
        ),
        "invalidated_report_count": len(invalidated),
    }
    return report, gate


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--pairs", type=Path, required=True)
    parser.add_argument("--calibration-pairs", type=Path, required=True)
    parser.add_argument("--phenomenon-pairs", type=Path, required=True)
    parser.add_argument("--annotated-pairs", type=Path, required=True)
    parser.add_argument("--symmetric-annotated-output", type=Path, required=True)
    parser.add_argument("--records", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--gate", type=Path, required=True)
    args = parser.parse_args()
    report, gate = build_report(
        args.protocol,
        args.pairs,
        args.calibration_pairs,
        args.phenomenon_pairs,
        args.annotated_pairs,
        args.symmetric_annotated_output,
        args.records,
    )
    for path, payload in ((args.report, report), (args.gate, gate)):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
