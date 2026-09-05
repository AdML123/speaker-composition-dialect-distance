"""Audit calibration source pools against the locked projection evaluation set."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


def _items(row: Mapping[str, Any], *keys: str) -> set[str]:
    for key in keys:
        value = row.get(key)
        if value is not None:
            if isinstance(value, (list, tuple, set)):
                return {str(item) for item in value}
            return {str(value)}
    return set()


def _canonical_digest(rows: Sequence[Mapping[str, Any]]) -> str:
    payload = json.dumps(list(rows), sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _source_summary(role: str, rows: Sequence[Mapping[str, Any]], evaluation: Mapping[str, set[str]]) -> dict[str, Any]:
    pair_ids = _items_from_rows(rows, "pair_id")
    pair_identities = {
        f"{row.get('pair_id', '')}|{'|'.join(sorted(_items(row, 'source_utterance_ids', 'utterance_ids')))}"
        for row in rows
    }
    speaker_ids = _items_from_rows(rows, "speaker_ids", "speaker_id")
    utterance_ids = _items_from_rows(rows, "source_utterance_ids", "utterance_ids")
    intersections = {
        "pair_ids": sorted(pair_ids & evaluation["pair_ids"]),
        "pair_identities": sorted(pair_identities & evaluation["pair_identities"]),
        "speaker_ids": sorted(speaker_ids & evaluation["speaker_ids"]),
        "utterance_ids": sorted(utterance_ids & evaluation["utterance_ids"]),
    }
    return {
        "role": role,
        "row_count": len(rows),
        "pair_count": len(pair_ids),
        "speaker_count": len(speaker_ids),
        "utterance_count": len(utterance_ids),
        "sha256": _canonical_digest(rows),
        "intersections": intersections,
        "pair_id_namespace_overlap_only": bool(intersections["pair_ids"])
        and not bool(intersections["pair_identities"]),
        "evaluation_rows_used_for_fitting": any(
            intersections[key]
            for key in ("pair_identities", "speaker_ids", "utterance_ids")
        ),
    }


def _items_from_rows(rows: Iterable[Mapping[str, Any]], *keys: str) -> set[str]:
    values: set[str] = set()
    for row in rows:
        values.update(_items(row, *keys))
    return values


def audit_projection_sources(
    calibration_pairs: Sequence[Mapping[str, Any]],
    cross_examples: Sequence[Mapping[str, Any]],
    evaluation_pairs: Sequence[Mapping[str, Any]],
    fitted_sources: Mapping[str, Sequence[Mapping[str, Any]]] | None = None,
) -> dict[str, Any]:
    """Return a fail-closed source/evaluation intersection audit.

    ``fitted_sources`` names the exact pools consumed by fitting. When omitted,
    the canonical projection pair and auxiliary cross-pair pools are used.
    Evaluation rows are always treated as scoring-only.
    """
    evaluation = {
        "pair_ids": _items_from_rows(evaluation_pairs, "pair_id"),
        "pair_identities": {
            f"{row.get('pair_id', '')}|{'|'.join(sorted(_items(row, 'source_utterance_ids', 'utterance_ids')))}"
            for row in evaluation_pairs
        },
        "speaker_ids": _items_from_rows(evaluation_pairs, "speaker_ids", "speaker_id"),
        "utterance_ids": _items_from_rows(evaluation_pairs, "source_utterance_ids", "utterance_ids"),
    }
    sources = fitted_sources or {
        "projection_calibration": calibration_pairs,
        "calibration_auxiliary_cross_pair": cross_examples,
    }
    rows = [
        _source_summary(str(role), list(pool), evaluation)
        for role, pool in sources.items()
    ]
    collisions = [
        row
        for row in rows
        if any(
            row["intersections"][key]
            for key in ("pair_identities", "speaker_ids", "utterance_ids")
        )
    ]
    if collisions:
        first = collisions[0]
        fields = ", ".join(
            key for key, values in first["intersections"].items() if values
        )
        raise ValueError(f"evaluation collision in {first['role']}: {fields}")
    return {
        "schema": "calibration-leakage-audit-v1",
        "status": "passed",
        "evaluation_manifest_role": "projection_evaluation",
        "evaluation_pair_count": len(evaluation["pair_ids"]),
        "evaluation_speaker_count": len(evaluation["speaker_ids"]),
        "evaluation_utterance_count": len(evaluation["utterance_ids"]),
        "sources": rows,
        "evaluation_rows_used_for_fitting": False,
        "raw_pair_id_namespace_overlap_is_not_identity": True,
        "fitted_source_roles": sorted(str(role) for role in sources),
        "legacy_ab_manifest_used_by_lp": False,
    }


def _load_pairs(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return [dict(row) for row in payload["pairs"]]


def _load_records(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return [dict(row) for row in payload.get("records", payload)]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--calibration-pairs", type=Path, required=True)
    parser.add_argument("--legacy-calibration-pairs", type=Path)
    parser.add_argument("--evaluation-pairs", type=Path, required=True)
    parser.add_argument("--records", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--gate", type=Path, required=True)
    args = parser.parse_args()
    calibration = _load_pairs(args.calibration_pairs)
    evaluation = _load_pairs(args.evaluation_pairs)
    cross: list[dict[str, Any]] = []
    if args.records:
        from .cross_dialect_projection_head import build_training_examples

        records = [row for row in _load_records(args.records) if str(row.get("split")) == "calibration"]
        reference_path = args.calibration_pairs.parent.parent / "references" / "taxonomy_matrix.json"
        if reference_path.is_file():
            reference_payload = json.loads(reference_path.read_text(encoding="utf-8"))
            reference = reference_payload["matrix"]
            cross = list(build_training_examples(records, calibration, reference)["cross_dialect_examples"])
    report = audit_projection_sources(calibration, cross, evaluation)
    report["calibration_manifest"] = str(args.calibration_pairs)
    report["evaluation_manifest"] = str(args.evaluation_pairs)
    if args.legacy_calibration_pairs:
        report["legacy_calibration_manifest"] = str(args.legacy_calibration_pairs)
        report["legacy_ab_manifest_used_by_lp"] = False
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.gate.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    gate = {
        "schema": "calibration-manifest-role-gate-v1",
        "status": report["status"],
        "audit": str(args.output),
        "evaluation_rows_used_for_fitting": report["evaluation_rows_used_for_fitting"],
        "legacy_ab_manifest_used_by_lp": report["legacy_ab_manifest_used_by_lp"],
        "source_roles": report["fitted_source_roles"],
        "raw_pair_id_namespace_overlap_only": all(
            bool(row.get("pair_id_namespace_overlap_only", False))
            or not row["intersections"]["pair_ids"]
            for row in report["sources"]
        ),
        "exact_pair_identity_intersections": sum(
            len(row["intersections"]["pair_identities"])
            for row in report["sources"]
        ),
        "speaker_intersections": sum(
            len(row["intersections"]["speaker_ids"]) for row in report["sources"]
        ),
        "utterance_intersections": sum(
            len(row["intersections"]["utterance_ids"]) for row in report["sources"]
        ),
    }
    args.gate.write_text(json.dumps(gate, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
