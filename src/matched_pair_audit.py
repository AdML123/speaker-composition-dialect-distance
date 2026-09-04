"""Construct condition- and content-matched A/B pairs with an auditable contract."""

from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from itertools import combinations
from typing import Any, Iterable, Mapping


class MatchingAuditError(ValueError):
    """Raised when the matched-pair contract cannot be validated."""


GROUPS = ("A", "B")
DEFAULT_GROUP_LIMITS = {"A": 32, "B": 32}
VALID_SPLITS = {"train", "calibration", "evaluation"}
MATCHED_FIELDS = ("split", "dialect_label", "recording_condition", "content_id")
REQUIRED_FIELDS = {
    "utterance_id",
    "audio_path",
    "speaker_id",
    "dialect_label",
    "sample_rate",
    "recording_condition",
    "split",
    "content_id",
}
BASE_REQUIRED_FIELDS = REQUIRED_FIELDS - {"content_id"}


def _validated_records(records: Iterable[Mapping[str, Any]], *, require_content: bool) -> list[dict[str, Any]]:
    checked: list[dict[str, Any]] = []
    utterance_ids: set[str] = set()
    speaker_splits: dict[str, str] = {}
    for index, raw_record in enumerate(records):
        if not isinstance(raw_record, Mapping):
            raise MatchingAuditError(f"record at index {index} must be a mapping")
        required = REQUIRED_FIELDS if require_content else BASE_REQUIRED_FIELDS
        missing = required.difference(raw_record)
        if missing:
            raise MatchingAuditError(f"missing required field: {sorted(missing)[0]}")
        record = dict(raw_record)
        for field in required - {"sample_rate"}:
            if not isinstance(record[field], str) or not record[field].strip():
                raise MatchingAuditError(f"malformed record field: {field}")
        if isinstance(record["sample_rate"], bool) or not isinstance(record["sample_rate"], int) or record["sample_rate"] <= 0:
            raise MatchingAuditError("malformed record field: sample_rate")
        if record["split"] not in VALID_SPLITS:
            raise MatchingAuditError("malformed record field: split")
        if record["utterance_id"] in utterance_ids:
            raise MatchingAuditError("duplicate utterance_id")
        utterance_ids.add(record["utterance_id"])
        prior_split = speaker_splits.setdefault(record["speaker_id"], record["split"])
        if prior_split != record["split"]:
            raise MatchingAuditError("speaker leakage across splits")
        checked.append(record)
    if not checked:
        raise MatchingAuditError("at least one record is required")
    return checked


def _record_hash(record: Mapping[str, Any]) -> str:
    payload = {
        key: record.get(key)
        for key in ("utterance_id", "speaker_id", "dialect_label", "split", "recording_condition", "content_id")
    }
    blob = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def _pair_payload(
    group: str,
    first: Mapping[str, Any],
    second: Mapping[str, Any],
    index: int,
    *,
    matched_fields: Iterable[str] = MATCHED_FIELDS,
) -> dict[str, Any]:
    ordered = sorted((dict(first), dict(second)), key=lambda item: item["utterance_id"])
    matched_stratum = "|".join(
        str(ordered[0].get(field, ""))
        for field in matched_fields
    )
    return {
        "pair_id": f"{group}-{index:06d}",
        "group": group,
        "split": ordered[0]["split"],
        "dialect_labels": sorted({item["dialect_label"] for item in ordered}),
        "speaker_ids": sorted({item["speaker_id"] for item in ordered}),
        "content_ids": [item.get("content_id") for item in ordered],
        "source_utterance_ids": [item["utterance_id"] for item in ordered],
        "utterance_ids": [item["utterance_id"] for item in ordered],
        "matched_stratum": matched_stratum,
        "recording_conditions": [item["recording_condition"] for item in ordered],
        "source_record_hashes": [_record_hash(item) for item in ordered],
        "matched_fields": sorted(set(matched_fields)),
    }


def _candidate_key(seed: int, group: str, first: Mapping[str, Any], second: Mapping[str, Any]) -> str:
    ids = sorted((first["utterance_id"], second["utterance_id"]))
    return hashlib.sha256(f"{seed}|{group}|{ids[0]}|{ids[1]}".encode("utf-8")).hexdigest()


def _select(candidates: Iterable[tuple[Mapping[str, Any], Mapping[str, Any]]], *, group: str, limit: int, seed: int) -> list[tuple[Mapping[str, Any], Mapping[str, Any]]]:
    if limit == 0:
        return []
    ordered = sorted(candidates, key=lambda pair: _candidate_key(seed, group, pair[0], pair[1]))
    selected: list[tuple[Mapping[str, Any], Mapping[str, Any]]] = []
    used: set[str] = set()
    for first, second in ordered:
        ids = {first["utterance_id"], second["utterance_id"]}
        if used.intersection(ids):
            continue
        selected.append((first, second))
        used.update(ids)
        if len(selected) >= limit:
            break
    return selected


def sample_matched_ab_pairs(
    records: Iterable[Mapping[str, Any]],
    *,
    group_limits: Mapping[str, int] | None = None,
    seed: int = 20260829,
    match_content: bool = True,
) -> dict[str, Any]:
    """Sample deterministic A/B pairs under exact-content or condition-only matching."""
    checked = _validated_records(records, require_content=match_content)
    limits = dict(DEFAULT_GROUP_LIMITS if group_limits is None else group_limits)
    if set(limits) != set(GROUPS):
        raise MatchingAuditError("group_limits must define A and B")
    for group, limit in limits.items():
        if isinstance(limit, bool) or not isinstance(limit, int) or limit < 0:
            raise MatchingAuditError(f"group_limits[{group}] must be a non-negative integer")

    matched_fields = ("split", "dialect_label", "recording_condition", "content_id") if match_content else (
        "split", "dialect_label", "recording_condition"
    )
    strata: dict[tuple[str, ...], list[dict[str, Any]]] = defaultdict(list)
    for record in checked:
        strata[tuple(record[field] for field in matched_fields)].append(record)

    candidates_a: list[tuple[Mapping[str, Any], Mapping[str, Any]]] = []
    candidates_b: list[tuple[Mapping[str, Any], Mapping[str, Any]]] = []
    for bucket in strata.values():
        by_speaker: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for record in bucket:
            by_speaker[record["speaker_id"]].append(record)
        for speaker_records in by_speaker.values():
            candidates_a.extend(combinations(sorted(speaker_records, key=lambda item: item["utterance_id"]), 2))
        for speaker_a, speaker_b in combinations(sorted(by_speaker), 2):
            candidates_b.extend((first, second) for first in by_speaker[speaker_a] for second in by_speaker[speaker_b])

    capacities = {"A": len(candidates_a), "B": len(candidates_b)}
    selected_by_group = {
        "A": _select(candidates_a, group="A", limit=limits["A"], seed=seed),
        "B": _select(candidates_b, group="B", limit=limits["B"], seed=seed),
    }
    pairs: list[dict[str, Any]] = []
    group_summary: dict[str, dict[str, Any]] = {}
    exclusions: list[dict[str, Any]] = []
    for group in GROUPS:
        selected = selected_by_group[group]
        requested = limits[group]
        if requested == 0:
            status = "disabled"
        elif len(selected) < requested:
            status = "partial"
            exclusions.append({"group": group, "reason": "insufficient_nonoverlapping_capacity", "requested": requested, "available": capacities[group], "sampled": len(selected)})
        else:
            status = "ok"
        pairs.extend(
            _pair_payload(group, first, second, index + 1, matched_fields=matched_fields)
            for index, (first, second) in enumerate(selected)
        )
        group_summary[group] = {"requested": requested, "available": capacities[group], "sampled": len(selected), "status": status}

    payload = {
        "schema": "matched-pair-audit-v1",
        "matching_mode": "exact_content" if match_content else "condition_aware",
        "seed": seed,
        "record_count": len(checked),
        "group_limits": limits,
        "group_summary": group_summary,
        "pair_count": len(pairs),
        "pairs": pairs,
        "exclusions": exclusions,
        "audit": {
            "matching_fields": list(matched_fields),
            "content_cluster_count": len({record.get("content_id") for record in checked if record.get("content_id") is not None}),
            "speaker_count": len({record["speaker_id"] for record in checked}),
            "dialect_count": len({record["dialect_label"] for record in checked}),
            "stratum_count": len(strata),
            "raw_transcripts_in_output": False,
        },
        "status": "passed",
    }
    return payload
