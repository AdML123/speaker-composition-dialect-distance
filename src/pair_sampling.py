"""Construct leakage-controlled A/B/C/D pairs from an audited manifest."""

from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from itertools import combinations
from pathlib import Path
from typing import Any, Iterable, Mapping

from .corpus_audit import REQUIRED_FIELDS


class PairSamplingError(ValueError):
    """Raised when pair construction cannot satisfy the locked sampling contract."""


DEFAULT_GROUP_LIMITS = {"A": 32, "B": 32, "C": 32, "D": 32}
GROUPS = ("A", "B", "C", "D")
VALID_SPLITS = ("train", "calibration", "evaluation")


def _require_mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise PairSamplingError(f"{label} must be a mapping")
    return value


def _validated_records(records: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
    checked: list[dict[str, Any]] = []
    speaker_splits: dict[str, str] = {}
    utterance_ids: set[str] = set()
    for index, raw_record in enumerate(records):
        record = _require_mapping(raw_record, f"record at index {index}")
        missing = REQUIRED_FIELDS.difference(record)
        if missing:
            raise PairSamplingError(f"missing required field: {sorted(missing)[0]}")
        checked_record = dict(record)
        for field in REQUIRED_FIELDS - {"sample_rate"}:
            value = checked_record[field]
            if not isinstance(value, str) or not value.strip():
                raise PairSamplingError(f"malformed record field: {field}")
        sample_rate = checked_record["sample_rate"]
        if isinstance(sample_rate, bool) or not isinstance(sample_rate, int) or sample_rate <= 0:
            raise PairSamplingError("malformed record field: sample_rate")
        if checked_record["split"] not in VALID_SPLITS:
            raise PairSamplingError("malformed record field: split")
        utterance_id = checked_record["utterance_id"]
        if utterance_id in utterance_ids:
            raise PairSamplingError("duplicate utterance_id")
        utterance_ids.add(utterance_id)
        prior_split = speaker_splits.setdefault(checked_record["speaker_id"], checked_record["split"])
        if prior_split != checked_record["split"]:
            raise PairSamplingError("speaker leakage across splits")
        checked.append(checked_record)
    return checked


def _canonical_record_hash(record: Mapping[str, Any]) -> str:
    payload = {
        key: record[key]
        for key in ("utterance_id", "speaker_id", "dialect_label", "split", "recording_condition")
    }
    blob = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def _pair_hash(group: str, record_a: Mapping[str, Any], record_b: Mapping[str, Any]) -> str:
    ordered = sorted(
        (
            _canonical_record_hash(record_a),
            _canonical_record_hash(record_b),
        )
    )
    payload = f"{group}|{ordered[0]}|{ordered[1]}"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _ordered_records(records: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
    return sorted(
        (dict(record) for record in records),
        key=lambda record: (
            record["recording_condition"],
            record["utterance_id"],
        ),
    )


def _record_pair_payload(group: str, record_a: Mapping[str, Any], record_b: Mapping[str, Any], index: int) -> dict[str, Any]:
    ordered = sorted((dict(record_a), dict(record_b)), key=lambda record: record["utterance_id"])
    source_hashes = [_canonical_record_hash(record) for record in ordered]
    dialect_labels = sorted({record["dialect_label"] for record in ordered})
    speaker_ids = sorted({record["speaker_id"] for record in ordered})
    return {
        "pair_id": f"{group}-{index:06d}",
        "group": group,
        "split": ordered[0]["split"],
        "dialect_labels": dialect_labels,
        "speaker_ids": speaker_ids,
        "source_utterance_ids": [record["utterance_id"] for record in ordered],
        "recording_conditions": [record["recording_condition"] for record in ordered],
        "source_record_hashes": source_hashes,
        "pair_hash": _pair_hash(group, ordered[0], ordered[1]),
    }


def _round_robin_pairs(
    strata: Mapping[str, Iterable[tuple[Mapping[str, Any], Mapping[str, Any]]]],
    limit: int,
) -> list[tuple[Mapping[str, Any], Mapping[str, Any]]]:
    selected: list[tuple[Mapping[str, Any], Mapping[str, Any]]] = []
    used_utterance_ids: set[str] = set()
    labels = [label for label, items in sorted(strata.items()) if items]
    iterators = {label: iter(items) for label, items in strata.items() if items}
    while labels and len(selected) < limit:
        progressed = False
        exhausted: list[str] = []
        for label in labels:
            while True:
                try:
                    candidate = next(iterators[label])
                except StopIteration:
                    exhausted.append(label)
                    break
                utterance_ids = {candidate[0]["utterance_id"], candidate[1]["utterance_id"]}
                if used_utterance_ids.intersection(utterance_ids):
                    continue
                selected.append(candidate)
                used_utterance_ids.update(utterance_ids)
                progressed = True
                break
            if len(selected) >= limit:
                break
        if not progressed:
            break
        for label in exhausted:
            labels.remove(label)
            iterators.pop(label, None)
    return selected


def _iter_same_speaker_pairs(
    by_dialect: Mapping[str, list[Mapping[str, Any]]],
    speakers: Iterable[str],
) -> Iterable[tuple[Mapping[str, Any], Mapping[str, Any]]]:
    for speaker in speakers:
        yield from combinations(_ordered_records(by_dialect[speaker]), 2)


def _iter_same_dialect_pairs(
    speakers: Mapping[str, list[Mapping[str, Any]]],
) -> Iterable[tuple[Mapping[str, Any], Mapping[str, Any]]]:
    for speaker_a, speaker_b in combinations(sorted(speakers), 2):
        for record_a in speakers[speaker_a]:
            for record_b in speakers[speaker_b]:
                yield record_a, record_b


def _iter_cross_dialect_same_speaker_pairs(
    dialects: Mapping[str, list[Mapping[str, Any]]],
) -> Iterable[tuple[Mapping[str, Any], Mapping[str, Any]]]:
    for dialect_a, dialect_b in combinations(sorted(dialects), 2):
        for record_a in dialects[dialect_a]:
            for record_b in dialects[dialect_b]:
                yield record_a, record_b


def _iter_cross_dialect_pairs(
    speakers_a: Mapping[str, list[Mapping[str, Any]]],
    speakers_b: Mapping[str, list[Mapping[str, Any]]],
) -> Iterable[tuple[Mapping[str, Any], Mapping[str, Any]]]:
    for speaker_a in sorted(speakers_a):
        for speaker_b in sorted(speakers_b):
            if speaker_a == speaker_b:
                continue
            for record_a in speakers_a[speaker_a]:
                for record_b in speakers_b[speaker_b]:
                    yield record_a, record_b


def _bucket_by(records: Iterable[Mapping[str, Any]], *fields: str) -> dict[Any, Any]:
    buckets: dict[Any, Any] = defaultdict(dict if len(fields) > 1 else list)
    for record in records:
        cursor = buckets
        for field in fields[:-1]:
            cursor = cursor.setdefault(record[field], {})
        cursor.setdefault(record[fields[-1]], []).append(record)
    return buckets


def _sample_group_a(records: list[dict[str, Any]], limit: int) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    pairs: list[dict[str, Any]] = []
    exclusions: list[dict[str, Any]] = []
    by_split = _bucket_by(records, "split", "dialect_label", "speaker_id")
    for split in sorted(by_split):
        strata: dict[str, Iterable[tuple[Mapping[str, Any], Mapping[str, Any]]]] = {}
        for dialect in sorted(by_split[split]):
            strata[dialect] = _iter_same_speaker_pairs(
                by_split[split][dialect],
                sorted(by_split[split][dialect]),
            )
        selected = _round_robin_pairs(strata, limit - len(pairs))
        for record_a, record_b in selected:
            pairs.append(_record_pair_payload("A", record_a, record_b, len(pairs) + 1))
        if len(pairs) >= limit:
            return pairs, exclusions
    return pairs, exclusions


def _sample_group_b(records: list[dict[str, Any]], limit: int) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    pairs: list[dict[str, Any]] = []
    exclusions: list[dict[str, Any]] = []
    by_split = _bucket_by(records, "split", "dialect_label", "speaker_id")
    for split in sorted(by_split):
        strata: dict[str, Iterable[tuple[Mapping[str, Any], Mapping[str, Any]]]] = {}
        for dialect in sorted(by_split[split]):
            speakers = {speaker: _ordered_records(bucket) for speaker, bucket in sorted(by_split[split][dialect].items())}
            strata[dialect] = _iter_same_dialect_pairs(speakers)
        selected = _round_robin_pairs(strata, limit - len(pairs))
        for record_a, record_b in selected:
            pairs.append(_record_pair_payload("B", record_a, record_b, len(pairs) + 1))
        if len(pairs) >= limit:
            return pairs, exclusions
    return pairs, exclusions


def _sample_group_c(records: list[dict[str, Any]], limit: int) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    pairs: list[dict[str, Any]] = []
    exclusions: list[dict[str, Any]] = []
    by_split = _bucket_by(records, "split", "speaker_id", "dialect_label")
    for split in sorted(by_split):
        strata: dict[str, Iterable[tuple[Mapping[str, Any], Mapping[str, Any]]]] = {}
        for speaker in sorted(by_split[split]):
            dialects = {dialect: _ordered_records(bucket) for dialect, bucket in sorted(by_split[split][speaker].items())}
            strata[speaker] = _iter_cross_dialect_same_speaker_pairs(dialects)
        selected = _round_robin_pairs(strata, limit - len(pairs))
        for record_a, record_b in selected:
            pairs.append(_record_pair_payload("C", record_a, record_b, len(pairs) + 1))
        if len(pairs) >= limit:
            return pairs, exclusions
    return pairs, exclusions


def _sample_group_d(records: list[dict[str, Any]], limit: int) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    pairs: list[dict[str, Any]] = []
    exclusions: list[dict[str, Any]] = []
    by_split = _bucket_by(records, "split", "dialect_label", "speaker_id")
    for split in sorted(by_split):
        strata: dict[str, Iterable[tuple[Mapping[str, Any], Mapping[str, Any]]]] = {}
        for dialect_a, dialect_b in combinations(sorted(by_split[split]), 2):
            speakers_a = {speaker: _ordered_records(bucket) for speaker, bucket in sorted(by_split[split][dialect_a].items())}
            speakers_b = {speaker: _ordered_records(bucket) for speaker, bucket in sorted(by_split[split][dialect_b].items())}
            strata[f"{dialect_a}::{dialect_b}"] = _iter_cross_dialect_pairs(speakers_a, speakers_b)
        selected = _round_robin_pairs(strata, limit - len(pairs))
        for record_a, record_b in selected:
            pairs.append(_record_pair_payload("D", record_a, record_b, len(pairs) + 1))
        if len(pairs) >= limit:
            return pairs, exclusions
    return pairs, exclusions


def _group_sampler(group: str):
    return {
        "A": _sample_group_a,
        "B": _sample_group_b,
        "C": _sample_group_c,
        "D": _sample_group_d,
    }[group]


def sample_pairs(
    records: Iterable[Mapping[str, Any]],
    *,
    group_limits: Mapping[str, int] | None = None,
    seed: int = 20260829,
) -> dict[str, Any]:
    """Sample deterministic, leakage-controlled A/B/C/D pairs from an audit manifest."""
    checked = _validated_records(records)
    limits = dict(DEFAULT_GROUP_LIMITS if group_limits is None else group_limits)
    if set(limits) != set(GROUPS):
        raise PairSamplingError("group_limits must define A, B, C, and D")
    for key, value in limits.items():
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise PairSamplingError(f"group_limits[{key}] must be a non-negative integer")

    capacities = {"A": 0, "B": 0, "C": 0, "D": 0}
    from .corpus_audit import compute_pair_capacity

    capacities.update(compute_pair_capacity(checked))

    pairs: list[dict[str, Any]] = []
    exclusions: list[dict[str, Any]] = []
    group_summary: dict[str, dict[str, Any]] = {}
    for group in GROUPS:
        requested = limits[group]
        available = capacities[group]
        if requested == 0:
            group_summary[group] = {"requested": 0, "available": available, "sampled": 0, "status": "disabled"}
            continue
        sampled_pairs, group_exclusions = _group_sampler(group)(checked, requested)
        if len(sampled_pairs) < requested:
            group_exclusions.append(
                {
                    "group": group,
                    "reason": "insufficient_capacity",
                    "requested": requested,
                    "available": available,
                    "sampled": len(sampled_pairs),
                }
            )
        pairs.extend(sampled_pairs)
        exclusions.extend(group_exclusions)
        if group == "C" and available < 200:
            status = "auxiliary"
        elif len(sampled_pairs) < requested:
            status = "partial"
        else:
            status = "ok"
        group_summary[group] = {
            "requested": requested,
            "available": available,
            "sampled": len(sampled_pairs),
            "status": status,
        }

    payload = {
        "schema": "pair-sampling-v1",
        "seed": seed,
        "record_count": len(checked),
        "speaker_count": len({record["speaker_id"] for record in checked}),
        "dialect_count": len({record["dialect_label"] for record in checked}),
        "split_counts": {split: sum(1 for record in checked if record["split"] == split) for split in VALID_SPLITS},
        "group_limits": limits,
        "group_summary": group_summary,
        "pair_count": len(pairs),
        "pairs": pairs,
        "exclusions": exclusions,
        "status": "passed",
    }
    return payload


def _atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    tmp.replace(path)


def write_pair_sampling_protocol(path: str | Path, payload: dict[str, Any]) -> None:
    _atomic_write_json(Path(path), payload)
