"""Fail-closed corpus manifest validation and audit helpers."""

from __future__ import annotations

import json
import math
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping


class AuditError(ValueError):
    """Raised when corpus evidence cannot satisfy the locked audit contract."""


REQUIRED_FIELDS = {
    "utterance_id",
    "audio_path",
    "speaker_id",
    "dialect_label",
    "sample_rate",
    "recording_condition",
    "split",
}


def load_manifest(path: str | Path) -> list[dict[str, Any]]:
    """Load a JSON manifest, accepting either a records list or wrapper object."""
    try:
        with Path(path).open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
    except (OSError, json.JSONDecodeError) as exc:
        raise AuditError("malformed manifest") from exc
    records = payload.get("records") if isinstance(payload, Mapping) else payload
    if not isinstance(records, list) or not all(isinstance(record, Mapping) for record in records):
        raise AuditError("malformed manifest")
    return [dict(record) for record in records]


def _validated_root(audio_root: str | Path) -> Path:
    root = Path(audio_root).resolve()
    if not root.is_dir():
        raise AuditError("validated audio root must be a directory")
    return root


def _path_below_root(value: str, root: Path) -> Path:
    candidate = Path(value)
    if not candidate.is_absolute():
        candidate = root / candidate
    resolved = candidate.resolve()
    try:
        resolved.relative_to(root)
    except ValueError as exc:
        raise AuditError("audio path outside validated audio root") from exc
    return resolved


def _validate_records(records: Iterable[Mapping[str, Any]], audio_root: str | Path) -> tuple[list[dict[str, Any]], Path]:
    root = _validated_root(audio_root)
    checked: list[dict[str, Any]] = []
    utterance_ids: set[str] = set()
    speaker_splits: dict[str, str] = {}
    for index, raw_record in enumerate(records):
        if not isinstance(raw_record, Mapping):
            raise AuditError(f"malformed record at index {index}")
        missing = REQUIRED_FIELDS.difference(raw_record)
        if missing:
            raise AuditError(f"missing required field: {sorted(missing)[0]}")
        record = dict(raw_record)
        for field in REQUIRED_FIELDS - {"sample_rate"}:
            if not isinstance(record[field], str) or not record[field].strip():
                raise AuditError(f"malformed record field: {field}")
        if isinstance(record["sample_rate"], bool) or not isinstance(record["sample_rate"], int) or record["sample_rate"] <= 0:
            raise AuditError("malformed record field: sample_rate")
        if record["utterance_id"] in utterance_ids:
            raise AuditError("duplicate utterance_id")
        utterance_ids.add(record["utterance_id"])
        audio_path = _path_below_root(record["audio_path"], root)
        if not audio_path.is_file():
            raise AuditError("audio file is missing")
        prior_split = speaker_splits.setdefault(record["speaker_id"], record["split"])
        if prior_split != record["split"]:
            raise AuditError("speaker leakage across splits")
        checked.append(record)
    return checked, root


def compute_pair_capacity(records: Iterable[Mapping[str, Any]]) -> dict[str, int]:
    """Count A/B/C/D unordered-pair capacity using one pass over records."""
    by_speaker: Counter[str] = Counter()
    by_dialect: Counter[str] = Counter()
    by_speaker_dialect: Counter[tuple[str, str]] = Counter()
    total = 0
    for record in records:
        speaker = record.get("speaker_id")
        dialect = record.get("dialect_label")
        if not isinstance(speaker, str) or not isinstance(dialect, str):
            raise AuditError("malformed record for pair capacity")
        total += 1
        by_speaker[speaker] += 1
        by_dialect[dialect] += 1
        by_speaker_dialect[(speaker, dialect)] += 1
    choose_two = lambda count: count * (count - 1) // 2
    group_a = sum(choose_two(count) for count in by_speaker_dialect.values())
    group_b = sum(choose_two(count) for count in by_dialect.values()) - group_a
    group_c = sum(choose_two(count) for count in by_speaker.values()) - group_a
    group_d = choose_two(total) - group_a - group_b - group_c
    return {"A": group_a, "B": group_b, "C": group_c, "D": group_d}


def _bounded_speaker_diagnostics(records: list[Mapping[str, Any]]) -> dict[str, Any]:
    speakers = sorted({str(record["speaker_id"]) for record in records})
    return {
        "speaker_count": len(speakers),
        "sample_speaker_ids": speakers[:10],
        "sample_limit": 10,
        "global_mixed_effect": {"status": "pending", "reason": "speaker-distance rows unavailable"},
        "per_dialect": {},
        "status": "pending",
    }


def audit_manifest(
    records: Iterable[Mapping[str, Any]],
    audio_root: str | Path,
    *,
    legally_usable_for_research: bool | None = None,
    raw_audio_redistributable: bool | None = None,
    minimum_dialects: int = 8,
    minimum_pairs: int = 200,
) -> dict[str, Any]:
    """Validate a manifest and return a path-redacted, fail-closed audit report."""
    checked, _ = _validate_records(records, audio_root)
    if not isinstance(legally_usable_for_research, bool):
        raise AuditError("legally_usable_for_research must be boolean")
    if not isinstance(raw_audio_redistributable, bool):
        raise AuditError("raw_audio_redistributable must be boolean")
    if minimum_dialects <= 0 or minimum_pairs <= 0:
        raise AuditError("audit thresholds must be positive")

    capacities = compute_pair_capacity(checked)
    dialect_count = len({record["dialect_label"] for record in checked})
    failed_checks: list[str] = []
    if dialect_count < minimum_dialects:
        failed_checks.append("dialect_label_count")
    if any(value < minimum_pairs for value in capacities.values()):
        failed_checks.append("pair_capacity")
    if not legally_usable_for_research:
        failed_checks.append("legally_usable_for_research")
    if legally_usable_for_research and raw_audio_redistributable:
        classification = "permitted"
    elif legally_usable_for_research:
        classification = "research_only"
    else:
        classification = "pending_audit"
    return {
        "schema": "corpus-audit-v1",
        "status": "passed" if not failed_checks else "failed",
        "local_input_path": "<local>",
        "record_count": len(checked),
        "speaker_count": len({record["speaker_id"] for record in checked}),
        "dialect_count": dialect_count,
        "pair_capacity": capacities,
        "thresholds": {"minimum_dialects": minimum_dialects, "minimum_pairs": minimum_pairs},
        "legally_usable_for_research": legally_usable_for_research,
        "raw_audio_redistributable": raw_audio_redistributable,
        "raw_audio_classification": classification,
        "speaker_label_consistency": _bounded_speaker_diagnostics(checked),
        "failed_checks": failed_checks,
    }


def _quantile(values: list[float], probability: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * probability
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def hierarchical_bootstrap_effect(
    rows: Iterable[Mapping[str, Any]], *, replicates: int, seed: int
) -> dict[str, Any]:
    """Bootstrap between-minus-within distance over dialect, speaker, then pair."""
    if replicates < 1000:
        raise AuditError("at least 1000 bootstrap replicates are required")
    grouped: dict[str, dict[str, dict[str, list[float]]]] = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    endpoints_by_dialect: dict[str, set[str]] = defaultdict(set)
    for row in rows:
        try:
            dialect, group, distance = row["dialect"], row["group"], row["distance"]
        except (KeyError, TypeError) as exc:
            raise AuditError("malformed bootstrap row") from exc
        if not isinstance(dialect, str) or group not in {"within", "between"}:
            raise AuditError("malformed bootstrap row")
        speaker_ids = row.get("speaker_ids")
        if speaker_ids is None:
            if group == "between":
                raise AuditError("between bootstrap row requires speaker_ids")
            speaker_id = row.get("speaker_id")
            if not isinstance(speaker_id, str) or not speaker_id.strip():
                raise AuditError("bootstrap row requires speaker_ids")
            speaker_ids = [speaker_id]
        if not isinstance(speaker_ids, (list, tuple)) or not speaker_ids or not all(
            isinstance(value, str) and value.strip() for value in speaker_ids
        ):
            raise AuditError("malformed bootstrap speaker_ids")
        if group == "within" and len(speaker_ids) != 1:
            raise AuditError("within bootstrap row requires one speaker endpoint")
        if group == "between" and len(speaker_ids) != 2:
            raise AuditError("between bootstrap row requires two speaker endpoints")
        speaker_ids = tuple(speaker_ids)
        endpoints_by_dialect[dialect].update(speaker_ids)
        if isinstance(distance, bool) or not isinstance(distance, (int, float)) or not math.isfinite(distance):
            raise AuditError("malformed bootstrap distance")
        # A between pair contributes to both endpoint clusters. Sorting makes
        # the representation invariant to the pair's input orientation.
        for speaker in speaker_ids:
            grouped[dialect][speaker][group].append(float(distance))

    usable = {
        dialect: {speaker: groups for speaker, groups in speakers.items() if groups["within"] and groups["between"]}
        for dialect, speakers in grouped.items()
    }
    usable = {dialect: speakers for dialect, speakers in usable.items() if speakers}
    if not usable:
        raise AuditError("no valid dialect-speaker bootstrap clusters")
    per_dialect = {
        dialect: {
            "speaker_count": len(speakers),
            "speaker_endpoint_count": len(endpoints_by_dialect[dialect]),
            "pair_count": sum(len(groups["within"]) + len(groups["between"]) for groups in speakers.values()),
        }
        for dialect, speakers in sorted(usable.items())
    }

    rng = random.Random(seed)
    dialects = tuple(sorted(usable))
    estimates: list[float] = []
    for _ in range(replicates):
        effects: list[float] = []
        for dialect in (rng.choice(dialects) for _ in dialects):
            speakers = tuple(sorted(usable[dialect]))
            for speaker in (rng.choice(speakers) for _ in speakers):
                groups = usable[dialect][speaker]
                within = [rng.choice(groups["within"]) for _ in groups["within"]]
                between = [rng.choice(groups["between"]) for _ in groups["between"]]
                effects.append(sum(between) / len(between) - sum(within) / len(within))
        estimate = sum(effects) / len(effects)
        if not math.isfinite(estimate):
            raise AuditError("non-finite bootstrap replicate")
        estimates.append(estimate)
    return {
        "effective_replicates": len(estimates),
        "replicates": estimates,
        "ci": {"lower": _quantile(estimates, 0.025), "upper": _quantile(estimates, 0.975), "confidence_level": 0.95},
        "per_dialect": per_dialect,
    }
