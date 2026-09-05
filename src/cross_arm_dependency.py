"""Audit and block-bootstrap cross-arm utterance dependence."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import yaml

from .dyadic_bootstrap import endpoint_multiplicity
from .speaker_effect_sensitivity import overlap_pair_weight, stratum_effects


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def audit_cross_arm_dependencies(
    pair_rows: Iterable[Mapping[str, Any]],
    record_rows: Iterable[Mapping[str, Any]],
    *,
    expected_shared: int,
) -> dict[str, Any]:
    """Verify global speaker and utterance identity across matched A/B arms."""
    pairs = [dict(row) for row in pair_rows]
    record_by_utterance: dict[str, dict[str, Any]] = {}
    for raw_record in record_rows:
        record = dict(raw_record)
        utterance = str(record.get("utterance_id", ""))
        if not utterance or utterance in record_by_utterance:
            raise ValueError("record utterance_id must be nonempty and unique")
        if not record.get("speaker_id") or not record.get("dialect_label"):
            raise ValueError("record speaker_id and dialect_label are required")
        record_by_utterance[utterance] = record

    by_arm: dict[str, list[str]] = {"A": [], "B": []}
    speaker_arms: dict[str, set[str]] = defaultdict(set)
    speaker_strata: dict[str, set[str]] = defaultdict(set)
    occurrence_metadata: dict[str, list[tuple[str, str, str]]] = defaultdict(list)
    for pair in pairs:
        group = str(pair.get("group", ""))
        if group not in by_arm:
            raise ValueError("dependency audit accepts only A/B pairs")
        utterances = pair.get("utterance_ids")
        if not isinstance(utterances, list) or len(utterances) != 2:
            raise ValueError("every pair must have two utterance_ids")
        stratum = str(pair.get("matched_stratum", ""))
        if not stratum:
            raise ValueError("matched_stratum is required")
        observed_speakers = set()
        observed_dialects = set()
        for raw_utterance in utterances:
            utterance = str(raw_utterance)
            if utterance not in record_by_utterance:
                raise ValueError(f"pair utterance missing from records: {utterance}")
            record = record_by_utterance[utterance]
            speaker = str(record["speaker_id"])
            dialect = str(record["dialect_label"])
            observed_speakers.add(speaker)
            observed_dialects.add(dialect)
            by_arm[group].append(utterance)
            speaker_arms[speaker].add(group)
            speaker_strata[speaker].add(stratum)
            occurrence_metadata[utterance].append((speaker, dialect, group))
        if set(map(str, pair.get("speaker_ids", []))) != observed_speakers:
            raise ValueError("pair speaker_ids disagree with record metadata")
        if set(map(str, pair.get("dialect_labels", []))) != observed_dialects:
            raise ValueError("pair dialect_labels disagree with record metadata")

    arm_sets = {group: set(values) for group, values in by_arm.items()}
    shared = sorted(arm_sets["A"].intersection(arm_sets["B"]))
    if len(shared) != expected_shared:
        raise ValueError(
            f"shared utterance count drift: expected {expected_shared}, got {len(shared)}"
        )
    nested_speaker = all(
        len({speaker for speaker, _, _ in occurrence_metadata[utterance]}) == 1
        for utterance in shared
    )
    nested_dialect = all(
        len({dialect for _, dialect, _ in occurrence_metadata[utterance]}) == 1
        for utterance in shared
    )
    return {
        "schema": "cross-arm-dependency-audit-v1",
        "pair_count": len(pairs),
        "cross_arm_shared_utterance_count": len(shared),
        "shared_utterance_id_sha256": hashlib.sha256(
            "\n".join(shared).encode("utf-8")
        ).hexdigest(),
        "shared_utterances_uniquely_nested_in_speaker": nested_speaker,
        "shared_utterances_uniquely_nested_in_dialect": nested_dialect,
        "nesting_status": (
            "utterance_uniquely_nested_in_speaker_and_dialect"
            if nested_speaker and nested_dialect
            else "non_nested"
        ),
        "within_arm_utterance_reuse": {
            group: len(values) - len(set(values)) for group, values in by_arm.items()
        },
        "arm_utterance_occurrences": {
            group: len(values) for group, values in by_arm.items()
        },
        "arm_unique_utterances": {
            group: len(set(values)) for group, values in by_arm.items()
        },
        "speakers_in_multiple_arms": sorted(
            speaker for speaker, arms in speaker_arms.items() if len(arms) > 1
        ),
        "speakers_in_multiple_strata": sorted(
            speaker
            for speaker, strata in speaker_strata.items()
            if len(strata) > 1
        ),
        "cluster_identity": {
            "speaker": "global speaker_id",
            "utterance": "global utterance_id",
        },
        "covariance_units": {
            "primary_speaker_interval": "global speaker_id across arms and strata",
            "utterance_sensitivity_interval": "global utterance_id across arms and strata",
            "arm_independent_diagnostic": "arm-specific utterance_id; not used for inference",
        },
    }


def _weighted_median(values: Sequence[float], weights: Sequence[int]) -> float:
    ordered = sorted(
        (float(value), int(weight))
        for value, weight in zip(values, weights)
        if int(weight) > 0
    )
    if not ordered:
        raise ValueError("weighted median has no positive support")
    cutoff = sum(weight for _, weight in ordered) / 2.0
    cumulative = 0
    for value, weight in ordered:
        cumulative += weight
        if cumulative >= cutoff:
            return value
    return ordered[-1][0]


def _checked_distance_rows(
    rows: Iterable[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    checked = [dict(row) for row in rows]
    if not checked:
        raise ValueError("at least one distance row is required")
    for row in checked:
        if row.get("group") not in {"A", "B"}:
            raise ValueError("groups must be A or B")
        if not row.get("matched_stratum"):
            raise ValueError("matched_stratum is required")
        if not isinstance(row.get("utterance_ids"), list) or not row["utterance_ids"]:
            raise ValueError("utterance_ids are required")
        if not isinstance(row.get("speaker_ids"), list) or not row["speaker_ids"]:
            raise ValueError("speaker_ids are required")
        row["utterance_ids"] = [str(value) for value in row["utterance_ids"]]
        row["speaker_ids"] = [str(value) for value in row["speaker_ids"]]
        row["distance"] = float(row["distance"])
    return checked


def utterance_block_bootstrap(
    rows: Iterable[Mapping[str, Any]],
    *,
    seed: int,
    replicates: int,
    share_cross_arm_clusters: bool = True,
) -> dict[str, Any]:
    """Bootstrap utterance endpoints, sharing IDs across arms when requested."""
    if replicates <= 0:
        raise ValueError("replicates must be positive")
    checked = _checked_distance_rows(rows)
    grouped: dict[str, dict[str, list[dict[str, Any]]]] = defaultdict(
        lambda: {"A": [], "B": []}
    )
    for row in checked:
        grouped[str(row["matched_stratum"])][str(row["group"])].append(row)
    if any(not arms["A"] or not arms["B"] for arms in grouped.values()):
        raise ValueError("every stratum must contain both A and B arms")
    fixed_weights = {
        stratum: overlap_pair_weight(len(arms["A"]), len(arms["B"]))
        for stratum, arms in grouped.items()
    }
    by_arm_units = {
        group: sorted(
            {
                utterance
                for row in checked
                if row["group"] == group
                for utterance in row["utterance_ids"]
            }
        )
        for group in ("A", "B")
    }
    global_units = sorted(set(by_arm_units["A"]).union(by_arm_units["B"]))
    rng = np.random.default_rng(seed)
    estimates = []
    missing: Counter[str] = Counter()
    for _ in range(replicates):
        if share_cross_arm_clusters:
            draw = rng.choice(global_units, size=len(global_units), replace=True)
            global_counts = Counter(map(str, draw.tolist()))
            counts_by_arm = {"A": global_counts, "B": global_counts}
        else:
            counts_by_arm = {}
            for group in ("A", "B"):
                units = by_arm_units[group]
                draw = rng.choice(units, size=len(units), replace=True)
                counts_by_arm[group] = Counter(map(str, draw.tolist()))

        effects = []
        for stratum, arms in grouped.items():
            arm_medians = {}
            for group in ("A", "B"):
                values = []
                weights = []
                for row in arms[group]:
                    weight = endpoint_multiplicity(
                        row["utterance_ids"], counts_by_arm[group]
                    )
                    if weight > 0:
                        values.append(float(row["distance"]))
                        weights.append(weight)
                if values:
                    arm_medians[group] = _weighted_median(values, weights)
            if set(arm_medians) == {"A", "B"}:
                effects.append(
                    (
                        arm_medians["B"] - arm_medians["A"],
                        fixed_weights[stratum],
                    )
                )
            else:
                missing[stratum] += 1
        if effects:
            total = sum(weight for _, weight in effects)
            estimates.append(sum(effect * weight for effect, weight in effects) / total)

    if not estimates:
        raise ValueError("all utterance bootstrap replicates lost matched support")
    lower, upper = np.quantile(estimates, [0.025, 0.975], method="linear")
    observed = stratum_effects(checked)
    point = sum(row["weighted_effect_contribution"] for row in observed) / sum(
        row["overlap_weight"] for row in observed
    )
    return {
        "schema": "utterance-block-bootstrap-v1",
        "point_estimate": float(point),
        "ci": {
            "lower": float(lower),
            "upper": float(upper),
            "confidence_level": 0.95,
            "method": "percentile",
        },
        "bootstrap_estimate_standard_deviation": float(
            np.std(estimates, ddof=1) if len(estimates) > 1 else 0.0
        ),
        "replicates_requested": replicates,
        "replicates_used": len(estimates),
        "seed": seed,
        "cross_arm_cluster_identity": (
            "global utterance_id" if share_cross_arm_clusters else "arm-specific utterance_id"
        ),
        "pair_multiplier": "product of unique utterance endpoint multiplicities",
        "stratum_medians_recomputed_per_replicate": True,
        "stratum_weights": "fixed observed overlap-pair weights",
        "missing_stratum_replicates": dict(sorted(missing.items())),
    }


def build_report(
    protocol_path: Path,
    pair_path: Path,
    records_path: Path,
    distance_dir: Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    protocol = yaml.safe_load(protocol_path.read_text(encoding="utf-8"))
    if protocol.get("status") != "locked_before_new_runs":
        raise ValueError("protocol is not locked")
    pairs_payload = json.loads(pair_path.read_text(encoding="utf-8"))
    records_payload = json.loads(records_path.read_text(encoding="utf-8"))
    expected_shared = int(protocol["dependence"]["shared_utterance_expected_count"])
    audit = audit_cross_arm_dependencies(
        pairs_payload["pairs"], records_payload["records"], expected_shared=expected_shared
    )
    if audit["nesting_status"] != "utterance_uniquely_nested_in_speaker_and_dialect":
        raise ValueError("non-nested utterances require a locked crossed-bootstrap amendment")
    if any(audit["within_arm_utterance_reuse"].values()):
        raise ValueError("within-arm utterance reuse violates the locked design")

    pair_ids = {str(row["pair_id"]) for row in pairs_payload["pairs"]}
    model_names = [
        str(protocol["primary_phenomenon"]["model"]),
        *map(str, protocol["supporting_phenomenon_family"]["models"]),
    ]
    seed = int(protocol["seeds"][0])
    replicates = int(protocol["bootstrap_replicates"])
    models = []
    for model_name in model_names:
        path = distance_dir / f"{model_name}.json"
        payload = json.loads(path.read_text(encoding="utf-8"))
        rows = payload["distances"]
        if {str(row["pair_id"]) for row in rows} != pair_ids:
            raise ValueError(f"pair identity mismatch for {model_name}")
        blocked = utterance_block_bootstrap(
            rows, seed=seed, replicates=replicates, share_cross_arm_clusters=True
        )
        independent = utterance_block_bootstrap(
            rows, seed=seed, replicates=replicates, share_cross_arm_clusters=False
        )
        models.append(
            {
                "model_name": model_name,
                "source_file": str(path),
                "source_sha256": _sha256(path),
                "utterance_block": blocked,
                "arm_independent_diagnostic": independent,
                "direction_preserved": blocked["point_estimate"] > 0,
                "interval_lower_positive": blocked["ci"]["lower"] > 0,
            }
        )

    direction_pass = all(row["direction_preserved"] for row in models)
    interval_pass = all(row["interval_lower_positive"] for row in models)
    report = {
        "schema": "speaker-effect-dependency-sensitivity-v1",
        "status": "evaluated",
        "declaration_status": "reviewer_driven_revision_not_preregistered",
        "protocol": str(protocol_path),
        "protocol_sha256": _sha256(protocol_path),
        "pair_manifest": str(pair_path),
        "pair_manifest_sha256": _sha256(pair_path),
        "records_manifest": str(records_path),
        "records_manifest_sha256": _sha256(records_path),
        "identity_audit": audit,
        "models": models,
    }
    gate = {
        "schema": "cross-arm-dependency-gate-v1",
        "status": "passed" if direction_pass else "failed",
        "direction_preserved_all_models": direction_pass,
        "interval_lower_positive_all_models": interval_pass,
        "shared_utterance_count_matches_lock": (
            audit["cross_arm_shared_utterance_count"] == expected_shared
        ),
        "nested_utterance_sensitivity_valid": True,
        "full_title_dependency_passed": direction_pass,
    }
    return report, gate


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--pairs", type=Path, required=True)
    parser.add_argument("--records", type=Path, required=True)
    parser.add_argument("--distance-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--gate-output", type=Path, required=True)
    args = parser.parse_args()
    report, gate = build_report(
        args.protocol, args.pairs, args.records, args.distance_dir
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
