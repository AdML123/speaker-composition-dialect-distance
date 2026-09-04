"""Cluster-aware inference for matched A/B distance comparisons.

The inferential unit is a speaker cluster within a matched stratum.  Pair rows
are retained as observations for the estimand, but are never treated as
independent draws when uncertainty is estimated.
"""

from __future__ import annotations

import math
import random
from collections import defaultdict
from statistics import median
from typing import Any, Iterable, Mapping


class ClusterStatisticsError(ValueError):
    """Raised when rows cannot support the declared clustered estimator."""


def _as_rows(rows: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
    checked: list[dict[str, Any]] = []
    for index, raw in enumerate(rows):
        if not isinstance(raw, Mapping):
            raise ClusterStatisticsError(f"row {index} must be a mapping")
        row = dict(raw)
        group = row.get("group")
        if group not in {"A", "B"}:
            raise ClusterStatisticsError(f"row {index} has invalid group")
        speakers = row.get("speaker_ids")
        if not isinstance(speakers, (list, tuple)) or not speakers or any(
            not isinstance(value, str) or not value for value in speakers
        ):
            raise ClusterStatisticsError("speaker_ids are required for clustered inference")
        distance = row.get("distance")
        if isinstance(distance, bool) or not isinstance(distance, (int, float)) or not math.isfinite(float(distance)):
            raise ClusterStatisticsError("distance must be finite")
        stratum = row.get("matched_stratum")
        if not isinstance(stratum, str) or not stratum:
            raise ClusterStatisticsError("matched_stratum is required for clustered inference")
        utterances = row.get("utterance_ids")
        if utterances is not None and (
            not isinstance(utterances, (list, tuple))
            or any(not isinstance(value, str) or not value for value in utterances)
        ):
            raise ClusterStatisticsError("utterance_ids must be a sequence of non-empty strings")
        row["speaker_ids"] = tuple(dict.fromkeys(speakers))
        row["distance"] = float(distance)
        checked.append(row)
    if not checked:
        raise ClusterStatisticsError("at least one row is required")
    return checked


def _strata(rows: list[dict[str, Any]]) -> dict[str, dict[str, list[dict[str, Any]]]]:
    grouped: dict[str, dict[str, list[dict[str, Any]]]] = defaultdict(lambda: {"A": [], "B": []})
    for row in rows:
        grouped[row["matched_stratum"]][row["group"]].append(row)
    usable = {key: value for key, value in grouped.items() if value["A"] and value["B"]}
    if not usable:
        raise ClusterStatisticsError("no matched A/B strata")
    return usable


def _stratum_effect(groups: Mapping[str, list[dict[str, Any]]]) -> float:
    if not groups["A"] or not groups["B"]:
        raise ClusterStatisticsError("each matched stratum needs A and B rows")
    return float(median(row["distance"] for row in groups["B"]) - median(row["distance"] for row in groups["A"]))


def _effect_from_rows(rows: list[dict[str, Any]]) -> float:
    strata = _strata(rows)
    return float(median(_stratum_effect(groups) for groups in strata.values()))


def _speaker_clusters(groups: Mapping[str, list[dict[str, Any]]]) -> list[str]:
    return sorted({speaker for side in groups.values() for row in side for speaker in row["speaker_ids"]})


def clustered_ab_effect(rows: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    """Return the observed B-minus-A effect and cluster bookkeeping."""
    checked = _as_rows(rows)
    strata = _strata(checked)
    clusters = {speaker for row in checked for speaker in row["speaker_ids"]}
    utterance_ids = {
        utterance
        for row in checked
        for utterance in (row.get("utterance_ids") or [])
    }
    return {
        "schema": "clustered-ab-effect-v1",
        "effect": _effect_from_rows(checked),
        "matched_stratum_count": len(strata),
        "speaker_cluster_count": len(clusters),
        "utterance_count": len(utterance_ids),
        "resampling_unit": "speaker_cluster_within_matched_stratum",
        "estimand": "median_of_stratum_median_B_minus_A",
    }


def _bootstrap_sample(rows: list[dict[str, Any]], rng: random.Random) -> list[dict[str, Any]]:
    sampled: list[dict[str, Any]] = []
    for groups in _strata(rows).values():
        # Resample each side separately so a replicate cannot lose one arm of
        # a matched stratum when the two arms share some speaker endpoints.
        for group in ("A", "B"):
            side_rows = groups[group]
            clusters = sorted({speaker for row in side_rows for speaker in row["speaker_ids"]})
            selected = [rng.choice(clusters) for _ in clusters]
            counts = {speaker: selected.count(speaker) for speaker in set(selected)}
            # Draw observations inside each selected speaker.  Pair rows with
            # multiple utterances remain intact, while their within-speaker
            # observation draw is repeated with replacement.
            for speaker, multiplicity in counts.items():
                eligible = [row for row in side_rows if speaker in row["speaker_ids"]]
                for _ in range(multiplicity):
                    sampled.extend(rng.choice(eligible) for _ in range(len(eligible)))
    if not sampled:
        raise ClusterStatisticsError("cluster bootstrap produced no rows")
    return sampled


def _bootstrap_effect_fast(rows: list[dict[str, Any]], rng: random.Random) -> float:
    """Compute the same nested bootstrap estimand from speaker summaries.

    Pair rows are never treated as independent.  Within each stratum and arm,
    speaker-level medians are resampled, and utterance rows are resampled only
    inside a selected speaker.  The summary representation avoids rebuilding
    large row lists for every replicate.
    """
    values: list[float] = []
    for groups in _strata(rows).values():
        arm_values: dict[str, list[float]] = {}
        for group in ("A", "B"):
            by_speaker: dict[str, list[float]] = defaultdict(list)
            for row in groups[group]:
                for speaker in row["speaker_ids"]:
                    by_speaker[speaker].append(row["distance"])
            speakers = sorted(by_speaker)
            sampled_observations: list[float] = []
            for _ in speakers:
                speaker = rng.choice(speakers)
                observations = by_speaker[speaker]
                sampled_observations.extend(rng.choices(observations, k=len(observations)))
            arm_values[group] = sampled_observations
        values.append(float(median(arm_values["B"]) - median(arm_values["A"])))
    return float(median(values))


def clustered_bootstrap(
    rows: Iterable[Mapping[str, Any]],
    *,
    seed: int,
    replicates: int,
) -> dict[str, Any]:
    """Estimate a CI by resampling speaker clusters within matched strata."""
    if replicates < 1000:
        raise ClusterStatisticsError("at least 1000 bootstrap replicates are required")
    checked = _as_rows(rows)
    observed = _effect_from_rows(checked)
    rng = random.Random(seed)
    estimates = [_bootstrap_effect_fast(checked, rng) for _ in range(replicates)]
    ordered = sorted(estimates)

    def quantile(probability: float) -> float:
        position = (len(ordered) - 1) * probability
        lower = math.floor(position)
        upper = math.ceil(position)
        if lower == upper:
            return ordered[lower]
        return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)

    return {
        "schema": "clustered-bootstrap-v1",
        "effect": observed,
        "ci": {"lower": quantile(0.025), "upper": quantile(0.975), "confidence_level": 0.95},
        "bootstrap_replicates": replicates,
        "bootstrap_tail_p_nonpositive": (sum(value <= 0.0 for value in estimates) + 1) / (replicates + 1),
        "seed": seed,
        "resampling": {
            "unit": "speaker_cluster_within_matched_stratum",
            "utterance_within_speaker": any(bool(row.get("utterance_ids")) for row in checked),
            "nested_utterance_sampling": "row observations resampled within selected speaker clusters",
        },
        "matched_stratum_count": len(_strata(checked)),
        "speaker_cluster_count": len({speaker for row in checked for speaker in row["speaker_ids"]}),
    }


def clustered_sign_flip_test(
    rows: Iterable[Mapping[str, Any]],
    *,
    seed: int,
    permutations: int,
) -> dict[str, Any]:
    """Run a cluster-level sign-flip null test for the B-minus-A contrast."""
    if permutations < 1000:
        raise ClusterStatisticsError("at least 1000 permutations are required")
    checked = _as_rows(rows)
    strata = _strata(checked)
    observed = _effect_from_rows(checked)
    cluster_effects: dict[str, list[float]] = {}
    for stratum, groups in strata.items():
        by_speaker: dict[str, dict[str, list[float]]] = defaultdict(lambda: {"A": [], "B": []})
        for group in ("A", "B"):
            for row in groups[group]:
                for speaker in row["speaker_ids"]:
                    by_speaker[speaker][group].append(row["distance"])
        cluster_effects[stratum] = [
            float(median(values["B"]) - median(values["A"]))
            for values in by_speaker.values()
            if values["A"] and values["B"]
        ]
        if not cluster_effects[stratum]:
            raise ClusterStatisticsError("no speaker cluster has observations in both A and B")
    rng = random.Random(seed)
    null: list[float] = []
    for _ in range(permutations):
        stratum_nulls = []
        for effects in cluster_effects.values():
            signed = [effect * (1 if rng.getrandbits(1) else -1) for effect in effects]
            stratum_nulls.append(median(signed))
        null.append(float(median(stratum_nulls)))
    exceed = sum(value >= observed for value in null)
    return {
        "schema": "cluster-sign-flip-v1",
        "observed_effect": observed,
        "null_type": "cluster_sign_flip",
        "exchangeability_unit": "speaker_cluster_within_matched_stratum",
        "permutations": permutations,
        "raw_p": (exceed + 1) / (permutations + 1),
        "seed": seed,
        "matched_stratum_count": len(strata),
        "speaker_cluster_count": sum(len(effects) for effects in cluster_effects.values()),
        "null_summary": {"min": min(null), "max": max(null), "median": median(null)},
    }
