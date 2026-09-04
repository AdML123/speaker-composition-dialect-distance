"""Deterministic schedules and clustered paired resampling for controls."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import random
from collections import defaultdict
from statistics import mean
from typing import Any, Iterable, Mapping, Sequence


@dataclass(frozen=True)
class PairedSchedule:
    """The random and data choices shared by B3, B4, and matched controls."""

    seed: int
    fold_seed: int
    batch_seed: int
    initialization_seed: int
    pair_indices: tuple[int, ...]
    cross_indices: tuple[int, ...]
    fold_assignments: tuple[tuple[str, ...], ...]
    optimizer: Mapping[str, float | int]

    @property
    def schedule_id(self) -> str:
        blob = json.dumps(asdict(self), sort_keys=True, separators=(",", ":"), default=list)
        return hashlib.sha256(blob.encode("utf-8")).hexdigest()

    def to_dict(self) -> dict[str, Any]:
        return {**asdict(self), "schedule_id": self.schedule_id}


def make_paired_schedule(
    *,
    seed: int,
    pair_count: int,
    cross_count: int,
    epochs: int,
    batch_size: int,
    fold_assignments: Sequence[Sequence[str]] = (),
    optimizer: Mapping[str, float | int] | None = None,
) -> PairedSchedule:
    """Create a deterministic schedule independent of the control variant."""
    if min(pair_count, cross_count, epochs, batch_size) < 0:
        raise ValueError("schedule dimensions must be non-negative")
    rng = random.Random(seed)
    n_cross = min(max(batch_size // 4, 1), cross_count) if cross_count else 0
    n_pair = min(batch_size - n_cross, pair_count) if pair_count else 0
    pair_indices = tuple(rng.randrange(pair_count) for _ in range(epochs * n_pair)) if pair_count else ()
    cross_indices = tuple(rng.randrange(cross_count) for _ in range(epochs * n_cross)) if cross_count else ()
    return PairedSchedule(
        seed=seed,
        fold_seed=seed,
        batch_seed=seed,
        initialization_seed=seed,
        pair_indices=pair_indices,
        cross_indices=cross_indices,
        fold_assignments=tuple(tuple(sorted(map(str, fold))) for fold in fold_assignments),
        optimizer=dict(optimizer or {}),
    )


def clustered_paired_bootstrap(
    deltas: Iterable[Mapping[str, Any]],
    *,
    seed: int,
    replicates: int,
) -> dict[str, Any]:
    """Bootstrap paired deltas by speaker cluster within matched stratum."""
    if replicates < 1000:
        raise ValueError("at least 1000 bootstrap replicates are required")
    rows = [dict(row) for row in deltas]
    if not rows:
        raise ValueError("at least one paired delta is required")
    strata: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for row in rows:
        stratum = str(row.get("matched_stratum", ""))
        speakers = row.get("speaker_ids")
        if not stratum or not isinstance(speakers, (list, tuple)) or not speakers:
            raise ValueError("paired delta requires matched_stratum and speaker_ids")
        delta = float(row["delta"])
        # A pair belongs to every endpoint cluster; duplicates are collapsed to
        # keep shared endpoints from receiving an artificial multiplicity.
        for speaker in dict.fromkeys(map(str, speakers)):
            strata[stratum][speaker].append(delta)
    usable = {key: value for key, value in strata.items() if value}
    if not usable:
        raise ValueError("no usable matched strata")
    def stratum_cluster_mean(speakers: Mapping[str, Sequence[float]]) -> float:
        return mean(mean(values) for values in speakers.values())

    observed = mean(stratum_cluster_mean(speakers) for speakers in usable.values())
    rng = random.Random(seed)
    estimates: list[float] = []
    for _ in range(replicates):
        stratum_means = []
        for speakers in usable.values():
            selected = [rng.choice(sorted(speakers)) for _ in speakers]
            cluster_means = []
            for speaker in selected:
                values = speakers[speaker]
                cluster_means.append(mean(rng.choices(values, k=len(values))))
            stratum_means.append(mean(cluster_means))
        estimates.append(mean(stratum_means))
    ordered = sorted(estimates)
    def quantile(p: float) -> float:
        position = (len(ordered) - 1) * p
        lower, upper = int(position), int(position + 1)
        if lower == upper:
            return ordered[lower]
        return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)
    return {
        "schema": "clustered-paired-bootstrap-v1",
        "observed_delta": observed,
        "ci": {"lower": quantile(0.025), "upper": quantile(0.975), "confidence_level": 0.95},
        "bootstrap_tail_p_nonpositive": (
            sum(value <= 0.0 for value in estimates) + 1
        )
        / (replicates + 1),
        "bootstrap_replicates": replicates,
        "seed": seed,
        "resampling_unit": "speaker_cluster_within_matched_stratum",
        "estimand": "equal speaker-cluster means within stratum, then equal stratum means",
        "nested_utterance_sampling": True,
    }
