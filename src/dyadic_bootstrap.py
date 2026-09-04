"""Dyadic endpoint sensitivity bootstrap for matched A/B distance contrasts."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from statistics import median
from typing import Any, Iterable, Mapping, Sequence

import numpy as np


def endpoint_multiplicity(
    speaker_ids: Sequence[str], multiplicities: Mapping[str, int]
) -> int:
    """Return one factor for a self pair and the product for a dyad."""
    endpoints = tuple(dict.fromkeys(map(str, speaker_ids)))
    if not endpoints:
        return 0
    product = 1
    for endpoint in endpoints:
        product *= int(multiplicities.get(endpoint, 0))
    return product


def _weighted_median(values: Sequence[float], weights: Sequence[int]) -> float:
    positive = sorted(
        (float(value), int(weight))
        for value, weight in zip(values, weights)
        if int(weight) > 0
    )
    if not positive:
        raise ValueError("weighted median has no positive weight")
    total = sum(weight for _, weight in positive)
    threshold = total / 2.0
    cumulative = 0
    for value, weight in positive:
        cumulative += weight
        if cumulative >= threshold:
            return value
    return positive[-1][0]


def _checked_rows(rows: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
    checked = [dict(row) for row in rows]
    if not checked:
        raise ValueError("at least one row is required")
    for row in checked:
        if row.get("group") not in {"A", "B"}:
            raise ValueError("groups must be A or B")
        if not row.get("speaker_ids") or not row.get("matched_stratum"):
            raise ValueError("speaker_ids and matched_stratum are required")
        row["distance"] = float(row["distance"])
    return checked


def _observed_effect(rows: Sequence[Mapping[str, Any]]) -> float:
    strata: dict[str, dict[str, list[float]]] = defaultdict(
        lambda: {"A": [], "B": []}
    )
    for row in rows:
        strata[str(row["matched_stratum"])][str(row["group"])].append(
            float(row["distance"])
        )
    contrasts = [
        median(arms["B"]) - median(arms["A"])
        for arms in strata.values()
        if arms["A"] and arms["B"]
    ]
    if not contrasts:
        raise ValueError("no matched A/B stratum")
    return float(median(contrasts))


def _quantiles(values: Sequence[float], probs: Sequence[float]) -> list[float]:
    if not values:
        return [0.0 for _ in probs]
    return [float(value) for value in np.quantile(values, probs, method="linear")]


def dyadic_ab_bootstrap(
    rows: Iterable[Mapping[str, Any]], *, seed: int, replicates: int
) -> dict[str, Any]:
    """Resample speaker endpoints within strata and recompute the A/B estimand."""
    if replicates <= 0:
        raise ValueError("replicates must be positive")
    checked = _checked_rows(rows)
    by_stratum: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in checked:
        by_stratum[str(row["matched_stratum"])].append(row)
    rng = np.random.default_rng(seed)
    estimates: list[float] = []
    effective_counts: dict[str, list[int]] = {"A": [], "B": []}
    zero_effective = 0

    for _ in range(replicates):
        stratum_effects: list[float] = []
        replicate_counts = {"A": 0, "B": 0}
        for stratum_rows in by_stratum.values():
            speakers = sorted(
                {
                    str(speaker)
                    for row in stratum_rows
                    for speaker in row["speaker_ids"]
                }
            )
            sampled = rng.choice(speakers, size=len(speakers), replace=True)
            multiplicities = Counter(map(str, sampled.tolist()))
            arm_values: dict[str, list[float]] = {"A": [], "B": []}
            arm_weights: dict[str, list[int]] = {"A": [], "B": []}
            for row in stratum_rows:
                group = str(row["group"])
                weight = endpoint_multiplicity(row["speaker_ids"], multiplicities)
                if weight > 0:
                    arm_values[group].append(float(row["distance"]))
                    arm_weights[group].append(weight)
                    replicate_counts[group] += weight
            if arm_values["A"] and arm_values["B"]:
                stratum_effects.append(
                    _weighted_median(arm_values["B"], arm_weights["B"])
                    - _weighted_median(arm_values["A"], arm_weights["A"])
                )
        for group in ("A", "B"):
            effective_counts[group].append(replicate_counts[group])
        if stratum_effects:
            estimates.append(float(median(stratum_effects)))
        else:
            zero_effective += 1

    if not estimates:
        raise ValueError("all bootstrap replicates lost a matched A/B stratum")
    ci_lower, ci_upper = _quantiles(estimates, (0.025, 0.975))
    return {
        "schema": "dyadic-ab-bootstrap-v1",
        "point_estimate": _observed_effect(checked),
        "ci": {
            "lower": ci_lower,
            "upper": ci_upper,
            "confidence_level": 0.95,
        },
        "replicates_requested": replicates,
        "replicates_used": len(estimates),
        "zero_effective_arm_replicates": zero_effective,
        "effective_pair_count_quantiles": {
            group: _quantiles(values, (0.025, 0.5, 0.975))
            for group, values in effective_counts.items()
        },
        "seed": seed,
        "estimand": "median_of_stratum_weighted_median_B_minus_A",
        "resampling_unit": "speaker_endpoints_within_matched_stratum",
    }


def build_report(distance_root: Path, *, seed: int, replicates: int) -> dict[str, Any]:
    models = []
    for path in sorted(distance_root.glob("*.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        rows = payload.get("distances")
        if not isinstance(rows, list):
            continue
        result = dyadic_ab_bootstrap(rows, seed=seed, replicates=replicates)
        result["model_name"] = payload.get("model_name", path.stem)
        result["source_file"] = str(path)
        models.append(result)
    if not models:
        raise ValueError(f"no distance reports found in {distance_root}")
    return {
        "schema": "speaker-effect-dyadic-sensitivity-v1",
        "status": "evaluated",
        "models": models,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--distance-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260829)
    parser.add_argument("--replicates", type=int, default=1000)
    args = parser.parse_args()
    report = build_report(args.distance_root, seed=args.seed, replicates=args.replicates)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
