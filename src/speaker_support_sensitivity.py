"""Speaker-support and incidence-ESS sensitivities for the A/B contrast."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from statistics import median
from typing import Any, Iterable, Mapping, Sequence

import numpy as np


def speaker_overlap_weight(s_a: int, s_b: int) -> float:
    if s_a <= 0 or s_b <= 0:
        raise ValueError("both speaker supports must be positive")
    return float(s_a * s_b / (s_a + s_b))


def incidence_ess(counts: Sequence[int]) -> float:
    values = [float(value) for value in counts if float(value) > 0]
    if not values:
        raise ValueError("incidence counts must contain a positive value")
    return float(sum(values) ** 2 / sum(value * value for value in values))


def _rows_by_stratum(rows: Iterable[Mapping[str, Any]]) -> dict[str, dict[str, list[Mapping[str, Any]]]]:
    grouped: dict[str, dict[str, list[Mapping[str, Any]]]] = defaultdict(lambda: {"A": [], "B": []})
    for row in rows:
        group = str(row.get("group", ""))
        if group not in {"A", "B"}:
            continue
        stratum = str(row.get("matched_stratum", ""))
        if not stratum:
            raise ValueError("matched_stratum is required")
        grouped[stratum][group].append(row)
    if not grouped or any(not arms["A"] or not arms["B"] for arms in grouped.values()):
        raise ValueError("every stratum must contain both A and B arms")
    return dict(grouped)


def summarize_support(rows: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Return pair, speaker, and incidence-ESS support for every stratum."""
    grouped = _rows_by_stratum(rows)
    output = []
    for stratum in sorted(grouped):
        arms = grouped[stratum]
        support: dict[str, Any] = {"stratum": stratum}
        weights: dict[str, float] = {}
        for group in ("A", "B"):
            incident = Counter(
                str(speaker)
                for row in arms[group]
                for speaker in dict.fromkeys(row.get("speaker_ids", []))
            )
            pair_count = len(arms[group])
            speaker_count = len(incident)
            ess = incidence_ess(list(incident.values()))
            support[group] = {
                "pair_count": pair_count,
                "unique_speaker_count": speaker_count,
                "incidence_counts": dict(sorted(incident.items())),
                "incidence_ess": ess,
            }
            weights[group] = float(pair_count)
        support["w_pair"] = weights["A"] * weights["B"] / (weights["A"] + weights["B"])
        support["w_speaker"] = speaker_overlap_weight(
            support["A"]["unique_speaker_count"], support["B"]["unique_speaker_count"]
        )
        support["w_ess"] = speaker_overlap_weight(
            support["A"]["incidence_ess"], support["B"]["incidence_ess"]
        )
        output.append(support)
    return output


def aggregate_support_effects(
    summaries: Sequence[Mapping[str, Any]],
    effects: Mapping[str, float],
    weight_key: str,
) -> float:
    if weight_key not in {"w_pair", "w_speaker", "w_ess"}:
        raise ValueError("unknown support weight")
    weighted = [
        (float(row[weight_key]), float(effects[str(row["stratum"])]))
        for row in summaries
        if float(row[weight_key]) > 0
    ]
    if not weighted:
        raise ValueError("no positive support")
    denominator = sum(weight for weight, _ in weighted)
    return float(sum(weight * effect for weight, effect in weighted) / denominator)


def _weighted_median(values: Sequence[float], weights: Sequence[int]) -> float:
    ordered = sorted((float(value), int(weight)) for value, weight in zip(values, weights) if int(weight) > 0)
    if not ordered:
        raise ValueError("no positive bootstrap support")
    cutoff = sum(weight for _, weight in ordered) / 2.0
    cumulative = 0
    for value, weight in ordered:
        cumulative += weight
        if cumulative >= cutoff:
            return value
    return ordered[-1][0]


def bootstrap_support_estimands(
    rows: Sequence[Mapping[str, Any]], *, seed: int, replicates: int
) -> dict[str, Any]:
    """Bootstrap support-weighted effects by global endpoint speaker."""
    if replicates <= 0:
        raise ValueError("replicates must be positive")
    checked = list(rows)
    grouped = _rows_by_stratum(checked)
    speakers = sorted({str(speaker) for row in checked for speaker in row["speaker_ids"]})
    rng = np.random.default_rng(seed)
    values = {key: [] for key in ("w_pair", "w_speaker", "w_ess")}
    for _ in range(replicates):
        draws = rng.choice(speakers, size=len(speakers), replace=True)
        multiplicity = Counter(map(str, draws.tolist()))
        effects: dict[str, float] = {}
        supports: list[dict[str, Any]] = []
        for stratum, arms in grouped.items():
            arm_medians: dict[str, float] = {}
            active_rows: dict[str, list[Mapping[str, Any]]] = {"A": [], "B": []}
            for group in ("A", "B"):
                weighted_values = []
                weighted_rows = []
                for row in arms[group]:
                    unique = list(dict.fromkeys(map(str, row["speaker_ids"])))
                    weight = multiplicity.get(unique[0], 0) if len(unique) == 1 else multiplicity.get(unique[0], 0) * multiplicity.get(unique[-1], 0)
                    if weight > 0:
                        weighted_values.append(float(row["distance"]))
                        weighted_rows.append((row, weight))
                if weighted_values:
                    arm_medians[group] = _weighted_median(weighted_values, [weight for _, weight in weighted_rows])
                    active_rows[group] = [row for row, _ in weighted_rows]
            if set(arm_medians) != {"A", "B"}:
                continue
            effects[stratum] = arm_medians["B"] - arm_medians["A"]
            summary = summarize_support(active_rows["A"] + active_rows["B"])
            supports.extend(summary)
        if not supports:
            continue
        for key in values:
            values[key].append(aggregate_support_effects(supports, effects, key))
    result = {}
    for key, sample in values.items():
        if not sample:
            raise ValueError(f"no finite bootstrap estimates for {key}")
        lower, upper = np.quantile(sample, [0.025, 0.975])
        result[key] = {
            "estimate": float(np.median(sample)),
            "ci": {"lower": float(lower), "upper": float(upper), "confidence_level": 0.95, "method": "percentile"},
            "replicates_requested": replicates,
            "replicates_used": len(sample),
            "resampling_unit": "global_endpoint_speaker",
        }
    return result


def build_support_report(rows: Sequence[Mapping[str, Any]], *, excluded_stratum: str | None = None, bootstrap_seed: int | None = None, bootstrap_replicates: int = 1000) -> dict[str, Any]:
    summaries = summarize_support(rows)
    effect_rows = {}
    grouped = _rows_by_stratum(rows)
    for stratum, arms in grouped.items():
        effect_rows[stratum] = float(
            median(float(row["distance"]) for row in arms["B"])
            - median(float(row["distance"]) for row in arms["A"])
        )
    kept = [row for row in summaries if str(row["stratum"]) != excluded_stratum]
    result = {
        "schema": "speaker-support-sensitivity-v1",
        "status": "evaluated",
        "weight_definitions": {
            "w_pair": "n_A*n_B/(n_A+n_B)",
            "w_speaker": "s_A*s_B/(s_A+s_B)",
            "w_ess": "ESS_A*ESS_B/(ESS_A+ESS_B)",
            "incidence_ess": "(sum_s k_s)^2/sum_s k_s^2",
        },
        "strata": summaries,
        "effects": effect_rows,
        "primary": {
            "weight": "w_pair",
            "estimate": aggregate_support_effects(summaries, effect_rows, "w_pair"),
        },
        "sensitivities": {
            key: aggregate_support_effects(summaries, effect_rows, key)
            for key in ("w_speaker", "w_ess")
        },
        "excluded_stratum": excluded_stratum,
        "excluded_primary": (
            aggregate_support_effects(kept, effect_rows, "w_pair") if kept else None
        ),
        "M-A_visible": any("Mandarin|phase1:Accent" in str(row["stratum"]) for row in summaries),
        "stratum_count": len(summaries),
    }
    result["direction_agreement"] = all(
        (result["primary"]["estimate"] > 0) == (value > 0)
        for value in result["sensitivities"].values()
    )
    if bootstrap_seed is not None:
        result["clustered_intervals"] = bootstrap_support_estimands(
            list(rows), seed=bootstrap_seed, replicates=bootstrap_replicates
        )
    return result


def build_multi_model_support_report(distance_dir: Path, model_names: Sequence[str], *, excluded_stratum: str | None = None, bootstrap_seed: int | None = None, bootstrap_replicates: int = 1000) -> dict[str, Any]:
    models = []
    for model_name in model_names:
        path = distance_dir / f"{model_name}.json"
        payload = json.loads(path.read_text(encoding="utf-8"))
        model_report = build_support_report(payload["distances"], excluded_stratum=excluded_stratum, bootstrap_seed=bootstrap_seed, bootstrap_replicates=bootstrap_replicates)
        model_report["model_name"] = model_name
        model_report["source"] = str(path)
        models.append(model_report)
    primary = models[0] if models else None
    return {
        "schema": "speaker-support-sensitivity-v1",
        "status": "evaluated",
        "models": models,
        "stratum_count": primary["stratum_count"] if primary else 0,
        "M-A_visible": all(model["M-A_visible"] for model in models),
        "direction_agreement": all(model["direction_agreement"] for model in models),
        "excluded_stratum": excluded_stratum,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--distances", type=Path)
    parser.add_argument("--distance-dir", type=Path)
    parser.add_argument("--models", nargs="*", default=[])
    parser.add_argument("--bootstrap-seed", type=int)
    parser.add_argument("--bootstrap-replicates", type=int, default=1000)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--gate", type=Path, required=True)
    parser.add_argument("--exclude", default="evaluation|Mandarin|phase1:Accent")
    args = parser.parse_args()
    if args.distance_dir:
        report = build_multi_model_support_report(
            args.distance_dir,
            args.models,
            excluded_stratum=args.exclude,
            bootstrap_seed=args.bootstrap_seed,
            bootstrap_replicates=args.bootstrap_replicates,
        )
    elif args.distances:
        payload = json.loads(args.distances.read_text(encoding="utf-8"))
        report = build_support_report(payload["distances"], excluded_stratum=args.exclude, bootstrap_seed=args.bootstrap_seed, bootstrap_replicates=args.bootstrap_replicates)
        report["source"] = str(args.distances)
    else:
        parser.error("one of --distances or --distance-dir is required")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.gate.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    gate = {
        "schema": "speaker-support-gate-v1",
        "status": "passed" if report["direction_agreement"] and report["M-A_visible"] else "failed",
        "stratum_count": report["stratum_count"],
        "M-A_visible": report["M-A_visible"],
        "direction_agreement": report["direction_agreement"],
        "primary": report.get("primary"),
        "sensitivities": report.get("sensitivities"),
        "models": [
            {
                "model_name": model["model_name"],
                "primary": model["primary"],
                "sensitivities": model["sensitivities"],
                "excluded_primary": model["excluded_primary"],
            }
            for model in report.get("models", [])
        ],
    }
    args.gate.write_text(json.dumps(gate, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
