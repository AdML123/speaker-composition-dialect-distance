"""Primary speaker-effect statistical gate."""

from __future__ import annotations

import argparse
import json
import math
import random
from collections import defaultdict
from pathlib import Path
from statistics import mean, median
from typing import Any, Iterable, Mapping

from .cluster_statistics import (
    ClusterStatisticsError,
    clustered_ab_effect,
    clustered_bootstrap,
    clustered_sign_flip_test,
)


def _quantile(values: list[float], probability: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * probability
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def holm_adjust(p_values: Mapping[str, float]) -> dict[str, float]:
    ordered = sorted(p_values.items(), key=lambda item: item[1])
    adjusted: dict[str, float] = {}
    running = 0.0
    total = len(ordered)
    for rank, (name, p_value) in enumerate(ordered):
        value = min(1.0, (total - rank) * p_value)
        running = max(running, value)
        adjusted[name] = running
    return adjusted


def _dialect_key(row: Mapping[str, Any]) -> str | None:
    labels = row.get("dialect_labels")
    if isinstance(labels, list) and len(labels) == 1 and isinstance(labels[0], str):
        return labels[0]
    return None


def _model_rows(rows: Iterable[Mapping[str, Any]]) -> dict[str, list[Mapping[str, Any]]]:
    by_model: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        model = row.get("model_name")
        if not isinstance(model, str) or not model:
            raise ValueError("distance row missing model_name")
        by_model[model].append(row)
    return dict(sorted(by_model.items()))


def _stratified_effect(rows: Iterable[Mapping[str, Any]]) -> tuple[float, dict[str, dict[str, list[float]]]]:
    strata: dict[str, dict[str, list[float]]] = defaultdict(lambda: {"A": [], "B": []})
    for row in rows:
        group = row.get("group")
        dialect = _dialect_key(row)
        distance = row.get("distance")
        if group not in {"A", "B"} or dialect is None:
            continue
        if isinstance(distance, bool) or not isinstance(distance, (int, float)) or not math.isfinite(distance):
            raise ValueError("distance row has non-finite distance")
        strata[dialect][group].append(float(distance))
    usable = {dialect: values for dialect, values in strata.items() if values["A"] and values["B"]}
    if not usable:
        raise ValueError("no matched A/B dialect strata")
    effects = [mean(values["B"]) - mean(values["A"]) for values in usable.values()]
    return mean(effects), usable


def _bootstrap_effect(
    strata: Mapping[str, Mapping[str, list[float]]],
    *,
    seed: int,
    replicates: int,
) -> list[float]:
    if replicates < 1000:
        raise ValueError("at least 1000 bootstrap replicates are required")
    rng = random.Random(seed)
    dialects = sorted(strata)
    estimates: list[float] = []
    for _ in range(replicates):
        sampled_effects: list[float] = []
        for dialect in (rng.choice(dialects) for _ in dialects):
            within = strata[dialect]["A"]
            between = strata[dialect]["B"]
            sampled_within = [rng.choice(within) for _ in within]
            sampled_between = [rng.choice(between) for _ in between]
            sampled_effects.append(mean(sampled_between) - mean(sampled_within))
        estimate = mean(sampled_effects)
        if not math.isfinite(estimate):
            raise ValueError("non-finite bootstrap estimate")
        estimates.append(estimate)
    return estimates


def _one_sided_positive_p(replicates: list[float]) -> float:
    non_positive = sum(1 for value in replicates if value <= 0.0)
    return (non_positive + 1) / (len(replicates) + 1)


def _report_model(model_name: str, rows: list[Mapping[str, Any]], *, seed: int, replicates: int) -> dict[str, Any]:
    effect, strata = _stratified_effect(rows)
    estimates = _bootstrap_effect(strata, seed=seed, replicates=replicates)
    return {
        "model_name": model_name,
        "matched_dialect_count": len(strata),
        "counts": {
            dialect: {"A": len(values["A"]), "B": len(values["B"])}
            for dialect, values in sorted(strata.items())
        },
        "median_distance": {
            "A": median(value for values in strata.values() for value in values["A"]),
            "B": median(value for values in strata.values() for value in values["B"]),
        },
        "effect": effect,
        "ci": {
            "lower": _quantile(estimates, 0.025),
            "upper": _quantile(estimates, 0.975),
            "confidence_level": 0.95,
        },
        "raw_p": _one_sided_positive_p(estimates),
        "bootstrap_replicates": len(estimates),
    }


def run_speaker_effect_gate(
    rows: Iterable[Mapping[str, Any]],
    *,
    seed: int,
    replicates: int,
) -> dict[str, Any]:
    reports = [_report_model(model, model_rows, seed=seed, replicates=replicates) for model, model_rows in _model_rows(rows).items()]
    adjusted = holm_adjust({report["model_name"]: report["raw_p"] for report in reports})
    for report in reports:
        report["holm_adjusted_p"] = adjusted[report["model_name"]]
        report["status"] = (
            "passed"
            if report["effect"] > 0 and report["ci"]["lower"] > 0 and report["holm_adjusted_p"] < 0.05
            else "failed"
        )
    return {
        "schema": "speaker-effect-gate-v1",
        "seed": seed,
        "status": "passed" if reports and all(report["status"] == "passed" for report in reports) else "failed",
        "models": reports,
    }


def run_clustered_speaker_effect_gate(
    rows: Iterable[Mapping[str, Any]],
    *,
    seed: int,
    replicates: int,
    permutations: int,
) -> dict[str, Any]:
    """Run the primary matched A/B gate with speaker-cluster inference.

    The confidence interval and one-sided tail probability are obtained from
    the same nested cluster bootstrap.  A sign-flip null is reported when the
    matched strata contain speakers observed in both arms; it is retained as
    a diagnostic rather than silently substituting for the bootstrap result.
    """
    reports: list[dict[str, Any]] = []
    for model_name, model_rows in _model_rows(rows).items():
        try:
            effect = clustered_ab_effect(model_rows)
            bootstrap = clustered_bootstrap(model_rows, seed=seed, replicates=replicates)
            try:
                sign_flip = clustered_sign_flip_test(model_rows, seed=seed, permutations=permutations)
            except ClusterStatisticsError as error:
                sign_flip = {"status": "unavailable", "reason": str(error)}
            report = {
                "model_name": model_name,
                "estimand": effect["estimand"],
                "effect": effect["effect"],
                "ci": bootstrap["ci"],
                "bootstrap_tail_p_nonpositive": bootstrap["bootstrap_tail_p_nonpositive"],
                "clustered_bootstrap": bootstrap,
                "sign_flip": sign_flip,
                "matched_stratum_count": effect["matched_stratum_count"],
                "speaker_cluster_count": effect["speaker_cluster_count"],
                "utterance_count": effect["utterance_count"],
                "resampling_unit": effect["resampling_unit"],
                "test_type": "nested_speaker_cluster_bootstrap_with_optional_cluster_sign_flip",
            }
            report["raw_p"] = (
                sign_flip["raw_p"]
                if sign_flip.get("status", "") != "unavailable"
                else bootstrap["bootstrap_tail_p_nonpositive"]
            )
            report["status"] = (
                "passed"
                if report["effect"] > 0
                and report["ci"]["lower"] > 0
                and report["raw_p"] < 0.05
                else "failed"
            )
        except (ClusterStatisticsError, ValueError) as error:
            report = {
                "model_name": model_name,
                "status": "failed",
                "error": str(error),
            }
        reports.append(report)
    adjusted = holm_adjust({report["model_name"]: report.get("raw_p", 1.0) for report in reports})
    for report in reports:
        report["holm_adjusted_p"] = adjusted[report["model_name"]]
        if report.get("status") == "passed" and report["holm_adjusted_p"] >= 0.05:
            report["status"] = "failed"
    return {
        "schema": "speaker-effect-clustered-gate-v1",
        "seed": seed,
        "bootstrap_replicates": replicates,
        "permutations": permutations,
        "holm_family": "all configured extractors",
        "status": "passed" if reports and all(report["status"] == "passed" for report in reports) else "failed",
        "models": reports,
        "inference_contract": {
            "primary_uncertainty": "nested speaker-cluster bootstrap with utterance rows resampled within selected clusters",
            "p_value": "cluster sign-flip when exchangeable paired clusters exist; otherwise bootstrap tail proportion",
            "bootstrap_tail_is_not_permutation": True,
        },
    }


def _load_distance_rows(paths: Iterable[str | Path]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in paths:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        model = payload["model_name"]
        for row in payload["distances"]:
            item = dict(row)
            item["model_name"] = model
            rows.append(item)
    return rows


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--distances", nargs="+", action="append", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--seed", type=int, default=20260829)
    parser.add_argument("--replicates", type=int, default=1000)
    args = parser.parse_args(argv)
    args.distances = [path for group in args.distances for path in group]
    return args


def main() -> int:
    args = _parse_args()
    report = run_speaker_effect_gate(_load_distance_rows(args.distances), seed=args.seed, replicates=args.replicates)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return 0 if report["status"] == "passed" else 2


if __name__ == "__main__":
    raise SystemExit(main())
