"""Pool-exposure and gradient-budget diagnostics for pair-loss training."""

from __future__ import annotations

import math
from typing import Any, Mapping, Sequence

import numpy as np


EXPOSURE_RATIO_GRID = ((1, 1), (2, 1), (4, 1), (8, 1))


def build_exposure_ratio_grid() -> list[dict[str, int | float]]:
    return [
        {"generic_count": generic, "cross_count": cross, "rho": generic / cross}
        for generic, cross in EXPOSURE_RATIO_GRID
    ]


def build_candidate_inventory_grid(
    pair_examples: Sequence[Mapping[str, Any]],
    cross_examples: Sequence[Mapping[str, Any]],
    *,
    fractions: Sequence[float] = (0.25, 0.50, 1.0, 2.0),
) -> list[dict[str, Any]]:
    """Return result-blind inventory sizes while preserving exposure controls."""
    if not pair_examples or not cross_examples:
        raise ValueError("both candidate pools are required")
    return [
        {
            "fraction": float(fraction),
            "pair_inventory": max(1, min(len(pair_examples), int(round(len(pair_examples) * fraction)))),
            "cross_inventory": max(1, min(len(cross_examples), int(round(len(cross_examples) * fraction)))),
            "pair_ids_hash": _stable_ids(pair_examples, fraction),
            "cross_ids_hash": _stable_ids(cross_examples, fraction),
        }
        for fraction in fractions
    ]


def compute_aggregation_arms(
    pair_losses: Sequence[float],
    cross_losses: Sequence[float],
    lambda_cross: float,
) -> dict[str, float]:
    if not pair_losses or not cross_losses:
        raise ValueError("both loss pools are required")
    pair_mean = float(np.mean(pair_losses))
    cross_mean = float(np.mean(cross_losses))
    return {
        "pair_mean": pair_mean,
        "cross_mean": cross_mean,
        "separate": pair_mean + float(lambda_cross) * cross_mean,
        "mixed_mean": (sum(pair_losses) + float(lambda_cross) * sum(cross_losses))
        / (len(pair_losses) + len(cross_losses)),
    }


def compute_gradient_budget_readouts(
    pair_norm: float,
    cross_norm: float,
    *,
    lambda_cross: float,
    n_pair: int,
    n_cross: int,
    cosine: float,
) -> dict[str, float | int]:
    values = [pair_norm, cross_norm, lambda_cross, n_pair, n_cross, cosine]
    if any(not math.isfinite(float(value)) for value in values):
        raise ValueError("gradient readouts must be finite")
    if pair_norm < 0 or cross_norm < 0 or lambda_cross < 0 or n_pair < 0 or n_cross < 0:
        raise ValueError("gradient norms, weights, and counts must be non-negative")
    if not -1.0 <= cosine <= 1.0:
        raise ValueError("gradient cosine must be bounded")
    separate_denominator = pair_norm + lambda_cross * cross_norm
    mixed_denominator = n_pair * pair_norm + lambda_cross * n_cross * cross_norm
    return {
        "pair_norm": float(pair_norm),
        "cross_norm": float(cross_norm),
        "cosine": float(cosine),
        "lambda_cross": float(lambda_cross),
        "n_pair": int(n_pair),
        "n_cross": int(n_cross),
        "eta_sep": 0.0 if separate_denominator == 0 else float(lambda_cross * cross_norm / separate_denominator),
        "eta_mix": 0.0 if mixed_denominator == 0 else float(lambda_cross * n_cross * cross_norm / mixed_denominator),
    }


def classify_mechanism(
    *,
    prevalence_supported: bool,
    gradient_budget_supported: bool,
    interference_supported: bool,
    regularization_compatible: bool,
) -> dict[str, Any]:
    names = []
    if prevalence_supported:
        names.append("target_prevalence")
    if gradient_budget_supported:
        names.append("gradient_budget")
    if interference_supported:
        names.append("gradient_interference")
    if regularization_compatible:
        names.append("regularization_compatible")
    if len(names) == 1:
        status = "unique_supported_mechanism"
    elif names:
        status = "compatible_with_multiple_mechanisms"
    else:
        status = "mechanism_unresolved"
    return {"schema": "pool-ratio-mechanism-classification-v1", "status": status, "compatible_mechanisms": names}


def fit_log_rho_lambda_interaction(rows: Sequence[Mapping[str, Any]]) -> dict[str, float]:
    if len(rows) < 4:
        raise ValueError("at least four exposure rows are required")
    matrix = np.asarray([
        [1.0, math.log(float(row["rho"])), float(row["lambda_cross"]), math.log(float(row["rho"])) * float(row["lambda_cross"])]
        for row in rows
    ], dtype=np.float64)
    outcome = np.asarray([float(row["gain"]) for row in rows], dtype=np.float64)
    coefficients = np.linalg.lstsq(matrix, outcome, rcond=None)[0]
    return {
        "intercept": float(coefficients[0]),
        "log_rho": float(coefficients[1]),
        "lambda_cross": float(coefficients[2]),
        "log_rho_by_lambda_cross": float(coefficients[3]),
    }


def speaker_cluster_interaction_bootstrap(
    rows: Sequence[Mapping[str, Any]],
    *,
    seed: int,
    replicates: int = 1000,
) -> dict[str, Any]:
    if replicates < 1000:
        raise ValueError("at least 1000 bootstrap replicates are required")
    speakers = sorted({str(row["speaker_id"]) for row in rows})
    if not speakers:
        raise ValueError("speaker-cluster rows are required")
    by_speaker = {speaker: [row for row in rows if str(row["speaker_id"]) == speaker] for speaker in speakers}

    def aggregate(selected: Sequence[str]) -> list[dict[str, float]]:
        cells: dict[tuple[float, float], list[float]] = {}
        for speaker in selected:
            for row in by_speaker[speaker]:
                key = (float(row["rho"]), float(row["lambda_cross"]))
                cells.setdefault(key, []).append(float(row["gain"]))
        return [{"rho": rho, "lambda_cross": lam, "gain": float(np.mean(values))} for (rho, lam), values in sorted(cells.items())]

    observed = fit_log_rho_lambda_interaction(aggregate(speakers))["log_rho_by_lambda_cross"]
    rng = np.random.default_rng(seed)
    boot = []
    for _ in range(replicates):
        selected = [speakers[int(index)] for index in rng.integers(0, len(speakers), len(speakers))]
        boot.append(fit_log_rho_lambda_interaction(aggregate(selected))["log_rho_by_lambda_cross"])
    return {
        "schema": "speaker-cluster-interaction-bootstrap-v1",
        "estimate": float(observed),
        "ci": {"lower": float(np.quantile(boot, 0.025)), "upper": float(np.quantile(boot, 0.975)), "confidence_level": 0.95},
        "bootstrap_replicates": replicates,
        "resampling_unit": "evaluation_speaker_cluster",
        "speaker_cluster_count": len(speakers),
    }


def _stable_ids(examples: Sequence[Mapping[str, Any]], fraction: float) -> str:
    import hashlib
    import json
    count = max(1, min(len(examples), int(round(len(examples) * fraction))))
    ids = sorted(str(item.get("pair_id", "")) for item in examples)[:count]
    return hashlib.sha256(json.dumps(ids, separators=(",", ":")).encode("utf-8")).hexdigest()
