"""Target-prevalence controls for separately weighted pair losses."""

from __future__ import annotations

import hashlib
import json
import math
import random
from collections import Counter
from typing import Any, Mapping, Sequence

import numpy as np


TARGET_PREVALENCE_GRID = (0.0, 0.10, 0.25, 0.50, 0.75, 1.0)
LAMBDA_CROSS_GRID = (0.0, 0.25, 0.50, 1.0, 2.0)


def _hash(payload: object) -> str:
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()


def audit_prevalence_capacity(
    examples: Sequence[Mapping[str, Any]],
    *,
    q_grid: Sequence[float] = TARGET_PREVALENCE_GRID,
    minimum_cell: int = 200,
) -> dict[str, Any]:
    """Choose the largest common natural-pool size before outcome fitting."""
    zero = sum(float(item["target"]) == 0.0 for item in examples)
    nonzero = len(examples) - zero
    feasible = []
    for q in q_grid:
        limits = []
        if q > 0:
            limits.append(nonzero / q)
        if q < 1:
            limits.append(zero / (1 - q))
        feasible.append(math.floor(min(limits)) if limits else 0)
    common = (min(feasible) // 20) * 20 if feasible else 0
    return {
        "schema": "target-prevalence-capacity-v1",
        "q_grid": [float(q) for q in q_grid],
        "lambda_cross_grid": list(LAMBDA_CROSS_GRID),
        "zero_capacity": zero,
        "nonzero_capacity": nonzero,
        "common_pool_size": common,
        "minimum_cell": minimum_cell,
        "status": "passed" if common >= minimum_cell else "failed",
        "result_blind": True,
    }


def build_nested_fixed_pair_masks(
    nonzero_examples: Sequence[Mapping[str, Any]],
    *,
    pool_size: int,
    q_grid: Sequence[float] = TARGET_PREVALENCE_GRID,
    seed: int,
) -> dict[str, Any]:
    """Mask targets in a fixed identity pool using nested deterministic sets."""
    if pool_size <= 0 or pool_size > len(nonzero_examples):
        raise ValueError("fixed-pair pool_size exceeds nonzero support")
    ordered = sorted((dict(item) for item in nonzero_examples), key=lambda item: str(item["pair_id"]))
    rng = random.Random(seed)
    base = sorted(rng.sample(ordered, pool_size), key=lambda item: str(item["pair_id"]))
    rank = list(range(pool_size))
    rng.shuffle(rank)
    ranks = {index: position for position, index in enumerate(rank)}
    arms: dict[str, list[dict[str, Any]]] = {}
    mask_hashes = {}
    for q in q_grid:
        keep = int(round(float(q) * pool_size))
        arm = []
        nonzero_ids = []
        for index, item in enumerate(base):
            current = dict(item)
            if ranks[index] >= keep:
                current["target"] = 0.0
                current["target_masked"] = True
            else:
                current["target_masked"] = False
                nonzero_ids.append(str(current["pair_id"]))
            arm.append(current)
        key = format(float(q), ".2f")
        arms[key] = arm
        mask_hashes[key] = _hash(nonzero_ids)
    return {
        "schema": "fixed-pair-target-prevalence-v1",
        "seed": seed,
        "pool_size": pool_size,
        "pair_identity_hash": _hash([item["pair_id"] for item in base]),
        "mask_hashes": mask_hashes,
        "arms": arms,
    }


def build_natural_prevalence_pools(
    examples: Sequence[Mapping[str, Any]],
    *,
    pool_size: int,
    q_grid: Sequence[float] = TARGET_PREVALENCE_GRID,
    seed: int,
) -> dict[str, Any]:
    """Build natural zero/nonzero pools and report structural covariates."""
    zero = sorted((dict(item) for item in examples if float(item["target"]) == 0), key=lambda x: str(x["pair_id"]))
    nonzero = sorted((dict(item) for item in examples if float(item["target"]) != 0), key=lambda x: str(x["pair_id"]))
    rng = random.Random(seed)
    arms = {}
    audit = {}
    for q in q_grid:
        n_nonzero = int(round(pool_size * float(q)))
        n_zero = pool_size - n_nonzero
        if n_nonzero > len(nonzero) or n_zero > len(zero):
            raise ValueError("natural prevalence cell exceeds capacity")
        selected = rng.sample(nonzero, n_nonzero) + rng.sample(zero, n_zero)
        selected.sort(key=lambda item: str(item["pair_id"]))
        key = format(float(q), ".2f")
        arms[key] = selected
        audit[key] = summarize_pool_covariates(selected)
    return {"schema": "natural-target-prevalence-v1", "seed": seed, "pool_size": pool_size, "arms": arms, "covariates": audit}


def summarize_pool_covariates(examples: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    speakers = [str(s) for item in examples for s in item.get("speaker_ids", [])]
    endpoints = [str(u) for item in examples for u in item.get("utterance_ids", item.get("source_utterance_ids", []))]
    pairs = ["|".join(sorted(map(str, item.get("dialect_labels", [])))) for item in examples]
    conditions = [str(c) for item in examples for c in item.get("recording_conditions", [])]
    counts = Counter(pairs)
    total = max(sum(counts.values()), 1)
    entropy = -sum((count / total) * math.log(count / total) for count in counts.values())
    return {
        "unique_speaker_count": len(set(speakers)),
        "unique_dialect_pair_count": len(counts),
        "dialect_pair_entropy": entropy,
        "condition_histogram": dict(sorted(Counter(conditions).items())),
        "endpoint_count": len(endpoints),
        "unique_endpoint_count": len(set(endpoints)),
        "endpoint_reuse_fraction": 1.0 - len(set(endpoints)) / max(len(endpoints), 1),
    }


def prevalence_balanced_loss(zero_losses: Sequence[float], nonzero_losses: Sequence[float]) -> float:
    if not zero_losses or not nonzero_losses:
        raise ValueError("balanced loss requires both target strata")
    return 0.5 * float(np.mean(zero_losses)) + 0.5 * float(np.mean(nonzero_losses))


def slope_contrast(
    ordinary: Sequence[Mapping[str, float]],
    balanced: Sequence[Mapping[str, float]],
) -> dict[str, float]:
    """Return beta_ordinary - beta_balanced from aligned q cells."""
    q1 = [float(row["q"]) for row in ordinary]
    q2 = [float(row["q"]) for row in balanced]
    if q1 != q2 or len(q1) < 2:
        raise ValueError("ordinary and balanced q cells must align")
    beta_o = float(np.polyfit(q1, [float(row["gain"]) for row in ordinary], 1)[0])
    beta_b = float(np.polyfit(q2, [float(row["gain"]) for row in balanced], 1)[0])
    return {"beta_ordinary": beta_o, "beta_balanced": beta_b, "delta_beta": beta_o - beta_b}


def clustered_slope_bootstrap(
    rows: Sequence[Mapping[str, Any]],
    *,
    seed: int,
    replicates: int = 1000,
) -> dict[str, Any]:
    """Bootstrap ordinary-minus-balanced slope contrasts by speaker clusters."""
    if replicates < 1000:
        raise ValueError("at least 1000 bootstrap replicates are required")
    by_q: dict[float, dict[str, list[Mapping[str, Any]]]] = {}
    for row in rows:
        q = float(row["q"])
        speakers = row.get("speaker_ids") or row.get("speaker_id")
        if isinstance(speakers, (list, tuple)):
            cluster = "|".join(sorted(map(str, speakers)))
        else:
            cluster = str(speakers)
        if not cluster:
            raise ValueError("slope rows require speaker cluster identifiers")
        by_q.setdefault(q, {}).setdefault(cluster, []).append(row)
    q_values = sorted(by_q)
    if len(q_values) < 2:
        raise ValueError("at least two q cells are required")

    def estimate(selected: Mapping[float, Sequence[Mapping[str, Any]]]) -> float:
        ordinary = []
        balanced = []
        for q in q_values:
            cells = selected[q]
            ordinary.append({"q": q, "gain": float(np.mean([float(r["ordinary_gain"]) for r in cells]))})
            balanced.append({"q": q, "gain": float(np.mean([float(r["balanced_gain"]) for r in cells]))})
        return slope_contrast(ordinary, balanced)["delta_beta"]

    observed = estimate({q: [row for cluster in clusters.values() for row in cluster] for q, clusters in by_q.items()})
    rng = random.Random(seed)
    boot = []
    for _ in range(replicates):
        selected = {}
        for q, clusters in by_q.items():
            keys = sorted(clusters)
            chosen = [rng.choice(keys) for _ in keys]
            selected[q] = [row for key in chosen for row in rng.choices(clusters[key], k=len(clusters[key]))]
        boot.append(estimate(selected))
    ordered = sorted(boot)
    return {
        "schema": "clustered-slope-bootstrap-v1",
        "observed_delta_beta": float(observed),
        "ci": {"lower": float(np.quantile(ordered, 0.025)), "upper": float(np.quantile(ordered, 0.975)), "confidence_level": 0.95},
        "bootstrap_replicates": replicates,
        "resampling_unit": "speaker_cluster_within_q",
        "nested_utterance_sampling": True,
        "q_cells": q_values,
    }


def clustered_mean_gain_contrast(
    rows: Sequence[Mapping[str, Any]],
    *,
    seed: int,
    replicates: int = 1000,
) -> dict[str, Any]:
    """Bootstrap the balanced-minus-ordinary mean gain by speaker cluster."""
    if replicates < 1000:
        raise ValueError("at least 1000 bootstrap replicates are required")
    by_speaker: dict[str, list[float]] = {}
    for row in rows:
        speakers = row.get("speaker_ids") or row.get("speaker_id")
        if isinstance(speakers, (list, tuple)):
            cluster = "|".join(sorted(map(str, speakers)))
        else:
            cluster = str(speakers)
        if not cluster:
            raise ValueError("gain contrast rows require speaker cluster identifiers")
        by_speaker.setdefault(cluster, []).append(float(row["balanced_gain"]) - float(row["ordinary_gain"]))
    if not by_speaker:
        raise ValueError("at least one speaker cluster is required")
    cluster_means = {speaker: float(np.mean(values)) for speaker, values in by_speaker.items()}
    observed = float(np.mean(list(cluster_means.values())))
    keys = sorted(cluster_means)
    rng = random.Random(seed)
    boot = [float(np.mean([cluster_means[rng.choice(keys)] for _ in keys])) for _ in range(replicates)]
    return {
        "schema": "clustered-mean-gain-contrast-v1",
        "contrast": "prevalence_balanced_minus_ordinary_mean_gain_across_internal_q",
        "estimate": observed,
        "ci": {
            "lower": float(np.quantile(boot, 0.025)),
            "upper": float(np.quantile(boot, 0.975)),
            "confidence_level": 0.95,
        },
        "bootstrap_replicates": replicates,
        "resampling_unit": "evaluation_speaker_cluster",
        "speaker_cluster_count": len(keys),
    }
