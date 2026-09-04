"""Relation-level summaries for operational dialect-distance references."""

from __future__ import annotations

from collections import defaultdict
from itertools import combinations
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
from scipy.stats import kendalltau, spearmanr


Relation = tuple[str, str]


def canonical_relation(labels: Iterable[Any]) -> Relation:
    """Return one unordered, off-diagonal dialect relation."""
    unique = sorted(dict.fromkeys(map(str, labels)))
    if len(unique) != 2:
        raise ValueError("relation must contain two distinct off-diagonal labels")
    return unique[0], unique[1]


def aggregate_relation_predictions(
    rows: Sequence[Mapping[str, Any]],
) -> dict[Relation, float]:
    """Average predicted distance for each unordered off-diagonal relation."""
    grouped: dict[Relation, list[float]] = defaultdict(list)
    for row in rows:
        labels = list(map(str, row["dialect_labels"]))
        if len(set(labels)) != 2:
            continue
        value = row.get("predicted_distance", row.get("distance"))
        if value is None:
            raise ValueError("row lacks predicted distance")
        grouped[canonical_relation(labels)].append(float(value))
    return {
        relation: float(np.mean(values))
        for relation, values in sorted(grouped.items())
    }


def reference_relations(
    reference: Mapping[str, Mapping[str, float]],
) -> dict[Relation, float]:
    labels = sorted(reference)
    return {
        (left, right): float(reference[left][right])
        for left, right in combinations(labels, 2)
    }


def ranking_metrics(
    reference: Mapping[Relation, float],
    prediction: Mapping[Relation, float],
) -> dict[str, Any]:
    """Compare relation ordering with a declared operational reference."""
    if any(left == right for left, right in reference):
        raise ValueError("ranking requires off-diagonal relations")
    keys = sorted(set(reference) & set(prediction))
    if len(keys) < 2:
        raise ValueError("ranking requires at least two shared relations")
    expected = np.asarray([float(reference[key]) for key in keys], dtype=float)
    observed = np.asarray([float(prediction[key]) for key in keys], dtype=float)
    if not np.isfinite(expected).all() or not np.isfinite(observed).all():
        raise ValueError("ranking inputs must be finite")

    correct = 0
    reversals = 0
    ties = 0
    comparable = 0
    for first, second in combinations(range(len(keys)), 2):
        expected_sign = np.sign(expected[first] - expected[second])
        observed_sign = np.sign(observed[first] - observed[second])
        if expected_sign == 0 or observed_sign == 0:
            ties += 1
            continue
        comparable += 1
        if expected_sign == observed_sign:
            correct += 1
        else:
            reversals += 1

    spearman = spearmanr(expected, observed).statistic
    kendall = kendalltau(expected, observed, variant="b").statistic
    return {
        "relation_count": len(keys),
        "spearman": float(spearman),
        "kendall_tau_b": float(kendall),
        "pairwise_order_accuracy": (
            float(correct / comparable) if comparable else None
        ),
        "order_reversals": reversals,
        "ties": ties,
        "comparable_relation_pairs": comparable,
    }


def ordering_changes(
    baseline: Mapping[Relation, float], method: Mapping[Relation, float]
) -> dict[str, int]:
    """Count pairwise relation-order reversals against a baseline ordering."""
    keys = sorted(set(baseline) & set(method))
    if len(keys) < 2:
        raise ValueError("ordering comparison requires at least two shared relations")
    reversals = 0
    ties = 0
    comparable = 0
    for first, second in combinations(keys, 2):
        baseline_sign = np.sign(float(baseline[first]) - float(baseline[second]))
        method_sign = np.sign(float(method[first]) - float(method[second]))
        if baseline_sign == 0 or method_sign == 0:
            ties += 1
            continue
        comparable += 1
        reversals += int(baseline_sign != method_sign)
    return {
        "reversals_vs_baseline": reversals,
        "ties_in_either_ordering": ties,
        "comparable_relation_pairs": comparable,
    }
