"""Auditable permutation of pair-distance targets used by the cross loss."""

from __future__ import annotations

import hashlib
import json
import random
from typing import Any, Mapping, Sequence


def target_hashes(examples: Sequence[Mapping[str, Any]]) -> list[str]:
    """Hash pair IDs and target values without retaining raw examples."""
    return [
        hashlib.sha256(
            json.dumps(
                {"pair_id": str(item.get("pair_id", "")), "target": float(item["target"])},
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        ).hexdigest()
        for item in examples
    ]


def permute_pair_distance_targets(
    examples: Sequence[Mapping[str, Any]],
    *,
    seed: int,
    preserve: str = "target_histogram",
) -> dict[str, Any]:
    """Return a target-permuted cross-loss pool and a provenance report."""
    if preserve != "target_histogram":
        raise ValueError("only target_histogram preservation is supported")
    items = [dict(item) for item in examples]
    before = target_hashes(items)
    order = list(range(len(items)))
    random.Random(seed).shuffle(order)
    if len(items) > 1 and all(items[i]["target"] == items[order[i]]["target"] for i in range(len(items))):
        order = order[1:] + order[:1]
    targets = [item["target"] for item in items]
    for index, item in enumerate(items):
        item["target"] = targets[order[index]]
        item["target_permutation_seed"] = seed
        item["target_permutation_source"] = "cross_loss_pair_distance_target"
    after = target_hashes(items)
    return {
        "schema": "target-permutation-control-v1",
        "seed": seed,
        "preserved": ["pool_size", "pair_id_set", "target_histogram"],
        "changed": ["target_to_pair_assignment"],
        "pool_size": len(items),
        "target_histogram_before": _histogram(examples),
        "target_histogram_after": _histogram(items),
        "target_hashes_before": before,
        "target_hashes_after": after,
        "pair_examples": items,
        "evaluation_targets_unchanged": True,
    }


def _histogram(examples: Sequence[Mapping[str, Any]]) -> dict[str, int]:
    result: dict[str, int] = {}
    for item in examples:
        key = format(float(item["target"]), ".12g")
        result[key] = result.get(key, 0) + 1
    return dict(sorted(result.items()))
