"""Distance functions for frozen speech embeddings."""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping
from typing import Any

import torch


def cosine_distance(first: torch.Tensor, second: torch.Tensor) -> float:
    """Return 1 - cosine similarity with finite and non-zero validation."""
    if first.ndim != 1 or second.ndim != 1 or first.shape != second.shape:
        raise ValueError("cosine distance requires equal one-dimensional vectors")
    if not torch.isfinite(first).all() or not torch.isfinite(second).all():
        raise ValueError("cosine distance requires finite vectors")
    first = first.to(dtype=torch.float64)
    second = second.to(dtype=torch.float64)
    first_norm = torch.linalg.vector_norm(first)
    second_norm = torch.linalg.vector_norm(second)
    if float(first_norm) == 0.0 or float(second_norm) == 0.0:
        raise ValueError("cosine distance is undefined for zero vectors")
    value = 1.0 - torch.dot(first, second) / (first_norm * second_norm)
    result = float(value.item())
    if not math.isfinite(result):
        raise ValueError("cosine distance is non-finite")
    return max(0.0, min(2.0, result))


def pairwise_distances(
    pairs: Iterable[Mapping[str, Any]],
    vectors: Mapping[str, torch.Tensor],
    *,
    model_name: str,
) -> dict[str, Any]:
    distances: list[dict[str, Any]] = []
    for pair in pairs:
        pair_id = pair.get("pair_id")
        utterance_ids = pair.get("source_utterance_ids")
        if not isinstance(pair_id, str) or not isinstance(utterance_ids, (list, tuple)) or len(utterance_ids) != 2:
            raise ValueError("malformed pair metadata")
        first_id, second_id = utterance_ids
        if first_id not in vectors or second_id not in vectors:
            raise ValueError(f"missing embedding for pair {pair_id}")
        distances.append(
            {
                "pair_id": pair_id,
                "group": pair.get("group"),
                "split": pair.get("split"),
                "dialect_labels": pair.get("dialect_labels"),
                "speaker_ids": pair.get("speaker_ids"),
                "utterance_ids": pair.get("utterance_ids", utterance_ids),
                "matched_stratum": pair.get("matched_stratum"),
                "matched_fields": pair.get("matched_fields"),
                "distance": cosine_distance(vectors[first_id], vectors[second_id]),
            }
        )
    return {
        "schema": "pair-distances-v1",
        "model_name": model_name,
        "metric": "cosine",
        "distance_count": len(distances),
        "distances": distances,
    }
