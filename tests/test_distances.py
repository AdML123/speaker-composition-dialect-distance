from __future__ import annotations

import pytest
import torch

from src.distances import cosine_distance, pairwise_distances


def test_cosine_distance_is_symmetric_and_zero_for_identical_vectors():
    first = torch.tensor([1.0, 2.0, 3.0])
    second = torch.tensor([-1.0, 0.5, 4.0])

    assert cosine_distance(first, first) == pytest.approx(0.0)
    assert cosine_distance(first, second) == pytest.approx(cosine_distance(second, first))


def test_cosine_distance_rejects_zero_or_non_finite_vectors():
    with pytest.raises(ValueError, match="zero"):
        cosine_distance(torch.zeros(3), torch.ones(3))
    with pytest.raises(ValueError, match="finite"):
        cosine_distance(torch.tensor([1.0, float("nan")]), torch.ones(2))


def test_pairwise_distances_preserves_pair_metadata():
    vectors = {
        "u1": torch.tensor([1.0, 0.0]),
        "u2": torch.tensor([0.0, 1.0]),
        "u3": torch.tensor([1.0, 0.0]),
    }
    pairs = [
        {"pair_id": "A-000001", "group": "A", "source_utterance_ids": ["u1", "u3"]},
        {"pair_id": "B-000001", "group": "B", "source_utterance_ids": ["u1", "u2"]},
    ]

    result = pairwise_distances(pairs, vectors, model_name="test-model")

    assert result["schema"] == "pair-distances-v1"
    assert result["model_name"] == "test-model"
    assert result["distances"][0]["pair_id"] == "A-000001"
    assert result["distances"][0]["distance"] == pytest.approx(0.0)
    assert result["distances"][1]["distance"] == pytest.approx(1.0)

