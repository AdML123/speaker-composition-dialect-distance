import pytest
import torch

from src.cross_dialect_gradient_isolation import (
    aggregate_gradient_isolation_gate,
    build_lambda_dose_response_grid,
    build_same_dialect_independent_pool,
    build_target_matched_random_pool,
    compute_gradient_probe_metrics,
)


def test_lambda_dose_response_grid_is_locked_and_deterministic():
    assert build_lambda_dose_response_grid() == [
        0.0,
        0.05,
        0.1,
        0.25,
        0.5,
        1.0,
        2.0,
        5.0,
    ]


def _toy_examples():
    return [
        {"pair_id": "z1", "speaker_ids": ["s1", "s2"], "dialect_labels": ["M", "M"], "target": 0.0},
        {"pair_id": "z2", "speaker_ids": ["s3", "s4"], "dialect_labels": ["M", "M"], "target": 0.0},
        {"pair_id": "n1", "speaker_ids": ["s1", "s2"], "dialect_labels": ["M", "R"], "target": 0.8},
        {"pair_id": "n2", "speaker_ids": ["s3", "s4"], "dialect_labels": ["M", "R"], "target": 0.6},
        {"pair_id": "n3", "speaker_ids": ["s1", "s3"], "dialect_labels": ["M", "E"], "target": 0.4},
    ]


def test_independent_loss_pools_have_common_count_and_record_histogram():
    generic = _toy_examples()
    cross = [item for item in generic if item["target"] > 0]
    common_count = min(len(cross), len([item for item in generic if item["target"] == 0]))
    same = build_same_dialect_independent_pool(generic, requested_count=len(cross), seed=7)
    random_pool = build_target_matched_random_pool(generic, requested_count=common_count, seed=7)
    assert len(same.examples) == common_count
    assert len(random_pool.examples) == common_count
    assert all(item["target"] == 0.0 for item in same.examples)
    assert same.requested_count == len(cross)
    assert same.achieved_count == common_count
    assert random_pool.requested_count == common_count
    assert random_pool.achieved_count == common_count
    assert random_pool.target_histogram[0.0] == 1
    assert sum(random_pool.target_histogram.values()) == common_count
    assert sum(count for target, count in random_pool.target_histogram.items() if target > 0) == 1


def test_gradient_probe_reports_normalized_norms_and_cosine():
    model = torch.nn.Linear(2, 1, bias=False)
    with torch.no_grad():
        model.weight[:] = torch.tensor([[1.0, 0.0]])
    pair_inputs = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    pair_targets = torch.tensor([0.0, 0.0])
    cross_inputs = torch.tensor([[1.0, 1.0]])
    cross_targets = torch.tensor([1.0])
    metrics = compute_gradient_probe_metrics(
        model,
        pair_inputs,
        pair_targets,
        cross_inputs,
        cross_targets,
    )
    assert metrics["pair_count"] == 2
    assert metrics["cross_count"] == 1
    assert metrics["cross_norm_per_example"] >= 0.0
    assert -1.0 <= metrics["pair_cross_cosine"] <= 1.0
    assert "mixed_nonzero_norm_per_example" in metrics


def test_gate_aggregator_is_fail_closed():
    failed = aggregate_gradient_isolation_gate(
        dose_response={"passed": True},
        independent_loss_control={"passed": True},
        gradient_probe={"passed": False},
    )
    assert failed["status"] == "failed"
    passed = aggregate_gradient_isolation_gate(
        dose_response={"passed": True},
        independent_loss_control={"passed": True},
        gradient_probe={"passed": True},
    )
    assert passed["status"] == "passed"


def test_gate_aggregator_accepts_status_based_reports():
    report = aggregate_gradient_isolation_gate(
        dose_response={"status": "passed"},
        independent_loss_control={"status": "passed"},
        gradient_probe={"status": "failed"},
    )
    assert report["criteria"] == {
        "dose_response": True,
        "independent_loss_control": True,
        "gradient_probe": False,
    }
    assert report["status"] == "failed"
