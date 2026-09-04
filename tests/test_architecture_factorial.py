import pytest

from src.architecture_factorial import (
    _load_protocol,
    build_factorial_jobs,
    difference_in_differences,
)


def test_protocol_loader_accepts_yaml_dates_without_json_roundtrip(tmp_path):
    path = tmp_path / "protocol.yaml"
    path.write_text("schema: test\ndate: 2026-09-04\n", encoding="utf-8")
    payload = _load_protocol(path)
    assert payload["schema"] == "test"
    assert str(payload["date"]) == "2026-09-04"


def test_primary_heads_share_budget_and_lambda_grid():
    jobs = build_factorial_jobs()
    primary = [
        job
        for job in jobs
        if job["head"] in {"linear", "mlp_parameter_matched"}
    ]
    assert {job["epochs"] for job in primary} == {30}
    assert {job["batch_size"] for job in primary} == {256}
    assert {job["weight_decay"] for job in primary} == {0.001}
    assert {job["lambda_cross"] for job in primary} == {0.0, 0.25, 0.5, 1.0}
    assert all(job["early_stopping"] is False for job in primary)
    assert len(jobs) == 120


def test_jobs_share_schedule_key_within_reference_and_seed():
    jobs = build_factorial_jobs()
    selected = [
        job
        for job in jobs
        if job["reference"] == "taxonomy" and job["seed"] == 20260829
    ]
    assert len({job["schedule_key"] for job in selected}) == 1


def test_difference_in_differences_compares_cross_loss_increments():
    value = difference_in_differences(
        linear_zero=0.40,
        linear_added=0.42,
        mlp_zero=0.45,
        mlp_added=0.43,
    )
    assert value == pytest.approx(0.04)
