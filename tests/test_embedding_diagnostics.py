import pytest

from src.embedding_diagnostics import compare_preprocessing_contract, summarize_embedding_cache


def test_embedding_diagnostics_exposes_norm_and_variance():
    report = summarize_embedding_cache({"a": [3.0, 4.0], "b": [0.0, 2.0]}, model_name="m", split="evaluation")
    assert report["embedding_count"] == 2
    assert report["norm_summary"]["minimum"] == pytest.approx(2.0)
    assert report["explicit_l2_normalization"] is False
    assert report["finite"] is True


def test_preprocessing_contract_rejects_hidden_pooling_mismatch():
    base = summarize_embedding_cache({"a": [1.0, 0.0]}, model_name="m1", split="evaluation")
    other = dict(base, model_name="m2", pooling="sum")
    report = compare_preprocessing_contract([base, other])
    assert report["status"] == "failed"
    assert "pooling" in report["mismatches"]
