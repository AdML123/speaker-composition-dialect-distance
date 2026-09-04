import pytest

from src.speaker_mean_normalization_gate import (
    evaluate_speaker_mean_normalization_gate,
    fit_split_speaker_mean_model,
    mean_normalize_embeddings,
    pairwise_mean_normalized_distance,
    _leave_pair_out_rows,
    _metadata_for_embedding_sets,
)


def _metadata(speaker, dialect="d1", condition="clean"):
    return {"speaker_id": speaker, "dialect_label": dialect, "recording_condition": condition}


def _reference():
    return {
        "name": "synthetic",
        "matrix": {
            "d1": {"d1": 0.0, "d2": 1.0},
            "d2": {"d1": 1.0, "d2": 0.0},
        },
    }


def _pair(pair_id, group, labels, utterance_ids):
    return {
        "pair_id": pair_id,
        "group": group,
        "dialect_labels": labels,
        "source_utterance_ids": utterance_ids,
    }


def _build_split(prefix):
    embeddings = {}
    metadata = {}
    pairs = []
    dialect_vectors = {"d1": [0.0, 1.0, 0.0], "d2": [0.0, 0.0, 1.0]}
    speaker_values = {"s1": 10.0, "s2": -10.0, "s3": 8.0, "s4": -8.0}
    for dialect, base in dialect_vectors.items():
        for speaker, value in speaker_values.items():
            for index in (1, 2):
                utterance_id = f"{prefix}-{dialect}-{speaker}-{index}"
                embeddings[utterance_id] = [value, base[1], base[2]]
                metadata[utterance_id] = _metadata(speaker, dialect)
    pairs.append(_pair(f"{prefix}-A-1", "A", ["d1"], [f"{prefix}-d1-s1-1", f"{prefix}-d1-s1-2"]))
    pairs.append(_pair(f"{prefix}-A-2", "A", ["d2"], [f"{prefix}-d2-s2-1", f"{prefix}-d2-s2-2"]))
    pairs.append(_pair(f"{prefix}-B-1", "B", ["d1"], [f"{prefix}-d1-s1-1", f"{prefix}-d1-s2-1"]))
    pairs.append(_pair(f"{prefix}-C-1", "C", ["d1", "d2"], [f"{prefix}-d1-s1-1", f"{prefix}-d2-s1-1"]))
    pairs.append(_pair(f"{prefix}-D-1", "D", ["d1", "d2"], [f"{prefix}-d1-s1-1", f"{prefix}-d2-s2-1"]))
    pairs.append(_pair(f"{prefix}-D-2", "D", ["d1", "d2"], [f"{prefix}-d1-s3-1", f"{prefix}-d2-s4-1"]))
    return embeddings, metadata, pairs


def test_mean_normalize_embeddings_removes_speaker_offset():
    embeddings = {
        "u1": [10.0, 1.0, 0.0],
        "u2": [10.0, 0.0, 1.0],
        "u3": [-10.0, 1.0, 0.0],
        "u4": [-10.0, 0.0, 1.0],
    }
    metadata = {
        "u1": _metadata("s1", "d1"),
        "u2": _metadata("s1", "d2"),
        "u3": _metadata("s2", "d1"),
        "u4": _metadata("s2", "d2"),
    }

    cleaned, summary = mean_normalize_embeddings(embeddings, metadata)

    assert summary["speaker_count"] == 2
    assert summary["fallback_speaker_count"] == 0
    assert cleaned["u1"][0] == pytest.approx(0.0)
    assert cleaned["u3"][0] == pytest.approx(0.0)
    assert cleaned["u1"][1:] == pytest.approx([1.0, 0.0])
    assert cleaned["u2"][1:] == pytest.approx([0.0, 1.0])


def test_pairwise_mean_normalized_distance_supports_leave_pair_out_and_fallback():
    embeddings = {
        "u1": [10.0, 1.0, 0.0],
        "u2": [10.0, 1.0, 0.0],
        "u3": [-10.0, 0.0, 1.0],
    }
    metadata = {
        "u1": _metadata("s1", "d1"),
        "u2": _metadata("s1", "d1"),
        "u3": _metadata("s2", "d2"),
    }
    pair = _pair("p1", "C", ["d1", "d1"], ["u1", "u2"])

    distance, report = pairwise_mean_normalized_distance(
        pair,
        embeddings,
        metadata,
        leave_pair_out=True,
        return_report=True,
    )

    assert distance == pytest.approx(0.0)
    assert report["fallback_endpoint_count"] == 2
    assert report["fallback_pair_count"] == 1


def test_leave_pair_out_rows_match_pairwise_helper():
    embeddings = {
        "u1": [10.0, 1.0, 0.0],
        "u2": [10.0, 1.0, 0.0],
        "u3": [-10.0, 0.0, 1.0],
        "u4": [-10.0, 0.0, 1.0],
    }
    metadata = {
        "u1": _metadata("s1", "d1"),
        "u2": _metadata("s1", "d1"),
        "u3": _metadata("s2", "d2"),
        "u4": _metadata("s2", "d2"),
    }
    pairs = [_pair("p1", "D", ["d1", "d2"], ["u1", "u3"])]

    rows, summary = _leave_pair_out_rows(pairs, embeddings, metadata, _reference())

    assert rows[0]["distance"] == pytest.approx(0.0)
    assert summary["fallback_pair_count"] == 0
    assert summary["fallback_endpoint_ratio"] == 0.0


def test_mean_normalization_ablation_helpers_are_deterministic():
    embeddings, metadata, _ = _build_split("cal")

    shuffled_a, summary_a = mean_normalize_embeddings(embeddings, metadata, shuffle_speaker_ids=True, seed=7)
    shuffled_b, summary_b = mean_normalize_embeddings(embeddings, metadata, shuffle_speaker_ids=True, seed=7)
    wrong_a, wrong_summary = mean_normalize_embeddings(embeddings, metadata, wrong_mean=True, seed=11)

    assert shuffled_a == shuffled_b
    assert summary_a == summary_b
    assert wrong_summary["mode"] == "wrong_mean"
    assert wrong_a != embeddings


def test_speaker_mean_normalization_gate_passes_synthetic_fixture():
    calibration_embeddings, calibration_metadata, calibration_pairs = _build_split("cal")
    evaluation_embeddings, evaluation_metadata, evaluation_pairs = _build_split("eval")

    report = evaluate_speaker_mean_normalization_gate(
        {"m1": calibration_embeddings},
        {"m1": evaluation_embeddings},
        calibration_metadata,
        evaluation_metadata,
        calibration_pairs,
        evaluation_pairs,
        references=[_reference()],
        bootstrap_replicates=1000,
        ablation_seeds=[1, 2, 3],
        seed=20260831,
    )

    reference_report = report["models"][0]["references"][0]
    assert report["status"] == "passed"
    assert reference_report["improvement_ratio"] >= 0.05
    assert reference_report["ci"]["lower"] > 0.0
    assert reference_report["matched_speaker_mae_increase_ratio"] <= 0.01
    assert reference_report["matched_speaker_groups"]["A"]["mae_increase_ratio"] <= 0.01
    assert reference_report["matched_speaker_groups"]["C"]["mae_increase_ratio"] <= 0.01
    assert reference_report["leave_pair_out"]["fallback_pair_ratio"] >= 0.0
    assert reference_report["ablation"]["shuffled_speaker"]["max_improvement_ratio"] < 0.05
    assert reference_report["ablation"]["wrong_mean"]["max_improvement_ratio"] < 0.05


def test_speaker_mean_normalization_gate_rejects_too_few_bootstrap_replicates():
    with pytest.raises(ValueError, match="1000"):
        evaluate_speaker_mean_normalization_gate({}, {}, {}, {}, [], [], references=[_reference()], bootstrap_replicates=999)


def test_metadata_loader_includes_embedding_pool_beyond_pair_endpoints(tmp_path):
    record_manifest = tmp_path / "records.json"
    record_manifest.write_text(
        """
        {
          "records": [
            {"utterance_id": "u1", "speaker_id": "s1", "dialect_label": "d1", "recording_condition": "clean"},
            {"utterance_id": "u2", "speaker_id": "s1", "dialect_label": "d1", "recording_condition": "clean"},
            {"utterance_id": "u3", "speaker_id": "s1", "dialect_label": "d1", "recording_condition": "clean"}
          ]
        }
        """,
        encoding="utf-8",
    )

    metadata = _metadata_for_embedding_sets(record_manifest, {"m1": {"u1": [1.0], "u2": [2.0], "u3": [3.0]}})

    assert sorted(metadata) == ["u1", "u2", "u3"]
