import pytest

from src.speaker_offset_alpha_sweep import run_alpha_sweep


def _reference():
    return {
        "name": "synthetic",
        "matrix": {
            "d1": {"d1": 0.0, "d2": 1.0},
            "d2": {"d1": 1.0, "d2": 0.0},
        },
    }


def _metadata(speaker, dialect):
    return {"speaker_id": speaker, "dialect_label": dialect, "recording_condition": "clean"}


def _embeddings(scale):
    return {
        "u1": [scale, 1.0, 0.0],
        "u2": [scale, 0.0, 1.0],
        "u3": [-scale, 1.0, 0.0],
        "u4": [-scale, 0.0, 1.0],
    }


def _pairs():
    return [
        {"pair_id": "a1", "group": "A", "dialect_labels": ["d1"], "source_utterance_ids": ["u1", "u1"]},
        {"pair_id": "b1", "group": "B", "dialect_labels": ["d1"], "source_utterance_ids": ["u1", "u3"]},
        {"pair_id": "c1", "group": "C", "dialect_labels": ["d1", "d2"], "source_utterance_ids": ["u1", "u2"]},
        {"pair_id": "d1", "group": "D", "dialect_labels": ["d1", "d2"], "source_utterance_ids": ["u1", "u4"]},
    ]


def test_alpha_sweep_reports_all_models_and_references():
    calibration_embeddings = {
        "chinese_hubert_large": _embeddings(10.0),
        "chinese_wav2vec2_large": _embeddings(4.0),
        "wavlm_large": _embeddings(7.0),
    }
    evaluation_embeddings = calibration_embeddings
    calibration_metadata = {utterance_id: _metadata("s1" if utterance_id in {"u1", "u2"} else "s2", "d1" if utterance_id in {"u1", "u3"} else "d2") for utterance_id in ["u1", "u2", "u3", "u4"]}
    evaluation_metadata = calibration_metadata

    report = run_alpha_sweep(
        calibration_embeddings=calibration_embeddings,
        evaluation_embeddings=evaluation_embeddings,
        calibration_metadata=calibration_metadata,
        evaluation_metadata=evaluation_metadata,
        calibration_pairs=_pairs(),
        evaluation_pairs=_pairs(),
        references=[_reference(), _reference()],
        alpha_grid=[0.0, 0.5, 1.0],
        bootstrap_replicates=1000,
    )

    assert report["schema"] == "speaker-offset-alpha-sweep-v1"
    assert len(report["models"]) == 3
    assert len(report["models"][0]["references"]) == 2
    assert report["models"][0]["references"][0]["curve"][0]["alpha"] == 0.0

