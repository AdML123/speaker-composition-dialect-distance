from src.speaker_offset_decomposition import run_cross_dialect_decomposition


def test_cross_dialect_decomposition_reports_eligible_speakers_and_residuals():
    embeddings = {
        "chinese_hubert_large": {
            "u1": [10.0, 1.0],
            "u2": [10.0, 2.0],
            "u3": [10.0, 3.0],
            "u4": [-10.0, 1.0],
        },
        "chinese_wav2vec2_large": {
            "u1": [9.0, 1.0],
            "u2": [9.0, 2.0],
            "u3": [9.0, 3.0],
            "u4": [-9.0, 1.0],
        },
        "wavlm_large": {
            "u1": [8.0, 1.0],
            "u2": [8.0, 2.0],
            "u3": [8.0, 3.0],
            "u4": [-8.0, 1.0],
        },
    }
    metadata = {
        "u1": {"speaker_id": "s1", "dialect_label": "d1", "recording_condition": "clean"},
        "u2": {"speaker_id": "s1", "dialect_label": "d2", "recording_condition": "clean"},
        "u3": {"speaker_id": "s1", "dialect_label": "d3", "recording_condition": "clean"},
        "u4": {"speaker_id": "s2", "dialect_label": "d1", "recording_condition": "clean"},
    }

    report = run_cross_dialect_decomposition(embeddings=embeddings, metadata=metadata)

    assert report["schema"] == "speaker-offset-cross-dialect-decomposition-v1"
    assert len(report["models"]) == 3
    assert report["models"][0]["cross_dialect_speaker_count"] == 1
    assert report["models"][0]["eligible_speaker_count"] == 1
    assert report["models"][0]["speaker_means"]["s1"]["dialect_count"] == 3
    assert "delta_norm_ratio" in report["models"][0]["speaker_means"]["s1"]
    assert "speaker_mean_ratio_summary" in report["models"][0]
    assert "residual_cosine_summary" in report["models"][0]
    assert "group_a_mae_change" in report["models"][0]
    assert "group_c_mae_change" in report["models"][0]
