from src.speaker_offset_cca import _probe_accuracy, run_speaker_offset_cca


def test_cca_report_includes_probe_accuracy_and_correlations():
    embeddings = {
        "m": {
            "u1": [2.0, 0.0],
            "u2": [2.0, 1.0],
            "u3": [-2.0, 0.0],
            "u4": [-2.0, 1.0],
        }
    }
    metadata = {
        "u1": {"speaker_id": "s1", "dialect_label": "d1", "recording_condition": "clean"},
        "u2": {"speaker_id": "s1", "dialect_label": "d2", "recording_condition": "clean"},
        "u3": {"speaker_id": "s2", "dialect_label": "d1", "recording_condition": "clean"},
        "u4": {"speaker_id": "s2", "dialect_label": "d2", "recording_condition": "clean"},
    }

    report = run_speaker_offset_cca(embeddings=embeddings, metadata=metadata)

    assert report["schema"] == "speaker-offset-cca-v1"
    assert report["models"][0]["speaker_probe_accuracy"] == 1.0
    assert report["models"][0]["dialect_probe_accuracy"] == 1.0
    assert len(report["models"][0]["canonical_correlations"]) > 0


def test_probe_accuracy_ignores_singleton_labels_instead_of_zeroing_out():
    x = [[float(index), 0.0] for index in range(10)]
    y = ["s1"] * 10
    x.extend([[100.0 + float(index), 0.0] for index in range(10)])
    y.extend(["s2"] * 10)
    x.append([1000.0, 0.0])
    y.append("singleton")

    accuracy = _probe_accuracy(x, y, seed=20260829)

    assert accuracy > 0.9
