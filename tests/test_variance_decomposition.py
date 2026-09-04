from src.variance_decomposition import audit_variance_identifiability


def test_identifiability_fails_on_singleton_speaker_dialect_cell():
    report = audit_variance_identifiability([
        {"speaker_id": "s1", "dialect_label": "d1", "recording_condition": "c1"},
        {"speaker_id": "s1", "dialect_label": "d2", "recording_condition": "c2"},
        {"speaker_id": "s1", "dialect_label": "d2", "recording_condition": "c2"},
    ])
    assert report["status"] == "failed"
    assert report["singleton_cell_count"] == 1
    assert report["selected_wording"] == "factorial bookkeeping notation"


def test_identifiability_passes_when_cells_repeat_and_conditions_cross():
    rows = []
    for speaker in ("s1", "s2"):
        for dialect in ("d1", "d2"):
            for condition in ("c1", "c2"):
                rows.append({"speaker_id": speaker, "dialect_label": dialect, "recording_condition": condition})
    report = audit_variance_identifiability(rows)
    assert report["status"] == "passed"
