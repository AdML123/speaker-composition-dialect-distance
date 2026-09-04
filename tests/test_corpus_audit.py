import json
import math
from pathlib import Path

import pytest

from src.corpus_audit import (
    AuditError,
    audit_manifest,
    compute_pair_capacity,
    hierarchical_bootstrap_effect,
    load_manifest,
)


def record(uid, speaker, dialect, split, audio="audio.wav"):
    return {
        "utterance_id": uid,
        "audio_path": audio,
        "speaker_id": speaker,
        "dialect_label": dialect,
        "sample_rate": 16000,
        "recording_condition": "phone-near",
        "split": split,
    }


def test_pair_capacity_counts_abcd_in_linear_time_contract():
    records = [
        record("u1", "s1", "d1", "train"),
        record("u2", "s1", "d1", "train"),
        record("u3", "s1", "d2", "train"),
        record("u4", "s2", "d1", "train"),
    ]

    assert compute_pair_capacity(records) == {"A": 1, "B": 2, "C": 2, "D": 1}


def test_manifest_rejects_missing_fields_duplicates_and_root_escape(tmp_path):
    root = tmp_path / "audio"
    root.mkdir()
    valid = record("u1", "s1", "d1", "train", str(root / "u1.wav"))
    (root / "u1.wav").write_bytes(b"RIFF")

    with pytest.raises(AuditError, match="missing required field"):
        audit_manifest([{k: v for k, v in valid.items() if k != "speaker_id"}], root)
    with pytest.raises(AuditError, match="duplicate utterance_id"):
        audit_manifest([valid, valid], root)

    escaped = dict(valid, utterance_id="u2", audio_path=str(tmp_path / "outside.wav"))
    with pytest.raises(AuditError, match="outside validated audio root"):
        audit_manifest([escaped], root)


def test_manifest_requires_speaker_disjoint_splits_and_license_booleans(tmp_path):
    root = tmp_path / "audio"
    root.mkdir()
    (root / "u1.wav").write_bytes(b"RIFF")
    (root / "u2.wav").write_bytes(b"RIFF")
    records = [
        record("u1", "s1", "d1", "train", str(root / "u1.wav")),
        record("u2", "s1", "d1", "evaluation", str(root / "u2.wav")),
    ]
    with pytest.raises(AuditError, match="speaker leakage"):
        audit_manifest(records, root, legally_usable_for_research=True, raw_audio_redistributable=False)
    with pytest.raises(AuditError, match="must be boolean"):
        audit_manifest([records[0]], root, legally_usable_for_research="yes", raw_audio_redistributable=False)


def test_audit_report_redacts_paths_and_keeps_gate_pending(tmp_path):
    root = tmp_path / "audio"
    root.mkdir()
    records = [record("u1", "s1", "d1", "train", str(root / "u1.wav"))]
    (root / "u1.wav").write_bytes(b"RIFF")

    report = audit_manifest(
        records,
        root,
        legally_usable_for_research=False,
        raw_audio_redistributable=False,
        minimum_dialects=8,
        minimum_pairs=200,
    )

    serialized = json.dumps(report)
    assert str(tmp_path) not in serialized
    assert report["local_input_path"] == "<local>"
    assert report["status"] == "failed"
    assert report["speaker_label_consistency"]["global_mixed_effect"]["status"] == "pending"
    assert report["speaker_label_consistency"]["per_dialect"] == {}


def test_research_use_can_pass_when_raw_audio_redistribution_is_restricted(tmp_path):
    root = tmp_path / "audio"
    root.mkdir()
    records = [record("u1", "s1", "d1", "train", str(root / "u1.wav")),
               record("u2", "s1", "d1", "train", str(root / "u2.wav")),
               record("u3", "s1", "d2", "train", str(root / "u3.wav")),
               record("u4", "s2", "d1", "train", str(root / "u4.wav")),
               record("u5", "s2", "d2", "train", str(root / "u5.wav"))]
    for record_item in records:
        Path(record_item["audio_path"]).write_bytes(b"RIFF")

    report = audit_manifest(records, root, legally_usable_for_research=True,
                            raw_audio_redistributable=False,
                            minimum_dialects=1, minimum_pairs=1)

    assert report["status"] == "passed"
    assert report["raw_audio_classification"] == "research_only"
    assert "raw_audio_redistributable" not in report["failed_checks"]


def test_load_manifest_rejects_malformed_records(tmp_path):
    path = tmp_path / "manifest.json"
    path.write_text("not json", encoding="utf-8")
    with pytest.raises(AuditError, match="malformed manifest"):
        load_manifest(path)


def test_three_level_bootstrap_is_complete_finite_and_dialect_reported():
    rows = []
    for dialect_index, dialect in enumerate(("d1", "d2")):
        for speaker_index in range(4):
            speaker = f"{dialect}-s{speaker_index}"
            rows.append({"dialect": dialect, "speaker_ids": [speaker], "group": "within", "distance": 0.1 + dialect_index * 0.01})
            rows.append({"dialect": dialect, "speaker_ids": [speaker, f"{dialect}-other"], "group": "between", "distance": 0.4 + dialect_index * 0.01})

    result = hierarchical_bootstrap_effect(rows, replicates=1000, seed=20260829)

    assert result["effective_replicates"] == 1000
    assert len(result["replicates"]) == 1000
    assert all(math.isfinite(value) for value in result["replicates"])
    assert result["ci"]["lower"] > 0
    assert set(result["per_dialect"]) == {"d1", "d2"}
    assert result["per_dialect"]["d1"]["speaker_endpoint_count"] >= 5


def test_bootstrap_rejects_between_rows_without_both_speaker_endpoints():
    rows = [{"dialect": "d1", "speaker_id": "s1", "group": "within", "distance": 0.1},
            {"dialect": "d1", "speaker_id": "s1", "group": "between", "distance": 0.4}]
    with pytest.raises(AuditError, match="speaker_ids"):
        hierarchical_bootstrap_effect(rows, replicates=1000, seed=1)


def test_bootstrap_is_invariant_to_between_pair_endpoint_order():
    rows = [
        {"dialect": "d1", "speaker_ids": ["s1"], "group": "within", "distance": 0.1},
        {"dialect": "d1", "speaker_ids": ["s2"], "group": "within", "distance": 0.2},
        {"dialect": "d1", "speaker_ids": ["s1", "s2"], "group": "between", "distance": 0.6},
    ]
    reversed_rows = [dict(row, speaker_ids=list(reversed(row["speaker_ids"]))) for row in rows]

    first = hierarchical_bootstrap_effect(rows, replicates=1000, seed=7)
    second = hierarchical_bootstrap_effect(reversed_rows, replicates=1000, seed=7)

    assert first["replicates"] == second["replicates"]


def test_bootstrap_rejects_too_few_requested_replicates():
    with pytest.raises(AuditError, match="at least 1000"):
        hierarchical_bootstrap_effect([], replicates=999, seed=1)
