import io
import json
import tarfile
from pathlib import Path

import pytest

from src.extract_kespeech_audio import ExtractionError, extract_selected_audio


def _write_split_tar(parts_dir: Path, members: dict[str, bytes]) -> None:
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w:gz") as archive:
        for name, content in members.items():
            info = tarfile.TarInfo(name)
            info.size = len(content)
            archive.addfile(info, io.BytesIO(content))
    blob = buffer.getvalue()
    midpoint = len(blob) // 2
    (parts_dir / "KeSpeech.tar.gz.aa").write_bytes(blob[:midpoint])
    (parts_dir / "KeSpeech.tar.gz.ab").write_bytes(blob[midpoint:])


def test_extract_selected_audio_reads_concatenated_split_archive(tmp_path):
    parts_dir = tmp_path / "parts"
    parts_dir.mkdir()
    _write_split_tar(
        parts_dir,
        {
            "KeSpeech/Audio/spk1/u1.wav": b"wav-1",
            "KeSpeech/Audio/spk2/u2.wav": b"wav-2",
            "KeSpeech/Audio/spk3/u3.wav": b"wav-3",
        },
    )
    records = {
        "records": [
            {"utterance_id": "u1", "audio_path": "DATA_ROOT/KeSpeech/Audio/spk1/u1.wav"},
            {"utterance_id": "u2", "audio_path": "DATA_ROOT/KeSpeech/Audio/spk2/u2.wav"},
        ]
    }
    pairs = {"pairs": [{"source_utterance_ids": ["u1", "u2"]}]}
    records_path = tmp_path / "records.json"
    pairs_path = tmp_path / "pairs.json"
    audit_path = tmp_path / "audit.json"
    records_path.write_text(json.dumps(records), encoding="utf-8")
    pairs_path.write_text(json.dumps(pairs), encoding="utf-8")

    audit = extract_selected_audio(
        parts_dir=parts_dir,
        pair_manifest_paths=[pairs_path],
        record_manifest_path=records_path,
        output_root=tmp_path / "extract",
        audit_output=audit_path,
    )

    assert audit["status"] == "passed"
    assert audit["selected_utterance_count"] == 2
    assert audit["extracted_count"] == 2
    assert (tmp_path / "extract" / "Audio" / "spk1" / "u1.wav").read_bytes() == b"wav-1"
    assert not (tmp_path / "extract" / "Audio" / "spk3" / "u3.wav").exists()


def test_extract_selected_audio_rejects_malformed_audio_path(tmp_path):
    parts_dir = tmp_path / "parts"
    parts_dir.mkdir()
    _write_split_tar(parts_dir, {"KeSpeech/Audio/spk1/u1.wav": b"wav-1"})
    records_path = tmp_path / "records.json"
    pairs_path = tmp_path / "pairs.json"
    records_path.write_text(
        json.dumps({"records": [{"utterance_id": "u1", "audio_path": "E:/outside/u1.wav"}]}),
        encoding="utf-8",
    )
    pairs_path.write_text(json.dumps({"pairs": [{"source_utterance_ids": ["u1"]}]}), encoding="utf-8")

    with pytest.raises(ExtractionError, match="KeSpeech"):
        extract_selected_audio(
            parts_dir=parts_dir,
            pair_manifest_paths=[pairs_path],
            record_manifest_path=records_path,
            output_root=tmp_path / "extract",
            audit_output=tmp_path / "audit.json",
        )
