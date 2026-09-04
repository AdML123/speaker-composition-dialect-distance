from __future__ import annotations

import hashlib
import unicodedata
from pathlib import Path

import pytest

from src.prepare_kespeech_manifest import build_manifest


def _write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def test_build_manifest_uses_per_utterance_phase_metadata(tmp_path):
    metadata = tmp_path / "Metadata"
    tasks = tmp_path / "Tasks" / "ASR"
    _write(metadata / "phase1.utt2style", "u1 Dialect\n")
    _write(metadata / "phase2.utt2env", "")
    _write(metadata / "phase2.utt2env", "u2 noisy\n")

    for task in ["train_phase1", "train_phase2", "dev_phase1", "dev_phase2"]:
        _write(tasks / task / "wav.scp", "")
        _write(tasks / task / "utt2spk", "")
        _write(tasks / task / "utt2subdialect", "")

    _write(tasks / "test" / "wav.scp", "u1 Audio/s1/phase1/u1.wav\nu2 Audio/s1/phase2/u2.wav\n")
    _write(tasks / "test" / "utt2spk", "u1 s1\nu2 s1\n")
    _write(tasks / "test" / "utt2subdialect", "u1 Mandarin\nu2 Mandarin\n")

    manifest = build_manifest(metadata)

    conditions = {record["utterance_id"]: record["recording_condition"] for record in manifest["records"]}
    assert conditions == {"u1": "phase1:Dialect", "u2": "phase2:noisy"}
    assert {record["split"] for record in manifest["records"]} == {"evaluation"}


def test_build_manifest_adds_hashed_transcript_content_without_raw_text(tmp_path):
    metadata = tmp_path / "Metadata"
    tasks = tmp_path / "Tasks" / "ASR"
    _write(metadata / "phase1.utt2style", "u1 Dialect\n")
    _write(metadata / "phase2.utt2env", "")
    _write(metadata / "phase1.text", "u1 你好，世界\n")
    _write(tasks / "test" / "wav.scp", "u1 Audio/s1/phase1/u1.wav\n")
    _write(tasks / "test" / "utt2spk", "u1 s1\n")
    _write(tasks / "test" / "utt2subdialect", "u1 Mandarin\n")
    for task in ["train_phase1", "train_phase2", "dev_phase1", "dev_phase2"]:
        _write(tasks / task / "wav.scp", "")
        _write(tasks / task / "utt2spk", "")
        _write(tasks / task / "utt2subdialect", "")

    manifest = build_manifest(metadata)
    record = manifest["records"][0]

    normalized = " ".join(unicodedata.normalize("NFKC", "你好，世界").split())
    expected = hashlib.sha256(normalized.encode("utf-8")).hexdigest()
    assert record["content_id"] == expected
    assert "text" not in record
    assert manifest["metadata_capabilities"]["content_id"] is True


def test_build_manifest_fails_closed_when_discourse_is_required_but_unavailable(tmp_path):
    metadata = tmp_path / "Metadata"
    tasks = tmp_path / "Tasks" / "ASR"
    _write(metadata / "phase1.utt2style", "u1 Dialect\n")
    _write(tasks / "test" / "wav.scp", "u1 Audio/s1/phase1/u1.wav\n")
    _write(tasks / "test" / "utt2spk", "u1 s1\n")
    _write(tasks / "test" / "utt2subdialect", "u1 Mandarin\n")
    for task in ["train_phase1", "train_phase2", "dev_phase1", "dev_phase2"]:
        _write(tasks / task / "wav.scp", "")
        _write(tasks / task / "utt2spk", "")
        _write(tasks / task / "utt2subdialect", "")

    with pytest.raises(ValueError, match="discourse"):
        build_manifest(metadata, require_discourse=True)
