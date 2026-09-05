import json
from pathlib import Path

import torch

from src.ecapa_embeddings import (
    _load_classifier,
    extract_ecapa_embeddings,
    run_ecapa_embeddings,
    run_ecapa_full_embeddings,
)
from src.model_smoke import ModelSpec


class FakeClassifier:
    def encode_batch(self, waveform):
        scale = float(waveform.numel())
        return torch.tensor([[[scale, scale + 1.0]]], dtype=torch.float32)


def test_load_classifier_pins_speechbrain_revision(monkeypatch, tmp_path):
    captured = {}

    def fake_from_hparams(**kwargs):
        captured.update(kwargs)
        return FakeClassifier()

    monkeypatch.setattr(
        "speechbrain.inference.speaker.EncoderClassifier.from_hparams",
        fake_from_hparams,
    )
    spec = ModelSpec(
        name="ecapa",
        source="speechbrain/spkrec-ecapa-voxceleb",
        revision="0f99f2d0ebe89ac095bcc5903c4dd8f72b367286",
        provenance_record="prov.yaml",
        dimension=192,
        status="locked",
        checkpoint="embedding_model.ckpt",
        checkpoint_hash="hash",
        cache_root=str(tmp_path),
    )
    _load_classifier(spec, torch.device("cpu"), tmp_path / "saved")
    assert captured["fetch_config"].revision == spec.revision


def test_extract_ecapa_embeddings_uses_injected_classifier(monkeypatch, tmp_path):
    audio_path = tmp_path / "u1.wav"
    audio_path.write_bytes(b"unused")

    def fake_load(path):
        assert path == audio_path
        return torch.ones(3), 16000

    monkeypatch.setattr("src.ecapa_embeddings.load_mono_audio", fake_load)
    spec = ModelSpec(
        name="ecapa",
        source="speechbrain/spkrec-ecapa-voxceleb",
        revision="rev",
        provenance_record="prov.yaml",
        dimension=2,
        status="locked",
        checkpoint_hash="hash",
    )

    vectors = extract_ecapa_embeddings(
        spec,
        [{"utterance_id": "u1", "audio_path": str(audio_path)}],
        audio_root=tmp_path,
        device=torch.device("cpu"),
        classifier=FakeClassifier(),
    )

    assert vectors["u1"].tolist() == [3.0, 4.0]


def test_run_ecapa_embeddings_writes_selected_vectors(monkeypatch, tmp_path):
    config_path = tmp_path / "config.yaml"
    records_path = tmp_path / "records.json"
    pairs_path = tmp_path / "pairs.json"
    output_dir = tmp_path / "out"
    config_path.write_text("unused: true\n", encoding="utf-8")
    records_path.write_text(
        json.dumps({"records": [{"utterance_id": "u1", "audio_path": "E:/root/KeSpeech/Audio/s/u1.wav"}]}),
        encoding="utf-8",
    )
    pairs_path.write_text(json.dumps({"pairs": [{"source_utterance_ids": ["u1"]}]}), encoding="utf-8")

    def fake_extract(spec, records, *, audio_root, device, classifier=None, savedir=None):
        return {"u1": torch.tensor([1.0, 2.0])}

    monkeypatch.setattr(
        "src.ecapa_embeddings.load_config",
        lambda _: {
            "speaker_proxy": {
                "name": "ecapa",
                "source": "speechbrain/spkrec-ecapa-voxceleb",
                "status": "locked",
                "checkpoint": "embedding_model.ckpt",
                "revision": "rev",
                "checkpoint_hash": "hash",
                "provenance_record": "prov.yaml",
                "dimension": 2,
            }
        },
    )
    monkeypatch.setattr("src.ecapa_embeddings.extract_ecapa_embeddings", fake_extract)

    summary = run_ecapa_embeddings(
        config_path=config_path,
        pair_manifest_path=pairs_path,
        record_manifest_path=records_path,
        audio_root=tmp_path,
        output_dir=output_dir,
    )

    assert summary["status"] == "passed"
    payload = json.loads((output_dir / "ecapa.json").read_text(encoding="utf-8"))
    assert payload["embeddings"]["u1"] == [1.0, 2.0]


def test_run_ecapa_full_embeddings_uses_evaluation_split_only(monkeypatch, tmp_path):
    config_path = tmp_path / "config.yaml"
    records_path = tmp_path / "records.json"
    output_dir = tmp_path / "out"
    config_path.write_text("unused: true\n", encoding="utf-8")
    records_path.write_text(
        json.dumps(
            {
                "records": [
                    {"utterance_id": "u1", "split": "evaluation", "audio_path": "E:/root/KeSpeech/Audio/s/u1.wav"},
                    {"utterance_id": "u2", "split": "train", "audio_path": "E:/root/KeSpeech/Audio/s/u2.wav"},
                    {"utterance_id": "u3", "split": "evaluation", "audio_path": "E:/root/KeSpeech/Audio/s/u3.wav"},
                ]
            }
        ),
        encoding="utf-8",
    )

    seen = {}

    def fake_extract(spec, records, *, audio_root, device, classifier=None, savedir=None):
        seen["utterance_ids"] = [record["utterance_id"] for record in records]
        return {record["utterance_id"]: torch.tensor([1.0, 2.0]) for record in records}

    monkeypatch.setattr("src.ecapa_embeddings.extract_ecapa_embeddings", fake_extract)
    monkeypatch.setattr(
        "src.ecapa_embeddings._speaker_proxy_spec",
        lambda config: ModelSpec(
            name="ecapa",
            source="speechbrain/spkrec-ecapa-voxceleb",
            revision="rev",
            provenance_record="prov.yaml",
            dimension=2,
            status="locked",
            checkpoint_hash="hash",
            checkpoint="embedding_model.ckpt",
            cache_root="cache",
        ),
    )
    monkeypatch.setattr(
        "src.ecapa_embeddings.load_config",
        lambda _: {
            "speaker_proxy": {
                "name": "ecapa",
                "source": "speechbrain/spkrec-ecapa-voxceleb",
                "status": "locked",
                "checkpoint": "embedding_model.ckpt",
                "revision": "rev",
                "checkpoint_hash": "hash",
                "provenance_record": "prov.yaml",
                "dimension": 2,
            }
        },
    )

    summary = run_ecapa_full_embeddings(
        config_path=config_path,
        record_manifest_path=records_path,
        audio_root=tmp_path,
        output_dir=output_dir,
    )

    assert seen["utterance_ids"] == ["u1", "u3"]
    assert summary["status"] == "passed"
    assert summary["embedding_count"] == 2
    payload = json.loads((output_dir / "ecapa.json").read_text(encoding="utf-8"))
    assert sorted(payload["embeddings"]) == ["u1", "u3"]
