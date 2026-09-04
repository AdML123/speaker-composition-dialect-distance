from __future__ import annotations

import wave
import hashlib
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import torchaudio

from src.config import ConfigError
from src.model_smoke import (
    ModelSpec,
    load_audio_paths,
    load_mono_audio,
    mean_pool_hidden_state,
    smoke_transformer,
    verify_speaker_proxy_checkpoint,
)


def _write_wav(path: Path, waveform: torch.Tensor, sample_rate: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if waveform.ndim == 1:
        waveform = waveform.unsqueeze(0)
    clipped = waveform.clamp(-1.0, 1.0)
    pcm16 = (clipped * 32767.0).to(torch.int16).transpose(0, 1).contiguous().numpy()
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(clipped.size(0))
        handle.setsampwidth(2)
        handle.setframerate(sample_rate)
        handle.writeframes(pcm16.tobytes())


def test_load_audio_paths_requires_exactly_ten_readable_utterances(tmp_path):
    audio_paths = []
    for index in range(9):
        path = tmp_path / f"{index}.wav"
        _write_wav(path, torch.zeros(1, 1600), 16000)
        audio_paths.append(path)
    list_path = tmp_path / "audio.txt"
    list_path.write_text("\n".join(str(path) for path in audio_paths), encoding="utf-8")

    with pytest.raises(ConfigError, match="exactly 10"):
        load_audio_paths(list_path)


def test_load_mono_audio_resamples_stereo_audio_to_16k(tmp_path):
    path = tmp_path / "stereo.wav"
    stereo = torch.stack([torch.linspace(-1, 1, 8000), torch.linspace(1, -1, 8000)])
    _write_wav(path, stereo, 8000)

    waveform, sample_rate = load_mono_audio(path)

    assert sample_rate == 16000
    assert waveform.ndim == 1
    assert waveform.numel() > 0


def test_mean_pool_hidden_state_returns_expected_shape_and_finite_values():
    hidden_state = torch.ones(1, 4, 1024)

    pooled = mean_pool_hidden_state(hidden_state)

    assert pooled.shape == (1024,)
    assert torch.isfinite(pooled).all()


def test_smoke_transformer_validates_dimension_and_finite_outputs(tmp_path, monkeypatch):
    audio_paths = []
    for index in range(10):
        path = tmp_path / f"{index}.wav"
        _write_wav(path, torch.zeros(1, 1600), 16000)
        audio_paths.append(path)

    class FakeProcessor:
        def __call__(self, waveform, sampling_rate, return_tensors, padding):
            assert sampling_rate == 16000
            assert return_tensors == "pt"
            assert padding is True
            return {"input_values": torch.ones(1, 1600)}

    class FakeModel:
        def __init__(self):
            self.config = SimpleNamespace(hidden_size=1024)

        def eval(self):
            return self

        def to(self, device):
            self.device = device
            return self

        def __call__(self, **inputs):
            assert "input_values" in inputs
            return SimpleNamespace(last_hidden_state=torch.ones(1, 3, 1024))

    monkeypatch.setattr("src.model_smoke.AutoFeatureExtractor.from_pretrained", lambda *args, **kwargs: FakeProcessor())
    monkeypatch.setattr("src.model_smoke.AutoModel.from_pretrained", lambda *args, **kwargs: FakeModel())

    spec = ModelSpec(
        name="wavlm_large",
        source="microsoft/wavlm-large",
        revision="f2443e98b0a97613614f31258463dcb3e95c904c",
        provenance_record="results/provenance/model_inventory.yaml",
        dimension=1024,
        status="locked",
        license="MIT",
        checkpoint_hash="f2443e98b0a97613614f31258463dcb3e95c904c",
        cache_root=str(tmp_path),
    )

    report = smoke_transformer(spec, audio_paths, torch.device("cpu"))

    assert report["audio_count"] == 10
    assert report["dimension"] == 1024
    assert report["status"] == "passed"
    assert len(report["embedding_norms"]) == 10


def test_smoke_transformer_uses_local_directory_without_hub_revision(tmp_path, monkeypatch):
    audio_paths = []
    for index in range(10):
        path = tmp_path / f"{index}.wav"
        _write_wav(path, torch.zeros(1, 1600), 16000)
        audio_paths.append(path)

    local_model_dir = tmp_path / "local-model"
    local_model_dir.mkdir()

    class FakeProcessor:
        def __call__(self, waveform, sampling_rate, return_tensors, padding):
            assert sampling_rate == 16000
            return {"input_values": torch.ones(1, 1600)}

    class FakeModel:
        def __init__(self):
            self.config = SimpleNamespace(hidden_size=1024)

        def eval(self):
            return self

        def to(self, device):
            self.device = device
            return self

        def __call__(self, **inputs):
            return SimpleNamespace(last_hidden_state=torch.ones(1, 3, 1024))

    def fake_processor_from_pretrained(source, **kwargs):
        assert source == str(local_model_dir)
        assert kwargs == {"local_files_only": True}
        return FakeProcessor()

    def fake_model_from_pretrained(source, **kwargs):
        assert source == str(local_model_dir)
        assert kwargs == {"local_files_only": True}
        return FakeModel()

    monkeypatch.setattr("src.model_smoke.AutoFeatureExtractor.from_pretrained", fake_processor_from_pretrained)
    monkeypatch.setattr("src.model_smoke.AutoModel.from_pretrained", fake_model_from_pretrained)

    spec = ModelSpec(
        name="wavlm_large",
        source=str(local_model_dir),
        revision="unused",
        provenance_record="results/provenance/model_inventory.yaml",
        dimension=1024,
        status="locked",
        license="MIT",
        checkpoint_hash="unused",
        cache_root=str(tmp_path),
    )

    report = smoke_transformer(spec, audio_paths, torch.device("cpu"))

    assert report["audio_count"] == 10
    assert report["status"] == "passed"


def test_verify_speaker_proxy_checkpoint_hash(tmp_path, monkeypatch):
    checkpoint = tmp_path / "embedding_model.ckpt"
    checkpoint.write_bytes(b"checkpoint-bytes")
    expected_hash = hashlib.sha256(checkpoint.read_bytes()).hexdigest()

    monkeypatch.setattr("src.model_smoke.hf_hub_download", lambda **kwargs: str(checkpoint))

    spec = ModelSpec(
        name="ecapa",
        source="speechbrain/spkrec-ecapa-voxceleb",
        revision="0f99f2d",
        provenance_record="results/provenance/model_inventory.yaml",
        dimension=192,
        status="locked",
        license="apache-2.0",
        checkpoint_hash=expected_hash,
        checkpoint="embedding_model.ckpt",
        cache_root=str(tmp_path),
    )

    report = verify_speaker_proxy_checkpoint(spec)

    assert report["checkpoint_hash"] == expected_hash
    assert report["dimension"] == 192
