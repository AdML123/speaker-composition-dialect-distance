"""Validate locked speech models against a bounded ten-utterance smoke set."""

from __future__ import annotations

import argparse
import hashlib
import json
import wave
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch
from huggingface_hub import hf_hub_download
from transformers import AutoFeatureExtractor, AutoModel

from .config import ConfigError, load_config


EXPECTED_AUDIO_COUNT = 10
TARGET_SAMPLE_RATE = 16000


@dataclass(frozen=True)
class ModelSpec:
    name: str
    source: str
    revision: str
    provenance_record: str
    dimension: int
    status: str
    license: str | None = None
    checkpoint_hash: str | None = None
    checkpoint: str | None = None
    cache_root: str | None = None


def _read_nonempty_lines(path: Path) -> list[str]:
    try:
        lines = [line.strip() for line in path.read_text(encoding="utf-8").splitlines()]
    except OSError as exc:
        raise ConfigError(f"unable to read audio list: {exc}") from exc
    return [line for line in lines if line and not line.startswith("#")]


def load_audio_paths(path: str | Path) -> list[Path]:
    list_path = Path(path)
    entries = _read_nonempty_lines(list_path)
    if len(entries) != EXPECTED_AUDIO_COUNT:
        raise ConfigError(f"audio list must contain exactly {EXPECTED_AUDIO_COUNT} readable utterances")
    resolved: list[Path] = []
    for entry in entries:
        candidate = Path(entry)
        if not candidate.is_absolute():
            candidate = (list_path.parent / candidate).resolve()
        if not candidate.is_file():
            raise ConfigError(f"audio file is missing: {candidate}")
        resolved.append(candidate)
    return resolved


def load_mono_audio(path: str | Path) -> tuple[torch.Tensor, int]:
    try:
        with wave.open(str(path), "rb") as handle:
            sample_rate = handle.getframerate()
            channels = handle.getnchannels()
            sampwidth = handle.getsampwidth()
            nframes = handle.getnframes()
            frames = handle.readframes(nframes)
    except (OSError, wave.Error) as exc:
        raise ConfigError(f"invalid audio file: {path}") from exc

    if channels <= 0 or sampwidth not in {1, 2, 4}:
        raise ConfigError(f"unsupported audio format: {path}")
    if sampwidth == 1:
        dtype = np.uint8
        offset = 128.0
        scale = 128.0
    elif sampwidth == 2:
        dtype = np.int16
        offset = 0.0
        scale = 32768.0
    else:
        dtype = np.int32
        offset = 0.0
        scale = 2147483648.0

    audio = np.frombuffer(frames, dtype=dtype)
    if audio.size == 0:
        raise ConfigError(f"empty audio: {path}")
    audio = audio.reshape(-1, channels).astype(np.float32)
    if sampwidth == 1:
        audio = audio - offset
    audio = audio / scale
    waveform = torch.from_numpy(audio.T)
    if waveform.size(0) > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    if sample_rate != TARGET_SAMPLE_RATE:
        target_frames = max(1, round(waveform.size(-1) * TARGET_SAMPLE_RATE / sample_rate))
        waveform = torch.nn.functional.interpolate(
            waveform.unsqueeze(0), size=target_frames, mode="linear", align_corners=False
        ).squeeze(0)
        sample_rate = TARGET_SAMPLE_RATE
    waveform = waveform.squeeze(0).contiguous()
    if waveform.numel() == 0:
        raise ConfigError(f"empty audio: {path}")
    return waveform, sample_rate


def mean_pool_hidden_state(hidden_state: torch.Tensor) -> torch.Tensor:
    if hidden_state.ndim != 3:
        raise ConfigError("hidden_state must be [batch, frames, dim]")
    pooled = hidden_state.mean(dim=1)
    if pooled.ndim != 2 or pooled.shape[0] != 1:
        raise ConfigError("mean pooling must preserve batch size 1")
    return pooled.squeeze(0)


def _finite_tensor(tensor: torch.Tensor, label: str) -> None:
    if not torch.isfinite(tensor).all():
        raise ConfigError(f"non-finite values in {label}")


def _pretrained_kwargs(spec: ModelSpec) -> dict[str, Any]:
    source_path = Path(spec.source)
    if source_path.exists():
        return {"local_files_only": True}
    return {"revision": spec.revision, "cache_dir": spec.cache_root}


def _model_specs(config: dict[str, Any]) -> list[ModelSpec]:
    specs: list[ModelSpec] = []
    for entry in config["extractors"]["models"]:
        specs.append(
            ModelSpec(
                name=str(entry["name"]),
                source=str(entry["source"]),
                revision=str(entry["revision"]),
                provenance_record=str(entry["provenance_record"]),
                dimension=int(entry["dimension"]),
                status=str(entry["status"]),
                license=str(entry.get("license")) if entry.get("license") is not None else None,
                checkpoint_hash=str(entry.get("checkpoint_hash")) if entry.get("checkpoint_hash") is not None else None,
                cache_root=str(entry.get("cache_root")) if entry.get("cache_root") is not None else None,
            )
        )
    return specs


def _speaker_proxy_spec(config: dict[str, Any]) -> ModelSpec:
    entry = config["speaker_proxy"]
    return ModelSpec(
        name=str(entry["name"]),
        source=str(entry["source"]),
        revision=str(entry["revision"]),
        provenance_record=str(entry["provenance_record"]),
        dimension=int(entry["dimension"]),
        status=str(entry["status"]),
        license=str(entry.get("license")) if entry.get("license") is not None else None,
        checkpoint_hash=str(entry.get("checkpoint_hash")) if entry.get("checkpoint_hash") is not None else None,
        checkpoint=str(entry.get("checkpoint")) if entry.get("checkpoint") is not None else None,
        cache_root=str(entry.get("cache_root")) if entry.get("cache_root") is not None else None,
    )


def validate_model_spec(spec: ModelSpec) -> None:
    missing = [field for field in ("source", "revision", "provenance_record", "checkpoint_hash", "cache_root") if not getattr(spec, field)]
    if missing:
        raise ConfigError(f"model provenance missing fields: {', '.join(missing)}")
    if spec.dimension <= 0:
        raise ConfigError("model dimension must be positive")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _local_hub_cache_file(cache_root: str | None, repo_id: str, filename: str) -> Path | None:
    if not cache_root:
        return None
    repo_dir = Path(cache_root) / f"models--{repo_id.replace('/', '--')}"
    snapshots = repo_dir / "snapshots"
    if not snapshots.is_dir():
        return None
    for snapshot_dir in snapshots.iterdir():
        candidate = snapshot_dir / filename
        if candidate.is_file():
            return candidate
    return None


def verify_speaker_proxy_checkpoint(spec: ModelSpec) -> dict[str, Any]:
    validate_model_spec(spec)
    if not spec.checkpoint:
        raise ConfigError("speaker proxy checkpoint filename is missing")
    checkpoint_path = _local_hub_cache_file(spec.cache_root, spec.source, spec.checkpoint)
    if checkpoint_path is None:
        checkpoint_path = Path(
            hf_hub_download(
                repo_id=spec.source,
                filename=spec.checkpoint,
                revision=spec.revision,
                cache_dir=spec.cache_root,
            )
        )
    observed_hash = _sha256(checkpoint_path)
    if spec.checkpoint_hash and observed_hash != spec.checkpoint_hash:
        raise ConfigError("speaker proxy checkpoint hash mismatch")
    return {
        "source": spec.source,
        "revision": spec.revision,
        "checkpoint": spec.checkpoint,
        "checkpoint_hash": observed_hash,
        "cache_path": str(checkpoint_path),
        "dimension": spec.dimension,
        "license": spec.license,
    }


def smoke_transformer(spec: ModelSpec, audio_paths: Iterable[Path], device: torch.device) -> dict[str, Any]:
    validate_model_spec(spec)
    pretrained_kwargs = _pretrained_kwargs(spec)
    processor = AutoFeatureExtractor.from_pretrained(spec.source, **pretrained_kwargs)
    model = AutoModel.from_pretrained(spec.source, **pretrained_kwargs)
    model.eval().to(device)
    observed_dimension = int(getattr(model.config, "hidden_size"))
    if observed_dimension != spec.dimension:
        raise ConfigError(f"{spec.name} hidden size mismatch: expected {spec.dimension}, observed {observed_dimension}")

    embedding_norms: list[float] = []
    for audio_path in audio_paths:
        waveform, sample_rate = load_mono_audio(audio_path)
        if sample_rate != TARGET_SAMPLE_RATE:
            raise ConfigError("audio preprocessing failed to lock sample rate")
        inputs = processor(
            waveform.cpu().numpy(),
            sampling_rate=TARGET_SAMPLE_RATE,
            return_tensors="pt",
            padding=True,
        )
        inputs = {key: value.to(device) for key, value in inputs.items()}
        with torch.no_grad():
            outputs = model(**inputs)
        hidden_state = outputs.last_hidden_state
        pooled = mean_pool_hidden_state(hidden_state)
        _finite_tensor(pooled, spec.name)
        if pooled.numel() != spec.dimension:
            raise ConfigError(f"{spec.name} pooled dimension mismatch")
        embedding_norms.append(float(torch.linalg.vector_norm(pooled).item()))
    return {
        "name": spec.name,
        "source": spec.source,
        "revision": spec.revision,
        "checkpoint_hash": spec.checkpoint_hash,
        "dimension": spec.dimension,
        "license": spec.license,
        "cache_root": spec.cache_root,
        "audio_count": len(embedding_norms),
        "embedding_norms": embedding_norms,
        "device": str(device),
        "status": "passed",
    }


def _atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    tmp.replace(path)


def run_smoke(config_path: str | Path, audio_list_path: str | Path, output_path: str | Path) -> dict[str, Any]:
    config = load_config(config_path)
    audio_paths = load_audio_paths(audio_list_path)
    extractor_specs = _model_specs(config)
    speaker_proxy_spec = _speaker_proxy_spec(config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    extractor_reports = [smoke_transformer(spec, audio_paths, device) for spec in extractor_specs]
    speaker_report = verify_speaker_proxy_checkpoint(speaker_proxy_spec)

    payload = {
        "schema": "model-smoke-v1",
        "status": "passed",
        "audio_count": len(audio_paths),
        "audio_list": str(Path(audio_list_path)),
        "device": str(device),
        "extractors": extractor_reports,
        "speaker_proxy": speaker_report,
    }
    _atomic_write_json(Path(output_path), payload)
    return payload


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run the locked model smoke gate")
    parser.add_argument("--config", required=True)
    parser.add_argument("--audio-list", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    run_smoke(args.config, args.audio_list, args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
