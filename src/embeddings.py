"""Frozen transformer embedding extraction for selected pilot utterances."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Iterable, Mapping

import torch
from transformers import AutoFeatureExtractor, AutoModel

from .config import load_config
from .model_smoke import ModelSpec, TARGET_SAMPLE_RATE, _model_specs, _pretrained_kwargs, load_mono_audio, mean_pool_hidden_state


def _resolve_external_audio_path(audio_path: str, audio_root: Path) -> Path:
    marker = "/KeSpeech/"
    normalized = audio_path.replace("\\", "/")
    if marker in normalized:
        relative = normalized.split(marker, 1)[1]
        return audio_root / Path(*relative.split("/"))
    return Path(audio_path)


def _selected_records(pair_manifest: Mapping[str, Any], records: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
    wanted = {
        utterance_id
        for pair in pair_manifest["pairs"]
        for utterance_id in pair["source_utterance_ids"]
    }
    by_id = {record["utterance_id"]: dict(record) for record in records}
    missing = sorted(wanted.difference(by_id))
    if missing:
        raise ValueError(f"pair manifest references missing utterance: {missing[0]}")
    return [by_id[utterance_id] for utterance_id in sorted(wanted)]


def extract_transformer_embeddings(
    spec: ModelSpec,
    records: Iterable[Mapping[str, Any]],
    *,
    audio_root: str | Path,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    processor = AutoFeatureExtractor.from_pretrained(spec.source, **_pretrained_kwargs(spec))
    model = AutoModel.from_pretrained(spec.source, **_pretrained_kwargs(spec))
    model.eval().to(device)
    observed_dimension = int(getattr(model.config, "hidden_size"))
    if observed_dimension != spec.dimension:
        raise ValueError(f"{spec.name} hidden size mismatch")

    vectors: dict[str, torch.Tensor] = {}
    root = Path(audio_root)
    for record in records:
        utterance_id = str(record["utterance_id"])
        audio_path = _resolve_external_audio_path(str(record["audio_path"]), root)
        waveform, sample_rate = load_mono_audio(audio_path)
        if sample_rate != TARGET_SAMPLE_RATE:
            raise ValueError("audio preprocessing failed to lock sample rate")
        inputs = processor(
            waveform.cpu().numpy(),
            sampling_rate=TARGET_SAMPLE_RATE,
            return_tensors="pt",
            padding=True,
        )
        inputs = {key: value.to(device) for key, value in inputs.items()}
        with torch.no_grad():
            outputs = model(**inputs)
        pooled = mean_pool_hidden_state(outputs.last_hidden_state).detach().cpu()
        if pooled.numel() != spec.dimension or not torch.isfinite(pooled).all():
            raise ValueError(f"invalid embedding for {utterance_id}")
        vectors[utterance_id] = pooled
    return vectors


def _write_embedding_file(path: Path, vectors: Mapping[str, torch.Tensor], spec: ModelSpec) -> dict[str, Any]:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema": "frozen-embeddings-v1",
        "model_name": spec.name,
        "source": spec.source,
        "revision": spec.revision,
        "checkpoint_hash": spec.checkpoint_hash,
        "dimension": spec.dimension,
        "embeddings": {key: value.tolist() for key, value in sorted(vectors.items())},
    }
    path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    return payload


def load_embedding_file(path: str | Path) -> dict[str, torch.Tensor]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    return {key: torch.tensor(value, dtype=torch.float32) for key, value in payload["embeddings"].items()}


def run_pilot_embeddings(
    *,
    config_path: str | Path,
    pair_manifest_path: str | Path,
    record_manifest_path: str | Path,
    audio_root: str | Path,
    output_dir: str | Path,
) -> dict[str, Any]:
    config = load_config(config_path)
    pairs = json.loads(Path(pair_manifest_path).read_text(encoding="utf-8"))
    records = json.loads(Path(record_manifest_path).read_text(encoding="utf-8"))["records"]
    selected = _selected_records(pairs, records)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    reports: list[dict[str, Any]] = []
    for spec in _model_specs(config):
        vectors = extract_transformer_embeddings(spec, selected, audio_root=audio_root, device=device)
        embedding_path = Path(output_dir) / f"{spec.name}.json"
        _write_embedding_file(embedding_path, vectors, spec)
        reports.append(
            {
                "model_name": spec.name,
                "embedding_path": str(embedding_path),
                "embedding_count": len(vectors),
                "dimension": spec.dimension,
                "device": str(device),
            }
        )
    summary = {"schema": "pilot-embedding-summary-v1", "status": "passed", "reports": reports}
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    (Path(output_dir) / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return summary


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--pairs", required=True)
    parser.add_argument("--records", required=True)
    parser.add_argument("--audio-root", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    run_pilot_embeddings(
        config_path=args.config,
        pair_manifest_path=args.pairs,
        record_manifest_path=args.records,
        audio_root=args.audio_root,
        output_dir=args.output_dir,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
