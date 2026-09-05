"""Frozen ECAPA speaker-proxy embedding extraction for selected utterances."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Iterable, Mapping

import torch

from .config import load_config
from .embeddings import _resolve_external_audio_path, _selected_records
from .model_smoke import TARGET_SAMPLE_RATE, ModelSpec, _speaker_proxy_spec, load_mono_audio


def _load_classifier(spec: ModelSpec, device: torch.device, savedir: str | Path | None):
    from speechbrain.inference.speaker import EncoderClassifier
    from speechbrain.utils.fetching import FetchConfig, LocalStrategy

    return EncoderClassifier.from_hparams(
        source=spec.source,
        savedir=str(savedir or Path("DATASET_PATH_REDACTED/models/speechbrain-spkrec-ecapa-voxceleb")),
        run_opts={"device": str(device)},
        local_strategy=LocalStrategy.COPY,
        fetch_config=FetchConfig(
            allow_updates=False,
            revision=spec.revision,
            huggingface_cache_dir=spec.cache_root,
        ),
    )


def extract_ecapa_embeddings(
    spec: ModelSpec,
    records: Iterable[Mapping[str, Any]],
    *,
    audio_root: str | Path,
    device: torch.device,
    classifier: Any | None = None,
    savedir: str | Path | None = None,
) -> dict[str, torch.Tensor]:
    encoder = classifier or _load_classifier(spec, device, savedir)
    root = Path(audio_root)
    vectors: dict[str, torch.Tensor] = {}
    for record in records:
        utterance_id = str(record["utterance_id"])
        audio_path = _resolve_external_audio_path(str(record["audio_path"]), root)
        waveform, sample_rate = load_mono_audio(audio_path)
        if sample_rate != TARGET_SAMPLE_RATE:
            raise ValueError("audio preprocessing failed to lock sample rate")
        batch = waveform.to(device).unsqueeze(0)
        with torch.no_grad():
            encoded = encoder.encode_batch(batch)
        vector = encoded.detach().cpu().reshape(-1).float()
        if vector.numel() != spec.dimension or not torch.isfinite(vector).all():
            raise ValueError(f"invalid ECAPA embedding for {utterance_id}")
        vectors[utterance_id] = vector
    return vectors


def _write_embedding_file(path: Path, vectors: Mapping[str, torch.Tensor], spec: ModelSpec) -> dict[str, Any]:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema": "speaker-proxy-embeddings-v1",
        "model_name": spec.name,
        "source": spec.source,
        "revision": spec.revision,
        "checkpoint_hash": spec.checkpoint_hash,
        "dimension": spec.dimension,
        "embeddings": {key: value.tolist() for key, value in sorted(vectors.items())},
    }
    path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    return payload


def run_ecapa_embeddings(
    *,
    config_path: str | Path,
    pair_manifest_path: str | Path,
    record_manifest_path: str | Path,
    audio_root: str | Path,
    output_dir: str | Path,
    savedir: str | Path | None = None,
) -> dict[str, Any]:
    config = load_config(config_path)
    pair_manifest = json.loads(Path(pair_manifest_path).read_text(encoding="utf-8"))
    records = json.loads(Path(record_manifest_path).read_text(encoding="utf-8"))["records"]
    selected = _selected_records(pair_manifest, records)
    spec = _speaker_proxy_spec(config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    vectors = extract_ecapa_embeddings(spec, selected, audio_root=audio_root, device=device, savedir=savedir)
    output_path = Path(output_dir) / "ecapa.json"
    _write_embedding_file(output_path, vectors, spec)
    summary = {
        "schema": "speaker-proxy-embedding-summary-v1",
        "status": "passed",
        "model_name": spec.name,
        "embedding_path": str(output_path).replace("\\", "/"),
        "embedding_count": len(vectors),
        "dimension": spec.dimension,
        "device": str(device),
    }
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    (Path(output_dir) / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return summary


def _records_for_split(records: Iterable[Mapping[str, Any]], split: str) -> list[dict[str, Any]]:
    selected = [dict(record) for record in records if str(record.get("split")) == split]
    if not selected:
        raise ValueError(f"no records found for split: {split}")
    return sorted(selected, key=lambda record: str(record["utterance_id"]))


def run_ecapa_full_embeddings(
    *,
    config_path: str | Path,
    record_manifest_path: str | Path,
    audio_root: str | Path,
    output_dir: str | Path,
    split: str = "evaluation",
    savedir: str | Path | None = None,
) -> dict[str, Any]:
    config = load_config(config_path)
    records = json.loads(Path(record_manifest_path).read_text(encoding="utf-8"))["records"]
    selected = _records_for_split(records, split)
    spec = _speaker_proxy_spec(config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    vectors = extract_ecapa_embeddings(spec, selected, audio_root=audio_root, device=device, savedir=savedir)
    output_path = Path(output_dir) / "ecapa.json"
    _write_embedding_file(output_path, vectors, spec)
    summary = {
        "schema": "speaker-proxy-embedding-summary-v1",
        "status": "passed",
        "model_name": spec.name,
        "model_source": spec.source,
        "revision": spec.revision,
        "checkpoint_hash": spec.checkpoint_hash,
        "embedding_path": str(output_path).replace("\\", "/"),
        "embedding_count": len(vectors),
        "dimension": spec.dimension,
        "device": str(device),
        "source_record_split": split,
        "source_record_count": len(selected),
    }
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    (Path(output_dir) / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--pairs")
    parser.add_argument("--records", required=True)
    parser.add_argument("--audio-root", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--full", action="store_true")
    parser.add_argument("--split", default="evaluation")
    parser.add_argument("--savedir")
    args = parser.parse_args()
    if args.full:
        run_ecapa_full_embeddings(
            config_path=args.config,
            record_manifest_path=args.records,
            audio_root=args.audio_root,
            output_dir=args.output_dir,
            split=args.split,
            savedir=args.savedir,
        )
    else:
        if not args.pairs:
            raise ValueError("--pairs is required unless --full is set")
        run_ecapa_embeddings(
            config_path=args.config,
            pair_manifest_path=args.pairs,
            record_manifest_path=args.records,
            audio_root=args.audio_root,
            output_dir=args.output_dir,
            savedir=args.savedir,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
