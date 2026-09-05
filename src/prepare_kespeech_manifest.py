"""Build a KeSpeech audit manifest from the official Kaldi metadata files."""

from __future__ import annotations

import argparse
import hashlib
import json
import unicodedata
from pathlib import Path
from typing import Any


TASK_SPLITS = {
    "train_phase1": "train",
    "train_phase2": "train",
    "dev_phase1": "calibration",
    "dev_phase2": "calibration",
    "test": "evaluation",
}


def _read_kv(path: Path) -> dict[str, str]:
    result: dict[str, str] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            fields = line.strip().split(maxsplit=1)
            if len(fields) == 2:
                result[fields[0]] = fields[1]
    return result


def _read_optional_kv(path: Path) -> dict[str, str] | None:
    return _read_kv(path) if path.is_file() else None


def _normalize_transcript(value: str) -> str:
    """Normalize transcript text before hashing, without retaining the text."""
    return " ".join(unicodedata.normalize("NFKC", value).split())


def _content_id(value: str) -> str:
    return hashlib.sha256(_normalize_transcript(value).encode("utf-8")).hexdigest()


def _records_for_task(
    metadata_root: Path,
    task_root: Path,
    task_name: str,
    *,
    transcript_maps: dict[str, dict[str, str] | None],
    discourse_map: dict[str, str] | None,
) -> list[dict[str, Any]]:
    task = task_root / task_name
    wav = _read_kv(task / "wav.scp")
    speakers = _read_kv(task / "utt2spk")
    dialects = _read_kv(task / "utt2subdialect")
    phase1_conditions = _read_kv(metadata_root / "phase1.utt2style")
    phase2_conditions = _read_kv(metadata_root / "phase2.utt2env")

    records: list[dict[str, Any]] = []
    split = TASK_SPLITS[task_name]
    for utterance_id in sorted(wav):
        relative_audio = wav[utterance_id]
        phase = "phase2" if "/phase2/" in f"/{relative_audio}" else "phase1"
        speaker_id = speakers.get(utterance_id)
        dialect_label = dialects.get(utterance_id)
        condition = (phase2_conditions if phase == "phase2" else phase1_conditions).get(utterance_id)
        if not speaker_id or not dialect_label or not condition:
            raise ValueError(f"incomplete metadata for {utterance_id} in {task_name}")
        records.append(
            {
                "utterance_id": utterance_id,
                "audio_path": f"DATASET_PATH_REDACTED/kespeech/KeSpeech/{relative_audio}",
                "speaker_id": speaker_id,
                "dialect_label": dialect_label,
                "sample_rate": 16000,
                "recording_condition": f"{phase}:{condition}",
                "split": split,
            }
        )
        transcript_map = transcript_maps.get(phase)
        if transcript_map is not None:
            transcript = transcript_map.get(utterance_id)
            if transcript is None:
                raise ValueError(f"missing transcript metadata for {utterance_id} in {task_name}")
            records[-1]["content_id"] = _content_id(transcript)
        if discourse_map is not None:
            discourse_level = discourse_map.get(utterance_id)
            if discourse_level is None:
                raise ValueError(f"missing discourse metadata for {utterance_id} in {task_name}")
            records[-1]["discourse_level"] = discourse_level
    return records


def build_manifest(metadata_root: str | Path, *, require_discourse: bool = False) -> dict[str, Any]:
    metadata = Path(metadata_root)
    task_root = metadata.parent / "Tasks" / "ASR"
    transcript_maps = {
        "phase1": _read_optional_kv(metadata / "phase1.text"),
        "phase2": _read_optional_kv(metadata / "phase2.text"),
    }
    discourse_candidates = (
        "discourse_level",
        "utt2discourse",
        "utt2session",
        "utt2dialogue",
    )
    discourse_path = next((metadata / name for name in discourse_candidates if (metadata / name).is_file()), None)
    if require_discourse and discourse_path is None:
        raise ValueError("discourse-level metadata was requested but is unavailable")
    discourse_map = _read_kv(discourse_path) if discourse_path is not None else None
    records: list[dict[str, Any]] = []
    for task_name in TASK_SPLITS:
        records.extend(
            _records_for_task(
                metadata,
                task_root,
                task_name,
                transcript_maps=transcript_maps,
                discourse_map=discourse_map,
            )
        )
    return {
        "schema": "corpus-manifest-v1",
        "corpus": "KeSpeech",
        "metadata_source": "KeSpeech/Metadata + KeSpeech/Tasks/ASR",
        "sample_rate_basis": "KeSpeech protocol metadata; verify selected WAV headers after extraction",
        "metadata_capabilities": {
            "content_id": any(value is not None for value in transcript_maps.values()),
            "discourse_level": discourse_map is not None,
            "raw_transcript_in_manifest": False,
        },
        "records": records,
    }


def write_manifest(metadata_root: str | Path, output_path: str | Path) -> None:
    payload = build_manifest(metadata_root)
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--metadata-root", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    write_manifest(args.metadata_root, args.output)


if __name__ == "__main__":
    main()
