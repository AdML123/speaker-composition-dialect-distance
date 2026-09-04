"""Extract selected KeSpeech audio from split tar.gz fragments."""

from __future__ import annotations

import argparse
import json
import shutil
import tarfile
from pathlib import Path
from typing import Any, Iterable, Mapping


class ExtractionError(RuntimeError):
    """Raised when selected-audio extraction cannot be completed safely."""


class ConcatenatedFiles:
    """Minimal binary reader that streams a sequence of split archive files."""

    def __init__(self, paths: Iterable[Path]) -> None:
        self._paths = list(paths)
        if not self._paths:
            raise ExtractionError("no split archive parts found")
        self._index = 0
        self._handle = self._paths[0].open("rb")

    def read(self, size: int = -1) -> bytes:
        if size == 0:
            return b""
        chunks: list[bytes] = []
        remaining = size
        while self._handle:
            request_size = -1 if size < 0 else remaining
            chunk = self._handle.read(request_size)
            if chunk:
                chunks.append(chunk)
                if size > 0:
                    remaining -= len(chunk)
                    if remaining <= 0:
                        break
                continue
            self._handle.close()
            self._index += 1
            if self._index >= len(self._paths):
                self._handle = None
                break
            self._handle = self._paths[self._index].open("rb")
        return b"".join(chunks)

    def close(self) -> None:
        if self._handle:
            self._handle.close()
            self._handle = None


def _selected_utterance_ids(pair_manifest_paths: Iterable[str | Path]) -> set[str]:
    selected: set[str] = set()
    for path in pair_manifest_paths:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        for pair in payload.get("pairs", []):
            selected.update(str(utterance_id) for utterance_id in pair["source_utterance_ids"])
    return selected


def _relative_audio_path(audio_path: str) -> str:
    normalized = audio_path.replace("\\", "/")
    marker = "/KeSpeech/"
    if marker not in normalized:
        raise ExtractionError(f"audio_path is not under KeSpeech root: {audio_path}")
    relative = normalized.split(marker, 1)[1]
    if not relative.startswith("Audio/") or ".." in Path(relative).parts:
        raise ExtractionError(f"unsafe KeSpeech audio path: {audio_path}")
    return relative


def _selected_archive_members(record_manifest_path: str | Path, utterance_ids: set[str]) -> dict[str, str]:
    records = json.loads(Path(record_manifest_path).read_text(encoding="utf-8"))["records"]
    by_id = {str(record["utterance_id"]): record for record in records}
    missing = sorted(utterance_ids.difference(by_id))
    if missing:
        raise ExtractionError(f"pair manifest references missing utterance: {missing[0]}")
    members: dict[str, str] = {}
    for utterance_id in sorted(utterance_ids):
        relative = _relative_audio_path(str(by_id[utterance_id]["audio_path"]))
        members[f"KeSpeech/{relative}"] = relative
    return members


def _safe_output_path(output_root: Path, relative: str) -> Path:
    root = output_root.resolve()
    target = (output_root / Path(*relative.split("/"))).resolve()
    if root != target and root not in target.parents:
        raise ExtractionError(f"unsafe output path: {relative}")
    return target


def extract_selected_audio(
    *,
    parts_dir: str | Path,
    pair_manifest_paths: Iterable[str | Path],
    record_manifest_path: str | Path,
    output_root: str | Path,
    audit_output: str | Path,
) -> dict[str, Any]:
    parts = sorted(Path(parts_dir).glob("KeSpeech.tar.gz.*"))
    selected_ids = _selected_utterance_ids(pair_manifest_paths)
    wanted_members = _selected_archive_members(record_manifest_path, selected_ids)
    remaining = set(wanted_members)
    root = Path(output_root)
    extracted_count = 0
    skipped_existing_count = 0
    reader = ConcatenatedFiles(parts)
    try:
        with tarfile.open(fileobj=reader, mode="r|gz") as archive:
            for member in archive:
                if member.name not in remaining:
                    continue
                if not member.isfile():
                    raise ExtractionError(f"selected member is not a file: {member.name}")
                relative = wanted_members[member.name]
                target = _safe_output_path(root, relative)
                if target.exists() and target.stat().st_size == member.size:
                    skipped_existing_count += 1
                    remaining.remove(member.name)
                    continue
                source = archive.extractfile(member)
                if source is None:
                    raise ExtractionError(f"unable to read selected member: {member.name}")
                target.parent.mkdir(parents=True, exist_ok=True)
                tmp = target.with_name(f".{target.name}.tmp")
                with tmp.open("wb") as handle:
                    shutil.copyfileobj(source, handle)
                tmp.replace(target)
                extracted_count += 1
                remaining.remove(member.name)
                if not remaining:
                    break
    finally:
        reader.close()

    audit = {
        "schema": "kespeech-selected-audio-extraction-v1",
        "status": "passed" if not remaining else "failed",
        "parts_dir": str(Path(parts_dir)).replace("\\", "/"),
        "output_root": str(root).replace("\\", "/"),
        "selected_utterance_count": len(selected_ids),
        "selected_member_count": len(wanted_members),
        "extracted_count": extracted_count,
        "skipped_existing_count": skipped_existing_count,
        "missing_member_count": len(remaining),
        "missing_members": sorted(remaining)[:20],
    }
    Path(audit_output).parent.mkdir(parents=True, exist_ok=True)
    Path(audit_output).write_text(json.dumps(audit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    if remaining:
        raise ExtractionError(f"missing {len(remaining)} selected KeSpeech archive members")
    return audit


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parts-dir", required=True)
    parser.add_argument("--pairs", action="append", required=True)
    parser.add_argument("--records", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--audit-output", required=True)
    args = parser.parse_args()
    audit = extract_selected_audio(
        parts_dir=args.parts_dir,
        pair_manifest_paths=args.pairs,
        record_manifest_path=args.records,
        output_root=args.output_root,
        audit_output=args.audit_output,
    )
    print(json.dumps(audit, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
