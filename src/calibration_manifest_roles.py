"""Identity and role checks for the locked KeSpeech pair manifests."""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any, Mapping

import yaml


_ROOT = Path(__file__).resolve().parents[1]
_DEFAULT_REGISTRY = _ROOT / "docs" / "revision" / "calibration-manifest-role-registry.yaml"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_role_registry(path: str | Path = _DEFAULT_REGISTRY) -> dict[str, Any]:
    """Load the explicit manifest-role registry keyed by repository-relative path."""
    registry_path = Path(path)
    payload = yaml.safe_load(registry_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or not isinstance(payload.get("manifests"), list):
        raise ValueError("role registry must contain a manifests list")
    registry: dict[str, Any] = {}
    for row in payload["manifests"]:
        if not isinstance(row, dict) or not row.get("path") or not row.get("role"):
            raise ValueError("each role registry row needs path and role")
        key = str(Path(str(row["path"])).as_posix())
        if key in registry:
            raise ValueError(f"duplicate registry path: {key}")
        registry[key] = dict(row)
    return {"schema": payload.get("schema"), "manifests": registry}


def _split_from_payload(payload: Mapping[str, Any], pairs: list[Mapping[str, Any]]) -> str:
    pair_splits = {str(row.get("split", "")) for row in pairs}
    pair_splits.discard("")
    if len(pair_splits) > 1:
        raise ValueError("mixed split in manifest pairs")
    if pair_splits:
        return next(iter(pair_splits))
    split_counts = payload.get("split_counts", {})
    if not isinstance(split_counts, Mapping):
        raise ValueError("manifest has no usable split metadata")
    active = [str(name) for name, count in split_counts.items() if int(count) > 0]
    if len(active) != 1:
        raise ValueError("manifest has mixed or missing split counts")
    return active[0]


def _repo_relative(path: Path) -> str:
    try:
        return path.resolve().relative_to(_ROOT.resolve()).as_posix()
    except ValueError:
        return path.as_posix()


def summarize_manifest(path: str | Path, registry_path: str | Path = _DEFAULT_REGISTRY) -> dict[str, Any]:
    """Validate one manifest and return its explicit role plus identity summary."""
    manifest_path = Path(path)
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or not isinstance(payload.get("pairs"), list):
        raise ValueError("manifest must contain a pairs list")
    pairs = [row for row in payload["pairs"] if isinstance(row, dict)]
    if len(pairs) != len(payload["pairs"]):
        raise ValueError("manifest pair rows must be objects")
    pair_ids = [str(row.get("pair_id", "")) for row in pairs]
    duplicate_ids = sorted(pair_id for pair_id, count in Counter(pair_ids).items() if not pair_id or count > 1)
    if duplicate_ids:
        raise ValueError(f"duplicate pair ID: {duplicate_ids[0]}")
    split = _split_from_payload(payload, pairs)
    groups = dict(sorted(Counter(str(row.get("group", "")) for row in pairs).items()))
    endpoint_speakers = {
        str(speaker)
        for row in pairs
        for speaker in row.get("speaker_ids", [])
    }
    split_speaker_count = int(payload.get("speaker_count") or payload.get("audit", {}).get("speaker_count") or len(endpoint_speakers))
    registry = load_role_registry(registry_path)["manifests"]
    role_row = registry.get(_repo_relative(manifest_path))
    if role_row is None:
        raise ValueError(f"manifest path is not registered: {_repo_relative(manifest_path)}")
    expected_split = str(role_row.get("split", ""))
    if split != expected_split:
        raise ValueError(f"registered split mismatch: {split} != {expected_split}")
    return {
        "path": _repo_relative(manifest_path),
        "role": str(role_row["role"]),
        "split": split,
        "pair_count": len(pairs),
        "groups": groups,
        "endpoint_speaker_count": len(endpoint_speakers),
        "split_speaker_count": split_speaker_count,
        "sha256": _sha256(manifest_path),
        "duplicate_pair_ids": duplicate_ids,
        "source_manifest_sha256": payload.get("source_manifest_sha256"),
        "used_by_lp": bool(role_row.get("used_by_lp", False)),
        "used_by_lc": bool(role_row.get("used_by_lc", False)),
        "used_by_model_selection": bool(role_row.get("used_by_model_selection", False)),
        "used_by_evaluation": bool(role_row.get("used_by_evaluation", False)),
    }


def assert_role(summary: Mapping[str, Any], expected: Mapping[str, Any]) -> None:
    """Raise with a focused message when a role summary differs from expectation."""
    for key, value in expected.items():
        if summary.get(key) != value:
            raise AssertionError(f"role mismatch for {key}: {summary.get(key)!r} != {value!r}")


def summarize_registered_manifests(registry_path: str | Path = _DEFAULT_REGISTRY) -> dict[str, Any]:
    """Summarize every registered manifest for a machine-readable identity report."""
    registry = load_role_registry(registry_path)
    rows = []
    for relative_path in sorted(registry["manifests"]):
        rows.append(summarize_manifest(_ROOT / relative_path, registry_path))
    return {"schema": "calibration-manifest-roles-v1", "manifests": rows}


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registry", type=Path, default=_DEFAULT_REGISTRY)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(summarize_registered_manifests(args.registry), indent=2) + "\n",
        encoding="utf-8",
    )
