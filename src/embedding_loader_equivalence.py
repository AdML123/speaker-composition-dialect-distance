"""Audit strict-loader embeddings against the locked pilot embedding files."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load(path: Path) -> dict[str, list[float]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    embeddings = payload.get("embeddings")
    if not isinstance(embeddings, dict) or not embeddings:
        raise ValueError(f"missing embeddings: {path}")
    return embeddings


def audit_loader_equivalence(
    locked_dir: str | Path,
    strict_dir: str | Path,
    output_path: str | Path,
) -> dict[str, Any]:
    locked_root = Path(locked_dir)
    strict_root = Path(strict_dir)
    models = sorted(path.stem for path in locked_root.glob("*.json") if path.name != "summary.json")
    if not models:
        raise ValueError("no locked pilot embeddings found")

    reports: list[dict[str, Any]] = []
    for model in models:
        locked_path = locked_root / f"{model}.json"
        strict_path = strict_root / f"{model}.json"
        if not strict_path.is_file():
            raise ValueError(f"missing strict-loader pilot embeddings: {strict_path}")
        locked = _load(locked_path)
        strict = _load(strict_path)
        if set(locked) != set(strict):
            raise ValueError(f"utterance IDs differ for {model}")
        maximum = max(
            float(
                np.max(
                    np.abs(
                        np.asarray(locked[utterance_id], dtype=np.float64)
                        - np.asarray(strict[utterance_id], dtype=np.float64)
                    )
                )
            )
            for utterance_id in locked
        )
        reports.append(
            {
                "model_name": model,
                "utterance_count": len(locked),
                "maximum_absolute_difference": maximum,
                "locked_file_sha256": _sha256(locked_path),
                "strict_file_sha256": _sha256(strict_path),
                "status": "equivalent" if maximum == 0.0 else "different",
            }
        )

    passed = all(report["status"] == "equivalent" for report in reports)
    payload = {
        "schema": "transformer-loader-equivalence-v1",
        "comparison": "locked-pilot-vectors-versus-strict-state-dict-loader",
        "exact_equality_required": True,
        "status": "passed" if passed else "failed",
        "models": reports,
    }
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    if not passed:
        raise ValueError("strict-loader pilot embeddings differ from locked embeddings")
    return payload


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--locked", required=True)
    parser.add_argument("--strict", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    audit_loader_equivalence(args.locked, args.strict, args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
