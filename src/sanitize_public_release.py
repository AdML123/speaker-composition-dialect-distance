"""Remove fields that would reconstruct an unlicensed continuous reference."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


REDACTION = {
    "schema": "public-release-redaction-v1",
    "reason": "upstream_license_not_specified",
    "scope": "fields_that_would_reconstruct_the_continuous_reference",
    "reconstruction_route": "obtain_pinned_upstream_commit_and_rebuild_locally",
}


def _without_key(value: Any, key: str) -> Any:
    if isinstance(value, dict):
        return {
            item_key: _without_key(item_value, key)
            for item_key, item_value in value.items()
            if item_key != key
        }
    if isinstance(value, list):
        return [_without_key(item, key) for item in value]
    return value


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def sanitize_release_reports(release_root: str | Path) -> dict[str, int]:
    analysis = Path(release_root) / "results" / "analysis"

    architecture_path = analysis / "architecture_cross_loss_factorial.json"
    architecture = _without_key(_load(architecture_path), "per_pair")
    architecture["public_release_redaction"] = {
        **REDACTION,
        "removed": ["cells[*].per_pair"],
    }
    _write(architecture_path, architecture)

    ranking_path = analysis / "metric_baseline_and_ranking.json"
    ranking = _without_key(_load(ranking_path), "per_pair")
    ranking["public_release_redaction"] = {
        **REDACTION,
        "removed": ["references.*.methods.*.per_pair", "seed_results[*].per_pair"],
    }
    _write(ranking_path, ranking)

    sensitivity_path = analysis / "reference_sensitivity_clustered.json"
    sensitivity = _load(sensitivity_path)
    continuous = sensitivity["references"]["continuous_sinitic"]["target_distribution"]
    continuous.pop("target_histogram", None)
    sensitivity["public_release_redaction"] = {
        **REDACTION,
        "removed": ["references.continuous_sinitic.target_distribution.target_histogram"],
    }
    _write(sensitivity_path, sensitivity)

    return {
        "architecture_cells": len(architecture["cells"]),
        "ranking_references": len(ranking["references"]),
        "sensitivity_references": len(sensitivity["references"]),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--release-root", required=True)
    args = parser.parse_args()
    print(json.dumps(sanitize_release_reports(args.release_root), indent=2))


if __name__ == "__main__":
    main()
