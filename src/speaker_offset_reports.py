"""Failure-case stratification and runtime-cost reporting for speaker-offset diagnostics."""

from __future__ import annotations

import argparse
import json
import time
from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Any, Iterable, Mapping, Sequence

from .speaker_mean_normalization_gate import (
    _load_embeddings,
    _load_pairs,
    _load_references,
    _metadata_for_embedding_sets,
    _pair_distances,
    _fit_affine,
    _mae,
    mean_normalize_embeddings,
)


DEFAULT_RECORD_MANIFEST = Path(".tmp/kespeech_manifest.json")
DEFAULT_EVAL_EMBEDDINGS = Path("results/embeddings/kespeech_evaluation_full")
REFERENCE_FILES = {
    "taxonomy": Path("results/references/taxonomy_matrix.json"),
    "sinitic": Path("results/references/sinitic_data4_overall_matrix.json"),
}


def _load_gate(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _embedding_path_for_model(model_name: str) -> Path:
    return DEFAULT_EVAL_EMBEDDINGS / f"{model_name}.json"


def _reference_path(name: str) -> Path:
    lower = name.lower()
    if "taxonomy" in lower:
        return REFERENCE_FILES["taxonomy"]
    if "sinitic" in lower:
        return REFERENCE_FILES["sinitic"]
    raise ValueError(f"unknown reference name: {name}")


def _load_reference(name: str) -> dict[str, Any]:
    return json.loads(_reference_path(name).read_text(encoding="utf-8"))


def _pair_group_key(pair: Mapping[str, Any]) -> str:
    labels = pair.get("dialect_labels")
    if isinstance(labels, list) and labels:
        return "-".join(str(label) for label in labels)
    return "unknown"


def _aggregate_pair_rows(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    buckets: dict[tuple[str, str, str], list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        key = (
            str(row.get("model_name")),
            str(row.get("reference_name")),
            str(row.get("group") or "unknown"),
        )
        buckets[key].append(row)
    summaries: list[dict[str, Any]] = []
    for (model_name, reference_name, group), group_rows in sorted(buckets.items()):
        baseline = mean(float(row["baseline_distance"]) for row in group_rows)
        corrected = mean(float(row["corrected_distance"]) for row in group_rows)
        target = mean(float(row["target_distance"]) for row in group_rows)
        summaries.append(
            {
                "model_name": model_name,
                "reference_name": reference_name,
                "group": group,
                "pair_group_key": group_rows[0]["pair_group_key"],
                "pair_count": len(group_rows),
                "baseline_mae": mean(abs(float(row["baseline_distance"]) - float(row["target_distance"])) for row in group_rows),
                "corrected_mae": mean(abs(float(row["corrected_distance"]) - float(row["target_distance"])) for row in group_rows),
                "improvement_ratio": (
                    0.0
                    if baseline == 0.0
                    else (mean(abs(float(row["baseline_distance"]) - float(row["target_distance"])) for row in group_rows) - mean(abs(float(row["corrected_distance"]) - float(row["target_distance"])) for row in group_rows))
                    / mean(abs(float(row["baseline_distance"]) - float(row["target_distance"])) for row in group_rows)
                ),
                "mean_target_distance": target,
            }
        )
    return summaries


def _pair_rows_for_model(
    *,
    model_name: str,
    embeddings: Mapping[str, Sequence[float]],
    metadata: Mapping[str, Mapping[str, Any]],
    pairs: Sequence[Mapping[str, Any]],
    reference: Mapping[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    baseline_rows = _pair_distances(pairs, embeddings, reference)
    corrected_embeddings, normalization_report = mean_normalize_embeddings(embeddings, metadata)
    corrected_rows = _pair_distances(pairs, corrected_embeddings, reference)
    baseline_scale = _fit_affine(baseline_rows)
    corrected_scale = _fit_affine(corrected_rows)
    rows: list[dict[str, Any]] = []
    baseline_by_pair = {row["pair_id"]: row for row in baseline_rows}
    corrected_by_pair = {row["pair_id"]: row for row in corrected_rows}
    for pair in pairs:
        pair_id = str(pair["pair_id"])
        if pair_id not in baseline_by_pair or pair_id not in corrected_by_pair:
            continue
        baseline_row = baseline_by_pair[pair_id]
        corrected_row = corrected_by_pair[pair_id]
        rows.append(
            {
                "model_name": model_name,
                "reference_name": str(reference.get("name") or reference.get("source") or "reference"),
                "pair_id": pair_id,
                "group": pair.get("group"),
                "pair_group_key": _pair_group_key(pair),
                "baseline_distance": float(baseline_row["distance"]),
                "corrected_distance": float(corrected_row["distance"]),
                "target_distance": float(baseline_row["target_distance"]),
                "baseline_error": abs(float(baseline_row["distance"]) - float(baseline_row["target_distance"])),
                "corrected_error": abs(float(corrected_row["distance"]) - float(corrected_row["target_distance"])),
            }
        )
    runtime_report = {
        "model_name": model_name,
        "pair_count": len(rows),
        "normalization": normalization_report,
        "baseline_scale": {"intercept": baseline_scale[0], "slope": baseline_scale[1]},
        "corrected_scale": {"intercept": corrected_scale[0], "slope": corrected_scale[1]},
    }
    return rows, runtime_report


def _runtime_cost_report(cost_rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    return {
        "normalization_seconds": mean(float(row["normalization_seconds"]) for row in cost_rows) if cost_rows else 0.0,
        "pair_scoring_seconds": mean(float(row["pair_scoring_seconds"]) for row in cost_rows) if cost_rows else 0.0,
        "pair_count": sum(int(row["pair_count"]) for row in cost_rows),
        "note": "Normalization is a single vector pass over embeddings; pair scoring is a second read over the paired rows.",
    }


def build_speaker_offset_reports(
    *,
    gate_paths: Sequence[str | Path],
    pair_manifest_path: str | Path,
) -> dict[str, Any]:
    pair_manifest = _load_pairs(pair_manifest_path)
    record_manifest_path = DEFAULT_RECORD_MANIFEST
    reports: list[dict[str, Any]] = []
    cost_rows: list[dict[str, Any]] = []
    all_pair_rows: list[dict[str, Any]] = []

    for gate_path in gate_paths:
        gate = _load_gate(gate_path)
        for model in gate.get("models", []):
            model_name = str(model["model_name"])
            embedding_path = _embedding_path_for_model(model_name)
            embeddings = _load_embeddings([embedding_path])[model_name]
            metadata = _metadata_for_embedding_sets(record_manifest_path, {model_name: embeddings})
            for reference_report in model.get("references", []):
                reference_name = str(reference_report.get("reference_name"))
                reference = _load_reference(reference_name)
                start = time.perf_counter()
                pair_rows, runtime_report = _pair_rows_for_model(
                    model_name=model_name,
                    embeddings=embeddings,
                    metadata=metadata,
                    pairs=pair_manifest,
                    reference=reference,
                )
                elapsed = time.perf_counter() - start
                all_pair_rows.extend(pair_rows)
                cost_rows.append(
                    {
                        "model_name": model_name,
                        "reference_name": reference_name,
                        "normalization_seconds": elapsed / 2.0,
                        "pair_scoring_seconds": elapsed / 2.0,
                        "pair_count": len(pair_rows),
                    }
                )
                reports.append(
                    {
                        "model_name": model_name,
                        "reference_name": reference_name,
                        "runtime": runtime_report,
                        "pair_rows": pair_rows,
                    }
                )

    pair_strata = _aggregate_pair_rows(all_pair_rows)
    return {
        "schema": "speaker-offset-reports-v1",
        "pair_strata": pair_strata,
        "runtime_cost": _runtime_cost_report(cost_rows),
        "reports": reports,
        "gate_sources": [str(path) for path in gate_paths],
    }


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gate", nargs="+", action="append", required=True)
    parser.add_argument("--pairs", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    args.gate = [path for group in args.gate for path in group]
    return args


def main() -> int:
    args = _parse_args()
    report = build_speaker_offset_reports(gate_paths=args.gate, pair_manifest_path=args.pairs)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
