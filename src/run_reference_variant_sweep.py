"""Run a fixed HuBERT linear pair-only sweep over continuous references."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from .config import load_config
from .cross_dialect_gradient_isolation import _run_training_condition
from .estimand_sensitivity import WEIGHTINGS, weighted_mae
from .run_target_prevalence_mechanism import (
    _copy_config,
    _load_embeddings,
    _load_pairs,
    _load_records,
    _load_reference,
)


LOCKED_SEEDS = (20260829, 20260830, 20260831, 20260901, 20260902)
DEFAULT_PAIR_MANIFEST = Path("results/pairs/kespeech_evaluation_1000.json")
DEFAULT_PATHS = {
    "calibration_embedding": Path(
        "results/embeddings/kespeech_calibration_1000/chinese_hubert_large.json"
    ),
    "evaluation_embedding": Path(
        "results/embeddings/kespeech_evaluation_full/chinese_hubert_large.json"
    ),
    "records": Path("results/provenance/kespeech_manifest.json"),
    "calibration_pairs": Path("results/pairs/kespeech_calibration_1000.json"),
    "evaluation_pairs": DEFAULT_PAIR_MANIFEST,
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def make_variant_jobs(
    variants: Sequence[str], pair_manifest: Path = DEFAULT_PAIR_MANIFEST
) -> list[dict[str, Any]]:
    pair_hash = _sha256(pair_manifest)
    return [
        {
            "variant": str(variant),
            "seeds": LOCKED_SEEDS,
            "head_kind": "linear",
            "lambda_cross": 0.0,
            "lambda_dialect": 0.1,
            "learning_rate": 0.0003,
            "weight_decay": 0.001,
            "fixed_epochs": 29,
            "pair_manifest_hash": pair_hash,
            "selection_unit": "calibration_speaker_folds",
        }
        for variant in variants
    ]


def _estimands(rows: Sequence[Mapping[str, Any]]) -> dict[str, float | None]:
    values: dict[str, float | None] = {}
    for weighting in WEIGHTINGS:
        try:
            values[weighting] = weighted_mae(rows, weighting)
        except ValueError:
            if weighting != "matched_stratum":
                raise
            values[weighting] = None
    return values


def _weighted_result(run: Mapping[str, Any]) -> dict[str, Any]:
    baseline = _estimands(run["baseline_per_pair"])
    method = _estimands(run["per_pair"])
    gain = {
        weighting: (
            None
            if baseline[weighting] is None or method[weighting] is None
            else float(
                (float(baseline[weighting]) - float(method[weighting]))
                / float(baseline[weighting])
            )
        )
        for weighting in WEIGHTINGS
    }
    return {"baseline_mae": baseline, "method_mae": method, "gain": gain}


def _distribution(values: Sequence[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=float)
    return {
        "median": float(np.median(array)),
        "iqr": float(np.quantile(array, 0.75) - np.quantile(array, 0.25)),
        "minimum": float(np.min(array)),
        "maximum": float(np.max(array)),
    }


def run_sweep(
    config: Mapping[str, Any], matrix_paths: Sequence[Path]
) -> dict[str, Any]:
    variants = [path.stem.removeprefix("sinitic_data4_") for path in matrix_paths]
    jobs = make_variant_jobs(variants)
    calibration_embeddings = _load_embeddings(str(DEFAULT_PATHS["calibration_embedding"]))
    evaluation_embeddings = _load_embeddings(str(DEFAULT_PATHS["evaluation_embedding"]))
    records = _load_records(str(DEFAULT_PATHS["records"]))
    calibration_pairs = _load_pairs(str(DEFAULT_PATHS["calibration_pairs"]))
    evaluation_pairs = _load_pairs(str(DEFAULT_PATHS["evaluation_pairs"]))
    calibration_ids = set(calibration_embeddings)
    evaluation_ids = set(evaluation_embeddings)
    calibration_records = [
        row
        for row in records
        if str(row.get("split")) == "calibration"
        and str(row["utterance_id"]) in calibration_ids
    ]
    evaluation_records = [
        row
        for row in records
        if str(row.get("split")) == "evaluation"
        and str(row["utterance_id"]) in evaluation_ids
    ]
    calibration_pairs = [
        row
        for row in calibration_pairs
        if set(map(str, row["source_utterance_ids"])) <= calibration_ids
    ]
    evaluation_pairs = [
        row
        for row in evaluation_pairs
        if set(map(str, row["source_utterance_ids"])) <= evaluation_ids
    ]
    run_config = _copy_config(config, fixed_epochs=29)
    head = run_config["projection_head"]
    head["lambda_dialect_grid"] = [0.1]
    head["learning_rate_grid"] = [0.0003]
    head["weight_decay_grid"] = [0.001]
    outputs = []

    for job, matrix_path in zip(jobs, matrix_paths):
        reference = _load_reference(str(matrix_path))
        seed_rows = []
        for seed in job["seeds"]:
            run = _run_training_condition(
                calibration_embeddings=calibration_embeddings,
                evaluation_embeddings=evaluation_embeddings,
                calibration_records=calibration_records,
                evaluation_records=evaluation_records,
                calibration_pairs=calibration_pairs,
                evaluation_pairs=evaluation_pairs,
                reference=reference,
                config=run_config,
                seed=int(seed),
                lambda_cross=0.0,
                cross_examples=[],
                fixed_epochs=29,
                head_kind="linear",
            )
            seed_rows.append(
                {
                    "seed": int(seed),
                    "selected_epoch": 29,
                    **_weighted_result(run),
                }
            )
        outputs.append(
            {
                **job,
                "seeds": list(job["seeds"]),
                "matrix_path": str(matrix_path),
                "matrix_sha256": _sha256(matrix_path),
                "seed_results": seed_rows,
                "gain_distribution": {
                    weighting: _distribution(
                        [
                            float(row["gain"][weighting])
                            for row in seed_rows
                            if row["gain"][weighting] is not None
                        ]
                    )
                    if any(row["gain"][weighting] is not None for row in seed_rows)
                    else None
                    for weighting in WEIGHTINGS
                },
            }
        )
    return {
        "schema": "reference-variant-linear-sweep-v1",
        "status": "evaluated",
        "model_name": "chinese_hubert_large",
        "selection_policy": "fixed_before_evaluation_no_variant_selection",
        "pair_manifest": str(DEFAULT_PAIR_MANIFEST),
        "pair_manifest_sha256": _sha256(DEFAULT_PAIR_MANIFEST),
        "variants": outputs,
    }


def update_report_and_gate(
    report_path: Path, gate_path: Path, sweep: Mapping[str, Any]
) -> tuple[dict[str, Any], dict[str, Any]]:
    report = json.loads(report_path.read_text(encoding="utf-8"))
    report["projection_sweep"] = dict(sweep)
    positive_by_variant = {}
    for variant in sweep["variants"]:
        gains = variant["gain_distribution"]
        pair_seed_values = [row["gain"]["pair"] for row in variant["seed_results"]]
        positive_by_variant[variant["variant"]] = {
            "pair_weighted_median_positive": gains["pair"]["median"] > 0.0,
            "at_least_four_of_five_pair_seeds_positive": sum(
                float(value) > 0.0 for value in pair_seed_values
            )
            >= 4,
            "speaker_weighted_median_positive": gains["endpoint_speaker"]["median"]
            > 0.0,
            "relation_weighted_median_positive": gains["dialect_relation"]["median"]
            > 0.0,
            "matched_stratum_available": gains["matched_stratum"] is not None,
        }
    stable = all(
        row["pair_weighted_median_positive"]
        and row["at_least_four_of_five_pair_seeds_positive"]
        and row["speaker_weighted_median_positive"]
        and row["relation_weighted_median_positive"]
        for row in positive_by_variant.values()
    )
    report["status"] = "reference_stable" if stable else "construction_specific"
    gate = json.loads(gate_path.read_text(encoding="utf-8"))
    gate.update(
        {
            "phase": "construction_and_projection",
            "status": "passed" if stable else "failed",
            "projection_stability_status": report["status"],
            "variant_checks": positive_by_variant,
            "matched_stratum_note": "not defined in the locked A-D projection pair manifest",
        }
    )
    return report, gate


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--matrices", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--gate", type=Path, required=True)
    args = parser.parse_args()
    matrix_paths = [Path(value.strip()) for value in args.matrices.split(",") if value.strip()]
    sweep = run_sweep(load_config(args.config), matrix_paths)
    report, gate = update_report_and_gate(args.output, args.gate, sweep)
    args.output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    args.gate.write_text(
        json.dumps(gate, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
