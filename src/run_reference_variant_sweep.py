"""Run a fixed HuBERT linear pair-only sweep over continuous references."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from .config import load_config
from .architecture_factorial import _run_cell
from .cross_dialect_gradient_isolation import _run_training_condition
from .cross_dialect_projection_head import build_training_examples
from .estimand_sensitivity import WEIGHTINGS, weighted_mae
from .metric_baselines import (
    _available_pairs,
    _available_split,
    _fit_affine_for_vectors,
    _raw_and_pca,
    _score_vectors,
    _train_diagonal,
    _transform_diagonal,
)
from .run_target_prevalence_mechanism import (
    _copy_config,
    _load_embeddings,
    _load_pairs,
    _load_records,
    _load_reference,
)


LOCKED_SEEDS = (20260829, 20260830, 20260831, 20260901, 20260902)
REQUIRED_REFERENCES = (
    "taxonomy", "city_nearest", "subgroup_medoid", "subgroup_aggregate"
)
REQUIRED_METHODS = (
    "frozen_affine", "principal_component", "diagonal_metric",
    "linear", "matched_mlp", "wide_mlp",
)
ARCHITECTURES = {
    "linear": {
        "head": "linear", "head_kind": "linear", "hidden_dim": None,
        "dropout": 0.0,
    },
    "matched_mlp": {
        "head": "mlp_parameter_matched", "head_kind": "mlp",
        "hidden_dim": 205, "dropout": 0.0,
    },
    "wide_mlp": {
        "head": "mlp_wide", "head_kind": "mlp", "hidden_dim": 512,
        "dropout": 0.2,
    },
}
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


def make_architecture_jobs(variants: Sequence[str]) -> list[dict[str, Any]]:
    jobs = []
    for variant in variants:
        for seed in LOCKED_SEEDS:
            for method, architecture in ARCHITECTURES.items():
                jobs.append({
                    "reference": str(variant),
                    "method": method,
                    "seed": int(seed),
                    **architecture,
                    "lambda_cross": 0.0,
                    "lambda_dialect": 0.1,
                    "fixed_epochs": 30,
                    "epochs": 30,
                    "batch_size": 256,
                    "learning_rate": 0.0003,
                    "weight_decay": 0.001,
                    "early_stopping": False,
                    "schedule_key": hashlib.sha256(
                        f"{variant}|{seed}|pair-only-reference-variant".encode()
                    ).hexdigest(),
                })
    return jobs


def validate_complete_sweep(report: Mapping[str, Any]) -> None:
    references = report.get("references", {})
    if set(references) != set(REQUIRED_REFERENCES):
        raise ValueError("reference set is incomplete")
    for reference, reference_row in references.items():
        methods = reference_row.get("methods", {})
        if set(methods) != set(REQUIRED_METHODS):
            raise ValueError(f"method set is incomplete for {reference}")
        for method, method_row in methods.items():
            seed_results = method_row.get("seed_results", [])
            expected = 1 if method in {"frozen_affine", "principal_component"} else 5
            if len(seed_results) != expected:
                raise ValueError(f"seed count mismatch for {reference}:{method}")
            if any(len(row.get("per_pair", [])) != 4000 for row in seed_results):
                raise ValueError(f"pair count mismatch for {reference}:{method}")


def _normalized_primary_methods(
    baselines: Mapping[str, Any], architecture: Mapping[str, Any], reference: str
) -> dict[str, Any]:
    source = baselines["references"][reference]["methods"]
    methods = {
        "frozen_affine": {
            "seed_results": [{"seed": 0, "per_pair": source["frozen_affine"]["per_pair"]}]
        },
        "principal_component": {
            "seed_results": [{"seed": 0, "per_pair": source["frozen_pca256_affine"]["per_pair"]}]
        },
        "diagonal_metric": {
            "seed_results": [
                {"seed": int(row["seed"]), "per_pair": row["per_pair"]}
                for row in source["diagonal_metric"]["seed_results"]
            ]
        },
    }
    for method, head in (
        ("linear", "linear"),
        ("matched_mlp", "mlp_parameter_matched"),
        ("wide_mlp", "mlp_wide"),
    ):
        methods[method] = {
            "seed_results": [
                {"seed": int(row["seed"]), "per_pair": row["per_pair"]}
                for row in architecture["cells"]
                if str(row["reference"]) == reference
                and str(row["head"]) == head
                and float(row["lambda_cross"]) == 0.0
            ]
        }
    return methods


def run_complete_architecture_sweep(
    config: Mapping[str, Any],
    matrix_paths: Mapping[str, Path],
    architecture_path: Path,
    baselines_path: Path,
    checkpoint_root: Path,
) -> dict[str, Any]:
    architecture = json.loads(architecture_path.read_text(encoding="utf-8"))
    baselines = json.loads(baselines_path.read_text(encoding="utf-8"))
    calibration_embeddings = _load_embeddings(str(DEFAULT_PATHS["calibration_embedding"]))
    evaluation_embeddings = _load_embeddings(str(DEFAULT_PATHS["evaluation_embedding"]))
    records = _load_records(str(DEFAULT_PATHS["records"]))
    calibration_records = _available_split(records, calibration_embeddings, "calibration")
    evaluation_records = _available_split(records, evaluation_embeddings, "evaluation")
    calibration_pairs = _available_pairs(
        _load_pairs(str(DEFAULT_PATHS["calibration_pairs"])), calibration_embeddings
    )
    evaluation_pairs = _available_pairs(
        _load_pairs(str(DEFAULT_PATHS["evaluation_pairs"])), evaluation_embeddings
    )
    references = {
        name: _load_reference(str(path)) for name, path in matrix_paths.items()
    }
    output = {
        reference: {
            "reference_path": str(matrix_paths[reference]),
            "reference_sha256": _sha256(matrix_paths[reference]),
            "methods": _normalized_primary_methods(
                baselines, architecture, reference
            ),
        }
        for reference in ("taxonomy", "city_nearest")
    }
    frozen_vectors = _raw_and_pca(calibration_embeddings, evaluation_embeddings)
    checkpoint_root.mkdir(parents=True, exist_ok=True)
    for variant in ("subgroup_medoid", "subgroup_aggregate"):
        reference = references[variant]
        methods: dict[str, Any] = {}
        for source_name, result_name in (
            ("frozen_affine", "frozen_affine"),
            ("frozen_pca256_affine", "principal_component"),
        ):
            calibration_vectors, evaluation_vectors = frozen_vectors[source_name]
            affine = _fit_affine_for_vectors(
                calibration_pairs, calibration_records, calibration_vectors, reference
            )
            rows = _score_vectors(
                evaluation_pairs, evaluation_records, evaluation_vectors, reference, affine
            )
            methods[result_name] = {
                "seed_results": [{"seed": 0, "per_pair": rows}]
            }
        diagonal_rows = []
        for seed in LOCKED_SEEDS:
            _, state, _ = _train_diagonal(
                calibration_embeddings, calibration_pairs, calibration_records,
                reference, seed=int(seed), epochs=30, batch_size=256,
                learning_rate=0.0003, weight_decay=0.001,
            )
            vectors = _transform_diagonal(evaluation_embeddings, state)
            rows = _score_vectors(
                evaluation_pairs, evaluation_records, vectors, reference, state
            )
            diagonal_rows.append({"seed": int(seed), "per_pair": rows})
        methods["diagonal_metric"] = {"seed_results": diagonal_rows}

        examples = build_training_examples(calibration_records, calibration_pairs, reference)
        for job in make_architecture_jobs([variant]):
            method = str(job["method"])
            path = checkpoint_root / (
                f"{variant}__{job['seed']}__{method}__lambda_0.json"
            )
            if path.exists():
                cell = json.loads(path.read_text(encoding="utf-8"))
            else:
                cell = _run_cell(
                    job,
                    base_config=config,
                    calibration_embeddings=calibration_embeddings,
                    evaluation_embeddings=evaluation_embeddings,
                    calibration_records=calibration_records,
                    evaluation_records=evaluation_records,
                    calibration_pairs=calibration_pairs,
                    evaluation_pairs=evaluation_pairs,
                    reference=reference,
                    cross_examples=examples["cross_dialect_examples"],
                    pair_count=len(examples["pair_examples"]),
                )
                path.write_text(
                    json.dumps(cell, ensure_ascii=False, separators=(",", ":")) + "\n",
                    encoding="utf-8",
                )
            methods.setdefault(method, {"seed_results": []})["seed_results"].append({
                "seed": int(cell["seed"]), "per_pair": cell["per_pair"],
                "head_parameter_count": int(cell["head_parameter_count"]),
            })
        output[variant] = {
            "reference_path": str(matrix_paths[variant]),
            "reference_sha256": _sha256(matrix_paths[variant]),
            "methods": methods,
        }
    report = {
        "schema": "reference-variant-architecture-sweep-v1",
        "status": "evaluated",
        "model": "chinese_hubert_large",
        "pair_manifest": str(DEFAULT_PAIR_MANIFEST),
        "pair_manifest_sha256": _sha256(DEFAULT_PAIR_MANIFEST),
        "training_policy": "reference-specific pair-only training; fixed common budget",
        "references": output,
    }
    validate_complete_sweep(report)
    return report


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
    keep = (
        "pair_id", "group", "dialect_labels", "speaker_ids", "utterance_ids",
        "predicted_distance", "reference_distance", "absolute_error",
    )
    def compact(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
        output = []
        for row in rows:
            predicted = row.get("predicted_distance", row.get("distance"))
            reference = row.get("reference_distance", row.get("target"))
            checked = {key: row.get(key) for key in keep}
            checked["predicted_distance"] = float(predicted)
            checked["reference_distance"] = float(reference)
            checked["absolute_error"] = abs(float(predicted) - float(reference))
            output.append(checked)
        return output
    return {
        "baseline_mae": baseline,
        "method_mae": method,
        "gain": gain,
        "baseline_per_pair": compact(run["baseline_per_pair"]),
        "per_pair": compact(run["per_pair"]),
    }


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
    parser.add_argument("--output", type=Path)
    parser.add_argument("--gate", type=Path)
    parser.add_argument("--complete-output", type=Path)
    parser.add_argument("--architecture", type=Path)
    parser.add_argument("--baselines", type=Path)
    parser.add_argument("--checkpoint-root", type=Path)
    args = parser.parse_args()
    matrix_paths = [Path(value.strip()) for value in args.matrices.split(",") if value.strip()]
    if args.complete_output:
        if not args.architecture or not args.baselines or not args.checkpoint_root:
            parser.error(
                "--complete-output requires --architecture, --baselines, and --checkpoint-root"
            )
        by_name = {
            path.stem.removeprefix("sinitic_data4_"): path for path in matrix_paths
        }
        by_name.setdefault("taxonomy", Path("results/references/taxonomy_matrix.json"))
        report = run_complete_architecture_sweep(
            load_config(args.config), by_name, args.architecture, args.baselines,
            args.checkpoint_root,
        )
        args.complete_output.parent.mkdir(parents=True, exist_ok=True)
        args.complete_output.write_text(
            json.dumps(report, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        return 0
    if not args.output or not args.gate:
        parser.error("the legacy sweep requires --output and --gate")
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
