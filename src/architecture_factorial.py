"""Run the parameter- and budget-matched architecture by cross-loss factorial."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import yaml

from .config import load_config
from .calibration_leakage_audit import audit_projection_sources
from .cross_dialect_gradient_isolation import _run_training_condition
from .cross_dialect_projection_head import (
    _make_projection_model,
    build_training_examples,
    parameter_count,
)
from .estimand_sensitivity import WEIGHTINGS, weighted_mae
from .paired_randomness import clustered_paired_bootstrap
from .projection_seed_sweep import _schedule_digest
from .run_target_prevalence_mechanism import (
    _load_embeddings,
    _load_pairs,
    _load_records,
    _load_reference,
)
from .statistical_gate import holm_adjust


SEEDS = (20260829, 20260830, 20260831, 20260901, 20260902)
REFERENCES = ("taxonomy", "city_nearest")
LAMBDA_CROSS = (0.0, 0.25, 0.5, 1.0)
HEADS = {
    "linear": {"head_kind": "linear", "hidden_dim": None, "dropout": 0.0},
    "mlp_parameter_matched": {
        "head_kind": "mlp",
        "hidden_dim": 205,
        "dropout": 0.0,
    },
    "mlp_wide": {"head_kind": "mlp", "hidden_dim": 512, "dropout": 0.2},
}
PATHS = {
    "calibration_embedding": Path(
        "results/embeddings/kespeech_calibration_1000/chinese_hubert_large.json"
    ),
    "evaluation_embedding": Path(
        "results/embeddings/kespeech_evaluation_full/chinese_hubert_large.json"
    ),
    "records": Path("results/provenance/kespeech_manifest.json"),
    "calibration_pairs": Path("results/pairs/kespeech_calibration_1000.json"),
    "evaluation_pairs": Path("results/pairs/kespeech_evaluation_1000.json"),
    "taxonomy": Path("results/references/taxonomy_matrix.json"),
    "city_nearest": Path("results/references/sinitic_data4_city_nearest.json"),
}
CHECKPOINT_ROOT = Path(
    ".tmp/estimand-reference-architecture-revision/architecture-factorial-cells"
)


def _load_protocol(path: Path) -> dict[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("protocol must be a YAML mapping")
    return payload


def _hash_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_factorial_jobs() -> list[dict[str, Any]]:
    jobs = []
    for reference in REFERENCES:
        for seed in SEEDS:
            schedule_key = _hash_text(f"{reference}|{seed}|common-pair-cross-schedule")
            for head, spec in HEADS.items():
                for lambda_cross in LAMBDA_CROSS:
                    jobs.append(
                        {
                            "reference": reference,
                            "seed": seed,
                            "head": head,
                            "head_kind": spec["head_kind"],
                            "hidden_dim": spec["hidden_dim"],
                            "dropout": spec["dropout"],
                            "lambda_cross": lambda_cross,
                            "lambda_dialect": 0.1,
                            "epochs": 30,
                            "batch_size": 256,
                            "learning_rate": 0.0003,
                            "weight_decay": 0.001,
                            "early_stopping": False,
                            "schedule_key": schedule_key,
                        }
                    )
    return jobs


def difference_in_differences(
    *,
    linear_zero: float,
    linear_added: float,
    mlp_zero: float,
    mlp_added: float,
) -> float:
    """Return the nonlinear minus linear cross-loss error improvement."""
    linear_increment = float(linear_zero) - float(linear_added)
    mlp_increment = float(mlp_zero) - float(mlp_added)
    return mlp_increment - linear_increment


def _estimands(rows: Sequence[Mapping[str, Any]]) -> dict[str, float | None]:
    output: dict[str, float | None] = {}
    for weighting in WEIGHTINGS:
        try:
            output[weighting] = weighted_mae(rows, weighting)
        except ValueError:
            if weighting != "matched_stratum":
                raise
            output[weighting] = None
    return output


def _compact_rows(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    keep = (
        "pair_id",
        "group",
        "dialect_labels",
        "speaker_ids",
        "utterance_ids",
        "matched_stratum",
        "predicted_distance",
        "reference_distance",
        "absolute_error",
    )
    return [{key: row.get(key) for key in keep} for row in rows]


def _cell_path(root: Path, job: Mapping[str, Any]) -> Path:
    lambda_token = str(job["lambda_cross"]).replace(".", "p")
    return root / (
        f"{job['reference']}__{job['seed']}__{job['head']}__lambda_{lambda_token}.json"
    )


def _run_cell(
    job: Mapping[str, Any],
    *,
    base_config: Mapping[str, Any],
    calibration_embeddings: Mapping[str, Sequence[float]],
    evaluation_embeddings: Mapping[str, Sequence[float]],
    calibration_records: Sequence[Mapping[str, Any]],
    evaluation_records: Sequence[Mapping[str, Any]],
    calibration_pairs: Sequence[Mapping[str, Any]],
    evaluation_pairs: Sequence[Mapping[str, Any]],
    reference: Mapping[str, Mapping[str, float]],
    cross_examples: Sequence[Mapping[str, Any]],
    pair_count: int,
    source_audit: Mapping[str, Any],
) -> dict[str, Any]:
    config = copy.deepcopy(dict(base_config))
    head_config = config["projection_head"]
    head_config.update(
        {
            "hidden_dim": int(job["hidden_dim"] or 512),
            "dropout": float(job["dropout"]),
            "max_epochs": int(job["epochs"]),
            "batch_size": int(job["batch_size"]),
            "lambda_cross_grid": [float(job["lambda_cross"])],
            "lambda_dialect_grid": [float(job["lambda_dialect"])],
            "learning_rate_grid": [float(job["learning_rate"])],
            "weight_decay_grid": [float(job["weight_decay"])],
        }
    )
    model = _make_projection_model(head_config, str(job["head_kind"]))
    head_parameters = parameter_count(model)
    start = time.perf_counter()
    run = _run_training_condition(
        calibration_embeddings=calibration_embeddings,
        evaluation_embeddings=evaluation_embeddings,
        calibration_records=calibration_records,
        evaluation_records=evaluation_records,
        calibration_pairs=calibration_pairs,
        evaluation_pairs=evaluation_pairs,
        reference=reference,
        config=config,
        seed=int(job["seed"]),
        lambda_cross=float(job["lambda_cross"]),
        cross_examples=cross_examples,
        fixed_epochs=int(job["epochs"]),
        head_kind=str(job["head_kind"]),
    )
    wall_seconds = time.perf_counter() - start
    composition = run["fitted"]["batch_composition"]
    schedule_digest = _schedule_digest(
        seed=int(job["seed"]),
        pair_count=pair_count,
        cross_count=len(cross_examples),
        epochs=int(job["epochs"]),
        n_pair=int(composition["n_pair"]),
        n_cross=int(composition["n_cross"]),
    )
    method_rows = _compact_rows(run["per_pair"])
    baseline_rows = _compact_rows(run["baseline_per_pair"])
    dialect_count = len(run["fitted"]["dialect_to_index"])
    classifier_parameters = (int(head_config["output_dim"]) + 1) * dialect_count
    return {
        **dict(job),
        "status": "complete",
        "head_parameter_count": head_parameters,
        "classifier_parameter_count": classifier_parameters,
        "wall_clock_seconds": wall_seconds,
        "actual_schedule_digest": schedule_digest,
        "evaluation_labels_used": False,
        "mae": float(run["mae"]),
        "gain": float(run["improvement_ratio"]),
        "calibration_mae_last_epoch": float(run["calibration_mae"]),
        "method_estimands": _estimands(method_rows),
        "baseline_estimands": _estimands(baseline_rows),
        "loss_history": run["fitted"]["loss_history"],
        "per_pair": method_rows,
        "calibration_source_audit": dict(source_audit),
    }


def _interaction_rows(
    cells: Sequence[Mapping[str, Any]], reference: str
) -> list[dict[str, Any]]:
    indexed = {
        (str(cell["head"]), float(cell["lambda_cross"]), int(cell["seed"])): cell
        for cell in cells
        if cell["reference"] == reference
    }
    by_pair: dict[str, list[dict[str, Any]]] = {}
    for seed in SEEDS:
        needed = {
            key: {
                str(row["pair_id"]): row
                for row in indexed[(head, weight, seed)]["per_pair"]
            }
            for key, head, weight in (
                ("linear_zero", "linear", 0.0),
                ("linear_added", "linear", 0.5),
                ("mlp_zero", "mlp_parameter_matched", 0.0),
                ("mlp_added", "mlp_parameter_matched", 0.5),
            )
        }
        common = set.intersection(*(set(rows) for rows in needed.values()))
        for pair_id in common:
            anchor = needed["linear_zero"][pair_id]
            delta = difference_in_differences(
                linear_zero=needed["linear_zero"][pair_id]["absolute_error"],
                linear_added=needed["linear_added"][pair_id]["absolute_error"],
                mlp_zero=needed["mlp_zero"][pair_id]["absolute_error"],
                mlp_added=needed["mlp_added"][pair_id]["absolute_error"],
            )
            by_pair.setdefault(pair_id, []).append(
                {
                    "delta": delta,
                    "group": anchor["group"],
                    "speaker_ids": anchor["speaker_ids"],
                }
            )
    return [
        {
            "pair_id": pair_id,
            "delta": float(np.mean([row["delta"] for row in rows])),
            "matched_stratum": str(rows[0]["group"]),
            "speaker_ids": rows[0]["speaker_ids"],
        }
        for pair_id, rows in sorted(by_pair.items())
        if len(rows) == len(SEEDS)
    ]


def _gate(cells: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    run_count_ok = len(cells) == 120
    schedule_groups: dict[tuple[str, int], set[str]] = {}
    for cell in cells:
        key = (str(cell["reference"]), int(cell["seed"]))
        schedule_groups.setdefault(key, set()).add(str(cell["actual_schedule_digest"]))
    schedule_ok = all(len(values) == 1 for values in schedule_groups.values())
    interactions = {}
    raw_p = {}
    for reference in REFERENCES:
        rows = _interaction_rows(cells, reference)
        result = clustered_paired_bootstrap(rows, seed=SEEDS[0], replicates=1000)
        interactions[reference] = result
        raw_p[reference] = float(result["bootstrap_tail_p_nonpositive"])
    adjusted = holm_adjust(raw_p)
    for reference, value in adjusted.items():
        interactions[reference]["holm_adjusted_p"] = value
        interactions[reference]["passed"] = bool(
            interactions[reference]["ci"]["lower"] > 0.0 and value < 0.05
        )
    passed = bool(
        run_count_ok
        and schedule_ok
        and all(not cell["evaluation_labels_used"] for cell in cells)
        and all(result["passed"] for result in interactions.values())
    )
    return {
        "schema": "architecture-factorial-gate-v1",
        "status": "passed" if passed else "failed",
        "run_count": len(cells),
        "run_count_ok": run_count_ok,
        "schedule_match": schedule_ok,
        "evaluation_labels_used": False,
        "primary_lambda_cross": 0.5,
        "interaction_definition": "(matched MLP error at lambda 0 minus lambda 0.5) minus (linear error at lambda 0 minus lambda 0.5)",
        "interactions": interactions,
        "failure_action": "describe_added_loss_as_capacity_or_optimization_contingent",
    }


def run_factorial(
    config: Mapping[str, Any], protocol: Mapping[str, Any], checkpoint_root: Path
) -> tuple[dict[str, Any], dict[str, Any]]:
    jobs = build_factorial_jobs()
    expected = protocol["architecture_factorial"]
    if expected["hidden_dims"] != [None, 205, 512]:
        raise ValueError("protocol hidden dimensions do not match runner")
    checkpoint_root.mkdir(parents=True, exist_ok=True)
    calibration_embeddings = _load_embeddings(str(PATHS["calibration_embedding"]))
    evaluation_embeddings = _load_embeddings(str(PATHS["evaluation_embedding"]))
    records = _load_records(str(PATHS["records"]))
    calibration_pairs = _load_pairs(str(PATHS["calibration_pairs"]))
    evaluation_pairs = _load_pairs(str(PATHS["evaluation_pairs"]))
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
    cells = []
    baseline_by_reference: dict[str, list[dict[str, Any]]] = {}
    references = {name: _load_reference(str(PATHS[name])) for name in REFERENCES}
    examples = {
        name: build_training_examples(calibration_records, calibration_pairs, reference)
        for name, reference in references.items()
    }
    source_audit = audit_projection_sources(
        calibration_pairs=calibration_pairs,
        cross_examples=examples[REFERENCES[0]]["cross_dialect_examples"],
        evaluation_pairs=evaluation_pairs,
        fitted_sources={
            "projection_calibration": calibration_pairs,
            "calibration_auxiliary_cross_pair": examples[REFERENCES[0]]["cross_dialect_examples"],
        },
    )
    for position, job in enumerate(jobs, start=1):
        path = _cell_path(checkpoint_root, job)
        if path.exists():
            cell = json.loads(path.read_text(encoding="utf-8"))
            cell.setdefault("calibration_source_audit", dict(source_audit))
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
                reference=references[str(job["reference"])],
                cross_examples=examples[str(job["reference"])]["cross_dialect_examples"],
                pair_count=len(examples[str(job["reference"])]["pair_examples"]),
                source_audit=source_audit,
            )
            path.write_text(
                json.dumps(cell, ensure_ascii=False, separators=(",", ":")) + "\n",
                encoding="utf-8",
            )
        cells.append(cell)
        if str(job["reference"]) not in baseline_by_reference:
            # The affine baseline is invariant to head and cross-loss weight.
            baseline_by_reference[str(job["reference"])] = []
        print(
            f"[{position:03d}/120] {job['reference']} {job['seed']} "
            f"{job['head']} lambda={job['lambda_cross']} gain={cell['gain']:.6f}",
            flush=True,
        )
    gate = _gate(cells)
    gate["calibration_source_audit"] = source_audit
    report = {
        "schema": "architecture-cross-loss-factorial-v1",
        "status": "evaluated",
        "protocol": dict(expected),
        "source_hashes": {
            key: _sha256(path) for key, path in PATHS.items() if path.is_file()
        },
        "run_count": len(cells),
        "cells": cells,
        "gate_summary": gate,
    }
    return report, gate


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--gate", type=Path, required=True)
    parser.add_argument("--checkpoint-root", type=Path, default=CHECKPOINT_ROOT)
    args = parser.parse_args()
    protocol = _load_protocol(args.protocol)
    report, gate = run_factorial(
        load_config(args.config), protocol, args.checkpoint_root
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.gate.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    args.gate.write_text(
        json.dumps(gate, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
