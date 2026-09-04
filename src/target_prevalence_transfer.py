"""Evaluate prevalence balancing on a frozen cross-backbone condition."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from .config import load_config
from .cross_dialect_gradient_isolation import _run_training_condition
from .cross_dialect_projection_head import build_training_examples
from .run_target_prevalence_mechanism import (
    _copy_config,
    _gain_by_speaker,
    _load_embeddings,
    _load_pairs,
    _load_records,
    _load_reference,
    build_slope_rows,
    seed_slope_distribution,
)
from .target_prevalence_experiment import (
    TARGET_PREVALENCE_GRID,
    build_nested_fixed_pair_masks,
    clustered_mean_gain_contrast,
    clustered_slope_bootstrap,
)


LOCKED_SEEDS = (20260829, 20260830, 20260831, 20260901, 20260902)


def _sha(payload: object) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _condition_gate(condition: Mapping[str, Any]) -> dict[str, bool]:
    slope = condition["slope_contrast"]
    mean = condition["mean_gain_contrast"]
    seeds = condition["seed_slope_distribution"]
    return {
        "q_sensitivity_reduced": bool(float(slope["ci"]["lower"]) > 0),
        "mean_gain_improved": bool(float(mean["ci"]["lower"]) > 0),
        "all_seed_slope_contrasts_positive": bool(seeds["all_positive"]),
    }


def evaluate_transfer_gate(
    primary: Mapping[str, Any],
    orthogonal: Mapping[str, Any],
) -> dict[str, Any]:
    primary_gate = _condition_gate(primary)
    orthogonal_gate = _condition_gate(orthogonal)
    passed = all(primary_gate.values()) and all(orthogonal_gate.values())
    return {
        "primary": primary_gate,
        "orthogonal": orthogonal_gate,
        "rule_transfer_supported": bool(passed),
        "selected_wording": (
            "Prevalence balancing transferred across the tested backbone under the locked rule."
            if passed
            else "Target prevalence is associated with the loss-structure effect under the tested configuration; a transferable balancing rule was not established."
        ),
    }


def _seed_mode_summary(q_cells: Mapping[str, Any], q_keys: Sequence[str]) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for mode in ("ordinary", "prevalence_balanced"):
        by_seed: dict[int, list[float]] = {}
        for q_key in q_keys:
            for row in q_cells[q_key][mode]["seed_results"]:
                by_seed.setdefault(int(row["seed"]), []).append(float(row["gain"]))
        values = np.asarray([np.mean(by_seed[seed]) for seed in sorted(by_seed)], dtype=np.float64)
        output[mode] = {
            "seed_count": len(values),
            "median_gain_across_internal_q": float(np.median(values)),
            "iqr": float(np.quantile(values, 0.75) - np.quantile(values, 0.25)),
            "minimum": float(np.min(values)),
            "maximum": float(np.max(values)),
        }
    return output


def run_fixed_pair_transfer(
    *,
    config: Mapping[str, Any],
    calibration_embeddings: Mapping[str, Sequence[float]],
    evaluation_embeddings: Mapping[str, Sequence[float]],
    records: Sequence[Mapping[str, Any]],
    calibration_pairs: Sequence[Mapping[str, Any]],
    evaluation_pairs: Sequence[Mapping[str, Any]],
    reference: Mapping[str, Mapping[str, float]],
    expected_pair_identity_hash: str,
    fixed_epochs: int,
    lambda_cross: float,
) -> dict[str, Any]:
    calibration_ids = set(calibration_embeddings)
    evaluation_ids = set(evaluation_embeddings)
    calibration_records = [
        row for row in records
        if str(row.get("split")) == "calibration" and str(row["utterance_id"]) in calibration_ids
    ]
    evaluation_records = [
        row for row in records
        if str(row.get("split")) == "evaluation" and str(row["utterance_id"]) in evaluation_ids
    ]
    calibration_pairs = [
        row for row in calibration_pairs
        if set(map(str, row["source_utterance_ids"])) <= calibration_ids
    ]
    evaluation_pairs = [
        row for row in evaluation_pairs
        if set(map(str, row["source_utterance_ids"])) <= evaluation_ids
    ]
    examples = build_training_examples(calibration_records, calibration_pairs, reference)
    nonzero = [row for row in examples["cross_dialect_examples"] if float(row["target"]) != 0.0]
    pool_size = min(1000, len(nonzero))
    masks = build_nested_fixed_pair_masks(nonzero, pool_size=pool_size, seed=LOCKED_SEEDS[0])
    if masks["pair_identity_hash"] != expected_pair_identity_hash:
        raise ValueError("orthogonal condition does not reproduce the frozen primary pair-identity hash")

    run_config = _copy_config(config, fixed_epochs=fixed_epochs)
    q_cells: dict[str, Any] = {}
    trace_digests = []
    for q_key, pool in masks["arms"].items():
        q_cells[q_key] = {}
        for mode in ("ordinary", "prevalence_balanced"):
            rows = []
            for seed in LOCKED_SEEDS:
                run = _run_training_condition(
                    calibration_embeddings=calibration_embeddings,
                    evaluation_embeddings=evaluation_embeddings,
                    calibration_records=calibration_records,
                    evaluation_records=evaluation_records,
                    calibration_pairs=calibration_pairs,
                    evaluation_pairs=evaluation_pairs,
                    reference=reference,
                    config=run_config,
                    seed=seed,
                    lambda_cross=lambda_cross,
                    cross_examples=pool,
                    cross_loss_mode=mode,
                    fixed_epochs=fixed_epochs,
                )
                rows.append({
                    "seed": seed,
                    "gain": float(run["improvement_ratio"]),
                    "mae": float(run["mae"]),
                    "speaker_gain": _gain_by_speaker(run),
                })
                trace_digests.append(_sha({
                    "q": q_key,
                    "mode": mode,
                    "seed": seed,
                    "history": run["fitted"]["loss_history"],
                }))
            gains = np.asarray([row["gain"] for row in rows], dtype=np.float64)
            q_cells[q_key][mode] = {
                "seed_results": [
                    {"seed": row["seed"], "gain": row["gain"], "mae": row["mae"]}
                    for row in rows
                ],
                "median_gain": float(np.median(gains)),
                "iqr_gain": float(np.quantile(gains, 0.75) - np.quantile(gains, 0.25)),
                "minimum_gain": float(np.min(gains)),
                "maximum_gain": float(np.max(gains)),
                "speaker_gain": rows,
            }

    internal_q = [format(q, ".2f") for q in TARGET_PREVALENCE_GRID if 0.0 < q < 1.0]
    slope_rows = build_slope_rows(q_cells, internal_q)
    return {
        "schema": "target-prevalence-transfer-condition-v1",
        "pair_identity_hash": masks["pair_identity_hash"],
        "mask_hashes": masks["mask_hashes"],
        "pool_size": pool_size,
        "q_grid": list(TARGET_PREVALENCE_GRID),
        "internal_q_cells": internal_q,
        "fixed_epochs": fixed_epochs,
        "lambda_cross": lambda_cross,
        "seeds": list(LOCKED_SEEDS),
        "q_cells": q_cells,
        "slope_contrast": clustered_slope_bootstrap(slope_rows, seed=LOCKED_SEEDS[0], replicates=1000),
        "mean_gain_contrast": clustered_mean_gain_contrast(slope_rows, seed=LOCKED_SEEDS[0], replicates=1000),
        "seed_slope_distribution": seed_slope_distribution(q_cells, internal_q),
        "gain_distribution": _seed_mode_summary(q_cells, internal_q),
        "process_trace_digest": _sha(trace_digests),
    }


def _primary_condition(report: Mapping[str, Any]) -> dict[str, Any]:
    fixed = report["fixed_pair"]
    return {
        "model_name": "chinese_hubert_large",
        "reference": report["reference"],
        "pair_identity_hash": fixed["pair_identity_hash"],
        "slope_contrast": fixed["slope_contrast"],
        "mean_gain_contrast": fixed["mean_gain_contrast"],
        "seed_slope_distribution": fixed["seed_slope_distribution"],
    }


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in (
        "config", "calibration_embedding", "evaluation_embedding", "records",
        "calibration_pairs", "evaluation_pairs", "reference", "primary_report",
        "output", "gate_output",
    ):
        parser.add_argument(f"--{name.replace('_', '-')}", required=True)
    parser.add_argument("--model-name", required=True)
    parser.add_argument("--reference-name", required=True)
    parser.add_argument("--transfer-axis", choices=("cross_backbone", "cross_reference"), required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    primary_report = json.loads(Path(args.primary_report).read_text(encoding="utf-8"))
    primary = _primary_condition(primary_report)
    orthogonal = run_fixed_pair_transfer(
        config=load_config(args.config),
        calibration_embeddings=_load_embeddings(args.calibration_embedding),
        evaluation_embeddings=_load_embeddings(args.evaluation_embedding),
        records=_load_records(args.records),
        calibration_pairs=_load_pairs(args.calibration_pairs),
        evaluation_pairs=_load_pairs(args.evaluation_pairs),
        reference=_load_reference(args.reference),
        expected_pair_identity_hash=primary["pair_identity_hash"],
        fixed_epochs=int(primary_report["fixed_epochs"]),
        lambda_cross=float(primary_report["lambda_cross"]),
    )
    orthogonal["model_name"] = args.model_name
    orthogonal["reference"] = args.reference_name
    decision = evaluate_transfer_gate(primary, orthogonal)
    report = {
        "schema": "target-prevalence-rule-transfer-v1",
        "status": "evaluated",
        "corpus": "KeSpeech",
        "transfer_axis": args.transfer_axis,
        "rule": "0.5*mean(zero_target_loss)+0.5*mean(nonzero_target_loss)",
        "rule_selection_unit": "calibration_speaker_folds_only",
        "common_randomness": "same q grid, pair identities, masks, seeds, epochs, optimizer settings, and evaluation pair IDs",
        "primary": primary,
        "orthogonal": orthogonal,
        "decision": decision,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    gate = {
        "schema": "target-prevalence-rule-transfer-gate-v1",
        "status": "passed" if decision["rule_transfer_supported"] else "failed",
        "transfer_axis": args.transfer_axis,
        "primary": decision["primary"],
        "orthogonal": decision["orthogonal"],
        "rule_transfer_supported": decision["rule_transfer_supported"],
        "selected_wording": decision["selected_wording"],
        "rule_selection_unit": report["rule_selection_unit"],
    }
    gate_path = Path(args.gate_output)
    gate_path.parent.mkdir(parents=True, exist_ok=True)
    gate_path.write_text(json.dumps(gate, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
