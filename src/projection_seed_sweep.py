"""Run paired B3/B4 projection-head comparisons over the locked seeds."""

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
from .paired_randomness import clustered_paired_bootstrap
from .run_target_prevalence_mechanism import (
    _copy_config,
    _load_embeddings,
    _load_pairs,
    _load_records,
    _load_reference,
)


LOCKED_SEEDS = (20260829, 20260830, 20260831, 20260901, 20260902)


def _distribution(values: Sequence[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    return {
        "median": float(np.median(array)),
        "iqr": float(np.quantile(array, 0.75) - np.quantile(array, 0.25)),
        "minimum": float(np.min(array)),
        "maximum": float(np.max(array)),
    }


def summarize_seed_distribution(runs: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    if not runs:
        raise ValueError("at least one seed result is required")
    seeds = [int(row["seed"]) for row in runs]
    if len(set(seeds)) != len(seeds):
        raise ValueError("seed identifiers must be unique")
    b3_mae = [float(row["b3_mae"]) for row in runs]
    b4_mae = [float(row["b4_mae"]) for row in runs]
    return {
        "seed_count": len(runs),
        "seeds": seeds,
        "b3_mae": _distribution(b3_mae),
        "b4_mae": _distribution(b4_mae),
        "b3_gain": _distribution([float(row["b3_gain"]) for row in runs]),
        "b4_gain": _distribution([float(row["b4_gain"]) for row in runs]),
        "b3_minus_b4_mae": _distribution([first - second for first, second in zip(b3_mae, b4_mae)]),
        "all_b4_better_than_b3": all(second < first for first, second in zip(b3_mae, b4_mae)),
    }


def _schedule_digest(*, seed: int, pair_count: int, cross_count: int, epochs: int, n_pair: int, n_cross: int) -> str:
    rng = np.random.default_rng(seed)
    schedule = []
    for _ in range(epochs):
        schedule.append({
            "pair": rng.integers(0, pair_count, size=n_pair).tolist() if n_pair else [],
            "cross": rng.integers(0, cross_count, size=n_cross).tolist() if n_cross else [],
        })
    payload = {"seed": seed, "epochs": epochs, "n_pair": n_pair, "n_cross": n_cross, "schedule": schedule}
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()


def _paired_rows(b3: Sequence[Mapping[str, Any]], b4: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    left = {str(row["pair_id"]): row for row in b3}
    right = {str(row["pair_id"]): row for row in b4}
    if set(left) != set(right):
        raise ValueError("B3 and B4 evaluation pair IDs differ")
    return [
        {
            "pair_id": pair_id,
            "delta": float(left[pair_id]["absolute_error"]) - float(right[pair_id]["absolute_error"]),
            "matched_stratum": str(left[pair_id].get("matched_stratum") or left[pair_id].get("group")),
            "speaker_ids": list(left[pair_id]["speaker_ids"]),
        }
        for pair_id in sorted(left)
    ]


def _mean_delta_rows(seed_rows: Sequence[Sequence[Mapping[str, Any]]]) -> list[dict[str, Any]]:
    by_pair: dict[str, list[Mapping[str, Any]]] = {}
    for rows in seed_rows:
        for row in rows:
            by_pair.setdefault(str(row["pair_id"]), []).append(row)
    if any(len(rows) != len(seed_rows) for rows in by_pair.values()):
        raise ValueError("every evaluation pair must occur in every seed")
    return [
        {
            "pair_id": pair_id,
            "delta": float(np.mean([float(row["delta"]) for row in rows])),
            "matched_stratum": str(rows[0]["matched_stratum"]),
            "speaker_ids": list(rows[0]["speaker_ids"]),
        }
        for pair_id, rows in sorted(by_pair.items())
    ]


def run_seed_sweep(
    *,
    config: Mapping[str, Any],
    calibration_embeddings: Mapping[str, Sequence[float]],
    evaluation_embeddings: Mapping[str, Sequence[float]],
    records: Sequence[Mapping[str, Any]],
    calibration_pairs: Sequence[Mapping[str, Any]],
    evaluation_pairs: Sequence[Mapping[str, Any]],
    reference: Mapping[str, Mapping[str, float]],
    model_name: str,
    reference_name: str,
    lambda_cross: float,
    lambda_dialect: float,
    learning_rate: float,
    weight_decay: float,
    fixed_epochs: int,
    head_kind: str = "mlp",
    seeds: Sequence[int] = LOCKED_SEEDS,
) -> dict[str, Any]:
    calibration_ids = set(calibration_embeddings)
    evaluation_ids = set(evaluation_embeddings)
    calibration_records = [row for row in records if str(row.get("split")) == "calibration" and str(row["utterance_id"]) in calibration_ids]
    evaluation_records = [row for row in records if str(row.get("split")) == "evaluation" and str(row["utterance_id"]) in evaluation_ids]
    calibration_pairs = [row for row in calibration_pairs if set(map(str, row["source_utterance_ids"])) <= calibration_ids]
    evaluation_pairs = [row for row in evaluation_pairs if set(map(str, row["source_utterance_ids"])) <= evaluation_ids]
    examples = build_training_examples(calibration_records, calibration_pairs, reference)
    cross_examples = list(examples["cross_dialect_examples"])
    run_config = _copy_config(config, fixed_epochs=fixed_epochs)
    head_config = run_config["projection_head"]
    head_config["lambda_dialect_grid"] = [float(lambda_dialect)]
    head_config["learning_rate_grid"] = [float(learning_rate)]
    head_config["weight_decay_grid"] = [float(weight_decay)]
    seed_results = []
    paired_by_seed = []
    for seed in seeds:
        common = dict(
            calibration_embeddings=calibration_embeddings,
            evaluation_embeddings=evaluation_embeddings,
            calibration_records=calibration_records,
            evaluation_records=evaluation_records,
            calibration_pairs=calibration_pairs,
            evaluation_pairs=evaluation_pairs,
            reference=reference,
            config=run_config,
            seed=int(seed),
            cross_examples=cross_examples,
            fixed_epochs=fixed_epochs,
        )
        b3 = _run_training_condition(lambda_cross=0.0, head_kind=head_kind, **common)
        b4 = _run_training_condition(lambda_cross=float(lambda_cross), head_kind=head_kind, **common)
        paired = _paired_rows(b3["per_pair"], b4["per_pair"])
        paired_by_seed.append(paired)
        composition = b4["fitted"]["batch_composition"]
        seed_results.append({
            "seed": int(seed),
            "b3_mae": float(b3["mae"]),
            "b4_mae": float(b4["mae"]),
            "b3_gain": float(b3["improvement_ratio"]),
            "b4_gain": float(b4["improvement_ratio"]),
            "mae_delta_b3_minus_b4": float(b3["mae"] - b4["mae"]),
            "selected_epoch": fixed_epochs,
            "schedule_digest": _schedule_digest(
                seed=int(seed),
                pair_count=len(examples["pair_examples"]),
                cross_count=len(cross_examples),
                epochs=fixed_epochs,
                n_pair=int(composition["n_pair"]),
                n_cross=int(composition["n_cross"]),
            ),
        })
    averaged = _mean_delta_rows(paired_by_seed)
    clustered = clustered_paired_bootstrap(averaged, seed=int(seeds[0]), replicates=1000)
    distribution = summarize_seed_distribution(seed_results)
    return {
        "schema": "projection-seed-sweep-v1",
        "status": "evaluated",
        "model_name": model_name,
        "reference": reference_name,
        "selected": {
            "lambda_cross": float(lambda_cross),
            "lambda_dialect": float(lambda_dialect),
            "learning_rate": float(learning_rate),
            "weight_decay": float(weight_decay),
            "fixed_epochs": int(fixed_epochs),
            "head_kind": head_kind,
            "selection_unit": "calibration_speaker_folds",
        },
        "pair_count": len(evaluation_pairs),
        "calibration_pair_count": len(calibration_pairs),
        "cross_pool_count": len(cross_examples),
        "common_randomness": "same seed, initialization, pair batches, cross batches, optimizer, and evaluation pairs within each B3/B4 contrast",
        "seed_results": seed_results,
        "distribution": distribution,
        "clustered_b3_minus_b4": clustered,
        "passed": bool(distribution["all_b4_better_than_b3"] and clustered["ci"]["lower"] > 0),
    }


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("config", "calibration_embedding", "evaluation_embedding", "records", "calibration_pairs", "evaluation_pairs", "reference", "model_name", "reference_name", "output", "gate_output"):
        parser.add_argument(f"--{name.replace('_', '-')}", required=True)
    parser.add_argument("--lambda-cross", type=float, required=True)
    parser.add_argument("--lambda-dialect", type=float, required=True)
    parser.add_argument("--learning-rate", type=float, required=True)
    parser.add_argument("--weight-decay", type=float, required=True)
    parser.add_argument("--fixed-epochs", type=int, required=True)
    parser.add_argument("--head-kind", choices=["mlp", "linear"], default="mlp")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    report = run_seed_sweep(
        config=load_config(args.config),
        calibration_embeddings=_load_embeddings(args.calibration_embedding),
        evaluation_embeddings=_load_embeddings(args.evaluation_embedding),
        records=_load_records(args.records),
        calibration_pairs=_load_pairs(args.calibration_pairs),
        evaluation_pairs=_load_pairs(args.evaluation_pairs),
        reference=_load_reference(args.reference),
        model_name=args.model_name,
        reference_name=args.reference_name,
        lambda_cross=args.lambda_cross,
        lambda_dialect=args.lambda_dialect,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        fixed_epochs=args.fixed_epochs,
        head_kind=args.head_kind,
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    gate = {
        "schema": "projection-seed-sweep-gate-v1",
        "status": "passed" if report["passed"] else "failed",
        "model_name": report["model_name"],
        "reference": report["reference"],
        "selected": report["selected"],
        "distribution": report["distribution"],
        "clustered_b3_minus_b4": report["clustered_b3_minus_b4"],
        "common_randomness": report["common_randomness"],
    }
    gate_output = Path(args.gate_output)
    gate_output.parent.mkdir(parents=True, exist_ok=True)
    gate_output.write_text(json.dumps(gate, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
