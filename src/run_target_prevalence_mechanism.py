"""Run the locked target-prevalence mechanism experiment on KeSpeech."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from .config import load_config
from .cross_dialect_gradient_isolation import _run_training_condition
from .cross_dialect_projection_head import build_training_examples
from .target_prevalence_experiment import (
    TARGET_PREVALENCE_GRID,
    audit_prevalence_capacity,
    build_nested_fixed_pair_masks,
    build_natural_prevalence_pools,
    clustered_mean_gain_contrast,
    clustered_slope_bootstrap,
    slope_contrast,
    summarize_pool_covariates,
)


def _load_json(path: str) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _load_embeddings(path: str) -> dict[str, list[float]]:
    payload = _load_json(path)
    return dict(payload.get("embeddings", payload))


def _load_records(path: str) -> list[dict[str, Any]]:
    payload = _load_json(path)
    return list(payload.get("records", payload))


def _load_pairs(path: str) -> list[dict[str, Any]]:
    payload = _load_json(path)
    return list(payload.get("pairs", payload))


def _load_reference(path: str) -> dict[str, dict[str, float]]:
    payload = _load_json(path)
    return payload.get("matrix", payload)


def _sha(payload: object) -> str:
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()


def _copy_config(config: Mapping[str, Any], *, fixed_epochs: int) -> dict[str, Any]:
    result = copy.deepcopy(dict(config))
    head = result.setdefault("projection_head", {})
    head["max_epochs"] = int(fixed_epochs)
    head["lambda_cross_grid"] = [0.5]
    head["lambda_dialect_grid"] = [0.05]
    head["learning_rate_grid"] = [0.0003]
    head["weight_decay_grid"] = [0.0001]
    head["bootstrap_replicates"] = 1000
    return result


def _gain_by_speaker(result: Mapping[str, Any]) -> dict[str, float]:
    baseline = {str(row["pair_id"]): float(row["absolute_error"]) for row in result["baseline_per_pair"]}
    values: dict[str, list[float]] = {}
    for row in result["per_pair"]:
        delta = baseline[str(row["pair_id"])] - float(row["absolute_error"])
        for speaker in row.get("speaker_ids", []):
            values.setdefault(str(speaker), []).append(delta)
    return {speaker: float(np.mean(items)) for speaker, items in values.items() if items}


def build_slope_rows(
    arm: Mapping[str, Any],
    q_keys: Sequence[str],
) -> list[dict[str, Any]]:
    """Average paired seed results within speaker before cluster inference."""
    output = []
    for q_key in q_keys:
        ordinary = arm[q_key]["ordinary"]["speaker_gain"]
        balanced = arm[q_key]["prevalence_balanced"]["speaker_gain"]
        ordinary_by_speaker: dict[str, list[float]] = {}
        balanced_by_speaker: dict[str, list[float]] = {}
        for row in ordinary:
            for speaker, value in row["speaker_gain"].items():
                ordinary_by_speaker.setdefault(str(speaker), []).append(float(value))
        for row in balanced:
            for speaker, value in row["speaker_gain"].items():
                balanced_by_speaker.setdefault(str(speaker), []).append(float(value))
        for speaker in sorted(set(ordinary_by_speaker) & set(balanced_by_speaker)):
            output.append({
                "q": float(q_key),
                "speaker_ids": [speaker],
                "ordinary_gain": float(np.mean(ordinary_by_speaker[speaker])),
                "balanced_gain": float(np.mean(balanced_by_speaker[speaker])),
                "seed_count": len(ordinary_by_speaker[speaker]),
            })
    return output


def seed_slope_distribution(
    arm: Mapping[str, Any],
    q_keys: Sequence[str],
) -> dict[str, Any]:
    seeds = sorted({
        int(row["seed"])
        for q_key in q_keys
        for mode in ("ordinary", "prevalence_balanced")
        for row in arm[q_key][mode]["speaker_gain"]
    })
    seed_rows = []
    for seed in seeds:
        ordinary = []
        balanced = []
        for q_key in q_keys:
            by_mode = {}
            for mode in ("ordinary", "prevalence_balanced"):
                match = next(row for row in arm[q_key][mode]["speaker_gain"] if int(row["seed"]) == seed)
                by_mode[mode] = float(np.mean(list(match["speaker_gain"].values())))
            ordinary.append({"q": float(q_key), "gain": by_mode["ordinary"]})
            balanced.append({"q": float(q_key), "gain": by_mode["prevalence_balanced"]})
        contrast = slope_contrast(ordinary, balanced)
        seed_rows.append({"seed": seed, **contrast})
    values = np.asarray([row["delta_beta"] for row in seed_rows], dtype=np.float64)
    return {
        "seed_count": len(seed_rows),
        "seed_results": seed_rows,
        "median_delta_beta": float(np.median(values)),
        "iqr_delta_beta": float(np.quantile(values, 0.75) - np.quantile(values, 0.25)),
        "minimum_delta_beta": float(np.min(values)),
        "maximum_delta_beta": float(np.max(values)),
        "all_positive": bool(np.all(values > 0)),
    }


def run_experiment(
    *,
    config: Mapping[str, Any],
    calibration_embeddings: Mapping[str, Sequence[float]],
    evaluation_embeddings: Mapping[str, Sequence[float]],
    records: Sequence[Mapping[str, Any]],
    calibration_pairs: Sequence[Mapping[str, Any]],
    evaluation_pairs: Sequence[Mapping[str, Any]],
    reference: Mapping[str, Mapping[str, float]],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    calibration_ids = set(calibration_embeddings)
    evaluation_ids = set(evaluation_embeddings)
    calibration_records = [r for r in records if str(r.get("split")) == "calibration" and str(r["utterance_id"]) in calibration_ids]
    evaluation_records = [r for r in records if str(r.get("split")) == "evaluation" and str(r["utterance_id"]) in evaluation_ids]
    calibration_pairs = [p for p in calibration_pairs if set(map(str, p["source_utterance_ids"])) <= calibration_ids]
    evaluation_pairs = [p for p in evaluation_pairs if set(map(str, p["source_utterance_ids"])) <= evaluation_ids]
    base = build_training_examples(calibration_records, calibration_pairs, reference)
    generic = list(base["pair_examples"])
    cross = [item for item in base["cross_dialect_examples"] if float(item["target"]) != 0.0]
    capacity = {
        "schema": "fixed-pair-target-prevalence-capacity-v1",
        "nonzero_identity_capacity": len(cross),
        "common_pool_size": min(1000, len(cross)),
        "q_grid": list(TARGET_PREVALENCE_GRID),
        "minimum_cell": 200,
        "status": "passed" if len(cross) >= 200 else "failed",
        "result_blind": True,
        "note": "zero targets are generated by masking the fixed nonzero identity pool; no natural zero support is consumed",
    }
    pool_size = int(capacity["common_pool_size"])
    if pool_size < 200:
        raise ValueError("target-prevalence fixed-pair arm lacks 200-example support")
    masks = build_nested_fixed_pair_masks(cross, pool_size=pool_size, seed=20260829)
    natural_capacity = audit_prevalence_capacity(generic, minimum_cell=200)
    natural_size = min(pool_size, int(natural_capacity["common_pool_size"]))
    if natural_size < 200:
        raise ValueError("target-prevalence natural arm lacks 200-example support")
    natural = build_natural_prevalence_pools(generic, pool_size=natural_size, seed=20260829)
    seeds = [20260829, 20260830, 20260831, 20260901, 20260902]
    lambda_cross = 0.5
    fixed_epochs = int(config.get("projection_head", {}).get("max_epochs", 30))
    run_config = _copy_config(config, fixed_epochs=fixed_epochs)
    fixed_results: dict[str, dict[str, Any]] = {}
    natural_results: dict[str, dict[str, Any]] = {}
    traces: list[dict[str, Any]] = []
    for arm_name, arm_pool in (("fixed_pair", masks["arms"]), ("natural_pool", natural["arms"])):
        target = fixed_results if arm_name == "fixed_pair" else natural_results
        for q_key, pool in arm_pool.items():
            target[q_key] = {}
            for mode in ("ordinary", "prevalence_balanced"):
                rows = []
                for seed in seeds:
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
                    rows.append({"seed": seed, "gain": float(run["improvement_ratio"]), "mae": float(run["mae"]), "speaker_gain": _gain_by_speaker(run)})
                    traces.append({"arm": arm_name, "q": float(q_key), "mode": mode, "seed": seed, "history": run["fitted"]["loss_history"]})
                gains = [r["gain"] for r in rows]
                target[q_key][mode] = {
                    "seed_results": [{"seed": r["seed"], "gain": r["gain"], "mae": r["mae"]} for r in rows],
                    "median_gain": float(np.median(gains)),
                    "iqr_gain": [float(np.quantile(gains, 0.25)), float(np.quantile(gains, 0.75))],
                    "min_gain": float(min(gains)),
                    "max_gain": float(max(gains)),
                    "speaker_gain": rows,
                }
    internal_q = [format(q, ".2f") for q in TARGET_PREVALENCE_GRID if 0.0 < q < 1.0]

    fixed_slope_rows = build_slope_rows(fixed_results, internal_q)
    natural_slope_rows = build_slope_rows(natural_results, internal_q)
    fixed_slope = clustered_slope_bootstrap(fixed_slope_rows, seed=20260829, replicates=1000)
    natural_slope = clustered_slope_bootstrap(natural_slope_rows, seed=20260829, replicates=1000)
    fixed_mean_gain = clustered_mean_gain_contrast(fixed_slope_rows, seed=20260829, replicates=1000)
    natural_mean_gain = clustered_mean_gain_contrast(natural_slope_rows, seed=20260829, replicates=1000)
    fixed_seed_slopes = seed_slope_distribution(fixed_results, internal_q)
    natural_seed_slopes = seed_slope_distribution(natural_results, internal_q)
    report = {
        "schema": "target-prevalence-mechanism-v1",
        "status": "evaluated",
        "corpus": "KeSpeech",
        "reference": "taxonomy_binary_subgroup_proxy",
        "lambda_cross": lambda_cross,
        "lambda_dialect": 0.05,
        "fixed_epochs": fixed_epochs,
        "seeds": seeds,
        "capacity": {"fixed_pair": capacity, "natural_pool": natural_capacity},
        "fixed_pair": {"pool_size": pool_size, "pair_identity_hash": masks["pair_identity_hash"], "mask_hashes": masks["mask_hashes"], "q_cells": fixed_results, "slope_contrast": fixed_slope, "mean_gain_contrast": fixed_mean_gain, "seed_slope_distribution": fixed_seed_slopes},
        "natural_pool": {"pool_size": natural_size, "q_cells": natural_results, "covariates": {q: summarize_pool_covariates(pool) for q, pool in natural["arms"].items()}, "slope_contrast": natural_slope, "mean_gain_contrast": natural_mean_gain, "seed_slope_distribution": natural_seed_slopes, "interpretation": "ecological_sensitivity_only_due_to_q_linked_pair_identity_and_speaker_coverage"},
        "internal_q_cells": internal_q,
        "process_trace_count": len(traces),
        "trace_digest": _sha(traces),
    }
    return report, traces


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("config", "calibration_embedding", "evaluation_embedding", "records", "calibration_pairs", "evaluation_pairs", "reference", "output", "process_output", "gate_output"):
        parser.add_argument(f"--{name.replace('_', '-')}", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    report, traces = run_experiment(config=load_config(args.config), calibration_embeddings=_load_embeddings(args.calibration_embedding), evaluation_embeddings=_load_embeddings(args.evaluation_embedding), records=_load_records(args.records), calibration_pairs=_load_pairs(args.calibration_pairs), evaluation_pairs=_load_pairs(args.evaluation_pairs), reference=_load_reference(args.reference))
    output = Path(args.output); output.parent.mkdir(parents=True, exist_ok=True); output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    process = Path(args.process_output); process.parent.mkdir(parents=True, exist_ok=True); process.write_text(json.dumps({"schema": "target-prevalence-process-traces-v1", "trace_count": len(traces), "trace_digest": report["trace_digest"], "traces": traces}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    fixed_supported = report["fixed_pair"]["slope_contrast"]["ci"]["lower"] > 0 and report["fixed_pair"]["seed_slope_distribution"]["all_positive"]
    gate = {"schema": "target-prevalence-mechanism-gate-v1", "status": "evaluated", "fixed_pair_association_supported": fixed_supported, "natural_pool_ecological_only": True, "balanced_slope_contrast": report["fixed_pair"]["slope_contrast"], "mean_gain_contrast": report["fixed_pair"]["mean_gain_contrast"], "seed_slope_distribution": report["fixed_pair"]["seed_slope_distribution"], "decision": "association_only_unless_balanced_slope_and_process_chain_pass", "q_grid": list(TARGET_PREVALENCE_GRID), "inference": "seed-paired speaker-cluster means within q, 1000 cluster bootstrap replicates"}
    gate_path = Path(args.gate_output); gate_path.parent.mkdir(parents=True, exist_ok=True); gate_path.write_text(json.dumps(gate, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
