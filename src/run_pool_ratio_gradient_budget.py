"""Run the locked pool-exposure, gradient-budget, and placebo audit."""

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
from .pool_ratio_gradient_budget import (
    build_candidate_inventory_grid,
    build_exposure_ratio_grid,
    classify_mechanism,
    speaker_cluster_interaction_bootstrap,
)
from .run_target_prevalence_mechanism import (
    _copy_config,
    _gain_by_speaker,
    _load_embeddings,
    _load_pairs,
    _load_records,
    _load_reference,
)
from .target_prevalence_experiment import build_nested_fixed_pair_masks


def _digest(payload: object) -> str:
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()


def _paired_speaker_bootstrap(
    first: Mapping[str, float],
    second: Mapping[str, float],
    *,
    seed: int,
    replicates: int = 1000,
) -> dict[str, Any]:
    speakers = sorted(set(first) & set(second))
    if not speakers:
        raise ValueError("paired speaker contrast has no common clusters")
    deltas = np.asarray([float(first[s]) - float(second[s]) for s in speakers], dtype=np.float64)
    rng = np.random.default_rng(seed)
    boot = [float(np.mean(deltas[rng.integers(0, len(deltas), len(deltas))])) for _ in range(replicates)]
    return {
        "estimate": float(np.mean(deltas)),
        "ci": {"lower": float(np.quantile(boot, 0.025)), "upper": float(np.quantile(boot, 0.975)), "confidence_level": 0.95},
        "bootstrap_replicates": replicates,
        "resampling_unit": "evaluation_speaker_cluster",
        "speaker_cluster_count": len(speakers),
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
    prevalence_gate: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    calibration_ids = set(calibration_embeddings)
    evaluation_ids = set(evaluation_embeddings)
    calibration_records = [r for r in records if str(r.get("split")) == "calibration" and str(r["utterance_id"]) in calibration_ids]
    evaluation_records = [r for r in records if str(r.get("split")) == "evaluation" and str(r["utterance_id"]) in evaluation_ids]
    calibration_pairs = [p for p in calibration_pairs if set(map(str, p["source_utterance_ids"])) <= calibration_ids]
    evaluation_pairs = [p for p in evaluation_pairs if set(map(str, p["source_utterance_ids"])) <= evaluation_ids]
    base = build_training_examples(calibration_records, calibration_pairs, reference)
    generic = list(base["pair_examples"])
    cross = [item for item in base["cross_dialect_examples"] if float(item["target"]) != 0.0]
    fixed = build_nested_fixed_pair_masks(cross, pool_size=1000, seed=20260829)
    q0 = 0.50
    cross_pool = fixed["arms"][format(q0, ".2f")]
    ratios = build_exposure_ratio_grid()
    lambdas = [0.0, 0.25, 0.5, 1.0, 2.0]
    seeds = [20260829, 20260830, 20260831, 20260901, 20260902]
    epochs = int(config.get("projection_head", {}).get("max_epochs", 100))
    run_config = _copy_config(config, fixed_epochs=epochs)
    separate: dict[str, dict[str, Any]] = {}
    interaction_rows = []
    for ratio in ratios:
        rho_key = format(float(ratio["rho"]), ".1f")
        separate[rho_key] = {}
        exposure = (int(ratio["generic_count"]), int(ratio["cross_count"]))
        for lam in lambdas:
            runs = []
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
                    lambda_cross=lam,
                    cross_examples=cross_pool,
                    fixed_epochs=epochs,
                    aggregation_mode="separate",
                    exposure_ratio=exposure,
                    record_gradient_budget=True,
                )
                speaker_gain = _gain_by_speaker(run)
                runs.append({"seed": seed, "calibration_mae": run["calibration_mae"], "gain": run["improvement_ratio"], "speaker_gain": speaker_gain, "gradient_trace": [row for row in run["fitted"]["loss_history"] if row["gradient_budget"] is not None]})
                interaction_rows.extend({"speaker_id": speaker, "rho": float(ratio["rho"]), "lambda_cross": lam, "gain": gain} for speaker, gain in speaker_gain.items())
            separate[rho_key][format(lam, ".2f")] = {
                "median_calibration_mae": float(np.median([r["calibration_mae"] for r in runs])),
                "median_gain": float(np.median([r["gain"] for r in runs])),
                "seed_results": runs,
            }
    lambda_star = {
        rho: min(cells, key=lambda key: (cells[key]["median_calibration_mae"], float(key)))
        for rho, cells in separate.items()
    }
    mixed = {}
    selected_separate_speaker: dict[str, dict[str, float]] = {}
    for ratio in ratios:
        rho_key = format(float(ratio["rho"]), ".1f")
        lam_key = lambda_star[rho_key]
        lam = float(lam_key)
        exposure = (int(ratio["generic_count"]), int(ratio["cross_count"]))
        selected_separate_speaker[rho_key] = {
            speaker: float(np.mean(values))
            for speaker in {s for run in separate[rho_key][lam_key]["seed_results"] for s in run["speaker_gain"]}
            for values in [[run["speaker_gain"][speaker] for run in separate[rho_key][lam_key]["seed_results"] if speaker in run["speaker_gain"]]]
        }
        runs = []
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
                lambda_cross=lam,
                cross_examples=cross_pool,
                fixed_epochs=epochs,
                aggregation_mode="mixed_mean",
                exposure_ratio=exposure,
                record_gradient_budget=True,
            )
            runs.append({"seed": seed, "calibration_mae": run["calibration_mae"], "gain": run["improvement_ratio"], "speaker_gain": _gain_by_speaker(run), "gradient_trace": [row for row in run["fitted"]["loss_history"] if row["gradient_budget"] is not None]})
        mixed[rho_key] = {"lambda_star": lam, "median_gain": float(np.median([r["gain"] for r in runs])), "seed_results": runs}
    strengths = [0.0, 1e-5, 1e-4, 1e-3, 1e-2]
    placebo_candidates = {}
    for strength in strengths:
        runs = []
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
                lambda_cross=0.0,
                cross_examples=cross_pool,
                fixed_epochs=epochs,
                exposure_ratio=(1, 1),
                regularization_strength=strength,
                replace_cross_with_regularization=True,
            )
            runs.append({"seed": seed, "calibration_mae": run["calibration_mae"], "gain": run["improvement_ratio"], "speaker_gain": _gain_by_speaker(run)})
        placebo_candidates[format(strength, ".1e")] = {"median_calibration_mae": float(np.median([r["calibration_mae"] for r in runs])), "median_gain": float(np.median([r["gain"] for r in runs])), "seed_results": runs}
    placebo_star = min(placebo_candidates, key=lambda key: (placebo_candidates[key]["median_calibration_mae"], float(key)))
    placebo_speaker = {
        speaker: float(np.mean(values))
        for speaker in {s for run in placebo_candidates[placebo_star]["seed_results"] for s in run["speaker_gain"]}
        for values in [[run["speaker_gain"][speaker] for run in placebo_candidates[placebo_star]["seed_results"] if speaker in run["speaker_gain"]]]
    }
    primary_speaker = selected_separate_speaker["1.0"]
    placebo_contrast = _paired_speaker_bootstrap(primary_speaker, placebo_speaker, seed=20260829)
    interaction = speaker_cluster_interaction_bootstrap(interaction_rows, seed=20260829, replicates=1000)
    selected_gradients = [row["gradient_budget"] for rho, lam in lambda_star.items() for run in separate[rho][lam]["seed_results"] for row in run["gradient_trace"]]
    eta = np.asarray([float(row["eta_sep"]) for row in selected_gradients if row["eta_sep"] is not None])
    cosine = np.asarray([float(row["cosine"]) for row in selected_gradients if row["cosine"] is not None])
    eta_cv = float(np.std(eta) / max(abs(np.mean(eta)), 1e-12))
    gradient_budget_supported = len(set(lambda_star.values())) > 1 and eta_cv <= 0.20 and not (interaction["ci"]["lower"] <= 0 <= interaction["ci"]["upper"])
    regularization_compatible = placebo_contrast["ci"]["lower"] <= 0 <= placebo_contrast["ci"]["upper"]
    fixed_pair_association_supported = bool(prevalence_gate["fixed_pair_association_supported"])
    classification = classify_mechanism(
        prevalence_supported=bool(prevalence_gate.get("prevalence_supported", False)),
        gradient_budget_supported=gradient_budget_supported,
        interference_supported=False,
        regularization_compatible=regularization_compatible,
    )
    inventory = {"schema": "pool-inventory-sensitivity-v1", "status": "result_blind_inventory_audit", "grid": build_candidate_inventory_grid(generic, cross_pool), "pair_identity_digest": _digest(sorted(str(row["pair_id"]) for row in generic)), "cross_identity_digest": _digest(sorted(str(row["pair_id"]) for row in cross_pool)), "note": "Inventory changes are reported as coverage/noise sensitivity and are not used for the mechanism classification."}
    placebo = {"schema": "loss-structure-placebo-v1", "status": "evaluated", "candidate_strengths": strengths, "selected_strength": float(placebo_star), "selection_unit": "median calibration MAE across five locked seeds", "candidates": placebo_candidates, "cross_term_minus_placebo": placebo_contrast, "regularization_compatible": regularization_compatible}
    report = {
        "schema": "pool-ratio-gradient-budget-v1",
        "status": "evaluated",
        "q0": q0,
        "fixed_pair_identity_hash": fixed["pair_identity_hash"],
        "target_mask_hash": fixed["mask_hashes"]["0.50"],
        "ratio_grid": ratios,
        "lambda_grid": lambdas,
        "seeds": seeds,
        "epochs": epochs,
        "separate_objective": "mean(pair_loss)+lambda_cross*mean(cross_loss)",
        "mixed_objective": "(sum(pair_loss)+lambda_cross*sum(cross_loss))/(n_pair+n_cross)",
        "separate": separate,
        "lambda_star_by_rho": {rho: float(value) for rho, value in lambda_star.items()},
        "mixed_at_lambda_star": mixed,
        "primary_interaction": interaction,
        "gradient_summary": {"eta_sep_median": float(np.median(eta)), "eta_sep_cv": eta_cv, "cosine_median": float(np.median(cosine)), "logging_interval_epochs": 5, "readout_count": len(selected_gradients)},
        "fixed_pair_association_supported": fixed_pair_association_supported,
        "mechanism_classification": classification,
        "selected_wording": "Target prevalence affects the loss-structure result under the tested configuration; the pool-ratio audit does not promote a unique gradient-budget mechanism." if not gradient_budget_supported else "The calibration-selected weight and gradient share vary with exposure ratio, consistent with gradient-budget allocation under the tested configuration.",
    }
    return report, inventory, placebo


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("config", "calibration_embedding", "evaluation_embedding", "records", "calibration_pairs", "evaluation_pairs", "reference", "prevalence_gate", "output", "inventory_output", "placebo_output", "gate_output"):
        parser.add_argument(f"--{name.replace('_', '-')}", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    report, inventory, placebo = run_experiment(config=load_config(args.config), calibration_embeddings=_load_embeddings(args.calibration_embedding), evaluation_embeddings=_load_embeddings(args.evaluation_embedding), records=_load_records(args.records), calibration_pairs=_load_pairs(args.calibration_pairs), evaluation_pairs=_load_pairs(args.evaluation_pairs), reference=_load_reference(args.reference), prevalence_gate=json.loads(Path(args.prevalence_gate).read_text(encoding="utf-8")))
    for path, payload in ((args.output, report), (args.inventory_output, inventory), (args.placebo_output, placebo)):
        target = Path(path); target.parent.mkdir(parents=True, exist_ok=True); target.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    gate = {"schema": "gradient-budget-mechanism-gate-v1", "status": "evaluated", "q0": report["q0"], "ratio_grid": report["ratio_grid"], "primary_interaction": report["primary_interaction"], "lambda_star_by_rho": report["lambda_star_by_rho"], "gradient_summary": report["gradient_summary"], "placebo_compatible": placebo["regularization_compatible"], "classification": report["mechanism_classification"], "selected_wording": report["selected_wording"]}
    gate_path = Path(args.gate_output); gate_path.parent.mkdir(parents=True, exist_ok=True); gate_path.write_text(json.dumps(gate, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
