"""Build a compact two-reference sensitivity report for the manuscript."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any, Mapping, Sequence


def _reference_result(payload: Mapping[str, Any]) -> Mapping[str, Any]:
    return payload["models"][0]["references"][0]


def _metrics(payload: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "mae": float(payload["mae"]),
        "gain": float(payload["improvement_ratio"]),
        "ci": payload["ci"],
    }


def _target_distribution(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    values = [float(row["target"]) for row in rows]
    rounded = Counter(format(value, ".6f") for value in values)
    return {
        "pair_count": len(values),
        "zero_count": sum(value == 0.0 for value in values),
        "nonzero_count": sum(value != 0.0 for value in values),
        "unique_target_count": len(set(values)),
        "target_histogram": dict(sorted(rounded.items())),
    }


def architecture_boundary(
    mlp_payload: Mapping[str, Any],
    linear_payload: Mapping[str, Any],
) -> dict[str, Any]:
    linear_b3 = linear_payload["comparisons"]["lambda_cross_zero"]
    gains = {
        "mlp_b4": float(mlp_payload["improvement_ratio"]),
        "linear_b3": float(linear_b3["improvement_ratio"]),
        "linear_b4": float(linear_payload["improvement_ratio"]),
    }
    mlp_increment = bool(mlp_payload["paired_contrast_b4_vs_b3"]["passed"])
    linear_increment = bool(linear_payload["paired_contrast_b4_vs_b3"]["passed"])
    return {
        "strongest_projection": max(gains, key=gains.get),
        "gains": gains,
        "mlp_cross_loss_increment_supported": mlp_increment,
        "linear_cross_loss_increment_supported": linear_increment,
        "cross_loss_increment_architecture_robust": mlp_increment and linear_increment,
        "selected_wording": (
            "The independently weighted cross loss improved the multilayer perceptron but not the linear projection; the increment was architecture-dependent."
            if mlp_increment and not linear_increment
            else "The architecture control did not establish a common cross-loss increment."
        ),
    }


def summarize_linear_seed(payload: Mapping[str, Any]) -> dict[str, Any]:
    if payload.get("selected", {}).get("head_kind") != "linear":
        raise ValueError("linear seed report must declare head_kind=linear")
    rows = payload["seed_results"]
    all_b3_better = all(float(row["b3_mae"]) < float(row["b4_mae"]) for row in rows)
    clustered = payload["clustered_b3_minus_b4"]
    return {
        "selected": payload["selected"],
        "distribution": payload["distribution"],
        "clustered_b3_minus_b4": clustered,
        "all_b3_better_than_b4": all_b3_better,
        "clustered_negative_increment_supported": bool(
            float(clustered["observed_delta"]) < 0
            and float(clustered["ci"]["upper"]) < 0
        ),
    }
def summarize_reference(
    main_payload: Mapping[str, Any],
    seed_payload: Mapping[str, Any],
    linear_payload: Mapping[str, Any],
    linear_seed_payload: Mapping[str, Any],
) -> dict[str, Any]:
    main = _reference_result(main_payload)
    linear = _reference_result(linear_payload)
    linear_b3 = linear["comparisons"]["lambda_cross_zero"]
    b3 = main["comparisons"]["lambda_cross_zero"]
    shuffled = main["ablations"]["shuffled_cross_dialect"]
    permuted = main["ablations"]["permuted_pair_distance_target"]
    b4_gain = float(main["improvement_ratio"])
    max_control = max(float(shuffled["improvement_ratio"]), float(permuted["improvement_ratio"]))
    return {
        "reference": main["reference"],
        "selected": main["selected"],
        "target_distribution": _target_distribution(main["per_pair"]),
        "b0": {"mae": float(main["baseline_mae"])},
        "b3": _metrics(b3),
        "b4": _metrics(main),
        "linear": {
            "b3": _metrics(linear_b3),
            "b4": _metrics(linear),
            "paired_b4_vs_b3": linear["paired_contrast_b4_vs_b3"],
            "seed_sweep": summarize_linear_seed(linear_seed_payload),
        },
        "shuffled": _metrics(shuffled),
        "target_permuted": _metrics(permuted),
        "paired_b4_vs_b3": main["paired_contrast_b4_vs_b3"],
        "seed_sweep": {
            "passed": bool(seed_payload["passed"]),
            "distribution": seed_payload["distribution"],
            "clustered_b3_minus_b4": seed_payload["clustered_b3_minus_b4"],
        },
        "semantic_specificity_supported": bool(b4_gain > max_control),
        "architecture_boundary": architecture_boundary(main, linear),
    }


def reference_gate(binary: Mapping[str, Any], continuous: Mapping[str, Any]) -> dict[str, Any]:
    efficacy = all(
        float(item["b4"]["gain"]) >= 0.05
        and float(item["b4"]["ci"]["lower"]) > 0
        and bool(item["seed_sweep"]["passed"])
        for item in (binary, continuous)
    )
    semantic = all(bool(item["semantic_specificity_supported"]) for item in (binary, continuous))
    architecture_boundaries = [item.get("architecture_boundary") for item in (binary, continuous)]
    architecture_robust = bool(
        all(boundary is not None for boundary in architecture_boundaries)
        and all(bool(boundary["cross_loss_increment_architecture_robust"]) for boundary in architecture_boundaries)
    )
    linear_b3_strongest = bool(
        all(boundary is not None for boundary in architecture_boundaries)
        and all(boundary["strongest_projection"] == "linear_b3" for boundary in architecture_boundaries)
    )
    return {
        "efficacy_across_references_supported": bool(efficacy),
        "semantic_specificity_across_references_supported": bool(semantic),
        "cross_loss_increment_architecture_robust": architecture_robust,
        "linear_b3_strongest_across_references": linear_b3_strongest,
        "selected_wording": (
            "Trainable projection benefits persisted across both references. The linear pair-only head was strongest, and semantic controls plus the linear architecture blocked a general cross-loss mechanism claim."
            if efficacy and not semantic and linear_b3_strongest
            else "The projection benefit persisted across the binary and continuous references, but shuffled and target-permuted controls blocked a semantic-specificity claim."
            if efficacy and not semantic
            else "The reference-sensitivity result did not support a common efficacy claim."
        ),
    }


def _load(path: str) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in (
        "binary_main", "binary_seed", "binary_linear", "binary_linear_seed",
        "continuous_main", "continuous_seed", "continuous_linear",
        "continuous_linear_seed", "output", "gate_output",
    ):
        parser.add_argument(f"--{name.replace('_', '-')}", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    binary = summarize_reference(
        _load(args.binary_main),
        _load(args.binary_seed),
        _load(args.binary_linear),
        _load(args.binary_linear_seed),
    )
    continuous = summarize_reference(
        _load(args.continuous_main),
        _load(args.continuous_seed),
        _load(args.continuous_linear),
        _load(args.continuous_linear_seed),
    )
    decision = reference_gate(binary, continuous)
    report = {
        "schema": "reference-sensitivity-clustered-v1",
        "status": "evaluated",
        "model_name": "chinese_hubert_large",
        "references": {"binary_taxonomy": binary, "continuous_sinitic": continuous},
        "decision": decision,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    gate = {"schema": "reference-sensitivity-gate-v1", "status": "passed" if decision["efficacy_across_references_supported"] else "failed", **decision}
    gate_output = Path(args.gate_output)
    gate_output.parent.mkdir(parents=True, exist_ok=True)
    gate_output.write_text(json.dumps(gate, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
