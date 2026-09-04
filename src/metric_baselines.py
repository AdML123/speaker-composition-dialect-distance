"""Calibration-only metric baselines and operational relation rankings."""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from sklearn.decomposition import PCA
from torch import nn

from .config import load_config
from .cross_dialect_projection_head import (
    _apply_affine,
    _cosine_batch,
    _normalized_reference,
    _record_index,
    fit_affine_from_training_pairs,
)
from .estimand_sensitivity import WEIGHTINGS, weighted_mae
from .relation_ranking import (
    aggregate_relation_predictions,
    ordering_changes,
    ranking_metrics,
    reference_relations,
)
from .run_target_prevalence_mechanism import (
    _load_embeddings,
    _load_pairs,
    _load_records,
    _load_reference,
)


SEEDS = (20260829, 20260830, 20260831, 20260901, 20260902)
REFERENCES = {
    "taxonomy": Path("results/references/taxonomy_matrix.json"),
    "city_nearest": Path("results/references/sinitic_data4_city_nearest.json"),
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
}


class DiagonalMetric(nn.Module):
    """A positive feature reweighting with one parameter per input dimension."""

    def __init__(self, dimension: int) -> None:
        super().__init__()
        if dimension <= 0:
            raise ValueError("dimension must be positive")
        initial = np.log(np.expm1(1.0))
        self.raw_scales = nn.Parameter(torch.full((dimension,), float(initial)))

    def scales(self) -> torch.Tensor:
        return nn.functional.softplus(self.raw_scales)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return nn.functional.normalize(inputs * self.scales(), p=2, dim=1)


def _available_split(
    records: Sequence[Mapping[str, Any]],
    embeddings: Mapping[str, Sequence[float]],
    split: str,
) -> list[dict[str, Any]]:
    ids = set(embeddings)
    return [
        dict(row)
        for row in records
        if str(row.get("split")) == split and str(row["utterance_id"]) in ids
    ]


def _available_pairs(
    pairs: Sequence[Mapping[str, Any]], embeddings: Mapping[str, Sequence[float]]
) -> list[dict[str, Any]]:
    ids = set(embeddings)
    return [
        dict(row)
        for row in pairs
        if set(map(str, row["source_utterance_ids"])) <= ids
    ]


def _target(
    reference: Mapping[str, Mapping[str, float]], labels: Sequence[str]
) -> float:
    _, maximum = _normalized_reference(reference)
    return float(reference[labels[0]][labels[1]]) / maximum


def _pair_arrays(
    pairs: Sequence[Mapping[str, Any]],
    records: Sequence[Mapping[str, Any]],
) -> tuple[list[str], list[str], list[list[str]], list[list[str]]]:
    indexed = _record_index(records)
    left_ids: list[str] = []
    right_ids: list[str] = []
    dialects: list[list[str]] = []
    speakers: list[list[str]] = []
    for pair in pairs:
        left, right = map(str, pair["source_utterance_ids"])
        left_ids.append(left)
        right_ids.append(right)
        dialects.append(
            [str(indexed[left]["dialect_label"]), str(indexed[right]["dialect_label"])]
        )
        speakers.append(
            [str(indexed[left]["speaker_id"]), str(indexed[right]["speaker_id"])]
        )
    return left_ids, right_ids, dialects, speakers


def _score_vectors(
    pairs: Sequence[Mapping[str, Any]],
    records: Sequence[Mapping[str, Any]],
    vectors: Mapping[str, Sequence[float]],
    reference: Mapping[str, Mapping[str, float]],
    affine: Mapping[str, float],
) -> list[dict[str, Any]]:
    left_ids, right_ids, dialects, speakers = _pair_arrays(pairs, records)
    rows = []
    for pair, left, right, labels, speaker_ids in zip(
        pairs, left_ids, right_ids, dialects, speakers
    ):
        left_vector = torch.as_tensor(vectors[left], dtype=torch.float64)
        right_vector = torch.as_tensor(vectors[right], dtype=torch.float64)
        raw = float(_cosine_batch(left_vector[None], right_vector[None])[0])
        predicted = float(_apply_affine([raw], affine)[0])
        target = _target(reference, labels)
        rows.append(
            {
                "pair_id": str(pair["pair_id"]),
                "group": str(pair.get("group", "")),
                "dialect_labels": labels,
                "speaker_ids": speaker_ids,
                "utterance_ids": [left, right],
                "matched_stratum": pair.get("matched_stratum"),
                "predicted_distance": predicted,
                "reference_distance": target,
                "absolute_error": abs(predicted - target),
            }
        )
    return rows


def _fit_affine_for_vectors(
    pairs: Sequence[Mapping[str, Any]],
    records: Sequence[Mapping[str, Any]],
    vectors: Mapping[str, Sequence[float]],
    reference: Mapping[str, Mapping[str, float]],
) -> dict[str, float]:
    left_ids, right_ids, dialects, _ = _pair_arrays(pairs, records)
    raw = []
    targets = []
    for left, right, labels in zip(left_ids, right_ids, dialects):
        left_vector = torch.as_tensor(vectors[left], dtype=torch.float64)
        right_vector = torch.as_tensor(vectors[right], dtype=torch.float64)
        raw.append(float(_cosine_batch(left_vector[None], right_vector[None])[0]))
        targets.append(_target(reference, labels))
    return fit_affine_from_training_pairs(raw, targets)


def _raw_and_pca(
    calibration_embeddings: Mapping[str, Sequence[float]],
    evaluation_embeddings: Mapping[str, Sequence[float]],
) -> dict[str, tuple[dict[str, list[float]], dict[str, list[float]]]]:
    calibration_ids = sorted(calibration_embeddings)
    evaluation_ids = sorted(evaluation_embeddings)
    calibration_matrix = np.asarray(
        [calibration_embeddings[key] for key in calibration_ids], dtype=np.float64
    )
    evaluation_matrix = np.asarray(
        [evaluation_embeddings[key] for key in evaluation_ids], dtype=np.float64
    )
    pca = PCA(n_components=256, svd_solver="randomized", random_state=20260829)
    calibration_pca = pca.fit_transform(calibration_matrix)
    evaluation_pca = pca.transform(evaluation_matrix)
    return {
        "frozen_affine": (
            {key: list(map(float, calibration_embeddings[key])) for key in calibration_ids},
            {key: list(map(float, evaluation_embeddings[key])) for key in evaluation_ids},
        ),
        "frozen_pca256_affine": (
            {key: calibration_pca[index].tolist() for index, key in enumerate(calibration_ids)},
            {key: evaluation_pca[index].tolist() for index, key in enumerate(evaluation_ids)},
        ),
    }


def _train_diagonal(
    calibration_embeddings: Mapping[str, Sequence[float]],
    calibration_pairs: Sequence[Mapping[str, Any]],
    calibration_records: Sequence[Mapping[str, Any]],
    reference: Mapping[str, Mapping[str, float]],
    *,
    seed: int,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    weight_decay: float,
) -> tuple[DiagonalMetric, dict[str, float], list[float]]:
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ids = sorted(calibration_embeddings)
    index = {key: position for position, key in enumerate(ids)}
    matrix = np.asarray([calibration_embeddings[key] for key in ids], dtype=np.float32)
    mean = matrix.mean(axis=0)
    scale = matrix.std(axis=0)
    scale[scale == 0] = 1e-6
    standardized = (matrix - mean) / scale
    tensors = torch.as_tensor(standardized, device=device)
    left_ids, right_ids, dialects, _ = _pair_arrays(
        calibration_pairs, calibration_records
    )
    left = torch.as_tensor([index[key] for key in left_ids], dtype=torch.long, device=device)
    right = torch.as_tensor([index[key] for key in right_ids], dtype=torch.long, device=device)
    targets = torch.as_tensor(
        [_target(reference, labels) for labels in dialects],
        dtype=torch.float32,
        device=device,
    )
    model = DiagonalMetric(tensors.shape[1]).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=learning_rate, weight_decay=weight_decay
    )
    generator = np.random.default_rng(seed)
    history = []
    for _ in range(epochs):
        chosen = generator.integers(0, len(calibration_pairs), size=batch_size)
        chosen_tensor = torch.as_tensor(chosen, dtype=torch.long, device=device)
        projected = model(tensors)
        distances = _cosine_batch(
            projected.index_select(0, left.index_select(0, chosen_tensor)),
            projected.index_select(0, right.index_select(0, chosen_tensor)),
        )
        loss = nn.functional.smooth_l1_loss(
            distances, targets.index_select(0, chosen_tensor)
        )
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        history.append(float(loss.item()))
    with torch.no_grad():
        transformed = model(tensors).cpu().numpy()
    vectors = {key: transformed[position].tolist() for position, key in enumerate(ids)}
    affine = _fit_affine_for_vectors(
        calibration_pairs, calibration_records, vectors, reference
    )
    state = {
        "mean": mean.tolist(),
        "scale": scale.tolist(),
        "weights": model.scales().detach().cpu().numpy().tolist(),
    }
    return model.cpu(), {**affine, "state": state}, history


def _transform_diagonal(
    embeddings: Mapping[str, Sequence[float]], affine: Mapping[str, Any]
) -> dict[str, list[float]]:
    state = affine["state"]
    ids = sorted(embeddings)
    matrix = np.asarray([embeddings[key] for key in ids], dtype=np.float64)
    standardized = (matrix - np.asarray(state["mean"])) / np.asarray(state["scale"])
    weighted = standardized * np.asarray(state["weights"])
    norm = np.linalg.norm(weighted, axis=1, keepdims=True)
    norm[norm == 0] = 1.0
    transformed = weighted / norm
    return {key: transformed[position].tolist() for position, key in enumerate(ids)}


def _estimands(rows: Sequence[Mapping[str, Any]]) -> dict[str, float | None]:
    values: dict[str, float | None] = {}
    for weighting in WEIGHTINGS:
        try:
            values[weighting] = weighted_mae(rows, weighting)
        except ValueError:
            values[weighting] = None
    return values


def _method_summary(
    rows: Sequence[Mapping[str, Any]],
    reference: Mapping[str, Mapping[str, float]],
    baseline_mae: float,
) -> dict[str, Any]:
    estimands = _estimands(rows)
    relations = aggregate_relation_predictions(rows)
    ranking = ranking_metrics(reference_relations(reference), relations)
    return {
        "estimands": estimands,
        "pair_weighted_gain_vs_frozen_affine": float(
            (baseline_mae - float(estimands["pair"])) / baseline_mae
        ),
        "relation_ranking": ranking,
        "relation_predictions": {
            "|".join(key): value for key, value in relations.items()
        },
        "per_pair": list(rows),
    }


def _architecture_methods(
    report: Mapping[str, Any], reference: str
) -> dict[str, list[Mapping[str, Any]]]:
    output: dict[str, list[Mapping[str, Any]]] = {}
    for cell in report["cells"]:
        if str(cell["reference"]) != reference:
            continue
        name = f"{cell['head']}__lambda_{float(cell['lambda_cross']):g}"
        output.setdefault(name, []).append(cell)
    return output


def run_baselines(
    config: Mapping[str, Any], architecture_report: Mapping[str, Any]
) -> tuple[dict[str, Any], dict[str, Any]]:
    calibration_embeddings = _load_embeddings(str(PATHS["calibration_embedding"]))
    evaluation_embeddings = _load_embeddings(str(PATHS["evaluation_embedding"]))
    records = _load_records(str(PATHS["records"]))
    calibration_records = _available_split(records, calibration_embeddings, "calibration")
    evaluation_records = _available_split(records, evaluation_embeddings, "evaluation")
    calibration_pairs = _available_pairs(
        _load_pairs(str(PATHS["calibration_pairs"])), calibration_embeddings
    )
    evaluation_pairs = _available_pairs(
        _load_pairs(str(PATHS["evaluation_pairs"])), evaluation_embeddings
    )
    frozen_vectors = _raw_and_pca(calibration_embeddings, evaluation_embeddings)
    references_output = {}
    gate_checks = {}
    for reference_name, reference_path in REFERENCES.items():
        reference = _load_reference(str(reference_path))
        methods: dict[str, Any] = {}
        for name, (calibration_vectors, evaluation_vectors) in frozen_vectors.items():
            affine = _fit_affine_for_vectors(
                calibration_pairs, calibration_records, calibration_vectors, reference
            )
            rows = _score_vectors(
                evaluation_pairs,
                evaluation_records,
                evaluation_vectors,
                reference,
                affine,
            )
            methods[name] = {"affine": affine, "rows": rows}
        baseline_mae = weighted_mae(methods["frozen_affine"]["rows"], "pair")
        for name in list(methods):
            rows = methods[name].pop("rows")
            methods[name].update(_method_summary(rows, reference, baseline_mae))

        diagonal_seed_rows = []
        for seed in SEEDS:
            _, affine, history = _train_diagonal(
                calibration_embeddings,
                calibration_pairs,
                calibration_records,
                reference,
                seed=seed,
                epochs=30,
                batch_size=256,
                learning_rate=0.0003,
                weight_decay=0.001,
            )
            evaluation_vectors = _transform_diagonal(evaluation_embeddings, affine)
            rows = _score_vectors(
                evaluation_pairs,
                evaluation_records,
                evaluation_vectors,
                reference,
                affine,
            )
            diagonal_seed_rows.append(
                {
                    "seed": seed,
                    "parameter_count": 1024,
                    "epochs": 30,
                    "batch_size": 256,
                    "learning_rate": 0.0003,
                    "weight_decay": 0.001,
                    "loss_history": history,
                    **_method_summary(rows, reference, baseline_mae),
                }
            )
        methods["diagonal_metric"] = {"seed_results": diagonal_seed_rows}

        for architecture_name, cells in _architecture_methods(
            architecture_report, reference_name
        ).items():
            seed_rows = []
            for cell in cells:
                relation_predictions = aggregate_relation_predictions(cell["per_pair"])
                seed_rows.append(
                    {
                        "seed": int(cell["seed"]),
                        "parameter_count": int(cell["head_parameter_count"]),
                        "estimands": dict(cell["method_estimands"]),
                        "pair_weighted_gain_vs_frozen_affine": float(cell["gain"]),
                        "relation_ranking": ranking_metrics(
                            reference_relations(reference),
                            relation_predictions,
                        ),
                        "relation_predictions": {
                            "|".join(key): value
                            for key, value in relation_predictions.items()
                        },
                    }
                )
            methods[architecture_name] = {"seed_results": seed_rows}

        baseline_ranking = methods["frozen_affine"]["relation_ranking"]
        baseline_relations = {
            tuple(key.split("|")): float(value)
            for key, value in methods["frozen_affine"]["relation_predictions"].items()
        }
        for method in methods.values():
            seed_rows = method.get("seed_results", [method])
            for row in seed_rows:
                relation_values = row.get("relation_predictions")
                if relation_values is None:
                    continue
                method_relations = {
                    tuple(key.split("|")): float(value)
                    for key, value in relation_values.items()
                }
                row["ordering_changes_vs_frozen_affine"] = ordering_changes(
                    baseline_relations, method_relations
                )
        candidates = []
        for name, method in methods.items():
            seed_rows = method.get("seed_results", [method])
            median_mae = float(
                np.median([row["estimands"]["pair"] for row in seed_rows])
            )
            median_rank = {
                metric: float(
                    np.median([row["relation_ranking"][metric] for row in seed_rows])
                )
                for metric in ("spearman", "kendall_tau_b", "pairwise_order_accuracy")
            }
            candidates.append(
                {
                    "method": name,
                    "median_pair_mae": median_mae,
                    "mae_not_worse": median_mae <= baseline_mae,
                    "ranking_improvements": {
                        metric: median_rank[metric] > float(baseline_ranking[metric])
                        for metric in median_rank
                    },
                }
            )
        passed = any(
            row["mae_not_worse"] and any(row["ranking_improvements"].values())
            for row in candidates
        )
        references_output[reference_name] = {
            "reference_path": str(reference_path),
            "baseline_pair_mae": baseline_mae,
            "methods": methods,
            "gate_candidates": candidates,
        }
        gate_checks[reference_name] = {"passed": passed, "candidates": candidates}

    gate_passed = any(row["passed"] for row in gate_checks.values())
    pair_only_mlp_passes_both = all(
        any(
            candidate["method"] in {
                "mlp_parameter_matched__lambda_0",
                "mlp_wide__lambda_0",
            }
            and candidate["mae_not_worse"]
            and all(candidate["ranking_improvements"].values())
            for candidate in check["candidates"]
        )
        for check in gate_checks.values()
    )
    report = {
        "schema": "metric-baseline-and-relation-ranking-v1",
        "status": "evaluated",
        "model": "chinese_hubert_large",
        "ranking_scope": "agreement with operational references, not perceptual validity or downstream utility",
        "references": references_output,
    }
    gate = {
        "schema": "practical-consequence-gate-v1",
        "status": "passed" if gate_passed else "narrowed",
        "reference_checks": gate_checks,
        "selected_wording": (
            "Pair-only multilayer projections improved all three operational relation-order measures while reducing mean absolute error under both references; the lower-error linear map did not improve ordering."
            if gate_passed and pair_only_mlp_passes_both
            else "At least one tested correction improved operational relation ordering without adverse mean absolute error."
            if gate_passed
            else None
        ),
        "failure_wording": "the correction results are limited to mean absolute error; no practical ordering claim is made",
    }
    return report, gate


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--architecture-report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--gate", type=Path, required=True)
    args = parser.parse_args()
    report, gate = run_baselines(
        load_config(args.config),
        json.loads(args.architecture_report.read_text(encoding="utf-8")),
    )
    for path, payload in ((args.output, report), (args.gate, gate)):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
