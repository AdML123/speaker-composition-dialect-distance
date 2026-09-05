"""Projection-head primitives for same-speaker cross-dialect supervision."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from math import isfinite
import random
import json
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from torch import nn

from .calibration_leakage_audit import audit_projection_sources


class SupportError(ValueError):
    """Raised when a locked calibration support requirement is not met."""


@dataclass(frozen=True)
class FeatureStandardizer:
    mean: np.ndarray
    scale: np.ndarray

    @property
    def dimension(self) -> int:
        return int(self.mean.shape[0])

    def transform(self, embeddings: Mapping[str, Sequence[float]]) -> dict[str, list[float]]:
        transformed: dict[str, list[float]] = {}
        for key in sorted(embeddings):
            vector = np.asarray(embeddings[key], dtype=np.float64)
            if vector.ndim != 1 or vector.shape[0] != self.dimension:
                raise ValueError(f"embedding {key} has invalid dimension")
            if not np.isfinite(vector).all():
                raise ValueError(f"embedding {key} contains non-finite values")
            transformed[key] = ((vector - self.mean) / self.scale).tolist()
        return transformed


def fit_standardizer(embeddings: Mapping[str, Sequence[float]]) -> FeatureStandardizer:
    if not embeddings:
        raise ValueError("at least one embedding is required")
    keys = sorted(embeddings)
    matrix = np.asarray([embeddings[key] for key in keys], dtype=np.float64)
    if matrix.ndim != 2 or not np.isfinite(matrix).all():
        raise ValueError("embeddings must be a finite two-dimensional matrix")
    mean = matrix.mean(axis=0)
    scale = matrix.std(axis=0)
    scale = np.where(scale == 0.0, 1e-6, scale)
    return FeatureStandardizer(mean=mean, scale=scale)


def _reference_value(reference: Mapping[str, Mapping[str, float]], left: str, right: str) -> float:
    try:
        value = float(reference[left][right])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"unknown reference dialect pair: {left}, {right}") from exc
    if not isfinite(value):
        raise ValueError("reference contains non-finite value")
    return value


def _record_index(records: Sequence[Mapping[str, object]]) -> dict[str, Mapping[str, object]]:
    indexed: dict[str, Mapping[str, object]] = {}
    splits_by_speaker: dict[str, set[str]] = {}
    for record in records:
        utterance_id = str(record["utterance_id"])
        if utterance_id in indexed:
            raise ValueError(f"duplicate utterance_id: {utterance_id}")
        speaker = str(record["speaker_id"])
        split = str(record.get("split", ""))
        indexed[utterance_id] = record
        splits_by_speaker.setdefault(speaker, set()).add(split)
    if any(len(splits) > 1 for splits in splits_by_speaker.values()):
        raise ValueError("speaker occurs across multiple splits")
    return indexed


def _normalized_reference(reference: Mapping[str, Mapping[str, float]]) -> tuple[dict[str, dict[str, float]], float]:
    values = [
        _reference_value(reference, left, right)
        for left in sorted(reference)
        for right in sorted(reference[left])
    ]
    maximum = max(values) if values else 0.0
    if maximum <= 0:
        raise ValueError("reference maximum must be positive")
    normalized = {
        left: {
            right: _reference_value(reference, left, right) / maximum
            for right in sorted(reference[left])
        }
        for left in sorted(reference)
    }
    return normalized, maximum


def build_training_examples(
    records: Sequence[Mapping[str, object]],
    pair_manifest: Sequence[Mapping[str, object]],
    reference: Mapping[str, Mapping[str, float]],
) -> dict[str, object]:
    """Build deterministic generic and same-speaker cross-dialect pools."""
    indexed = _record_index(records)
    normalized_reference, reference_max = _normalized_reference(reference)
    pair_examples: list[dict[str, object]] = []
    pair_keys: set[tuple[str, str]] = set()

    for pair in pair_manifest:
        endpoints = list(pair.get("source_utterance_ids", []))
        if len(endpoints) != 2:
            raise ValueError("each pair must contain exactly two endpoints")
        left_id, right_id = map(str, endpoints)
        if left_id not in indexed or right_id not in indexed:
            raise ValueError("pair endpoint is missing from records")
        left = indexed[left_id]
        right = indexed[right_id]
        if left.get("split") != right.get("split"):
            raise ValueError("pair endpoints cross splits")
        left_dialect = str(left["dialect_label"])
        right_dialect = str(right["dialect_label"])
        target = normalized_reference[left_dialect][right_dialect]
        key = tuple(sorted((left_id, right_id)))
        pair_keys.add(key)
        pair_examples.append(
            {
                "pair_id": str(pair.get("pair_id", f"pair-{left_id}-{right_id}")),
                "utterance_ids": [left_id, right_id],
                "speaker_ids": [str(left["speaker_id"]), str(right["speaker_id"])],
                "dialect_labels": [left_dialect, right_dialect],
                "group": str(pair.get("group", "")),
                "target": target,
            }
        )

    by_speaker_dialect: dict[str, dict[str, list[str]]] = {}
    for record in records:
        speaker = str(record["speaker_id"])
        dialect = str(record["dialect_label"])
        by_speaker_dialect.setdefault(speaker, {}).setdefault(dialect, []).append(str(record["utterance_id"]))

    cross_examples: list[dict[str, object]] = []
    for speaker in sorted(by_speaker_dialect):
        dialects = by_speaker_dialect[speaker]
        if len(dialects) < 2:
            continue
        for left_dialect, right_dialect in combinations(sorted(dialects), 2):
            for left_id in sorted(dialects[left_dialect]):
                for right_id in sorted(dialects[right_dialect]):
                    key = tuple(sorted((left_id, right_id)))
                    cross_examples.append(
                        {
                            "pair_id": f"cross-{left_id}-{right_id}",
                            "utterance_ids": [left_id, right_id],
                            "speaker_id": speaker,
                            "speaker_ids": [speaker, speaker],
                            "dialect_labels": [left_dialect, right_dialect],
                            "group": "cross_dialect",
                            "target": normalized_reference[left_dialect][right_dialect],
                        }
                    )

    cross_keys = {
        tuple(sorted(map(str, example["utterance_ids"])))
        for example in cross_examples
    }
    return {
        "pair_examples": sorted(pair_examples, key=lambda item: (str(item["pair_id"]), tuple(item["utterance_ids"]))),
        "cross_dialect_examples": sorted(
            cross_examples,
            key=lambda item: (
                str(item["speaker_id"]),
                tuple(item["dialect_labels"]),
                tuple(item["utterance_ids"]),
            ),
        ),
        "same_speaker_cross_dialect_count": len(cross_examples),
        "pair_cross_overlap_count": len(pair_keys & cross_keys),
        "reference_max": reference_max,
    }


class ProjectionHead(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, dropout: float) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return nn.functional.normalize(self.network(inputs), p=2, dim=1)


class LinearProjectionHead(nn.Module):
    def __init__(self, input_dim: int, output_dim: int) -> None:
        super().__init__()
        self.linear = nn.Linear(input_dim, output_dim)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return nn.functional.normalize(self.linear(inputs), p=2, dim=1)


def parameter_count(model: nn.Module) -> int:
    """Count trainable model parameters without auxiliary classifier weights."""
    return sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )


def _make_projection_model(head_config: Mapping[str, object], head_kind: str) -> nn.Module:
    if head_kind == "mlp":
        return ProjectionHead(
            int(head_config["input_dim"]),
            int(head_config["hidden_dim"]),
            int(head_config["output_dim"]),
            float(head_config["dropout"]),
        )
    if head_kind == "linear":
        return LinearProjectionHead(
            int(head_config["input_dim"]),
            int(head_config["output_dim"]),
        )
    raise ValueError(f"unknown head_kind: {head_kind}")


def compute_projection_loss(
    pair_distances: torch.Tensor,
    pair_targets: torch.Tensor,
    cross_distances: torch.Tensor,
    cross_targets: torch.Tensor,
    dialect_logits: torch.Tensor,
    dialect_targets: torch.Tensor,
    lambda_cross: float,
    lambda_dialect: float,
    cross_loss_mode: str = "ordinary",
    aggregation_mode: str = "separate",
) -> dict[str, torch.Tensor]:
    smooth_l1 = nn.SmoothL1Loss()
    smooth_l1_none = nn.SmoothL1Loss(reduction="none")
    if pair_distances.numel() == 0:
        pair = cross_distances.new_zeros(())
    else:
        pair = smooth_l1(pair_distances, pair_targets)
    if cross_distances.numel() == 0:
        cross = pair.new_zeros(())
    elif cross_loss_mode == "prevalence_balanced":
        nonzero = cross_targets != 0
        zero = ~nonzero
        terms = []
        if bool(zero.any()):
            terms.append(smooth_l1(cross_distances[zero], cross_targets[zero]))
        if bool(nonzero.any()):
            terms.append(smooth_l1(cross_distances[nonzero], cross_targets[nonzero]))
        cross = torch.stack(terms).mean() if terms else pair.new_zeros(())
    elif cross_loss_mode == "ordinary":
        cross = smooth_l1(cross_distances, cross_targets)
    else:
        raise ValueError(f"unknown cross_loss_mode: {cross_loss_mode}")
    if dialect_logits.numel() == 0:
        dialect = pair.new_zeros(())
    else:
        dialect = nn.CrossEntropyLoss()(dialect_logits, dialect_targets)
    if aggregation_mode == "separate":
        distance_total = pair + lambda_cross * cross
    elif aggregation_mode == "mixed_mean":
        if cross_loss_mode != "ordinary":
            raise ValueError("mixed_mean aggregation requires ordinary cross loss")
        denominator = pair_distances.numel() + cross_distances.numel()
        if denominator == 0:
            distance_total = pair.new_zeros(())
        else:
            distance_total = (
                smooth_l1_none(pair_distances, pair_targets).sum()
                + lambda_cross * smooth_l1_none(cross_distances, cross_targets).sum()
            ) / denominator
    else:
        raise ValueError(f"unknown aggregation_mode: {aggregation_mode}")
    total = distance_total + lambda_dialect * dialect
    return {"pair": pair, "cross": cross, "dialect": dialect, "total": total}


def make_grouped_calibration_folds(
    records: Sequence[Mapping[str, object]],
    n_splits: int,
    seed: int,
) -> list[dict[str, object]]:
    if n_splits < 2:
        raise ValueError("n_splits must be at least 2")
    by_speaker: dict[str, list[Mapping[str, object]]] = {}
    for record in records:
        if str(record.get("split")) != "calibration":
            raise SupportError("grouped calibration folds require calibration records")
        by_speaker.setdefault(str(record["speaker_id"]), []).append(record)
    speakers = sorted(by_speaker)
    if len(speakers) < n_splits:
        raise SupportError("not enough speakers for requested folds")
    rng = random.Random(seed)
    shuffled = list(speakers)
    rng.shuffle(shuffled)
    fold_speakers = [sorted(shuffled[index::n_splits]) for index in range(n_splits)]
    folds: list[dict[str, object]] = []
    for validation_speakers in fold_speakers:
        dialect_support = sum(
            len({str(record["dialect_label"]) for record in by_speaker[speaker]}) >= 2
            for speaker in validation_speakers
        )
        if dialect_support < 1:
            raise SupportError("validation fold lacks a multi-dialect speaker")
        validation_ids = {
            str(record["utterance_id"])
            for speaker in validation_speakers
            for record in by_speaker[speaker]
        }
        folds.append(
            {
                "validation_speakers": validation_speakers,
                "training_speakers": sorted(set(speakers) - set(validation_speakers)),
                "validation_utterance_ids": sorted(validation_ids),
                "cross_dialect_validation_count": dialect_support,
            }
        )
    return folds


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _pair_tensors(
    examples: Sequence[Mapping[str, object]],
    vectors: Mapping[str, Sequence[float]],
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, list[str]]:
    first = []
    second = []
    targets = []
    ids = []
    for example in examples:
        left, right = map(str, example["utterance_ids"])
        if left not in vectors or right not in vectors:
            raise ValueError(f"missing embedding for pair {example.get('pair_id')}")
        first.append(vectors[left])
        second.append(vectors[right])
        targets.append(float(example["target"]))
        ids.append(str(example.get("pair_id", "")))
    if not first:
        return (
            torch.empty((0, 0), dtype=torch.float32, device=device),
            torch.empty((0, 0), dtype=torch.float32, device=device),
            torch.empty((0,), dtype=torch.float32, device=device),
        )
    return (
        torch.tensor(np.asarray(first), dtype=torch.float32, device=device),
        torch.tensor(np.asarray(second), dtype=torch.float32, device=device),
        torch.tensor(targets, dtype=torch.float32, device=device),
    )


def _cosine_batch(first: torch.Tensor, second: torch.Tensor) -> torch.Tensor:
    if first.numel() == 0:
        return first.new_empty((0,))
    return 1.0 - nn.functional.cosine_similarity(first, second, dim=1)


def _fit_affine(predicted: Sequence[float], targets: Sequence[float]) -> dict[str, float]:
    x = np.asarray(predicted, dtype=np.float64)
    y = np.asarray(targets, dtype=np.float64)
    if len(x) == 0:
        return {"slope": 1.0, "intercept": 0.0}
    design = np.column_stack((x, np.ones(len(x))))
    slope, intercept = np.linalg.lstsq(design, y, rcond=None)[0]
    return {"slope": float(slope), "intercept": float(intercept)}


def fit_affine_from_training_pairs(predicted: Sequence[float], targets: Sequence[float]) -> dict[str, float]:
    """Fit calibration only from training-pair predictions and targets."""
    return _fit_affine(predicted, targets)


def _apply_affine(values: Sequence[float], affine: Mapping[str, float]) -> np.ndarray:
    return float(affine["slope"]) * np.asarray(values, dtype=np.float64) + float(affine["intercept"])


def _mae(predicted: Sequence[float], targets: Sequence[float]) -> float:
    if not predicted:
        return float("nan")
    return float(np.mean(np.abs(np.asarray(predicted) - np.asarray(targets))))


def _dialect_index(records: Sequence[Mapping[str, object]]) -> dict[str, int]:
    return {label: index for index, label in enumerate(sorted({str(r["dialect_label"]) for r in records}))}


def train_projection_head(
    train_embeddings: Mapping[str, Sequence[float]],
    train_records: Sequence[Mapping[str, object]],
    train_pairs: Sequence[Mapping[str, object]],
    reference: Mapping[str, Mapping[str, float]],
    *,
    lambda_cross: float,
    lambda_dialect: float,
    learning_rate: float,
    weight_decay: float,
    config: Mapping[str, object],
    seed: int,
    validation_embeddings: Mapping[str, Sequence[float]],
    validation_records: Sequence[Mapping[str, object]],
    validation_pairs: Sequence[Mapping[str, object]],
    pair_examples_override: Sequence[Mapping[str, object]] | None = None,
    cross_examples_override: Sequence[Mapping[str, object]] | None = None,
    dialect_label_map: Mapping[str, str] | None = None,
    head_kind: str = "mlp",
    fixed_epochs: int | None = None,
    cross_loss_mode: str = "ordinary",
    aggregation_mode: str = "separate",
    exposure_ratio: tuple[int, int] | None = None,
    regularization_strength: float = 0.0,
    replace_cross_with_regularization: bool = False,
    record_gradient_budget: bool = False,
    gradient_log_interval: int = 5,
) -> dict[str, object]:
    _set_seed(seed)
    head_config = config.get("projection_head", config)
    available_train_ids = set(train_embeddings)
    effective_train_records = [
        record
        for record in train_records
        if str(record["utterance_id"]) in available_train_ids
    ]
    standardizer = fit_standardizer(train_embeddings)
    train_x = standardizer.transform(train_embeddings)
    val_x = standardizer.transform(validation_embeddings)
    examples = build_training_examples(effective_train_records, train_pairs, reference)
    pair_examples = list(
        examples["pair_examples"] if pair_examples_override is None else pair_examples_override
    )
    cross_examples = list(
        examples["cross_dialect_examples"]
        if cross_examples_override is None
        else cross_examples_override
    )
    validation_examples = build_training_examples(validation_records, validation_pairs, reference)
    validation_pair_examples = validation_examples["pair_examples"]
    dialect_to_index = _dialect_index(effective_train_records)
    model = _make_projection_model(head_config, head_kind)
    classifier = nn.Linear(int(head_config["output_dim"]), len(dialect_to_index))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    classifier.to(device)
    optimizer = torch.optim.AdamW(
        list(model.parameters()) + list(classifier.parameters()),
        lr=float(learning_rate),
        weight_decay=float(weight_decay),
    )
    train_matrix = torch.tensor(np.asarray([train_x[key] for key in sorted(train_x)]), dtype=torch.float32, device=device)
    train_labels = torch.tensor(
        [
            dialect_to_index[
                str((dialect_label_map or {}).get(str(record["dialect_label"]), str(record["dialect_label"])))
            ]
            for record in sorted(effective_train_records, key=lambda r: str(r["utterance_id"]))
        ],
        dtype=torch.long,
        device=device,
    )
    pair_left, pair_right, pair_targets = _pair_tensors(pair_examples, train_x, device)
    cross_left, cross_right, cross_targets = _pair_tensors(cross_examples, train_x, device)
    epochs = int(head_config.get("max_epochs", 100)) if fixed_epochs is None else int(fixed_epochs)
    if epochs <= 0:
        raise ValueError("fixed_epochs must be positive")
    patience = int(head_config.get("early_stopping_patience", 10))
    batch_size = max(1, int(head_config.get("batch_size", 256)))
    pair_count = int(pair_targets.numel())
    cross_count = int(cross_targets.numel())
    cross_zero_indices = np.flatnonzero(cross_targets.detach().cpu().numpy() == 0)
    cross_nonzero_indices = np.flatnonzero(cross_targets.detach().cpu().numpy() != 0)
    if pair_count and cross_count and exposure_ratio is not None:
        pair_weight, cross_weight = map(int, exposure_ratio)
        if pair_weight <= 0 or cross_weight <= 0:
            raise ValueError("exposure_ratio values must be positive")
        n_cross = max(1, round(batch_size * cross_weight / (pair_weight + cross_weight)))
        n_cross = min(batch_size - 1, n_cross)
        n_pair = batch_size - n_cross
        cross_fraction = n_cross / batch_size
    elif pair_count and cross_count:
        cross_fraction = min(0.5, cross_count / max(pair_count + cross_count, 1))
        n_cross = min(batch_size - 1, max(1, int(round(batch_size * cross_fraction))))
        n_pair = batch_size - n_cross
    elif pair_count:
        cross_fraction = 0.0
        n_pair, n_cross = batch_size, 0
    elif cross_count:
        cross_fraction = 1.0
        n_pair, n_cross = 0, batch_size
    else:
        raise SupportError("training requires at least one pair example")
    rng = np.random.default_rng(seed)
    val_ids = sorted(val_x)
    val_index = {key: index for index, key in enumerate(val_ids)}
    val_matrix = torch.tensor(
        np.asarray([val_x[key] for key in val_ids]),
        dtype=torch.float32,
        device=device,
    )
    validation_left_indices = torch.tensor(
        [val_index[str(example["utterance_ids"][0])] for example in validation_pair_examples],
        dtype=torch.long,
        device=device,
    )
    validation_right_indices = torch.tensor(
        [val_index[str(example["utterance_ids"][1])] for example in validation_pair_examples],
        dtype=torch.long,
        device=device,
    )
    validation_targets = [float(example["target"]) for example in validation_pair_examples]
    best_state = None
    best_affine = {"slope": 1.0, "intercept": 0.0}
    best_mae = float("inf")
    stale = 0
    history = []
    for epoch in range(epochs):
        model.train()
        classifier.train()
        optimizer.zero_grad()
        z = model(train_matrix)
        if n_pair:
            pair_indices = rng.integers(0, pair_count, size=n_pair)
            pair_index_tensor = torch.tensor(pair_indices, dtype=torch.long, device=device)
            pair_distances = _cosine_batch(
                model(pair_left.index_select(0, pair_index_tensor)),
                model(pair_right.index_select(0, pair_index_tensor)),
            )
            pair_batch_targets = pair_targets.index_select(0, pair_index_tensor)
        else:
            pair_distances = z.new_empty((0,))
            pair_batch_targets = z.new_empty((0,))
        if n_cross:
            if cross_loss_mode == "prevalence_balanced" and len(cross_zero_indices) and len(cross_nonzero_indices):
                n_zero = n_cross // 2
                n_nonzero = n_cross - n_zero
                cross_indices = np.concatenate((
                    rng.choice(cross_zero_indices, size=n_zero, replace=True),
                    rng.choice(cross_nonzero_indices, size=n_nonzero, replace=True),
                ))
                rng.shuffle(cross_indices)
            else:
                cross_indices = rng.integers(0, cross_count, size=n_cross)
            cross_index_tensor = torch.tensor(cross_indices, dtype=torch.long, device=device)
            cross_distances = _cosine_batch(
                model(cross_left.index_select(0, cross_index_tensor)),
                model(cross_right.index_select(0, cross_index_tensor)),
            )
            cross_batch_targets = cross_targets.index_select(0, cross_index_tensor)
        else:
            cross_distances = z.new_empty((0,))
            cross_batch_targets = z.new_empty((0,))
        logits = classifier(z)
        losses = compute_projection_loss(
            pair_distances,
            pair_batch_targets,
            cross_distances,
            cross_batch_targets,
            logits,
            train_labels,
            0.0 if replace_cross_with_regularization else lambda_cross,
            lambda_dialect,
            cross_loss_mode,
            aggregation_mode,
        )
        if replace_cross_with_regularization:
            regularizer = torch.stack([parameter.square().mean() for parameter in model.parameters()]).mean()
            losses["regularizer"] = regularizer
            losses["total"] = losses["total"] + float(regularization_strength) * regularizer
        else:
            losses["regularizer"] = losses["total"].new_zeros(())
        gradient_budget = None
        should_log_gradient = record_gradient_budget and (
            epoch == 0 or (epoch + 1) % max(int(gradient_log_interval), 1) == 0 or epoch + 1 == epochs
        )
        if should_log_gradient and pair_distances.numel() and cross_distances.numel():
            parameters = tuple(model.parameters())
            pair_grad = torch.autograd.grad(losses["pair"], parameters, retain_graph=True, allow_unused=True)
            cross_grad = torch.autograd.grad(losses["cross"], parameters, retain_graph=True, allow_unused=True)
            pair_flat = torch.cat([(value if value is not None else torch.zeros_like(parameter)).reshape(-1) for value, parameter in zip(pair_grad, parameters)])
            cross_flat = torch.cat([(value if value is not None else torch.zeros_like(parameter)).reshape(-1) for value, parameter in zip(cross_grad, parameters)])
            pair_norm = torch.linalg.vector_norm(pair_flat)
            cross_norm = torch.linalg.vector_norm(cross_flat)
            denominator = pair_norm * cross_norm
            cosine = torch.dot(pair_flat, cross_flat) / denominator if float(denominator) > 0 else pair_norm.new_zeros(())
            effective_lambda = 0.0 if replace_cross_with_regularization else float(lambda_cross)
            eta_denominator = pair_norm + effective_lambda * cross_norm
            eta_sep = effective_lambda * cross_norm / eta_denominator if float(eta_denominator) > 0 else pair_norm.new_zeros(())
            eta_mix_denominator = n_pair * pair_norm + effective_lambda * n_cross * cross_norm
            eta_mix = effective_lambda * n_cross * cross_norm / eta_mix_denominator if float(eta_mix_denominator) > 0 else pair_norm.new_zeros(())
            gradient_budget = {"pair_norm": float(pair_norm), "cross_norm": float(cross_norm), "cosine": float(cosine), "eta_sep": float(eta_sep), "eta_mix": float(eta_mix)}
        losses["total"].backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            train_pred = _cosine_batch(model(pair_left), model(pair_right)).detach().cpu().numpy().astype(float).tolist()
            train_affine = fit_affine_from_training_pairs(train_pred, pair_targets.detach().cpu().numpy().astype(float).tolist())
            val_z = model(val_matrix)
            val_pred = _cosine_batch(
                val_z.index_select(0, validation_left_indices),
                val_z.index_select(0, validation_right_indices),
            ).detach().cpu().numpy().astype(float).tolist()
        affine = train_affine
        val_mae = _mae(_apply_affine(val_pred, affine).tolist(), validation_targets)
        cross_zero = cross_batch_targets == 0
        cross_nonzero = ~cross_zero
        history.append({
            "epoch": epoch + 1,
            "loss": float(losses["total"].item()),
            "pair_loss": float(losses["pair"].item()),
            "cross_loss": float(losses["cross"].item()),
            "dialect_loss": float(losses["dialect"].item()),
            "regularizer": float(losses["regularizer"].item()),
            "cross_zero_count": int(cross_zero.sum().item()),
            "cross_nonzero_count": int(cross_nonzero.sum().item()),
            "cross_zero_distance": float(cross_distances[cross_zero].mean().item()) if bool(cross_zero.any()) else None,
            "cross_nonzero_distance": float(cross_distances[cross_nonzero].mean().item()) if bool(cross_nonzero.any()) else None,
            "validation_mae": val_mae,
            "gradient_budget": gradient_budget,
        })
        if fixed_epochs is not None:
            continue
        if val_mae < best_mae:
            best_mae = val_mae
            best_state = {
                "model": {key: value.detach().cpu().clone() for key, value in model.state_dict().items()},
                "classifier": {key: value.detach().cpu().clone() for key, value in classifier.state_dict().items()},
            }
            best_affine = affine
            stale = 0
        else:
            stale += 1
            if stale >= patience:
                break
    if fixed_epochs is not None:
        best_state = {
            "model": {key: value.detach().cpu().clone() for key, value in model.state_dict().items()},
            "classifier": {key: value.detach().cpu().clone() for key, value in classifier.state_dict().items()},
        }
        best_affine = train_affine
        best_mae = float("nan")
    if best_state is None:
        raise RuntimeError("training produced no finite validation checkpoint")
    return {
        "model_state": best_state["model"],
        "classifier_state": best_state["classifier"],
        "standardizer": standardizer,
        "affine_scale": best_affine,
        "selected_epoch": int(fixed_epochs if fixed_epochs is not None else len(history) - stale),
        "validation_mae": best_mae,
        "loss_history": history,
        "same_speaker_cross_dialect_count": len(cross_examples),
        "dialect_to_index": dialect_to_index,
        "hidden_dim": int(head_config["hidden_dim"]),
        "output_dim": int(head_config["output_dim"]),
        "head_kind": head_kind,
        "cross_loss_mode": cross_loss_mode,
        "aggregation_mode": aggregation_mode,
        "exposure_ratio": list(exposure_ratio) if exposure_ratio is not None else None,
        "regularization_strength": float(regularization_strength),
        "replace_cross_with_regularization": bool(replace_cross_with_regularization),
        "batch_composition": {
            "batch_size": batch_size,
            "n_pair": n_pair,
            "n_cross": n_cross,
            "cross_fraction": cross_fraction,
        },
    }


def fit_projection_head_cv(
    embeddings: Mapping[str, Sequence[float]],
    records: Sequence[Mapping[str, object]],
    pair_manifest: Sequence[Mapping[str, object]],
    reference: Mapping[str, Mapping[str, float]],
    config: Mapping[str, object],
    seed: int,
    progress_label: str | None = None,
    head_kind: str = "mlp",
) -> dict[str, object]:
    head_config = config.get("projection_head", config)
    folds = make_grouped_calibration_folds(records, int(head_config["inner_cv_folds"]), seed)
    records_by_id = {str(record["utterance_id"]): record for record in records}
    available_ids = set(embeddings)
    candidates = [
        (float(cross), float(dialect), float(lr), float(wd))
        for cross in head_config["lambda_cross_grid"]
        for dialect in head_config["lambda_dialect_grid"]
        for lr in head_config["learning_rate_grid"]
        for wd in head_config["weight_decay_grid"]
    ]
    results = []
    for candidate_index, candidate in enumerate(candidates, start=1):
        fold_scores = []
        selected_epochs: list[int] = []
        for fold_index, fold in enumerate(folds):
            train_speakers = set(fold["training_speakers"])
            validation_speakers = set(fold["validation_speakers"])
            train_records = [
                r
                for r in records
                if str(r["speaker_id"]) in train_speakers and str(r["utterance_id"]) in available_ids
            ]
            val_records = [
                r
                for r in records
                if str(r["speaker_id"]) in validation_speakers and str(r["utterance_id"]) in available_ids
            ]
            train_ids = {str(r["utterance_id"]) for r in train_records}
            val_ids = {str(r["utterance_id"]) for r in val_records}
            train_pairs = [
                p
                for p in pair_manifest
                if set(map(str, p["source_utterance_ids"])) <= train_ids
            ]
            val_pairs = [
                p
                for p in pair_manifest
                if set(map(str, p["source_utterance_ids"])) <= val_ids
            ]
            if not train_pairs or not val_pairs:
                raise SupportError("fold lacks pair support")
            fitted = train_projection_head(
                {key: value for key, value in embeddings.items() if key in train_ids},
                train_records,
                train_pairs,
                reference,
                lambda_cross=candidate[0],
                lambda_dialect=candidate[1],
                learning_rate=candidate[2],
                weight_decay=candidate[3],
                config=config,
                seed=seed + fold_index,
                validation_embeddings={key: value for key, value in embeddings.items() if key in val_ids},
                validation_records=val_records,
                validation_pairs=val_pairs,
                head_kind=head_kind,
            )
            fold_scores.append(float(fitted["validation_mae"]))
            selected_epochs.append(int(fitted["selected_epoch"]))
        results.append({
            "candidate": candidate,
            "mean_validation_mae": float(np.mean(fold_scores)),
            "selected_epochs": selected_epochs,
        })
        if progress_label:
            print(
                "projection-head cv "
                f"{progress_label} candidate={candidate_index}/{len(candidates)} "
                f"lambda_cross={candidate[0]} lambda_dialect={candidate[1]} "
                f"learning_rate={candidate[2]} weight_decay={candidate[3]} "
                f"mean_validation_mae={float(np.mean(fold_scores)):.6f}",
                file=sys.stderr,
                flush=True,
            )
    selected = min(results, key=lambda item: (item["mean_validation_mae"], item["candidate"]))
    final_epochs = int(round(float(np.median(selected["selected_epochs"]))))
    final = train_projection_head(
        {key: value for key, value in embeddings.items() if key in available_ids},
        [record for record in records if str(record["utterance_id"]) in available_ids],
        pair_manifest,
        reference,
        lambda_cross=selected["candidate"][0],
        lambda_dialect=selected["candidate"][1],
        learning_rate=selected["candidate"][2],
        weight_decay=selected["candidate"][3],
        config=config,
        seed=seed,
        validation_embeddings=embeddings,
        validation_records=records,
        validation_pairs=pair_manifest,
        head_kind=head_kind,
        fixed_epochs=final_epochs,
    )
    final["selected"] = {
        "lambda_cross": selected["candidate"][0],
        "lambda_dialect": selected["candidate"][1],
        "learning_rate": selected["candidate"][2],
        "weight_decay": selected["candidate"][3],
    }
    final["folds_used"] = len(folds)
    final["selected_epoch_from_training_folds"] = final_epochs
    final["best_validation_mae"] = selected["mean_validation_mae"]
    final["cv_results"] = results
    return final


def transform_embeddings(
    embeddings: Mapping[str, Sequence[float]],
    fitted_model: Mapping[str, object],
) -> dict[str, list[float]]:
    standardizer = fitted_model["standardizer"]
    model = _make_projection_model(
        {
            "input_dim": standardizer.dimension,
            "hidden_dim": int(fitted_model.get("hidden_dim", 512)),
            "output_dim": int(fitted_model.get("output_dim", 256)),
            "dropout": 0.0,
        },
        str(fitted_model.get("head_kind", "mlp")),
    )
    model.load_state_dict(fitted_model["model_state"])
    model.eval()
    transformed = standardizer.transform(embeddings)
    with torch.no_grad():
        matrix = torch.tensor(
            np.asarray([transformed[key] for key in sorted(transformed)]),
            dtype=torch.float32,
        )
        output = model(matrix).cpu().numpy()
    return {key: output[index].astype(float).tolist() for index, key in enumerate(sorted(transformed))}


def score_pair_distances(
    pairs: Sequence[Mapping[str, object]],
    transformed_embeddings: Mapping[str, Sequence[float]],
    reference: Mapping[str, Mapping[str, float]],
    affine_scale: Mapping[str, float],
    record_index: Mapping[str, Mapping[str, object]],
) -> list[dict[str, object]]:
    _, maximum = _normalized_reference(reference)
    rows: list[dict[str, object]] = []
    for pair in pairs:
        left, right = map(str, pair["source_utterance_ids"])
        if left not in transformed_embeddings or right not in transformed_embeddings:
            raise ValueError(f"missing embedding for pair {pair.get('pair_id')}")
        left_record = record_index.get(left)
        right_record = record_index.get(right)
        if left_record is None or right_record is None:
            raise ValueError(f"missing record for pair {pair.get('pair_id')}")
        left_tensor = torch.tensor(transformed_embeddings[left], dtype=torch.float64)
        right_tensor = torch.tensor(transformed_embeddings[right], dtype=torch.float64)
        raw = float(_cosine_batch(left_tensor[None, :], right_tensor[None, :])[0])
        labels = [str(left_record["dialect_label"]), str(right_record["dialect_label"])]
        target = _reference_value(reference, labels[0], labels[1]) / maximum
        scaled = float(_apply_affine([raw], affine_scale)[0])
        rows.append(
            {
                "pair_id": str(pair.get("pair_id", f"{pair.get('group', '?')}::{left}::{right}")),
                "group": pair.get("group"),
                "dialect_labels": pair.get("dialect_labels"),
                "speaker_ids": pair.get("speaker_ids"),
                "utterance_ids": pair.get("utterance_ids", pair.get("source_utterance_ids")),
                "matched_stratum": pair.get("matched_stratum"),
                "matched_fields": pair.get("matched_fields"),
                "raw_distance": raw,
                "distance": scaled,
                "target": target,
            }
        )
    return rows


def build_shuffled_cross_dialect_control(
    records: Sequence[Mapping[str, object]],
    seed: int,
) -> list[dict[str, object]]:
    by_speaker: dict[str, dict[str, list[str]]] = {}
    by_dialect: dict[str, list[tuple[str, str]]] = {}
    for record in records:
        speaker = str(record["speaker_id"])
        dialect = str(record["dialect_label"])
        utterance_id = str(record["utterance_id"])
        by_speaker.setdefault(speaker, {}).setdefault(dialect, []).append(utterance_id)
        by_dialect.setdefault(dialect, []).append((speaker, utterance_id))
    examples = []
    for speaker in sorted(by_speaker):
        dialects = by_speaker[speaker]
        for left_dialect, right_dialect in combinations(sorted(dialects), 2):
            for left_id in sorted(dialects[left_dialect]):
                for right_id in sorted(dialects[right_dialect]):
                    examples.append(
                        {
                            "pair_id": f"cross-{left_id}-{right_id}",
                            "utterance_ids": [left_id, right_id],
                            "speaker_id": speaker,
                            "speaker_ids": [speaker, speaker],
                            "dialect_labels": [left_dialect, right_dialect],
                            "group": "cross_dialect",
                            "target": 0.0,
                        }
                    )
    rng = random.Random(seed)
    shuffled_examples: list[dict[str, object]] = []
    for index, example in enumerate(examples):
        left_id, right_id = map(str, example["utterance_ids"])
        left_dialect, right_dialect = map(str, example["dialect_labels"])
        left_speaker = str(example["speaker_id"])
        candidates = [
            (speaker, utterance_id)
            for speaker, utterance_id in by_dialect.get(right_dialect, [])
            if speaker != left_speaker
        ]
        if not candidates:
            candidates = list(by_dialect.get(right_dialect, []))
        if not candidates:
            continue
        right_speaker, shuffled_right_id = candidates[index % len(candidates)]
        shuffled_examples.append(
            {
                **example,
                "utterance_ids": [left_id, shuffled_right_id],
                "speaker_id": f"{left_speaker}|{right_speaker}",
                "speaker_ids": [left_speaker, right_speaker],
            }
        )
    rng.shuffle(shuffled_examples)
    return sorted(
        shuffled_examples,
        key=lambda item: (str(item["pair_id"]), tuple(item["utterance_ids"])),
    )


def _shuffle_cross_examples(
    cross_examples: Sequence[Mapping[str, object]],
    records: Sequence[Mapping[str, object]],
    seed: int,
) -> list[dict[str, object]]:
    """Build a cross-speaker control while preserving dialect targets."""
    by_dialect: dict[str, list[tuple[str, str]]] = {}
    for record in records:
        by_dialect.setdefault(str(record["dialect_label"]), []).append(
            (str(record["speaker_id"]), str(record["utterance_id"]))
        )
    rng = random.Random(seed)
    shuffled: list[dict[str, object]] = []
    for index, example in enumerate(cross_examples):
        left_id, right_id = map(str, example["utterance_ids"])
        left_dialect, right_dialect = map(str, example["dialect_labels"])
        speaker_ids = example.get("speaker_ids", [])
        original_speaker = (
            str(example["speaker_id"])
            if "speaker_id" in example
            else str(speaker_ids[0] if speaker_ids else "")
        )
        left_candidates = list(by_dialect.get(left_dialect, []))
        right_candidates = list(by_dialect.get(right_dialect, []))
        rng.shuffle(left_candidates)
        rng.shuffle(right_candidates)
        selected = None
        for left_speaker, candidate_left in left_candidates:
            for right_speaker, candidate_right in right_candidates:
                if left_speaker != right_speaker:
                    selected = (left_speaker, candidate_left, right_speaker, candidate_right)
                    break
            if selected is not None:
                break
        if selected is None:
            continue
        left_speaker, candidate_left, right_speaker, candidate_right = selected
        shuffled.append(
            {
                **example,
                "utterance_ids": [candidate_left, candidate_right],
                "speaker_id": f"{left_speaker}|{right_speaker}",
                "speaker_ids": [left_speaker, right_speaker],
                "original_speaker_id": original_speaker,
            }
        )
    return sorted(
        shuffled,
        key=lambda item: (str(item["pair_id"]), tuple(item["utterance_ids"])),
    )


def build_permuted_dialect_targets(
    records: Sequence[Mapping[str, object]],
    seed: int,
) -> dict[str, str]:
    labels = sorted({str(record["dialect_label"]) for record in records})
    permutation = list(labels)
    random.Random(seed).shuffle(permutation)
    return dict(zip(labels, permutation))


def build_permuted_target_cross_examples(
    cross_examples: Sequence[Mapping[str, object]],
    seed: int,
) -> list[dict[str, object]]:
    """Permute the targets used by the cross-loss while preserving its pool.

    This is deliberately separate from ``build_permuted_dialect_targets``:
    the latter is an auxiliary classifier-label control, whereas this control
    changes the actual pair-distance supervision consumed by the cross loss.
    Pair identities, speaker endpoints, pool size, and target histogram are
    retained; only the target-to-example assignment changes.
    """
    from .target_permutation_control import permute_pair_distance_targets

    return list(
        permute_pair_distance_targets(cross_examples, seed=seed)["pair_examples"]
    )


def build_identity_head_control(
    calibration_embeddings: Mapping[str, Sequence[float]],
    output_dim: int,
    seed: int,
) -> dict[str, object]:
    standardizer = fit_standardizer(calibration_embeddings)
    matrix = np.asarray([standardizer.transform(calibration_embeddings)[key] for key in sorted(calibration_embeddings)])
    _, _, vh = np.linalg.svd(matrix, full_matrices=False)
    components = vh[:output_dim]
    return {"standardizer": standardizer, "components": components, "output_dim": output_dim, "seed": seed}


def transform_identity_embeddings(
    embeddings: Mapping[str, Sequence[float]],
    fitted_control: Mapping[str, object],
) -> dict[str, list[float]]:
    standardizer = fitted_control["standardizer"]
    transformed = standardizer.transform(embeddings)
    components = np.asarray(fitted_control["components"], dtype=np.float64)
    matrix = np.asarray([transformed[key] for key in sorted(transformed)], dtype=np.float64)
    projected = matrix @ components.T
    norms = np.linalg.norm(projected, axis=1, keepdims=True)
    projected = projected / np.maximum(norms, 1e-12)
    return {
        key: projected[index].astype(float).tolist()
        for index, key in enumerate(sorted(transformed))
    }


def build_same_count_no_cross_control(examples: Mapping[str, object]) -> dict[str, object]:
    pair_examples = list(examples.get("pair_examples", []))
    cross_examples = list(examples.get("cross_dialect_examples", []))
    return {
        "pair_examples": pair_examples + cross_examples[: len(cross_examples)],
        "lambda_cross": 0.0,
    }


def _dialect_pair_key(example: Mapping[str, object]) -> tuple[str, str]:
    labels = sorted(map(str, example.get("dialect_labels", [])))
    if len(labels) != 2:
        return ("unknown", "unknown")
    return (labels[0], labels[1])


def _entropy(values: Sequence[str]) -> float:
    if not values:
        return 0.0
    counts: dict[str, int] = {}
    for value in values:
        counts[value] = counts.get(value, 0) + 1
    total = float(len(values))
    return float(-sum((count / total) * np.log(count / total) for count in counts.values()))


def summarize_cross_pool_diversity(examples: Sequence[Mapping[str, object]]) -> dict[str, object]:
    speaker_values: list[str] = []
    dialect_pairs: list[str] = []
    targets: list[float] = []
    for example in examples:
        speakers = example.get("speaker_ids", [])
        if isinstance(speakers, Sequence) and not isinstance(speakers, (str, bytes)):
            speaker_values.extend(str(speaker) for speaker in speakers)
        elif "speaker_id" in example:
            speaker_values.append(str(example["speaker_id"]))
        pair = "|".join(_dialect_pair_key(example))
        dialect_pairs.append(pair)
        targets.append(float(example.get("target", 0.0)))
    return {
        "pair_count": len(examples),
        "unique_speaker_count": len(set(speaker_values)),
        "unique_dialect_pair_count": len(set(dialect_pairs)),
        "speaker_entropy": _entropy(speaker_values),
        "dialect_pair_entropy": _entropy(dialect_pairs),
        "target_mean": float(np.mean(targets)) if targets else 0.0,
        "target_std": float(np.std(targets)) if targets else 0.0,
    }


def validate_pair_count_matched_conditions(
    conditions: Mapping[str, Sequence[Mapping[str, object]]],
) -> None:
    counts = {name: len(examples) for name, examples in conditions.items() if examples}
    if len(set(counts.values())) > 1:
        raise ValueError(f"pair count mismatch across diversity conditions: {counts}")


def _records_by_dialect(records: Sequence[Mapping[str, object]]) -> dict[str, list[Mapping[str, object]]]:
    by_dialect: dict[str, list[Mapping[str, object]]] = {}
    for record in records:
        by_dialect.setdefault(str(record["dialect_label"]), []).append(record)
    return {dialect: sorted(items, key=lambda r: str(r["utterance_id"])) for dialect, items in by_dialect.items()}


def _make_cross_example(
    template: Mapping[str, object],
    left: Mapping[str, object],
    right: Mapping[str, object],
    *,
    prefix: str,
    index: int,
) -> dict[str, object]:
    return {
        **dict(template),
        "pair_id": f"{prefix}-{index}-{left['utterance_id']}-{right['utterance_id']}",
        "utterance_ids": [str(left["utterance_id"]), str(right["utterance_id"])],
        "speaker_id": f"{left['speaker_id']}|{right['speaker_id']}",
        "speaker_ids": [str(left["speaker_id"]), str(right["speaker_id"])],
        "dialect_labels": [str(left["dialect_label"]), str(right["dialect_label"])],
    }


def _sample_for_dialect_pairs(
    records: Sequence[Mapping[str, object]],
    templates: Sequence[Mapping[str, object]],
    *,
    seed: int,
    broaden_dialect_pairs: bool,
) -> list[dict[str, object]]:
    rng = random.Random(seed)
    by_dialect = _records_by_dialect(records)
    dialect_pairs = [_dialect_pair_key(template) for template in templates]
    if broaden_dialect_pairs:
        all_pairs = sorted(
            pair
            for pair in combinations(sorted(by_dialect), 2)
            if by_dialect.get(pair[0]) and by_dialect.get(pair[1])
        )
        if all_pairs:
            dialect_pairs = [all_pairs[index % len(all_pairs)] for index in range(len(templates))]
    examples: list[dict[str, object]] = []
    for index, template in enumerate(templates):
        left_dialect, right_dialect = dialect_pairs[index]
        left_candidates = list(by_dialect.get(left_dialect, []))
        right_candidates = list(by_dialect.get(right_dialect, []))
        rng.shuffle(left_candidates)
        rng.shuffle(right_candidates)
        selected = None
        for left in left_candidates:
            for right in right_candidates:
                if str(left["speaker_id"]) != str(right["speaker_id"]):
                    selected = (left, right)
                    break
            if selected is not None:
                break
        if selected is None and left_candidates and right_candidates:
            selected = (left_candidates[0], right_candidates[0])
        if selected is not None:
            examples.append(
                _make_cross_example(
                    template,
                    selected[0],
                    selected[1],
                    prefix="diversity",
                    index=index,
                )
            )
    return examples


def build_pair_diversity_sweep_conditions(
    records: Sequence[Mapping[str, object]],
    base_examples: Sequence[Mapping[str, object]],
    seed: int,
) -> dict[str, list[dict[str, object]]]:
    same_speaker = [dict(example) for example in base_examples]
    base_speakers = {
        str(speaker)
        for example in base_examples
        for speaker in example.get("speaker_ids", [])
    }
    coverage_records = [
        record
        for record in records
        if str(record["speaker_id"]) in base_speakers
    ]
    coverage = _shuffle_cross_examples(base_examples, coverage_records, seed)
    speaker_broadened = _sample_for_dialect_pairs(
        records,
        base_examples,
        seed=seed + 1,
        broaden_dialect_pairs=False,
    )
    speaker_and_dialect = _sample_for_dialect_pairs(
        records,
        base_examples,
        seed=seed + 2,
        broaden_dialect_pairs=True,
    )
    target_count = len(base_examples)
    return {
        "same_speaker": same_speaker[:target_count],
        "coverage_matched_shuffled": coverage[:target_count],
        "speaker_broadened_shuffled": speaker_broadened[:target_count],
        "speaker_and_dialect_broadened_shuffled": speaker_and_dialect[:target_count],
    }


def make_synthetic_fitted_projection_head(input_dim: int, output_dim: int, seed: int) -> dict[str, object]:
    _set_seed(seed)
    hidden_dim = max(2, input_dim)
    model = ProjectionHead(input_dim, hidden_dim, output_dim, 0.0)
    return {
        "model_state": {key: value.detach().cpu().clone() for key, value in model.state_dict().items()},
        "standardizer": FeatureStandardizer(np.zeros(input_dim), np.ones(input_dim)),
        "hidden_dim": hidden_dim,
        "output_dim": output_dim,
    }


def _raw_pair_rows(
    pairs: Sequence[Mapping[str, object]],
    embeddings: Mapping[str, Sequence[float]],
    reference: Mapping[str, Mapping[str, float]],
    record_index: Mapping[str, Mapping[str, object]],
) -> list[dict[str, object]]:
    _, maximum = _normalized_reference(reference)
    rows = []
    for pair in pairs:
        left, right = map(str, pair["source_utterance_ids"])
        left_record = record_index.get(left)
        right_record = record_index.get(right)
        if left_record is None or right_record is None:
            raise ValueError(f"missing record for pair {pair.get('pair_id')}")
        first = torch.tensor(embeddings[left], dtype=torch.float64)
        second = torch.tensor(embeddings[right], dtype=torch.float64)
        labels = [str(left_record["dialect_label"]), str(right_record["dialect_label"])]
        target = _reference_value(reference, labels[0], labels[1]) / maximum
        rows.append(
            {
                "pair_id": str(pair.get("pair_id", f"{pair.get('group', '?')}::{left}::{right}")),
                "group": pair.get("group"),
                "dialect_labels": labels,
                "speaker_ids": pair.get("speaker_ids"),
                "utterance_ids": pair.get("utterance_ids", pair.get("source_utterance_ids")),
                "matched_stratum": pair.get("matched_stratum"),
                "matched_fields": pair.get("matched_fields"),
                "raw_distance": float(_cosine_batch(first[None, :], second[None, :])[0]),
                "target": target,
            }
        )
    return rows


def _enrich_per_pair(rows: Sequence[Mapping[str, object]]) -> list[dict[str, object]]:
    enriched = []
    for row in rows:
        predicted = float(row["distance"])
        reference = float(row["target"])
        enriched.append(
            {
                **dict(row),
                "predicted_distance": predicted,
                "reference_distance": reference,
                "absolute_error": abs(predicted - reference),
            }
        )
    return enriched


def _summarize_rows(rows: Sequence[Mapping[str, object]], baseline_rows: Sequence[Mapping[str, object]], seed: int, replicates: int) -> dict[str, object]:
    method_values = np.asarray([float(row["distance"]) for row in rows], dtype=np.float64)
    baseline_values = np.asarray([float(row["distance"]) for row in baseline_rows], dtype=np.float64)
    targets = np.asarray([float(row["target"]) for row in rows], dtype=np.float64)
    baseline_targets = np.asarray([float(row["target"]) for row in baseline_rows], dtype=np.float64)
    if len(method_values) == 0:
        raise ValueError("cannot summarize empty rows")
    baseline_mae = float(np.mean(np.abs(baseline_values - baseline_targets)))
    method_mae = float(np.mean(np.abs(method_values - targets)))
    improvement = (baseline_mae - method_mae) / max(abs(baseline_mae), 1e-12)
    rng = np.random.default_rng(seed)
    boot = []
    for _ in range(replicates):
        indices = rng.integers(0, len(rows), len(rows))
        b = np.mean(np.abs(baseline_values[indices] - baseline_targets[indices]))
        m = np.mean(np.abs(method_values[indices] - targets[indices]))
        boot.append(float((b - m) / max(abs(b), 1e-12)))
    groups: dict[str, dict[str, float]] = {}
    for group in sorted({str(row.get("group")) for row in rows}):
        indices = [index for index, row in enumerate(rows) if str(row.get("group")) == group]
        groups[group] = {
            "count": len(indices),
            "baseline_mae": float(np.mean(np.abs(baseline_values[indices] - baseline_targets[indices]))),
            "method_mae": float(np.mean(np.abs(method_values[indices] - targets[indices]))),
        }
        groups[group]["improvement_ratio"] = (
            groups[group]["baseline_mae"] - groups[group]["method_mae"]
        ) / max(abs(groups[group]["baseline_mae"]), 1e-12)
    return {
        "baseline_mae": baseline_mae,
        "mae": method_mae,
        "improvement_ratio": improvement,
        "ci": {
            "lower": float(np.quantile(boot, 0.025)),
            "upper": float(np.quantile(boot, 0.975)),
            "confidence_level": 0.95,
        },
        "bootstrap_replicates": replicates,
        "groups": groups,
        "per_pair": _enrich_per_pair(rows),
    }


def _metrics_only(summary: Mapping[str, object], *, include_per_pair: bool = False) -> dict[str, object]:
    if include_per_pair:
        return dict(summary)
    return {key: value for key, value in summary.items() if key != "per_pair"}


def _unique_per_pair(rows: Sequence[Mapping[str, object]], label: str) -> dict[str, Mapping[str, object]]:
    indexed = {}
    for row in rows:
        pair_id = str(row.get("pair_id", ""))
        if not pair_id:
            raise ValueError(f"{label} row missing matched pair_id")
        if pair_id in indexed:
            raise ValueError(f"{label} has duplicate matched pair_id: {pair_id}")
        indexed[pair_id] = row
    return indexed


def paired_bootstrap_b4_minus_b3(
    b4_per_pair: Sequence[Mapping[str, object]],
    b3_per_pair: Sequence[Mapping[str, object]],
    *,
    seed: int,
    replicates: int,
) -> dict[str, object]:
    """Paired bootstrap of the B3 MAE minus B4 MAE on matched evaluation pairs."""
    b4 = _unique_per_pair(b4_per_pair, "B4")
    b3 = _unique_per_pair(b3_per_pair, "B3")
    if set(b4) != set(b3):
        raise ValueError("B4 and B3 must use matched pair_id values")
    pair_ids = sorted(b4)
    if not pair_ids:
        raise ValueError("at least one matched pair_id is required")
    deltas = np.asarray(
        [
            float(b3[pair_id]["absolute_error"]) - float(b4[pair_id]["absolute_error"])
            for pair_id in pair_ids
        ],
        dtype=np.float64,
    )
    observed = float(np.mean(deltas))
    clustered_rows = []
    for pair_id in pair_ids:
        candidate = b4[pair_id]
        if not candidate.get("matched_stratum") or not candidate.get("speaker_ids"):
            clustered_rows = []
            break
        clustered_rows.append(
            {
                "delta": float(b3[pair_id]["absolute_error"])
                - float(candidate["absolute_error"]),
                "matched_stratum": str(candidate["matched_stratum"]),
                "speaker_ids": candidate["speaker_ids"],
                "utterance_ids": candidate.get("utterance_ids"),
            }
        )
    if clustered_rows:
        from .paired_randomness import clustered_paired_bootstrap

        clustered = clustered_paired_bootstrap(
            clustered_rows, seed=seed, replicates=replicates
        )
        return {
            "pair_count": len(pair_ids),
            "mae_delta_b3_minus_b4": clustered["observed_delta"],
            "ci": clustered["ci"],
            "bootstrap_replicates": replicates,
            "passed": clustered["ci"]["lower"] > 0.0,
            "resampling_unit": clustered["resampling_unit"],
            "nested_utterance_sampling": clustered["nested_utterance_sampling"],
        }
    rng = np.random.default_rng(seed)
    boot = []
    for _ in range(replicates):
        indices = rng.integers(0, len(deltas), len(deltas))
        boot.append(float(np.mean(deltas[indices])))
    lower = float(np.quantile(boot, 0.025))
    upper = float(np.quantile(boot, 0.975))
    return {
        "pair_count": len(pair_ids),
        "mae_delta_b3_minus_b4": observed,
        "ci": {
            "lower": lower,
            "upper": upper,
            "confidence_level": 0.95,
        },
        "bootstrap_replicates": replicates,
        "passed": lower > 0.0,
        "resampling_unit": "pair_id_fallback_for_legacy_rows",
    }


def _combined_group_change(
    rows: Sequence[Mapping[str, object]],
    baseline_rows: Sequence[Mapping[str, object]],
    groups: set[str],
) -> dict[str, float]:
    selected = [
        index
        for index, row in enumerate(rows)
        if str(row.get("group")) in groups
    ]
    if not selected:
        return {"baseline_mae": float("nan"), "method_mae": float("nan"), "relative_change": float("nan")}
    baseline_mae = float(
        np.mean(
            [
                abs(float(baseline_rows[index]["distance"]) - float(baseline_rows[index]["target"]))
                for index in selected
            ]
        )
    )
    method_mae = float(
        np.mean(
            [
                abs(float(rows[index]["distance"]) - float(rows[index]["target"]))
                for index in selected
            ]
        )
    )
    return {
        "baseline_mae": baseline_mae,
        "method_mae": method_mae,
        "relative_change": (method_mae - baseline_mae) / max(abs(baseline_mae), 1e-12),
    }


def _improvement_value(container: Mapping[str, object], key: str) -> float:
    try:
        return float(container[key]["improvement_ratio"])  # type: ignore[index]
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"missing improvement_ratio for {key}") from exc


def _reference_gate_decisions(reference_report: Mapping[str, object]) -> dict[str, object]:
    """Evaluate the pre-committed projection-head gates for one model/reference row."""
    improvement = float(reference_report["improvement_ratio"])
    ci_lower = float(reference_report["ci"]["lower"])  # type: ignore[index]
    matched_change = float(reference_report["matched_speaker"]["relative_change"])  # type: ignore[index]
    comparisons = reference_report["comparisons"]  # type: ignore[index]
    ablations = reference_report["ablations"]  # type: ignore[index]

    permuted_key = (
        "permuted_pair_distance_target"
        if "permuted_pair_distance_target" in ablations
        else "permuted_dialect"
    )
    controls = {
        "lambda_cross_zero": _improvement_value(comparisons, "lambda_cross_zero"),  # type: ignore[arg-type]
        "same_count_no_cross": _improvement_value(comparisons, "same_count_no_cross"),  # type: ignore[arg-type]
        "shuffled_cross_dialect": _improvement_value(ablations, "shuffled_cross_dialect"),  # type: ignore[arg-type]
        "permuted_pair_distance_target": _improvement_value(ablations, permuted_key),  # type: ignore[arg-type]
        "identity_head": _improvement_value(ablations, "identity_head"),  # type: ignore[arg-type]
    }
    paired_contrast = reference_report.get("paired_contrast_b4_vs_b3", {})
    paired_passed = bool(paired_contrast.get("passed", False)) if isinstance(paired_contrast, Mapping) else False
    gate_2 = improvement >= 0.05
    gate_3 = ci_lower > 0.0
    gate_5 = all(improvement > value for value in controls.values()) and paired_passed
    gate_7 = matched_change <= 0.01
    return {
        "gate_2_efficacy": {
            "passed": gate_2,
            "threshold": 0.05,
            "improvement_ratio": improvement,
        },
        "gate_3_statistical_significance": {
            "passed": gate_3,
            "ci_lower": ci_lower,
            "threshold": 0.0,
        },
        "gate_5_mechanism_specificity": {
            "passed": gate_5,
            "method_improvement_ratio": improvement,
            "control_improvement_ratios": controls,
            "paired_b4_vs_b3": paired_contrast,
        },
        "gate_7_matched_speaker_protection": {
            "passed": gate_7,
            "relative_mae_change": matched_change,
            "threshold": 0.01,
            "groups": ["A", "C"],
        },
        "passed": gate_2 and gate_3 and gate_5 and gate_7,
    }


def _fit_control_variant(
    *,
    variant: str,
    calibration_embeddings: Mapping[str, Sequence[float]],
    calibration_records: Sequence[Mapping[str, object]],
    calibration_pairs: Sequence[Mapping[str, object]],
    reference: Mapping[str, Mapping[str, float]],
    config: Mapping[str, object],
    seed: int,
    selected: Mapping[str, float],
    pair_examples_override: Sequence[Mapping[str, object]] | None = None,
    cross_examples_override: Sequence[Mapping[str, object]] | None = None,
    dialect_label_map: Mapping[str, str] | None = None,
    lambda_cross_override: float | None = None,
    head_kind: str = "mlp",
    fixed_epochs: int | None = None,
) -> dict[str, object]:
    """Fit one calibration-only control with the selected full-head settings."""
    return train_projection_head(
        calibration_embeddings,
        calibration_records,
        calibration_pairs,
        reference,
        lambda_cross=(
            float(selected["lambda_cross"])
            if lambda_cross_override is None
            else float(lambda_cross_override)
        ),
        lambda_dialect=float(selected["lambda_dialect"]),
        learning_rate=float(selected["learning_rate"]),
        weight_decay=float(selected["weight_decay"]),
        config=config,
        seed=seed,
        validation_embeddings=calibration_embeddings,
        validation_records=calibration_records,
        validation_pairs=calibration_pairs,
        pair_examples_override=pair_examples_override,
        cross_examples_override=cross_examples_override,
        dialect_label_map=dialect_label_map,
        head_kind=head_kind,
        fixed_epochs=fixed_epochs,
    )


def evaluate_projection_head_gate(
    *,
    calibration_embeddings_by_model: Mapping[str, Mapping[str, Sequence[float]]],
    evaluation_embeddings_by_model: Mapping[str, Mapping[str, Sequence[float]]],
    records: Sequence[Mapping[str, object]],
    calibration_pairs: Sequence[Mapping[str, object]],
    evaluation_pairs: Sequence[Mapping[str, object]],
    references: Sequence[Mapping[str, object]],
    config: Mapping[str, object],
    seed: int,
    head_kind: str = "mlp",
) -> dict[str, object]:
    calibration_records = [r for r in records if str(r.get("split")) == "calibration"]
    evaluation_records = [r for r in records if str(r.get("split")) == "evaluation"]
    calibration_speakers = {str(r["speaker_id"]) for r in calibration_records}
    evaluation_speakers = {str(r["speaker_id"]) for r in evaluation_records}
    if calibration_speakers & evaluation_speakers:
        raise SupportError("calibration and evaluation speakers overlap")
    head_config = config.get("projection_head", config)
    replicates = int(head_config.get("bootstrap_replicates", 1000))
    registry = {
        "B0": {"name": "frozen_ssl_affine", "role": "zero_training_baseline"},
        "B1": {"name": "global_ecapa_regression", "role": "post_processing_boundary"},
        "B2": {"name": "rank1_low_rank_dialect_modulation", "role": "post_processing_boundary"},
        "B3": {"name": "projection_head_pair_only", "role": "standard_training_baseline"},
        "B4": {"name": "projection_head_pair_plus_same_speaker_cross", "role": "proposed_method"},
    }
    model_reports = []
    training_reports = []
    for model_name in sorted(calibration_embeddings_by_model):
        calibration_embeddings = calibration_embeddings_by_model[model_name]
        evaluation_embeddings = evaluation_embeddings_by_model[model_name]
        for reference_item in references:
            reference_name = str(reference_item["name"])
            reference = reference_item["matrix"]
            source_examples = build_training_examples(
                calibration_records, calibration_pairs, reference
            )
            source_audit = audit_projection_sources(
                calibration_pairs=calibration_pairs,
                cross_examples=source_examples["cross_dialect_examples"],
                evaluation_pairs=evaluation_pairs,
                fitted_sources={
                    "projection_calibration": calibration_pairs,
                    "calibration_auxiliary_cross_pair": source_examples["cross_dialect_examples"],
                },
            )
            record_index = {str(record["utterance_id"]): record for record in records}
            baseline_cal = _raw_pair_rows(calibration_pairs, calibration_embeddings, reference, record_index)
            baseline_eval = _raw_pair_rows(evaluation_pairs, evaluation_embeddings, reference, record_index)
            affine = _fit_affine(
                [row["raw_distance"] for row in baseline_cal],
                [row["target"] for row in baseline_cal],
            )
            baseline_rows = [
                {**row, "distance": float(_apply_affine([float(row["raw_distance"])], affine)[0])}
                for row in baseline_eval
            ]
            available_calibration_ids = set(calibration_embeddings)
            available_calibration_records = [
                record
                for record in calibration_records
                if str(record["utterance_id"]) in available_calibration_ids
            ]
            available_calibration_pairs = [
                pair
                for pair in calibration_pairs
                if set(map(str, pair["source_utterance_ids"])) <= available_calibration_ids
            ]
            fitted = fit_projection_head_cv(
                calibration_embeddings,
                calibration_records,
                calibration_pairs,
                reference,
                config,
                seed,
                progress_label=f"model={model_name} reference={reference_name}",
                head_kind=head_kind,
            )
            transformed = transform_embeddings(evaluation_embeddings, fitted)
            method_rows = score_pair_distances(
                evaluation_pairs,
                transformed,
                reference,
                fitted["affine_scale"],
                record_index,
            )
            report = _summarize_rows(method_rows, baseline_rows, seed, replicates)
            base_examples = build_training_examples(
                available_calibration_records,
                available_calibration_pairs,
                reference,
            )
            selected = fitted["selected"]
            available_calibration = {
                key: value
                for key, value in calibration_embeddings.items()
                if key in available_calibration_ids
            }
            pair_only = _fit_control_variant(
                variant="lambda_cross_zero",
                calibration_embeddings=available_calibration,
                calibration_records=available_calibration_records,
                calibration_pairs=available_calibration_pairs,
                reference=reference,
                config=config,
                seed=seed,
                selected=selected,
                cross_examples_override=base_examples["cross_dialect_examples"],
                lambda_cross_override=0.0,
                head_kind=head_kind,
                fixed_epochs=int(fitted["selected_epoch_from_training_folds"]),
            )
            same_count = build_same_count_no_cross_control(base_examples)
            same_count_fit = _fit_control_variant(
                variant="same_count_no_cross",
                calibration_embeddings=available_calibration,
                calibration_records=available_calibration_records,
                calibration_pairs=available_calibration_pairs,
                reference=reference,
                config=config,
                seed=seed,
                selected=selected,
                pair_examples_override=same_count["pair_examples"],
                cross_examples_override=[],
                lambda_cross_override=0.0,
                head_kind=head_kind,
                fixed_epochs=int(fitted["selected_epoch_from_training_folds"]),
            )
            shuffled_cross = _shuffle_cross_examples(
                base_examples["cross_dialect_examples"],
                available_calibration_records,
                seed,
            )
            shuffled_fit = _fit_control_variant(
                variant="shuffled_cross_dialect",
                calibration_embeddings=available_calibration,
                calibration_records=available_calibration_records,
                calibration_pairs=available_calibration_pairs,
                reference=reference,
                config=config,
                seed=seed,
                selected=selected,
                cross_examples_override=shuffled_cross,
                head_kind=head_kind,
                fixed_epochs=int(fitted["selected_epoch_from_training_folds"]),
            )
            permuted_targets = build_permuted_target_cross_examples(
                base_examples["cross_dialect_examples"], seed
            )
            permuted_fit = _fit_control_variant(
                variant="permuted_pair_distance_target",
                calibration_embeddings=available_calibration,
                calibration_records=available_calibration_records,
                calibration_pairs=available_calibration_pairs,
                reference=reference,
                config=config,
                seed=seed,
                selected=selected,
                cross_examples_override=permuted_targets,
                head_kind=head_kind,
                fixed_epochs=int(fitted["selected_epoch_from_training_folds"]),
            )

            def score_fitted(control_fit: Mapping[str, object]) -> dict[str, object]:
                control_transformed = transform_embeddings(evaluation_embeddings, control_fit)
                control_rows = score_pair_distances(
                    evaluation_pairs,
                    control_transformed,
                    reference,
                    control_fit["affine_scale"],
                    record_index,
                )
                return _summarize_rows(control_rows, baseline_rows, seed, replicates)

            pair_only_summary = score_fitted(pair_only)
            same_count_summary = score_fitted(same_count_fit)
            shuffled_summary = score_fitted(shuffled_fit)
            permuted_summary = score_fitted(permuted_fit)
            paired_contrast = paired_bootstrap_b4_minus_b3(
                report["per_pair"],
                pair_only_summary["per_pair"],
                seed=seed,
                replicates=replicates,
            )

            identity_fit = build_identity_head_control(
                available_calibration,
                int(head_config["output_dim"]),
                seed,
            )
            identity_cal = transform_identity_embeddings(available_calibration, identity_fit)
            identity_eval = transform_identity_embeddings(evaluation_embeddings, identity_fit)
            identity_cal_raw = _raw_pair_rows(
                calibration_pairs,
                identity_cal,
                reference,
                record_index,
            )
            identity_eval_raw = _raw_pair_rows(
                evaluation_pairs,
                identity_eval,
                reference,
                record_index,
            )
            identity_affine = _fit_affine(
                [row["raw_distance"] for row in identity_cal_raw],
                [row["target"] for row in identity_cal_raw],
            )
            identity_eval_rows = [
                {
                    **row,
                    "distance": float(
                        _apply_affine([float(row["raw_distance"])], identity_affine)[0]
                    ),
                }
                for row in identity_eval_raw
            ]
            identity_summary = _summarize_rows(
                identity_eval_rows,
                baseline_rows,
                seed,
                replicates,
            )
            matched_change = _combined_group_change(method_rows, baseline_rows, {"A", "C"})
            report.update(
                {
                    "model_name": model_name,
                    "reference": reference_name,
                    "selected": fitted["selected"],
                    "training": {
                        "backbone_frozen": True,
                        "evaluation_labels_used": False,
                        "calibration_source_audit": source_audit,
                        "same_speaker_cross_dialect_count": fitted["same_speaker_cross_dialect_count"],
                        "folds_used": fitted["folds_used"],
                        "head_kind": head_kind,
                    },
                    "group_a": report["groups"].get("A", {}),
                    "group_c": report["groups"].get("C", {}),
                    "matched_speaker": matched_change,
                    "paired_contrast_b4_vs_b3": paired_contrast,
                    "comparisons": {
                        "lambda_cross_zero": _metrics_only(pair_only_summary, include_per_pair=True),
                        "same_count_no_cross": _metrics_only(same_count_summary, include_per_pair=True),
                    },
                    "ablations": {
                        "shuffled_cross_dialect": {
                            "status": "evaluated",
                            **_metrics_only(shuffled_summary, include_per_pair=True),
                        },
                        "permuted_pair_distance_target": {
                            "status": "evaluated",
                            "target_permutation": "cross_loss_pair_distance_target",
                            **_metrics_only(permuted_summary, include_per_pair=True),
                        },
                        "permuted_dialect": {
                            "status": "evaluated",
                            "alias_of": "permuted_pair_distance_target",
                            "note": "Compatibility alias; the reported control permutes cross-loss pair-distance targets, not only classifier labels.",
                            **_metrics_only(permuted_summary, include_per_pair=True),
                        },
                        "same_count_no_cross": {
                            "status": "evaluated",
                            **_metrics_only(same_count_summary, include_per_pair=True),
                        },
                        "identity_head": {
                            "status": "evaluated",
                            **_metrics_only(identity_summary),
                        },
                    },
                }
            )
            report["gates"] = _reference_gate_decisions(report)
            print(
                "projection-head gate "
                f"model={model_name} reference={reference_name} "
                f"improvement={float(report['improvement_ratio']):.6f} "
                f"ci_lower={float(report['ci']['lower']):.6f} "
                f"gate_passed={bool(report['gates']['passed'])}",
                file=sys.stderr,
                flush=True,
            )
            model_reports.append(report)
            training_reports.append(
                {
                    "model_name": model_name,
                    "reference": reference_name,
                    "selected": fitted["selected"],
                    "folds_used": fitted["folds_used"],
                    "cross_fraction": min(
                        0.5,
                        fitted["same_speaker_cross_dialect_count"]
                        / max(len(calibration_pairs) + fitted["same_speaker_cross_dialect_count"], 1),
                    ),
                    "loss_history": fitted["loss_history"],
                }
            )
    passed = all(bool(report["gates"]["passed"]) for report in model_reports)
    return {
        "schema": "cross-dialect-projection-head-gate-v1",
        "seed": seed,
        "status": "passed" if passed else "failed",
        "baseline_registry": registry,
        "training": {
            "backbone_frozen": True,
            "evaluation_labels_used": False,
            "calibration_speaker_count": len(calibration_speakers),
            "evaluation_speaker_count": len(evaluation_speakers),
            "same_speaker_cross_dialect_count": max(
                [r["training"]["same_speaker_cross_dialect_count"] for r in model_reports],
                default=0,
            ),
        },
        "comparisons": {
            "lambda_cross_zero": {"status": "evaluated"},
            "same_count_no_cross": {"status": "evaluated"},
        },
        "ablations": {
            "shuffled_cross_dialect": {"status": "evaluated"},
            "permuted_dialect": {"status": "evaluated"},
            "same_count_no_cross": {"status": "evaluated"},
            "identity_head": {"status": "evaluated"},
        },
        "models": [
            {
                "model_name": model_name,
                "references": [report for report in model_reports if report["model_name"] == model_name],
            }
            for model_name in sorted(calibration_embeddings_by_model)
        ],
        "training_runs": training_reports,
    }


def _load_embedding_file(path: str) -> tuple[str, dict[str, list[float]]]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    name = str(payload["model_name"])
    embeddings = payload["embeddings"]
    return name, embeddings


def _load_records(path: str) -> list[dict[str, object]]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    return list(payload.get("records", payload))


def _load_pairs(path: str) -> list[dict[str, object]]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    return list(payload.get("pairs", payload))


def _parse_cli(argv: Sequence[str] | None = None) -> Any:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--calibration-embedding", action="append", required=True)
    parser.add_argument("--evaluation-embedding", action="append", required=True)
    parser.add_argument("--records", required=True)
    parser.add_argument("--calibration-pairs", required=True)
    parser.add_argument("--evaluation-pairs", required=True)
    parser.add_argument("--reference-matrix", action="append", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--training-output", required=True)
    parser.add_argument("--head-kind", choices=["mlp", "linear"], default="mlp")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    from src.config import load_config

    args = _parse_cli(argv)
    config = load_config(args.config)
    calibration_by_model = {}
    evaluation_by_model = {}
    for path in args.calibration_embedding:
        name, values = _load_embedding_file(path)
        calibration_by_model[name] = values
    for path in args.evaluation_embedding:
        name, values = _load_embedding_file(path)
        evaluation_by_model[name] = values
    if set(calibration_by_model) != set(evaluation_by_model):
        raise SupportError("calibration and evaluation extractor sets differ")
    if not calibration_by_model:
        raise SupportError("at least one extractor embedding is required")
    records = _load_records(args.records)
    calibration_pairs = _load_pairs(args.calibration_pairs)
    evaluation_pairs = _load_pairs(args.evaluation_pairs)
    references = []
    for path in args.reference_matrix:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        matrix = payload.get("matrix", payload)
        name = "taxonomy" if "taxonomy" in Path(path).name else "sincomp"
        references.append({"name": name, "matrix": matrix})
    report = evaluate_projection_head_gate(
        calibration_embeddings_by_model=calibration_by_model,
        evaluation_embeddings_by_model=evaluation_by_model,
        records=records,
        calibration_pairs=calibration_pairs,
        evaluation_pairs=evaluation_pairs,
        references=references,
        config=config,
        seed=int(config["protocol"]["seed"]),
        head_kind=args.head_kind,
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")
    training = {
        "schema": "cross-dialect-projection-head-training-v1",
        "seed": int(config["protocol"]["seed"]),
        "runs": report["training_runs"],
    }
    training_output = Path(args.training_output)
    training_output.parent.mkdir(parents=True, exist_ok=True)
    training_output.write_text(json.dumps(training, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")
    return 0 if report["status"] == "passed" else 2


if __name__ == "__main__":
    raise SystemExit(main())
