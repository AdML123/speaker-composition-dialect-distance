"""CCA overlap diagnostic for frozen speaker and dialect directions."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import mean
from typing import Any, Mapping, Sequence

import numpy as np
from sklearn.cross_decomposition import CCA
from sklearn.linear_model import SGDClassifier
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler


def _matrix(embeddings: Mapping[str, Sequence[float]], ids: Sequence[str]) -> np.ndarray:
    values = np.asarray([embeddings[utterance_id] for utterance_id in ids], dtype=np.float64)
    if values.ndim != 2 or values.shape[0] != len(ids) or not np.isfinite(values).all():
        raise ValueError("embeddings must be a finite 2D matrix")
    return values


def _label_array(metadata: Mapping[str, Mapping[str, Any]], ids: Sequence[str], key: str) -> np.ndarray:
    values = [str(metadata[utterance_id][key]) for utterance_id in ids]
    if not values:
        raise ValueError("no labels available")
    return np.asarray(values, dtype=object)


def _linear_probe(seed: int):
    return make_pipeline(
        StandardScaler(),
        SGDClassifier(loss="log_loss", max_iter=2000, tol=1e-4, random_state=seed),
    )


def _supported_label_mask(labels: np.ndarray, *, min_count: int = 2) -> np.ndarray:
    classes, counts = np.unique(labels, return_counts=True)
    supported = {label for label, count in zip(classes, counts) if count >= min_count}
    return np.asarray([label in supported for label in labels], dtype=bool)


def _probe_support(y: np.ndarray) -> dict[str, int]:
    labels = np.asarray(y)
    classes, counts = np.unique(labels, return_counts=True)
    supported = counts >= 2
    return {
        "sample_count": int(len(labels)),
        "class_count": int(len(classes)),
        "cv_sample_count": int(sum(count for count in counts if count >= 2)),
        "cv_class_count": int(np.sum(supported)),
        "singleton_class_count": int(np.sum(counts == 1)),
    }


def _probe_accuracy(x: np.ndarray, y: np.ndarray, *, seed: int) -> float:
    features = np.asarray(x, dtype=np.float64)
    labels = np.asarray(y)
    mask = _supported_label_mask(labels)
    features = features[mask]
    labels = labels[mask]
    classes, counts = np.unique(labels, return_counts=True)
    if len(features) < 2 or len(classes) < 2:
        return 0.0
    if len(features) < max(20, len(classes) * 3):
        classifier = _linear_probe(seed)
        classifier.fit(features, labels)
        return float(classifier.score(features, labels))
    folds = min(5, int(counts.min()), len(features))
    if folds < 2:
        return 0.0
    classifier = _linear_probe(seed)
    splitter = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
    scores = cross_val_score(classifier, features, labels, cv=splitter)
    return float(scores.mean())


def _orthonormal_basis(weights: np.ndarray) -> np.ndarray:
    if weights.ndim != 2 or weights.size == 0:
        return np.zeros((weights.shape[1] if weights.ndim == 2 else 0, 0), dtype=np.float64)
    _, singular_values, vt = np.linalg.svd(weights, full_matrices=False)
    rank = int(np.sum(singular_values > 1e-10))
    return vt[:rank].T


def _principal_correlations(left_weights: np.ndarray, right_weights: np.ndarray) -> list[float]:
    left_basis = _orthonormal_basis(left_weights)
    right_basis = _orthonormal_basis(right_weights)
    if left_basis.size == 0 or right_basis.size == 0:
        return []
    singular_values = np.linalg.svd(left_basis.T @ right_basis, compute_uv=False)
    return [float(value) for value in singular_values[: min(left_basis.shape[1], right_basis.shape[1])]]


def _cca_summary(x: np.ndarray, y: np.ndarray, *, n_components: int) -> list[float]:
    if n_components <= 0:
        return []
    cca = CCA(n_components=n_components, max_iter=1000)
    x_scores, y_scores = cca.fit_transform(x, y)
    correlations: list[float] = []
    for index in range(min(x_scores.shape[1], y_scores.shape[1])):
        left = x_scores[:, index]
        right = y_scores[:, index]
        if np.std(left) <= 0 or np.std(right) <= 0:
            correlations.append(0.0)
        else:
            correlations.append(float(np.corrcoef(left, right)[0, 1]))
    return correlations


def run_speaker_offset_cca(
    *,
    embeddings: Mapping[str, Mapping[str, Sequence[float]]],
    metadata: Mapping[str, Mapping[str, Any]],
    seed: int = 20260829,
) -> dict[str, Any]:
    models: list[dict[str, Any]] = []
    for model_name, model_embeddings in sorted(embeddings.items()):
        ids = sorted(utterance_id for utterance_id in model_embeddings if utterance_id in metadata)
        if not ids:
            continue
        x = _matrix(model_embeddings, ids)
        speaker_labels = _label_array(metadata, ids, "speaker_id")
        dialect_labels = _label_array(metadata, ids, "dialect_label")
        speaker_accuracy = _probe_accuracy(x, speaker_labels, seed=seed)
        dialect_accuracy = _probe_accuracy(x, dialect_labels, seed=seed)

        speaker_encoder = LabelEncoder().fit(speaker_labels)
        dialect_encoder = LabelEncoder().fit(dialect_labels)
        speaker_classifier = _linear_probe(seed)
        dialect_classifier = _linear_probe(seed)
        speaker_classifier.fit(x, speaker_encoder.transform(speaker_labels))
        dialect_classifier.fit(x, dialect_encoder.transform(dialect_labels))
        correlations = _principal_correlations(
            speaker_classifier.named_steps["sgdclassifier"].coef_,
            dialect_classifier.named_steps["sgdclassifier"].coef_,
        )
        speaker_scores = speaker_classifier.decision_function(x)
        dialect_scores = dialect_classifier.decision_function(x)
        if speaker_scores.ndim == 1:
            speaker_scores = speaker_scores[:, np.newaxis]
        if dialect_scores.ndim == 1:
            dialect_scores = dialect_scores[:, np.newaxis]
        try:
            cca_correlations = _cca_summary(
                speaker_scores,
                dialect_scores,
                n_components=max(1, min(speaker_scores.shape[1], dialect_scores.shape[1], 5)),
            )
        except Exception:
            cca_correlations = []
        if not cca_correlations:
            cca_correlations = correlations

        models.append(
            {
                "model_name": model_name,
                "sample_count": len(ids),
                "speaker_probe_accuracy": speaker_accuracy,
                "speaker_probe_support": _probe_support(speaker_labels),
                "dialect_probe_accuracy": dialect_accuracy,
                "dialect_probe_support": _probe_support(dialect_labels),
                "subspace_principal_correlations": correlations,
                "canonical_correlations": cca_correlations,
                "canonical_correlation_mean": float(mean(cca_correlations)) if cca_correlations else 0.0,
            }
        )
    return {"schema": "speaker-offset-cca-v1", "seed": seed, "models": models}


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--embedding", nargs="+", action="append", required=True)
    parser.add_argument("--records", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    args.embedding = [path for group in args.embedding for path in group]
    return args


def main() -> int:
    args = _parse_args()
    payload = json.loads(Path(args.records).read_text(encoding="utf-8"))["records"]
    metadata = {
        str(record["utterance_id"]): {
            "speaker_id": str(record["speaker_id"]),
            "dialect_label": str(record["dialect_label"]),
            "recording_condition": str(record["recording_condition"]),
        }
        for record in payload
    }
    embeddings = {}
    for path in args.embedding:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        embeddings[payload["model_name"]] = payload["embeddings"]
    report = run_speaker_offset_cca(embeddings=embeddings, metadata=metadata)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
