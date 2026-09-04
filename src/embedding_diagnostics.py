"""Protocol and scale diagnostics for frozen embedding caches."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np


def summarize_embedding_cache(
    embeddings: Mapping[str, Sequence[float]],
    *,
    model_name: str,
    split: str,
    layer: str = "last_hidden_state",
    pooling: str = "mean",
    sample_rate_hz: int = 16000,
    channels: int = 1,
    explicit_l2_normalization: bool = False,
) -> dict[str, Any]:
    if not embeddings:
        raise ValueError("embedding cache is empty")
    matrix = np.asarray([embeddings[key] for key in sorted(embeddings)], dtype=np.float64)
    if matrix.ndim != 2 or not np.isfinite(matrix).all():
        raise ValueError("embedding cache must be finite and rectangular")
    norms = np.linalg.norm(matrix, axis=1)
    variances = np.var(matrix, axis=0)
    if np.any(norms == 0):
        raise ValueError("zero-norm embedding is not valid for cosine scoring")
    return {
        "schema": "embedding-diagnostics-v1",
        "model_name": model_name,
        "split": split,
        "layer": layer,
        "pooling": pooling,
        "sample_rate_hz": sample_rate_hz,
        "channels": channels,
        "explicit_l2_normalization": explicit_l2_normalization,
        "embedding_count": int(matrix.shape[0]),
        "dimension": int(matrix.shape[1]),
        "norm_summary": {
            "minimum": float(np.min(norms)),
            "median": float(np.median(norms)),
            "maximum": float(np.max(norms)),
        },
        "per_dimension_variance": {
            "minimum": float(np.min(variances)),
            "median": float(np.median(variances)),
            "maximum": float(np.max(variances)),
            "mean": float(np.mean(variances)),
        },
        "finite": True,
    }


def compare_preprocessing_contract(reports: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    if not reports:
        raise ValueError("at least one report is required")
    fields = ("split", "layer", "pooling", "sample_rate_hz", "channels", "explicit_l2_normalization")
    mismatches = {field: sorted({report.get(field) for report in reports}) for field in fields if len({report.get(field) for report in reports}) > 1}
    dimensions = sorted({report.get("dimension") for report in reports})
    return {
        "schema": "embedding-preprocessing-contract-v1",
        "status": "passed" if not mismatches and len(dimensions) == 1 else "failed",
        "mismatches": mismatches,
        "dimensions": dimensions,
        "required_same_fields": list(fields),
    }


def load_and_diagnose(paths: Sequence[str | Path], *, split: str) -> dict[str, Any]:
    reports = []
    for path in paths:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        reports.append(
            summarize_embedding_cache(
                payload["embeddings"],
                model_name=str(payload["model_name"]),
                split=split,
                layer="last_hidden_state",
                pooling="mean",
                explicit_l2_normalization=False,
            )
        )
    return {"schema": "embedding-diagnostics-batch-v1", "reports": reports, "contract": compare_preprocessing_contract(reports)}
