"""Construct and compare representative-robust Sinitic Data4 references."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import pickle
import zipfile
from itertools import combinations
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import yaml
from scipy.stats import pearsonr, spearmanr

from .reference_matrices import (
    DIALECT_SPECS,
    KESPEECH_DIALECTS,
    _nested_matrix,
    select_sinitic_representatives,
    subgroup_aggregate_matrix,
    subgroup_medoid_indices,
    validate_reference_matrix,
)


EXPECTED_ARCHIVE_SHA256 = (
    "020f399824a9e4073f4092078099b5c9b6a995f1ef7d6eae6887f6b1250923c5"
)
PINNED_COMMIT_ARCHIVE_SHA256 = (
    "621c8bce1fc49e5bc8e05103c810d675447c6d43d572bac3334bdff385a74692"
)
ALLOWED_ARCHIVE_SHA256 = {
    EXPECTED_ARCHIVE_SHA256,
    PINNED_COMMIT_ARCHIVE_SHA256,
}


def validate_source_archive_hash(archive_hash: str) -> str:
    """Accept only either byte-verified container for the pinned source commit."""
    if archive_hash not in ALLOWED_ARCHIVE_SHA256:
        raise ValueError("Sinitic source archive hash does not match the locked protocol")
    return archive_hash


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_archive(path: Path) -> tuple[dict[str, Any], np.ndarray]:
    with zipfile.ZipFile(path) as archive:
        prefix = next(
            name[: -len("Data4/processed_info.pkl")]
            for name in archive.namelist()
            if name.endswith("Data4/processed_info.pkl")
        )
        info = pickle.loads(archive.read(f"{prefix}Data4/processed_info.pkl"))
        with np.load(
            io.BytesIO(archive.read(f"{prefix}Data4/distance_matrices.npz"))
        ) as matrices:
            overall = np.asarray(matrices["overall"], dtype=float)
    return info, overall


def normalize_reference_values(values: np.ndarray) -> tuple[np.ndarray, float]:
    checked = np.asarray(values, dtype=float).copy()
    checked = (checked + checked.T) / 2.0
    np.fill_diagonal(checked, 0.0)
    maximum = float(np.max(checked))
    if not np.isfinite(maximum) or maximum <= 0.0:
        raise ValueError("reference maximum must be positive and finite")
    checked /= maximum
    np.fill_diagonal(checked, 0.0)
    return checked, maximum


def _relation_values(values: np.ndarray) -> np.ndarray:
    return np.asarray(
        [values[i, j] for i in range(values.shape[0]) for j in range(i + 1, values.shape[1])],
        dtype=float,
    )


def _rank_reversals(left: np.ndarray, right: np.ndarray) -> tuple[int, int]:
    reversals = 0
    comparable = 0
    for first, second in combinations(range(len(left)), 2):
        left_delta = float(left[first] - left[second])
        right_delta = float(right[first] - right[second])
        if left_delta == 0.0 or right_delta == 0.0:
            continue
        comparable += 1
        if left_delta * right_delta < 0.0:
            reversals += 1
    return reversals, comparable


def compare_reference_matrices(
    matrices: Mapping[str, np.ndarray], labels: Sequence[str]
) -> dict[str, Any]:
    relation_count = len(labels) * (len(labels) - 1) // 2
    pairs = []
    for left_name, right_name in combinations(sorted(matrices), 2):
        left = _relation_values(np.asarray(matrices[left_name], dtype=float))
        right = _relation_values(np.asarray(matrices[right_name], dtype=float))
        reversals, comparable = _rank_reversals(left, right)
        pairs.append(
            {
                "left": left_name,
                "right": right_name,
                "pearson": round(float(pearsonr(left, right).statistic), 12),
                "spearman": round(float(spearmanr(left, right).statistic), 12),
                "rank_order_reversals": reversals,
                "comparable_relation_orderings": comparable,
            }
        )
    return {
        "relation_count": relation_count,
        "pairs": pairs,
    }


def _groups(info: Mapping[str, Any]) -> dict[str, list[int]]:
    return {
        label: [
            index
            for index, area in enumerate(info["areas"])
            if area == DIALECT_SPECS[label]["area"]
        ]
        for label in KESPEECH_DIALECTS
    }


def _selected_values(
    overall: np.ndarray, indices: Mapping[str, int]
) -> np.ndarray:
    selected = np.asarray([indices[label] for label in KESPEECH_DIALECTS], dtype=int)
    values = np.asarray(overall[np.ix_(selected, selected)], dtype=float)
    values = (values + values.T) / 2.0
    np.fill_diagonal(values, 0.0)
    return values


def _aggregate_values(
    overall: np.ndarray, groups: Mapping[str, Sequence[int]]
) -> np.ndarray:
    nested = subgroup_aggregate_matrix(overall, groups)
    return np.asarray(
        [[nested[left][right] for right in KESPEECH_DIALECTS] for left in KESPEECH_DIALECTS],
        dtype=float,
    )


def _payload(
    name: str,
    values: np.ndarray,
    raw_maximum: float,
    construction: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "schema": "reference-matrix-v1",
        "name": name,
        "labels": list(KESPEECH_DIALECTS),
        "distance_scale": "Sinitic_Data Data4 overall distance normalized by the construction-specific maximum",
        "source": "YiYang-github/Sinitic_Data Data4 overall_distance",
        "coverage": {
            "status": "complete",
            "labels": list(KESPEECH_DIALECTS),
            "missing_labels": [],
        },
        "normalization": {"raw_maximum": raw_maximum, "normalized_maximum": 1.0},
        "construction": dict(construction),
        "notes": [
            "This is an externally derived reference construction, not perceptual ground truth.",
            "Mandarin retains the Beijing-standard proxy, so Beijing-Mandarin distance is zero.",
        ],
        "matrix": _nested_matrix(KESPEECH_DIALECTS, values),
    }


def build_artifacts(
    source_archive: Path,
    current_provenance: Path,
    output_dir: Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    archive_hash = _sha256(source_archive)
    validate_source_archive_hash(archive_hash)
    info, overall = _load_archive(source_archive)
    groups = _groups(info)
    source_counts = {label: len(indices) for label, indices in groups.items()}

    city_records = select_sinitic_representatives(info)
    city_indices = {label: int(record["index"]) for label, record in city_records.items()}
    medoid_indices = subgroup_medoid_indices(overall, groups)
    medoid_indices["Mandarin"] = medoid_indices["Beijing"]
    raw = {
        "city_nearest": _selected_values(overall, city_indices),
        "subgroup_medoid": _selected_values(overall, medoid_indices),
        "subgroup_aggregate": _aggregate_values(overall, groups),
    }

    matrices: dict[str, np.ndarray] = {}
    payloads: dict[str, dict[str, Any]] = {}
    paths: dict[str, Path] = {}
    for name, raw_values in raw.items():
        normalized, maximum = normalize_reference_values(raw_values)
        matrices[name] = normalized
        if name == "city_nearest":
            construction = {
                "kind": name,
                "representative_indices": city_indices,
                "representatives": city_records,
            }
        elif name == "subgroup_medoid":
            construction = {
                "kind": name,
                "representative_indices": medoid_indices,
                "source_counts": source_counts,
            }
        else:
            construction = {
                "kind": name,
                "operation": "mean_all_cross_subgroup_source_distances",
                "source_counts": source_counts,
            }
        payloads[name] = _payload(
            f"sinitic_data4_{name}", normalized, maximum, construction
        )
        paths[name] = output_dir / f"sinitic_data4_{name}.json"
        paths[name].parent.mkdir(parents=True, exist_ok=True)
        paths[name].write_text(
            json.dumps(payloads[name], ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )

    provenance = yaml.safe_load(current_provenance.read_text(encoding="utf-8"))
    prior_indices = {
        label: int(record["index"])
        for label, record in provenance["sincomp"]["representatives"].items()
    }
    validations = {
        name: validate_reference_matrix(payload) for name, payload in payloads.items()
    }
    report = {
        "schema": "reference-representative-sensitivity-v1",
        "status": "construction_valid_awaiting_projection_sweep",
        "source_archive": str(source_archive),
        "source_archive_sha256": archive_hash,
        "source_counts": source_counts,
        "city_nearest_indices": city_indices,
        "subgroup_medoid_indices": medoid_indices,
        "city_nearest_matches_locked_provenance": city_indices == prior_indices,
        "normalization_maxima": {
            name: payload["normalization"]["raw_maximum"]
            for name, payload in payloads.items()
        },
        "matrix_comparison": compare_reference_matrices(matrices, KESPEECH_DIALECTS),
        "validations": validations,
        "matrix_paths": {name: str(path) for name, path in paths.items()},
        "matrix_sha256": {name: _sha256(path) for name, path in paths.items()},
        "projection_sweep": None,
    }
    checks = {
        "source_archive_hash": archive_hash in ALLOWED_ARCHIVE_SHA256,
        "city_nearest_matches_locked_provenance": city_indices == prior_indices,
        "all_matrices_validate": all(
            record["status"] == "passed" for record in validations.values()
        ),
        "all_subgroups_have_source_rows": all(count > 0 for count in source_counts.values()),
        "mandarin_proxy_preserved": all(
            payload["matrix"]["Beijing"]["Mandarin"] == 0.0
            for payload in payloads.values()
        ),
    }
    gate = {
        "schema": "reference-representative-gate-v1",
        "status": "passed" if all(checks.values()) else "failed",
        "phase": "construction_only",
        "checks": checks,
        "projection_stability_status": "awaiting_reference_variant_sweep",
        "failure_action": "report_construction_specific_result_only",
    }
    return report, gate


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-archive", type=Path, required=True)
    parser.add_argument("--current-provenance", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--gate", type=Path, required=True)
    args = parser.parse_args()
    report, gate = build_artifacts(
        args.source_archive, args.current_provenance, args.output_dir
    )
    for path, payload in ((args.report, report), (args.gate, gate)):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
    return 0 if gate["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
