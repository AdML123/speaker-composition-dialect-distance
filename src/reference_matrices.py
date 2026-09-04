"""Build auditable dialect-distance reference matrices for KeSpeech labels."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import pickle
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import yaml


KESPEECH_DIALECTS = [
    "Beijing",
    "Ji-Lu",
    "Jiang-Huai",
    "Jiao-Liao",
    "Lan-Yin",
    "Mandarin",
    "Northeastern",
    "Southwestern",
    "Zhongyuan",
]

SINITIC_DATA_COMMIT = "820b9d15a74cbee82109f0bf54cf791fe16596ef"
SINITIC_DATA_SOURCE_URL = (
    "https://github.com/YiYang-github/Sinitic_Data/tree/"
    f"{SINITIC_DATA_COMMIT}"
)
SINITIC_DATA_ANALYSIS_ARCHIVE_SHA256 = (
    "020f399824a9e4073f4092078099b5c9b6a995f1ef7d6eae6887f6b1250923c5"
)
SINITIC_DATA_COMMIT_ARCHIVE_SHA256 = (
    "621c8bce1fc49e5bc8e05103c810d675447c6d43d572bac3334bdff385a74692"
)
SINITIC_DERIVED_MATRIX_SHA256 = {
    "city_nearest": "6aa0375f7c028a7b89471225f5327c66b919b79bc1c850f0e27128835465bb2e",
    "overall": "463f787aa316b3750f806d16157d822e55080fbdfb8f8ca8d393efaa07c5ddde",
    "subgroup_aggregate": "24e4cf95cac3a3820fe8c876476f22e9911cfde4834851190a5d30b3c45972e0",
    "subgroup_medoid": "4f8bdc1f23257e889c56d4f52a0b1ff7b049265417cedaab5c9e77c0a18ed746",
}

DIALECT_SPECS: dict[str, dict[str, Any]] = {
    "Beijing": {"area": "北京官话", "target_lon_lat": (116.4074, 39.9042), "target_name": "Beijing"},
    "Ji-Lu": {"area": "冀鲁官话", "target_lon_lat": (117.1201, 36.6512), "target_name": "Jinan"},
    "Jiang-Huai": {"area": "江淮官话", "target_lon_lat": (117.2272, 31.8206), "target_name": "Hefei"},
    "Jiao-Liao": {"area": "胶辽官话", "target_lon_lat": (120.3826, 36.0671), "target_name": "Qingdao"},
    "Lan-Yin": {"area": "兰银官话", "target_lon_lat": (103.8343, 36.0611), "target_name": "Lanzhou"},
    "Mandarin": {
        "area": "北京官话",
        "target_lon_lat": (116.4074, 39.9042),
        "target_name": "Beijing standard Mandarin proxy",
    },
    "Northeastern": {"area": "东北官话", "target_lon_lat": (123.4315, 41.8057), "target_name": "Shenyang"},
    "Southwestern": {"area": "西南官话", "target_lon_lat": (104.0668, 30.5728), "target_name": "Chengdu"},
    "Zhongyuan": {"area": "中原官话", "target_lon_lat": (108.9398, 34.3416), "target_name": "Xi'an"},
}


def validate_reference_matrix(payload: Mapping[str, Any], *, required_labels: Sequence[str] = KESPEECH_DIALECTS) -> dict[str, Any]:
    labels = list(map(str, payload.get("labels", [])))
    matrix = payload.get("matrix", {})
    missing = sorted(set(required_labels) - set(labels))
    shape_ok = len(labels) == len(required_labels) and all(set(map(str, matrix.get(label, {}))) == set(labels) for label in labels)
    finite = True
    symmetric = True
    diagonal_zero = True
    maximum = 0.0
    for left in labels:
        for right in labels:
            try:
                value = float(matrix[left][right])
                reverse = float(matrix[right][left])
            except (KeyError, TypeError, ValueError):
                finite = symmetric = diagonal_zero = False
                continue
            finite = finite and math.isfinite(value)
            symmetric = symmetric and math.isclose(value, reverse, rel_tol=0.0, abs_tol=1e-10)
            if left == right:
                diagonal_zero = diagonal_zero and math.isclose(value, 0.0, abs_tol=1e-12)
            maximum = max(maximum, value)
    passed = shape_ok and not missing and finite and symmetric and diagonal_zero and maximum > 0
    return {
        "schema": "reference-matrix-validation-v1",
        "name": str(payload.get("name", "reference")),
        "status": "passed" if passed else "failed",
        "label_count": len(labels),
        "shape": [len(labels), len(labels)] if shape_ok else None,
        "missing_labels": missing,
        "finite": finite,
        "symmetric": symmetric,
        "diagonal_zero": diagonal_zero,
        "maximum": maximum,
    }


def _haversine_km(origin: Sequence[float], destination: Sequence[float]) -> float:
    lon1, lat1 = origin
    lon2, lat2 = destination
    radius_km = 6371.0088
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    delta_phi = math.radians(lat2 - lat1)
    delta_lambda = math.radians(lon2 - lon1)
    value = math.sin(delta_phi / 2.0) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(delta_lambda / 2.0) ** 2
    return 2.0 * radius_km * math.atan2(math.sqrt(value), math.sqrt(1.0 - value))


def _require_info(info: Mapping[str, Any]) -> dict[str, Sequence[Any]]:
    required = {"areas", "slice", "slices", "coords"}
    missing = required.difference(info)
    if missing:
        raise ValueError(f"Sinitic_Data info missing field: {sorted(missing)[0]}")
    checked = {field: info[field] for field in required}
    lengths = {len(value) for value in checked.values()}
    if len(lengths) != 1:
        raise ValueError("Sinitic_Data info fields must have equal length")
    return checked


def select_sinitic_representatives(
    info: Mapping[str, Any],
    *,
    dialect_specs: Mapping[str, Mapping[str, Any]] = DIALECT_SPECS,
) -> dict[str, dict[str, Any]]:
    """Select the nearest Data4 coordinate inside each requested dialect area."""
    checked = _require_info(info)
    representatives: dict[str, dict[str, Any]] = {}
    for label, spec in sorted(dialect_specs.items()):
        area = str(spec["area"])
        target = tuple(float(value) for value in spec["target_lon_lat"])
        candidates: list[tuple[float, int]] = []
        for index, candidate_area in enumerate(checked["areas"]):
            if candidate_area != area:
                continue
            candidate_coord = tuple(float(value) for value in checked["coords"][index])
            candidates.append((_haversine_km(target, candidate_coord), index))
        if not candidates:
            raise ValueError(f"Sinitic_Data has no representative candidate for {label} / {area}")
        distance_km, index = min(candidates, key=lambda item: (item[0], item[1]))
        representatives[label] = {
            "index": index,
            "area": area,
            "slice": str(checked["slice"][index]),
            "slices": str(checked["slices"][index]),
            "coordinate_lon_lat": [float(value) for value in checked["coords"][index]],
            "target_name": str(spec["target_name"]),
            "target_lon_lat": [float(value) for value in target],
            "target_distance_km": round(float(distance_km), 6),
        }
    return representatives


def _nested_matrix(labels: Sequence[str], values: np.ndarray) -> dict[str, dict[str, float]]:
    if values.shape != (len(labels), len(labels)):
        raise ValueError("matrix shape does not match labels")
    matrix: dict[str, dict[str, float]] = {}
    for i, row_label in enumerate(labels):
        matrix[row_label] = {}
        for j, col_label in enumerate(labels):
            value = 0.0 if i == j else float(values[i, j])
            if not math.isfinite(value):
                raise ValueError("reference matrix contains non-finite value")
            matrix[row_label][col_label] = value
    return matrix


def subgroup_medoid_indices(
    distance: np.ndarray, groups: Mapping[str, Sequence[int]]
) -> dict[str, int]:
    """Select the lowest-index minimum-mean-distance member of each group."""
    values = np.asarray(distance, dtype=float)
    if values.ndim != 2 or values.shape[0] != values.shape[1]:
        raise ValueError("distance must be square")
    result: dict[str, int] = {}
    for label, raw_indices in groups.items():
        indices = sorted({int(index) for index in raw_indices})
        if not indices:
            raise ValueError(f"group {label} has no members")
        if indices[0] < 0 or indices[-1] >= values.shape[0]:
            raise ValueError(f"group {label} contains an out-of-range index")
        within = values[np.ix_(indices, indices)]
        means = np.mean(within, axis=1)
        minimum = float(np.min(means))
        tied = [
            index
            for index, mean_value in zip(indices, means)
            if math.isclose(float(mean_value), minimum, rel_tol=0.0, abs_tol=1e-12)
        ]
        result[str(label)] = min(tied)
    return result


def subgroup_aggregate_matrix(
    distance: np.ndarray, groups: Mapping[str, Sequence[int]]
) -> dict[str, dict[str, float]]:
    """Average all cross-group source distances for every named relation."""
    values = np.asarray(distance, dtype=float)
    if values.ndim != 2 or values.shape[0] != values.shape[1]:
        raise ValueError("distance must be square")
    checked = {
        str(label): sorted({int(index) for index in indices})
        for label, indices in groups.items()
    }
    if any(not indices for indices in checked.values()):
        raise ValueError("every subgroup must contain at least one source index")
    labels = list(checked)
    result = {left: {} for left in labels}
    for i, left in enumerate(labels):
        for j, right in enumerate(labels):
            if i == j or checked[left] == checked[right]:
                result[left][right] = 0.0
                continue
            block = values[np.ix_(checked[left], checked[right])]
            result[left][right] = float(np.mean(block))
    for i, left in enumerate(labels):
        for right in labels[i + 1 :]:
            mean_value = (result[left][right] + result[right][left]) / 2.0
            result[left][right] = result[right][left] = float(mean_value)
    return result


def build_taxonomy_matrix(labels: Sequence[str] = KESPEECH_DIALECTS) -> dict[str, Any]:
    """Build a complete low-resolution Mandarin-subgroup taxonomy matrix."""
    specs = {label: DIALECT_SPECS[label] for label in labels}
    values = np.zeros((len(labels), len(labels)), dtype=float)
    for i, label_a in enumerate(labels):
        for j, label_b in enumerate(labels):
            if i == j:
                continue
            values[i, j] = 0.0 if specs[label_a]["area"] == specs[label_b]["area"] else 1.0
    return {
        "schema": "reference-matrix-v1",
        "name": "taxonomy_mandarin_subgroup",
        "labels": list(labels),
        "distance_scale": "0=same Mandarin subgroup proxy, 1=different Mandarin subgroup inside Mandarin macrogenus",
        "source": "Language Atlas of China 2012 Mandarin subgroup taxonomy, as operationalized for KeSpeech labels",
        "coverage": {"status": "complete", "labels": list(labels), "missing_labels": []},
        "notes": [
            "All KeSpeech labels are Mandarin-family labels, so this reference is intentionally low-resolution.",
            "Mandarin is treated as a Beijing-standard proxy; therefore Beijing-Mandarin taxonomy distance is 0.",
        ],
        "matrix": _nested_matrix(labels, values),
    }


def build_sinitic_data_matrix(
    info: Mapping[str, Any],
    overall_distance: np.ndarray,
    *,
    labels: Sequence[str] = KESPEECH_DIALECTS,
    dialect_specs: Mapping[str, Mapping[str, Any]] = DIALECT_SPECS,
) -> dict[str, Any]:
    """Build a KeSpeech-label submatrix from Sinitic_Data Data4 overall distances."""
    representatives = select_sinitic_representatives(
        info,
        dialect_specs={label: dialect_specs[label] for label in labels},
    )
    if overall_distance.shape[0] != overall_distance.shape[1]:
        raise ValueError("Sinitic_Data overall_distance must be square")
    if overall_distance.shape[0] < len(info["areas"]):
        raise ValueError("Sinitic_Data overall_distance is smaller than info rows")
    selected = np.array([representatives[label]["index"] for label in labels], dtype=int)
    values = np.array(overall_distance[np.ix_(selected, selected)], dtype=float)
    values = (values + values.T) / 2.0
    np.fill_diagonal(values, 0.0)
    return {
        "schema": "reference-matrix-v1",
        "name": "sinitic_data4_overall_distance",
        "labels": list(labels),
        "distance_scale": "Sinitic_Data Data4 precomputed overall distance, raw repository scale",
        "source": "YiYang-github/Sinitic_Data Data4 overall_distance",
        "coverage": {"status": "complete", "labels": list(labels), "missing_labels": []},
        "representatives": representatives,
        "notes": [
            "Each KeSpeech label is mapped to the nearest Sinitic_Data Data4 coordinate within the corresponding Mandarin subgroup.",
            "Mandarin is treated as a Beijing-standard proxy and therefore shares the Beijing representative.",
        ],
        "matrix": _nested_matrix(labels, values),
    }


def load_sinitic_data4(root: str | Path) -> tuple[dict[str, Any], np.ndarray]:
    root_path = Path(root)
    info_path = root_path / "Data4" / "processed_info.pkl"
    matrix_path = root_path / "Data4" / "distance_matrices.npz"
    with info_path.open("rb") as handle:
        info = pickle.load(handle)
    with np.load(matrix_path) as matrices:
        overall = np.array(matrices["overall"], dtype=float)
    return info, overall


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    tmp.replace(path)


def _write_yaml(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp")
    tmp.write_text(yaml.safe_dump(dict(payload), allow_unicode=True, sort_keys=False), encoding="utf-8")
    tmp.replace(path)


def build_reference_artifacts(
    *,
    sinitic_root: str | Path,
    taxonomy_output: str | Path,
    sinitic_output: str | Path,
    provenance_output: str | Path,
    source_archive: str | Path | None = None,
) -> dict[str, Any]:
    info, overall = load_sinitic_data4(sinitic_root)
    taxonomy = build_taxonomy_matrix()
    sinitic = build_sinitic_data_matrix(info, overall)
    _write_json(Path(taxonomy_output), taxonomy)
    _write_json(Path(sinitic_output), sinitic)

    area_counts = {
        area: sum(1 for value in info["areas"] if value == area)
        for area in sorted({spec["area"] for spec in DIALECT_SPECS.values()})
    }
    provenance = {
        "schema": "reference-matrix-provenance-v1",
        "taxonomy": {
            "status": "locked",
            "source": taxonomy["source"],
            "matrix": str(taxonomy_output).replace("\\", "/"),
            "coverage": taxonomy["coverage"],
        },
        "sincomp": {
            "status": "locked",
            "source": sinitic["source"],
            "matrix": str(sinitic_output).replace("\\", "/"),
            "coverage": sinitic["coverage"],
            "area_counts": area_counts,
            "representatives": sinitic["representatives"],
            "source_commit": SINITIC_DATA_COMMIT,
            "source_url": SINITIC_DATA_SOURCE_URL,
            "source_accessed": "2026-09-04",
            "upstream_license": "not_specified",
            "source_archive_sha256": _sha256(Path(source_archive)) if source_archive else None,
            "source_commit_archive_sha256": SINITIC_DATA_COMMIT_ARCHIVE_SHA256,
            "redistribution": {
                "status": "not_redistributed",
                "reason": "upstream_license_not_specified",
                "public_route": "obtain_the_pinned_upstream_commit_and_rebuild_locally",
            },
            "derived_matrix_sha256": dict(SINITIC_DERIVED_MATRIX_SHA256),
        },
    }
    _write_yaml(Path(provenance_output), provenance)
    return provenance


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sinitic-root", required=True)
    parser.add_argument("--taxonomy-output", required=True)
    parser.add_argument("--sinitic-output", required=True)
    parser.add_argument("--provenance-output", required=True)
    parser.add_argument("--source-archive")
    args = parser.parse_args()
    provenance = build_reference_artifacts(
        sinitic_root=args.sinitic_root,
        taxonomy_output=args.taxonomy_output,
        sinitic_output=args.sinitic_output,
        provenance_output=args.provenance_output,
        source_archive=args.source_archive,
    )
    print(json.dumps(provenance, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
