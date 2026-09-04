import numpy as np

from src.reference_matrices import (
    KESPEECH_DIALECTS,
    build_sinitic_data_matrix,
    build_taxonomy_matrix,
    select_sinitic_representatives,
    subgroup_aggregate_matrix,
    subgroup_medoid_indices,
    validate_reference_matrix,
)


def test_taxonomy_matrix_is_complete_symmetric_and_low_resolution():
    payload = build_taxonomy_matrix()

    assert payload["schema"] == "reference-matrix-v1"
    assert payload["labels"] == KESPEECH_DIALECTS
    assert payload["matrix"]["Beijing"]["Mandarin"] == 0.0
    assert payload["matrix"]["Mandarin"]["Beijing"] == 0.0
    assert payload["matrix"]["Beijing"]["Ji-Lu"] == 1.0
    assert payload["matrix"]["Ji-Lu"]["Beijing"] == 1.0
    assert all(payload["matrix"][label][label] == 0.0 for label in KESPEECH_DIALECTS)
    assert validate_reference_matrix(payload)["status"] == "passed"


def test_sinitic_representatives_choose_nearest_point_within_area():
    info = {
        "areas": ["北京官话", "北京官话", "东北官话", "兰银官话"],
        "slice": ["far", "near", "near", "near"],
        "slices": ["far", "near", "near", "near"],
        "coords": [
            [110.0, 40.0],
            [116.40, 39.90],
            [123.43, 41.81],
            [103.83, 36.06],
        ],
    }

    reps = select_sinitic_representatives(
        info,
        dialect_specs={
            "Beijing": {"area": "北京官话", "target_lon_lat": (116.4074, 39.9042), "target_name": "Beijing"},
            "Mandarin": {"area": "北京官话", "target_lon_lat": (116.4074, 39.9042), "target_name": "Beijing"},
            "Northeastern": {
                "area": "东北官话",
                "target_lon_lat": (123.4315, 41.8057),
                "target_name": "Shenyang",
            },
            "Lan-Yin": {"area": "兰银官话", "target_lon_lat": (103.8343, 36.0611), "target_name": "Lanzhou"},
        },
    )

    assert reps["Beijing"]["index"] == 1
    assert reps["Mandarin"]["index"] == 1
    assert reps["Northeastern"]["area"] == "东北官话"


def test_sinitic_data_matrix_uses_representative_indices_and_normalizes_diagonal():
    info = {
        "areas": ["北京官话", "冀鲁官话", "江淮官话"],
        "slice": ["京师片", "石济片", "洪巢片"],
        "slices": ["京师片", "石济片", "洪巢片"],
        "coords": [
            [116.40, 39.90],
            [117.12, 36.65],
            [117.23, 31.82],
        ],
    }
    distances = np.array(
        [
            [0.0, 0.2, 0.4],
            [0.2, 0.0, 0.6],
            [0.4, 0.6, 0.0],
        ]
    )

    payload = build_sinitic_data_matrix(
        info,
        distances,
        labels=["Beijing", "Mandarin", "Ji-Lu", "Jiang-Huai"],
        dialect_specs={
            "Beijing": {"area": "北京官话", "target_lon_lat": (116.4074, 39.9042), "target_name": "Beijing"},
            "Mandarin": {"area": "北京官话", "target_lon_lat": (116.4074, 39.9042), "target_name": "Beijing"},
            "Ji-Lu": {"area": "冀鲁官话", "target_lon_lat": (117.1201, 36.6512), "target_name": "Jinan"},
            "Jiang-Huai": {"area": "江淮官话", "target_lon_lat": (117.2272, 31.8206), "target_name": "Hefei"},
        },
    )

    assert payload["coverage"]["status"] == "complete"
    assert payload["matrix"]["Beijing"]["Mandarin"] == 0.0
    assert payload["matrix"]["Ji-Lu"]["Jiang-Huai"] == 0.6
    assert payload["matrix"]["Jiang-Huai"]["Ji-Lu"] == 0.6


def test_medoid_minimizes_within_group_mean_distance():
    distance = np.array(
        [[0.0, 1.0, 4.0], [1.0, 0.0, 2.0], [4.0, 2.0, 0.0]]
    )
    assert subgroup_medoid_indices(distance, {"A": [0, 1, 2]})["A"] == 1


def test_medoid_uses_lowest_index_on_exact_tie():
    distance = np.array([[0.0, 1.0], [1.0, 0.0]])
    assert subgroup_medoid_indices(distance, {"A": [1, 0]})["A"] == 0


def test_aggregate_is_mean_cross_group_distance():
    distance = np.array(
        [[0.0, 2.0, 4.0], [2.0, 0.0, 6.0], [4.0, 6.0, 0.0]]
    )
    result = subgroup_aggregate_matrix(distance, {"A": [0, 1], "B": [2]})
    assert result["A"]["B"] == 5.0
    assert result["A"]["A"] == 0.0


def test_aggregate_keeps_identical_proxy_groups_at_zero():
    distance = np.array([[0.0, 2.0], [2.0, 0.0]])
    result = subgroup_aggregate_matrix(
        distance, {"Beijing": [0, 1], "Mandarin": [0, 1]}
    )
    assert result["Beijing"]["Mandarin"] == 0.0
