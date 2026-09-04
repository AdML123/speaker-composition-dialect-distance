from src.target_permutation_control import permute_pair_distance_targets


def test_target_permutation_preserves_pool_and_histogram_but_changes_assignment():
    examples = [
        {"pair_id": "p1", "utterance_ids": ["a", "b"], "target": 0.0},
        {"pair_id": "p2", "utterance_ids": ["c", "d"], "target": 0.4},
        {"pair_id": "p3", "utterance_ids": ["e", "f"], "target": 0.8},
    ]
    report = permute_pair_distance_targets(examples, seed=7)
    assert report["pool_size"] == 3
    assert report["target_histogram_before"] == report["target_histogram_after"]
    assert [item["pair_id"] for item in report["pair_examples"]] == ["p1", "p2", "p3"]
    assert [item["target"] for item in report["pair_examples"]] != [0.0, 0.4, 0.8]
    assert report["evaluation_targets_unchanged"] is True
    assert report["changed"] == ["target_to_pair_assignment"]
