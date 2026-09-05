from collections import Counter

import pytest

from src.dyadic_bootstrap import (
    dyadic_ab_bootstrap,
    endpoint_multiplicity,
    global_speaker_multiplicities,
)


def test_endpoint_multiplicity_uses_one_factor_for_self_pair():
    counts = {"a": 3, "b": 2}
    assert endpoint_multiplicity(["a"], counts) == 3
    assert endpoint_multiplicity(["a", "b"], counts) == 6


def test_global_speaker_multiplicity_is_shared_across_arms_and_strata():
    rows = [
        {"group": "A", "distance": 0.1, "speaker_ids": ["shared"], "matched_stratum": "s1"},
        {"group": "B", "distance": 0.2, "speaker_ids": ["shared", "b"], "matched_stratum": "s1"},
        {"group": "A", "distance": 0.3, "speaker_ids": ["shared"], "matched_stratum": "s2"},
        {"group": "B", "distance": 0.4, "speaker_ids": ["c", "d"], "matched_stratum": "s2"},
    ]
    counts = global_speaker_multiplicities(rows, seed=17)
    assert isinstance(counts, Counter)
    assert set(counts).issubset({"shared", "b", "c", "d"})
    assert sum(counts.values()) == 4
    assert endpoint_multiplicity(["shared"], counts) == counts["shared"]
    assert endpoint_multiplicity(["shared", "shared"], counts) == counts["shared"]


def test_dyadic_bootstrap_is_seed_reproducible():
    rows = [
        {
            "group": "A",
            "distance": 0.1,
            "speaker_ids": ["a"],
            "matched_stratum": "s",
        },
        {
            "group": "B",
            "distance": 0.4,
            "speaker_ids": ["a", "b"],
            "matched_stratum": "s",
        },
    ]
    left = dyadic_ab_bootstrap(rows, seed=7, replicates=1000)
    right = dyadic_ab_bootstrap(rows, seed=7, replicates=1000)
    assert left == right
    assert left["point_estimate"] == pytest.approx(0.3)


def test_dyadic_bootstrap_reports_zero_effective_replicates():
    rows = [
        {
            "group": "A",
            "distance": 0.1,
            "speaker_ids": ["a"],
            "matched_stratum": "s",
        },
        {
            "group": "B",
            "distance": 0.4,
            "speaker_ids": ["b", "c"],
            "matched_stratum": "s",
        },
    ]
    report = dyadic_ab_bootstrap(rows, seed=2, replicates=50)
    assert report["replicates_requested"] == 50
    assert report["zero_effective_arm_replicates"] >= 0
    assert len(report["effective_pair_count_quantiles"]["A"]) == 3
