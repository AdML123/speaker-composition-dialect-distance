import pytest

from src.run_target_prevalence_mechanism import build_slope_rows, seed_slope_distribution

from src.target_prevalence_experiment import (
    audit_prevalence_capacity,
    build_nested_fixed_pair_masks,
    build_natural_prevalence_pools,
    prevalence_balanced_loss,
    slope_contrast,
    clustered_slope_bootstrap,
    clustered_mean_gain_contrast,
)


def _examples():
    rows = []
    for index in range(1200):
        target = 0.0 if index < 600 else 1.0
        rows.append({
            "pair_id": f"p{index}", "target": target,
            "utterance_ids": [f"u{index}a", f"u{index}b"],
            "speaker_ids": [f"s{index%20}", f"s{(index+1)%20}"],
            "dialect_labels": ["M", "M" if target == 0 else "R"],
            "recording_conditions": ["clean", "clean"],
        })
    return rows


def test_capacity_audit_freezes_largest_multiple_of_twenty():
    report = audit_prevalence_capacity(_examples())
    assert report["status"] == "passed"
    assert report["common_pool_size"] == 600
    assert report["result_blind"] is True


def test_fixed_pair_masks_are_nested_and_hold_pair_identity():
    nonzero = [row for row in _examples() if row["target"] > 0]
    report = build_nested_fixed_pair_masks(nonzero, pool_size=200, seed=7)
    identities = [[item["pair_id"] for item in arm] for arm in report["arms"].values()]
    assert all(ids == identities[0] for ids in identities)
    nonzero_sets = [
        {item["pair_id"] for item in report["arms"][q] if item["target"] > 0}
        for q in ["0.00", "0.10", "0.25", "0.50", "0.75", "1.00"]
    ]
    assert all(left <= right for left, right in zip(nonzero_sets, nonzero_sets[1:]))


def test_natural_pool_reports_structural_covariates():
    report = build_natural_prevalence_pools(_examples(), pool_size=200, seed=7)
    assert report["covariates"]["0.00"]["unique_dialect_pair_count"] == 1
    assert report["covariates"]["1.00"]["unique_dialect_pair_count"] == 1
    assert "endpoint_reuse_fraction" in report["covariates"]["0.50"]


def test_balanced_loss_and_slope_contrast():
    assert prevalence_balanced_loss([1.0, 3.0], [5.0, 7.0]) == pytest.approx(4.0)
    result = slope_contrast(
        [{"q": 0.0, "gain": 0.0}, {"q": 1.0, "gain": 1.0}],
        [{"q": 0.0, "gain": 0.0}, {"q": 1.0, "gain": 0.2}],
    )
    assert result["delta_beta"] == pytest.approx(0.8)


def test_clustered_slope_bootstrap_reports_speaker_unit():
    rows = [
        {"q": 0.25, "speaker_ids": ["s1"], "ordinary_gain": 0.1, "balanced_gain": 0.05},
        {"q": 0.25, "speaker_ids": ["s2"], "ordinary_gain": 0.2, "balanced_gain": 0.1},
        {"q": 0.75, "speaker_ids": ["s1"], "ordinary_gain": 0.4, "balanced_gain": 0.2},
        {"q": 0.75, "speaker_ids": ["s2"], "ordinary_gain": 0.5, "balanced_gain": 0.25},
    ]
    report = clustered_slope_bootstrap(rows, seed=20260829, replicates=1000)
    assert report["resampling_unit"] == "speaker_cluster_within_q"
    assert report["ci"]["upper"] >= report["ci"]["lower"]


def test_slope_rows_average_all_seeds_instead_of_overwriting():
    arm = {}
    for q in ("0.10", "0.25"):
        arm[q] = {
            "ordinary": {"speaker_gain": [
                {"seed": 1, "speaker_gain": {"s1": 1.0, "s2": 3.0}},
                {"seed": 2, "speaker_gain": {"s1": 5.0, "s2": 7.0}},
            ]},
            "prevalence_balanced": {"speaker_gain": [
                {"seed": 1, "speaker_gain": {"s1": 0.0, "s2": 2.0}},
                {"seed": 2, "speaker_gain": {"s1": 4.0, "s2": 6.0}},
            ]},
        }
    rows = build_slope_rows(arm, ("0.10", "0.25"))
    assert rows[0]["ordinary_gain"] == pytest.approx(3.0)
    assert rows[0]["balanced_gain"] == pytest.approx(2.0)
    distribution = seed_slope_distribution(arm, ("0.10", "0.25"))
    assert distribution["seed_count"] == 2


def test_clustered_mean_gain_contrast_reports_balanced_minus_ordinary():
    rows = [
        {"q": 0.1, "speaker_ids": ["s1"], "ordinary_gain": 0.1, "balanced_gain": 0.3},
        {"q": 0.5, "speaker_ids": ["s1"], "ordinary_gain": 0.2, "balanced_gain": 0.4},
        {"q": 0.1, "speaker_ids": ["s2"], "ordinary_gain": 0.0, "balanced_gain": 0.1},
        {"q": 0.5, "speaker_ids": ["s2"], "ordinary_gain": 0.1, "balanced_gain": 0.2},
    ]
    report = clustered_mean_gain_contrast(rows, seed=9, replicates=1000)
    assert report["estimate"] == pytest.approx(0.15)
    assert report["contrast"] == "prevalence_balanced_minus_ordinary_mean_gain_across_internal_q"
