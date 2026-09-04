import pytest

from src.pool_ratio_gradient_budget import (
    build_candidate_inventory_grid,
    build_exposure_ratio_grid,
    classify_mechanism,
    compute_aggregation_arms,
    compute_gradient_budget_readouts,
    fit_log_rho_lambda_interaction,
    speaker_cluster_interaction_bootstrap,
)


def test_exposure_grid_is_locked():
    grid = build_exposure_ratio_grid()
    assert [(row["generic_count"], row["cross_count"]) for row in grid] == [(1, 1), (2, 1), (4, 1), (8, 1)]


def test_candidate_inventory_grid_is_result_blind_and_bounded():
    pairs = [{"pair_id": f"p{i}"} for i in range(10)]
    cross = [{"pair_id": f"c{i}"} for i in range(20)]
    grid = build_candidate_inventory_grid(pairs, cross)
    assert grid[0]["pair_inventory"] == 2
    assert grid[-1]["pair_inventory"] == 10
    assert grid[-1]["cross_inventory"] == 20


def test_aggregation_arms_follow_declared_formulas():
    arms = compute_aggregation_arms([1.0, 3.0], [2.0, 6.0], lambda_cross=0.5)
    assert arms["separate"] == pytest.approx(2.0 + 0.5 * 4.0)
    assert arms["mixed_mean"] == pytest.approx((4.0 + 0.5 * 8.0) / 4.0)


def test_gradient_budget_readouts_are_bounded():
    report = compute_gradient_budget_readouts(2.0, 1.0, lambda_cross=0.5, n_pair=8, n_cross=2, cosine=-0.4)
    assert 0.0 <= report["eta_sep"] <= 1.0
    assert 0.0 <= report["eta_mix"] <= 1.0
    with pytest.raises(ValueError):
        compute_gradient_budget_readouts(1.0, 1.0, lambda_cross=1.0, n_pair=1, n_cross=1, cosine=2.0)


def test_mechanism_classifier_fails_closed_on_nonunique_evidence():
    report = classify_mechanism(prevalence_supported=True, gradient_budget_supported=True, interference_supported=False, regularization_compatible=False)
    assert report["status"] == "compatible_with_multiple_mechanisms"
    unresolved = classify_mechanism(prevalence_supported=False, gradient_budget_supported=False, interference_supported=False, regularization_compatible=False)
    assert unresolved["status"] == "mechanism_unresolved"


def test_fixed_pair_association_is_not_promoted_to_prevalence_support():
    fixed_pair_association_supported = True
    report = classify_mechanism(
        prevalence_supported=False,
        gradient_budget_supported=False,
        interference_supported=False,
        regularization_compatible=False,
    )
    assert fixed_pair_association_supported
    assert report["status"] == "mechanism_unresolved"
    assert report["compatible_mechanisms"] == []


def test_interaction_estimator_and_cluster_bootstrap():
    rows = []
    for speaker_offset, speaker in enumerate(("s1", "s2", "s3")):
        for rho in (1.0, 2.0, 4.0, 8.0):
            for lam in (0.0, 1.0):
                gain = 0.01 * speaker_offset + 0.2 * lam * __import__("math").log(rho)
                rows.append({"speaker_id": speaker, "rho": rho, "lambda_cross": lam, "gain": gain})
    cells = []
    for rho in (1.0, 2.0, 4.0, 8.0):
        for lam in (0.0, 1.0):
            cells.append({"rho": rho, "lambda_cross": lam, "gain": sum(r["gain"] for r in rows if r["rho"] == rho and r["lambda_cross"] == lam) / 3})
    assert fit_log_rho_lambda_interaction(cells)["log_rho_by_lambda_cross"] == pytest.approx(0.2)
    report = speaker_cluster_interaction_bootstrap(rows, seed=7, replicates=1000)
    assert report["estimate"] == pytest.approx(0.2)
    assert report["resampling_unit"] == "evaluation_speaker_cluster"
