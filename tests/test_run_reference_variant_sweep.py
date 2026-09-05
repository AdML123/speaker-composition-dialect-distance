import pytest

from src.run_reference_variant_sweep import (
    REQUIRED_METHODS,
    make_architecture_jobs,
    make_variant_jobs,
    validate_complete_sweep,
)


def test_variant_jobs_share_pairs_seeds_and_budget():
    jobs = make_variant_jobs(
        ["city_nearest", "subgroup_medoid", "subgroup_aggregate"]
    )
    assert {job["seeds"] for job in jobs} == {
        (20260829, 20260830, 20260831, 20260901, 20260902)
    }
    assert {job["head_kind"] for job in jobs} == {"linear"}
    assert {job["lambda_cross"] for job in jobs} == {0.0}
    assert {job["fixed_epochs"] for job in jobs} == {29}
    assert len({job["pair_manifest_hash"] for job in jobs}) == 1


def test_variant_jobs_do_not_encode_evaluation_selection():
    jobs = make_variant_jobs(["city_nearest", "subgroup_medoid"])
    assert all(job["selection_unit"] == "calibration_speaker_folds" for job in jobs)
    assert all("evaluation_metric" not in job for job in jobs)


def test_architecture_jobs_cover_three_pair_only_heads_and_five_seeds():
    jobs = make_architecture_jobs(["subgroup_medoid", "subgroup_aggregate"])
    assert len(jobs) == 2 * 3 * 5
    assert {job["method"] for job in jobs} == {
        "linear", "matched_mlp", "wide_mlp"
    }
    assert all(job["lambda_cross"] == 0.0 for job in jobs)
    assert all(job["fixed_epochs"] == 30 for job in jobs)


def _complete_report():
    deterministic = {
        "seed_results": [
            {"seed": None, "per_pair": [{"pair_id": str(i)} for i in range(4000)]}
        ]
    }
    stochastic = {
        "seed_results": [
            {
                "seed": seed,
                "per_pair": [{"pair_id": str(i)} for i in range(4000)],
            }
            for seed in (20260829, 20260830, 20260831, 20260901, 20260902)
        ]
    }
    return {
        "references": {
            reference: {
                "methods": {
                    method: (
                        deterministic
                        if method in {"frozen_affine", "principal_component"}
                        else stochastic
                    )
                    for method in REQUIRED_METHODS
                }
            }
            for reference in (
                "taxonomy", "city_nearest", "subgroup_medoid", "subgroup_aggregate"
            )
        }
    }


def test_complete_sweep_requires_four_references_six_methods_and_pair_rows():
    validate_complete_sweep(_complete_report())


def test_complete_sweep_rejects_missing_reference_method():
    report = _complete_report()
    del report["references"]["subgroup_medoid"]["methods"]["wide_mlp"]
    with pytest.raises(ValueError, match="method set"):
        validate_complete_sweep(report)
