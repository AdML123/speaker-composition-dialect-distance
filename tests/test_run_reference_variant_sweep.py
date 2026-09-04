from src.run_reference_variant_sweep import make_variant_jobs


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
