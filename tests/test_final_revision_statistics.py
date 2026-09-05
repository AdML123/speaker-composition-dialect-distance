import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_final_weighting_report_covers_all_declared_estimands():
    payload = json.loads(
        (ROOT / "results/analysis/estimand_weighting_intervals.json").read_text()
    )
    assert payload["status"] == "evaluated"
    assert set(payload["weightings"]) == {
        "pair",
        "endpoint_speaker",
        "dialect_relation",
        "matched_stratum",
    }
    assert payload["primary_method"] == "hubert_linear_pair_only"
    assert payload["seed_count"] == 5
    assert payload["source_hashes"]["projection_manifest"] == (
        "59404e971b7b5c7a6e2b828be3fd935c9b09e2b9ed4db942b1ca4843dae5d18d"
    )
    assert len(payload["source_hashes"]["design_strata_manifest"]) == 64
    assert set(payload["references"]) == {"taxonomy", "city_nearest"}
    for reference in payload["references"].values():
        assert set(reference["estimands"]) == set(payload["weightings"])
        for row in reference["estimands"].values():
            assert row["gain_unit"] == "fraction"
            assert row["sample_size"]["pairs"] == 4000
            assert row["independent_unit"] == "global_endpoint_speaker"
            assert len(row["seed_values"]) == 5
            assert row["ci"]["lower"] <= row["gain"] <= row["ci"]["upper"]


def test_statistical_semantics_gate_is_terminal_and_has_wording():
    payload = json.loads(
        (ROOT / "results/gates/statistical_semantics_gate.json").read_text()
    )
    assert payload["status"] in {"passed", "narrowed", "failed"}
    assert set(payload["gates"]) == {"G2", "G3", "G4", "G5", "G6"}
    assert all(
        row["selected_wording"] and row["failure_wording"]
        for row in payload["gates"].values()
    )
