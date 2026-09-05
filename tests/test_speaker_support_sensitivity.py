import pytest

from src.speaker_support_sensitivity import (
    aggregate_support_effects,
    bootstrap_support_estimands,
    incidence_ess,
    speaker_overlap_weight,
    summarize_support,
)


def _row(group, stratum, speaker, distance=0.0):
    return {
        "group": group,
        "matched_stratum": stratum,
        "speaker_ids": [speaker],
        "distance": distance,
    }


def test_support_weight_formulas():
    assert speaker_overlap_weight(2, 4) == pytest.approx(4 / 3)
    assert incidence_ess([4, 1, 1]) == pytest.approx(36 / 18)


def test_support_summary_reports_all_strata_and_m_a():
    rows = [
        _row("A", "M-A", "a1"),
        _row("B", "M-A", "b1"),
        _row("B", "M-A", "b2"),
        _row("A", "balanced", "a2"),
        _row("A", "balanced", "a3"),
        _row("B", "balanced", "b3"),
        _row("B", "balanced", "b4"),
    ]
    result = summarize_support(rows)
    sparse = next(row for row in result if row["stratum"] == "M-A")
    assert sparse["A"]["pair_count"] == 1
    assert sparse["B"]["unique_speaker_count"] == 2
    assert sparse["w_speaker"] < 1.0
    assert {row["stratum"] for row in result} == {"M-A", "balanced"}


def test_support_aggregate_uses_selected_weight_family():
    rows = [
        {"stratum": "a", "w_pair": 1.0, "w_speaker": 1.0, "w_ess": 1.0},
        {"stratum": "b", "w_pair": 3.0, "w_speaker": 1.0, "w_ess": 2.0},
    ]
    effects = {"a": 0.0, "b": 2.0}
    assert aggregate_support_effects(rows, effects, "w_pair") == pytest.approx(1.5)
    assert aggregate_support_effects(rows, effects, "w_speaker") == pytest.approx(1.0)


def test_support_bootstrap_reports_all_weight_families():
    rows = [
        _row("A", "s", "a1", 0.0),
        _row("A", "s", "a2", 0.1),
        _row("B", "s", "b1", 1.0),
        _row("B", "s", "b2", 1.1),
    ]
    result = bootstrap_support_estimands(rows, seed=7, replicates=100)
    assert set(result) == {"w_pair", "w_speaker", "w_ess"}
    assert all(row["ci"]["lower"] > 0 for row in result.values())
    assert all(row["resampling_unit"] == "global_endpoint_speaker" for row in result.values())
