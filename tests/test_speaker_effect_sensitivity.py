import pytest

from src.speaker_effect_sensitivity import (
    global_dyadic_speaker_bootstrap,
    resolve_excluded_stratum,
    overlap_pair_weight,
    speaker_effect_sensitivity,
    stratum_effects,
)


def _row(group, distance, speakers, stratum):
    return {
        "group": group,
        "distance": distance,
        "speaker_ids": speakers,
        "matched_stratum": stratum,
    }


def _sparse_fixture():
    rows = [_row("A", 0.0, ["ma"], "Mandarin|phase1:Accent")]
    rows.extend(
        _row("B", 100.0, [f"mb-{index}-a", f"mb-{index}-b"], "Mandarin|phase1:Accent")
        for index in range(160)
    )
    rows.extend(
        _row("A", 0.0, [f"a-{index}"], "balanced") for index in range(50)
    )
    rows.extend(
        _row("B", 1.0, [f"b-{index}-a", f"b-{index}-b"], "balanced")
        for index in range(50)
    )
    return rows


def test_overlap_pair_weight_uses_harmonic_overlap_support():
    assert overlap_pair_weight(1, 160) == pytest.approx(160 / 161)
    assert overlap_pair_weight(50, 50) == pytest.approx(25.0)
    with pytest.raises(ValueError, match="positive"):
        overlap_pair_weight(0, 10)


def test_stratum_effects_report_counts_speakers_weights_and_contributions():
    report = {row["stratum"]: row for row in stratum_effects(_sparse_fixture())}
    sparse = report["Mandarin|phase1:Accent"]
    assert sparse["A_pair_count"] == 1
    assert sparse["B_pair_count"] == 160
    assert sparse["A_unique_speaker_count"] == 1
    assert sparse["B_unique_speaker_count"] == 320
    assert sparse["effect"] == pytest.approx(100.0)
    assert sparse["overlap_weight"] == pytest.approx(160 / 161)
    assert sparse["weighted_effect_contribution"] == pytest.approx(
        sparse["effect"] * sparse["overlap_weight"]
    )


def test_sparse_stratum_cannot_dominate_overlap_weighted_headline():
    report = speaker_effect_sensitivity(
        _sparse_fixture(), excluded_stratum="Mandarin|phase1:Accent"
    )
    expected = ((160 / 161) * 100.0 + 25.0) / ((160 / 161) + 25.0)
    assert report["primary"]["estimate"] == pytest.approx(expected)
    assert report["primary"]["estimate"] < 5.0
    assert report["legacy_equal_stratum"]["estimate"] == pytest.approx(50.5)
    assert report["excluded_stratum"]["estimate"] == pytest.approx(1.0)
    assert report["excluded_stratum"]["excluded"] == "Mandarin|phase1:Accent"
    assert len(report["leave_one_stratum_out"]) == 2
    assert report["pooled_pair_weighted"]["status"] == "descriptive_only"
    assert report["support_qualified_equal_stratum"]["estimate"] == pytest.approx(1.0)
    assert report["support_qualified_equal_stratum"]["excluded_strata"] == [
        "Mandarin|phase1:Accent"
    ]


def test_stratum_effects_reject_an_arm_missing_from_any_stratum():
    rows = [_row("A", 0.1, ["a"], "only-a")]
    with pytest.raises(ValueError, match="both A and B"):
        stratum_effects(rows)


def test_bootstrap_recomputes_arm_medians_inside_each_replicate():
    rows = [
        _row("A", 0.0, ["a-low"], "s"),
        _row("A", 10.0, ["a-high"], "s"),
        _row("B", 20.0, ["b-low"], "s"),
        _row("B", 30.0, ["b-high"], "s"),
    ]
    report = global_dyadic_speaker_bootstrap(rows, seed=7, replicates=500)
    assert report["stratum_medians_recomputed_per_replicate"] is True
    assert report["bootstrap_estimate_standard_deviation"] > 0
    assert report["ci"]["upper"] > report["ci"]["lower"]


def test_protocol_amendment_explicitly_maps_conceptual_stratum_to_source_id():
    amendment = {
        "schema": "sparse-stratum-protocol-amendment-v1",
        "status": "locked_before_successful_analysis",
        "conceptual_stratum": "Mandarin|phase1:Accent",
        "source_stratum": "evaluation|Mandarin|phase1:Accent",
        "changes_estimand": False,
    }
    rows = [
        _row("A", 0.0, ["a"], "evaluation|Mandarin|phase1:Accent"),
        _row("B", 1.0, ["b"], "evaluation|Mandarin|phase1:Accent"),
    ]
    assert resolve_excluded_stratum(
        "Mandarin|phase1:Accent", rows, amendment
    ) == "evaluation|Mandarin|phase1:Accent"


def test_protocol_amendment_rejects_silent_or_estimand_changing_mapping():
    rows = [
        _row("A", 0.0, ["a"], "evaluation|Mandarin|phase1:Accent"),
        _row("B", 1.0, ["b"], "evaluation|Mandarin|phase1:Accent"),
    ]
    with pytest.raises(ValueError, match="explicit amendment"):
        resolve_excluded_stratum("Mandarin|phase1:Accent", rows, None)
    with pytest.raises(ValueError, match="cannot change the estimand"):
        resolve_excluded_stratum(
            "Mandarin|phase1:Accent",
            rows,
            {
                "schema": "sparse-stratum-protocol-amendment-v1",
                "status": "locked_before_successful_analysis",
                "conceptual_stratum": "Mandarin|phase1:Accent",
                "source_stratum": "evaluation|Mandarin|phase1:Accent",
                "changes_estimand": True,
            },
        )
