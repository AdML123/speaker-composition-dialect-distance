from src.reference_sensitivity import architecture_boundary, reference_gate, summarize_linear_seed


def _reference(*, gain: float, seed_passed: bool, semantic: bool):
    return {
        "b4": {"gain": gain, "ci": {"lower": gain - 0.01}},
        "seed_sweep": {"passed": seed_passed},
        "semantic_specificity_supported": semantic,
    }


def test_reference_gate_separates_efficacy_from_semantic_specificity():
    gate = reference_gate(
        _reference(gain=0.08, seed_passed=True, semantic=False),
        _reference(gain=0.07, seed_passed=True, semantic=False),
    )
    assert gate["efficacy_across_references_supported"] is True
    assert gate["semantic_specificity_across_references_supported"] is False


def test_reference_gate_fails_efficacy_when_one_seed_sweep_fails():
    gate = reference_gate(
        _reference(gain=0.08, seed_passed=True, semantic=False),
        _reference(gain=0.07, seed_passed=False, semantic=False),
    )
    assert gate["efficacy_across_references_supported"] is False


def test_architecture_boundary_keeps_linear_b3_and_b4_distinct():
    boundary = architecture_boundary(
        {
            "improvement_ratio": 0.087,
            "paired_contrast_b4_vs_b3": {"passed": True},
        },
        {
            "improvement_ratio": 0.109,
            "comparisons": {
                "lambda_cross_zero": {"improvement_ratio": 0.113},
            },
            "paired_contrast_b4_vs_b3": {"passed": False},
        },
    )
    assert boundary["strongest_projection"] == "linear_b3"
    assert boundary["linear_cross_loss_increment_supported"] is False
    assert boundary["cross_loss_increment_architecture_robust"] is False


def test_linear_seed_summary_records_consistent_negative_increment():
    summary = summarize_linear_seed({
        "selected": {"head_kind": "linear"},
        "seed_results": [
            {"b3_mae": 0.40, "b4_mae": 0.41},
            {"b3_mae": 0.39, "b4_mae": 0.40},
        ],
        "distribution": {"b3_gain": {"median": 0.10}, "b4_gain": {"median": 0.08}},
        "clustered_b3_minus_b4": {"observed_delta": -0.01, "ci": {"upper": -0.002}},
    })
    assert summary["all_b3_better_than_b4"] is True
    assert summary["clustered_negative_increment_supported"] is True
