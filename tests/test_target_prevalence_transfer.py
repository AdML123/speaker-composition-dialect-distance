from src.target_prevalence_transfer import evaluate_transfer_gate


def _condition(*, slope_lower: float, mean_lower: float, all_positive: bool = True):
    return {
        "slope_contrast": {"ci": {"lower": slope_lower}},
        "mean_gain_contrast": {"ci": {"lower": mean_lower}},
        "seed_slope_distribution": {"all_positive": all_positive},
    }


def test_transfer_gate_requires_slope_and_mean_gain_on_both_conditions():
    gate = evaluate_transfer_gate(
        _condition(slope_lower=0.01, mean_lower=0.01),
        _condition(slope_lower=0.02, mean_lower=0.02),
    )
    assert gate["rule_transfer_supported"] is True


def test_transfer_gate_fails_when_primary_mean_gain_crosses_zero():
    gate = evaluate_transfer_gate(
        _condition(slope_lower=0.01, mean_lower=-0.001),
        _condition(slope_lower=0.02, mean_lower=0.02),
    )
    assert gate["rule_transfer_supported"] is False
    assert gate["primary"]["mean_gain_improved"] is False


def test_transfer_gate_fails_when_seed_slopes_are_not_reproducible():
    gate = evaluate_transfer_gate(
        _condition(slope_lower=0.01, mean_lower=0.01, all_positive=False),
        _condition(slope_lower=0.02, mean_lower=0.02),
    )
    assert gate["rule_transfer_supported"] is False

