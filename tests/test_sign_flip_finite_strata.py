import pytest

from src.speaker_effect_sensitivity import exact_weighted_sign_flip


def test_exact_sign_flip_records_finite_stratum_family():
    rows = [
        {"effect": 0.1, "overlap_weight": 1.0}
        for _ in range(12)
    ]
    result = exact_weighted_sign_flip(rows)
    assert result["permutations"] == 4096
    assert result["enumeration_family_size"] == 4096
    assert result["exchangeability_unit"] == "stratum_effect_sign"
    assert result["tail_resolution"] == pytest.approx(2 / 4096)


def test_exact_sign_flip_rejects_missing_stratum_effect():
    with pytest.raises(ValueError, match="stratum effect"):
        exact_weighted_sign_flip([{"effect": 0.1, "overlap_weight": 1.0}, {}])
