from __future__ import annotations

import pytest

from src.statistical_gate import _parse_args, holm_adjust, run_speaker_effect_gate


def _row(model: str, group: str, dialect: str, distance: float, speaker: str) -> dict[str, object]:
    return {
        "model_name": model,
        "pair_id": f"{model}-{group}-{dialect}-{speaker}-{distance}",
        "group": group,
        "split": "calibration",
        "dialect_labels": [dialect],
        "speaker_ids": [speaker] if group == "A" else [speaker, f"{speaker}x"],
        "distance": distance,
    }


def test_holm_adjust_is_monotone_and_caps_at_one():
    adjusted = holm_adjust({"a": 0.001, "b": 0.02, "c": 0.9})

    assert adjusted["a"] == pytest.approx(0.003)
    assert adjusted["b"] == pytest.approx(0.04)
    assert adjusted["c"] == pytest.approx(0.9)


def test_speaker_effect_gate_passes_positive_fixture():
    rows = []
    for model in ["m1", "m2", "m3"]:
        for dialect in ["d1", "d2", "d3"]:
            for index in range(8):
                rows.append(_row(model, "A", dialect, 0.1 + index * 0.001, f"{dialect}-a{index}"))
                rows.append(_row(model, "B", dialect, 0.4 + index * 0.001, f"{dialect}-b{index}"))

    report = run_speaker_effect_gate(rows, seed=20260829, replicates=1000)

    assert report["status"] == "passed"
    assert all(model["effect"] > 0 for model in report["models"])
    assert all(model["ci"]["lower"] > 0 for model in report["models"])
    assert all(model["holm_adjusted_p"] < 0.05 for model in report["models"])


def test_speaker_effect_gate_fails_null_fixture():
    rows = []
    for model in ["m1", "m2", "m3"]:
        for dialect in ["d1", "d2"]:
            for index in range(8):
                rows.append(_row(model, "A", dialect, 0.2, f"{dialect}-a{index}"))
                rows.append(_row(model, "B", dialect, 0.2, f"{dialect}-b{index}"))

    report = run_speaker_effect_gate(rows, seed=20260829, replicates=1000)

    assert report["status"] == "failed"
    assert all(model["effect"] == pytest.approx(0.0) for model in report["models"])


def test_cli_accepts_repeated_distance_flags():
    args = _parse_args(["--distances", "a.json", "--distances", "b.json", "c.json", "--output", "out.json"])

    assert args.distances == ["a.json", "b.json", "c.json"]
