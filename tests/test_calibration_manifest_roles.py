from pathlib import Path

from src.calibration_manifest_roles import summarize_manifest


ROOT = Path(__file__).resolve().parents[1]


def role(name: str) -> dict:
    summary = summarize_manifest(ROOT / "results" / "pairs" / name)
    return {
        "role": summary["role"],
        "split": summary["split"],
        "pair_count": summary["pair_count"],
        "groups": summary["groups"],
    }


def test_locked_manifest_roles_have_distinct_counts():
    assert role("kespeech_calibration_matched.json") == {
        "role": "legacy_ab_calibration_audit",
        "split": "calibration",
        "pair_count": 2000,
        "groups": {"A": 1000, "B": 1000},
    }
    assert role("kespeech_calibration_1000.json") == {
        "role": "projection_calibration",
        "split": "calibration",
        "pair_count": 3123,
        "groups": {"A": 1000, "B": 1000, "C": 688, "D": 435},
    }
    assert role("kespeech_evaluation_matched.json") == {
        "role": "phenomenon_evaluation",
        "split": "evaluation",
        "pair_count": 12980,
        "groups": {"A": 3123, "B": 9857},
    }
    assert role("kespeech_evaluation_1000.json") == {
        "role": "projection_evaluation",
        "split": "evaluation",
        "pair_count": 4000,
        "groups": {"A": 1000, "B": 1000, "C": 1000, "D": 1000},
    }


def test_manifest_role_summary_preserves_identity_and_support():
    summary = summarize_manifest(ROOT / "results" / "pairs" / "kespeech_calibration_1000.json")
    assert summary["sha256"]
    assert summary["endpoint_speaker_count"] == 75
    assert summary["split_speaker_count"] == 100
    assert summary["duplicate_pair_ids"] == []


def test_mixed_split_is_rejected(tmp_path):
    import json

    source = ROOT / "results" / "pairs" / "kespeech_calibration_1000.json"
    payload = json.loads(source.read_text(encoding="utf-8"))
    payload["pairs"][0]["split"] = "evaluation"
    mixed = tmp_path / "mixed.json"
    mixed.write_text(json.dumps(payload), encoding="utf-8")
    try:
        summarize_manifest(mixed)
    except ValueError as exc:
        assert "mixed split" in str(exc)
    else:
        raise AssertionError("mixed split was accepted")
