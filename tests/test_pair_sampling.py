from __future__ import annotations

import pytest

from src.pair_sampling import PairSamplingError, sample_pairs


def _record(utterance_id: str, speaker_id: str, dialect_label: str, split: str, recording_condition: str = "phone-near") -> dict[str, object]:
    return {
        "utterance_id": utterance_id,
        "audio_path": f"DATASET_PATH_REDACTED/audio/{utterance_id}.wav",
        "speaker_id": speaker_id,
        "dialect_label": dialect_label,
        "sample_rate": 16000,
        "recording_condition": recording_condition,
        "split": split,
    }


def test_sample_pairs_is_deterministic_and_grouped_without_leakage():
    records = [
        _record("a1", "s1", "d1", "train", "phone-near"),
        _record("a2", "s1", "d1", "train", "phone-far"),
        _record("b1", "s2", "d1", "train", "studio"),
        _record("b2", "s3", "d1", "train", "studio"),
        _record("c1", "s1", "d2", "train", "phone-near"),
        _record("c2", "s1", "d3", "train", "phone-far"),
        _record("d1", "s4", "d2", "calibration", "phone-near"),
        _record("d2", "s5", "d3", "calibration", "phone-far"),
        _record("e1", "s6", "d4", "evaluation", "studio"),
        _record("e2", "s7", "d5", "evaluation", "studio"),
    ]

    first = sample_pairs(records, group_limits={"A": 1, "B": 1, "C": 1, "D": 1}, seed=20260829)
    second = sample_pairs(records, group_limits={"A": 1, "B": 1, "C": 1, "D": 1}, seed=20260829)

    assert first == second
    assert first["schema"] == "pair-sampling-v1"
    assert [pair["group"] for pair in first["pairs"]] == ["A", "B", "C", "D"]
    assert all(len(pair["source_utterance_ids"]) == 2 for pair in first["pairs"])

    by_group = {pair["group"]: pair for pair in first["pairs"]}
    assert by_group["A"]["speaker_ids"] == ["s1"]
    assert by_group["A"]["dialect_labels"] == ["d1"]
    assert by_group["B"]["dialect_labels"] == ["d1"]
    assert len(set(by_group["B"]["speaker_ids"])) == 2
    assert len(set(by_group["C"]["dialect_labels"])) == 2
    assert len(set(by_group["C"]["speaker_ids"])) == 1
    assert len(set(by_group["D"]["speaker_ids"])) == 2
    assert len(set(by_group["D"]["dialect_labels"])) == 2


def test_sample_pairs_logs_exclusion_when_group_c_limit_exceeds_capacity():
    records = [
        _record("u1", "s1", "d1", "train"),
        _record("u2", "s1", "d2", "train"),
        _record("u3", "s2", "d1", "train"),
        _record("u4", "s3", "d1", "train"),
    ]

    result = sample_pairs(records, group_limits={"A": 0, "B": 0, "C": 2, "D": 0}, seed=20260829)

    assert result["group_summary"]["C"]["requested"] == 2
    assert result["group_summary"]["C"]["available"] == 1
    assert result["group_summary"]["C"]["sampled"] == 1
    assert result["group_summary"]["C"]["status"] == "auxiliary"
    assert any(exclusion["group"] == "C" and exclusion["reason"] == "insufficient_capacity" for exclusion in result["exclusions"])


def test_group_b_spreads_across_dialects_before_filling_one_dialect():
    records = [
        _record("d1a", "s1", "d1", "train"),
        _record("d1b", "s2", "d1", "train"),
        _record("d1c", "s3", "d1", "train"),
        _record("d1d", "s4", "d1", "train"),
        _record("d2a", "s5", "d2", "train"),
        _record("d2b", "s6", "d2", "train"),
        _record("d2c", "s7", "d2", "train"),
        _record("d2d", "s8", "d2", "train"),
    ]

    result = sample_pairs(records, group_limits={"A": 0, "B": 4, "C": 0, "D": 0}, seed=20260829)
    b_pairs = [pair for pair in result["pairs"] if pair["group"] == "B"]

    assert len(b_pairs) == 4
    assert {pair["dialect_labels"][0] for pair in b_pairs} == {"d1", "d2"}


def test_group_a_spreads_across_dialects_before_filling_one_dialect():
    records = [
        _record("a1", "s1", "d1", "train"),
        _record("a2", "s1", "d1", "train"),
        _record("a3", "s1", "d1", "train"),
        _record("a4", "s1", "d1", "train"),
        _record("b1", "s2", "d2", "train"),
        _record("b2", "s2", "d2", "train"),
        _record("b3", "s2", "d2", "train"),
        _record("b4", "s2", "d2", "train"),
    ]

    result = sample_pairs(records, group_limits={"A": 4, "B": 0, "C": 0, "D": 0}, seed=20260829)
    a_pairs = [pair for pair in result["pairs"] if pair["group"] == "A"]

    assert len(a_pairs) == 4
    assert [pair["dialect_labels"][0] for pair in a_pairs[:2]] == ["d1", "d2"]


def test_group_d_spreads_across_dialect_pairs_before_filling_one_bucket():
    records = []
    for dialect in ["d1", "d2", "d3"]:
        for speaker_index in range(4):
            records.append(_record(f"{dialect}-u{speaker_index}", f"{dialect}-s{speaker_index}", dialect, "train"))

    result = sample_pairs(records, group_limits={"A": 0, "B": 0, "C": 0, "D": 4}, seed=20260829)
    d_pairs = [pair for pair in result["pairs"] if pair["group"] == "D"]

    assert len(d_pairs) == 4
    assert [tuple(pair["dialect_labels"]) for pair in d_pairs[:3]] == [("d1", "d2"), ("d1", "d3"), ("d2", "d3")]


def test_sample_pairs_does_not_reuse_utterances_within_each_group():
    records = []
    for dialect in ["d1", "d2", "d3"]:
        for speaker_index in range(6):
            speaker = f"{dialect}-s{speaker_index}"
            for utterance_index in range(4):
                records.append(
                    {
                        "utterance_id": f"{speaker}-u{utterance_index}",
                        "speaker_id": speaker,
                        "dialect_label": dialect,
                        "split": "evaluation",
                        "audio_path": f"Audio/{speaker}/u{utterance_index}.wav",
                        "sample_rate": 16000,
                        "recording_condition": "clean",
                    }
                )

    result = sample_pairs(records, group_limits={"A": 9, "B": 9, "C": 0, "D": 9}, seed=20260829)

    for group in ["A", "B", "D"]:
        utterance_ids = [
            utterance_id
            for pair in result["pairs"]
            if pair["group"] == group
            for utterance_id in pair["source_utterance_ids"]
        ]
        assert len(utterance_ids) == len(set(utterance_ids))


def test_group_d_excludes_same_speaker_cross_dialect_pairs():
    records = [
        _record("a1", "s1", "d1", "evaluation"),
        _record("b1", "s1", "d2", "evaluation"),
        _record("b2", "s2", "d2", "evaluation"),
    ]

    result = sample_pairs(records, group_limits={"A": 0, "B": 0, "C": 0, "D": 1}, seed=20260829)
    d_pairs = [pair for pair in result["pairs"] if pair["group"] == "D"]

    assert len(d_pairs) == 1
    assert d_pairs[0]["speaker_ids"] == ["s1", "s2"]


def test_group_d_uses_later_utterances_when_first_utterance_is_already_used():
    records = []
    for dialect in ["d1", "d2"]:
        for speaker_index in range(3):
            speaker = f"{dialect}-s{speaker_index}"
            for utterance_index in range(2):
                records.append(_record(f"{speaker}-u{utterance_index}", speaker, dialect, "evaluation"))

    result = sample_pairs(records, group_limits={"A": 0, "B": 0, "C": 0, "D": 6}, seed=20260829)
    d_pairs = [pair for pair in result["pairs"] if pair["group"] == "D"]

    assert len(d_pairs) == 6
    utterance_ids = [utterance_id for pair in d_pairs for utterance_id in pair["source_utterance_ids"]]
    assert len(utterance_ids) == len(set(utterance_ids))


def test_sample_pairs_rejects_missing_pair_limits():
    records = [_record("u1", "s1", "d1", "train")]

    with pytest.raises(PairSamplingError, match="group_limits"):
        sample_pairs(records, group_limits={"A": 1, "B": 1, "C": 1}, seed=20260829)
