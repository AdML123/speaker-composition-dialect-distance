from __future__ import annotations

import pytest

from src.matched_pair_audit import MatchingAuditError, sample_matched_ab_pairs


def _record(
    utterance_id: str,
    speaker_id: str,
    dialect_label: str = "d1",
    condition: str = "phase1:Dialect",
    content_id: str | None = "c1",
    split: str = "evaluation",
) -> dict[str, object]:
    record: dict[str, object] = {
        "utterance_id": utterance_id,
        "audio_path": f"Audio/{utterance_id}.wav",
        "speaker_id": speaker_id,
        "dialect_label": dialect_label,
        "sample_rate": 16000,
        "recording_condition": condition,
        "split": split,
    }
    if content_id is not None:
        record["content_id"] = content_id
    return record


def test_matched_ab_pairs_hold_content_condition_and_split_fixed():
    records = [
        _record("a1", "s1"),
        _record("a2", "s1"),
        _record("b1", "s2"),
        _record("b2", "s2"),
        _record("other", "s3", content_id="c2"),
    ]

    result = sample_matched_ab_pairs(records, group_limits={"A": 1, "B": 1}, seed=9)

    assert result["status"] == "passed"
    assert [pair["group"] for pair in result["pairs"]] == ["A", "B"]
    for pair in result["pairs"]:
        assert pair["matched_fields"] == ["content_id", "dialect_label", "recording_condition", "split"]
        assert len(set(pair["content_ids"])) == 1
        assert len(set(pair["recording_conditions"])) == 1
        assert pair["split"] == "evaluation"


def test_matched_ab_pairs_reject_missing_required_content_metadata():
    with pytest.raises(MatchingAuditError, match="content_id"):
        sample_matched_ab_pairs([_record("u1", "s1", content_id=None)])


def test_matched_ab_pairs_do_not_match_different_conditions():
    records = [
        _record("a1", "s1", condition="phase1:Dialect"),
        _record("a2", "s1", condition="phase1:Dialect", content_id="c2"),
        _record("b1", "s2", condition="phase1:Accent"),
        _record("b2", "s2", condition="phase1:Accent", content_id="c2"),
    ]

    result = sample_matched_ab_pairs(records, group_limits={"A": 1, "B": 1})

    assert result["group_summary"]["A"]["sampled"] == 0
    assert result["group_summary"]["B"]["sampled"] == 0
    assert result["group_summary"]["A"]["status"] == "partial"


def test_matched_ab_pairs_report_cluster_level_counts_without_raw_text():
    records = [
        _record("a1", "s1"),
        _record("a2", "s1"),
        _record("b1", "s2"),
        _record("b2", "s2"),
    ]

    result = sample_matched_ab_pairs(records, group_limits={"A": 1, "B": 1})

    assert result["audit"]["content_cluster_count"] == 1
    assert result["audit"]["speaker_count"] == 2
    assert "raw transcript text" not in str(result)


def test_matched_ab_pairs_keep_disabled_groups_empty():
    records = [_record("a1", "s1"), _record("a2", "s1")]

    result = sample_matched_ab_pairs(records, group_limits={"A": 0, "B": 0})

    assert result["group_summary"]["A"]["sampled"] == 0
    assert result["group_summary"]["B"]["sampled"] == 0
    assert result["pairs"] == []


def test_condition_aware_mode_accepts_records_without_content_id():
    records = [
        _record("a1", "s1", content_id=None),
        _record("a2", "s1", content_id=None),
        _record("b1", "s2", content_id=None),
        _record("b2", "s2", content_id=None),
    ]

    result = sample_matched_ab_pairs(
        records,
        group_limits={"A": 1, "B": 1},
        match_content=False,
    )

    assert result["matching_mode"] == "condition_aware"
    assert result["audit"]["matching_fields"] == ["split", "dialect_label", "recording_condition"]
    assert result["group_summary"]["A"]["sampled"] == 1
    assert result["group_summary"]["B"]["sampled"] == 1
    assert result["pairs"][0]["content_ids"] == [None, None]
