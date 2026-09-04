from src.matched_design_summary import summarize_matched_design


def test_summary_counts_strata_and_does_not_duplicate_same_speaker_endpoint():
    pairs = [
        {
            "group": "A",
            "matched_stratum": "s1",
            "speaker_ids": ["a"],
            "utterance_ids": ["u1", "u2"],
            "recording_conditions": ["c", "c"],
        },
        {
            "group": "B",
            "matched_stratum": "s1",
            "speaker_ids": ["a", "b"],
            "utterance_ids": ["u3", "u4"],
            "recording_conditions": ["c", "c"],
        },
    ]
    report = summarize_matched_design(pairs)
    assert report["stratum_count"] == 1
    assert report["groups"]["A"]["unique_speaker_count"] == 1
    assert report["groups"]["B"]["unique_speaker_count"] == 2
    assert report["groups"]["A"]["reused_utterance_count"] == 0


def test_summary_reports_reused_utterance_and_max_speaker_participation():
    pairs = [
        {
            "group": "B",
            "matched_stratum": "s",
            "speaker_ids": ["a", "b"],
            "utterance_ids": ["u1", "u2"],
            "recording_conditions": ["c", "c"],
        },
        {
            "group": "B",
            "matched_stratum": "s",
            "speaker_ids": ["a", "c"],
            "utterance_ids": ["u1", "u3"],
            "recording_conditions": ["c", "c"],
        },
    ]
    report = summarize_matched_design(pairs)
    assert report["groups"]["B"]["reused_utterance_count"] == 1
    assert report["groups"]["B"]["max_speaker_pair_count"] == 2


def test_summary_orders_strata_and_reports_cross_group_overlap():
    pairs = [
        {
            "group": "B",
            "matched_stratum": "z",
            "speaker_ids": ["b", "c"],
            "utterance_ids": ["u2", "u3"],
            "recording_conditions": ["z", "z"],
        },
        {
            "group": "A",
            "matched_stratum": "a",
            "speaker_ids": ["a"],
            "utterance_ids": ["u1", "u2"],
            "recording_conditions": ["a", "a"],
        },
    ]
    report = summarize_matched_design(pairs)
    assert [row["matched_stratum"] for row in report["strata"]] == ["a", "z"]
    assert report["cross_group_utterance_overlap_count"] == 1
