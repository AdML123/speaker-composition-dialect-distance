from pathlib import Path

import pytest

from src.speaker_offset_reports import build_speaker_offset_reports


def test_reports_include_pair_stratification_and_runtime_fields():
    if not Path("results/embeddings/kespeech_evaluation_full/chinese_hubert_large.json").is_file():
        pytest.skip("private extracted embeddings are intentionally excluded from the public release")
    report = build_speaker_offset_reports(
        gate_paths=["results/gates/speaker_mean_normalization_gate.json"],
        pair_manifest_path="results/pairs/kespeech_evaluation_1000.json",
    )
    assert "pair_strata" in report
    assert "runtime_cost" in report
