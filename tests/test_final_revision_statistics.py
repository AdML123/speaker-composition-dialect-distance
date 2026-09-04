import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_final_weighting_report_covers_all_declared_estimands():
    payload = json.loads(
        (ROOT / "results/analysis/estimand_weighting_sensitivity.json").read_text()
    )
    assert payload["status"] == "evaluated"
    assert set(payload["weightings"]) == {
        "pair",
        "endpoint_speaker",
        "dialect_relation",
        "matched_stratum",
    }
    assert payload["run_count"] == 134
    assert all(
        set(run["estimands"]) == set(payload["weightings"])
        for run in payload["runs"]
    )


def test_statistical_semantics_gate_is_terminal_and_has_wording():
    payload = json.loads(
        (ROOT / "results/gates/statistical_semantics_gate.json").read_text()
    )
    assert payload["status"] in {"passed", "narrowed", "failed"}
    assert set(payload["gates"]) == {"G2", "G3", "G4", "G5", "G6"}
    assert all(
        row["selected_wording"] and row["failure_wording"]
        for row in payload["gates"].values()
    )
