import json
import hashlib
from pathlib import Path

from src.build_revision_reports import build_consolidated_gate


ROOT = Path(__file__).resolve().parents[1]


def test_consolidated_gate_references_all_scientific_risk_closures():
    payload = json.loads(
        (ROOT / "results/gates/sparse_stratum_dual_endpoint_gate.json").read_text(
            encoding="utf-8"
        )
    )
    assert payload["schema"] == "sparse-stratum-dual-endpoint-gate-v1"
    assert payload["gates"]["G1_sparse_stratum"]["status"] == "passed"
    assert payload["gates"]["G3_projection_manifest"]["status"] == "passed"
    assert payload["gates"]["G5_mae_intervals"]["status"] == "passed"
    assert payload["gates"]["G6_ordering_intervals"]["status"] == "passed"
    assert payload["gates"]["G9_citation_provenance"]["status"] == "passed"
    assert payload["manuscript_status"] in {
        "scientific_gates_passed_pending_prose_visual_and_release_sync",
        "scientific_and_visual_gates_passed_pending_release_sync",
        "ready_to_submit",
    }
    assert "global winner" in payload["boundary"]


def test_consolidated_gate_hashes_match_current_artifacts():
    payload = json.loads(
        (ROOT / "results/gates/sparse_stratum_dual_endpoint_gate.json").read_text(
            encoding="utf-8"
        )
    )
    for key, path in payload["source_artifacts"].items():
        target = ROOT / path
        assert target.is_file(), key
        assert hashlib.sha256(target.read_bytes()).hexdigest() == payload["source_sha256"][key]
    assert payload["locked_input_verification"]["status"] == "passed"


def test_consolidated_gate_is_reproducible_from_current_tree():
    rebuilt = build_consolidated_gate(ROOT)
    stored = json.loads(
        (ROOT / "results/gates/sparse_stratum_dual_endpoint_gate.json").read_text(
            encoding="utf-8"
        )
    )
    assert rebuilt == stored
