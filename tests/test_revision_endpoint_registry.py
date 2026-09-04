import json
from pathlib import Path

from src.build_revision_reports import main


ROOT = Path(__file__).resolve().parents[1]


def test_endpoint_registry_separates_confirmatory_and_exploratory_families():
    payload = json.loads(
        (ROOT / "results/analysis/revision_endpoint_registry.json").read_text()
    )
    assert payload["primary"]["phenomenon"]["model"] == "chinese_hubert_large"
    assert payload["supporting"]["extractors"]["adjustment"] == "holm"
    assert payload["confirmatory"]["architecture_interactions"]["adjustment"] == "holm"
    assert payload["exploratory"]["mechanism_probes"]["familywise_claim"] is False
    assert payload["declaration_status"] == "revision_time_declared_not_preregistered"


def test_endpoint_registry_cli_returns_success(tmp_path):
    output = tmp_path / "registry.json"
    assert main(["endpoint-registry", "--output", str(output)]) == 0
    assert output.exists()
