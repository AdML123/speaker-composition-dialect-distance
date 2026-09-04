from pathlib import Path

import json
import yaml


ROOT = Path(__file__).resolve().parents[1]
RELEASE = ROOT / "release" / "github" if (ROOT / "release" / "github").is_dir() else ROOT
COMMIT = "820b9d15a74cbee82109f0bf54cf791fe16596ef"


def test_release_excludes_manuscript_pdf_and_latex_content():
    assert not list(RELEASE.rglob("*.tex"))
    assert not list(RELEASE.rglob("*.pdf"))


def test_unlicensed_sinitic_matrices_are_not_redistributed():
    reference_dir = RELEASE / "results" / "references"
    assert (reference_dir / "taxonomy_matrix.json").is_file()
    assert not list(reference_dir.glob("sinitic_data4_*.json"))


def test_sinitic_provenance_pins_source_and_records_nonredistribution():
    provenance = yaml.safe_load(
        (RELEASE / "results" / "provenance" / "reference_matrices.yaml").read_text(
            encoding="utf-8"
        )
    )
    sinitic = provenance["sincomp"]
    assert sinitic["source_commit"] == COMMIT
    assert sinitic["source_url"].endswith(f"/tree/{COMMIT}")
    assert sinitic["redistribution"]["status"] == "not_redistributed"
    assert sinitic["redistribution"]["reason"] == "upstream_license_not_specified"
    assert set(sinitic["derived_matrix_sha256"]) == {
        "city_nearest",
        "overall",
        "subgroup_aggregate",
        "subgroup_medoid",
    }


def test_release_metadata_limits_license_scope():
    third_party = (RELEASE / "THIRD_PARTY_DATA.md").read_text(encoding="utf-8")
    assert COMMIT in third_party
    assert "not redistributed" in third_party.lower()

    zenodo = json.loads((RELEASE / "zenodo.json").read_text(encoding="utf-8"))
    assert zenodo["license"] == "MIT"
    assert "third-party Sinitic_Data matrices are not included" in zenodo["description"]

    manifest = yaml.safe_load((RELEASE / "manifest.yaml").read_text(encoding="utf-8"))
    assert "third_party_reference_matrices" in manifest["excluded_classes"]


def test_public_reports_do_not_reconstruct_the_continuous_matrix():
    analysis = RELEASE / "results" / "analysis"
    architecture = json.loads(
        (analysis / "architecture_cross_loss_factorial.json").read_text(encoding="utf-8")
    )
    assert all("per_pair" not in cell for cell in architecture["cells"])

    ranking = json.loads(
        (analysis / "metric_baseline_and_ranking.json").read_text(encoding="utf-8")
    )

    def contains_key(value, key):
        if isinstance(value, dict):
            return key in value or any(contains_key(item, key) for item in value.values())
        if isinstance(value, list):
            return any(contains_key(item, key) for item in value)
        return False

    assert not contains_key(ranking, "per_pair")

    sensitivity = json.loads(
        (analysis / "reference_sensitivity_clustered.json").read_text(encoding="utf-8")
    )
    continuous = sensitivity["references"]["continuous_sinitic"]["target_distribution"]
    assert "target_histogram" not in continuous


def test_manuscript_citation_uses_the_pinned_commit_when_source_is_present():
    bibliography = ROOT / "submission" / "latex" / "references.bib"
    if not bibliography.is_file():
        return
    text = bibliography.read_text(encoding="utf-8")
    assert "@misc{yang2025sinitic" in text
    assert COMMIT in text
    assert "@misc{yang2026sinitic" not in text
