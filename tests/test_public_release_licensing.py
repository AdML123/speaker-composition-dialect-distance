from pathlib import Path

import json
import yaml


ROOT = Path(__file__).resolve().parents[1]
RELEASE = ROOT / "release" / "github" if (ROOT / "release" / "github").is_dir() else ROOT
COMMIT = "820b9d15a74cbee82109f0bf54cf791fe16596ef"


def test_release_excludes_manuscript_pdf_and_latex_content():
    assert not list(RELEASE.rglob("*.tex"))
    assert not list(RELEASE.rglob("*.pdf"))


def test_release_contains_sparse_stratum_and_dual_endpoint_audits():
    required = (
        "results/analysis/speaker_effect_support_sensitivity.json",
        "results/analysis/speaker_effect_dependency_sensitivity.json",
        "results/analysis/estimand_weighting_intervals.json",
        "results/analysis/relation_ranking_clustered.json",
        "results/pairs/kespeech_projection_evaluation_summary.json",
        "results/gates/sparse_stratum_dual_endpoint_gate.json",
    )
    assert all((RELEASE / path).is_file() for path in required)


def test_release_text_has_no_local_paths_or_secret_names():
    forbidden = (
        "c:" + "/users/",
        "d:" + "/paper48",
        "e:" + "/paper48",
        "access" + "token.txt",
        "key" + ".txt",
    )
    for path in RELEASE.rglob("*"):
        if any(part.startswith(".pytest") or part == "__pycache__" for part in path.parts):
            continue
        if not path.is_file() or path.name == "MANIFEST.sha256":
            continue
        try:
            text = path.read_text(encoding="utf-8").lower()
        except UnicodeDecodeError:
            continue
        assert not any(token in text for token in forbidden), path


def test_unlicensed_sinitic_matrices_are_not_redistributed():
    reference_dir = RELEASE / "results" / "references"
    assert (reference_dir / "taxonomy_matrix.json").is_file()
    assert not list(reference_dir.glob("sinitic_data4_*.json"))


def test_release_excludes_upstream_sinitic_archives_and_matrix_payloads():
    forbidden_suffixes = {".zip", ".tar", ".gz", ".tgz", ".npz", ".npy", ".pkl"}
    for path in RELEASE.rglob("*"):
        if not path.is_file():
            continue
        assert path.suffix.lower() not in forbidden_suffixes, path
        lower_name = path.name.lower()
        assert "sinitic_data4" not in lower_name, path

    provenance = yaml.safe_load(
        (RELEASE / "results/provenance/reference_matrices.yaml").read_text(encoding="utf-8")
    )["sincomp"]
    assert "matrix" not in provenance
    assert all("private_artifact" not in item for item in provenance["mappings"].values())


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
    assert "https://zhongguoyuyan.cn/index" in third_party
    assert all(value in third_party for value in ("1,289", "999", "1,084", "915", "overall_distance"))

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
