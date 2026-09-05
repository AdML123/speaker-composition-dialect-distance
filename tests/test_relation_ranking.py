from itertools import combinations

import pytest

from src.relation_ranking import (
    _sha256,
    aggregate_relation_predictions,
    build_practical_gate,
    canonical_relation,
    clustered_ranking_bootstrap,
    ordering_changes,
    ranking_metrics,
    validate_complete_clustered_report,
)


def test_sha256_records_the_exact_manifest_identity(tmp_path):
    manifest = tmp_path / "pairs.json"
    manifest.write_bytes(b'{"pairs":[]}\n')
    assert _sha256(manifest) == "fb116a3f11e2f1150d5192752159e65a35a2cadf20a3f6ce7a90b04044df47d4"


def test_relation_is_unordered():
    assert canonical_relation(["Ji-Lu", "Beijing"]) == ("Beijing", "Ji-Lu")


def test_perfect_relation_ranking_scores_one():
    reference = {("A", "B"): 0.2, ("A", "C"): 0.8, ("B", "C"): 0.5}
    metrics = ranking_metrics(reference, dict(reference))
    assert metrics["spearman"] == pytest.approx(1.0)
    assert metrics["kendall_tau_b"] == pytest.approx(1.0)
    assert metrics["pairwise_order_accuracy"] == pytest.approx(1.0)
    assert metrics["ties"] == 0


def test_ranking_metrics_report_reversed_order():
    reference = {("A", "B"): 0.2, ("A", "C"): 0.8, ("B", "C"): 0.5}
    prediction = {("A", "B"): 0.8, ("A", "C"): 0.2, ("B", "C"): 0.5}
    metrics = ranking_metrics(reference, prediction)
    assert metrics["pairwise_order_accuracy"] == pytest.approx(0.0)
    assert metrics["order_reversals"] == 3


def test_diagonal_relations_are_rejected():
    with pytest.raises(ValueError, match="off-diagonal"):
        ranking_metrics({("A", "A"): 0.0}, {("A", "A"): 0.0})


def test_ordering_changes_count_reversals_against_baseline():
    baseline = {("A", "B"): 0.2, ("A", "C"): 0.8, ("B", "C"): 0.5}
    method = {("A", "B"): 0.8, ("A", "C"): 0.2, ("B", "C"): 0.5}
    changes = ordering_changes(baseline, method)
    assert changes["reversals_vs_baseline"] == 3
    assert changes["ties_in_either_ordering"] == 0


def test_reference_ties_are_excluded_and_predicted_ties_receive_half_credit():
    reference = {("A", "B"): 0.2, ("A", "C"): 0.2, ("B", "C"): 0.8}
    prediction = {("A", "B"): 0.1, ("A", "C"): 0.4, ("B", "C"): 0.4}
    metrics = ranking_metrics(reference, prediction)
    assert metrics["reference_ties_excluded"] == 1
    assert metrics["predicted_ties_half_credit"] == 1
    assert metrics["comparable_relation_pairs"] == 2
    assert metrics["pairwise_order_accuracy"] == pytest.approx(0.75)


def test_aggregation_covers_36_unordered_relations_for_nine_labels():
    labels = list("ABCDEFGHI")
    rows = [
        {
            "dialect_labels": [left, right],
            "predicted_distance": index / 36,
        }
        for index, (left, right) in enumerate(combinations(labels, 2), start=1)
    ]
    assert len(aggregate_relation_predictions(rows)) == 36


def _bootstrap_rows(reverse: bool = False):
    relations = [("A", "B"), ("A", "C"), ("B", "C")]
    values = [0.2, 0.8, 0.5]
    if reverse:
        values = [0.8, 0.2, 0.5]
    return [
        {
            "pair_id": f"p{index}",
            "speaker_ids": [f"s{index}", f"t{index}"],
            "utterance_ids": [f"u{index}", f"v{index}"],
            "dialect_labels": list(relation),
            "predicted_distance": value,
            "absolute_error": abs(value - target),
        }
        for index, (relation, value, target) in enumerate(
            zip(relations, values, [0.2, 0.8, 0.5]), start=1
        )
    ]


def test_clustered_ranking_bootstrap_is_paired_and_deterministic():
    methods = {
        "linear": [{"seed": 1, "rows": _bootstrap_rows(reverse=True)}],
        "mlp": [{"seed": 1, "rows": _bootstrap_rows(reverse=False)}],
    }
    reference = {("A", "B"): 0.2, ("A", "C"): 0.8, ("B", "C"): 0.5}
    first = clustered_ranking_bootstrap(
        methods, reference, seed=9, replicates=100,
        contrasts=[("mlp", "linear")], required_relation_count=3,
    )
    second = clustered_ranking_bootstrap(
        methods, reference, seed=9, replicates=100,
        contrasts=[("mlp", "linear")], required_relation_count=3,
    )
    assert first == second
    assert first["resampling_unit"] == "global_endpoint_speaker"
    assert first["shared_resamples_across_methods"] is True
    assert set(first["methods"]["linear"]["ordering"]) == {
        "spearman", "kendall_tau_b", "pairwise_order_accuracy"
    }
    contrast = first["contrasts"]["mlp_minus_linear"]
    assert contrast["spearman"]["point_estimate"] > 0
    assert contrast["spearman"]["ci"]["lower"] > 0


def test_clustered_report_requires_all_methods_and_nonempty_variant_contrasts():
    metric = {"point_estimate": 0.1, "ci": {"lower": 0.0, "upper": 0.2}}
    method = {
        "mae": metric,
        "ordering": {
            "spearman": metric,
            "kendall_tau_b": metric,
            "pairwise_order_accuracy": metric,
        },
    }
    methods = {
        name: method
        for name in (
            "frozen_affine", "principal_component", "diagonal_metric",
            "linear", "matched_mlp", "wide_mlp",
        )
    }
    ordering_contrast = {
        name: metric
        for name in (
            "spearman", "kendall_tau_b", "pairwise_order_accuracy"
        )
    }
    report = {
        "references": {
            reference: {
                "methods": methods,
                "contrasts": {
                    "matched_mlp_minus_linear": ordering_contrast,
                    "linear_minus_matched_mlp": {"mae": metric},
                },
            }
            for reference in (
                "taxonomy", "city_nearest", "subgroup_medoid", "subgroup_aggregate"
            )
        }
    }
    validate_complete_clustered_report(report)
    report["references"]["subgroup_aggregate"]["contrasts"] = {}
    with pytest.raises(ValueError, match="contrast"):
        validate_complete_clustered_report(report)


def test_practical_gate_separates_ordering_and_mae_interval_support():
    ordering = {
        reference: {
            metric: True
            for metric in ("spearman", "kendall_tau_b", "pairwise_order_accuracy")
        }
        for reference in ("taxonomy", "city_nearest")
    }
    mae = {"taxonomy": True, "city_nearest": False}

    gate = build_practical_gate(ordering, mae)

    assert gate["ordering_contrast_support"] == ordering
    assert gate["mae_contrast_support"] == mae
    assert "point estimates" in gate["selected_wording"]
