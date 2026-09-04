import pytest

from src.relation_ranking import canonical_relation, ordering_changes, ranking_metrics


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
