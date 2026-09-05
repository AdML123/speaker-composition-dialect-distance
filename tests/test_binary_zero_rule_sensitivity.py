import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from src.binary_zero_rule_sensitivity import (
    alternate_binary_matrix,
    matrix_digest,
    compare_zero_rule_matrices,
    build_zero_rule_report,
)
from src.reference_matrices import KESPEECH_DIALECTS, build_taxonomy_matrix


def test_alternate_rule_changes_only_symmetric_beijing_mandarin_relation():
    primary = build_taxonomy_matrix()
    alternate = alternate_binary_matrix(primary)
    assert primary["matrix"]["Beijing"]["Mandarin"] == 0.0
    assert alternate["matrix"]["Beijing"]["Mandarin"] == 1.0
    assert alternate["matrix"]["Mandarin"]["Beijing"] == 1.0
    for label in KESPEECH_DIALECTS:
        assert alternate["matrix"][label][label] == 0.0
    changed = {
        (left, right)
        for left in KESPEECH_DIALECTS
        for right in KESPEECH_DIALECTS
        if primary["matrix"][left][right] != alternate["matrix"][left][right]
    }
    assert changed == {("Beijing", "Mandarin"), ("Mandarin", "Beijing")}


def test_zero_rule_payloads_are_distinct_and_reproducible():
    primary = build_taxonomy_matrix()
    alternate = alternate_binary_matrix(primary)
    assert matrix_digest(primary) != matrix_digest(alternate)
    assert matrix_digest(primary) == matrix_digest(primary)
    comparison = compare_zero_rule_matrices(primary, alternate)
    assert comparison["changed_relations"] == ["Beijing--Mandarin"]
    assert comparison["primary_perceptual_status"] == "operational_proxy"
    assert comparison["alternate_perceptual_status"] == "operational_proxy"


def test_report_requires_both_target_identities_before_claim_branch():
    primary = build_taxonomy_matrix()
    alternate = alternate_binary_matrix(primary)
    report = build_zero_rule_report(primary, alternate, projection_rows=[])
    assert report["status"] == "construction_valid_projection_pending"
    assert report["claim_branch"] == "pending_alternate_projection_cells"
    assert report["targets"]["primary"]["perceptual_status"] == "operational_proxy"
    assert report["targets"]["alternate"]["perceptual_status"] == "operational_proxy"
