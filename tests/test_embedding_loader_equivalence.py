from __future__ import annotations

import json

import pytest

from src.embedding_loader_equivalence import audit_loader_equivalence


def _write(path, vectors):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"embeddings": vectors}), encoding="utf-8")


def test_audit_loader_equivalence_requires_exact_vector_identity(tmp_path):
    locked = tmp_path / "locked"
    strict = tmp_path / "strict"
    _write(locked / "model.json", {"u1": [1.0, 2.0], "u2": [3.0, 4.0]})
    _write(strict / "model.json", {"u1": [1.0, 2.0], "u2": [3.0, 4.0]})

    report = audit_loader_equivalence(locked, strict, tmp_path / "audit.json")

    assert report["status"] == "passed"
    assert report["models"][0]["maximum_absolute_difference"] == 0.0


def test_audit_loader_equivalence_fails_on_any_numeric_change(tmp_path):
    locked = tmp_path / "locked"
    strict = tmp_path / "strict"
    _write(locked / "model.json", {"u1": [1.0, 2.0]})
    _write(strict / "model.json", {"u1": [1.0, 2.000001]})

    with pytest.raises(ValueError, match="differ"):
        audit_loader_equivalence(locked, strict, tmp_path / "audit.json")
