"""Identifiability gate for speaker-by-dialect variance decomposition."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence


def audit_variance_identifiability(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    if not records:
        raise ValueError("records are required")
    cells = Counter((str(row["speaker_id"]), str(row["dialect_label"])) for row in records)
    condition_by_dialect: dict[str, set[str]] = defaultdict(set)
    dialect_by_condition: dict[str, set[str]] = defaultdict(set)
    for row in records:
        dialect = str(row["dialect_label"])
        condition = str(row.get("recording_condition", "missing"))
        condition_by_dialect[dialect].add(condition)
        dialect_by_condition[condition].add(dialect)
    singleton = sorted([{"speaker_id": speaker, "dialect_label": dialect, "count": count} for (speaker, dialect), count in cells.items() if count < 2], key=lambda row: (row["speaker_id"], row["dialect_label"]))
    dialect_condition_aliases = sorted(dialect for dialect, conditions in condition_by_dialect.items() if len(conditions) == 1 and len(dialect_by_condition[next(iter(conditions))]) == 1)
    repeated_support = not singleton
    condition_separable = not dialect_condition_aliases
    passed = repeated_support and condition_separable
    return {
        "schema": "speaker-dialect-variance-identifiability-v1",
        "status": "passed" if passed else "failed",
        "record_count": len(records),
        "speaker_count": len({str(row["speaker_id"]) for row in records}),
        "dialect_count": len({str(row["dialect_label"]) for row in records}),
        "speaker_dialect_cell_count": len(cells),
        "singleton_cell_count": len(singleton),
        "singleton_cells": singleton,
        "dialect_condition_aliases": dialect_condition_aliases,
        "criteria": {"all_speaker_dialect_cells_repeated": repeated_support, "dialect_not_fully_aliased_with_condition": condition_separable},
        "selected_wording": "factorial bookkeeping notation" if not passed else "identified speaker, dialect, interaction, and utterance variance components",
        "reason": "speaker-by-dialect interaction cannot be separated from utterance residuals in singleton cells" if singleton else ("dialect is aliased with recording condition" if dialect_condition_aliases else "identifiable"),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--records", required=True)
    parser.add_argument("--split", default="calibration")
    parser.add_argument("--output", required=True)
    parser.add_argument("--gate-output", required=True)
    args = parser.parse_args()
    payload = json.loads(Path(args.records).read_text(encoding="utf-8"))
    records = [row for row in payload.get("records", payload) if str(row.get("split")) == args.split]
    report = audit_variance_identifiability(records)
    target = Path(args.output); target.parent.mkdir(parents=True, exist_ok=True); target.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    gate = {"schema": "speaker-dialect-variance-gate-v1", "status": report["status"], "identifiability": report["criteria"], "reason": report["reason"], "selected_wording": report["selected_wording"], "variance_components_reported": report["status"] == "passed"}
    gate_target = Path(args.gate_output); gate_target.parent.mkdir(parents=True, exist_ok=True); gate_target.write_text(json.dumps(gate, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
