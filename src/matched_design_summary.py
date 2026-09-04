"""Summarize matched-pair strata, endpoint participation, and pair reuse."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np


QUANTILES = (0.0, 0.25, 0.5, 0.75, 0.95, 1.0)


def _unique_strings(values: Iterable[Any]) -> tuple[str, ...]:
    return tuple(dict.fromkeys(str(value) for value in values))


def _participation_counts(
    rows: Iterable[Mapping[str, Any]], field: str
) -> Counter[str]:
    counts: Counter[str] = Counter()
    for row in rows:
        counts.update(_unique_strings(row.get(field, ())))
    return counts


def _quantile_record(counts: Counter[str]) -> dict[str, float]:
    values = np.asarray(list(counts.values()), dtype=float)
    if values.size == 0:
        return {str(q): 0.0 for q in QUANTILES}
    return {
        str(q): float(np.quantile(values, q, method="linear")) for q in QUANTILES
    }


def _group_summary(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    speakers = _participation_counts(rows, "speaker_ids")
    utterances = _participation_counts(rows, "utterance_ids")
    reused = {key: value for key, value in utterances.items() if value > 1}
    return {
        "pair_count": len(rows),
        "unique_speaker_count": len(speakers),
        "unique_utterance_count": len(utterances),
        "reused_utterance_count": len(reused),
        "nonreused_utterance_count": sum(value == 1 for value in utterances.values()),
        "max_utterance_pair_count": max(utterances.values(), default=0),
        "max_speaker_pair_count": max(speakers.values(), default=0),
        "speaker_pair_participation_quantiles": _quantile_record(speakers),
    }


def _recording_condition(rows: Sequence[Mapping[str, Any]]) -> str:
    labels = sorted(
        {
            str(value)
            for row in rows
            for value in row.get("recording_conditions", ())
            if value is not None and str(value)
        }
    )
    if labels:
        return " | ".join(labels)
    stratum = str(rows[0].get("matched_stratum", "")) if rows else ""
    return stratum.rsplit("|", 1)[-1]


def summarize_matched_design(
    pairs: Iterable[Mapping[str, Any]],
) -> dict[str, Any]:
    """Return a deterministic disclosure summary for one matched-pair split."""
    rows = [dict(pair) for pair in pairs]
    strata: dict[str, list[dict[str, Any]]] = defaultdict(list)
    by_group: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        group = str(row["group"])
        stratum = str(row["matched_stratum"])
        if group not in {"A", "B"}:
            raise ValueError(f"unsupported group: {group}")
        strata[stratum].append(row)
        by_group[group].append(row)

    stratum_rows: list[dict[str, Any]] = []
    for stratum in sorted(strata):
        members = strata[stratum]
        arm_rows = {
            group: [row for row in members if row["group"] == group]
            for group in ("A", "B")
        }
        speakers = _participation_counts(members, "speaker_ids")
        utterances = _participation_counts(members, "utterance_ids")
        stratum_rows.append(
            {
                "matched_stratum": stratum,
                "recording_condition": _recording_condition(members),
                "A_pair_count": len(arm_rows["A"]),
                "B_pair_count": len(arm_rows["B"]),
                "unique_speaker_count": len(speakers),
                "unique_utterance_count": len(utterances),
                "max_speaker_pair_count": max(speakers.values(), default=0),
                "A_unique_speaker_count": len(
                    _participation_counts(arm_rows["A"], "speaker_ids")
                ),
                "B_unique_speaker_count": len(
                    _participation_counts(arm_rows["B"], "speaker_ids")
                ),
                "A_unique_utterance_count": len(
                    _participation_counts(arm_rows["A"], "utterance_ids")
                ),
                "B_unique_utterance_count": len(
                    _participation_counts(arm_rows["B"], "utterance_ids")
                ),
            }
        )

    utterances_by_group = {
        group: set(_participation_counts(by_group[group], "utterance_ids"))
        for group in ("A", "B")
    }
    return {
        "schema": "matched-design-summary-v1",
        "pair_count": len(rows),
        "stratum_count": len(strata),
        "strata": stratum_rows,
        "groups": {
            group: _group_summary(by_group[group]) for group in ("A", "B")
        },
        "cross_group_utterance_overlap_count": len(
            utterances_by_group["A"].intersection(utterances_by_group["B"])
        ),
    }


def _load_pairs(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    pairs = payload.get("pairs") if isinstance(payload, dict) else payload
    if not isinstance(pairs, list):
        raise ValueError(f"matched-pair list missing from {path}")
    return pairs


def build_disclosure(
    calibration_path: Path, evaluation_path: Path
) -> tuple[dict[str, Any], dict[str, Any]]:
    calibration = summarize_matched_design(_load_pairs(calibration_path))
    evaluation = summarize_matched_design(_load_pairs(evaluation_path))
    checks = {
        "evaluation_A_pair_count": evaluation["groups"]["A"]["pair_count"]
        == 3123,
        "evaluation_B_pair_count": evaluation["groups"]["B"]["pair_count"]
        == 9857,
        "evaluation_stratum_count": evaluation["stratum_count"] == 12,
        "evaluation_A_no_within_group_utterance_reuse": evaluation["groups"]["A"][
            "reused_utterance_count"
        ]
        == 0,
        "evaluation_B_no_within_group_utterance_reuse": evaluation["groups"]["B"][
            "reused_utterance_count"
        ]
        == 0,
    }
    status = "passed" if all(checks.values()) else "failed"
    report = {
        "schema": "matched-design-disclosure-v1",
        "status": status,
        "calibration": calibration,
        "evaluation": evaluation,
    }
    gate = {
        "schema": "matched-design-disclosure-gate-v1",
        "status": status,
        "checks": checks,
        "failure_action": "pause_before_downstream_analysis",
    }
    return report, gate


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--calibration", type=Path, required=True)
    parser.add_argument("--evaluation", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--gate", type=Path, required=True)
    args = parser.parse_args()

    report, gate = build_disclosure(args.calibration, args.evaluation)
    for path, payload in ((args.output, report), (args.gate, gate)):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
    return 0 if gate["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
