"""Refresh target-prevalence slope inference from stored five-seed cells."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from .run_target_prevalence_mechanism import build_slope_rows, seed_slope_distribution
from .target_prevalence_experiment import (
    TARGET_PREVALENCE_GRID,
    clustered_mean_gain_contrast,
    clustered_slope_bootstrap,
)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", required=True)
    parser.add_argument("--gate", required=True)
    args = parser.parse_args(argv)
    report_path = Path(args.report)
    report = json.loads(report_path.read_text(encoding="utf-8"))
    internal_q = list(report.get("internal_q_cells") or [format(q, ".2f") for q in TARGET_PREVALENCE_GRID if 0 < q < 1])
    for arm_name in ("fixed_pair", "natural_pool"):
        arm = report[arm_name]
        slope_rows = build_slope_rows(arm["q_cells"], internal_q)
        arm["slope_contrast"] = clustered_slope_bootstrap(
            slope_rows,
            seed=20260829,
            replicates=1000,
        )
        arm["mean_gain_contrast"] = clustered_mean_gain_contrast(slope_rows, seed=20260829, replicates=1000)
        arm["seed_slope_distribution"] = seed_slope_distribution(arm["q_cells"], internal_q)
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    fixed_supported = report["fixed_pair"]["slope_contrast"]["ci"]["lower"] > 0 and report["fixed_pair"]["seed_slope_distribution"]["all_positive"]
    gate = {
        "schema": "target-prevalence-mechanism-gate-v1",
        "status": "evaluated",
        "fixed_pair_association_supported": fixed_supported,
        "natural_pool_ecological_only": True,
        "balanced_slope_contrast": report["fixed_pair"]["slope_contrast"],
        "mean_gain_contrast": report["fixed_pair"]["mean_gain_contrast"],
        "seed_slope_distribution": report["fixed_pair"]["seed_slope_distribution"],
        "decision": "association_only_unless_balanced_slope_and_process_chain_pass",
        "q_grid": list(TARGET_PREVALENCE_GRID),
        "inference": "seed-paired speaker-cluster means within q, 1000 cluster bootstrap replicates",
    }
    gate_path = Path(args.gate)
    gate_path.parent.mkdir(parents=True, exist_ok=True)
    gate_path.write_text(json.dumps(gate, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
