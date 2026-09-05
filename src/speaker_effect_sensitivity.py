"""Support-weighted sensitivity analysis for the matched A/B phenomenon."""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
from collections import Counter, defaultdict
from pathlib import Path
from statistics import median
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import yaml

from .dyadic_bootstrap import endpoint_multiplicity
from .statistical_gate import holm_adjust
from .speaker_support_sensitivity import build_support_report


def overlap_pair_weight(n_a: int, n_b: int) -> float:
    """Return the overlap-support weight n_A n_B / (n_A + n_B)."""
    if n_a <= 0 or n_b <= 0:
        raise ValueError("both arm counts must be positive")
    return float(n_a * n_b / (n_a + n_b))


def _checked_rows(rows: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
    checked = [dict(row) for row in rows]
    if not checked:
        raise ValueError("at least one row is required")
    for row in checked:
        if row.get("group") not in {"A", "B"}:
            raise ValueError("groups must be A or B")
        if not row.get("matched_stratum"):
            raise ValueError("matched_stratum is required")
        speakers = row.get("speaker_ids")
        if not isinstance(speakers, list) or not speakers:
            raise ValueError("speaker_ids are required")
        if any(str(speaker).strip() == "" for speaker in speakers):
            raise ValueError("speaker_ids cannot be empty")
        row["speaker_ids"] = [str(speaker) for speaker in speakers]
        row["distance"] = float(row["distance"])
    return checked


def _group_rows(
    rows: Sequence[Mapping[str, Any]],
) -> dict[str, dict[str, list[Mapping[str, Any]]]]:
    grouped: dict[str, dict[str, list[Mapping[str, Any]]]] = defaultdict(
        lambda: {"A": [], "B": []}
    )
    for row in rows:
        grouped[str(row["matched_stratum"])][str(row["group"])].append(row)
    missing = [
        stratum
        for stratum, arms in grouped.items()
        if not arms["A"] or not arms["B"]
    ]
    if missing:
        raise ValueError(f"every stratum must contain both A and B arms: {missing}")
    return dict(grouped)


def stratum_effects(rows: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Summarize arm support and median B-minus-A effect by stratum."""
    checked = _checked_rows(rows)
    grouped = _group_rows(checked)
    report = []
    for stratum in sorted(grouped):
        arms = grouped[stratum]
        a_distances = [float(row["distance"]) for row in arms["A"]]
        b_distances = [float(row["distance"]) for row in arms["B"]]
        a_median = float(median(a_distances))
        b_median = float(median(b_distances))
        weight = overlap_pair_weight(len(a_distances), len(b_distances))
        effect = b_median - a_median
        report.append(
            {
                "stratum": stratum,
                "A_pair_count": len(a_distances),
                "B_pair_count": len(b_distances),
                "A_unique_speaker_count": len(
                    {speaker for row in arms["A"] for speaker in row["speaker_ids"]}
                ),
                "B_unique_speaker_count": len(
                    {speaker for row in arms["B"] for speaker in row["speaker_ids"]}
                ),
                "A_median": a_median,
                "B_median": b_median,
                "effect": effect,
                "overlap_weight": weight,
                "weighted_effect_contribution": weight * effect,
            }
        )
    return report


def _overlap_aggregate(summaries: Sequence[Mapping[str, Any]]) -> float:
    total_weight = sum(float(row["overlap_weight"]) for row in summaries)
    if total_weight <= 0:
        raise ValueError("overlap aggregate has no positive support")
    return float(
        sum(
            float(row["effect"]) * float(row["overlap_weight"])
            for row in summaries
        )
        / total_weight
    )


def _legacy_median(summaries: Sequence[Mapping[str, Any]]) -> float:
    if not summaries:
        raise ValueError("equal-stratum estimate has no strata")
    return float(median(float(row["effect"]) for row in summaries))


def speaker_effect_sensitivity(
    rows: Iterable[Mapping[str, Any]], *, excluded_stratum: str
) -> dict[str, Any]:
    """Compute the locked headline and mandatory deterministic sensitivities."""
    checked = _checked_rows(rows)
    summaries = stratum_effects(checked)
    names = {str(row["stratum"]) for row in summaries}
    if excluded_stratum not in names:
        raise ValueError(f"excluded stratum is absent: {excluded_stratum}")

    without_excluded = [
        row for row in summaries if row["stratum"] != excluded_stratum
    ]
    leave_one_out = []
    for omitted in sorted(names):
        kept = [row for row in summaries if row["stratum"] != omitted]
        leave_one_out.append(
            {
                "omitted_stratum": omitted,
                "estimate": _overlap_aggregate(kept),
                "remaining_strata": len(kept),
            }
        )

    qualified = [
        row
        for row in summaries
        if int(row["A_unique_speaker_count"]) >= 2
        and int(row["B_unique_speaker_count"]) >= 2
    ]
    excluded_qualified = sorted(names - {str(row["stratum"]) for row in qualified})
    pooled_a = [float(row["distance"]) for row in checked if row["group"] == "A"]
    pooled_b = [float(row["distance"]) for row in checked if row["group"] == "B"]

    return {
        "schema": "speaker-effect-estimand-sensitivity-v1",
        "strata": summaries,
        "primary": {
            "estimand": "overlap_pair_weighted_stratum_median_B_minus_A",
            "weight_formula": "n_A*n_B/(n_A+n_B)",
            "estimate": _overlap_aggregate(summaries),
            "stratum_count": len(summaries),
        },
        "legacy_equal_stratum": {
            "estimand": "median_of_stratum_median_B_minus_A",
            "estimate": _legacy_median(summaries),
            "status": "sensitivity_only",
        },
        "excluded_stratum": {
            "estimand": "overlap_pair_weighted_stratum_median_B_minus_A",
            "excluded": excluded_stratum,
            "estimate": _overlap_aggregate(without_excluded),
            "legacy_equal_stratum_estimate": _legacy_median(without_excluded),
        },
        "leave_one_stratum_out": leave_one_out,
        "pooled_pair_weighted": {
            "estimand": "pooled_pair_median_B_minus_median_A",
            "estimate": float(median(pooled_b) - median(pooled_a)),
            "status": "descriptive_only",
        },
        "support_qualified_equal_stratum": {
            "estimand": "median_of_stratum_median_B_minus_A",
            "minimum_unique_speakers_per_arm": 2,
            "estimate": _legacy_median(qualified),
            "included_strata": [row["stratum"] for row in qualified],
            "excluded_strata": excluded_qualified,
            "status": "sensitivity_only",
        },
    }


def resolve_excluded_stratum(
    conceptual_stratum: str,
    rows: Iterable[Mapping[str, Any]],
    amendment: Mapping[str, Any] | None,
) -> str:
    """Resolve a source identifier only through an explicit locked amendment."""
    available = {str(row.get("matched_stratum", "")) for row in rows}
    if conceptual_stratum in available:
        return conceptual_stratum
    if amendment is None:
        raise ValueError(
            "source stratum differs from the protocol; an explicit amendment is required"
        )
    if amendment.get("schema") != "sparse-stratum-protocol-amendment-v1":
        raise ValueError("unexpected protocol amendment schema")
    if amendment.get("status") != "locked_before_successful_analysis":
        raise ValueError("protocol amendment is not locked")
    if amendment.get("changes_estimand") is not False:
        raise ValueError("an identifier amendment cannot change the estimand")
    if amendment.get("conceptual_stratum") != conceptual_stratum:
        raise ValueError("protocol amendment conceptual stratum does not match")
    source_stratum = str(amendment.get("source_stratum", ""))
    if source_stratum not in available:
        raise ValueError("amended source stratum is absent")
    return source_stratum


def _weighted_median(values: Sequence[float], weights: Sequence[int]) -> float:
    ordered = sorted(
        (float(value), int(weight))
        for value, weight in zip(values, weights)
        if int(weight) > 0
    )
    if not ordered:
        raise ValueError("weighted median has no positive support")
    threshold = sum(weight for _, weight in ordered) / 2.0
    cumulative = 0
    for value, weight in ordered:
        cumulative += weight
        if cumulative >= threshold:
            return value
    return ordered[-1][0]


def global_dyadic_speaker_bootstrap(
    rows: Sequence[Mapping[str, Any]],
    *,
    seed: int,
    replicates: int,
    excluded_stratum: str | None = None,
) -> dict[str, Any]:
    if replicates <= 0:
        raise ValueError("replicates must be positive")
    checked = _checked_rows(rows)
    grouped = _group_rows(checked)
    if excluded_stratum is not None:
        if excluded_stratum not in grouped:
            raise ValueError(f"excluded stratum is absent: {excluded_stratum}")
        grouped = {
            stratum: arms
            for stratum, arms in grouped.items()
            if stratum != excluded_stratum
        }
    fixed_weights = {
        stratum: overlap_pair_weight(len(arms["A"]), len(arms["B"]))
        for stratum, arms in grouped.items()
    }
    speakers = sorted(
        {
            str(speaker)
            for arms in grouped.values()
            for arm_rows in arms.values()
            for row in arm_rows
            for speaker in row["speaker_ids"]
        }
    )
    rng = np.random.default_rng(seed)
    estimates: list[float] = []
    missing_stratum_counts: Counter[str] = Counter()

    for _ in range(replicates):
        sampled = rng.choice(speakers, size=len(speakers), replace=True)
        multiplicities = Counter(map(str, sampled.tolist()))
        replicate_effects = []
        for stratum, arms in grouped.items():
            medians: dict[str, float] = {}
            for arm in ("A", "B"):
                values = []
                weights = []
                for row in arms[arm]:
                    weight = endpoint_multiplicity(row["speaker_ids"], multiplicities)
                    if weight > 0:
                        values.append(float(row["distance"]))
                        weights.append(weight)
                if values:
                    medians[arm] = _weighted_median(values, weights)
            if set(medians) == {"A", "B"}:
                replicate_effects.append(
                    (medians["B"] - medians["A"], fixed_weights[stratum])
                )
            else:
                missing_stratum_counts[stratum] += 1
        if not replicate_effects:
            continue
        total_weight = sum(weight for _, weight in replicate_effects)
        estimates.append(
            sum(effect * weight for effect, weight in replicate_effects) / total_weight
        )

    if not estimates:
        raise ValueError("all bootstrap replicates lost every matched stratum")
    lower, upper = np.quantile(estimates, [0.025, 0.975], method="linear")
    return {
        "ci": {
            "lower": float(lower),
            "upper": float(upper),
            "confidence_level": 0.95,
            "method": "percentile",
        },
        "replicates_requested": replicates,
        "replicates_used": len(estimates),
        "seed": seed,
        "resampling_unit": "speaker_id_global_across_arms_and_strata",
        "same_speaker_pair_weight": "one_speaker_multiplicity",
        "different_speaker_pair_weight": "product_of_endpoint_multiplicities",
        "stratum_weights": "fixed_observed_overlap_pair_weights",
        "stratum_medians_recomputed_per_replicate": True,
        "bootstrap_estimate_standard_deviation": float(
            np.std(estimates, ddof=1) if len(estimates) > 1 else 0.0
        ),
        "missing_stratum_replicates": dict(sorted(missing_stratum_counts.items())),
    }


def exact_weighted_sign_flip(
    summaries: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Enumerate the two-sided sign-flip null over matched stratum effects."""
    if not summaries or any("effect" not in row or "overlap_weight" not in row for row in summaries):
        raise ValueError("stratum effect and overlap_weight are required")
    effects = np.asarray([float(row["effect"]) for row in summaries], dtype=float)
    weights = np.asarray([float(row["overlap_weight"]) for row in summaries], dtype=float)
    observed = float(np.average(effects, weights=weights))
    null = []
    for signs in itertools.product((-1.0, 1.0), repeat=len(effects)):
        null.append(float(np.average(effects * np.asarray(signs), weights=weights)))
    tolerance = 1e-15
    extreme = sum(abs(value) >= abs(observed) - tolerance for value in null)
    return {
        "method": "exact_weighted_stratum_sign_flip",
        "sidedness": "two_sided",
        "permutations": len(null),
        "enumeration_family_size": len(null),
        "exchangeability_unit": "stratum_effect_sign",
        "sharp_null": "stratum signs exchangeable at fixed magnitudes and weights",
        "tail_resolution": 2.0 / len(null),
        "observed": observed,
        "p_value": extreme / len(null),
    }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_protocol(path: Path) -> dict[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if payload.get("schema") != "sparse-stratum-dual-endpoint-protocol-v1":
        raise ValueError("unexpected protocol schema")
    if payload.get("status") != "locked_before_new_runs":
        raise ValueError("protocol is not locked")
    for source, expected in payload.get("source_hashes", {}).items():
        source_path = Path(source)
        if source_path.exists() and _sha256(source_path) != expected:
            raise ValueError(f"source hash drift: {source}")
    return payload


def _load_amendment(path: Path) -> dict[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if payload.get("schema") != "sparse-stratum-protocol-amendment-v1":
        raise ValueError("unexpected protocol amendment schema")
    if payload.get("status") != "locked_before_successful_analysis":
        raise ValueError("protocol amendment is not locked")
    return payload


def build_report(
    protocol_path: Path,
    amendment_path: Path,
    pair_path: Path,
    distance_dir: Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    protocol = _load_protocol(protocol_path)
    amendment = _load_amendment(amendment_path)
    recorded_protocol = amendment.get("protocol", {})
    if recorded_protocol.get("sha256") != _sha256(protocol_path):
        raise ValueError("amendment does not match the locked protocol hash")
    pairs = json.loads(pair_path.read_text(encoding="utf-8"))
    pair_ids = {str(row["pair_id"]) for row in pairs["pairs"]}
    conceptual_excluded = str(
        protocol["mandatory_sensitivities"]["exclude_stratum"]
    )
    replicates = int(protocol["bootstrap_replicates"])
    seed = int(protocol["seeds"][0])
    model_order = [
        str(protocol["primary_phenomenon"]["model"]),
        *map(str, protocol["supporting_phenomenon_family"]["models"]),
    ]
    models = []
    for model_name in model_order:
        distance_path = distance_dir / f"{model_name}.json"
        payload = json.loads(distance_path.read_text(encoding="utf-8"))
        rows = payload["distances"]
        distance_pair_ids = {str(row["pair_id"]) for row in rows}
        if distance_pair_ids != pair_ids:
            raise ValueError(f"pair identity mismatch for {model_name}")
        excluded = resolve_excluded_stratum(conceptual_excluded, rows, amendment)
        deterministic = speaker_effect_sensitivity(rows, excluded_stratum=excluded)
        primary_bootstrap = global_dyadic_speaker_bootstrap(
            rows, seed=seed, replicates=replicates
        )
        excluded_bootstrap = global_dyadic_speaker_bootstrap(
            rows,
            seed=seed,
            replicates=replicates,
            excluded_stratum=excluded,
        )
        sign_flip = exact_weighted_sign_flip(deterministic["strata"])
        models.append(
            {
                "model_name": model_name,
                "source_file": str(distance_path),
                "source_sha256": _sha256(distance_path),
                **deterministic,
                "primary_bootstrap": primary_bootstrap,
                "excluded_stratum_bootstrap": excluded_bootstrap,
                "sign_flip": sign_flip,
                "speaker_support": build_support_report(
                    rows, excluded_stratum=excluded
                ),
            }
        )

    primary_name = str(protocol["primary_phenomenon"]["model"])
    primary = next(row for row in models if row["model_name"] == primary_name)
    hubert_supported = (
        primary["primary_bootstrap"]["ci"]["lower"] > 0
        and primary["excluded_stratum_bootstrap"]["ci"]["lower"] > 0
        and all(
            row["estimate"] > 0 for row in primary["leave_one_stratum_out"]
        )
    )
    supporting = [row for row in models if row["model_name"] != primary_name]
    raw_supporting_p = {
        row["model_name"]: float(row["sign_flip"]["p_value"])
        for row in supporting
    }
    adjusted = holm_adjust(raw_supporting_p)
    for row in supporting:
        row["sign_flip"]["holm_adjusted_p"] = float(adjusted[row["model_name"]])
    supporting_pass = all(
        row["primary"]["estimate"] > 0
        and row["primary_bootstrap"]["ci"]["lower"] > 0
        and adjusted[row["model_name"]] < 0.05
        for row in supporting
    )
    if hubert_supported and supporting_pass:
        branch = "three_encoder_supported"
    elif hubert_supported:
        branch = "hubert_only_supported"
    else:
        branch = "descriptive_audit_only"

    report = {
        "schema": "speaker-effect-support-sensitivity-v1",
        "status": "evaluated",
        "declaration_status": "reviewer_driven_revision_not_preregistered",
        "protocol": str(protocol_path),
        "protocol_sha256": _sha256(protocol_path),
        "protocol_amendment": str(amendment_path),
        "protocol_amendment_sha256": _sha256(amendment_path),
        "conceptual_excluded_stratum": conceptual_excluded,
        "source_excluded_stratum": amendment["source_stratum"],
        "pair_manifest": str(pair_path),
        "pair_manifest_sha256": _sha256(pair_path),
        "models": models,
        "multiplicity": {
            "primary_model": primary_name,
            "supporting_family": [row["model_name"] for row in supporting],
            "adjustment": "holm",
        },
    }
    gate = {
        "schema": "speaker-effect-support-gate-v1",
        "status": "passed" if branch != "descriptive_audit_only" else "failed",
        "claim_branch": branch,
        "hubert_support_gate": hubert_supported,
        "supporting_family_gate": supporting_pass,
        "supporting_holm_adjusted_p": adjusted,
        "utterance_block_dependency": "pending_task_2",
        "full_title_ready": False,
        "reason": (
            "Statistical support branch is provisional until the locked "
            "utterance-block dependency sensitivity is evaluated."
        ),
    }
    return report, gate


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--amendment", type=Path, required=True)
    parser.add_argument("--pairs", type=Path, required=True)
    parser.add_argument("--distance-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--gate-output", type=Path, required=True)
    args = parser.parse_args()
    report, gate = build_report(
        args.protocol, args.amendment, args.pairs, args.distance_dir
    )
    for path, payload in ((args.output, report), (args.gate_output, gate)):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
