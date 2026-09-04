# Reproduction routes

## Route A: audit released evidence without restricted inputs

From the repository root:

~~~bash
python -m pip install -r requirements.txt
python -m pytest tests -q
~~~

These commands validate pair-design summaries, dyadic and weighted estimands,
reference construction, paired randomness, architecture-factorial contracts,
ranking metrics, and endpoint semantics using synthetic fixtures and the
released report files.

The following command rebuilds the matched-strata summary from the released
sanitized pair manifests:

~~~bash
python -m src.matched_design_summary \
  --calibration results/pairs/kespeech_calibration_matched.json \
  --evaluation results/pairs/kespeech_evaluation_matched.json \
  --output results/pairs/kespeech_matched_strata_summary.rebuilt.json \
  --gate results/gates/matched_design_disclosure_gate.rebuilt.json
~~~

The release intentionally ships the locked aggregate analysis JSON used to
support the article. Manuscript, PDF, LaTeX, TikZ, and table sources are not
distributed. Public copies are passed through `src.sanitize_public_release`: aggregate
errors, intervals, seed summaries, and ordering metrics remain, but per-pair
continuous targets and their exact histogram are removed because they could
reconstruct the excluded third-party-derived matrix.

## Route B: full rerun with licensed KeSpeech, Sinitic_Data, and local models

1. Obtain KeSpeech and accept its license.
2. Obtain Sinitic_Data commit `820b9d15a74cbee82109f0bf54cf791fe16596ef`
   from its provider. The upstream repository does not state a redistribution
   license, so neither its archive nor our derived continuous matrices are
   included in this release.
3. Verify the downloaded commit archive SHA-256 is
   `621c8bce1fc49e5bc8e05103c810d675447c6d43d572bac3334bdff385a74692`.
   The analysis-time `main` archive has SHA-256
   `020f399824a9e4073f4092078099b5c9b6a995f1ef7d6eae6887f6b1250923c5`;
   all 37 extracted files match the pinned commit archive byte-for-byte.
4. Place data and model files outside this repository.
5. Replace MODEL_CACHE in configs/experiment.yaml with the local model root.
6. Build the transcript-hashed corpus manifest.
7. Recreate the matched pair manifests and extract frozen embeddings.
8. Run the estimand, reference, architecture, baseline, and mechanism commands.

Representative command forms are:

~~~bash
python -m src.prepare_kespeech_manifest \
  --metadata-root <KESPEECH_ROOT>/Metadata \
  --output <WORK_ROOT>/kespeech_manifest.json

python -m src.matched_design_summary \
  --calibration <WORK_ROOT>/kespeech_calibration_matched.json \
  --evaluation <WORK_ROOT>/kespeech_evaluation_matched.json \
  --output <WORK_ROOT>/kespeech_matched_strata_summary.json \
  --gate <WORK_ROOT>/matched_design_disclosure_gate.json

python -m src.estimand_sensitivity \
  --speaker-effect <WORK_ROOT>/speaker_effect_dyadic.json \
  --projection-report <WORK_ROOT>/projection_report.json \
  --output <WORK_ROOT>/estimand_weighting_sensitivity.json \
  --gate <WORK_ROOT>/estimand_weighting_gate.json

python -m src.reference_representative_sensitivity \
  --source-archive <SINITIC_DATA_ARCHIVE> \
  --current-provenance results/provenance/reference_matrices.yaml \
  --output-dir <WORK_ROOT>/reference_variants \
  --report <WORK_ROOT>/reference_representative_sensitivity.json \
  --gate <WORK_ROOT>/reference_representative_gate.json

python -m src.run_reference_variant_sweep \
  --config configs/experiment.yaml \
  --matrices <WORK_ROOT>/reference_variants \
  --output <WORK_ROOT>/reference_variant_sweep.json \
  --gate <WORK_ROOT>/reference_variant_sweep_gate.json

python -m src.architecture_factorial \
  --config configs/experiment.yaml \
  --protocol configs/architecture_factorial.yaml \
  --output <WORK_ROOT>/architecture_cross_loss_factorial.json \
  --gate <WORK_ROOT>/architecture_factorial_gate.json \
  --checkpoint-root <WORK_ROOT>/checkpoints

python -m src.metric_baselines \
  --config configs/experiment.yaml \
  --architecture-report <WORK_ROOT>/architecture_cross_loss_factorial.json \
  --output <WORK_ROOT>/metric_baseline_and_ranking.json \
  --gate <WORK_ROOT>/practical_consequence_gate.json
~~~

The full rerun also invokes the mechanism modules
src.target_permutation_control, src.run_target_prevalence_mechanism, and
src.run_pool_ratio_gradient_budget using paths declared in the local working
configuration. Their released reports are audit references, not substitutes
for restricted embeddings.

## Estimand and privacy notes

The pair manifests contain KeSpeech record identifiers, pseudonymous speaker
identifiers, transcript hashes, condition labels, and source-record hashes.
They contain no transcript text, waveform, or embedding vector. Evaluation
uncertainty resamples speakers with their incident pairs. The main mean
absolute error is pair-weighted; speaker-, dialect-relation-, and
stratum-weighted alternatives are sensitivity estimands.

Exact content matching is too sparse for the headline contrast. No
discourse-level metadata field is available in the audited KeSpeech metadata.
The continuous references are constructed from an external source and are not
perceptual labels. Their expected output hashes and representative mapping are
recorded in `results/provenance/reference_matrices.yaml`; hash records are
verification metadata, not substitutes for provider permission.
