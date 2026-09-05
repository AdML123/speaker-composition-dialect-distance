# Speaker-Composition Sensitivity of Frozen-Embedding Dialect Distance

This repository is the publication staging package for an IEEE Signal
Processing Letters study of pair construction in frozen speech-embedding
dialect distance. It contains shareable code, tests, sanitized KeSpeech pair
manifests, the author-defined binary reference, and report-level results. It
does not contain the manuscript, paper PDFs, LaTeX or TikZ sources, audio,
extracted embeddings, model checkpoints, third-party continuous-reference
matrices, local caches, or credentials.

## What the evidence supports

Within twelve available KeSpeech recording-condition strata, changing speaker
composition increases same-dialect cosine distance for Chinese HuBERT, Chinese
wav2vec 2.0, and WavLM. The estimand is a condition-aware association, not an
exclusive speaker-identity effect. Exact content matching is sparse and no
discourse-level field is available.

For HuBERT calibration, a linear projection lowers pair-weighted held-out mean
absolute error by 10.97 percent under a binary subgroup proxy and 9.49 percent
under an externally derived continuous reference. Both references are
constructed targets, not perceptual ground truth. Speaker-, relation-, and
stratum-weighted analyses preserve the binary-reference improvement direction.

The lower-error linear projection does not improve dialect-relation ordering.
Pair-only parameter-matched and wider multilayer projections improve Spearman
and Kendall association and pairwise order accuracy under both references while
also reducing mean absolute error. A controlled architecture-by-cross-loss
factorial finds different cross-loss responses for linear and multilayer
projections. The prespecified semantic, prevalence, transfer, exposure,
gradient, aggregation, and placebo probes do not identify or disprove a unique
mechanism.

## Data access

KeSpeech must be obtained separately under its provider terms.

- Project and access instructions: https://github.com/KeSpeech/KeSpeech
- Dataset license: https://github.com/KeSpeech/KeSpeech/blob/main/dataset_license.md
- Dataset paper: https://datasets-benchmarks-proceedings.neurips.cc/paper_files/paper/2021/hash/0336dcbab05b9d5ad24f4333c7658a0e-Abstract-round2.html

No official KeSpeech dataset DOI was located in the provider or proceedings
records. DOI `10.5281/zenodo.22340177` identifies only this derived software
and data package.

The continuous reference uses Data4 from Sinitic_Data commit
`820b9d15a74cbee82109f0bf54cf791fe16596ef`. That repository does not state a
redistribution license. Its archive and the four continuous matrices derived
from it are therefore not included here. `THIRD_PARTY_DATA.md` gives the pinned
source, verified hashes, and local reconstruction route.

## Repository map

- src: analysis, matching, inference, projection, sensitivity, and report code
- tests: synthetic unit tests and locked report-contract tests
- configs/experiment.yaml: revisions, hashes, seeds, grids, and protocol values
- results/pairs: selected pair identifiers, transcript hashes, and strata audit
- results/references: the author-defined binary reference; continuous outputs are rebuilt locally
- results/analysis and results/gates: report-level evidence used by the paper
- repro: exact audit and full-rerun routes

`python -m src.sanitize_public_release --release-root .` removes per-pair
continuous-target fields from a freshly staged public tree. The checked-in
aggregate reports already passed this redaction step.

## Quick audit

Python 3.10 or later is recommended.

~~~bash
python -m venv .venv
. .venv/bin/activate
python -m pip install -r requirements.txt
python -m pytest tests -q
~~~

On Windows PowerShell, activate the environment with
.venv\Scripts\Activate.ps1.

The test suite uses synthetic fixtures or locked report JSON. Model loading and
audio-extraction tests use mocks. A full experimental rerun requires licensed
KeSpeech access and local copies of the model revisions named in the
configuration. Replace MODEL_CACHE with the local model root and never commit
that path.

## Scope

KeSpeech is the only experimental corpus. The trainable-correction study uses
HuBERT; WavLM and wav2vec 2.0 appear in the frozen speaker-composition contrast.
The continuous-reference alternatives change subgroup representatives within
the same external source. Their reported hashes support identity checks but do
not grant access or redistribution rights. Cross-corpus, cross-language,
device-aware, and perceptual validation remain open.

The public software package deliberately excludes the manuscript and all
LaTeX content. Article files remain in the private submission workspace.

## Citation and license

Citation metadata are in CITATION.cff. The MIT License covers this project's
code and author-created release material only. KeSpeech, model checkpoints,
and Sinitic_Data retain their own terms and are not relicensed by this package.
