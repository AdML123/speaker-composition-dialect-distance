# Release audit

Date: 2026-09-04

Status: final public package passed; external publication in progress.

## Scope

- Experimental corpus: KeSpeech only.
- 3D-Speaker: related-work citation only; no 3D-Speaker adapter, trial, result,
  or test is present.
- Included: source, tests, configuration, pair manifests, the author-created
  binary reference, report-level evidence, and provenance.
- Excluded: raw audio, dataset archives, extracted embeddings, model
  checkpoints, credentials, local caches, private paths, and author-only
  correspondence, manuscript or paper PDF, and LaTeX or TikZ content. The
  upstream Sinitic\_Data files and four continuous
  reference matrices derived from them are also excluded because the upstream
  repository states no redistribution license.

## Verification

- Public tests: 226 passed, 3 dependency warnings; exit code zero.
- Private-path and secret-pattern scan: zero matches.
- Manuscript/PDF/LaTeX/TikZ exclusion test: passed.
- `MANIFEST.sha256`: 154 files, SHA-256
  `2bb8c0096470207e0d4d9a8d506a6c10346d1374078eb3b9233060708c492f15`;
  the checksum and this audit file are excluded from the checksum set.
- Every manifest entry was recomputed with zero missing or mismatched files.
- Old Matplotlib figure assets and their test are absent.
- No cache, bytecode, credential, restricted embedding path, private absolute
  path, or 3D-Speaker experiment file is included in the checksum set.

The upload package contains no manuscript, paper PDF, LaTeX, TikZ, figure-source,
or table-source file. GitHub and Zenodo identifiers are recorded in
`manifest.yaml` and `CITATION.cff`.
