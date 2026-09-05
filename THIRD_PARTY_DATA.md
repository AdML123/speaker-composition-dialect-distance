# Third-party data and reference reconstruction

## Sinitic_Data

The continuous dialect references reported in the paper were computed from
Data4 in the public Sinitic_Data repository:

- Provider: `YiYang-github/Sinitic_Data`
- Commit: `820b9d15a74cbee82109f0bf54cf791fe16596ef`
- Commit URL: <https://github.com/YiYang-github/Sinitic_Data/tree/820b9d15a74cbee82109f0bf54cf791fe16596ef>
- Official Data4 source: <https://zhongguoyuyan.cn/index>
- Accessed: 2026-09-04
- Commit archive SHA-256: `621c8bce1fc49e5bc8e05103c810d675447c6d43d572bac3334bdff385a74692`
- Analysis-time archive SHA-256: `020f399824a9e4073f4092078099b5c9b6a995f1ef7d6eae6887f6b1250923c5`

The two archives use different container metadata and root directory names.
Their 37 extracted files were compared by relative path and SHA-256 and are
identical.

The upstream repository does not specify a redistribution license. Its source
files and the four continuous matrices derived from them are not redistributed
in this GitHub or Zenodo release. Readers should obtain the pinned commit from
the provider, review its terms, and run the released reconstruction code
locally.

## Data4 reconstruction summary

The pinned Data4 source contains 1,289 raw dialect locations and 999 available
transcription columns (the 1,000-word inventory has one fully missing word).
Transcriptions marked `wrong` and values occurring fewer than 1,000 times are
treated as missing or unreliable. Rows and columns with missing proportion
strictly below 30% are retained, yielding 1,084 locations and 915 words. The
released reconstruction reads `Data4/processed_info.pkl` and the `overall`
array in `Data4/distance_matrices.npz`; `overall_distance` is the arithmetic
mean of initials, finals, and tones distances.

The four construction-specific hashes and mappings are recorded in
`results/provenance/reference_matrices.yaml`: city-nearest representatives,
the raw overall submatrix, minimum-mean within-subgroup medoids (lowest-index
tie break), and mean cross-subgroup aggregates. These hashes are verification
metadata only; no upstream archive or derived continuous matrix is included.

Expected continuous-reference output hashes are recorded under
`derived_matrix_sha256` in `results/provenance/reference_matrices.yaml`. The
representative mapping and construction algorithm are included so that a
locally generated matrix can be checked without treating the expected matrix
as openly licensed data.

The repository's MIT License applies only to code and original material created
for this project. It does not relicense KeSpeech, Sinitic_Data, model weights,
audio, embeddings, or any other third-party material.
