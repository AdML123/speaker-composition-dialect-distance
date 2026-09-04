# Third-party data and reference reconstruction

## Sinitic_Data

The continuous dialect references reported in the paper were computed from
Data4 in the public Sinitic_Data repository:

- Provider: `YiYang-github/Sinitic_Data`
- Commit: `820b9d15a74cbee82109f0bf54cf791fe16596ef`
- Commit URL: <https://github.com/YiYang-github/Sinitic_Data/tree/820b9d15a74cbee82109f0bf54cf791fe16596ef>
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

Expected continuous-reference output hashes are recorded under
`derived_matrix_sha256` in `results/provenance/reference_matrices.yaml`. The
representative mapping and construction algorithm are included so that a
locally generated matrix can be checked without treating the expected matrix
as openly licensed data.

The repository's MIT License applies only to code and original material created
for this project. It does not relicense KeSpeech, Sinitic_Data, model weights,
audio, embeddings, or any other third-party material.
