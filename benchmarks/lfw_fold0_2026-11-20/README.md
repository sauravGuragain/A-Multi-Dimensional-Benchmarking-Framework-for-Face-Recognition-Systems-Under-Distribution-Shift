# LFW Fold-0 benchmark archive (2026-11-20)

Canonical archive of the six-model custom Fold-0 verification benchmark and of its post-hoc analysis. The protocol is in
`docs/PROTOCOL.md`, the verification and regeneration commands in `docs/REPRODUCING.md`, the meaning of every result
file in `docs/RESULTS_MAP.md`.

**Custom identity-disjoint Fold 0, not the official LFW View 2 protocol.** One split, one seed (42). Do not compare with
published View 2 numbers.

## Contents

| Path | What it is |
|---|---|
| `classical/` | Run `5ecfdd995ffc4e528b0c17563798a7d4`: Fisherfaces, LBPH, HOG-SVM; 288 rows; code `34608f4eb8f1e073c254d53cbae9e64c82150710`; status COMPLETE. Results CSV/JSON, split manifest, pair files. |
| `eigenfaces/` | Run `a55a4a37f1664bf89bb2987f01c45683`: Eigenfaces; 96 rows; code `34608f4…`; status COMPLETE. |
| `deep_core/` | Run **`5ee3d007eb43445aa60c7fcd062ed50a`**: FaceNet and ArcFace in one run; **192 rows** (2 models x 96 conditions); code `f6d5e12f08e0bf484f44dd68f4a04bafe423403c`, `code_dirty: false`; status COMPLETE, no failures, no stub backbones. Includes `scores/` (192 per-condition `.npz` score files plus `test_pairs.npz`) and `verification_rows.jsonl` (the per-row record stream written during the run). |
| `configs/` | The classical and deep-core YAML configs. |
| `requirements-freeze.txt` | Package freeze committed with the archive. |
| `ARCHIVE_INFO.txt` | Note written when the archive was assembled (2026-11-20 21:19 +0545). It lists the canonical result code version `34608f4…` (the classical runs); the deep run's own code version is `f6d5e12…`. It does not record the deep run id. |
| `SHA256SUMS` | SHA-256 of the 216 files of the original archive (every file above except this README, `SHA256SUMS` itself and `analysis/`). Unchanged since the archive was assembled. |
| `analysis/` | The post-hoc analysis (29 files) with its own `SHA256SUMS`; see below. |

The run ids appear in each `verification_results.json` as part of `manifest_path`
(for example `experiments/lfw_fold0_deep_core/verification/5ee3d007…/split_manifest_fold0.tsv`).

Relation to other run ids in older documents: `docs/EXPERIMENT_EVIDENCE_RECORD.md` also names FaceNet run `b567f880…` and
ArcFace run `4bdf6982…` (earlier separate sweeps reported from the Mac). Their result files are **not** in this
repository, and their relation to `5ee3d007…` is **not recorded** here. The six-model analysis described in these
documents uses only the runs listed above. The historical per-image-noise Eigenfaces run `c408e0d2…` is under
`reports/experiment_results/eigenfaces_per_image_noise/` and is never mixed into the primary results.

Provenance: the deep run records a `provenance` block (platform, package versions, weight SHA-256). The classical and
Eigenfaces runs record only Python, NumPy and code version; everything else for them is **NOT RECORDED**.

## Verify

From the repository root (use `sha256sum -c` on Linux):

```bash
(cd benchmarks/lfw_fold0_2026-11-20 && shasum -a 256 -c SHA256SUMS)                   # all entries must print OK
(cd benchmarks/lfw_fold0_2026-11-20/analysis && shasum -a 256 -c SHA256SUMS)          # all entries must print OK
(cd benchmarks/lfw_fold0_2026-11-20/analysis && find . -type f | wc -l)                                  # expected 29
grep -l caBX benchmarks/lfw_fold0_2026-11-20/analysis/six_model/figures/*.png                           # expected: no output
```

## `analysis/`

Generated post hoc in a Linux cloud sandbox from the data above (`analysis/RUN_RECORD.txt` has the commands, inputs and
environment). **29 files**: 28 listed in `analysis/SHA256SUMS` plus that file itself (RUN_RECORD, 13 six-model tables,
7 figures, the six-model provenance JSON, and three files each for FaceNet and ArcFace valid-pairs outputs). No
benchmark was rerun and no benchmark data was touched to produce them. The maintainer reports that all 28 checksums
verified locally on the Mac after copying.

- The provenance JSON files contain the **sandbox paths** (`/tmp/...`) used when the files were generated; those are not
  paths in this repository. The input SHA-256 values in the same files identify the inputs.
- **Figure bytes and a recorded discrepancy.** The seven PNGs committed here are the bytes listed in
  `analysis/SHA256SUMS`. During preparation, copies of the same seven PNGs obtained from the loose delivery folder were
  found to differ from those bytes: each carried one extra PNG chunk, `caBX` (a C2PA content-credentials manifest,
  5,770 bytes including the chunk header), and therefore failed the checksum. The image data was identical: all seven pairs
  had the same dimensions, zero differing pixels, and byte-identical IDAT data. The 21 non-figure files matched. When or
  why the extra chunk was added is **unresolved**; the filesystem timestamps do not show a later modification. If a
  figure fails the checksum and `grep -l caBX` lists it, it is such a copy; use the files whose bytes match the manifest.

## What this archive does not contain

- Calibration, fairness, speed and memory results, other splits or seeds: never run on real data.
- Models other than the six (PCA+SVM, k-NN, InsightFace, dlib) on real LFW.
- The raw LFW images and the model weight files (their hashes are in the deep run's `provenance`).
