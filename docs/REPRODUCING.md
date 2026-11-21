# Reproducing and verifying the FaceEval-X Fold-0 results

Three levels, from cheapest to most expensive. Levels 1 and 2 need neither the LFW images nor any model weights, and run
no experiment. **Level 3 (re-running the benchmark) is not needed to verify the committed results.**

Statements about what was run say where it was run. Nothing in this repository has been verified on a clean machine, and
the authors of these documents have no access to the maintainer's Mac: anything marked "maintainer-reported" was reported,
not independently checked.

## 1. Verify the committed archive and analysis (read-only)

From the repository root:

```bash
# (a) the benchmark archive: 216 entries, all must print OK
(cd benchmarks/lfw_fold0_2026-11-20 && shasum -a 256 -c SHA256SUMS)  # all 216 entries must print OK

# (b) the analysis outputs: 28 entries (29 files; SHA256SUMS does not list itself)
(cd benchmarks/lfw_fold0_2026-11-20/analysis && shasum -a 256 -c SHA256SUMS)  # all 28 entries must print OK
(cd benchmarks/lfw_fold0_2026-11-20/analysis && find . -type f | wc -l)                           # expected: 29

# (c) none of the PNG figures carries an extra credential chunk (see benchmarks/lfw_fold0_2026-11-20/README.md)
grep -l caBX benchmarks/lfw_fold0_2026-11-20/analysis/six_model/figures/*.png                    # expected: no output
```

On Linux use `sha256sum -c` instead of `shasum -a 256 -c`. The command prints each file's verification result; any line marked `FAILED` indicates a mismatch.

The checksums verify that the files are the bytes that were reviewed. They do not verify the science.

## 2. Regenerate the analysis into a NEW directory and compare

The analysis tools read archived result files only. They refuse to write into an input run directory and refuse a
non-empty output directory unless `--overwrite` is given. Always write to a **new** directory outside the archive.

```bash
python3 -m pip install -e ".[dev]"          # core + test tools; no PyTorch needed for the analysis
OUT=$(mktemp -d)

# six-model analysis: classical runs from the results root, FaceNet + ArcFace from ONE combined run
python3 -m faceeval.analysis.fold0_results \
    --results-root reports/experiment_results \
    --deep-run benchmarks/lfw_fold0_2026-11-20/deep_core \
    --out "$OUT/six_model"

# valid-pairs-only analysis, one model of the combined run at a time (--model is required for a multi-model run)
python3 -m faceeval.analysis.valid_pairs --run benchmarks/lfw_fold0_2026-11-20/deep_core --model facenet --out "$OUT/valid_pairs/facenet"
python3 -m faceeval.analysis.valid_pairs --run benchmarks/lfw_fold0_2026-11-20/deep_core --model arcface --out "$OUT/valid_pairs/arcface"

# compare the CSV tables with the committed ones
diff -r "$OUT/six_model/tables" benchmarks/lfw_fold0_2026-11-20/analysis/six_model/tables
for model in facenet arcface; do
  for file in "$OUT/valid_pairs/$model"/*.csv; do
    name="$(basename "$file")"
    cmp "$file" "benchmarks/lfw_fold0_2026-11-20/analysis/valid_pairs/$model/$name" || exit 1
  done
done
```

Options:

- `--deep-run DIR` takes one run that holds exactly FaceNet and ArcFace (192 rows, 96 per model). It cannot be combined with
  `--facenet-run` / `--arcface-run` (each deep model must come from exactly one run). With separate per-model runs, use
  `--facenet-run DIR --arcface-run DIR` instead. With no deep option, the analysis covers the four classical models only
  (384 primary rows).
- `--model NAME` (valid-pairs) selects one model of a multi-model run. Without it, a multi-model run is refused.
- `--results-root` (default `reports/experiment_results`) is the folder holding the committed `classical/`, `eigenfaces_final/` and
  `eigenfaces_per_image_noise/` archives. `--out` defaults to `<results-root>/analysis`; give an explicit new directory instead.
- `--no-figures` skips the figures. `--overwrite` allows a non-empty output directory: nothing is deleted, and files of an
  earlier analysis that were not rewritten are listed in `provenance.json` under `stale_files_not_rewritten`.
- Every input run is validated on its own (complete, full LFW, 96 rows per model, constant frozen threshold, consistent
  clean baseline), and the runs must agree on split content hash, pair hashes, dataset fingerprint, fold, seed and noise
  protocol. The historical run `c408e0d2` is read separately and appears only in `table11`.
- `valid_pairs --allow-archive-mismatch` records archive mismatches instead of refusing the run; for inspection only.
- A failed check prints `ANALYSIS REFUSED: <reason>` and exits with status 2.

What to expect. In the cloud sandbox (Python 3.12.3, NumPy 1.26.4, matplotlib 3.11.2) this exact regeneration, run on a copy of
the committed archive, reproduced all 13 six-model tables and all six valid-pairs CSV files **byte-for-byte**, and the seven
PNG files byte-for-byte as well; the provenance JSON files differed only in their path fields (the committed ones record
sandbox paths, so `diff` will show those lines, while the input SHA-256 values match). With another matplotlib version the
PNG bytes or pixels can differ while the tables do not. This was **not** repeated on the Mac, so whether it reproduces
there is unverified.

## 3. Environment

| | Recorded for the archived runs |
|---|---|
| Deep run `5ee3d007eb43445aa60c7fcd062ed50a` | code `f6d5e12f08e0bf484f44dd68f4a04bafe423403c` (`code_dirty: false`), macOS arm64, Python 3.12.14, NumPy 1.26.4, torch 2.2.2, torchvision 0.17.2, facenet-pytorch 2.6.0, insightface 2.0, onnxruntime 1.30.0, onnx 1.23.0, opencv-python 4.11.0.86, scikit-image 0.24.0, scikit-learn 1.9.1, scipy 1.13.1, Pillow 10.2.0; SHA-256 of all five ArcFace `buffalo_l` model files and of the FaceNet VGGFace2 weights. All of it is in `verification_results.json` (`provenance`). |
| Classical runs `5ecfdd99…`, `a55a4a37…` | code `34608f4eb8f1e073c254d53cbae9e64c82150710`, Python 3.12.14, NumPy 1.26.4. **No `provenance` block**: package versions, platform and weight hashes are **NOT RECORDED** for these runs. |
| Full environment | `benchmarks/lfw_fold0_2026-11-20/requirements-freeze.txt` (119 lines; the pip freeze committed with the archive). |

Notes:

- **`insightface 2.0`** is what the run recorded (and what the freeze lists as `insightface==2.0`). The author of this
  document could not check that version string against the package index. Confirm it on your machine before citing it:
  `python3 -m pip show insightface`, and `python3 -m faceeval.analysis.env_pins --run benchmarks/lfw_fold0_2026-11-20/deep_core --print`
  prints the versions the run recorded.
- `pyproject.toml` pins `numpy>=1.24,<2` and `opencv-python>=4.8,<5`; the `deep` extra needs Python 3.11 or 3.12
  (`facenet-pytorch` 2.6.0 requires `torch>=2.2,<2.3`). A clean-machine installation has **not** been tested.
- The deep run used `device: cpu` (config). The ONNX Runtime providers actually used are not recorded.
- The exact dataset used is identified by the content fingerprint and, for the original archive, the SHA-256 given in
  `docs/PROTOCOL.md`, section 1.

## 4. Tests

```bash
python3 -m pytest tests/test_fold0_analysis.py tests/test_valid_pairs.py -q -p no:cacheprovider -rs
```

- The tests marked `needs_deep_core` read the committed archive and are skipped if it is missing; with `-rs` skips are
  listed. They need `benchmarks/lfw_fold0_2026-11-20/deep_core` and `reports/experiment_results/classical`.
- **What has been run, and where.** The 40 tests added by PR #10 (30 in `tests/test_fold0_analysis.py`, 10 in
  `tests/test_valid_pairs.py`) were run in a **Linux cloud sandbox** on a tree whose four changed files are byte-identical to
  those merged in `10e901b`; the two test files then gave **258 passed, 0 skipped** (94.84 s). That was not a run on the
  Mac and not in GitHub CI, and the full suite was not run on `10e901b`.
- **Do not quote a total test count for the current commit** unless you have run the full suite on it and recorded the
  commit with the summary line. Earlier counts in `docs/EXPERIMENT_EVIDENCE_RECORD.md` (983 on the Mac, 1,172 in a sandbox
  at `98774f3`) belong to older commits.
- CI (`.github/workflows/ci.yml`) runs the lightweight suite on Python 3.11 and 3.12 plus a `--help` smoke test of the
  analysis CLIs; it does not run real models or real LFW. Its status on `10e901b` was not inspected.

## 5. Re-running the benchmark (not needed to verify the results)

Re-running needs the LFW images at `data/lfw/<Person_Name>/<Person_Name>_0001.jpg`, the real model weights, and hours of
compute; it writes only into a new run directory and has no resume. The runs behind the archive used
`configs/lfw_fold0_classical.yaml`, `configs/lfw_fold0_eigenfaces_final.yaml` and `configs/lfw_fold0_deep_core.yaml`
(copies are in `benchmarks/lfw_fold0_2026-11-20/configs/` for the classical and deep configs). Start commands are in the
README and `docs/USER_GUIDE.md`. A repeat of a run has never been performed, so run-to-run repeatability is **unknown**.
