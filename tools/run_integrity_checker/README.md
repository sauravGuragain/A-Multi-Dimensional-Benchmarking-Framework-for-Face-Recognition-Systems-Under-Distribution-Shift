# FaceEval-X run-integrity checker (read-only)

## Purpose and limits
`verify_run_npz.py` checks the **structural integrity** of ONE existing FaceEval-X Fold 0 verification run directory
(96 conditions = 16 perturbations x 6 severities, per deep model). It only reads files. It does not run, resume,
repair or modify a benchmark, load a model, or touch the network.

A PASS means the run artifacts are internally consistent and match the protocol constants hard-coded in the checker.
It does NOT prove that the models actually ran, that perturbations were actually applied, that the dataset is the real
LFW, or that EER/FRR numbers are right. See REVIEW_REPORT.md for the exact coverage list.

**Synthetic fixtures** (from `generate_synthetic_fixture.py`) use made-up scores and invented embeddings. They validate
the checker's structure only. They are NOT real LFW and NOT real model output; a synthetic PASS says nothing about
scientific validity.

## Prerequisites
Python 3.9+, numpy. scikit-learn is optional (used only by the generator for AUC). The generator and the test harness
also need a local clone of the FaceEval-X repo (`--repo PATH`) because they reuse its pair/split writers.

## Commands (run one at a time)
1. Generate a synthetic fixture (writes only into the new, empty directory you name):
   `python3 generate_synthetic_fixture.py --out /tmp/fx --profile production --repo /path/to/repo`
   Profiles: `production`, `stub-deep`, `stub-mixed`. `--force` replaces only the `run/` of a fixture the generator itself created.
2. Run the regression tests (works on disposable copies; originals are hash-checked before and after):
   `python3 test_verify_run_npz_regressions.py --repo /path/to/repo [--real-stubs DIR_OF_STUB_RUNS]`
3. Check a real run directory (strict, production):
   `python -I verify_run_npz.py RUN_DIR --expect-models facenet,arcface --expect-per-model 96`

## Modes
- **Strict (default)**: requires `--expect-models` and `--expect-per-model 96`; enforces the full grid, official LFW counts,
  Fold 0 constants. Symlinks are rejected for the run dir, `scores/`, every top-level artifact and every score file.
- **`--test-mode`**: rehearsal only. Tolerates stub backbones and a non-official dataset; skips grid/official-count checks;
  follows symlinks for the run directory, `scores/` and top-level artifacts. A symlinked per-condition score file is rejected in every mode.
- **`--partial-grid`**: reduced real-model smoke run; needs `--expect-per-model` below 96; not combinable with `--test-mode`.

## Exit codes
0 PASS (only then is "RESULT: ALL CHECKS PASSED" printed) | 1 FAIL | 2 usage error | 3 INCOMPLETE VERIFICATION.
