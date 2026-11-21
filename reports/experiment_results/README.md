# FaceEval-X Experiment Results Archive

## Project
Multi-dimensional benchmarking framework for face recognition
under distribution shift.

## 1. Primary comparative experiments

### Classical models
- Run: 5ecfdd995ffc4e528b0c17563798a7d4
- Models: Fisherfaces, LBPH, HOG-SVM
- Rows: 288
- Conditions: 16 perturbations x 6 severities x 3 models
- Code commit: 34608f4eb8f1e073c254d53cbae9e64c82150710
- Manifest SHA256:
  d36e8ae060eae1bea9df663351b00f44f5156330c06f55ad77234ef8fc7bdd05
- Status: COMPLETE
- Failures: 0
- Stub backbones: 0

### Eigenfaces final
- Run: a55a4a37f1664bf89bb2987f01c45683
- Model: Eigenfaces
- Rows: 96
- Conditions: 16 perturbations x 6 severities
- Code commit: 34608f4eb8f1e073c254d53cbae9e64c82150710
- Manifest SHA256:
  d36e8ae060eae1bea9df663351b00f44f5156330c06f55ad77234ef8fc7bdd05
- Status: COMPLETE
- Failures: 0
- Stub backbones: 0

These two experiments share the same Fold 0 manifest
and code version. They are the primary comparative results.

## 2. Historical experiment

### Eigenfaces per-image noise
- Run: c408e0d2352e4de780a9d7a45a3ae7cc
- Rows: 96
- Code commit: e0d0eae8444e93b6110740aae81442385048316b
- Manifest SHA256:
  3c1cf7a77791a2c42cdf05f5115594fefb904cc602f3be5a77249f967b0a7116
- Status: COMPLETE
- Failures: 0
- Stub backbones: 0

This is a separate historical run with a different manifest
and code version. Do not silently combine it with primary
comparative results.

## 3. Reproducibility protocol

- Dataset: Full LFW
- Split: Custom identity-disjoint Fold 0
- Seed: 42
- K folds: 10
- Threshold: Maximum accuracy on clean balanced validation
  pairs; frozen for test evaluation.
- Noise protocol: per_image_v2
- Noise master seed: 42
- Recorded NumPy version: 1.26.4
- Recorded Python version: 3.12.14

## 4. Required analysis

1. Inspect CSV and JSON together.
2. Verify all metrics and result completeness.
3. Compare four classical/model families without mixing
   incompatible protocol versions.
4. Analyse clean performance and robustness degradation.
5. Inspect AUC, EER, TAR, FAR, FRR and accuracy.
6. Check missing values and detector miss rates.
7. Produce reproducible tables and figures.
8. Identify remaining experiments and resource requirements.

Do not fabricate metrics or claim unperformed experiments.

## 5. Deep-model runs and six-model analysis (archived elsewhere in Git)

Updated 2026-11-21. The archives in this folder are the four classical models. The deep-model results (FaceNet and ArcFace,
one combined run, **`5ee3d007eb43445aa60c7fcd062ed50a`**, 192 rows, code `f6d5e12`) and the six-model analysis built from
them are committed in `benchmarks/lfw_fold0_2026-11-20/` (see its README):

- `benchmarks/lfw_fold0_2026-11-20/deep_core/`: the deep run.
- `benchmarks/lfw_fold0_2026-11-20/analysis/`: 13 tables, 7 figures, provenance, FaceNet and ArcFace valid-pairs outputs.

Earlier separate deep sweeps (FaceNet `b567f88089c142d3b7732d993f34c979`, ArcFace `4bdf6982668f4f498fa3fe36d41c885b`, as
reported by the maintainer) are not in Git, and their relation to `5ee3d007...` is not recorded.

The analysis command for the archived combined run is
`python3 -m faceeval.analysis.fold0_results --results-root reports/experiment_results --deep-run benchmarks/lfw_fold0_2026-11-20/deep_core --out NEW_DIR`
(with separate per-model runs: `--facenet-run DIR --arcface-run DIR`). See `docs/REPRODUCING.md`.

Deep-model values are end-to-end: a failed embedding is a zero vector and its pairs are still scored, not removed, so
ArcFace AUC/EER are not valid-pairs-only performance. FaceNet (centre crop, no detector) and ArcFace (detector plus
5-point alignment) use different preprocessing pipelines. Brightness, contrast and gamma are centred at severity 0.5.
This protocol is different from, and never merged with, the 200-subject identification experiment (run `2bad8efd`).
Full protocol: `docs/PROTOCOL.md`; result map: `docs/RESULTS_MAP.md`.
