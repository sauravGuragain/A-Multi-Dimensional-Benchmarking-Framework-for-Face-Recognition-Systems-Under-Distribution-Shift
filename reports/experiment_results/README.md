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
