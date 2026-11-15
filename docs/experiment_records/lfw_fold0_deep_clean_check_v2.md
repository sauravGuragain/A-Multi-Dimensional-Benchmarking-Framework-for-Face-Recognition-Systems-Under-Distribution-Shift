# LFW Fold 0 Deep Clean Verification v2

## Execution
- Commit: 930b49d96a39ce3de4bc072758eb842c3f981a51
- Run ID: c069be84b7f84008a2b90975d384d8bf
- Date: 2026-11-15
- Protocol: LFW Fold 0
- Models: FaceNet, ArcFace
- Conditions: Clean only
- Result rows: 2
- Run status: COMPLETE
- Failures: 0
- Stub backbones: 0
- Exit status: 0

## Validation
- FaceNet threshold: 0.41684
- FaceNet balanced accuracy: 0.9905
- ArcFace threshold: 0.20875
- ArcFace balanced accuracy: 0.9956

These are validation metrics, not test-set performance.

## Pair counts
- Validation: 684 genuine, 6840 impostor
- Test: 764 genuine, 7640 impostor
- Fold 0 shuffled-order vs V25: match

## Preprocessing
- FaceNet: facenet_fixed, centre_crop, output 160
- ArcFace: arcface, affine_fixed_landmarks, output 112

## Isolation and safety
- Original results.db SHA256:
  68823c0e3e057eacb422fa84f93d457eeafb4321b325a0e14de061dfd7b8f941
- Original backup SHA256:
  68823c0e3e057eacb422fa84f93d457eeafb4321b325a0e14de061dfd7b8f941
- Isolated results.db SHA256:
  7dd8feeefee378d684a3bd328b6aa6b471ef51de9d0c31d0d7c9a5c73746e618
- Previous clean verification output retained.
- Git working tree was clean before adding this report.

## Limitations
- This is a two-model clean-condition verification, not the full 96-condition benchmark.
- Validation metrics are not test metrics.
- CUDAExecutionProvider was unavailable; ArcFace loaded successfully with available providers.
