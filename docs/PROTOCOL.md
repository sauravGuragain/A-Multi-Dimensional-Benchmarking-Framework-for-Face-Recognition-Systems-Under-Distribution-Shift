# FaceEval-X Fold-0 verification protocol

This file specifies the **custom identity-disjoint LFW Fold-0 verification benchmark** that produced the six-model results.
It is a specification of what the code does and what the archived run recorded. Every number below was read from the
source, the configs or the archive at commit `10e901b`; where a fact is not recorded, it says so.

**This is not the official LFW View 2 protocol.** The split and the pairs are this project's own. Do not compare these
numbers with published View 2 results, and never put them in one table with the closed-set identification run `2bad8efd`
(a different protocol; see `docs/EXPERIMENT_EVIDENCE_RECORD.md`).

## 1. Dataset

| Item | Value | Where it is recorded |
|---|---|---|
| Dataset | Official LFW, all identities | `dataset` block of every `verification_results.json` |
| Images / identities | 13,233 / 5,749 (`counts_match_official_lfw: true`) | same |
| Dataset content fingerprint (SHA-256) | `df3e734c6e5033224011c2587fd2c7671ce2f86d499cb57404f50f0d12337230` | same (`fingerprint_sha256`) |
| Official `lfw.tgz` archive SHA-256 | `055f7d9c632d7370e6fb4afc7468d40f970c34a80d4c6f50ffec63f5a8d536c0` | **Only in code comments** (`faceeval/orchestration/verification.py`, next to `OFFICIAL_LFW_FINGERPRINTS`; also mentioned in `tools/run_integrity_checker/verify_run_npz.py`). The comment calls it the published scikit-learn LFW checksum and says the fingerprint above was independently reproduced from a fresh extraction of that archive. No run output records the archive hash. |

A run is accepted as "full LFW" only when its content fingerprint is in `OFFICIAL_LFW_FINGERPRINTS`.

## 2. Split: identity-disjoint Fold 0

- Fold 0 is `DatasetSplitter(seed).all_k_folds(records, k=10)[0]` (`verification.py`): ten folds, seed 42, fold index 0.
  Seed 42 with fold 0 is the **reference split**; other (seed, fold) choices are supported by the code but have never
  been run on real data.
- The three roles are disjoint by **identity**. I checked the archived manifest of the deep run: no identity occurs in
  more than one role.

| Role | Identities | Images |
|---|---|---|
| train | 4,601 | 10,636 |
| validation | 574 | 1,259 |
| test | 574 | 1,338 |

(about 8 : 1 : 1 by identity; the counts were counted from `benchmarks/lfw_fold0_2026-11-20/deep_core/split_manifest_fold0.tsv`.)

- The split manifest is `split_manifest_fold0.tsv` (version `faceeval-split-manifest/r4.1/1`); its SHA-256 is written next
  to it (`.sha256`) and in the run JSON (`manifest_sha256`, `shuffled_order_sha256`).
- Classical models are **trained on the train identities**. FaceNet and ArcFace are frozen pretrained backbones; the
  train split is not used to fit them.
- The YAML keys `split_strategy`, `test_fraction` and `val_fraction` in the Fold-0 configs belong to the identification
  protocol. The verification runner builds its split as above.

## 3. Pairs

Pairs are built by `faceeval/evaluation/pairs.py`, algorithm version `faceeval-pairs/r4.1/1`, with seed 42, `cap = 10`,
`rho = 10`. Validation uses partition code 1, test uses partition code 2.

- **Genuine pairs (G):** per identity, all within-identity pairs if C(n, 2) <= cap, otherwise `cap` pairs drawn uniformly
  without replacement.
- **Impostor pool (K = rho x G):** uniform draws of cross-identity image pairs, rejecting same-image, same-identity and
  duplicate pairs, until K are accepted.
- **Balanced set:** G impostor pairs drawn uniformly from the pool (without replacement). Metrics reported as the primary
  results use the balanced set: G genuine plus G impostor pairs.
- The pair counts are checked against acceptance values (`Fold0AcceptanceError`):

| Partition | Genuine (G) | Impostor pool (K) | Balanced impostors | Balanced pairs total |
|---|---|---|---|---|
| validation | 684 | 6,840 | 684 | 1,368 |
| test | 764 | 7,640 | 764 | 1,528 |

- Pair files: `pairs_validation_fold0.tsv` and `pairs_test_fold0.tsv`; hashes in `pair_manifest_sha256`; parameters in
  `pair_params` of the run JSON.
- The pool-based ("full") columns (for example `auc_full_pool` in `table1`) use all G genuine plus all K pool pairs.
  The balanced metrics are the primary ones.

## 4. Scoring and threshold calibration

- **Score:** cosine similarity of L2-normalised embeddings (the dot product of unit vectors).
- **Threshold:** chosen **once per model on the clean validation balanced pairs only**, as the threshold with maximum
  accuracy (`best_accuracy_threshold`: accept iff `score >= t`; tied scores are accepted or rejected together; ties in
  accuracy resolve to the strictest, i.e. highest, threshold). It is **frozen before any test evaluation** and is the
  same in all 96 conditions of that model. Recorded as `threshold_protocol`:
  "max-accuracy on clean validation balanced pairs; frozen for test".
- Consequence: accuracy, TAR and FAR under perturbation use a threshold that was not re-tuned. AUC and EER are
  threshold-free.
- The deep-model thresholds in the archived run are 0.4168 (FaceNet) and 0.2087 (ArcFace); the classical thresholds are on
  different score scales (`table1_clean_baseline.csv`).

## 5. Metrics

Each row is one (model, perturbation, severity) condition on the 764 genuine plus 764 balanced impostor test pairs.

- `accuracy`: fraction of the 1,528 balanced pairs classified correctly at the frozen threshold.
- `tar` (true accept rate): accepted genuine pairs / 764. FRR = 1 - TAR.
- `far` (false accept rate): accepted impostor pairs / 764.
- `auc`: area under the ROC curve over the balanced pairs (ties count one half). `eer`: equal error rate (FAR = FRR).

## 6. Models

| Model | Kind | What the code does |
|---|---|---|
| Eigenfaces, Fisherfaces, LBPH, HOG-SVM | classical, trained on the train identities | Embeddings from the fitted model are scored with the same cosine rule. Eigenfaces were run separately (`eigenfaces/`, run `a55a4a37…`); Fisherfaces, LBPH and HOG-SVM together (`classical/`, run `5ecfdd99…`). Both used code `34608f4`. |
| FaceNet | frozen pretrained | InceptionResNetV1 (VGGFace2 weights `20180402-114759-vggface2.pt`), **centre crop** (fraction 0.8) to 160 x 160, **no face detector**. Because there is no detector, `detector_miss_rate` is NaN by design and it has no detection failures. |
| ArcFace | frozen pretrained | InsightFace pack `buffalo_l`; recognition model `w600k_r50.onnx`; face detector `det_10g.onnx`; 5-point landmark alignment to 112 x 112. The cropping fallback is switched off (`CROP_FALLBACK = False`), so an image where the detector finds no face gets **no embedding**. |

- FaceNet versus ArcFace is a **pipeline comparison, not a backbone-only comparison**: crop or alignment, input size and
  detection differ along with the backbone.
- The archived deep run used `device: cpu` (`configs/lfw_fold0_deep_core.yaml`). The exact ONNX Runtime providers used
  are not recorded in the archive.

## 7. Perturbations

Sixteen perturbation types, six severities `0.0, 0.2, 0.4, 0.6, 0.8, 1.0`: 16 x 6 = **96 conditions per model**.
Perturbations are applied to the **test images only**; validation data (and so the threshold) are clean.
Severity 0.0 returns an exact copy of the image for every type (`BasePerturbation.apply`), so the 16 severity-0.0
conditions of a model are all the clean condition.

The values below come from each class's `_severity_to_params` (computed from the code, not re-run on images).

| Type | Parameter | 0.2 | 0.4 | 0.6 | 0.8 | 1.0 | Rule |
|---|---|---|---|---|---|---|---|
| `gaussian_blur` | sigma | 1.6 | 3.2 | 4.8 | 6.4 | 8.0 | sigma = 8 x s |
| `motion_blur` | kernel length (px) | 8 | 16 | 24 | 32 | 40 | length = max(1, round(40 x s)); fixed 45 deg direction |
| `gaussian_noise` | std (fraction of pixel range) | 0.04 | 0.08 | 0.12 | 0.16 | 0.20 | 0.20 x s |
| `salt_pepper_noise` | corrupted-pixel density | 0.04 | 0.08 | 0.12 | 0.16 | 0.20 | 0.20 x s |
| `speckle_noise` | variance | 0.03 | 0.06 | 0.09 | 0.12 | 0.15 | 0.15 x s |
| `brightness` | additive shift | -0.24 | -0.08 | +0.08 | +0.24 | +0.40 | (s - 0.5) x 0.8 |
| `contrast` | scale factor | 0.475 | 0.754 | 1.194 | 1.893 | 3.0 | log-interpolated 0.3 to 3.0 |
| `gamma` | gamma | 0.435 | 0.758 | 1.320 | 2.297 | 4.0 | log-interpolated 0.25 to 4.0 |
| `rotation` | angle (deg) | 9 | 18 | 27 | 36 | 45 | 45 x s |
| `scaling` | scale (zoom-out, padded) | 0.9 | 0.8 | 0.7 | 0.6 | 0.5 | 1 - 0.5 x s |
| `cropping` | crop fraction | 0.08 | 0.16 | 0.24 | 0.32 | 0.40 | 0.40 x s; fixed-seed offsets, resized back |
| `random_occlusion` | occluded area fraction | 0.08 | 0.16 | 0.24 | 0.32 | 0.40 | grey rectangle at a fixed position |
| `face_mask` | mask opacity | 0.2 | 0.4 | 0.6 | 0.8 | 1.0 | opacity = s |
| `sunglasses` | glasses opacity | 0.2 | 0.4 | 0.6 | 0.8 | 1.0 | opacity = s |
| `jpeg_compression` | JPEG quality | 77 | 59 | 41 | 23 | 5 | int(95 - 90 x s) |
| `resolution_degradation` | downscale factor | 2.4 | 3.8 | 5.2 | 6.6 | 8.0 | 1 + 7 x s, then upscaled back |

**Centred perturbations.** `brightness`, `contrast` and `gamma` are two-sided. Brightness is neutral at s = 0.5 and
gamma is neutral at s = 0.5. Contrast is log-interpolated, so its scale is 1.0 at s of about 0.523 (at s = 0.5 it is 0.949);
the module docstring's "identity at 0.5" is therefore approximate. No grid severity equals 0.5, so the neutral point is
never evaluated and severity 0.0 (the clean copy) is not a darkening or washed-out level. Their distance-from-neutral curves are V-shaped.
Read them as distance from the neutral centre and do not read the mean over severities 0.2-1.0 as "average
degradation" for these three types.

## 8. Noise reproducibility (`per_image_v2`, master seed 42)

Applies to `gaussian_noise`, `salt_pepper_noise` and `speckle_noise` (recorded in `noise_seeding` of the run JSON):

- seed = first 8 bytes (big-endian unsigned 64-bit) of `SHA-256(master seed | image_id | perturbation type)`, passed to
  `numpy.random.default_rng`; `image_id` is Unicode-NFC normalised with backslashes turned into forward slashes.
- **Severity is not part of the seed**, so an image's noise pattern is the same at every severity and is scaled by it.
  For salt-and-pepper, the corrupted set is `U1 < density` and polarity is `U2 < 0.5`, both drawn from the per-image
  stream in that order at full size, so pixel sets are nested across severities and a pixel keeps its polarity.
- The run records the NumPy version (1.26.4) and a stream fingerprint of `default_rng(42)`; the analysis refuses to
  combine runs whose noise protocol, master seed or stream fingerprint differ.
- The historical Eigenfaces run `c408e0d2` used a different noise setup and is kept apart (`table11` only).

## 9. Failed embeddings: what is scored and what is not

- A failed embedding (zero, NaN or Inf row; for ArcFace mainly a detector miss) is replaced by the **zero vector**, so
  every pair that contains it has cosine 0, and **the pair is still scored**. Failed pairs are not removed.
- Each deep-model condition records `n_test_images`, `n_failed_embeddings`, `n_failed_pairs_balanced`,
  `n_failed_pairs_full` and `val_n_failed_embeddings`. Counts summed over conditions are not unique failed images (the same
  image is counted once per condition). Classical models have no failure counters; their fields are blank.
- The archived deep-model metrics are therefore **end-to-end**: they mix recognition quality with face-acquisition
  failure. A **valid-pairs-only** result (pairs where both images have a real embedding, same frozen threshold) is a
  different, conditional quantity: failed images are not a random sample, so it is selection-biased, is not comparable
  across models or conditions, and carries no interval. It never replaces the end-to-end result.
- The runner refuses a deep model when more than 50% of its clean validation embeddings fail (`MAX_FAILED_EMBEDDING_FRACTION`).

## 10. Limits of inference

- One fold and one seed. The intervals printed in `table2` and the scale in `table9` are **indicative only** (Hanley-McNeil
  standard error for AUC and Wilson score intervals for accuracy, TAR and FAR, treating pairs as independent; no interval is
  computed for EER); they are not population-level guarantees, and no
  significance test is performed or claimed.
- The pretrained FaceNet and ArcFace weights were trained outside this project; whether LFW identities occur in their
  training data has **not** been checked here.
- Custom split, not View 2: no comparison with published numbers.
- Calibration, fairness, speed and memory, and other splits or seeds, have not been run on real data.
