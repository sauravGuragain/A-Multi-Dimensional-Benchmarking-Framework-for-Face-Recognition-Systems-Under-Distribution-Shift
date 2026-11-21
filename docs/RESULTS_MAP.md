# FaceEval-X results map

What each committed result file contains and how to read it. The protocol behind every number is in
`docs/PROTOCOL.md`; how to verify or regenerate the files is in `docs/REPRODUCING.md`.

All files below live in `benchmarks/lfw_fold0_2026-11-20/analysis/` (29 files: 28 listed in its `SHA256SUMS`, plus the
`SHA256SUMS` file itself). They were generated **post hoc, in a Linux cloud sandbox** (Python 3.12.3, NumPy 1.26.4,
matplotlib 3.11.2; see `RUN_RECORD.txt`) from the archived benchmark data. No benchmark was run to make them. Their
checksums have been verified by the maintainer on the Mac (reported, not independently verified by the author of this
document).

## Layout

```
analysis/
  RUN_RECORD.txt                      how and where the outputs were produced (commands, inputs, environment, notes)
  SHA256SUMS                          SHA-256 of the 28 other files
  six_model/
    provenance.json                   input files and hashes, run compatibility checks, failure semantics
    tables/table1 ... table13 (CSV)   see below
    figures/fig1 ... fig7 (PNG)       see below
  valid_pairs/
    facenet/  valid_pairs_by_condition.csv  valid_pairs_model_summary.csv  valid_pairs_provenance.json
    arcface/  (the same three files)
```

## Six-model tables (`six_model/tables/`)

Six models: Eigenfaces, Fisherfaces, LBPH, HOG-SVM, FaceNet, ArcFace; 16 perturbations x 6 severities = 96 conditions
each; **576 primary rows**. `harm` is defined so that **positive harm always means worse than clean**
(`delta = value - clean`; harm = -delta for accuracy, TAR and AUC, harm = +delta for FAR and EER).

| File | Rows | Content |
|---|---|---|
| `table1_clean_baseline.csv` | 6 | Clean accuracy, TAR, FAR, AUC, EER, frozen threshold, full-pool AUC/EER, ranks. |
| `table2_clean_intervals_indicative.csv` | 6 | Clean metrics with **indicative** intervals (Hanley-McNeil for AUC, Wilson for rates; independence assumed; no EER interval). |
| `table3_auc_harm_heatmap_severity1.csv` | 16 | AUC harm at severity 1.0, perturbation x model. |
| `table4_auc_harm_heatmap_mean_severity.csv` | 16 | Mean AUC harm over severities 0.2 to 1.0, perturbation x model. |
| `table5_auc_by_severity.csv` | 36 | Mean, min and max AUC per model and severity. |
| `table6_degradation_rankings_auc.csv` | 96 | Per model and perturbation: mean harm over severities 0.2 to 1.0, harm at severity 1.0, ranks (1 = most damaging). |
| `table7_degradation_summary.csv` | 18 | Per model and metric: clean value, mean harm, worst and best condition, counts. |
| `table8_perturbation_across_models_auc.csv` | 16 | Per perturbation, harm averaged across models and its rank. |
| `table9_auc_cells_indicative_scale.csv` | 480 | AUC per perturbed cell with an indicative scale and a flag `abs_harm_exceeds_scale`. **Indicative only.** |
| `table10_all_cells_long.csv` | 2,880 | Every condition x five metrics (576 x 5): value, clean value, delta, harm. |
| `table11_historical_vs_final_eigenfaces.csv` | 96 | Eigenfaces `eigenfaces_final` versus the **historical** per-image-noise run `c408e0d2`. The only place that run appears. |
| `table12_failure_diagnostics_by_condition.csv` | 576 | Per condition: images, failed embeddings, failed pairs (balanced and full), rates with explicit denominators. Blank for classical models (`diagnostics_applicable = false`). |
| `table13_failure_diagnostics_model_summary.csv` | 6 | Per model: pooled clean and perturbed failure counts and rates; sums are across conditions, **not unique failed images**. |

Notes on tables 3 to 9: AUC is threshold-free and is the harm measure used for rankings. Brightness, contrast and gamma
are two-sided (see `docs/PROTOCOL.md`, section 7), so their "harm" is distance from the neutral centre and the mean
over severities is not an "average degradation" for them.

## Six-model figures (`six_model/figures/`)

| File | Shows |
|---|---|
| `fig1_clean_baseline.png` | Clean baseline per model. |
| `fig2_auc_harm_heatmap_severity1.png` | AUC harm at severity 1.0 (as `table3`). |
| `fig3_auc_harm_heatmap_mean_severity.png` | Mean AUC harm over severities 0.2 to 1.0 (as `table4`). |
| `fig4_auc_severity_mean.png` | Mean AUC versus severity per model (as `table5`). |
| `fig5_auc_severity_by_perturbation.png` | AUC versus severity, one panel per perturbation. |
| `fig6_degradation_rankings.png` | Degradation rankings (as `table6`). |
| `fig7_failure_rates.png` | Failed-embedding rates (deep models only; as `table12`). |

The PNG files were rendered with matplotlib 3.11.2; another version can change the pixels or bytes but not the tables.

## Valid-pairs outputs (`valid_pairs/<model>/`)

Produced by `python3 -m faceeval.analysis.valid_pairs --run <deep_core> --model <name>`. The tool first **reproduces every
archived end-to-end value** (accuracy, TAR, FAR, AUC, EER) and failure count from the saved `scores/*.npz` and refuses
the run if any differ. For the archive, `archive_mismatches` is empty in both provenance JSON files (96 conditions each,
`archive_mismatches_allowed` false).

- `valid_pairs_by_condition.csv` (96 rows per model): archived `e2e_*` values next to `valid_*` values on the pairs where both
  images have a real embedding (same frozen threshold, no re-tuning), `valid_minus_e2e_*`, valid genuine and impostor
  counts, and `n_common_valid`.
- `valid_pairs_model_summary.csv` (1 row): summary over conditions.
- `valid_pairs_provenance.json`: input hashes, `semantics`, `model_filter`, `models_in_run`, `n_rows_in_run`.
- The `path` and `run_directory` fields inside the provenance JSON files are the **cloud-sandbox paths** the files were
  generated from (not paths in this repository). The input SHA-256 values in the same files identify the inputs.

## Reading the deep-model results: end-to-end versus valid-pairs-only

- The six-model tables report **end-to-end** results. A failed embedding is a zero vector and its pairs are still scored.
- `valid_*` columns are **conditional** on both images being acquired, are **selection-biased**, are **not comparable
  across models or conditions**, and have no interval. They never replace the end-to-end numbers.

**FaceNet** (no detector): 0 failed embeddings in all 96 conditions; valid-pairs and end-to-end values are identical.

**ArcFace** (read from `table12`, `table13` and `valid_pairs/arcface/`):

- Clean conditions: 0 failures. Failures occur only in perturbed conditions: **31 of the 80 perturbed conditions** have at
  least one failed embedding (across `gaussian_blur`, `motion_blur`, `gaussian_noise`, `salt_pepper_noise`,
  `speckle_noise`, `contrast`, `gamma`, `cropping`, `random_occlusion`, `face_mask`, `jpeg_compression` and
  `resolution_degradation`).
- Failed embeddings **summed over conditions: 12,969** (not unique images). Pooled over the 80 perturbed conditions that is
  12,969 of 107,040 image-condition cases (12.1%); failed balanced pairs 18,058 of 122,240 (14.8%).
- **Five conditions have fewer than 10 valid pairs in a class**, so their valid-pairs metrics are undefined (NaN) and are
  not meaningful:

| Condition | Failed / test images | Valid genuine / impostor pairs | End-to-end AUC |
|---|---|---|---|
| `gaussian_blur` 1.0 | 1,284 / 1,338 | 1 / 0 | 0.5007 |
| `motion_blur` 0.6 | 1,307 / 1,338 | 0 / 1 | 0.4993 |
| `motion_blur` 0.8 | 1,338 / 1,338 | 0 / 0 | 0.5000 |
| `motion_blur` 1.0 | 1,338 / 1,338 | 0 / 0 | 0.5000 |
| `salt_pepper_noise` 1.0 | 1,295 / 1,338 | 0 / 0 | 0.5000 |

  When the detector misses (nearly) every image, ArcFace collapses to chance in the end-to-end results: that reflects
  face acquisition, not recognition. `valid_pairs_model_summary.csv` lists `n_conditions_valid_status_ok = 91` of 96.
- No image is valid in every condition (`n_images_valid_in_every_condition = 0`), so `n_common_valid_pairs = 0` and every
  `common_*` metric is NaN for ArcFace. (FaceNet: 1,338 images and 1,528 pairs are valid in every condition.)
- **Mean AUC comparisons need care.** The summary row gives mean end-to-end AUC 0.9181 over all 96 conditions and mean
  valid-pairs AUC 0.9772 over the **91 conditions where it is defined**. These two means cover different condition sets.
  Over the same 91 conditions the mean end-to-end AUC is 0.9410 (computed from `valid_pairs_by_condition.csv`). Do not
  present 0.9772 as "ArcFace robustness", and do not compute the difference from the two summary-row means.

## Clean baseline (from `table1_clean_baseline.csv`)

| Model | Accuracy | TAR | FAR | AUC | EER |
|---|---|---|---|---|---|
| Eigenfaces | 0.5936 | 0.4843 | 0.2971 | 0.6352 | 0.4097 |
| Fisherfaces | 0.6270 | 0.5340 | 0.2801 | 0.6850 | 0.3809 |
| LBPH | 0.5478 | 0.4136 | 0.3181 | 0.5808 | 0.4463 |
| HOG-SVM | 0.5792 | 0.3730 | 0.2147 | 0.6186 | 0.4110 |
| FaceNet | 0.9784 | 0.9764 | 0.0196 | 0.9959 | 0.0223 |
| ArcFace | 0.9849 | 0.9699 | 0.0000 | 0.9896 | 0.0249 |

Custom Fold 0, one split, one seed, balanced test pairs (764 + 764), frozen validation threshold. These are not View 2
numbers and carry no significance claim.

## Known documentation caveat in the code

The module docstring of `faceeval/analysis/fold0_results.py` refers to a file `STATISTICAL_LIMITATIONS.md`. That file does
not exist and the module does not write it; the limits are stated in `docs/PROTOCOL.md`, section 10.
