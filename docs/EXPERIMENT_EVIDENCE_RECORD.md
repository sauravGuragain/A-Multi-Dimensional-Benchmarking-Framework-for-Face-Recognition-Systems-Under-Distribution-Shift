# FaceEval-X Experiment Evidence Record

Written 2026-11-18; the addenda and the rows marked 2026-11-19 were updated against `main` at commit `44c305c`. This file records **only evidence that exists**. Every fact carries one of four labels:

| Label | Meaning |
|---|---|
| **REPO** | Observed in files committed to this repository (commit `98774f3`, the state read when this record was written). |
| **MAC (reported)** | Reported by the maintainer from their Mac. The author of this record did not inspect those files. |
| **NOT RECORDED** | The run or the project never saved this information. It is not guessed and not reconstructed. |
| **NEEDS NEW RUN** | Could only be established by running something new. Not part of the thesis scope. |

## 1. Two protocols, never to be mixed (and Fold-0 is custom)

| | Closed-set **identification** | Identity-disjoint **verification** (custom Fold 0) |
|---|---|---|
| Question | Whose face is this, among known people? | Are these two faces the same person? |
| Split | Image-level stratified (the same people appear in training and test) | Identity-disjoint (test people are never seen in training) |
| Decision | Classifier or nearest enrolled centroid | Similarity threshold frozen on validation pairs |
| Metrics | Top-1 accuracy, F1 | Balanced accuracy, TAR, FAR, EER, AUC |
| Config key | `evaluation_protocol: identification` (default) | `lfw_fold0_verification` |
| Runs | 2bad8efd (primary), fe4696ee (superseded), 69878560 (invalid) | Classical: 5ecfdd99, a55a4a37. Deep (full 16 x 6 sweeps): b567f880 (FaceNet), 4bdf6982 (ArcFace). Preliminary clean-only deep checks: b8fbb5c3, c069be84. Plus `c408e0d2`, a **separate per-image-noise run** (see section 4) |

"Accuracy" under one protocol is not comparable with "accuracy" under the other. Published LFW numbers (about 99%) are verification numbers on the **official LFW View 2 protocol**. The Fold-0 protocol used here is this project's own custom identity-disjoint split (`reports/experiment_results/README.md`: "Custom identity-disjoint Fold 0", full LFW, seed 42, 10 folds, threshold chosen on clean validation pairs and frozen). It is **not** the official View 2 protocol, so its numbers must not be compared with published View 2 results either.

## 2. Primary result: run 2bad8efd (the thesis experiment)

| Field | Value | Label |
|---|---|---|
| Run ID | `2bad8efd2ac0468cb1bf54fc517e7119` | MAC (reported) |
| Experiment name | `lfw_intermediate_rerun` | MAC (reported) |
| Status in the local SQLite database | `completed` | MAC (reported) |
| Protocol | Closed-set identification | REPO (config default) |
| Models | Eigenfaces, HOG-SVM, FaceNet, ArcFace | MAC (reported) |
| Perturbations / severities | Gaussian blur, rotation, brightness / 0.0, 0.5, 1.0 | MAC (reported); mappings REPO |
| Result rows / test samples | 36 rows / 1,357 samples | MAC (reported) |
| Config base | `configs/lfw_intermediate.yaml` (REPO) and `configs/lfw_intermediate_rerun.yaml` (new names, `require_real_models: true`), which is **committed in the delivery PR**. Byte-for-byte identity of the committed file with the configuration file used for the historical run cannot be independently verified: the run database stores a `config_hash` of the configuration object, not a hash of the YAML file, and no file hash was recorded. | REPO (committed file); identity with the historical file NOT RECORDED |
| Seed and split | seed 42; test 0.30; validation 0.10; at least 8 images per subject; 2,734 / 461 / 1,357 images | REPO (config); split sizes MAC (reported) |
| Runtime and memory | 3,492 s (about 58 minutes), peak RSS about 3.56 GB, no swapping; ArcFace on CPU | MAC (reported, from the pasted terminal log) |
| Result files | `reports/lfw_intermediate_rerun/{evaluation_results,fingerprints,failure_clusters,deployment_rankings}_2bad8efd2ac0468cb1bf54fc517e7119.csv` on the Mac. **Not in Git.** | MAC (reported) |
| Local worktree | the local verification worktree, branch `main`, commit `98774f3af42fe032f5bb4b5a474d2204a828e9b5`, with untracked user files | MAC (reported) |
| Code commit **at the time of the run** | The run stores a short HEAD in its database record (`config.git_commit`). Its value has not been provided. | read it with `./run_faceeval.sh evidence 2bad8efd` |
| Whether the working copy was modified at run time | NOT RECORDED (the run stores the commit only, not a dirty flag) | NOT RECORDED |
| Python and package versions at run time | NOT RECORDED (the environment snapshot is captured but only its platform line is logged) | NOT RECORDED |
| Model-weight hashes | NOT RECORDED | NOT RECORDED |
| Dataset fingerprint at run time | NOT RECORDED | NOT RECORDED |
| SHA-256 of the result CSVs at run time | NOT RECORDED. Hashes computed today describe the files as they are now. | NOT RECORDED |
| Real (non-stub) backbones | The run was started with `require_real_models: true` and completed; clean accuracy FaceNet 99.78%, ArcFace 99.63%. | MAC (reported) |

Reported values (from the maintainer's pasted output; accuracy in %, severity 1.0 unless stated):

| Model | Clean | Blur 0.5 | Blur 1.0 | Rotation 0.5 | Rotation 1.0 | Brightness 1.0 |
|---|---|---|---|---|---|---|
| Eigenfaces | 21.6 | 16.2 | 12.5 | 9.7 | 5.0 | 13.8 |
| HOG-SVM | 22.3 | 14.4 | 11.9 | 12.3 | 11.7 | 21.1 |
| FaceNet | 99.8 | 59.9 | 1.0 | 99.2 | 77.3 | 98.0 |
| ArcFace | 99.6 | 57.8 | 0.2 | 99.6 | 99.6 | 99.3 |

Internal consistency, checked arithmetically on these reported values: each robustness AUC equals 0.25 a(0) + 0.5 a(0.5) + 0.25 a(1); each deployment `score_accuracy` equals the mean accuracy over the nine conditions of that model; each total equals 0.20 x accuracy score + 0.80 x 0.5.

### What `./run_faceeval.sh evidence 2bad8efd` adds

It is read-only. It reads the database from a temporary copy made in the system temp folder (deleted on exit), so no journal, WAL or lock file is ever created next to the original. Part A prints what the run recorded in the database (status, timestamps, duration, `config_hash`, `git_commit`, seed, protocol). Part B prints today's SHA-256 of the result CSVs with their modification times, the config files, the dataset file-list fingerprint and the cached weights. Part B is evidence about the historical run only if nothing changed since; the modification times let a reader judge that. Paste the printed block into section 8 below.

## 3. Interpretation limits that apply to run 2bad8efd

- Brightness: the shift is `(severity - 0.5) x 0.8`, so severity 0.5 equals the clean image, and severity 0.0 returns an unperturbed copy. Only one brightening level (+0.40) is tested; darkening is not.
- Only 3 of 16 perturbations and 3 of 6 severity levels were run. 200 of 5,749 subjects (3.5%). 4 of 10 models.
- The deployment ranking is accuracy-driven: robustness, calibration, fairness, latency, memory and cost are placeholders (0.5). It does not validate the multidimensional framework.
- The fingerprint has 21 dimensions; 3 are populated.
- One run, one split, seed 42; no confidence intervals. An approximate 95% binomial interval at n = 1,357 is about +/-2.7 points at 60% accuracy (a lower bound, because images of one person are not independent).
- Image-level split: near-duplicate photos of a person may appear in training and test.

## 4. Supplementary and historical experiments

| Run | Protocol | What is known | Label |
|---|---|---|---|
| `5ecfdd995ffc4e528b0c17563798a7d4` (`lfw_fold0_classical`): Fisherfaces, LBPH, HOG-SVM; 288 rows = 16 x 6 x 3 | Fold-0 verification | COMPLETE, failures 0, stub 0; code commit `34608f4eb8f1e073c254d53cbae9e64c82150710`; manifest SHA-256 `d36e8ae060eae1bea9df663351b00f44f5156330c06f55ad77234ef8fc7bdd05` (the committed sidecar `split_manifest_fold0.tsv.sha256` contains the same hash) | REPO: `reports/experiment_results/classical/`, README; also a local DB record |
| `a55a4a37f1664bf89bb2987f01c45683`: Eigenfaces final; 96 rows | Fold-0 verification | COMPLETE; same manifest and commit as the row above | REPO: `reports/experiment_results/eigenfaces_final/` |
| `c408e0d2352e4de780a9d7a45a3ae7cc`: Eigenfaces **per-image-noise run**; 96 rows | Separate per-image-noise run on the custom Fold-0 setup | Historical; different manifest (SHA-256 `3c1cf7a77791a2c42cdf05f5115594fefb904cc602f3be5a77249f967b0a7116`) and code commit `e0d0eae8444e93b6110740aae81442385048316b`; do not combine with other runs. The committed `verification_results.json` of all three archived runs (5ecfdd99, a55a4a37, c408e0d2) records noise protocol `per_image_v2`, the same noise-module hashes, the same split content hash (`073c6291…`) and the same pair content hashes; only the manifest SHA and the recorded code commit differ. Comparing the committed CSVs of a55a4a37 and c408e0d2 shows identical accuracy, TAR and FAR; only AUC and EER differ (largest difference 0.00026 AUC and 0.00365 EER), consistent with commit 34608f4 ("Fix exact verification AUC and EER calculations") lying between the two code commits. This is an inference from the repository, not a recorded statement. Use a55a4a37 for the thesis tables. | REPO: `reports/experiment_results/eigenfaces_per_image_noise/` |
| `b567f88089c142d3b7732d993f34c979`: FaceNet, full sweep; 96 rows = 16 x 6 | Fold-0 verification | COMPLETE, 96/96 conditions (details in the final addendum) | MAC (reported); config REPO (`configs/lfw_fold0_deep_facenet.yaml`). Result files not in Git. |
| `4bdf6982668f4f498fa3fe36d41c885b`: ArcFace, full sweep; 96 rows = 16 x 6 | Fold-0 verification | COMPLETE, 96/96 conditions (details in the final addendum) | MAC (reported); config REPO (`configs/lfw_fold0_deep_arcface.yaml`). Result files not in Git. |
| `b8fbb5c3` (`lfw_fold0_deep_clean_check`): FaceNet, ArcFace; 2 rows, clean only | Fold-0 verification | COMPLETE, failures none, per the 2026-11-15 handoff; present in the local database | REPO: `docs/experiment_records/2026-11-15_session_handoff.md`; MAC (reported) |
| `c069be84b7f84008a2b90975d384d8bf`: FaceNet, ArcFace; 2 rows, clean only ("v2") | Fold-0 verification | COMPLETE, stub 0; commit `930b49d96a39ce3de4bc072758eb842c3f981a51`; validation balanced accuracy FaceNet 0.9905, ArcFace 0.9956 (validation, not test) | REPO: `docs/experiment_records/lfw_fold0_deep_clean_check_v2.md` |
| `fe4696ee986c43f8b7920c5a5416f20f` (`lfw_intermediate_scale_validation`) | Identification (same design as the primary run) | **Superseded.** FaceNet numbers came from preprocessing and perturbation defects (clean 61.6% versus 99.78% in the rerun); other values also differ in places. Never place its numbers beside 2bad8efd. | REPO: `reports/lfw_intermediate_scale_validation/`; MAC (reported) as completed |
| `69878560...` | Identification | **Invalid.** ArcFace at chance level. Not in Git. | earlier audit output; MAC |
| Six-model Fold-0 analysis output (generated with `python -m faceeval.analysis.fold0_results`; 576 primary rows, 13 tables, 7 figures; `table10` 2,880 rows) | Fold-0 analysis | Reported by the maintainer as generated successfully on the Mac. The output directory is **not in Git** and was not inspected by the reviewer. | MAC (reported) |
| `reports/experiment_results/analysis/`, `analysis_20261114/` | Earlier (classical) Fold-0 analysis outputs | Reported to exist on the Mac. They were **not present** in the Git checkout of `main` that this record was written from (only `classical/`, `eigenfaces_final/`, `eigenfaces_per_image_noise/` and the README are tracked). The folder `reports/` is git-ignored, so such folders stay local unless force-added. | MAC (reported) |

The relationship between `b8fbb5c3` and `c069be84` (two clean-only deep checks, one described as "v2") is as written in the two committed documents; this record does not resolve it further. Both are two-row clean-condition checks and are **not** the 96-condition benchmark; the committed handoff and v2 documents say so and are historical records that were left unchanged.

## 5. Tests and checks

Only focused checks were run for this delivery. The full suite was **not** rerun.

| Statement | Label |
|---|---|
| The full suite passed with 983 tests in the maintainer's Mac environment. The commit and date of that run are not recorded here. | MAC (reported); commit NOT RECORDED |
| A separate cloud run at commit `98774f3` passed 1,172 tests. The two counts are not reconciled. | cloud (Linux sandbox), not the Mac |
| Real-backbone enforcement: `test_verification_stage.py::test_require_real_models_blocks_stub` and the `real`/`Preflight` tests in `test_audit2_fixes.py`: 5 passed. | cloud, not the Mac |
| Models touched by the docstring fixes: `pytest tests -k "arcface or hog or insightface or facenet"`: 83 passed, 1,089 deselected. | cloud, not the Mac |
| `tests/test_perturbation_value_range.py`: 386 passed (an earlier focused run). | cloud, not the Mac |
| `run_faceeval.sh` read-only commands (`help`, `check`, `results`, `results 2bad8efd`, `evidence`, `evidence 2bad8efd`): a hash, size and modification-time snapshot of the whole scratch tree, including hidden files and `.git`, was identical before and after, with a clean WAL-mode database and with a database held open by a live writer (`-wal` and `-shm` present). Scratch tree used **fixture data**, not your results. | cloud, not the Mac |
| Exit status through the tee and awk pipeline is propagated (child exit 0 -> 0, 3 -> 3) and the `PREDICTIONS DEBUG` filter hides lines on screen but not in the log. | cloud, not the Mac |
| Commit `44c305c` adds `faceeval/analysis/fold0_results.py` and `tests/test_fold0_analysis.py` (SHA-256 of both committed files equals that of the files the reviewer delivered and tested earlier in a Linux sandbox). These tests use **synthetic fixtures** and the committed classical archives, not the real FaceNet or ArcFace results. | cloud (Linux sandbox), not the Mac |
| `bash -n run_faceeval.sh`, `help`, `check` (0 warnings), `results 2bad8efd` and `evidence 2bad8efd` succeeded on the maintainer's macOS Bash 3.2. The SHA-256 of the file tested is **NOT RECORDED**. Experiment-starting commands and clean-machine installation have not been verified. | MAC (reported) |
| SHA-256 of `run_faceeval.sh` as committed in `4a9788282b9b3ab91ccd10d935644671cba18d3e`: `4c621819991195d59255b772216a5f8cb5539c1cb60eb59022f890f283132637`, computed from the GitHub source by the reviewer in a Linux sandbox. If `shasum -a 256 run_faceeval.sh` on the Mac prints a different value, the Mac-tested file was a different build. | REPO (derived); not a Mac result |

## 6. Capability versus validation

| Capability | Implemented | Validated in the thesis |
|---|---|---|
| Subjects | all of LFW (5,749) | 200 |
| Models | 10 registered | 4 (identification); 4 classical + 2 deep across 16 x 6 conditions (Fold 0) |
| Perturbations | 16 | 3 (identification); 16 for 4 classical + 2 deep models (Fold 0) |
| Severities | 6-level grid | 3 levels (identification); all 6 levels in Fold-0 verification |
| Deployment dimensions | 7 | 1 populated (accuracy) |
| Fingerprint dimensions | 21 | 3 populated |

Placeholders (0.5 in every output; not findings): robustness (the runner does not pass fingerprints to the scorer), calibration, fairness, latency, memory, cost. CSV `auc` and `eer` columns in identification output are 0.0 and must not be cited.

## 7. Code-level limitations established from the source (documented, not changed)

| Finding | Why it is documented rather than changed |
|---|---|
| `require_real_models` defaults to `false`, and `configs/lfw_intermediate.yaml`, `lfw_smoke_test.yaml`, `quick_test.yaml` and `lfw_full_benchmark.yaml` do not set it. Run directly from those YAML files, a missing backbone can fall back to a stub. The Fold-0 configs do set it. | Editing a committed config changes its hash and its traceability. `run_faceeval.sh` always writes a derived config with the flag on. |
| The deployment scorer receives no fingerprints, so the robustness dimension is 0.5 for every model. | Fixing it changes future outputs. The thesis states the ranking is accuracy-driven. |
| `ReportGenerator.export_json` exists, but the runner does not call it: it logs "JSON export requested but not yet implemented." No JSON report is produced by a run. | Wiring it adds a failure point after an hour-long run, and the thesis needs no JSON. |
| The environment snapshot (Python, packages, hardware) is captured at setup, but only the platform line is logged; it is not stored in the run record. The run records the short git commit but no dirty flag. | Explains the NOT RECORDED entries above. |
| `lfw_intermediate_rerun.yaml` sets `compute_calibration`, `compute_fairness` and `profile_resources` (and its description text speaks of fairness computation). These flags and the description were inherited from `lfw_intermediate.yaml`. The three inherited flags are **not consumed** by `ExperimentRunner`. Of the `compute_*` and `export_*` flags, the runner reads only `compute_fingerprints`, `compute_deployment_scores`, `export_csv` and `export_json`. No calibration, fairness or resource result exists for the thesis run. | The YAML is left unchanged so that it stays what it is; this note is the correction. |
| `ExperimentTracker` (MLflow) is defined but the runner never creates it; `mlflow_tracking_uri` is only copied into the configuration object. A run does not start MLflow tracking. | Documented, not changed. |
| `BasePerturbation.apply` returns an exact copy at severity 0.0; brightness 0.5 is the identity. | By design and covered by `test_perturbation_value_range.py`; the thesis now says so. |
| Docstrings said HOG length about 2,916 (actual 8,100) and ArcFace `buffalo_l` ResNet-100 (actual recognition model ResNet-50, `w600k_r50`). | **Fixed** in this change (comments and one description string only). Result CSVs are untouched; the thesis text was corrected in parallel. |
| The module docstring of `faceeval/analysis/fold0_results.py` refers to a file `STATISTICAL_LIMITATIONS.md`. The module writes only `tables/`, `figures/` and `provenance.json`; it writes no such file. | Documentation-only update; the source was not changed. The statistical limitations are stated in this record and in the README. |
| The deep-model metrics are end-to-end: failed embeddings are zero vectors and their pairs are still scored (`faceeval/orchestration/verification.py`). The six-model analysis module computes no valid-pairs-only metric; the separate `faceeval.analysis.valid_pairs` tool does (Addendum 3). | Changing it would change what a completed result means. The failure counts are in the CSVs and in `table12` / `table13`. |

No source change in this delivery alters the meaning of any completed result.

## 8. Evidence block from `./run_faceeval.sh evidence 2bad8efd`

```
NOT YET PASTED. The maintainer reports that the command succeeded on the Mac, but its output is not available to this record. Paste it here if you want the recorded commit, config_hash, seed and timing of 2bad8efd on record. Until then those values are NOT RECORDED in this document.
```

## 9. Evidence that would require a new run (not planned, not needed for the thesis claims)

- A repeat of 2bad8efd to test repeatability.
- The complete 10-model benchmark (only 6 models have completed the documented Fold-0 verification sweeps).
- Calibration, fairness, speed and memory measurements.
- Confidence intervals from multiple splits.
- A clean-machine installation.

## Addendum: provenance, centred perturbations and deep-model sweep status (repository evidence)

- **Fold-0 summaries recorded `environment` only** (python, numpy, code commit; no dirty flag). Runs started after this change also write a `provenance` block to `verification_results.json`: platform, machine, `code_dirty` (tracked files only), versions of torch, torchvision, facenet-pytorch, insightface, onnxruntime, onnx, opencv-python, scikit-image, scikit-learn, scipy and Pillow, and SHA-256 of the default cached FaceNet (`vggface2`) and ArcFace (`buffalo_l`) weight files. Anything undeterminable is written as `not_recorded`. Hardware and a config-file hash remain NOT RECORDED. Runs made earlier have no such block.
- **Centred perturbations.** In the 0.0-1.0 grid, brightness (shift = (s-0.5)*0.8), contrast (scale 0.3 to 3.0, near identity at about 0.5) and gamma (0.25 to 4.0, identity at 0.5) are two-sided. Severity 0.0 returns the clean image for every type, so these three types are not monotone in severity. Interpret their curves as distance from the neutral centre, and do not read the mean over 0.2-1.0 as "average degradation" for them. This is existing behaviour, documented in `tests/test_perturbation_value_range.py`; no code was changed.
- **Deep-model sweep status.** Both deep-model Fold-0 verification sweeps are COMPLETE, each with 96/96 conditions. Per-condition failed-embedding and failed-pair counts are retained in the verification CSVs; these are benchmark outcomes, not run-completion failures. FaceNet run ID `b567f88089c142d3b7732d993f34c979` completed in 7,887.1 seconds; its outputs and log are preserved in `runs/full_lfw_fold0_deep_facenet_20261118_173444/`. Its verified evidence archive is `~/Thesis/faceeval_x_full_evidence/facenet_full_20261118_173444.tar.gz` (SHA-256 in the adjacent `.sha256` file). ArcFace run ID `4bdf6982668f4f498fa3fe36d41c885b` completed in 15,826.8 seconds; its outputs and log are preserved in `runs/full_lfw_fold0_deep_arcface_20261118_203249/`. Its verified evidence archive is `~/Thesis/faceeval_x_full_evidence/arcface_full_20261118_203249.tar.gz`, SHA-256 `02fe2d1542adfd414cf3dc479582be5533d8867d19e74ae2d0532400dcd21f49`. Both configs require real pretrained weights. The verification stage has no resume: an interrupted run must be restarted into a new run directory.

## Addendum 2 (2026-11-19, repository at `44c305c`): six-model Fold-0 benchmark and analysis

Labels are the same as above. This addendum adds no new experimental result: every number from the Mac is the maintainer's report and the reviewer has not seen those files.

| Statement | Label |
|---|---|
| The Fold-0 verification benchmark covers six models (Eigenfaces, Fisherfaces, LBPH, HOG-SVM, FaceNet, ArcFace) over 16 perturbations x 6 severities = 96 conditions per model, 576 primary rows. | MAC (reported) for the deep models; REPO for the four classical models (384 rows committed) |
| FaceNet (`b567f880...`) and ArcFace (`4bdf6982...`) completed 96/96 conditions each. The FaceNet and ArcFace verification CSV/JSON files are not in Git. | MAC (reported); not in Git |
| The six-model analysis ran on the Mac and produced 13 CSV tables and 7 figures; `table10` has 2,880 rows (576 conditions x 5 metrics). | MAC (reported); NOT VERIFIED by the reviewer |
| The generated analysis output is a local artifact and is not committed. `reports/` is git-ignored. | MAC (reported) and REPO (`.gitignore`) |
| The analysis code and its tests are committed in `44c305c`. | REPO |
| The analysis command is `python -m faceeval.analysis.fold0_results --results-root ... --facenet-run DIR --arcface-run DIR --out NEW_DIR` (options documented in the README). The command line was read from the source; it was not run on the real archives by the reviewer. | REPO |
| Whether the FaceNet and ArcFace JSONs contain a `provenance` block (package versions, weight hashes) was not checked. The block exists only for runs started after commit `42c2f44`. | NOT VERIFIED |
| ArcFace failure counts: the maintainer's earlier read-only audit reported FaceNet with 0 failed embeddings and ArcFace with 12,969 failed embeddings **summed across the 96 conditions**. That sum counts the same image once per condition, so it is not a number of unique failed images. | MAC (reported); not re-derived by the reviewer |

**Failure semantics.** Deep-model metrics are end-to-end. A failed embedding becomes a zero vector and every pair containing it is still scored, so failed pairs are not removed. `n_failed_embeddings`, `n_failed_pairs_balanced`, `n_failed_pairs_full`, `n_test_images` and `val_n_failed_embeddings` are recorded per condition. ArcFace AUC/EER are therefore not valid-pairs-only recognition performance. A valid-pairs-only analysis from `scores/*.npz` is provided by `faceeval.analysis.valid_pairs` (Addendum 3); it has not been run on the real scores.

**Pipeline difference (read from source at `44c305c`).** FaceNet: `CentreCropAligner`, crop fraction 0.8, 160 x 160, InceptionResNetV1 (VGGFace2), no detector. ArcFace: `FaceAligner` with 5-point landmark alignment to 112 x 112, InsightFace `buffalo_l` (recognition model `w600k_r50`), internal face detector, cropping fallback off. FaceNet versus ArcFace is an end-to-end pipeline comparison, not a backbone-only comparison.

**Protocols stay separate.** Run `2bad8efd` is image-level closed-set identification on 200 subjects (4 models, 3 perturbations x 3 severities, 36 rows). The Fold-0 runs are custom identity-disjoint verification (test set: 764 genuine plus 764 balanced impostor pairs, with a 7,640-pair impostor pool; threshold chosen on clean validation pairs and frozen). Their numbers are not comparable with each other or with published LFW View 2 results and must not be put in one table. The historical run `c408e0d2` is kept out of the primary rows (it appears only in `table11` of the analysis).

**Centred perturbations.** Brightness, contrast and gamma are two-sided with neutral near severity 0.5; severity 0.0 is the clean image for every type; their curves are V-shaped (see the first addendum).

**Remaining limitations (documented, not changed).** One split and one seed; indicative intervals only, no significance tests; pretrained-weight training data not audited for LFW identity overlap; calibration, fairness, speed and memory not evaluated; a clean-machine installation has not been tested.

## Addendum 3: valid-pairs-only tool, efficiency harness and remaining-task status

| Item | Status | Label |
|---|---|---|
| `faceeval/analysis/valid_pairs.py` (+ `tests/test_valid_pairs.py`) | Implemented. Tested only on synthetic fixtures built with the production helpers of `verification.py`. Not run on the real FaceNet/ArcFace score files. | cloud tests (Linux sandbox); real-data run NEEDS MAC |
| `faceeval/analysis/efficiency.py` (+ `tests/test_efficiency.py`) | Implemented. Self-tested with a synthetic pseudo-model; the stub-refusal path is tested through the real `facenet` registry entry on a machine without FaceNet weights. No real timing or memory number exists. | cloud tests; real measurement NEEDS MAC |
| Calibration, fairness | Superseded by Addendum 4: calibration is now integrated into the Fold-0 runner (validation scores are saved in a sibling directory); fairness has an offline tool that needs external attribute labels. | See Addendum 4 |
| Runtime and memory | `ModelProfiler` uses `tracemalloc` (Python heap only), which under-reports PyTorch/ONNX memory; use the new harness. | Harness implemented; NOT measured |
| Multiple Fold-0 splits or seeds | Superseded by Addendum 4: `verification_fold` and `random_seed` select other identity-disjoint splits, with leakage and pair-count invariants replacing the official G/K counts for non-reference splits. The reference split (seed 42, fold 0) is unchanged. | Implemented; real multi-split run NEEDS MAC |
| 10-model benchmark | `configs/lfw_full_benchmark.yaml` exists and has never been run. Per-model Fold-0 configs now exist for the four extra models; all ten models execute through the same runner on fake data (Addendum 4). Real insightface/dlib weights were never loaded. | NOT validated on real data |

## Addendum 4 (2026-11-19): completion work on top of `aa8e83a` (cloud-tested; nothing here was run on the Mac)

Every row states what is evidenced. "Cloud" means the Linux sandbox with fake or synthetic data; it does not establish any real-model result.

| Item | Evidence level | Where / how to verify |
|---|---|---|
| Calibration in the runner | **Implemented; cloud-tested.** Platt fit on clean validation balanced pairs only; ECE/MCE/Brier/NLL before and after; validation scores saved; run directory, manifest and rows byte-identical with the flag on or off. No real-model result. | `faceeval/evaluation/verification_calibration.py`, `tests/test_verification_calibration.py` |
| Multi-split / multi-seed | **Implemented; cloud-tested.** `random_seed` and `verification_fold` select the split; invariants (identity/image disjointness, genuine/impostor validity, balanced = genuine count, pool = rho x genuine); aggregation by mean, sample SD, Student-t interval with the not-independent caveat. Reference split unchanged. | `faceeval/orchestration/verification.py`, `faceeval/evaluation/split_invariants.py`, `faceeval/analysis/multi_split.py`, `tests/test_multi_split.py` |
| Ten models | **Execution path validated on fake data**; placeholders are labelled `STUB` and `require_real_models: true` refuses them. Real insightface/dlib weights never loaded. | `tests/test_ten_models.py` |
| Fairness | **Interface implemented; synthetic tests only.** Needs an external `image_id,group` CSV. Nothing is inferred; missing labels are unknown. No LFW fairness result is claimed. | `faceeval/analysis/fairness.py`, `tests/test_fairness.py` |
| Runtime / memory | **Harness implemented; synthetic self-test only.** Adds CUDA synchronisation, GPU peak (PyTorch allocations) and GPU name. No real measurement. | `faceeval/analysis/efficiency.py`, `tests/test_efficiency.py` |
| Valid-pairs-only | Unchanged from Addendum 3: not yet run on the real archives. | `faceeval/analysis/valid_pairs.py` |
| JSON export | **Implemented; cloud-tested.** Atomic write, `schema_version` 2. | `faceeval/reporting/generator.py`, `tests/test_json_export.py` |
| Evidence index | **Implemented; cloud-tested.** Lists which evidence products exist; claims nothing about missing ones. | `faceeval/analysis/evidence_index.py` |
| Install | Lightweight install (`pip install -e ".[dev]"`) validated in a clean venv with `pip check`; pins `numpy<2`, `opencv-python<5`. Deep (`.[deep]`) and dlib extras **not installed or tested** (torch wheels exceed the sandbox disk). | `pyproject.toml`, `.github/workflows/ci.yml` |

**Validated** (cloud, real code paths): all items above on fake or synthetic data. **Implemented, not validated on real data:** every row marked so. **Unsupported:** fairness conclusions on LFW (no labels); comparison with published LFW numbers (custom Fold 0 is not official View 2). **Limitations:** single-fold results have no population-level interval; multi-split intervals measure split sensitivity only; the balanced 50% base rate affects calibration (ECE/Brier) interpretation.

**Audit corrections made after the first delivery of this addendum** (found by the pre-Mac audit; all cloud-tested):
- `ExperimentConfig.to_dict()` does not serialise `verification_fold` when it is the default 0, so the `config_hash` (hence `run_id` and the result store's duplicate-run detection) of every pre-existing shipped config is identical to its value at `aa8e83a` (`tests/test_config_hash_stability.py`). The free-text `description:` of `configs/lfw_fold0_classical.yaml` still says that calibration is "not read by the runner under any protocol"; that sentence is historical and was deliberately NOT edited, because the description is part of the hashed config. `# comment` lines are not hashed and were corrected.
- Weight-file SHA-256 is now recorded in `provenance.weights` also for `insightface` (antelopev2) and `dlib_fr` when their default cache files exist (otherwise `NOT_RECORDED`). `dlib_fr` downloaded its weights over plain HTTP without a checksum (SUPERSEDED in Addendum 5: HTTPS from a pinned commit with pinned SHA-256).
- The efficiency harness records `device_used`; GPU peak memory is reported only for a torch CUDA device (ONNX Runtime and dlib models are labelled as not measured). `auto` never selects MPS, so on a Mac the torch models run on the CPU (Addendum 5 adds an explicit `--device mps`).
- The fairness tool refuses (and writes nothing) when no attribute row matches a test image id, and records the attribute file's SHA-256 and a `--source` note in `fairness_coverage.json`.
- The multi-split aggregator reports whether the test identity sets are disjoint (`design` in `multi_split_provenance.json`) and words the interval interpretation accordingly.

## Addendum 5 (2026-11-19): remaining engineering issues from the pre-Mac review (cloud-tested; nothing here was run on the Mac)

| # | Issue | Resolution | Evidence level |
|---|---|---|---|
| 1 | `dlib_fr` used the generic ArcFace-template aligner; weights downloaded over HTTP | New `DlibChipAligner` (HOG detector, 5-point landmarks, `get_face_chip` 150 px / padding 0.25; largest face; see Addendum 6 for the no-face policy, which corrects this row). Opt-in through `pipeline_for_model(alignment="dlib_chip")` and config `dlib_alignment` (default `dlib_chip` for `dlib_fr` only; `legacy` keeps the old behaviour and changes the hash). Weights: HTTPS from `davisking/dlib-models@fd81b63...`, archive and file SHA-256 pinned, atomic write, mismatch refused. | Cloud: unit tests with a fake dlib; **real dlib 20.0.1 in the sandbox**: the pinned HTTPS download verified; the chip equals a direct dlib call; pipeline embeddings equal a direct dlib descriptor (cosine 1.0000 on two sample images); the generic aligner gives a different embedding (cosine about 0.86), which confirms the alignment matters. **No LFW run; no comparability claim.** |
| 2 | InsightFace version unpinned | The ArcFace run's package versions are not in the delivered files, so none can be derived. `faceeval.analysis.env_pins` prints/checks the pins from a run's `provenance.packages` and refuses when that block is missing. | Tool cloud-tested; the real pin needs the Mac run directory. |
| 3 | Efficiency harness | Separate init time, cold first calls, warm-up, preprocessing, batch-1 latency, batched throughput, peak RSS after load and overall; `--device auto/cpu/cuda/mps/best` with a pure, hardware-free selection function (refuses unavailable devices); MPS synchronised and its current (not peak) allocation recorded; CUDA peak recorded; ONNX/dlib models labelled as not torch devices; comparability block lists differing conditions. | Cloud: 42 tests including CPU/CUDA/MPS selection with stand-in torch. **No real timing.** |
| 4 | Multi-split statistics | Design classified (folds of one seed / different seeds / mixed), model kind labelled, interval status per row, Nadeau-Bengio corrected interval for trained models, no interval claimed for overlapping seeds or fewer than 5 splits. | Cloud: synthetic runs. |
| 5 | PCA/LDA randomness | PCA randomized solver and SVMs use the global NumPy stream (confirmed by test: output changes with prior stream use). Historical default unchanged (`classical_rng: global_seed`, not serialised). New separately hashed `isolated_seed` protocol. The resolved PCA solver is recorded in `provenance.classical_rng`. | Cloud. Whether the Mac's historical results equal the isolated protocol is unknown; do not assume it. |
| 6 | Stale classical `description:` | Not edited (hashed). Documented here and in USER_GUIDE; test shows comments are not hashed and the description is. The calibration flag IS read under `lfw_fold0_verification`. | Test. |
| 7 | Figures | `faceeval.analysis.figures` (calibration, multi_split, valid_pairs, efficiency): reads result files only, writes PNG + plotted data CSV + provenance with input SHA-256, refuses missing/invalid input, never overwrites. | Cloud: tests on synthetic input files. No figure from real results exists yet. |

## Addendum 6 (2026-11-19): corrective patch after the independent review of `f6082e4` (cloud-tested; nothing here was run on the Mac)

| Finding | Correction | Research impact |
|---|---|---|
| **Correction of Addendum 5, item 1 (no-face policy).** The experiment runner calls the aligner directly and treats `None` as a fatal error for the whole model, so "no face -> failure (zero embedding)" never applied to `dlib_fr` in a run: the first image dlib could not detect would have aborted the model. | `DlibChipAligner` now sends a fixed centred square (0.5 x shorter side) to the landmark model when dlib finds no face, labels the chip `dlib_5pt_face_chip(centre_box_fallback)`, counts it, and records the counters in `provenance.classical_rng.dlib_alignment_stats` (the key name is historical: it also holds alignment counters). A run where more than 50% of images need the fallback is refused. | For `dlib_fr`, an undetected face is NOT a zero embedding and is NOT counted in `n_failed_embeddings`; its end-to-end failure accounting is therefore **not comparable** with ArcFace's. The centre-box share must be reported next to any dlib_fr number. The box fraction (0.5), the 50% refusal and the 10% warning thresholds are engineering choices, not validated on LFW. |
| D1 | Unloadable, missing or checksum-invalid landmark weights raise `FatalPreprocessingError` (checked before any image is processed; `pipeline.process` re-raises it). `require_real_models: true` with dlib missing fails closed. Without dlib and without `require_real_models` the generic aligner is used, labelled truthfully, warned once. A deep model whose CLEAN validation embeddings fail above 50% is refused. | A broken installation can no longer produce a finished-looking run. The 50% guard is new run-level behaviour; it cannot trigger on a run that already completed with fewer failures. |
| D2 | `multi_split` refuses stub runs and runs that differ in protocol, threshold protocol, dataset scope, per-model preprocessing/alignment, classical RNG protocol or recorded weight hashes. Only seed and fold may differ. | Tables can no longer silently mix protocols or weights. Runs that predate provenance are accepted; their weights are listed as unverified. |
| D3 | `evidence_index` opens the efficiency files: synthetic rows give `INVALID`; the harness's `comparability` block is surfaced as a caveat. | A self-test can no longer be indexed as evidence. |
| D4 | Efficiency rows record `requested_batch_size`, `model_internal_batch`, `effective_device_batch`, `limited_by_model` and a label. | FaceNet batches above 32, and every ONNX/dlib batch above 1, are labelled as calls of B images, not network batches of B. |
| R6 / R7 | Calibration figure: markers per categorical condition, no connecting lines; all-NaN valid-pairs input raises `FigureError`. CI smoke-tests `--help` of every new analysis CLI. | Presentation and tooling only. |
| Mac-side failure (efficiency) | `current_rss_mb()` returned NOT_RECORDED on macOS without psutil, so `rss_before_load_mb`/`rss_after_load_mb` were empty and `test_setup_time_is_reported_separately_from_per_image_embedding` failed. It now uses Linux `/proc/self/statm`, then on macOS `proc_pidinfo(PROC_PIDTASKINFO)` through ctypes (libproc, no dependency; `pti_resident_size` in bytes), then optional psutil, else NOT_RECORDED. Current and peak RSS stay separate: `ru_maxrss` is never substituted for the current size. | Measurement tooling only. The macOS path is written from Apple's header and covered by stand-in tests in the Linux sandbox plus one real-libproc test that runs only on macOS; **its real behaviour is confirmed only by your Mac run**. |
| Review follow-up: `multi_split` silent acceptances | (1) The model set was taken from the first run only; every run must now contain exactly the same models, and a missing or extra model is named in a `MultiSplitError` (also enforced inside `check_comparable` for direct callers). (2) `provenance.weights.<model>` is parsed strictly: only the string `NOT_RECORDED` or a non-empty object of file name to 64-hex SHA-256 is accepted. An empty object, any other string, `null`, a list or a malformed digest is refused (also when every run has the same malformed value, and also with `--allow-unverified-weights`). Absent provenance/weights (older runs, or the runner's empty `weights: {}` for runs without pretrained models) stays legitimate. Unrecorded in all runs lists the model under `weights_unverified`; partial recording needs `--allow-unverified-weights`; differing hashes or file sets are always refused. (3) Same family: an unknown or non-object `classical_rng.protocol`, a non-list `stub_backbones`, a duplicated (model, perturbation, severity) row inside one run (it was counted as an extra split), an empty or non-numeric result row, and a condition-set difference now give specific errors; comparison fields recorded by no run (older files) are accepted but listed under `comparability.fields_not_recorded`. | Aggregation tooling only; no run, result file or provenance was modified and no protocol changed. Test fixtures that used placeholder digests (`aaa`) now use 64-hex digests, because that is the only form the runner writes. Cloud-tested only. |
