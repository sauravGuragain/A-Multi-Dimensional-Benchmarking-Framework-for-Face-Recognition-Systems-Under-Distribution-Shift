# FaceEval-X

A modular, configuration-driven benchmarking framework that evaluates face recognition systems under controlled distribution shift. It compares traditional machine-learning and pretrained deep models across perturbations (blur, rotation, brightness and 13 more), and produces accuracy tables, robustness fingerprints, failure clusters and a context-weighted deployment score.

It is the software foundation of the thesis project, *Multi-dimensional benchmarking framework for face recognition under distribution shift*.

## What is validated, and what is only implemented

**Read this before citing anything from the project.**

| Area | Status |
|---|---|
| Primary thesis experiment: LFW, 200 subjects, closed-set identification, Eigenfaces / HOG-SVM / FaceNet / ArcFace, Gaussian blur / rotation / brightness at severities 0.0, 0.5, 1.0 (36 conditions, 1,357 test images) | **Completed.** Run `2bad8efd2ac0468cb1bf54fc517e7119` (result files are on the maintainer's machine, not in Git). See `docs/EXPERIMENT_EVIDENCE_RECORD.md`. |
| Custom Fold-0 verification (identity-disjoint pairs, TAR/FAR), **classical models**: Eigenfaces, Fisherfaces, LBPH, HOG-SVM over 16 perturbations x 6 severities (96 conditions per model, 384 rows) | **Completed**; result files are committed in `reports/experiment_results/`. This is the project's own identity-disjoint split (see its README: "Custom identity-disjoint Fold 0"), **not the official LFW View 2 protocol**, so its numbers are not comparable with published View 2 results. A different protocol from identification; never combine them. |
| `c408e0d2...` Eigenfaces per-image-noise run | A **separate, historical** run with a different manifest and code version. Keep its results apart from the other Fold-0 runs. |
| Custom Fold-0 verification, **deep models**: FaceNet and ArcFace over the same 16 x 6 grid (96 conditions each) | **Completed** (FaceNet run `b567f880...`, ArcFace run `4bdf6982...`; both 96/96, as reported by the maintainer from the Mac; full IDs in `docs/EXPERIMENT_EVIDENCE_RECORD.md`). Their result files and evidence archives are on the maintainer's machine and are **not in Git**. Values are archived **end-to-end** results: an image with no embedding becomes a zero vector and its pairs are still scored (see "Six-model Fold-0 analysis" below). The earlier two-row clean-only deep checks (`b8fbb5c3`, `c069be84`) were a preliminary step and are not the benchmark. |
| Six-model Fold-0 analysis (Eigenfaces, Fisherfaces, LBPH, HOG-SVM, FaceNet, ArcFace; 6 x 16 x 6 = 576 primary rows) | **Code and tests are merged** (`44c305c`, `faceeval/analysis/fold0_results.py`, `tests/test_fold0_analysis.py`). **Reported by the maintainer from the Mac:** the analysis ran successfully and produced 13 CSV tables and 7 figures (`table10` has 2,880 rows = 576 conditions x 5 metrics). **The generated outputs are local artifacts and are not committed to Git.** Not independently verified by the reviewer. |
| Earlier identification runs `fe4696ee...` and `69878560...` | **Superseded.** Defective FaceNet preprocessing / non-real ArcFace. Kept only as audit history. |
| PCA+SVM, k-NN, InsightFace, dlib models | Implemented, **never run on LFW** |
| Calibration, fairness, speed, memory, sample efficiency | Code present, **not evaluated**. Scores are placeholders (0.5). |
| Deployment ranking | **Accuracy-only in practice.** Robustness, calibration, fairness, latency, memory and cost are placeholders. It is a demonstration of the scoring mechanism, not a validated multi-dimensional ranking. |
| Behavioral fingerprint | 3 of 21 dimensions populated (robustness AUC for blur, rotation, brightness). |
| Full benchmark (`configs/lfw_full_benchmark.yaml`: 10 models, 16 perturbations, 6 severities, all subjects) | **Never run.** Out of the thesis scope. |
| JSON report, figure export, Docker | Not produced / not built (a JSON exporter exists in `reporting/` but the runner does not call it; the thesis run exported 0 figures) |

### Evidence levels used in this repository

| Level | Meaning |
|---|---|
| **Implemented** | The code exists in this repository. |
| **Executed** | An experiment actually ran to completion (only the runs listed above). |
| **Verified on the maintainer's Mac (reported)** | The maintainer reports it from their own machine: the 983-test full suite (commit not recorded), the 2bad8efd run, the `results` view of that run, the two completed deep Fold-0 sweeps (FaceNet, ArcFace; 96/96 conditions each) and the six-model Fold-0 analysis run (576 rows, 13 tables, 7 figures). None of these result files is in Git. |
| **Verified in a Linux cloud sandbox (not the Mac)** | Focused tests and the runner's read-only behaviour: whole-tree before/after snapshots on fixture data for six commands and two database states (`docs/EXPERIMENT_EVIDENCE_RECORD.md` section 5). These do not establish success on macOS. |
| **Reported as run on the maintainer's macOS (Bash 3.2)** | The maintainer reports that `bash -n run_faceeval.sh`, `help`, `check` (0 warnings), `results 2bad8efd` and `evidence 2bad8efd` succeeded. The SHA-256 of the file that was tested is NOT RECORDED. The SHA-256 of the committed `run_faceeval.sh` is given in the evidence record, section 5. Experiment-starting commands and a clean-machine installation have not been verified. |
| **Planned / proposed only** | A repeat of the thesis run, the 10-model benchmark, calibration/fairness/speed/memory studies, multiple Fold-0 splits or seeds, a valid-pairs-only analysis of ArcFace from the saved score files, a clean-machine install. **None has been executed.** |

**Reduced scope of the identification validation (run `2bad8efd`):** 200 of 5,749 subjects (3.5%), 4 of 10 models, 3 of 16 perturbations, 3 of 6 severity levels. One run, one split, seed 42, no confidence intervals.

**Scope of the Fold-0 verification benchmark (a different protocol; never merged with the above):** 6 of 10 models, all 16 perturbations, all 6 severity levels, on one custom identity-disjoint split (seed 42; 764 genuine plus 764 balanced impostor test pairs). One split and one seed, so the intervals the analysis prints are indicative only and no significance test is claimed. The pretrained FaceNet and ArcFace weights were trained outside this project; whether LFW identities occur in their training data has not been checked here.

**Brightness note (identification run):** the brightness shift is `(severity - 0.5) x 0.8`, so severity 0.5 is the identity and 0.0 is returned unperturbed. Only one brightening level (+0.40) is effectively tested; darkening is not.

**Centred perturbations (all runs on the 0.0-1.0 grid):** brightness, contrast (scale 0.3 to 3.0) and gamma (0.25 to 4.0) are two-sided with their neutral point near severity 0.5, and severity 0.0 returns the clean image for every perturbation type. Their curves are therefore V-shaped, not monotone: read them as distance from the neutral centre, and do not read the mean over severities 0.2-1.0 as "average degradation" for these three types.

## Quick start (macOS / Linux)

```bash
git clone <this repository> && cd <repository folder>
python3.12 -m venv .venv && source .venv/bin/activate   # recommended: Python 3.12 (3.11 is also fine; see the note below)
# If Python 3.11 is the version you have installed, use it instead:
#   python3.11 -m venv .venv && source .venv/bin/activate
pip install -e ".[deep]"          # core + FaceNet/ArcFace packages
# put LFW at data/lfw/<Person_Name>/<Person_Name>_0001.jpg  (see docs/USER_GUIDE.md)
chmod +x run_faceeval.sh
./run_faceeval.sh check           # read-only environment / data / weights check
./run_faceeval.sh results         # show recorded results (read-only, no experiment)
# Only if you intentionally want to start a small experiment:
# ./run_faceeval.sh demo --classical
```

Full instructions, dependencies, model weights, troubleshooting: **[docs/USER_GUIDE.md](docs/USER_GUIDE.md)**. Presentation material: **[docs/PRESENTATION_GUIDE.md](docs/PRESENTATION_GUIDE.md)**.

**Python version for the `deep` extra: use Python 3.11 or 3.12.** The extra installs `facenet-pytorch`, whose PyPI metadata (checked on 2026-11-18, version 2.6.0) requires `torch>=2.2,<2.3`, `torchvision<0.18` and `numpy<2`. No torch 2.2.x wheel exists for Python 3.13 or 3.14 on macOS arm64, so `pip install -e ".[deep]"` is expected to fail there. The core package declares `requires-python >= 3.11`; `pyproject.toml` is unchanged. A clean-machine installation has **not** been tested. Recorded interpreter evidence is limited: `reports/experiment_results/README.md` records Python 3.12.14 and NumPy 1.26.4 for the Fold-0 results archive; the interpreter used for run `2bad8efd` and for the other runs is NOT RECORDED.

## The runner script

`run_faceeval.sh` is the repository's wrapper around the project's real Python entry point. Its read-only and experiment commands are documented below. **The first four commands are read-only** (they write nothing, download nothing and start nothing). **The last two start an experiment.**

| Command | Purpose | Writes? |
|---|---|---|
| `./run_faceeval.sh help` | list commands | no |
| `./run_faceeval.sh check` | Python, packages, dataset, cached weights, configs | **no** |
| `./run_faceeval.sh results [id]` | accuracy table, fingerprints, deployment ranking; default run `2bad8efd` | **no** |
| `./run_faceeval.sh evidence [id]` | what the run recorded in its database, plus current file hashes | **no** (redirect to save: `... evidence > file.md`) |
| `./run_faceeval.sh demo` / `demo --classical` | **starts** a small 20-subject experiment | only `runs/`, `logs/` |
| `./run_faceeval.sh full [config]` | **starts** a long experiment (thesis config: about an hour); needs you to type `RUN FULL` | only `runs/`, `logs/` |

Every started experiment uses `require_real_models: true` (a missing FaceNet/ArcFace backbone stops it instead of using fake embeddings), runs inside its own `runs/<name>_<time>/` folder (so `experiments/results.db` is never opened for writing) and never overwrites anything. You never need to start an experiment to inspect or present the thesis results.

## Architecture

```
LFW images -> stratified split -> preprocessing -> model -> perturbation -> metrics
                                                              |
                         fingerprint (robustness AUC) -> deployment score -> CSV + SQLite
```

- **Traditional models** train on the training split. **Deep models** are frozen pretrained backbones; each person is enrolled as the L2-normalised mean embedding (centroid) of their training images, and a probe is assigned to the highest cosine similarity.
- **Perturbations** are applied to test images only. Severity 0.0 is the unperturbed image. Thesis mappings: Gaussian blur sigma = 8 x severity; rotation = 45 degrees x severity; brightness shift = (severity - 0.5) x 0.8.
- **Robustness AUC** is the trapezoidal area under accuracy versus severity.
- Everything is driven by one YAML file plus the seed.

```
faceeval/
  core/           types, config (pydantic), registry, exceptions
  data/           dataset loaders (LFW), stratified splitter, cache
  preprocessing/  detection, alignment, normalisation
  models/         traditional/ (Eigenfaces, Fisherfaces, LBPH, PCA+SVM, HOG-SVM, k-NN)
                  deep/ (FaceNet, ArcFace, InsightFace, dlib)
  perturbation/   16 perturbation types
  evaluation/     metrics, calibration, profiler, fairness
  fingerprint/    behavioral fingerprint builder / comparator
  failure/        failure extraction and clustering
  deployment/     scenarios and scorer
  visualization/  figure generators
  orchestration/  ExperimentRunner, tracker, reproducibility
  reporting/      CSV report generation
  storage/        SQLite ResultStore
api/              FastAPI read layer over experiments/results.db (not needed for the thesis numbers)
dashboard/        React + Vite front end (not needed for the thesis numbers)
configs/          YAML experiment definitions
tests/            pytest suite
tools/            run integrity checker
docs/             user guide, presentation guide, experiment records
reports/          committed experiment artifacts (force-added; the folder is git-ignored by default)
```

## Running an experiment from Python

The project has no CLI of its own; this is the real entry point (it is what `run_faceeval.sh` calls):

```python
from faceeval.core.config import load_config
from faceeval.orchestration import ExperimentRunner

config = load_config("configs/lfw_intermediate.yaml")   # set require_real_models: true
run = ExperimentRunner(config).run()   # also saves the run to experiments/results.db in the current folder
```

Protocols are selected by `evaluation_protocol` in the YAML: `identification`, `lfw_fold0_verification` or `both`.

## Reproducibility

- Fixed seed (42) and deterministic stratified split: 2,734 training / 461 validation / 1,357 test images for the 200-subject experiment (min 8 images per subject, test 0.30, validation 0.10).
- Each identification run writes `evaluation_results`, `fingerprints`, `deployment_rankings` and `failure_clusters` CSVs named with its run id, plus a log. A Fold-0 verification run writes `verification_results.csv` and `.json`, split and pair files, and `scores/*.npz`; it has no resume, so an interrupted run must be restarted into a new run directory.
- `docs/EXPERIMENT_EVIDENCE_RECORD.md` records what is known about each experiment and labels every fact as repository evidence, reported from the Mac, NOT RECORDED, or requiring a new run. Nothing is invented. An identification run stores its short git commit and configuration hash in its database record; Python and package versions, weight hashes and a dataset fingerprint are **not** stored by it. Fold-0 verification summaries record an `environment` block (Python, NumPy, code version); runs started after commit `42c2f44` also record a `provenance` block (platform, package versions, weight SHA-256). Whether the FaceNet and ArcFace sweeps carry that block has **not been verified** by the reviewer: check the `verification_results.json` of each run for a `provenance` key, and treat anything absent as NOT RECORDED.
- Not yet shown: a repeat of the thesis run, an installation from a clean machine, confidence intervals.

## Six-model Fold-0 analysis

This is the analysis of the **custom Fold-0 verification benchmark**, not of the 200-subject identification run. It only reads archived `verification_results.csv` / `.json` files. It runs no experiment, loads no model and recomputes no metric, so it needs neither the dataset nor the model weights.

```bash
python -m faceeval.analysis.fold0_results \
    --results-root reports/experiment_results \
    --facenet-run <FACENET_VERIFICATION_RUN_DIR> \
    --arcface-run <ARCFACE_VERIFICATION_RUN_DIR> \
    --out <NEW_OUTPUT_DIRECTORY>
```

| Option | Meaning |
|---|---|
| `--results-root` | Folder holding the committed `classical/`, `eigenfaces_final/` and `eigenfaces_per_image_noise/` archives (default `reports/experiment_results`). |
| `--facenet-run DIR` | Directory containing the FaceNet `verification_results.csv` and `.json`. It is **not** under the results root; it lives in the run tree (`runs/...`) on the Mac. |
| `--arcface-run DIR` | The same for the ArcFace run. |
| `--out DIR` | A **new** output directory (default `<results-root>/analysis`). A non-empty directory is refused unless `--overwrite` is given; with `--overwrite` nothing is deleted and files of an earlier analysis that were not rewritten are listed in `provenance.json` under `stale_files_not_rewritten`. |
| `--overwrite`, `--no-figures` | Allow a non-empty output directory; skip the figures. |

- With **both** deep options: 6 models x 16 perturbations x 6 severities = 576 primary rows. With **neither**: the original four classical models (384 rows). Giving only one deep option is a different configuration from the six-model analysis; use both.
- Every run is validated on its own (complete, full LFW, 96 rows per model, constant frozen threshold, consistent clean baseline), and the runs must agree on split content hash, pair hashes, dataset fingerprint, fold/seed and noise protocol. If a check fails the command prints `ANALYSIS REFUSED: <reason>` and exits with status 2. It never writes into an archived run directory.
- Output (all under `--out`): `tables/table1_...` to `tables/table13_...` (13 CSV files; `table10_all_cells_long` has one row per condition and metric, 576 x 5 = 2,880 rows for six models), `figures/fig1_...` to `fig7_...` (7 PNG files for the six-model analysis; `fig7`, the failure-rate figure, is drawn only when deep-model results are included, so the four-model run gives 6; with `--no-figures` none), and `provenance.json`.
- **The generated analysis output is a local artifact and is not committed to Git.** `reports/` is git-ignored, so an output folder created under it stays local unless it is force-added. The FaceNet and ArcFace result files are likewise not in Git. Anyone reproducing the analysis needs those two run directories (the maintainer holds them, with verified archives, on the Mac) and can state their SHA-256 values from the evidence record.
- The historical per-image-noise run (`c408e0d2`) is read separately and appears only in `table11`; it is never mixed into the primary rows.

**ArcFace and FaceNet failure semantics (read before interpreting any deep-model number).** The archived deep-model metrics are **end-to-end**. An image whose embedding cannot be produced is replaced by a **zero vector**, and every pair containing it is **still scored**; failed pairs are **not removed**. For ArcFace this happens mainly when the internal InsightFace face detector finds no face (the cropping fallback is switched off). AUC, EER, TAR, FAR and accuracy of such a condition therefore mix recognition quality with face-acquisition failure and are **not** valid-pairs-only recognition performance. The verification CSVs carry `n_test_images`, `n_failed_embeddings`, `n_failed_pairs_balanced` and `n_failed_pairs_full`; the analysis reports them in `table12` (per condition) and `table13` (per model) and in `fig7` with explicit denominators. Counts summed across conditions are not unique failed images. Classical models have no failure counters, and their diagnostic fields are left blank rather than invented. A valid-pairs-only analysis from the saved `scores/*.npz` files has **not** been performed.

**FaceNet versus ArcFace is a pipeline comparison, not a backbone-only comparison.** FaceNet (InceptionResNetV1, VGGFace2 weights) uses a centre crop (fraction 0.8) resized to 160 x 160 and no face detector, so it has no detection failures by construction. ArcFace (InsightFace `buffalo_l`, recognition model `w600k_r50`) uses a face detector with 5-point landmark alignment to 112 x 112. Differences between the two therefore combine backbone, alignment, input size and detection. (Established by reading the source at `44c305c`; not re-run.)

Other caveats for the Fold-0 results: custom split, not the official LFW View 2 protocol, so no comparison with published numbers; the threshold is chosen once on clean validation pairs and frozen, so accuracy, TAR and FAR under perturbation use a threshold that was not re-tuned, while AUC and EER are threshold-free; brightness, contrast and gamma are centred (see above); one split, one seed.

## Tests

```bash
pip install -e ".[dev]"
pytest
```

The maintainer reports 983 tests passing in their Mac environment (the commit and date of that run are not recorded here). A separate run in a Linux cloud sandbox at commit `98774f3` reported 1,172 passed; the difference is not reconciled. Record the commit and the summary line together when citing a count.

## Optional: API and dashboard

```bash
cp .env.example .env
uvicorn api.main:app --port 8000          # serves experiments/results.db read-only
cd dashboard && npm install && npm run dev
```

`start.sh` opens these in Terminal tabs on macOS; edit `PROJECT_DIR` inside it first. The API and dashboard were not part of the verification reported in the thesis.

## Licence

MIT (see `pyproject.toml`). InsightFace and VGGFace2-trained weights carry their own licences (research use); LFW has its own terms.
