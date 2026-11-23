# FaceEval-X

A modular, configuration-driven benchmarking framework that evaluates face recognition systems under controlled distribution shift. It compares traditional machine-learning and pretrained deep models across perturbations (blur, rotation, brightness and 13 more), and produces accuracy tables, robustness fingerprints, failure clusters and a context-weighted deployment score.

It is the software foundation of the thesis project, *Multi-dimensional benchmarking framework for face recognition under distribution shift*.

## Where to look

| Question | Document |
|---|---|
| What exactly was benchmarked (dataset, split, pairs, threshold, perturbations, noise, failures)? | [docs/PROTOCOL.md](docs/PROTOCOL.md) |
| How do I verify or regenerate the results? | [docs/REPRODUCING.md](docs/REPRODUCING.md) |
| What is in each result table and figure, and how do I read it? | [docs/RESULTS_MAP.md](docs/RESULTS_MAP.md) |
| Which files are in the committed archive? | [benchmarks/lfw_fold0_2026-11-20/README.md](benchmarks/lfw_fold0_2026-11-20/README.md) |
| What is known about each run, and how do we know? | [docs/EXPERIMENT_EVIDENCE_RECORD.md](docs/EXPERIMENT_EVIDENCE_RECORD.md) |
| Step-by-step instructions, dependencies, troubleshooting | [docs/USER_GUIDE.md](docs/USER_GUIDE.md) |
| Presenting the thesis results | [docs/PRESENTATION_GUIDE.md](docs/PRESENTATION_GUIDE.md) |

## What is validated, and what is only implemented

**Read this before citing anything from the project.**

| Area | Status |
|---|---|
| **Primary six-model benchmark: custom Fold-0 verification** (identity-disjoint pairs; TAR, FAR, AUC, EER). Eigenfaces, Fisherfaces, LBPH, HOG-SVM, FaceNet, ArcFace over 16 perturbations x 6 severities = 96 conditions per model, **576 rows** | **Completed and committed** in `benchmarks/lfw_fold0_2026-11-20/`: classical run `5ecfdd99...` (288 rows), Eigenfaces run `a55a4a37...` (96 rows) and the deep run **`5ee3d007eb43445aa60c7fcd062ed50a`** (FaceNet + ArcFace, 192 rows, code `f6d5e12`); 216 files in a `SHA256SUMS` manifest. This is the project's own identity-disjoint split (seed 42, fold 0 of 10), **not the official LFW View 2 protocol**; do not compare it with published View 2 numbers, and never combine it with the identification run below. One split, one seed. |
| Six-model analysis and valid-pairs analyses of that archive | **Committed** in `benchmarks/lfw_fold0_2026-11-20/analysis/` (29 files: 13 tables, 7 figures, provenance, FaceNet and ArcFace valid-pairs outputs, `SHA256SUMS`, `RUN_RECORD.txt`). Generated post hoc in a Linux cloud sandbox from the archived data; the maintainer reports that all 28 listed checksums verified on the Mac. Map: [docs/RESULTS_MAP.md](docs/RESULTS_MAP.md). |
| **FaceNet and ArcFace values are end-to-end** | A failed embedding is a zero vector and its pairs are still scored. **FaceNet: 0 failures** (centre crop, no detector). **ArcFace: failures in 31 of its 80 perturbed conditions** (12,969 failed embeddings summed over conditions, not unique images); five conditions leave fewer than 10 valid pairs in a class, and in two the detector misses every image. Valid-pairs-only values are conditional and selection-biased and never replace the end-to-end numbers. Details: [docs/RESULTS_MAP.md](docs/RESULTS_MAP.md). |
| Primary thesis experiment (identification): LFW, 200 subjects, closed-set identification, Eigenfaces / HOG-SVM / FaceNet / ArcFace, Gaussian blur / rotation / brightness at severities 0.0, 0.5, 1.0 (36 conditions, 1,357 test images) | **Completed.** Run `2bad8efd2ac0468cb1bf54fc517e7119` (result files are on the maintainer's machine, **not in Git**). A different protocol from verification; see `docs/EXPERIMENT_EVIDENCE_RECORD.md`. |
| Historical and superseded runs | `c408e0d2...` Eigenfaces per-image-noise run: separate and historical (different manifest and code); kept apart in `reports/experiment_results/eigenfaces_per_image_noise/`. `b567f880...` (FaceNet) and `4bdf6982...` (ArcFace): earlier deep sweeps reported from the Mac; not in Git; their relation to `5ee3d007...` is not recorded. Identification runs `fe4696ee...` and `69878560...`: **superseded** (defective FaceNet preprocessing / non-real ArcFace). The two-row clean-only deep checks (`b8fbb5c3`, `c069be84`) were a preliminary step, not the benchmark. |
| PCA+SVM, k-NN, InsightFace, dlib models | Implemented and run through the same verification runner on fake data only (`tests/test_ten_models.py`: placeholder backbones labelled `STUB`). **Never run on real LFW; real InsightFace weights never loaded.** `dlib_fr` uses dlib's own face chip and pinned, checksum-verified weights (checked on sample images in the cloud sandbox only). Configs: `configs/lfw_fold0_pca_svm.yaml`, `lfw_fold0_knn.yaml`, `lfw_fold0_deep_insightface.yaml`, `lfw_fold0_deep_dlib_fr.yaml`. |
| Calibration, fairness, speed and memory, other splits or seeds | **Implemented; never run on real data.** Calibration (`compute_calibration: true`) and multi-split (`faceeval.analysis.multi_split`) are tested on fake or synthetic data only. Fairness (`faceeval.analysis.fairness`) needs an external attribute CSV that LFW does not provide. The efficiency harness (`faceeval.analysis.efficiency`) has no real measurement. Sample efficiency: not implemented. |
| Deployment ranking | **Accuracy-only in practice.** Robustness, calibration, fairness, latency, memory and cost are placeholders. A demonstration of the scoring mechanism, not a validated multi-dimensional ranking. |
| Behavioral fingerprint | 3 of 21 dimensions populated (robustness AUC for blur, rotation, brightness). |
| Full benchmark (`configs/lfw_full_benchmark.yaml`: 10 models, 16 perturbations, 6 severities, all subjects) | **Never run.** Out of the thesis scope. |
| JSON report | The runner writes `reports/.../report.json` atomically with `schema_version` 2 (`tests/test_json_export.py`). Figure export (`export_png`) and Docker: not exercised / not built. The archived thesis run exported 0 figures. |

### Evidence levels used in this repository

| Level | Meaning |
|---|---|
| **Implemented** | The code exists in this repository. |
| **Executed** | An experiment actually ran to completion (only the runs listed above). |
| **Committed archive** | The result files are in Git and covered by a checksum manifest (`benchmarks/lfw_fold0_2026-11-20/`). |
| **Verified on the maintainer's Mac (reported)** | The maintainer reports it from their own machine: for example the identification run `2bad8efd` and its `results` view, the Fold-0 runs, the checksum verification of the 28 analysis files, and an earlier full-suite run (983 tests; commit not recorded; it predates the current code). Not independently verified. |
| **Verified in a Linux cloud sandbox (not the Mac)** | Focused tests, the analysis runs on the archive, and the runner's read-only behaviour on fixture data (`docs/EXPERIMENT_EVIDENCE_RECORD.md` section 5). These do not establish success on macOS. |
| **Reported as run on the maintainer's macOS (Bash 3.2)** | The maintainer reports that `bash -n run_faceeval.sh`, `help`, `check` (0 warnings), `results 2bad8efd` and `evidence 2bad8efd` succeeded. The SHA-256 of the file that was tested is NOT RECORDED (the SHA-256 of the committed `run_faceeval.sh` is in the evidence record, section 5). Experiment-starting commands and a clean-machine installation have not been verified. |
| **Planned / proposed only** | A repeat of the thesis runs; the real-weights 10-model benchmark; calibration, multi-split and efficiency on real models; a clean-machine deep install (torch, facenet-pytorch, insightface, dlib); a fairness study (needs licensed attribute labels). **None has been executed.** |

**Reduced scope of the identification validation (run `2bad8efd`):** 200 of 5,749 subjects (3.5%), 4 of 10 models, 3 of 16 perturbations, 3 of 6 severity levels. One run, one split, seed 42, no confidence intervals. Brightness there is `(severity - 0.5) x 0.8`, so 0.5 is the identity and 0.0 is returned unperturbed: only one brightening level (+0.40) is tested and darkening is not.

**Scope of the Fold-0 verification benchmark:** 6 of 10 models, all 16 perturbations, all 6 severity levels, one custom identity-disjoint split (seed 42; test set 764 genuine plus 764 balanced impostor pairs). One split and one seed: the intervals the analysis prints are **indicative only** (they assume independent pairs) and **no significance test is claimed**. The pretrained FaceNet and ArcFace weights were trained outside this project; whether LFW identities occur in their training data has not been checked. FaceNet versus ArcFace is a **pipeline** comparison (crop or alignment, input size and detection differ), not a backbone-only comparison.

**Centred perturbations:** brightness, contrast and gamma are two-sided with their neutral point near severity 0.5, and severity 0.0 returns the clean image for every type. Their curves are V-shaped, not monotone: do not read the mean over severities 0.2-1.0 as "average degradation" for these three. See `docs/PROTOCOL.md`, section 7.

## Quick start (macOS / Linux)

```bash
git clone <this repository> && cd <repository folder>
python3.12 -m venv .venv && source .venv/bin/activate   # recommended: Python 3.12 (3.11 is also fine; see the note below)
# If Python 3.11 is the version you have installed, use it instead:
#   python3.11 -m venv .venv && source .venv/bin/activate
pip install -e ".[deep]"          # core + torch + FaceNet/ArcFace/InsightFace packages
# pip install -e "."               # lightweight core only (classical models, analysis tools, reports; no torch)
# pip install -e ".[dlib]"         # optional dlib_fr model (needs CMake and a C++ compiler)
# put LFW at data/lfw/<Person_Name>/<Person_Name>_0001.jpg  (see docs/USER_GUIDE.md)
chmod +x run_faceeval.sh
./run_faceeval.sh check           # read-only environment / data / weights check
./run_faceeval.sh results         # show recorded results (read-only, no experiment)
# Only if you intentionally want to start a small experiment:
# ./run_faceeval.sh demo --classical
```

Full instructions, dependencies, model weights, troubleshooting: **[docs/USER_GUIDE.md](docs/USER_GUIDE.md)**.

**Python version for the `deep` extra: use Python 3.11 or 3.12.** The extra installs `facenet-pytorch`, whose PyPI metadata (checked on 2026-11-18, version 2.6.0) requires `torch>=2.2,<2.3`, `torchvision<0.18` and `numpy<2`. No torch 2.2.x wheel exists for Python 3.13 or 3.14 on macOS arm64, so `pip install -e ".[deep]"` is expected to fail there. The core package declares `requires-python >= 3.11`; `pyproject.toml` is unchanged. A clean-machine installation has **not** been tested. The recorded environment of the archived Fold-0 runs (Python 3.12.14, NumPy 1.26.4; for the deep run also torch 2.2.2 and the package versions listed there) is in [docs/REPRODUCING.md](docs/REPRODUCING.md); the interpreter used for run `2bad8efd` and for the other runs is NOT RECORDED.


## Verify and regenerate the Fold-0 results

Reading the committed results needs neither the dataset nor any model weights and runs no experiment.

```bash
# verify the archive and the analysis outputs (expected: 0 lines that are not OK; 29 files in analysis/)
(cd benchmarks/lfw_fold0_2026-11-20 && shasum -a 256 -c SHA256SUMS | grep -vc ': OK')
(cd benchmarks/lfw_fold0_2026-11-20/analysis && shasum -a 256 -c SHA256SUMS | grep -vc ': OK')

# regenerate into a NEW directory (never into the archive)
python -m faceeval.analysis.fold0_results --results-root reports/experiment_results \
    --deep-run benchmarks/lfw_fold0_2026-11-20/deep_core --out <NEW_OUTPUT_DIRECTORY>
python -m faceeval.analysis.valid_pairs --run benchmarks/lfw_fold0_2026-11-20/deep_core --model arcface --out <NEW_DIRECTORY>
```

- `--deep-run DIR` takes one combined FaceNet + ArcFace run (192 rows). `--facenet-run DIR --arcface-run DIR` is the alternative for separate per-model runs; the two forms cannot be mixed. `valid_pairs --model NAME` selects one model of a multi-model run and is required for one.
- A failed validation prints `ANALYSIS REFUSED: <reason>` and exits with status 2. Nothing is written into a run directory, and a non-empty output directory is refused without `--overwrite`.
- `valid_pairs` refuses a run whose archived end-to-end values it cannot reproduce from the saved score files.
- The efficiency harness is separate: `python -m faceeval.analysis.efficiency --models facenet arcface --images-dir data/lfw --n-images 200 --out <NEW_DIRECTORY>` (needs real weights; no real measurement exists yet).

The full procedure and what to expect: [docs/REPRODUCING.md](docs/REPRODUCING.md).

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
  evaluation/     metrics, pairs, calibration, profiler, fairness
  analysis/       read-only analysis tools (fold0_results, valid_pairs, multi_split, efficiency, figures, ...)
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
benchmarks/       committed Fold-0 benchmark archive and its analysis outputs
tests/            pytest suite
tools/            run integrity checker
docs/             protocol, reproducing, results map, user guide, presentation guide, experiment records
reports/          committed classical Fold-0 archives (force-added; the folder is git-ignored by default)
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

- Fixed seed (42). Identification experiment: deterministic stratified split of 2,734 training / 461 validation / 1,357 test images for 200 subjects (min 8 images per subject, test 0.30, validation 0.10). Fold-0 verification: the identity-disjoint split of `docs/PROTOCOL.md` (4,601 / 574 / 574 identities).
- Each identification run writes `evaluation_results`, `fingerprints`, `deployment_rankings` and `failure_clusters` CSVs named with its run id, plus a log. A Fold-0 verification run writes `verification_results.csv` and `.json`, split and pair files, and `scores/*.npz`; it has no resume, so an interrupted run must be restarted into a new run directory.
- What each run recorded: an identification run stores its short git commit and configuration hash in its database; Python and package versions, weight hashes and a dataset fingerprint are **not** stored by it. Fold-0 summaries record an `environment` block (Python, NumPy, code version). The archived **deep** run also records a `provenance` block (platform, package versions, weight SHA-256); the archived **classical and Eigenfaces** runs do not, so their package versions and weight hashes are NOT RECORDED.
- `docs/EXPERIMENT_EVIDENCE_RECORD.md` labels every fact as repository evidence, reported from the Mac, NOT RECORDED, or requiring a new run. Nothing is invented.
- Not yet shown: a repeat of any run (so run-to-run repeatability is unknown), an installation from a clean machine, multi-split confidence intervals.

## Tests

```bash
pip install -e ".[dev]"
pytest
```

No total test count is claimed for the current commit. The 40 tests added with the combined-run analysis (`tests/test_fold0_analysis.py`, `tests/test_valid_pairs.py`) gave 258 passed, 0 skipped in a Linux cloud sandbox, on a tree whose changed files are identical to those in commit `10e901b`; that was not a Mac run and not GitHub CI. The older counts (983 on the Mac, 1,172 in a sandbox at `98774f3`) belong to earlier commits and are not reconciled. Record the commit and the summary line together when citing a count. CI (`.github/workflows/ci.yml`) runs the lightweight suite on Python 3.11 and 3.12, not real models or real LFW.

## Optional: API and dashboard

A read-only dashboard over `experiments/results.db` (the identification-protocol runs written by `ExperimentRunner`). It does not change the database, run models or regenerate any result, and it does not show the canonical Fold-0 six-model analysis, which lives in `benchmarks/lfw_fold0_2026-11-20/` (see `docs/RESULTS_MAP.md`).

macOS, from the repository root, in two Terminal windows:

```bash
# Window 1 - API (use .venv-facenet or .venv, whichever you installed)
source .venv-facenet/bin/activate
python -m pip install -e .                # once, if fastapi/uvicorn are missing
uvicorn api.main:app --host 127.0.0.1 --port 8000

# Window 2 - dashboard (Node 18+)
cd dashboard
npm ci                                    # once (installs exactly what package-lock.json pins)
npm run dev -- --host 127.0.0.1           # then open http://127.0.0.1:5173
```

`./start.sh` does the same in two Terminal windows. `./start.sh --dry-run` runs the checks and prints the commands without starting or writing anything. It uses `FACEEVAL_VENV` if set (and stops if that environment lacks fastapi/uvicorn); otherwise the first of `.venv-facenet` / `.venv` that has fastapi and uvicorn. It stops if port 8000 or 5173 is busy and rejects unknown arguments.

- Settings are environment variables (`.env.example` lists them); nothing reads `.env` automatically. Defaults: `FACEEVAL_DB_PATH=experiments/results.db`, `FACEEVAL_FIGURES_DIR=reports/figures`, `FACEEVAL_REPORTS_DIR=reports`.
- If the database path is wrong the dashboard says so; the API never creates a database.
- `http://127.0.0.1:8000/api/status` shows which database, figures and reports folders the API is using.
- Figures (`.png`/`.svg`) and reports (`.csv`/`.json`/`.pdf`, file name containing the run id) are served read-only. Checks: `pytest tests/test_api_dashboard.py tests/test_dashboard_clean_semantics.py tests/test_start_script.py` (needs `httpx`, included in `.[dev]`; the clean-semantics test also uses `node` if installed) and `cd dashboard && npm test && npm run lint && npm run build`.
- Model Comparison shows a *clean accuracy* (unperturbed rows, severity 0.0 for every perturbation type, and severity 0.5 for brightness and gamma only; equivalent rows count once, and disagreeing rows show a range) next to the *mean over all stored conditions*; they are different quantities. Contrast has no exact neutral severity.
- Deployment scores are weighted composites in which a missing component is set to a neutral 0.5 by the scorer, and fingerprint axes for perturbations that were not evaluated are stored as 0.0; the pages say so. They are not the Fold-0 evidence.

## Licence

MIT (see `pyproject.toml`). InsightFace and VGGFace2-trained weights carry their own licences (research use); LFW has its own terms.
