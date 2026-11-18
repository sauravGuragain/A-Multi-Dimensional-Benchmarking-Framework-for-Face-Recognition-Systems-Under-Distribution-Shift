# FaceEval-X

A modular, configuration-driven benchmarking framework that evaluates face recognition systems under controlled distribution shift. It compares traditional machine-learning and pretrained deep models across perturbations (blur, rotation, brightness and 13 more), and produces accuracy tables, robustness fingerprints, failure clusters and a context-weighted deployment score.

It is the software foundation of the thesis project, *Multi-dimensional benchmarking framework for face recognition under distribution shift*.

## What is validated, and what is only implemented

**Read this before citing anything from the project.**

| Area | Status |
|---|---|
| Primary thesis experiment: LFW, 200 subjects, closed-set identification, Eigenfaces / HOG-SVM / FaceNet / ArcFace, Gaussian blur / rotation / brightness at severities 0.0, 0.5, 1.0 (36 conditions, 1,357 test images) | **Completed.** Run `2bad8efd2ac0468cb1bf54fc517e7119` (result files are on the maintainer's machine, not in Git). See `docs/EXPERIMENT_EVIDENCE_RECORD.md`. |
| Fold-0 verification (identity-disjoint pairs, TAR/FAR) for Eigenfaces, Fisherfaces, LBPH, HOG-SVM over 16 perturbations x 6 severities | **Completed** (`reports/experiment_results/`). A different protocol; never combine with identification numbers. |
| FaceNet / ArcFace on Fold-0 | Clean condition only |
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
| **Verified on the maintainer's Mac (reported)** | The maintainer reports it from their own machine: the 983-test full suite (commit not recorded), the 2bad8efd run, and the `results` view of that run. |
| **Verified in Linux cloud sandbox and macOS** | The runner's exhaustive read-only non-mutation guarantees and focused tests were verified in a Linux cloud sandbox. The runner's syntax and four read-only commands were also successfully checked on macOS. |
| **Verified on macOS (read-only commands)** | `bash -n run_faceeval.sh`, `help`, `check` (0 warnings), `results 2bad8efd`, and `evidence 2bad8efd` all succeeded on the maintainer's macOS Bash 3.2. Experiment-starting commands and clean-machine installation have not been independently verified. |
| **Planned / proposed only** | A repeat of the thesis run, FaceNet/ArcFace over the 16 x 6 grid, the 10-model benchmark, calibration/fairness/speed/memory studies, a clean-machine install. **None has been executed.** |

**Reduced scope of the thesis validation:** 200 of 5,749 subjects (3.5%), 4 of 10 models, 3 of 16 perturbations, 3 of 6 severity levels. One run, one split, seed 42, no confidence intervals.

**Brightness note:** the brightness shift is `(severity - 0.5) x 0.8`, so severity 0.5 is the identity and 0.0 is returned unperturbed. Only one brightening level (+0.40) is effectively tested; darkening is not.

## Quick start (macOS / Linux)

```bash
git clone <this repository> && cd <repository folder>
python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[deep]"          # core + FaceNet/ArcFace packages
# put LFW at data/lfw/<Person_Name>/<Person_Name>_0001.jpg  (see docs/USER_GUIDE.md)
chmod +x run_faceeval.sh
./run_faceeval.sh check           # read-only environment / data / weights check
./run_faceeval.sh results         # show recorded results (read-only, no experiment)
# Only if you intentionally want to start a small experiment:
# ./run_faceeval.sh demo --classical
```

Full instructions, dependencies, model weights, troubleshooting: **[docs/USER_GUIDE.md](docs/USER_GUIDE.md)**. Presentation material: **[docs/PRESENTATION_GUIDE.md](docs/PRESENTATION_GUIDE.md)**.

A clean-machine installation has not yet been independently verified. The thesis runs used Python 3.12 in an existing virtual environment.

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
- Each run writes `evaluation_results`, `fingerprints`, `deployment_rankings` and `failure_clusters` CSVs named with its run id, plus a log.
- `docs/EXPERIMENT_EVIDENCE_RECORD.md` records what is known about each experiment and labels every fact as repository evidence, reported from the Mac, NOT RECORDED, or requiring a new run. Nothing is invented. A run stores its short git commit and configuration hash in its database record; Python and package versions, weight hashes and a dataset fingerprint are **not** stored by a run.
- Not yet shown: a repeat of the thesis run, an installation from a clean machine, confidence intervals.

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
