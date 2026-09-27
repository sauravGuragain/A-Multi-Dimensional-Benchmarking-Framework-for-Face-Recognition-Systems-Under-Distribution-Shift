# FaceEval-X

Multi-dimensional benchmarking framework for face recognition under
distribution shift — traditional ML vs. deep learning, evaluated across
16 perturbation types, with behavioral fingerprinting, failure analysis,
and context-weighted deployment scoring.

Built incrementally as the software foundation for a Master's thesis
(*Multi-Dimensional Evaluation Framework for Benchmarking Traditional ML
and Deep Learning Face Recognition Systems Under Distribution Shift*).

## Build status

| # | Layer | Status |
|---|---|---|
| 1 | Architecture & design | ✅ |
| 2 | Core (`faceeval/core`) — types, config, registry, exceptions | ✅ |
| 3 | Data (`faceeval/data`) — datasets, splitting, caching | ✅ |
| 4 | Preprocessing (`faceeval/preprocessing`) — detect/align/normalize | ✅ |
| 5 | Traditional ML models (6) | ✅ |
| 6 | Deep learning models (4) | ✅ |
| 7 | Perturbation engine (16 types) | ✅ |
| 8 | Evaluation engine — metrics, calibration, profiling, fairness | ✅ |
| 9 | Behavioral fingerprint module | ✅ |
| 10 | Failure analysis module | ✅ |
| 11 | Deployment scoring engine | ✅ |
| 12 | Visualization engine | ✅ |
| 13 | Orchestration & reporting | ✅ |
| 14 | Storage + FastAPI layer | ✅ |
| 15 | React dashboard | ✅ |
| 16 | Docker deployment | ⬜ not yet built |
| 17 | Unit test suite | ⬜ scaffolding only — see `tests/` |
| 18 | Documentation | ⬜ this README only |

Every layer above has been independently verified with executable
smoke tests during development (see the conversation history for the
specific assertions run against each module). A formal `pytest` suite
under `tests/` has not yet been written — `tests/conftest.py` contains
fixture scaffolding only.

## Project structure

```
faceeval-x/
├── faceeval/                 # Core Python package
│   ├── core/                 # types, config, registry, exceptions
│   ├── data/                 # dataset manager, splitter, cache, loaders
│   ├── preprocessing/        # detection, alignment, normalization
│   ├── models/
│   │   ├── traditional/      # Eigenfaces, Fisherfaces, LBPH, PCA+SVM, HOG+SVM, KNN
│   │   └── deep/              # FaceNet, ArcFace, InsightFace, Dlib
│   ├── perturbation/         # blur, noise, photometric, geometric, occlusion, compression
│   ├── evaluation/           # metrics, calibration, profiler, fairness
│   ├── fingerprint/          # ADC curves, sample efficiency, builder, comparator
│   ├── failure/               # extractor, clusterer, correlator, meta-model
│   ├── deployment/            # scenarios, scorer, sensitivity, ranker
│   ├── visualization/         # matplotlib figure generators (16 figure types)
│   ├── orchestration/         # runner, tracker, reproducibility
│   ├── reporting/             # CSV/JSON/PDF report generator
│   └── storage/               # SQLite ResultStore
├── api/                      # FastAPI application (read layer over ResultStore)
│   └── routers/               # runs, results, fingerprints, deployment, figures, reports, registry
├── dashboard/                 # React + Vite frontend
│   └── src/
│       ├── api/                # fetch client
│       ├── components/         # AppShell, ScoreBar, RadarChart, AdcChart, primitives
│       └── pages/               # Overview, Models, Fingerprint, Deployment, Figures
├── configs/                   # YAML experiment configurations
├── tests/                     # pytest scaffolding (fixtures only — see status table)
├── pyproject.toml
└── .env.example
```

## Setup

### Python package

```bash
pip install -e .
# Optional heavy dependencies (only needed for real DL inference,
# not for running the framework's own tests):
pip install facenet-pytorch insightface onnxruntime dlib
```

### API server

```bash
cp .env.example .env        # adjust paths if needed
uvicorn api.main:app --reload --port 8000
```

The API is a **read-only** layer: it never trains models or runs
inference on the request path. It serves whatever has been written to
`experiments/results.db` by `faceeval.orchestration.ExperimentRunner`.

### Dashboard

```bash
cd dashboard
npm install
npm run dev          # http://localhost:5173, proxies /api to :8000
```

### Running an experiment

```python
from faceeval.core.config import load_config
from faceeval.orchestration import ExperimentRunner
from faceeval.storage import ResultStore

config = load_config("configs/quick_test.yaml")
run = ExperimentRunner(config).run()

ResultStore("experiments/results.db").save(run)
```

Then open the dashboard — the run will appear in the left-rail picker.

## Notes on scope

This codebase was built module-by-module across many sessions, with each
module independently exercised against synthetic data immediately after
writing it. That verification confirms *internal correctness* (types
match, invariants hold, computations are numerically sound) but does
**not** constitute an end-to-end run against a real dataset like LFW —
that integration test, plus the Docker packaging (`docker/`) and the
formal `pytest` suite (Chapter "Unit tests" in the original build plan),
remain open work.
# FaceX
