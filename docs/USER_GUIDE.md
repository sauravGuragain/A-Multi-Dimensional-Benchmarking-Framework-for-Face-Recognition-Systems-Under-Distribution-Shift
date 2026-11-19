# FaceEval-X User Guide (for a beginner on macOS)

You do **not** need to run any experiment to look at the thesis results. Sections 1–5 only look at existing files. Starting an experiment is a separate, clearly marked step (section 8).

Honesty note: this guide is written from the repository and from checks in a Linux cloud sandbox. A command counts as "verified on your Mac" only when you have seen its real output yourself and, if you want it on record, pasted it into `docs/EXPERIMENT_EVIDENCE_RECORD.md`.

## 1. Open Terminal and go to the project folder

1. Press `Cmd + Space`, type `Terminal`, press Enter. A window with a text prompt opens.
2. Go to the project folder. The folder with the verified results on your Mac is:

```bash
cd ~/Thesis/faceeval_x_patch_verify
```

`cd` means "change directory". `~` means your home folder. If the folder name has spaces, put it in quotes, for example `cd "~/My Folder/faceeval"` (the script itself handles spaces).

3. Check you are in the right place:

```bash
ls
```

You should see `README.md`, `faceeval`, `configs`, `run_faceeval.sh` and others. `ls` only lists files.

## 2. The Python environment

The project needs its Python packages, which live in a "virtual environment" (a private folder of packages). `run_faceeval.sh` finds and uses it automatically (it looks for `.venv-facenet` or `.venv` in the project and in `~/Thesis/faceeval_x`). You do not need to activate it for the script.

If you want to use Python yourself (for example `pytest`), activate it once per Terminal window:

```bash
source ~/Thesis/faceeval_x/.venv-facenet/bin/activate
```

The prompt then shows `(.venv-facenet)`. To leave it, type `deactivate`.

To point the script to a different environment:

```bash
export FACEEVAL_VENV=/path/to/your/venv
```

## 3. Make the script runnable (once)

The runner is `run_faceeval.sh` in the project root.


```bash
chmod +x run_faceeval.sh
./run_faceeval.sh help
```

`chmod +x` allows the file to be run. `help` prints the command list. The list separates **read-only** commands from commands that **start an experiment**.

## 4. Look at the existing results (read-only)

```bash
./run_faceeval.sh check
```

Looks at Python, packages, the dataset and the cached model weights. It writes nothing, downloads nothing, and starts nothing. Green `[ OK ]` is good; `[WARN]` is a note; `[FAIL]` needs fixing before you would start an experiment (it does not affect reading results).

```bash
./run_faceeval.sh results
```

Shows the thesis run `2bad8efd` (or the newest run if it is not found): an accuracy table for the 4 models, the robustness fingerprints and the deployment ranking, and lists the runs recorded in any results database (read from a temporary copy; the originals are untouched). It never starts an experiment.

```bash
./run_faceeval.sh evidence 2bad8efd
```

Prints what the run recorded about itself in its database (timestamps, duration, configuration hash, short git commit, seed, protocol) and today's file hashes. Save it with `./run_faceeval.sh evidence 2bad8efd > evidence_2bad8efd.md` (the `>` writes the screen output to that file; this is the only write, and you chose it).

## 5. How to read the output

**Which protocol is this?** `results` shows **closed-set identification**: each test face is assigned to one of 200 known people; "accuracy" is the share assigned correctly. This is the thesis experiment.

**Verification (custom Fold 0) is different.** Verification asks "are these two photos the same person?" using pairs of images and a decision threshold, on people never seen in training. The split is this project's own identity-disjoint "Fold 0"; it is **not the official LFW View 2 protocol**, so it must not be compared with published View 2 numbers. Its numbers (TAR, FAR, balanced accuracy) for the four classical models live in `reports/experiment_results/` and are described in its README. The FaceNet and ArcFace Fold-0 results (96 conditions each) are on the maintainer's Mac and are not in Git (section 5b). Never compare an identification accuracy with a verification accuracy, and never put them in one table.

| Term | Meaning |
|---|---|
| Severity | 0.0 = original image, 1.0 = strongest change |
| Blur | Gaussian blur with sigma = 8 x severity |
| Rotation | angle = 45 degrees x severity |
| Brightness | shift = (severity - 0.5) x 0.8, so **0.5 is the original image**; only +0.40 is tested |
| Robustness AUC (fingerprint) | 0.25 x acc(0) + 0.5 x acc(0.5) + 0.25 x acc(1); closer to 1 = more stable |
| Deployment score | In the thesis run it depends on accuracy only; the other dimensions are placeholders (0.5). It is **not** a validated multi-dimensional ranking. |

Recorded values of run `2bad8efd` (accuracy %, reported from your Mac; `results` should show the same):

| Model | Clean | Blur 1.0 | Rotation 1.0 | Brightness 1.0 |
|---|---|---|---|---|
| Eigenfaces | 21.6 | 12.5 | 5.0 | 13.8 |
| HOG-SVM | 22.3 | 11.9 | 11.7 | 21.1 |
| FaceNet | 99.8 | 1.0 | 77.3 | 98.0 |
| ArcFace | 99.6 | 0.2 | 99.6 | 99.3 |

A difference of a few images is not meaningful: with the recorded clean accuracies (FaceNet 99.78% = 1,354 of 1,357 images, ArcFace 99.63% = 1,352 of 1,357) the two models differ by two images.

## 5b. The six-model Fold-0 analysis (read-only on archived files)

The Fold-0 verification benchmark covers six models (Eigenfaces, Fisherfaces, LBPH, HOG-SVM, FaceNet, ArcFace) over 16 perturbations x 6 severities. The analysis tool reads the archived result files, runs no experiment, loads no model and does not need the dataset. It writes only into the **new** folder you name:

```bash
python -m faceeval.analysis.fold0_results \
    --results-root reports/experiment_results \
    --facenet-run <FACENET_VERIFICATION_RUN_DIR> \
    --arcface-run <ARCFACE_VERIFICATION_RUN_DIR> \
    --out <NEW_OUTPUT_DIRECTORY>
```

Use the Python from your environment (section 2). `--facenet-run` and `--arcface-run` each point to a run directory that contains `verification_results.csv` and `verification_results.json` (under `runs/` on your Mac, not under the results root). `--out` must be a new or empty folder (`--overwrite` allows a non-empty one). If a check fails the tool prints `ANALYSIS REFUSED: ...` and writes nothing into the archives.

Result: `tables/` (13 CSV files), `figures/` (7 PNG files with the deep runs; 6 without them, because the failure-rate figure needs deep-model results) and `provenance.json`. **These outputs are local artifacts and are not committed to Git**; `reports/` is git-ignored.

How to read the deep-model rows: they are **end-to-end** results. When no embedding can be produced for an image (mainly ArcFace's face detector finding no face), it becomes a zero vector and its pairs are **still scored**, not removed. So ArcFace AUC and EER include detection failures; see `table12`, `table13` and figure 7 for the failure counts and rates. FaceNet and ArcFace also differ in crop or alignment, input size and detection, so the comparison is between pipelines, not between backbones. Brightness, contrast and gamma are centred at severity 0.5, so their curves are V-shaped. Never mix these Fold-0 results with the identification results of section 5.

## 5c. Additional analysis tools (read-only on run directories; every one writes to a NEW directory)

All commands below only READ archived run directories. Use the Python from your environment (section 2). **None of them has been run on the real FaceNet/ArcFace archives yet** (see `docs/EXPERIMENT_EVIDENCE_RECORD.md`, Addendum 4).

```bash
# valid-pairs-only re-analysis of ONE deep run (refuses if the archived numbers cannot be reproduced from scores/*.npz)
python -m faceeval.analysis.valid_pairs --run <DEEP_RUN_DIR> --out <NEW_DIR>

# runtime and peak memory, one subprocess per model (needs the real weights and a folder of face images).
# Reports separately: initialisation, cold first call, preprocessing, per-image latency (batch 1), batched throughput, peak RSS.
# --device: auto (what the experiments use: CUDA if present else CPU) | cpu | cuda | mps (Apple Silicon) | best. An unavailable
# explicit device is REFUSED, never replaced. Only FaceNet (torch) can move; ArcFace/InsightFace (ONNX Runtime) and dlib are cpu/auto.
python -m faceeval.analysis.efficiency --models facenet arcface --images-dir <IMAGES_DIR> --n-images 200 --warmup 5 --batch-sizes 8 32 --device auto --out <NEW_DIR>
# run it again with --device mps for FaceNet on Apple Silicon; the JSON's "comparability" block lists what differs between rows

# subgroup metrics, ONLY with an external CSV (image_id,group); nothing is inferred; LFW here has no such labels
python -m faceeval.analysis.fairness --run <DEEP_RUN_DIR> --attributes <ATTRIBUTES_CSV> --source "<where the labels come from and their licence>" --out <NEW_DIR>

# mean / SD over several identity-disjoint split runs (each run made with its own random_seed / verification_fold).
# Every row says how far its interval may be trusted (interval_status) and whether the model is pretrained or trained per split.
python -m faceeval.analysis.multi_split --runs <RUN_DIR_1> <RUN_DIR_2> ... --out <NEW_DIR>

# figures, drawn only from the files above (calibration_results.csv, multi_split_by_condition.csv, valid_pairs_by_condition.csv, efficiency_results.json)
python -m faceeval.analysis.figures calibration --input <RUN_ID.calibration dir> --out <NEW_DIR>
python -m faceeval.analysis.figures multi_split --input <MULTI_SPLIT_DIR> --metric auc --perturbation clean --severity 0 --out <NEW_DIR>
python -m faceeval.analysis.figures valid_pairs --input <VALID_PAIRS_DIR> --out <NEW_DIR>
python -m faceeval.analysis.figures efficiency  --input <EFFICIENCY_DIR> --out <NEW_DIR>

# which package versions a COMPLETED run recorded (to reproduce the environment; prints pins, refuses to guess if none were recorded)
python -m faceeval.analysis.env_pins --run <DEEP_RUN_DIR> --print          # or --check for installed-vs-recorded

# index of which evidence products exist (claims nothing about missing ones)
python -m faceeval.analysis.evidence_index --fold0-analysis <DIR> --valid-pairs <DIR> --out <NEW_DIR>
```

Calibration is a runner option, not a separate tool: set `compute_calibration: true` in a Fold-0 verification config. It fits Platt scaling on the clean validation pairs only and writes `<run_id>.calibration/` next to the run directory (the run directory itself is unchanged, so the integrity checker still passes).

Other splits: add `random_seed: <N>` and/or `verification_fold: <0-9>` to a copy of a Fold-0 config and use a different `output_dir`. Seed 42 with fold 0 is the reference split and behaves exactly as before; any other combination is an additional identity-disjoint split (leakage and pair-count invariants are checked) and must never be mixed with the reference results in one table without saying so.

### Reading the multi-split intervals (what is and is not claimed)

- **Folds of one seed**: the test sets hold different people (disjoint), but every fold's training set contains the other folds' test people. For a pretrained model (FaceNet, ArcFace, InsightFace, dlib) nothing is trained on a split, so the spread across folds is a fair picture of test-set sampling. For a model trained per split (eigenfaces, fisherfaces, pca_svm, lbph, hog_svm, knn) the plain t-interval is **too narrow**; use the `*_corrected_ci_*` columns (Nadeau-Bengio corrected resampled t, variance x (1/n + n_test/n_train)), an approximation that assumes the correlation between splits is about n_test/(n_test+n_train).
- **Different seeds**: independent shuffles of the same people. Test sets overlap, so results are positively correlated; the table is labelled `descriptive_only` and no interval in it is a valid confidence interval.
- Fewer than 5 splits is labelled `too_few_splits`. No interval is a population confidence interval: one dataset, one collection process.

### Configuration identity versus documentation-only changes

`config_hash` (and so `run_id` and the result store's duplicate-run detection) is computed from the parsed configuration values. The YAML `description:` **is** a value, so editing it creates a new identity. `#` comments are **not** values: edit them freely. Fields added later with a default value (`verification_fold: 0`, `dlib_alignment: dlib_chip`, `classical_rng: global_seed`) are left out of the hashed form while they hold the default, so every older config keeps its hash (`tests/test_config_hash_stability.py` pins the eight pre-existing hashes). The `description:` of `configs/lfw_fold0_classical.yaml` still says calibration is not read by the runner; that sentence is historical and stays, because changing it would change the hash. The corrected statement is in the `#` comments of that file and in `docs/EXPERIMENT_EVIDENCE_RECORD.md` (Addendum 5).

### dlib_fr alignment and weights, InsightFace version, PCA/SVM random state

- `dlib_fr` now defaults (config `dlib_alignment: dlib_chip`) to dlib's own pipeline: HOG detector, 5-point landmarks, `get_face_chip(150, padding 0.25)`. `dlib_alignment: legacy` reproduces the old generic aligner and changes the config hash. The method is recorded in the run's preprocessing provenance. No comparability with other models is claimed until a real run exists.
- dlib weights come from `github.com/davisking/dlib-models` at a pinned commit over HTTPS and must match pinned SHA-256 values; a file that does not match is refused. Set `FACEEVAL_NO_DOWNLOAD=1` to forbid downloads.
- The InsightFace version used for the ArcFace run is **not** in the delivered evidence. If `provenance.packages` exists in that run's `verification_results.json`, `env_pins --run <ARCFACE_RUN_DIR> --print` prints it as pins; if not, it refuses.
- Classical models: `classical_rng: global_seed` (default) is the historical protocol, where PCA's randomized solver and the SVMs draw from the global NumPy stream. `classical_rng: isolated_seed` is a separate, hashed protocol that gives each estimator `random_state=random_seed`; its numbers may differ and must not be merged with historical Fold-0 results.

## 6. What the project contains (one minute)

| Folder | Purpose |
|---|---|
| `faceeval/` | the Python package: data, models, perturbations, evaluation, scoring, orchestration |
| `configs/` | YAML files that define experiments |
| `tests/` | automated tests |
| `reports/` | tracked historical result files; newly generated analysis outputs are generally git-ignored and stay local |
| `docs/` | this guide, the presentation guide, the evidence record |
| `runs/`, `logs/` | created only when you start an experiment with the script |
| `data/` | the LFW images (not in Git) |

## 7. Dependencies, weights and data (only needed if you start experiments)

- Packages: `pip install -e ".[deep]"` in a fresh virtual environment.
  - **Use Python 3.11 or 3.12.** The `deep` extra installs `facenet-pytorch` (version 2.6.0 on PyPI when checked on 2026-11-18), which requires `torch>=2.2,<2.3`, `torchvision<0.18` and `numpy<2`. No torch 2.2.x wheel exists for Python 3.13 or 3.14 on macOS arm64, so the install is expected to fail there. `pyproject.toml` declares only `requires-python >= 3.11` and was not changed.
  - A clean-machine installation has **not** been tested.
  - Recorded interpreter evidence is limited: `reports/experiment_results/README.md` records Python 3.12.14 and NumPy 1.26.4 for the Fold-0 results archive. The interpreter used for run `2bad8efd` and for the other runs is NOT RECORDED.
  - If pip complains, keep your existing working environment.
- Dataset: `data/lfw/<Person_Name>/<Person_Name>_0001.jpg`, directly under `data/lfw/` (not `data/lfw/lfw/`). Override the location with `export FACEEVAL_DATA=/path/to/data`.
- Weights: FaceNet (VGGFace2) in `~/.cache/torch/checkpoints/`; ArcFace (`buffalo_l`, recognition model ResNet-50) in `~/.insightface/models/buffalo_l/`. They download on first use of a deep model; `check` only looks.

## 8. Intentionally starting an experiment

> **Warning.** These commands start real computation. `demo` takes minutes; `full` takes about an hour and uses several GB of memory (the recorded thesis run: about 58 minutes, 3.6 GB, ArcFace on CPU). Keep the Mac plugged in. You do **not** need either for presenting the thesis results. Each run goes into a new folder under `runs/`; nothing is overwritten and your `experiments/results.db` is not used.

```bash
./run_faceeval.sh demo --classical
```

Small run (20 people, 3 perturbations x 3 severities) with Eigenfaces and HOG-SVM only. The run always requires real models; `--classical` simply leaves FaceNet out, visibly.

```bash
./run_faceeval.sh demo
```

The same with FaceNet. If FaceNet or its weights are missing the run stops with an error; it does not use a fake model. If the weights are not cached it downloads them (internet needed).

```bash
./run_faceeval.sh full
```

Repeats the thesis experiment configuration (`configs/lfw_intermediate.yaml`). It first runs the checks, describes what will happen, and starts only if you type exactly `RUN FULL`. Typing anything else cancels. The result will be a *new* run; exact digits can differ slightly across machines and library versions, and the thesis results already exist.

`configs/lfw_full_benchmark.yaml` (10 models, 16 perturbations, 6 severities, all people) is refused unless you set `FACEEVAL_ALLOW_PAPER_SCALE=yes`. It has never been run and is not part of the thesis.

## 9. Troubleshooting

| What you see | What it means / what to do |
|---|---|
| `permission denied: ./run_faceeval.sh` | `chmod +x run_faceeval.sh` |
| `No such file or directory` for the script | You are in the wrong folder. Use `pwd` to see where you are and `cd` to the project (section 1). |
| `No Python environment found` | Set `export FACEEVAL_VENV=...` (section 2) or create one with Python 3.11 or 3.12: `python3.12 -m venv .venv && source .venv/bin/activate && pip install -e ".[deep]"` (if Python 3.11 is the version you have installed, use `python3.11 -m venv .venv` instead of `python3.12 -m venv .venv`) |
| `[FAIL] facenet-pytorch missing` / `insightface missing` | Only matters for deep runs. `pip install -e ".[deep]"` with Python 3.11 or 3.12, or use `demo --classical`. Reading results is unaffected. |
| `pip install -e ".[deep]"` fails resolving `torch` | The environment is probably Python 3.13 or 3.14, for which `facenet-pytorch`'s torch pin has no wheels. Use Python 3.11 or 3.12. |
| `[WARN] weights not cached` | Fine for `check`. A deep run would download them. |
| `LFW dataset not found` | Create or link `data/lfw/...`, or set `FACEEVAL_DATA`. |
| `nested layout` failure | The people folders are in `data/lfw/lfw/`; move them up one level. |
| `faceeval is imported from a DIFFERENT folder` | Your environment has an editable install of another copy. The script forces this folder; note it in the evidence record. |
| `Run '2bad8efd' was not found` | The result CSVs are in another folder. Find them: `find ~ -name "evaluation_results_2bad8efd*"` and run `results` from the project that holds them. |
| A run stops with a stub or "Preflight failed" message | Working as designed: a real backbone is unavailable. Fix the install; never remove `require_real_models`. |
| Run is slow | ArcFace uses the CPU on a Mac. Plug in the laptop; `caffeinate` keeps it awake. |
| Lots of `PREDICTIONS DEBUG` lines | Hidden on screen; present in the log file in `logs/`. |
| `mlflow` messages | `demo` and `full` do not start MLflow tracking, so no MLflow warnings are expected while they run. `check` imports `mlflow` only to report its version; any message printed at that point comes from the package itself and does not affect results. |
| Disk full | Delete old folders under `runs/` yourself; the script never deletes anything. |

## 10. Optional: API and dashboard

```bash
cp .env.example .env
uvicorn api.main:app --port 8000
cd dashboard && npm install && npm run dev      # http://localhost:5173
```

They only read `experiments/results.db`. They are not needed for the thesis numbers and were not part of the verification reported in the thesis.
