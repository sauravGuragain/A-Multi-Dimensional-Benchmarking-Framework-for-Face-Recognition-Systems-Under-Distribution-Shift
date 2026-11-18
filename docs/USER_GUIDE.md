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

The runner is delivered as a separate file, `run_faceeval.final.sh`, so that it never overwrites a runner you already have. Review it first (`diff run_faceeval.sh run_faceeval.final.sh | less`). If you accept it, copy it into the project folder as `run_faceeval.sh` (keep a backup of the old one). The commands below assume the final runner is installed under that name. If you prefer to leave your own runner in place, replace `./run_faceeval.sh` with `./run_faceeval.final.sh` in every command in this guide; the script works under any file name.


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

**Verification (Fold 0) is different.** Verification asks "are these two photos the same person?" using pairs of images and a decision threshold, on people never seen in training. Its numbers (TAR, FAR, balanced accuracy) live in `reports/experiment_results/` and are described in its README. Never compare an identification accuracy with a verification accuracy, and never put them in one table.

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

A difference of a few images (FaceNet 99.8% versus ArcFace 99.6% is three images) is not meaningful.

## 6. What the project contains (one minute)

| Folder | Purpose |
|---|---|
| `faceeval/` | the Python package: data, models, perturbations, evaluation, scoring, orchestration |
| `configs/` | YAML files that define experiments |
| `tests/` | automated tests |
| `reports/` | committed result files of earlier experiments |
| `docs/` | this guide, the presentation guide, the evidence record |
| `runs/`, `logs/` | created only when you start an experiment with the script |
| `data/` | the LFW images (not in Git) |

## 7. Dependencies, weights and data (only needed if you start experiments)

- Packages: `pip install -e ".[deep]"` in a fresh virtual environment (Python 3.11+). `facenet-pytorch` pins a `torch` range; if pip complains, keep your existing working environment.
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
| `No Python environment found` | Set `export FACEEVAL_VENV=...` (section 2) or create one: `python3 -m venv .venv && source .venv/bin/activate && pip install -e ".[deep]"` |
| `[FAIL] facenet-pytorch missing` / `insightface missing` | Only matters for deep runs. `pip install -e ".[deep]"`, or use `demo --classical`. Reading results is unaffected. |
| `[WARN] weights not cached` | Fine for `check`. A deep run would download them. |
| `LFW dataset not found` | Create or link `data/lfw/...`, or set `FACEEVAL_DATA`. |
| `nested layout` failure | The people folders are in `data/lfw/lfw/`; move them up one level. |
| `faceeval is imported from a DIFFERENT folder` | Your environment has an editable install of another copy. The script forces this folder; note it in the evidence record. |
| `Run '2bad8efd' was not found` | The result CSVs are in another folder. Find them: `find ~ -name "evaluation_results_2bad8efd*"` and run `results` from the project that holds them. |
| A run stops with a stub or "Preflight failed" message | Working as designed: a real backbone is unavailable. Fix the install; never remove `require_real_models`. |
| Run is slow | ArcFace uses the CPU on a Mac. Plug in the laptop; `caffeinate` keeps it awake. |
| Lots of `PREDICTIONS DEBUG` lines | Hidden on screen; present in the log file in `logs/`. |
| `mlflow` warnings | Tracking is optional; results are still saved. |
| Disk full | Delete old folders under `runs/` yourself; the script never deletes anything. |

## 10. Optional: API and dashboard

```bash
cp .env.example .env
uvicorn api.main:app --port 8000
cd dashboard && npm install && npm run dev      # http://localhost:5173
```

They only read `experiments/results.db`. They are not needed for the thesis numbers and were not part of the verification reported in the thesis.
