# FaceEval-X Presentation Guide (10–12 minutes, no new experiment)

Everything below uses **existing** results. You do not need to start any experiment to present. Rehearse the commands once on your own Mac before the day; only then do you know the exact output on your machine.

## 0. One-sentence pitch

"FaceEval-X is a configuration-driven framework that measures how face recognition models behave under image degradation. Clean accuracy hides failure modes; I show this with a 200-subject LFW identification validation of four models, and I keep a separate custom verification benchmark (six models, 16 perturbations x 6 severities) as supplementary evidence."

## 1. Preparation (day before; all read-only)

1. Open Terminal: `cd ~/Thesis/faceeval_x_patch_verify` and run `./run_faceeval.sh help`, `check`, `results`, `evidence 2bad8efd` once. Keep the `evidence` output (redirect it to a file) for the evidence record.
2. Increase the terminal font size, plug in the laptop, turn off notifications.
3. Open `configs/lfw_intermediate.yaml` and `reports/experiment_results/README.md` in a text editor for later.
4. Optional (not an experiment): run `pytest tests/test_perturbation_value_range.py -q` once so you know it passes on your machine. The full suite is not needed.
5. Optional: if you want to show Fold-0 figures, open the local six-model analysis output folder (the one you passed to `--out`). It is a **local artifact, not in Git**; check on your Mac that it is there and has 13 tables and 7 figures. Do not regenerate it live.

## 2. Sequence

| Min | Show | Command / action | Say |
|---|---|---|---|
| 0–1 | Scope slide | capability versus validation table (section 5) | "Implemented: 10 models, 16 perturbations. Validated here: 200 subjects, 4 models, 3 perturbations, 3 severities." |
| 1–3 | Architecture | pipeline diagram (section 3) | explain the flow once |
| 3–4 | Safe environment | `./run_faceeval.sh check` | "It only looks. Real packages and cached weights are present; a missing deep package would stop a run, never produce fake numbers." |
| 4–8 | The 36-row experiment | `./run_faceeval.sh results` | section 4 below |
| 8–9 | One file defines it | `less configs/lfw_intermediate.yaml` (`q` to quit) | "Models, perturbations, severities, seed 42." |
| 9–10 | Provenance | `./run_faceeval.sh evidence 2bad8efd` | "The run recorded its commit, configuration hash and timing. What it did not record, I label NOT RECORDED." |
| 10–11 | Fold-0 (separately) | open `reports/experiment_results/README.md`; optionally the local six-model analysis figures | section 4b below |
| 11–12 | Limits | limitations slide (section 6) | say them before you are asked |

## 3. Architecture and how the pieces work

```
YAML config -> load + validate -> LFW images -> stratified split (seed 42)
  -> preprocessing (per model) -> training / enrolment
  -> for each model x perturbation x severity: perturb TEST images -> predict -> metrics
  -> robustness fingerprint -> deployment scoring -> CSV + SQLite + log
```

| Module (`faceeval/`) | Purpose |
|---|---|
| `core` | types, validated YAML configuration, registries |
| `data` | LFW loading, stratified splitting, caching |
| `preprocessing` | detect / crop / align / normalise for each model |
| `models` | 6 traditional and 4 deep recognisers behind one interface |
| `perturbation` | 16 degradations with a severity scale from 0.0 to 1.0 |
| `evaluation` | accuracy / F1 and further metric code |
| `fingerprint` | robustness-AUC profile per model |
| `failure` | failure clustering |
| `deployment` | scenario weights and scoring |
| `orchestration` | `ExperimentRunner` ties everything together |
| `reporting`, `storage` | CSV reports and the SQLite store |

Deep models are pretrained and frozen: each person is enrolled as the normalised mean embedding of their training images, and a test image goes to the person with the highest cosine similarity. Perturbations touch test images only.

## 4. Explaining the 36-row experiment (run 2bad8efd, closed-set identification)

36 rows = 4 models x 3 perturbations x 3 severities, each on 1,357 test images of 200 people.

Recorded values (accuracy %; `results` should show the same):

| Model | Clean | Blur 0.5 | Blur 1.0 | Rotation 1.0 | Brightness 1.0 |
|---|---|---|---|---|---|
| Eigenfaces | 21.6 | 16.2 | 12.5 | 5.0 | 13.8 |
| HOG-SVM | 22.3 | 14.4 | 11.9 | 11.7 | 21.1 |
| FaceNet | 99.8 | 59.9 | 1.0 | 77.3 | 98.0 |
| ArcFace | 99.6 | 57.8 | 0.2 | 99.6 | 99.3 |

1. Deep models are far above traditional ones on clean images.
2. Strong blur (sigma 8) collapses both deep models below the traditional ones. Clean accuracy did not predict this.
3. Rotation separates the deep models: ArcFace is unaffected, FaceNet falls to 77% at 45 degrees.
4. Brightness is **one** brightening level (+0.40); severity 0.5 equals the clean image. Say "limited evidence".
5. The deployment ranking (ArcFace 0.568, FaceNet 0.563, HOG-SVM 0.436, Eigenfaces 0.432) uses accuracy only; the other dimensions are placeholders (0.5). It demonstrates the scoring mechanism and is **not** a validated multi-dimensional ranking. The 0.005 gap is not meaningful.
6. If asked about the earlier run `fe4696ee`: it was found defective (FaceNet 61.6%) and replaced; never show it beside these numbers.

### 4b. Fold-0 verification, discussed separately

Say: "This is a different protocol. It asks whether two photos show the same person, on people never seen in training, with a threshold frozen on validation pairs. The split is our own custom identity-disjoint Fold 0, not the official LFW View 2 protocol, so I do not compare it with published View 2 numbers. It is supplementary evidence of the implementation's rigour, not part of the 36-row comparison."

- Committed results: four classical models over 16 perturbations x 6 severities (`reports/experiment_results/`; Fisherfaces, LBPH, HOG-SVM 288 rows; Eigenfaces 96 rows; the separate historical per-image-noise run `c408e0d2` is not to be combined).
- FaceNet and ArcFace were also run over the full 16 x 6 grid (96 conditions each, both complete, as reported from the Mac). Their result files are **not in Git**. The earlier two-row clean-only checks were only a backbone sanity check.
- Six-model analysis: 6 models x 16 x 6 = 576 rows, 13 tables, 7 figures, generated locally by `python -m faceeval.analysis.fold0_results` (see the README); the outputs are **not committed**.
- Say about ArcFace: "These are end-to-end results. When ArcFace's detector finds no face, the image gets a zero embedding and its pairs are still scored, not removed, so ArcFace AUC and EER include detection failures. The failure rates are in the failure tables and figure." Do not call them recognition-only numbers.
- Say about FaceNet versus ArcFace: "They differ in crop or alignment, input size and detection as well as in backbone, so I compare pipelines, not backbones."
- Say about brightness, contrast and gamma: "Their neutral point is near severity 0.5, so the curves are V-shaped; read them as distance from neutral."
- Show the manifest SHA-256 in the README as a reproducibility feature. Do not put identification and verification numbers in one table.

## 5. Capability versus validation (slide)

| Dimension | Implemented | Validated in the thesis |
|---|---|---|
| Subjects | 5,749 | 200 (3.5%) |
| Models | 10 | 4 (identification); 4 classical + 2 deep, 96 conditions each (Fold 0) |
| Perturbations | 16 | 3 (identification); 16 for all 6 models (Fold 0) |
| Severities | 6-level grid | 3 levels (identification); all 6 levels (Fold 0) |
| Deployment dimensions | 7 | 1 populated (accuracy) |

## 6. Limitations slide (say it before you are asked)

- Reduced scope as above; one run, one split, seed 42; no confidence intervals (about +/-2.7 points at 60% accuracy for n = 1,357, a lower bound).
- Closed-set identification with an image-level split; not comparable with published LFW verification numbers.
- Brightness (identification run): one level; darkening untested.
- Fold-0: custom split, not View 2; one split and one seed; brightness, contrast and gamma are centred (V-shaped curves); ArcFace values are end-to-end with failed embeddings scored as zero vectors; FaceNet and ArcFace pipelines differ; no valid-pairs-only analysis yet.
- Calibration, fairness, latency, memory, cost not evaluated.
- A clean-machine installation and a repeat of the run have not been shown.

## 7. If something goes wrong live

| Problem | Action |
|---|---|
| A command prints a red `[FAIL]` | Say what it means (section 9 of the user guide), then continue with `results`, which does not need the deep packages. |
| `results` cannot find 2bad8efd | Show the CSV folder directly in Finder or with `ls reports/lfw_intermediate_rerun`. |
| Output too long | `./run_faceeval.sh results | less` |
| A question you cannot answer | "That was not evaluated in this validation; it is listed as a limitation or future work." |

You may mention that `demo` and `full` exist and that they start experiments, but do not run them live.

## 8. Likely examiner questions

1. **Why did FaceNet change from 61.6% to 99.8%?** The earlier run had preprocessing and perturbation defects found by audit and fixed; the thesis reports the rerun and documents the retired run.
2. **How do you know the deep models are real?** The rerun was started with `require_real_models: true`, which aborts on a stub, and 99.8% / 99.6% over 200 classes is impossible for random embeddings. The Fold-0 clean check also recorded zero stub backbones.
3. **Why do both deep models collapse under blur?** Sigma 8 removes the detail they rely on and they were not trained for it. That is an observation; remedies were not tested.
4. **Is brightness really tested?** One level, +0.40; severity 0.5 is the identity.
5. **Why is ArcFace first in the deployment ranking?** Only accuracy is populated; the gap is 0.005 and not meaningful.
6. **Why no calibration or fairness results?** The code exists but was not evaluated; they are stated limitations.
7. **Why are Eigenfaces and HOG-SVM only about 22%?** 200 classes with about 13 training images each, unconstrained photos, untuned defaults. They are baselines.
8. **Is 200 subjects representative?** No; it is a feasibility validation.
9. **Can I compare with published LFW numbers?** No; those are verification numbers.
10. **Isn't an image-level split leaky?** By design for closed-set identification; near-duplicate photos can inflate accuracy. The identity-disjoint verification protocol exists separately.
11. **Repeats or confidence intervals?** None.
12. **Can I reproduce it?** The configuration, split and scoring arithmetic are reproducible; the run's database record stores its commit and config hash; Python and package versions, weight hashes and dataset fingerprint were not recorded at run time, and the evidence record says so.
13. **Why not the full 10-model, 16 x 6 benchmark?** No claim in the thesis needs it; the scope is declared as an intermediate-scale validation.
14. **Why does ArcFace have failures but FaceNet none?** FaceNet uses a fixed centre crop and has no face-detection stage that can fail; ArcFace uses an internal face detector and a failed detection gives a zero embedding. The pairs are still scored, so the numbers are end-to-end. The counts are in the failure tables.
15. **Is ArcFace's Fold-0 AUC its true recognition performance?** No. It includes detection failures. A valid-pairs-only analysis from the saved scores has not been done.
16. **Can I put the identification and Fold-0 numbers together?** No. Different protocols (image-level closed-set identification versus identity-disjoint verification); they are kept apart.
17. **What is your contribution?** A modular, config-driven framework with robustness fingerprints and context-weighted scoring, a worked validation showing that clean accuracy hides failure modes, and an audit trail that caught and replaced invalid results.

## Evidence status to state honestly (updated 2026-11-19)

- Completed and reported: custom Fold-0 verification for six models (576 rows). Not official LFW View 2; do not compare with published LFW numbers.
- Implemented and tested on fake or synthetic data only: calibration in the runner, multi-split runs and aggregation, valid-pairs-only analysis, efficiency harness, fairness interface, JSON export. Say "implemented, not yet run on the real models" for each.
- Not available: any fairness result on LFW (no demographic labels), any real runtime/memory number, real insightface/dlib results.
- Deep-model numbers are end-to-end (failed embeddings are zero vectors). Say so whenever ArcFace is shown.
