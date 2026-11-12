from __future__ import annotations

import logging
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

import numpy as np

from faceeval.core.exceptions import OrchestratorError
from faceeval.core.registry import model_registry, perturbation_registry
from faceeval.orchestration.perturbed_inputs import needs_chips, perturb_batch
from faceeval.core.types import (
    BehavioralFingerprint,
    DataSplit,
    DeploymentRanking,
    EvaluationResult,
    ExperimentConfig,
    ExperimentRun,
    ExperimentStatus,
    FailureCluster,
    FairnessReport,
    ModelParadigm,
    PerturbationSpec,
    PerturbationType,
    RunID,
)

logger = logging.getLogger(__name__)

ProgressCallback = Callable[[str, int, int], None]


class ExperimentRunner:
    """
    Orchestrates a complete FaceEval-X experiment run.

    Parameters
    ----------
    config:
        Validated ``ExperimentConfig`` from ``load_config()``.
    progress_callback:
        Optional callback ``(stage, completed, total) → None``.
    """

    def __init__(
        self,
        config: ExperimentConfig,
        progress_callback: ProgressCallback | None = None,
    ) -> None:
        self._config = config
        self._cb = progress_callback or (lambda stage, done, total: None)
        self._run: ExperimentRun | None = None
        self.verification_summary: dict | None = None
        self._preflight_kept: dict = {}   # at most one preflight-loaded model, reused by its first user
        self.failures: list[dict] = []   # failed models/conditions (see _raise_if_failures)

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    def run(self) -> ExperimentRun:
        """
        Execute the full experiment and return the completed ``ExperimentRun``.

        This method:
        1.  Seeds all RNGs.
        2.  Sets up the tracker and output directories.
        3.  Loads datasets and builds preprocessing pipelines.
        4.  Runs every (model × perturbation × severity) condition.
        5.  Computes fingerprints, failure clusters, deployment scores.
        6.  Generates all figures and the structured report.
        7.  Logs everything to MLflow.
        """
        config = self._config
        run = ExperimentRun(
            run_id=config.run_id,
            config=config,
            status=ExperimentStatus.RUNNING,
            started_at=datetime.utcnow(),
        )
        self._run = run

        try:
            self._setup(run)
            self._execute_conditions(run)
            self._post_process(run)
            run.status = ExperimentStatus.COMPLETED
            run.completed_at = datetime.utcnow()

            from faceeval.storage.result_store import ResultStore
            ResultStore().save(run)

            logger.info(
                "Experiment '%s' completed in %.1f s.",
                config.experiment_name,
                run.duration_seconds or 0.0,
            )
        except Exception as exc:
            run.status = ExperimentStatus.FAILED
            run.error_message = str(exc)
            run.completed_at = datetime.utcnow()

            from faceeval.storage.result_store import ResultStore
            ResultStore().save(run)

            logger.error("Experiment '%s' FAILED: %s", config.experiment_name, exc)
            raise OrchestratorError(f"Run '{config.run_id}' failed: {exc}") from exc

        return run

    # ------------------------------------------------------------------
    # Stage 1: Setup
    # ------------------------------------------------------------------

    def _setup(self, run: ExperimentRun) -> None:
        """Seed RNGs, create output dirs, start tracker, register plugins."""
        self._cb("setup", 0, 1)
        from faceeval.orchestration.reproducibility import set_global_seeds, capture_environment_snapshot
        set_global_seeds(self._config.random_seed, self._config.deterministic_mode)

        # Create output directories
        Path(self._config.output_dir).mkdir(parents=True, exist_ok=True)
        Path(self._config.report_dir).mkdir(parents=True, exist_ok=True)
        Path(self._config.feature_cache_dir).mkdir(parents=True, exist_ok=True)

        # Import model and perturbation registrations
        import faceeval.models          # noqa: F401
        import faceeval.perturbation    # noqa: F401
        import faceeval.data.loaders    # noqa: F401  (triggers @register_dataset decorators)

        env = capture_environment_snapshot()
        logger.info("Environment: %s", env.get("platform", ""))
        self._cb("setup", 1, 1)

    # ------------------------------------------------------------------
    # Stage 2: Evaluation loop
    # ------------------------------------------------------------------

    def _execute_conditions(self, run: ExperimentRun) -> None:
        """
        Run every (model × perturbation × severity) evaluation condition
        through the real pipeline: preprocess → perturb → predict → evaluate.
        """
        from faceeval.data.splitter import DatasetSplitter
        from faceeval.evaluation.metrics import compute_identification_metrics
        from faceeval.perturbation.composer import PerturbationComposer
        from faceeval.preprocessing.pipeline import pipeline_for_model
        from faceeval.core.types import ModelParadigm, SplitStrategy

        config = self._config

        # --- 1. Data: synthetic in-memory dataset (no download required) ---
        self._cb("data", 0, 1)
        records, raw_images = self._load_or_generate_data()
        self._cb("data", 1, 1)
        protocol = getattr(config, "evaluation_protocol", "identification")

        splitter = DatasetSplitter(
            seed=config.random_seed,
            test_fraction=config.test_fraction,
            val_fraction=config.val_fraction,
        )
        # NOTE: STRATIFIED, not SUBJECT_DISJOINT, is used here deliberately.
        # This runner computes closed-set identification accuracy (predict
        # which known subject a test image belongs to), which requires every
        # subject to appear in training — only held-out *images* of each
        # subject should be excluded. SUBJECT_DISJOINT (no subject overlap
        # between train/test) is the correct strategy for open-set
        # verification benchmarks, not for the identification metrics this
        # runner reports. See faceeval.data.splitter module docstring.
        dataset_name = config.dataset_names[0] if config.dataset_names else "synthetic"
        run_identification = protocol in ("identification", "both")
        run_verification = protocol in ("lfw_fold0_verification", "both")
        if run_verification:
            if dataset_name != "lfw":
                raise OrchestratorError(
                    "evaluation_protocol requires dataset_names: [lfw] "
                    f"(got {config.dataset_names})."
                )
        split = (splitter.split(records, config.split_strategy, dataset_name=dataset_name)
                 if run_identification else None)

        records_by_id = {r.image_id: r for r in records}
        train_records = [records_by_id[i] for i in split.train_ids] if split else []
        test_records = [records_by_id[i] for i in split.test_ids] if split else []
        n_classes = len({r.subject_id for r in records})

        if run_identification and (not train_records or not test_records):
            raise OrchestratorError(
                "Synthetic split produced an empty train or test set — "
                "increase n_subjects in the synthetic data generator."
            )
                # DEBUG: Log split statistics
        train_subject_ids = sorted({r.subject_id for r in train_records})
        test_subject_ids = sorted({r.subject_id for r in test_records})
        logger.info(
            "[SPLIT DEBUG] Total subjects: %d, Train subjects: %d, Test subjects: %d",
            n_classes, len(train_subject_ids), len(test_subject_ids)
        )
        logger.info(
            "[SPLIT DEBUG] Total records: %d, Train records: %d, Test records: %d",
            len(records), len(train_records), len(test_records)
        )
        logger.info("[SPLIT DEBUG] Train subject IDs: %s", train_subject_ids[:5])
        logger.info("[SPLIT DEBUG] Test subject IDs: %s", test_subject_ids[:5])

        # --- 2. Perturbation schedule ---
        composer = PerturbationComposer(
            severity_levels=config.perturbation_schedule.severity_levels
            if config.perturbation_schedule else [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
        )
        perturbation_types = (
            config.perturbation_schedule.perturbation_types
            if config.perturbation_schedule
            else list(PerturbationType)[:3]
        )
        all_specs: list[PerturbationSpec] = composer.specs_for_types(perturbation_types)

        model_names = (
            [m.value for m in config.traditional_models]
            + [m.value for m in config.deep_models]
        )
        trad_names = {m.value for m in config.traditional_models}
        if config.require_real_models:
            self._preflight_models(model_names, trad_names)
        if run_verification:
            from faceeval.orchestration.verification import run_fold0_verification
            self.verification_summary = run_fold0_verification(
                config=config, records=records, raw_images=raw_images,
                model_names=model_names, trad_names=trad_names,
                composer=composer, specs=all_specs, dataset_name=dataset_name,
                preprocess=self._preprocess_records,
                preprocess_chips=self._aligned_chips,
                adapt_hyperparameters=self._adapt_hyperparameters_to_data_size,
                progress=self._cb,
                get_model=self._get_model,
            )
            self.failures.extend(self.verification_summary["failures"])
        if not run_identification:
            self._raise_if_failures()
            return

        total_conditions = len(model_names) * len(all_specs)
        self._cb("evaluation", 0, total_conditions)
        condition_count = 0

        for model_name in model_names:
            paradigm = (
                ModelParadigm.TRADITIONAL if model_name in trad_names
                else ModelParadigm.DEEP_LEARNING
            )

            try:
                model = self._get_model(model_name)
            except Exception as exc:
                if config.require_real_models and model_name not in trad_names:
                    raise OrchestratorError(
                        f"Model '{model_name}' could not be instantiated and "
                        f"require_real_models=true: {exc}") from exc
                logger.warning("Skipping model '%s': could not instantiate (%s)", model_name, exc)
                self.failures.append({"protocol": "identification", "model": model_name,
                                      "stage": "instantiate", "error": f"{type(exc).__name__}: {exc}"})
                condition_count += len(all_specs)
                self._cb("evaluation", condition_count, total_conditions)
                continue

            pipeline = pipeline_for_model(model_name, skip_detection_for_pre_cropped=True)

            # --- 3. Preprocess clean train/test arrays once per model ---
            try:
                X_train = self._preprocess_records(pipeline, train_records, raw_images)
                y_train = [r.subject_id for r in train_records]
                test_chips = None
                if needs_chips(pipeline.normalizer):
                    # Perturb the aligned uint8 chip, then normalise (see perturbed_inputs).
                    test_chips = self._aligned_chips(pipeline, test_records, raw_images)
                    X_test_clean = [pipeline.normalizer.normalize(c) for c in test_chips]
                else:
                    X_test_clean = self._preprocess_records(pipeline, test_records, raw_images)
                y_test = [r.subject_id for r in test_records]
            except Exception as exc:
                logger.warning("Preprocessing failed for '%s': %s", model_name, exc)
                self.failures.append({"protocol": "identification", "model": model_name,
                                      "stage": "preprocess", "error": f"{type(exc).__name__}: {exc}"})
                condition_count += len(all_specs)
                self._cb("evaluation", condition_count, total_conditions)
                continue

            # --- 4. Train once on clean data ---
            self._adapt_hyperparameters_to_data_size(model, model_name, len(X_train))
            try:
                model.fit(X_train, y_train)
            except Exception as exc:
                logger.warning("Training failed for '%s': %s", model_name, exc)
                self.failures.append({"protocol": "identification", "model": model_name,
                                      "stage": "fit", "error": f"{type(exc).__name__}: {exc}"})
                condition_count += len(all_specs)
                self._cb("evaluation", condition_count, total_conditions)
                continue

            if config.require_real_models and hasattr(model, "assert_real_backbone"):
                try:
                    model.assert_real_backbone()
                except Exception as exc:
                    raise OrchestratorError(str(exc)) from exc

            logger.info("[%s] trained on %d samples, %d test samples.",
                        model_name, len(X_train), len(X_test_clean))

            # --- 5. For each perturbation spec: perturb test set, predict, evaluate ---
            for spec in all_specs:
                try:
                    X_test_perturbed = self._apply_perturbation(
                        composer, spec, test_records, X_test_clean,
                        normalizer=pipeline.normalizer, chips=test_chips,
                    )
                    predictions = model.predict(X_test_perturbed, perturbation_spec=spec)
                    metrics = compute_identification_metrics(
                        predictions, y_test, class_labels=sorted(set(y_train) | set(y_test)),
                    )
                                        # DEBUG: Log sample predictions vs ground truth
                    if spec.severity == 0.0:
                        logger.info(
                            "[PREDICTIONS DEBUG] %s @ severity 0.0 — accuracy: %.4f",
                            model_name, metrics["accuracy"]
                        )

                        for i in range(min(5, len(predictions))):
                            pred_id = predictions[i].predicted_subject_id or "NONE"
                            true_id = y_test[i]
                            match = "✓" if pred_id == true_id else "✗"
                            logger.info(
                                "[PREDICTIONS DEBUG]   Sample %d: predicted=%s, true=%s %s",
                                i, pred_id, true_id, match
                            )

                        cm = metrics["confusion_matrix"]
                        cm_diag_sum = sum(
                            cm[i][i] for i in range(len(cm))
                        )
                        logger.info(
                            "[PREDICTIONS DEBUG]   Confusion matrix diagonal sum (correct): %d / %d",
                            cm_diag_sum, metrics["num_samples"]
                        )

                    result = EvaluationResult(
                        run_id=run.run_id,
                        model_name=model_name,
                        model_paradigm=paradigm,
                        dataset_name=dataset_name,
                        perturbation_spec=spec,
                        accuracy=metrics["accuracy"],
                        precision=metrics["precision_macro"],
                        recall=metrics["recall_macro"],
                        f1_score=metrics["f1_macro"],
                        top_k_accuracy=metrics["top_k_accuracy"],
                        confusion_matrix=metrics["confusion_matrix"],
                        class_labels=metrics["class_labels"],
                        num_samples=metrics["num_samples"],
                        num_subjects=n_classes,
                    )
                    run.evaluation_results.append(result)
                except Exception as exc:
                    logger.warning(
                        "Condition %s/%s failed: %s", model_name, spec.label, exc
                    )
                    self.failures.append({"protocol": "identification", "model": model_name,
                                          "stage": spec.label, "error": f"{type(exc).__name__}: {exc}"})

                condition_count += 1
                self._cb("evaluation", condition_count, total_conditions)

        logger.info(
            "Evaluation loop complete: %d real conditions recorded.",
            len(run.evaluation_results),
        )
        self._raise_if_failures()

    def _raise_if_failures(self) -> None:
        """A run with failed models/conditions must not be reported as successful."""
        if not self.failures or self._config.allow_partial_results:
            if self.failures:
                logger.error("%d model/condition failure(s) ignored (allow_partial_results=true).",
                             len(self.failures))
            return
        head = "; ".join(f"{f['protocol']}:{f['model']}:{f['stage']}: {f['error']}"
                         for f in self.failures[:5])
        raise OrchestratorError(
            f"{len(self.failures)} model/condition failure(s); partial outputs were kept on disk. "
            f"First: {head}. Set allow_partial_results: true to accept partial results."
        )

    def _get_model(self, name: str):
        """Registry instance; hands out the preflight-loaded model once instead of reloading it."""
        kept = self._preflight_kept.pop(name, None)
        return kept if kept is not None else model_registry.get(name)

    def _preflight_models(self, model_names: list, trad_names: set) -> None:
        """Load every deep model once, before any expensive work, and require a real backbone."""
        import gc
        deep = [n for n in model_names if n not in trad_names]
        # Check in reverse so the model still held at the end is the first one evaluated: it is
        # reused instead of loaded twice. Only one model is ever held (no extra memory).
        for name in reversed(deep):
            self._preflight_kept.clear()
            gc.collect()
            try:
                model = model_registry.get(name)
                if hasattr(model, "assert_real_backbone"):
                    model.assert_real_backbone()
            except Exception as exc:
                raise OrchestratorError(
                    f"Preflight failed for '{name}' (require_real_models=true): {exc}") from exc
            self._preflight_kept[name] = model
            del model
            logger.info("Preflight OK: '%s' has a real pretrained backbone.", name)

    def _load_or_generate_data(self) -> tuple[list, dict]:
        """
        Resolve the data source: synthetic (in-memory) or LFW (disk-based).

        Synthetic: Returns (records, raw_images_dict) for in-memory preprocessing.
        LFW: Returns (records, None); preprocessing uses PreprocessingPipeline.process().
        """
        config = self._config

        # Synthetic path: in-memory dataset with pre-loaded pixels
        if not config.dataset_names or "synthetic" in config.dataset_names:
            from faceeval.orchestration.synthetic_data import generate_synthetic_dataset
            logger.info("Generating synthetic dataset for run '%s'...", config.run_id)
            return generate_synthetic_dataset(
                n_subjects=20, images_per_subject=12, seed=config.random_seed,
            )

        # LFW path: disk-based dataset with on-demand pixel loading via pipeline
        if "lfw" in config.dataset_names:
            from faceeval.data.dataset_manager import DatasetManager
            manager = DatasetManager(data_root=config.data_root)
            try:
                manager.register("lfw")
            except Exception as e:
                raise OrchestratorError(
                    f"Failed to register LFW dataset: {e}. "
                    "Ensure LFW is downloaded to data/lfw/ from https://vis-www.cs.umass.edu/lfw/"
                )
            records = manager.get_records("lfw")

            # Optional deterministic subset for smoke tests / small validation runs.
            # Defaults (min_images_per_subject=0, max_subjects=None) preserve
            # the complete LFW dataset for the full benchmark.
            min_images = config.min_images_per_subject
            max_subjects = config.max_subjects

            if min_images > 0 or max_subjects is not None:
                from collections import defaultdict
                import random

                by_subject = defaultdict(list)
                for record in records:
                    by_subject[record.subject_id].append(record)

                eligible_subjects = sorted(
                    subject
                    for subject, subject_records in by_subject.items()
                    if len(subject_records) >= min_images
                )

                if max_subjects is not None and len(eligible_subjects) > max_subjects:
                    rng = random.Random(config.random_seed)
                    rng.shuffle(eligible_subjects)
                    eligible_subjects = sorted(eligible_subjects[:max_subjects])

                selected = set(eligible_subjects)
                records = [
                    record for record in records
                    if record.subject_id in selected
                ]

                logger.info(
                    "LFW subset selected: %d images, %d subjects "
                    "(min_images_per_subject=%d, max_subjects=%s, seed=%d)",
                    len(records),
                    len(selected),
                    min_images,
                    max_subjects,
                    config.random_seed,
                )

            logger.info(
                "LFW registered: %d total images, %d unique subjects",
                len(records), len(set(r.subject_id for r in records))
            )
            return records, None  # None signals pixel loading via pipeline

        raise OrchestratorError(
            f"Unknown dataset_names: {config.dataset_names}. "
            f"Valid: ['synthetic'], ['lfw'], or empty (defaults to synthetic)."
        )

    @staticmethod
    def _aligned_chips(pipeline, records, raw_images: dict) -> list:
        """
        Load, detect and align every record; return uint8 BGR face chips (not normalised).

        - Synthetic: raw_images dict contains pre-loaded pixels; use directly.
        - LFW/real: raw_images is None; use disk-based pixel loading via cv2.imread.
        """
        import cv2
        chips = []
        for rec in records:
            # Synthetic path: in-memory pixel data
            if raw_images is not None:
                img_bgr = raw_images[rec.image_id]
                detection = pipeline._whole_image_detection(img_bgr, rec.image_path)
                aligned = pipeline.aligner.align(img_bgr, detection)
                if aligned is None:
                    raise RuntimeError(f"Alignment failed for synthetic image '{rec.image_id}'")
                chips.append(aligned.image)
            # LFW/real path: disk-based pixel loading via pipeline
            else:
                img_bgr = cv2.imread(rec.image_path)
                if img_bgr is None:
                    raise RuntimeError(
                        f"Failed to load image from disk: '{rec.image_path}'. "
                        "File may not exist or is unreadable."
                    )
                detection = pipeline._whole_image_detection(img_bgr, rec.image_path)
                aligned = pipeline.aligner.align(img_bgr, detection)
                if aligned is None:
                    raise RuntimeError(f"Alignment failed for image '{rec.image_path}'")
                chips.append(aligned.image)
        return chips

    @staticmethod
    def _preprocess_records(pipeline, records, raw_images: dict) -> list:
        """Run the preprocessing pipeline (align, then normalise) on a list of records."""
        chips = ExperimentRunner._aligned_chips(pipeline, records, raw_images)
        return [pipeline.normalizer.normalize(chip) for chip in chips]

    @staticmethod
    def _apply_perturbation(composer, spec, records, clean_arrays: list,
                            normalizer=None, chips=None) -> list:
        """Apply one PerturbationSpec to a preprocessed batch.

        When ``chips`` (uint8 aligned faces) are given, the perturbation is applied to the
        chip and the result normalised; otherwise it is applied to ``clean_arrays`` directly.
        """
        return perturb_batch(composer, spec, normalizer, clean_arrays, chips, [r.image_id for r in records])

    @staticmethod
    def _adapt_hyperparameters_to_data_size(model, model_name: str, n_train: int) -> None:
        """
        Cap PCA-based component counts to the available training set size.

        PCA cannot extract more components than min(n_samples, n_features);
        with small training sets (e.g. the synthetic generator's default of
        a few dozen images, or a small real dataset) the default
        hyperparameters (tuned for thousands of LFW images) can exceed this
        bound and crash ``fit()``. This rescales any *_n_components-style
        private attribute on the model down to a safe ceiling before fitting.
        """
        safe_max = max(2, n_train - 1)
        for attr in ("_n_components", "_n_pca"):
            if hasattr(model, attr):
                current = getattr(model, attr)
                if isinstance(current, int) and current > safe_max:
                    logger.info(
                        "[%s] Capping %s from %d to %d for training set size %d.",
                        model_name, attr, current, safe_max, n_train,
                    )
                    setattr(model, attr, safe_max)

    # ------------------------------------------------------------------
    # Stage 3: Post-processing
    # ------------------------------------------------------------------

    def _post_process(self, run: ExperimentRun) -> None:
        """Fingerprinting, failure analysis, deployment scoring, reporting."""
        config = self._config
        self._cb("post_processing", 0, 4)

        # Fingerprints
        if config.compute_fingerprints and run.evaluation_results:
            self._cb("fingerprinting", 0, 1)
            from faceeval.fingerprint.builder import build_all_fingerprints
            model_paradigms = {
                m.value: ModelParadigm.TRADITIONAL for m in config.traditional_models
            }
            model_paradigms.update({
                m.value: ModelParadigm.DEEP_LEARNING for m in config.deep_models
            })
            run.fingerprints = build_all_fingerprints(
                evaluation_results=run.evaluation_results,
                model_paradigms=model_paradigms,
            )
            self._cb("fingerprinting", 1, 1)
            logger.info("Fingerprints built for %d models.", len(run.fingerprints))

        self._cb("post_processing", 1, 4)

        # Deployment scoring
        if config.compute_deployment_scores:
            from faceeval.deployment.scorer import DeploymentScorer
            # build_ranking is a scorer method
            from faceeval.core.types import DeploymentScenario

            scenarios = config.deployment_scenarios or [DeploymentScenario.RESEARCH_BENCHMARK]
            for scenario in scenarios:
                scorer = DeploymentScorer(scenario=scenario)
                model_data = {
                    m.value: {
                        "paradigm": ModelParadigm.TRADITIONAL,
                        "evaluation_results": [
                            r for r in run.evaluation_results if r.model_name == m.value
                        ],
                    }
                    for m in config.traditional_models
                }
                model_data.update({
                    m.value: {
                        "paradigm": ModelParadigm.DEEP_LEARNING,
                        "evaluation_results": [
                            r for r in run.evaluation_results if r.model_name == m.value
                        ],
                    }
                    for m in config.deep_models
                })
                if model_data:
                    scores = scorer.score_all_models(model_data)
                    ranking = scorer.build_ranking(scores)
                    run.deployment_rankings.append(ranking)

        self._cb("post_processing", 2, 4)

        # Report generation and CSV/JSON export
        if config.export_csv or config.export_json:
            from faceeval.reporting.generator import ReportGenerator

            reporter = ReportGenerator(
                output_dir=config.report_dir,
                run_id=run.run_id,
            )

            reporter.build_report(run)

            if config.export_csv:
                reporter.export_evaluation_results_csv(
                    run.evaluation_results
                )
                reporter.export_fingerprint_csv(
                    run.fingerprints
                )
                reporter.export_deployment_csv(
                    run.deployment_rankings
                )
                reporter.export_failure_cluster_csv(
                    run.failure_clusters
                )

                logger.info(
                    "CSV exports written to %s",
                    config.report_dir,
                )

            if config.export_json:
                logger.info(
                    "JSON export requested but not yet implemented."
                )

        self._cb("post_processing", 3, 4)
        self._cb("post_processing", 4, 4)
        logger.info("Post-processing complete.")
