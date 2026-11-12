"""LFW Fold 0 verification stage for the orchestration runner.

Wires the Revision 4.1 split manifest (``faceeval.evaluation.split_manifest``) and the
deterministic pair builder (``faceeval.evaluation.pairs``) into the experiment run:

1. Fold 0 = ``DatasetSplitter(seed).all_k_folds(records, k=10)[0]`` (identity-disjoint).
2. The split is frozen as a manifest (written with a ``.sha256`` sidecar) and validated.
3. Validation / test pair sets are built (genuine, impostor pool, balanced subset) and
   written as pair manifests.
4. For every model the embedding extractor is fitted on the *train* partition only, a
   decision threshold is chosen on *clean validation* pairs only, and the *test* pairs are
   scored under every perturbation spec with that frozen threshold.

Results go to ``<output_dir>/verification/<run_id>/`` and are kept separate from the
closed-set identification results.
"""
from __future__ import annotations

import csv
import hashlib
import json
import logging
import os
import platform
import random
import subprocess
from pathlib import Path
from typing import Any, Callable

import numpy as np

from faceeval.core.exceptions import EmbeddingError, OrchestratorError
from faceeval.evaluation import pairs as pairs_mod
from faceeval.evaluation import split_manifest as sm
from faceeval.evaluation.metrics import compute_verification_metrics
from faceeval.orchestration.perturbed_inputs import needs_chips, perturb_batch
from faceeval.perturbation import noise as _noise

logger = logging.getLogger(__name__)

FOLD_INDEX = 0
K_FOLDS = 10
FULL_LFW_IDENTITIES = 5749
FULL_LFW_IMAGES = 13233
# Content fingerprints (see ``dataset_fingerprint``) of verified official LFW copies.
# This fingerprint was independently reproduced from a fresh extraction of the original
# lfw.tgz archive. The archive SHA-256 is the published scikit-learn LFW checksum:
# 055f7d9c632d7370e6fb4afc7468d40f970c34a80d4c6f50ffec63f5a8d536c0
OFFICIAL_LFW_FINGERPRINTS: frozenset[str] = frozenset({
    "df3e734c6e5033224011c2587fd2c7671ce2f86d499cb57404f50f0d12337230",
})
PARTITION_CODE = {"validation": 1, "test": 2}
FAR_POINTS = [0.001, 0.01, 0.1]


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


# SHA-256 of the reviewed per-image noise-seeding patch (per_image_noise_seeding.patch). It identifies the
# implementation while that patch is still uncommitted; once committed, ``git_commit`` in the run metadata does.
NOISE_SEEDING_PATCH_SHA256 = "881d30a130bbbc5993a0fcb3551725860ad1ebf03bd378d1515f89157e597d7e"


def noise_seeding_metadata() -> dict[str, Any]:
    """Provenance of the noise seeding used by this run (recorded in the verification summary)."""
    src = Path(_noise.__file__).read_bytes()
    return {
        "master_seed": _noise.NOISE_MASTER_SEED,
        "scheme": "sha256(master|image_id|perturbation_type)",
        "image_id_normalization": "backslash -> forward slash; then UTF-8 encoded; no other change",
        "digest_to_seed": "first 8 bytes of the SHA-256 digest, big-endian unsigned 64-bit integer, "
                          "passed to numpy.random.default_rng",
        "severity_in_seed": False,
        "id_source": "ImageRecord.image_id",
        "reviewed_patch_sha256": NOISE_SEEDING_PATCH_SHA256,
        "noise_module_sha256": hashlib.sha256(src).hexdigest(),
    }


def shuffled_order_sha256(records, seed: int) -> str:
    """SHA-256 of the seed-shuffled sorted subject IDs (same shuffle the splitter performs)."""
    subjects = sorted({r.subject_id for r in records})
    random.Random(seed).shuffle(subjects)
    return hashlib.sha256("\n".join(subjects).encode("utf-8")).hexdigest()


def _code_version() -> str:
    try:
        out = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True,
                             timeout=10, check=True)
        return out.stdout.strip() or sm.NOT_RECORDED
    except Exception:
        return sm.NOT_RECORDED


def dataset_fingerprint(records) -> str:
    """SHA-256 over the sorted (identity, image_id, file-content SHA-256) rows of the dataset used.

    Raises ``OrchestratorError`` if any image file is missing, so a damaged or partial copy is
    caught before any model runs. Reads each image file once.
    """
    rows, missing = [], []
    for r in records:
        if not os.path.isfile(r.image_path):
            missing.append(r.image_path)
            continue
        h = hashlib.sha256()
        with open(r.image_path, "rb") as fh:
            for block in iter(lambda: fh.read(1 << 20), b""):
                h.update(block)
        rows.append(f"{r.subject_id}|{r.image_id}|{h.hexdigest()}")
    if missing:
        raise OrchestratorError(f"{len(missing)} dataset image file(s) missing, e.g. {missing[:3]}")
    return hashlib.sha256(("\n".join(sorted(rows)) + "\n").encode("utf-8")).hexdigest()


def split_content_sha256(manifest) -> str:
    """Hash of the split itself (partition rows + non-environment metadata).

    Unlike ``manifest_sha256`` it excludes python/numpy/code versions, so it is identical for the
    same dataset and seed on any machine or NumPy version."""
    keep = ("dataset_name", "fold", "seed", "shuffled_order_sha256", "manifest_version")
    lines = [f"{k}={manifest.metadata[k]}" for k in keep]
    for part in sm.PARTITIONS:
        lines += [f"{part}|{ident}|{imid}" for ident, imid in manifest.partitions[part]]
    return hashlib.sha256(("\n".join(lines) + "\n").encode("utf-8")).hexdigest()


def pair_content_sha256(ps) -> str:
    """Hash of the pair set (parameters except ``numpy_version``, plus every canonical row)."""
    head = "|".join(f"{k}={ps.params[k]}" for k in sorted(ps.params) if k != "numpy_version")
    body = "\n".join(pairs_mod.manifest_rows(ps)) + "\n"
    return hashlib.sha256((head + "\n" + body).encode("utf-8")).hexdigest()


def _l2(emb: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(emb, axis=1, keepdims=True)
    return emb / np.where(n > 0, n, 1.0)


def _scores(emb: np.ndarray, idx_by_pair: list[tuple[int, int]]) -> np.ndarray:
    a = np.fromiter((p[0] for p in idx_by_pair), dtype=np.int64, count=len(idx_by_pair))
    b = np.fromiter((p[1] for p in idx_by_pair), dtype=np.int64, count=len(idx_by_pair))
    return np.einsum("ij,ij->i", emb[a], emb[b])


def best_accuracy_threshold(scores, labels) -> float:
    """Threshold ``t`` (accept iff ``score >= t``) that maximises accuracy on labelled scores.

    Every distinct outcome is considered: reject every pair (``t`` just above the maximum
    score), and accept the top-k pairs for each distinct score value, where tied scores are
    always accepted or rejected together (the last candidate, ``t == min(score)``, accepts
    every pair). Ties in accuracy resolve to the strictest (highest) threshold.

    Raises ``ValueError`` for empty input, mismatched lengths, non-1-D input, non-finite
    scores, or labels that are not boolean / 0-1.
    """
    s = np.asarray(scores)
    y = np.asarray(labels)
    if s.ndim != 1 or y.ndim != 1:
        raise ValueError("scores and labels must be 1-D")
    if s.size == 0:
        raise ValueError("scores and labels must not be empty")
    if s.shape != y.shape:
        raise ValueError(f"scores and labels differ in length: {s.shape[0]} vs {y.shape[0]}")
    s = s.astype(np.float64)
    if not np.isfinite(s).all():
        raise ValueError("scores must be finite")
    if y.dtype != np.bool_:
        if not np.isin(y, (0, 1)).all():
            raise ValueError("labels must be boolean or 0/1")
        y = y.astype(bool)

    n = s.size
    n_neg = int((~y).sum())
    order = np.argsort(-s, kind="mergesort")
    s_sorted = s[order]
    y_sorted = y[order].astype(np.int64)
    tp = np.cumsum(y_sorted)                       # genuine accepted among top-(i+1)
    fp = np.cumsum(1 - y_sorted)                   # impostors accepted among top-(i+1)
    acc_topk = (tp + (n_neg - fp)) / n
    last_of_tie = np.r_[s_sorted[1:] != s_sorted[:-1], True]   # cut only between distinct values
    cand_acc = np.r_[n_neg / n, acc_topk[last_of_tie]]          # first candidate: reject all
    cand_thr = np.r_[np.nextafter(s_sorted[0], np.inf), s_sorted[last_of_tie]]
    return float(cand_thr[int(np.argmax(cand_acc))])               # argmax: first max = strictest


def _accuracy_at(scores: np.ndarray, labels: np.ndarray, thr: float) -> dict[str, float]:
    pred = scores >= thr
    pos, neg = labels, ~labels
    tar = float(pred[pos].mean()) if pos.any() else float("nan")
    far = float(pred[neg].mean()) if neg.any() else float("nan")
    return {"accuracy": float((pred == labels).mean()), "tar": tar, "far": far}


def _flat_metrics(scores: np.ndarray, labels: np.ndarray, suffix: str) -> dict[str, Any]:
    m = compute_verification_metrics(scores.tolist(), labels.tolist(),
                                     far_operating_points=FAR_POINTS)
    out = {f"auc{suffix}": m["auc"], f"eer{suffix}": m["eer"]}
    for k, v in (m.get("frr_at_far") or {}).items():
        out[f"frr_at_far_{k}{suffix}"] = v
    return out


# ---------------------------------------------------------------------------
# main entry point
# ---------------------------------------------------------------------------

def run_fold0_verification(*, config, **kwargs) -> dict[str, Any]:
    """Run the Fold 0 verification protocol; return a summary dict (also written to disk).

    Any failure after the output directory exists leaves a ``verification_results.json`` with
    ``status: INCOMPLETE`` (written by the implementation when it has rows, here otherwise).
    """
    out_dir = Path(config.output_dir) / "verification" / config.run_id
    out_dir.mkdir(parents=True, exist_ok=False)
    try:
        return _run_fold0_verification(out_dir, config=config, **kwargs)
    except Exception as exc:
        path = out_dir / "verification_results.json"
        if not path.exists():
            with open(path, "w") as fh:
                json.dump({"protocol": "lfw_fold0_verification", "status": "INCOMPLETE",
                           "error": f"{type(exc).__name__}: {exc}", "n_rows": 0, "failures": [],
                           "results": []}, fh, indent=2)
        raise


def _run_fold0_verification(
    out_dir: Path,
    *,
    config,
    records: list,
    raw_images: dict | None,
    model_names: list[str],
    trad_names: set[str],
    composer,
    specs: list,
    dataset_name: str,
    preprocess: Callable,
    adapt_hyperparameters: Callable,
    preprocess_chips: Callable | None = None,
    get_model: Callable | None = None,
    progress: Callable[[str, int, int], None] | None = None,
) -> dict[str, Any]:
    """Run the Fold 0 verification protocol; return a summary dict (also written to disk)."""
    from faceeval.core.types import SplitStrategy
    from faceeval.data.splitter import DatasetSplitter
    from faceeval.core.registry import model_registry
    from faceeval.preprocessing.pipeline import pipeline_for_model

    progress = progress or (lambda stage, done, total: None)
    seed = config.random_seed

    # ---- 0. Validate the dataset actually used (the selected subset, not necessarily full LFW) ----
    ds_fingerprint = dataset_fingerprint(records)
    n_identities = len({r.subject_id for r in records})

    # ---- 1. Fold 0 split -------------------------------------------------
    splits = DatasetSplitter(seed=seed).all_k_folds(records, k=K_FOLDS, dataset_name=dataset_name)
    fold = splits[FOLD_INDEX]
    by_id = {r.image_id: r for r in records}
    parts_ids = {"train": fold.train_ids, "validation": fold.val_ids, "test": fold.test_ids}
    parts_recs = {p: [by_id[i] for i in ids] for p, ids in parts_ids.items()}
    partitions = {p: [(r.subject_id, r.image_id) for r in rs] for p, rs in parts_recs.items()}

    counts_match = (n_identities, len(records)) == (FULL_LFW_IDENTITIES, FULL_LFW_IMAGES)
    full_lfw = counts_match and ds_fingerprint in OFFICIAL_LFW_FINGERPRINTS
    if counts_match and not full_lfw:
        logger.warning("Dataset has the official LFW counts but its fingerprint %s is not a verified "
                       "official copy: treated as a custom dataset.", ds_fingerprint)
    expected_ids = expected_imgs = None
    if not full_lfw:
        logger.info("Dataset has %d identities / %d images (not a verified official full LFW; "
                    "min_images_per_subject=%s, max_subjects=%s): Fold 0 is built on this dataset and "
                    "the Revision 4.1 G/K acceptance counts are not applied.",
                    n_identities, len(records), config.min_images_per_subject, config.max_subjects)
        expected_ids = {p: len({i for i, _ in rows}) for p, rows in partitions.items()}
        expected_imgs = {p: len(rows) for p, rows in partitions.items()}

    # ---- 2. Manifest ----------------------------------------------------------
    metadata = {
        "dataset_name": dataset_name, "fold": str(FOLD_INDEX), "seed": str(seed),
        "python_version": platform.python_version(), "numpy_version": np.__version__,
        "code_version": _code_version(),
        "shuffled_order_sha256": shuffled_order_sha256(records, seed),
    }
    manifest = sm.build_manifest(partitions, metadata, expected_ids, expected_imgs)
    manifest_path = out_dir / "split_manifest_fold0.tsv"
    manifest_sha = sm.write_manifest(manifest, str(manifest_path))
    shuffled_cmp = sm.compare_shuffled_order(manifest)
    logger.info("Fold 0 manifest written (%s) sha256=%s; shuffled-order vs V25: %s",
                manifest_path, manifest_sha, shuffled_cmp["status"])

    # ---- 3. Pair sets -------------------------------------------------------------
    pair_sets: dict[str, pairs_mod.PairSet] = {}
    pair_shas: dict[str, str] = {}
    for part in ("validation", "test"):
        code = PARTITION_CODE[part]
        rows = list(manifest.partitions[part])
        ps = (pairs_mod.build_fold0_pairs(rows, code) if full_lfw
              else pairs_mod.build_pairs(rows, partition_code=code, **pairs_mod.FOLD0_DEFAULTS))
        pair_sets[part] = ps
        pair_shas[part] = pairs_mod.write_manifest(ps, str(out_dir / f"pairs_{part}_fold0.tsv"))
        logger.info("Pairs[%s]: G=%d K=%d balanced=%d sha256=%s", part,
                    len(ps.genuine), len(ps.pool), len(ps.balanced), pair_shas[part])

    def pair_arrays(ps):
        bal = ps.balanced_pairs()
        pool = ps.sorted_pool()
        gen = list(ps.genuine)
        return {
            "balanced": (gen + bal, np.r_[np.ones(len(gen), bool), np.zeros(len(bal), bool)]),
            "full": (gen + pool, np.r_[np.ones(len(gen), bool), np.zeros(len(pool), bool)]),
        }

    pa = {p: pair_arrays(ps) for p, ps in pair_sets.items()}
    # PairSet.images is the sorted (identity, image_id) table; index -> record
    img_recs = {p: [by_id[im] for _, im in ps.images] for p, ps in pair_sets.items()}

    # ---- 4. Per-model evaluation -----------------------------------------------------
    rows_out: list[dict[str, Any]] = []
    failures: list[dict[str, str]] = []
    total = len(model_names) * len(specs)
    done = 0
    fatal: OrchestratorError | None = None   # require_real_models violation: stop, but write results first
    progress("verification", 0, total)

    for model_name in model_names:
        paradigm = "traditional" if model_name in trad_names else "deep_learning"
        try:
            model = (get_model or model_registry.get)(model_name)
            pipeline = pipeline_for_model(model_name, skip_detection_for_pre_cropped=True)
            is_deep = paradigm == "deep_learning"
            if is_deep:
                X_train, y_train = [], [r.subject_id for r in parts_recs["train"]]
            else:
                X_train = preprocess(pipeline, parts_recs["train"], raw_images)
                y_train = [r.subject_id for r in parts_recs["train"]]
                adapt_hyperparameters(model, model_name, len(X_train))
            X_val = preprocess(pipeline, img_recs["validation"], raw_images)
            test_chips = None
            test_image_ids = [r.image_id for r in img_recs["test"]]   # parallel to X_test; seeds per-image noise
            if preprocess_chips is not None and needs_chips(pipeline.normalizer):
                # Perturb the uint8 chip and normalise afterwards (see perturbed_inputs).
                test_chips = preprocess_chips(pipeline, img_recs["test"], raw_images)
                X_test = [pipeline.normalizer.normalize(c) for c in test_chips]
            else:
                X_test = preprocess(pipeline, img_recs["test"], raw_images)

            model.fit_embedding(X_train, y_train) if not is_deep else model.fit_embedding([], y_train)
            backbone = "n/a"
            if is_deep:
                backbone = "STUB" if model.is_stub_backbone() else "real"
                if backbone == "STUB":
                    if config.require_real_models:
                        model.assert_real_backbone()          # raises EmbeddingError
                    logger.warning("[%s] running on a STUB backbone: results are meaningless.", model_name)

            miss = {"rate": float("nan")}

            def embed(arrs):
                bb = getattr(model, "_backbone", None)
                t0, m0 = getattr(bb, "n_total", 0), getattr(bb, "n_detector_miss", 0)
                out = _l2(np.asarray(model.extract_features(arrs), dtype=np.float64))
                dt = getattr(bb, "n_total", 0) - t0
                miss["rate"] = (getattr(bb, "n_detector_miss", 0) - m0) / dt if dt else float("nan")
                return out

            # threshold from CLEAN VALIDATION balanced pairs only
            v_pairs, v_lab = pa["validation"]["balanced"]
            v_scores = _scores(embed(X_val), v_pairs)
            val_miss = miss["rate"]
            if val_miss == val_miss and val_miss > 0.05:
                logger.warning("[%s] detector missed %.1f%% of clean validation images "
                               "(scored as zero embeddings / non-matches).", model_name, 100 * val_miss)
            thr = best_accuracy_threshold(v_scores, v_lab)
            emb_dim = int(embed(X_val[:1]).shape[1])
            logger.info("[%s] validation threshold=%.5f (balanced acc=%.4f)", model_name, thr,
                        _accuracy_at(v_scores, v_lab, thr)["accuracy"])
        except EmbeddingError as exc:
            failures.append({"protocol": "verification", "model": model_name, "stage": "setup",
                             "error": f"EmbeddingError: {exc}"})
            if config.require_real_models:
                fatal = OrchestratorError(f"Model '{model_name}' has no real backbone and "
                                          f"require_real_models=true: {exc}")
                break
            done += len(specs); progress("verification", done, total); continue
        except Exception as exc:
            logger.warning("Verification skipped for '%s': %s", model_name, exc)
            failures.append({"protocol": "verification", "model": model_name, "stage": "setup", "error": f"{type(exc).__name__}: {exc}"})
            if config.require_real_models and model_name not in trad_names:
                fatal = OrchestratorError(f"Model '{model_name}' unavailable and require_real_models=true: {exc}")
                fatal.__cause__ = exc
                break
            done += len(specs); progress("verification", done, total); continue

        for spec in specs:
            try:
                emb = embed(perturb_batch(composer, spec, pipeline.normalizer, X_test, test_chips, test_image_ids))
                det_miss = miss["rate"]
                row: dict[str, Any] = {
                    "model": model_name, "paradigm": paradigm, "backbone": backbone,
                    "perturbation": spec.perturbation_type.value, "severity": spec.severity,
                    "label": spec.label, "embedding_dim": emb_dim, "threshold": thr,
                    "detector_miss_rate": det_miss, "val_detector_miss_rate": val_miss,
                }
                t_pairs, t_lab = pa["test"]["balanced"]
                s_bal = _scores(emb, t_pairs)
                row.update(_accuracy_at(s_bal, t_lab, thr))
                row["n_genuine"] = int(t_lab.sum()); row["n_impostor_balanced"] = int((~t_lab).sum())
                row.update(_flat_metrics(s_bal, t_lab, ""))
                f_pairs, f_lab = pa["test"]["full"]
                s_full = _scores(emb, f_pairs)
                row["n_impostor_full"] = int((~f_lab).sum())
                row.update(_flat_metrics(s_full, f_lab, "_full"))
                far_full = _accuracy_at(s_full, f_lab, thr)
                row["far_full"] = far_full["far"]
                rows_out.append(row)
            except Exception as exc:
                logger.warning("Verification condition %s/%s failed: %s", model_name, spec.label, exc)
                failures.append({"protocol": "verification", "model": model_name, "stage": spec.label,
                                 "error": f"{type(exc).__name__}: {exc}"})
            done += 1
            progress("verification", done, total)

    # ---- 5. Write results ------------------------------------------------------
    fields: list[str] = []
    for r in rows_out:
        for k in r:
            if k not in fields:
                fields.append(k)
    with open(out_dir / "verification_results.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        w.writerows(rows_out)
    summary = {
        "protocol": "lfw_fold0_verification",
        "full_lfw": full_lfw,
        "status": "INCOMPLETE" if (failures or fatal or not rows_out) else "COMPLETE",
        "dataset": {
            "n_identities": n_identities, "n_images": len(records),
            "scope": "official_lfw" if full_lfw else "custom_or_subset",
            "counts_match_official_lfw": counts_match,
            "min_images_per_subject": config.min_images_per_subject,
            "max_subjects": config.max_subjects,
            "fingerprint_sha256": ds_fingerprint,
        },
        "content_sha256": {
            "split": split_content_sha256(manifest),
            "pairs": {p: pair_content_sha256(ps) for p, ps in pair_sets.items()},
        },
        "environment": {k: metadata[k] for k in ("python_version", "numpy_version", "code_version")},
        "fold": FOLD_INDEX, "k_folds": K_FOLDS, "seed": seed,
        "manifest_path": str(manifest_path), "manifest_sha256": manifest_sha,
        "shuffled_order": shuffled_cmp,
        "pair_manifest_sha256": pair_shas,
        "pair_params": {p: ps.params for p, ps in pair_sets.items()},
        "noise_seeding": noise_seeding_metadata(),
        "threshold_protocol": "max-accuracy on clean validation balanced pairs; frozen for test",
        "n_rows": len(rows_out), "failures": failures,
        "stub_backbones": sorted({r["model"] for r in rows_out if r["backbone"] == "STUB"}),
        "results": rows_out,
    }
    with open(out_dir / "verification_results.json", "w") as fh:
        json.dump(summary, fh, indent=2, default=str)
    summary["output_dir"] = str(out_dir)
    if fatal is not None:
        raise fatal
    if not rows_out:
        raise OrchestratorError(f"Fold 0 verification produced no results; failures: {failures[:3]}")
    logger.info("Verification complete: %d rows -> %s", len(rows_out), out_dir)
    return summary
