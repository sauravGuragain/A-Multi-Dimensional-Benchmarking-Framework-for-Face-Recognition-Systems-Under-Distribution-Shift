#!/usr/bin/env python3
"""Read-only integrity check of ONE FaceEval-X Fold 0 verification run directory. It writes nothing.

Usage:
    python -I verify_run_npz.py RUN_DIR --expect-models a,b --expect-per-model 96 [--partial-grid | --test-mode]

    --expect-models a,b    the set of deep models in the run must be exactly this (e.g. facenet,arcface)
    --expect-per-model N   every deep model must have exactly N result rows; production requires 96 = 16 x 6
                           Both flags are REQUIRED unless --test-mode is given.
    --partial-grid         reduced real-model smoke run only (needs --expect-per-model below 96); still rejects any
                           condition outside the grid or duplicated, but does not require full coverage
    --test-mode            rehearsal only (a real benchmark run must NOT use this flag): tolerate STUB backbones and
                           a non-official dataset, make the two flags above optional, skip the grid / official-count /
                           Fold 0 constant checks, and follow symbolic links for the run directory, scores/ and the
                           top-level artifacts. A symlinked per-condition score file is rejected in EVERY mode.

Exit status:  0 = PASS (every required check ran and passed)
              1 = FAIL (at least one check failed; all failures are listed)
              2 = usage error
              3 = INCOMPLETE VERIFICATION (no failure, but a required check could not be run)
The success line is printed only with exit status 0.
"""
import csv
import hashlib
import json
import math
import os
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

STREAM_TYPES = {"plan", "row", "failure", "end", "abort"}
PAIR_KEYS = {"image_ids", "bal_pairs", "bal_labels", "full_pairs", "full_labels"}
SCORE_KEYS = {"valid", "s_bal", "s_full", "threshold"}
STR_FIELDS = ("model", "label", "perturbation", "paradigm", "backbone")
NUM_FIELDS = ("severity", "threshold", "accuracy", "tar", "far", "far_full", "auc", "auc_full",
              "val_balanced_accuracy")
INT_FIELDS = ("n_genuine", "n_impostor_balanced", "n_impostor_full", "n_test_images", "n_failed_embeddings",
              "val_n_failed_embeddings", "n_failed_pairs_balanced", "n_failed_pairs_full")
TOL = 1e-9
# Registered official LFW content fingerprint (faceeval.orchestration.verification.OFFICIAL_LFW_FINGERPRINTS).
# It is a SHA-256 over sorted (identity | image_id | image-file SHA-256) rows of the extracted dataset, not the
# archive hash (the archive hash 055f7d9c... is only recorded in a code comment, never in run output).
OFFICIAL_LFW_FINGERPRINT = "df3e734c6e5033224011c2587fd2c7671ce2f86d499cb57404f50f0d12337230"
OFFICIAL_LFW_IDENTITIES, OFFICIAL_LFW_IMAGES = 5749, 13233
# Authoritative benchmark grid: configs/lfw_fold0_verification_full.yaml (perturbation_schedule), identical to
# faceeval.core.types.PerturbationType and faceeval.analysis.fold0_results.PERTURBATION_ORDER / SEVERITIES.
GRID_TYPES = ("gaussian_blur", "motion_blur", "gaussian_noise", "salt_pepper_noise", "speckle_noise", "brightness",
              "contrast", "gamma", "rotation", "scaling", "cropping", "random_occlusion", "face_mask", "sunglasses",
              "jpeg_compression", "resolution_degradation")
GRID_SEVERITIES = (0.0, 0.2, 0.4, 0.6, 0.8, 1.0)
GRID = [(p, v) for p in GRID_TYPES for v in GRID_SEVERITIES]            # 16 x 6 = 96 conditions per model
# Authoritative Fold 0 partition sizes of the official LFW: faceeval.evaluation.split_manifest V1_IDENTITIES / V1_IMAGES.
V1_IDENTITIES = {"train": 4601, "validation": 574, "test": 574}
V1_IMAGES = {"train": 10636, "validation": 1259, "test": 1338}
PARTITIONS = ("train", "validation", "test")
MANIFEST_NAME = "split_manifest_fold0.tsv"
MANIFEST_VERSION = "faceeval-split-manifest/r4.1/1"
MANIFEST_META = {"dataset_name", "fold", "seed", "python_version", "numpy_version", "code_version",
                 "shuffled_order_sha256", "manifest_version"}
# Authoritative Fold 0 protocol values: configs/lfw_fold0_verification_full.yaml (random_seed: 42, dataset lfw) and
# faceeval.orchestration.verification (FOLD_INDEX = 0, K_FOLDS = 10).
FOLD, K_FOLDS, SEED, DATASET_NAME = 0, 10, 42, "lfw"
# Official pair sets: faceeval.evaluation.pairs  FOLD0_EXPECTED {partition_code: (G, K)} and FOLD0_DEFAULTS.
PAIR_ALGORITHM, PAIR_CAP, PAIR_RHO = "faceeval-pairs/r4.1/1", 10, 10
PAIR_PARTS = {"validation": 1, "test": 2}                                 # partition codes
PAIR_G_K = {"validation": (684, 6840), "test": (764, 7640)}              # genuine pairs G, impostor pool K (balanced = G)
PAIR_KEYS_REQUIRED = ("algorithm_version", "numpy_version", "seed", "cap", "rho", "partition_code", "n_images",
                      "n_genuine", "n_pool", "n_balanced")
SCORE_TOL = 1e-9          # scores are dot products of L2-normalised float64 embeddings (cosine), failed images score 0
# Every file the run writer creates in the run directory (verification.py: out_dir / ...). Nothing else is legitimate.
ROOT_FILES = frozenset({"verification_rows.jsonl", "verification_results.json", "verification_results.csv",
                        "split_manifest_fold0.tsv", "split_manifest_fold0.tsv.sha256",
                        "pairs_validation_fold0.tsv", "pairs_test_fold0.tsv"})
SIDECAR = re.compile(r"([0-9a-f]{64})  ([^\n\r]+)\n\Z")


class Ledger:
    """Counts what was checked. A required check that cannot run is SKIPPED, never silently passed."""

    def __init__(self):
        self.passed, self.fails, self.skips, self.notes = 0, [], [], []

    def check(self, cond, msg):
        if cond:
            self.passed += 1
        else:
            self.fails.append(msg)
        return bool(cond)

    def skip(self, msg):
        self.skips.append(msg)


def verdict(led):
    """(label, exit_code). PASS only if nothing failed AND nothing required was skipped."""
    if led.fails:
        return "FAIL", 1
    if led.skips or led.passed == 0:             # a run in which nothing was actually checked is not a pass
        return "INCOMPLETE VERIFICATION", 3
    return "PASS", 0


def isnum(x):
    return isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x)


def isint(x):
    return isinstance(x, int) and not isinstance(x, bool)


def auc(scores, labels):
    """Rank-based (Mann-Whitney) ROC AUC with average ranks for ties; needs both classes."""
    n_pos, n_neg = int(labels.sum()), int((~labels).sum())
    uniq, inv, counts = np.unique(scores, return_inverse=True, return_counts=True)
    avg_rank = np.cumsum(counts) - (counts - 1) / 2.0
    ranks = avg_rank[inv]
    return float((ranks[labels].sum() - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def close(a, b):
    return bool(np.isclose(float(a), float(b), rtol=0.0, atol=TOL))


def load_json(path, led, what):
    try:
        obj = json.loads(Path(path).read_bytes())
    except Exception as e:                                   # missing, unreadable, empty or malformed
        led.check(False, f"{what} cannot be read as JSON ({type(e).__name__}: {e})")
        return None
    return obj


def read_stream(path, led):
    raw = Path(path).read_bytes()
    led.check(raw.endswith(b"\n") and len(raw) > 0, "stream is empty or its last line is not newline-terminated (torn write)")
    recs = []
    lines = raw.split(b"\n")
    if raw.endswith(b"\n"):
        lines = lines[:-1]
    for i, line in enumerate(lines, 1):
        try:
            rec = json.loads(line)
        except Exception:
            led.check(False, f"stream line {i} is not valid JSON")
            continue
        if not (isinstance(rec, dict) and rec.get("type") in STREAM_TYPES):
            led.check(False, f"stream line {i} is not a record with a known type")
            continue
        recs.append(rec)
    return recs


def safe_name(name):
    return (isinstance(name, str) and name.endswith(".npz") and name not in ("", "test_pairs.npz")
            and "/" not in name and "\\" not in name and "\0" not in name and not name.startswith("."))


def load_npz(path, led, tag, expect_keys):
    """Returns {key: array} or None. The archive is always closed."""
    z = None
    try:
        z = np.load(path, allow_pickle=False)
        if not hasattr(z, "files"):
            led.check(False, f"{tag}: {path.name} is not an .npz archive")
            return None
        keys = set(z.files)
        if not led.check(keys == expect_keys, f"{tag}: {path.name} has arrays {sorted(keys)}, expected {sorted(expect_keys)}"):
            return None
        return {k: z[k] for k in keys}
    except Exception as e:
        led.check(False, f"{tag}: {path.name} is unreadable ({type(e).__name__}: {e})")
        return None
    finally:
        if z is not None and hasattr(z, "close"):
            z.close()


def check_pairs(P, led):
    """Validates test_pairs.npz; returns True if the arrays are usable for per-row checks."""
    ok = led.check(set(P) == PAIR_KEYS, "test_pairs.npz does not have exactly the arrays " + str(sorted(PAIR_KEYS)))
    if not ok:
        return False
    ids = P["image_ids"]
    ok &= led.check(ids.ndim == 1 and len(ids) > 0 and ids.dtype.kind == "U",
                    "test_pairs.npz: image_ids is empty, not 1-D or not a string array")
    ok &= led.check(all(x != "" and x == x.strip() for x in map(str, ids)), "test_pairs.npz: an image id is empty or has surrounding whitespace")
    ok &= led.check(len(set(map(str, ids))) == len(ids), "test_pairs.npz: image_ids are not unique")
    for nm in ("bal", "full"):
        pr, lb = P[f"{nm}_pairs"], P[f"{nm}_labels"]
        good = (pr.ndim == 2 and pr.shape[1] == 2 and len(pr) > 0 and np.issubdtype(pr.dtype, np.integer)
                and lb.ndim == 1 and lb.dtype == bool and len(lb) == len(pr))
        ok &= led.check(good, f"test_pairs.npz: {nm}_pairs/{nm}_labels are empty, mis-shaped or of the wrong dtype")
        if good:
            ok &= led.check(int(pr.min()) >= 0 and int(pr.max()) < len(ids), f"test_pairs.npz: {nm}_pairs index outside image_ids")
            ok &= led.check(bool(lb.any()) and bool((~lb).any()), f"test_pairs.npz: {nm}_labels has a single class (AUC undefined)")
    return bool(ok)


def check_row_fields(r, tag, led):
    ok = True
    for k in STR_FIELDS:
        ok &= led.check(isinstance(r.get(k), str), f"{tag}: field '{k}' missing or not a string")
    for k in NUM_FIELDS:
        ok &= led.check(isnum(r.get(k)), f"{tag}: field '{k}' missing, non-numeric or non-finite")
    for k in INT_FIELDS:
        good = led.check(isint(r.get(k)), f"{tag}: field '{k}' missing or not an integer")
        ok &= good
        if good:
            ok &= led.check(r[k] >= 0, f"{tag}: field '{k}' is negative ({r[k]})")
    for k in ("detector_miss_rate", "val_detector_miss_rate"):       # NaN when a model has no detector, else a rate
        v = r.get(k)
        ok &= led.check(isinstance(v, (int, float)) and not isinstance(v, bool) and (v != v or 0.0 <= v <= 1.0),
                        f"{tag}: field '{k}' is missing or not NaN / a rate in [0, 1]")
    return bool(ok)


def check_condition(r, tag, led):
    """The row's label must be the label of the condition its own perturbation/severity define.
    Rule (faceeval.core.types.PerturbationSpec.label): f"{perturbation}_s{severity:.2f}"; deep score files are
    f"{model}__{perturbation}__sev{severity:.2f}.npz" (verification.py). The plan holds labels only."""
    p, s, m = r.get("perturbation"), r.get("severity"), r.get("model")
    if not (isinstance(p, str) and isnum(s) and isinstance(m, str)):
        return                                  # field-type failures are reported by check_row_fields
    led.check(0.0 <= s <= 1.0, f"{tag}: severity {s} outside [0, 1]")
    led.check(r.get("label") == f"{p}_s{s:.2f}", f"{tag}: label {r.get('label')!r} does not match perturbation {p!r} / severity {s}")
    if r.get("paradigm") == "deep_learning":
        led.check(r.get("scores_file") == f"{m}__{p}__sev{s:.2f}.npz",
                  f"{tag}: scores_file {r.get('scores_file')!r} does not match model/perturbation/severity")


def unsafe_text(v):
    return v == "" or v != v.strip() or any(ord(c) < 32 or ord(c) == 127 for c in v)


def check_manifest(run, strict, summary, led):
    """Validates split_manifest_fold0.tsv (header line, then sorted 'partition|identity|image_id' rows) and its
    sidecar. Returns {'validation_images', 'test_ids'} or None when the manifest is missing or malformed (FAIL)."""
    mp, sp = run / MANIFEST_NAME, run / (MANIFEST_NAME + ".sha256")
    if strict and (mp.is_symlink() or sp.is_symlink()):
        led.check(False, "split manifest or its sidecar is a symbolic link")
        return None
    try:
        data, side = mp.read_bytes(), sp.read_bytes()
    except Exception as e:
        led.check(False, f"split manifest or its .sha256 sidecar cannot be read ({type(e).__name__})")
        return None
    sha = hashlib.sha256(data).hexdigest()
    try:
        m = SIDECAR.match(side.decode("utf-8"))
    except UnicodeDecodeError:
        m = None
    led.check(bool(m) and m.group(1) == sha and m.group(2) == MANIFEST_NAME, "manifest sidecar is malformed or does not match the manifest bytes")
    if isinstance(summary, dict):
        led.check(summary.get("manifest_sha256") == sha, "manifest SHA-256 differs from 'manifest_sha256' in the summary")
    try:
        text = data.decode("utf-8")
    except UnicodeDecodeError:
        led.check(False, "split manifest is not valid UTF-8")
        return None
    if not led.check(text.endswith("\n") and "\r" not in text, "split manifest is not LF-terminated, or contains CR"):
        return None
    lines = text.split("\n")[:-1]
    if not led.check(len(lines) >= 2, "split manifest has no header and rows"):
        return None
    meta, keys, ok = {}, [], True
    for f in lines[0].split("|"):
        k, eq, v = f.partition("=")
        if not (eq and k):
            ok &= led.check(False, f"manifest header has a malformed field {f!r}")
            continue
        ok &= led.check(k not in meta, f"manifest header repeats key {k!r}")
        meta[k] = v
        keys.append(k)
    ok &= led.check(set(meta) == MANIFEST_META, f"manifest header keys are {sorted(meta)}, expected {sorted(MANIFEST_META)}")
    ok &= led.check(meta.get("manifest_version") == MANIFEST_VERSION, f"manifest version is {meta.get('manifest_version')!r}")
    ok &= led.check(keys == sorted(keys), "manifest header keys are not in canonical (sorted) order")
    if strict:                                   # authoritative Fold 0 identity of the split
        for k, want in (("dataset_name", DATASET_NAME), ("fold", str(FOLD)), ("seed", str(SEED))):
            ok &= led.check(meta.get(k) == want, f"manifest {k} is {meta.get(k)!r}, the Fold 0 protocol requires {want!r}")
    if isinstance(summary, dict):                # manifest header and summary must describe the same split
        ok &= led.check(isint(summary.get("fold")) and str(summary["fold"]) == meta.get("fold")
                        and isint(summary.get("seed")) and str(summary["seed"]) == meta.get("seed"),
                        "manifest fold/seed differ from the summary's fold/seed")
        so = summary.get("shuffled_order")
        ok &= led.check(isinstance(so, dict) and so.get("recorded") == meta.get("shuffled_order_sha256"),
                        "summary shuffled_order.recorded differs from the manifest shuffled_order_sha256")
    rows, bad = [], []
    for ln in lines[1:]:
        f = ln.split("|")
        if len(f) == 3 and f[0] in PARTITIONS and not unsafe_text(f[1]) and not unsafe_text(f[2]):
            rows.append(tuple(f))
        else:
            bad.append(ln)
    if not led.check(not bad, f"{len(bad)} malformed manifest row(s) (need 3 fields, a valid partition and non-empty clean identity/image id), e.g. {bad[:3]!r}"):
        return None
    ok &= led.check(lines[1:] == sorted(lines[1:]), "manifest rows are not in canonical (sorted) order")
    img_all = [r[2] for r in rows]
    ok &= led.check(len(set(img_all)) == len(img_all), f"manifest image IDs are not unique ({len(img_all) - len(set(img_all))} repeated)")
    ids = {p: {r[1] for r in rows if r[0] == p} for p in PARTITIONS}
    imgs = {p: [r[2] for r in rows if r[0] == p] for p in PARTITIONS}
    for i, a in enumerate(PARTITIONS):
        for b in PARTITIONS[i + 1:]:
            ok &= led.check(not (ids[a] & ids[b]), f"identities appear in both {a} and {b}: {sorted(ids[a] & ids[b])[:3]}")
    ok &= led.check(all(imgs[p] for p in PARTITIONS), "a manifest partition is empty")
    parts = {p: sorted({(r[1], r[2]) for r in rows if r[0] == p}) for p in PARTITIONS}
    if isinstance(summary, dict) and set(meta) == MANIFEST_META:      # verification.split_content_sha256, recomputed
        body = [f"{k}={meta[k]}" for k in ("dataset_name", "fold", "seed", "shuffled_order_sha256", "manifest_version")]
        for pn in PARTITIONS:
            body += [f"{pn}|{i}|{im}" for i, im in parts[pn]]
        cs = summary.get("content_sha256")
        ok &= led.check(isinstance(cs, dict) and cs.get("split") == hashlib.sha256(("\n".join(body) + "\n").encode("utf-8")).hexdigest(),
                        "summary content_sha256.split differs from the value recomputed from the manifest")
    n_img, n_id = len(rows), len(set().union(*ids.values()))
    if isinstance(summary, dict) and isinstance(summary.get("dataset"), dict):
        ds = summary["dataset"]
        ok &= led.check(n_img == ds.get("n_images") and n_id == ds.get("n_identities"),
                        f"split manifest lists {n_img} images / {n_id} identities, summary dataset says {ds.get('n_images')} / {ds.get('n_identities')}")
    if strict:
        got_i, got_m = {p: len(ids[p]) for p in PARTITIONS}, {p: len(imgs[p]) for p in PARTITIONS}
        ok &= led.check(got_i == V1_IDENTITIES and got_m == V1_IMAGES,
                        f"manifest partition counts differ from the official Fold 0 protocol: identities {got_i} vs {V1_IDENTITIES}; images {got_m} vs {V1_IMAGES}")
    return {"validation_images": len(imgs["validation"]), "test_ids": set(imgs["test"]),
            "parts": {pn: set(v) for pn, v in parts.items()}}


def check_summary_protocol(data, strict, led):
    """Strict mode: the summary must state the authoritative Fold 0 run parameters."""
    for k, want in (("fold", FOLD), ("k_folds", K_FOLDS), ("seed", SEED)):
        led.check(isint(data.get(k)) and data[k] == want, f"summary {k} is {data.get(k)!r}, the Fold 0 protocol requires {want}")
    ns = data.get("noise_seeding")
    led.check(isinstance(ns, dict) and isint(ns.get("master_seed")) and ns["master_seed"] == SEED,
              f"summary noise_seeding.master_seed is not {SEED}")


def check_pair_manifests(run, data, man, strict, P, pairs_ok, led):
    """Pair manifests pairs_<part>_fold0.tsv: hash, header, row format, endpoints inside the manifest partition, counts,
    recomputed content hash, summary pair_params, and (test) pair-by-pair alignment with test_pairs.npz."""
    pp_all, shas, cshas = data.get("pair_params"), data.get("pair_manifest_sha256"), data.get("content_sha256")
    for part, code in PAIR_PARTS.items():
        tag = f"pairs_{part}_fold0.tsv"
        pp = pp_all.get(part) if isinstance(pp_all, dict) else None
        if not led.check(isinstance(pp, dict) and set(pp) == set(PAIR_KEYS_REQUIRED) and
                         all(isint(pp[k]) for k in PAIR_KEYS_REQUIRED if k not in ("algorithm_version", "numpy_version")) and
                         isinstance(pp.get("algorithm_version"), str) and isinstance(pp.get("numpy_version"), str),
                         f"summary pair_params[{part!r}] is missing or malformed"):
            continue
        led.check(pp["algorithm_version"] == PAIR_ALGORITHM and pp["partition_code"] == code and pp["n_balanced"] == pp["n_genuine"]
                  and isint(data.get("seed")) and pp["seed"] == data["seed"],
                  f"summary pair_params[{part!r}]: algorithm/partition_code/balanced size/seed are inconsistent")
        if man is not None:
            led.check(pp["n_images"] == len(man["parts"][part]), f"{part}: pair_params n_images {pp['n_images']} != {len(man['parts'][part])} manifest images")
        if strict:
            led.check(pp["seed"] == SEED and pp["cap"] == PAIR_CAP and pp["rho"] == PAIR_RHO, f"{part}: pair_params seed/cap/rho differ from the Fold 0 protocol")
            led.check((pp["n_genuine"], pp["n_pool"]) == PAIR_G_K[part], f"{part}: pair counts G={pp['n_genuine']}, K={pp['n_pool']}, official Fold 0 requires {PAIR_G_K[part]}")
        path = run / tag
        if strict and path.is_symlink():
            led.check(False, f"{tag} is a symbolic link")
            continue
        try:
            raw = path.read_bytes()
            text = raw.decode("utf-8")
        except Exception as e:
            led.check(False, f"{tag} cannot be read ({type(e).__name__})")
            continue
        led.check(isinstance(shas, dict) and shas.get(part) == hashlib.sha256(raw).hexdigest(), f"{tag}: SHA-256 differs from pair_manifest_sha256[{part!r}] in the summary")
        if not led.check(text.endswith("\n") and "\r" not in text, f"{tag} is not LF-terminated or contains CR"):
            continue
        lines = text.split("\n")[:-1]
        led.check(bool(lines) and lines[0] == "|".join(f"{k}={pp[k]}" for k in sorted(pp)), f"{tag}: header differs from the summary pair_params")
        body = lines[1:]
        led.check(isinstance(cshas, dict) and isinstance(cshas.get("pairs"), dict) and cshas["pairs"].get(part) == hashlib.sha256(
            ("|".join(f"{k}={pp[k]}" for k in sorted(pp) if k != "numpy_version") + "\n" + "".join(l + "\n" for l in body)).encode("utf-8")).hexdigest(),
            f"{tag}: summary content_sha256.pairs[{part!r}] differs from the recomputed value")
        members = man["parts"][part] if man is not None else None
        lookup = {f"{i}/{im}": (i, im) for i, im in members} if members is not None else None
        sizes = Counter(i for i, _ in members) if members is not None else None
        recs, bad, seen = [], [], set()
        for n, ln in enumerate(body):
            f = ln.split("|")
            good = len(f) == 8 and f[0] == f"{n:08d}" and f[1] in ("genuine", "impostor")
            if good:
                if lookup is not None:               # unambiguous: "<identity>/<image_id>" is looked up, never split
                    ra, rb = lookup.get(f[2]), lookup.get(f[3])
                else:
                    ra, rb = tuple(f[2].split("/", 1)), tuple(f[3].split("/", 1))
                good = ra is not None and rb is not None and len(ra) == 2 and len(rb) == 2
            if good:
                (ia, a), (ib, b) = ra, rb
                # faceeval.evaluation.pairs.manifest_rows: id_lo/id_hi are the identities of the lower/higher image index
                # (images are sorted by (identity, image_id)), so id_lo <= id_hi; genuine = same identity, impostor = different
                good = (a != b and f[4] <= f[5] and {ia, ib} == {f[4], f[5]} and ((f[1] == "genuine") == (ia == ib))
                        and ((f[6] == "genuine" and f[7] == "") if f[1] == "genuine" else (f[6] in ("pool", "pool+balanced") and f[7].isdigit()))
                        and frozenset((a, b)) not in seen)
                if good:
                    seen.add(frozenset((a, b)))
                    recs.append((f[1], a, b, f[6], ia, int(f[7]) if f[7] else None))
            if not good:
                bad.append(ln)
        if not led.check(not bad, f"{tag}: {len(bad)} malformed, out-of-partition, duplicate or inconsistent row(s), e.g. {bad[:2]!r}"):
            continue
        led.check([l.split("|", 1)[1] for l in body] == sorted(l.split("|", 1)[1] for l in body), f"{tag}: rows are not in canonical order")
        g = sum(1 for r in recs if r[0] == "genuine")
        k = sum(1 for r in recs if r[0] == "impostor")
        kb = sum(1 for r in recs if r[3] == "pool+balanced")
        led.check((g, k, kb) == (pp["n_genuine"], pp["n_pool"], pp["n_balanced"]), f"{tag}: genuine/pool/balanced row counts {(g, k, kb)} differ from pair_params")
        # rules of faceeval.evaluation.pairs.build_pairs / build_genuine, re-derived from the manifest alone
        led.check(pp["n_pool"] == pp["rho"] * pp["n_genuine"], f"{tag}: pair_params n_pool != rho * n_genuine")
        led.check(sorted(r[5] for r in recs if r[0] == "impostor") == list(range(k)), f"{tag}: impostor acceptance ranks are not exactly 0..K-1")
        if sizes is not None:
            per_id = Counter(r[4] for r in recs if r[0] == "genuine")
            exp = {i: (c * (c - 1) // 2 if c * (c - 1) // 2 <= pp["cap"] else pp["cap"]) for i, c in sizes.items() if c >= 2}
            led.check(per_id == Counter(exp), f"{tag}: genuine pairs per identity differ from the build_genuine rule (all pairs, or cap={pp['cap']})")
        if part == "test" and pairs_ok:          # index-to-image alignment: what was scored is exactly the frozen pair set
            ids = [str(x) for x in P["image_ids"]]
            def pset(pairs, labels):
                return Counter((frozenset((ids[i], ids[j])), bool(l)) for (i, j), l in zip(pairs.tolist(), labels.tolist()))
            want_full = Counter((frozenset((a, b)), lab == "genuine") for lab, a, b, m, _i, _r in recs)
            want_bal = Counter((frozenset((a, b)), lab == "genuine") for lab, a, b, m, _i, _r in recs if lab == "genuine" or m == "pool+balanced")
            led.check(pset(P["full_pairs"], P["full_labels"]) == want_full, "test_pairs.npz full pairs do not equal the frozen pair manifest (index-to-image misalignment?)")
            led.check(pset(P["bal_pairs"], P["bal_labels"]) == want_bal, "test_pairs.npz balanced pairs do not equal the frozen pair manifest (index-to-image misalignment?)")


def same_cell(v, s):
    """JSON value v against the CSV text s written by csv.DictWriter (str(value); None and missing -> '')."""
    if v is None:
        return s == ""
    if isinstance(v, float):
        try:
            f = float(s)
        except ValueError:
            return False
        return (v != v and f != f) or f == v
    return s == str(v)


def check_csv(run, results, strict, led):
    """verification_results.csv (read by the analysis code) must be exactly the JSON results, in the same order."""
    cp = run / "verification_results.csv"
    if strict and cp.is_symlink():
        led.check(False, "verification_results.csv is a symbolic link")
        return
    try:
        with open(cp, newline="", encoding="utf-8") as fh:
            table = list(csv.reader(fh))
    except Exception as e:
        led.check(False, f"verification_results.csv cannot be read ({type(e).__name__}: {e})")
        return
    fields = []
    for r in results:
        for k in (r if isinstance(r, dict) else {}):
            if k not in fields:
                fields.append(k)
    if not led.check(bool(table) and table[0] == fields, "verification_results.csv header differs from the columns of the JSON results"):
        return
    body = table[1:]
    if not led.check(len(body) == len(results), f"verification_results.csv has {len(body)} rows, the JSON has {len(results)}"):
        return
    bad = [i for i, (row, r) in enumerate(zip(body, results))
           if len(row) != len(fields) or not isinstance(r, dict) or not all(same_cell(r.get(k), c) for k, c in zip(fields, row))]
    led.check(not bad, f"verification_results.csv differs from the JSON results at row(s) {bad[:5]}")


def check_root(run, strict, led):
    """The run directory may hold exactly the files the run writer creates, plus scores/."""
    try:
        entries = sorted(os.scandir(run), key=lambda e: e.name)
    except Exception as e:
        led.check(False, f"run directory cannot be listed ({type(e).__name__})")
        return
    for e in entries:
        if e.name == ".DS_Store" and e.is_file(follow_symlinks=False):
            led.notes.append(".DS_Store (macOS Finder metadata) ignored in the run directory")
        elif e.name in ROOT_FILES:                # strict: a regular file; test mode: a link to one is tolerated
            led.check(e.is_file(follow_symlinks=not strict), f"{e.name} is not a regular file")
        elif e.name == "scores":
            led.check(e.is_dir(follow_symlinks=not strict), "scores is not a real directory")
        else:
            led.check(False, f"unexpected entry in the run directory: {e.name}")


def report_failure_rates(deep, nval, npairs, led):
    """Reporting only: per-model failed-embedding and detector-miss rates. No pass/fail threshold is applied."""
    for m in sorted({r.get("model") for r in deep}):
        rs = [r for r in deep if r.get("model") == m and isint(r.get("n_failed_embeddings")) and isint(r.get("n_test_images")) and r["n_test_images"] > 0]
        if not rs:
            continue
        fr = [r["n_failed_embeddings"] / r["n_test_images"] for r in rs]
        dm = [r["detector_miss_rate"] for r in rs if isinstance(r.get("detector_miss_rate"), (int, float)) and r["detector_miss_rate"] == r["detector_miss_rate"]]
        fp = [r["n_failed_pairs_full"] / npairs for r in rs if isint(r.get("n_failed_pairs_full")) and npairs]
        vf = rs[0].get("val_n_failed_embeddings")
        vm = rs[0].get("val_detector_miss_rate")
        led.notes.append(
            f"{m}: failed test embeddings per condition min/max = {min(fr):.4f}/{max(fr):.4f} of {rs[0]['n_test_images']} images; "
            f"failed full-pair fraction max = {max(fp):.4f}" if fp else f"{m}: failed test embeddings min/max = {min(fr):.4f}/{max(fr):.4f}")
        led.notes.append(
            f"{m}: detector miss rate on test conditions " + (f"min/max = {min(dm):.4f}/{max(dm):.4f}" if dm else "n/a (no detector)") +
            "; validation: failed embeddings " + (f"{vf}" + (f" of {nval} ({vf / nval:.4f})" if isint(vf) and nval else "")) +
            ", detector miss rate " + (f"{vm:.4f}" if isinstance(vm, (int, float)) and vm == vm else "n/a"))


def ckey(r):
    p, v = r.get("perturbation"), r.get("severity")
    return (p if isinstance(p, str) else repr(p), float(v) if isnum(v) else repr(v))


def check_grid(srows, pc, models, partial, led):
    """Strict mode: every expected deep model has exactly one row per authoritative (perturbation, severity) cell."""
    labels = [f"{p}_s{v:.2f}" for p, v in GRID]
    if isinstance(pc, list):
        if partial:
            led.check(len(set(pc)) == len(pc) and set(pc) <= set(labels), "plan has duplicate conditions or conditions outside the benchmark grid")
        else:
            led.check(sorted(pc) == sorted(labels), "plan conditions are not exactly the 16 x 6 benchmark grid")
    gset = set(GRID)
    for m in models:
        cnt = Counter(ckey(r) for r in srows if r.get("model") == m)
        led.check(not [k for k in cnt if k not in gset], f"{m}: rows outside the benchmark grid: {[k for k in cnt if k not in gset][:3]}")
        led.check(not [k for k, n in cnt.items() if n > 1], f"{m}: duplicate grid cells: {[k for k, n in cnt.items() if n > 1][:3]}")
        if not partial:
            miss = [k for k in GRID if k not in cnt]
            led.check(not miss, f"{m}: {len(miss)} benchmark-grid cell(s) missing, e.g. {miss[:3]}")


def check_score_file(r, tag, scores_dir, P, led):
    """Per-condition checks. Returns (s_bal, s_full) on success, else None."""
    name = r.get("scores_file")
    if not led.check(safe_name(name), f"{tag}: scores_file {name!r} is not a plain *.npz file name (path traversal?)"):
        return None
    path = scores_dir / name
    if not led.check(not path.is_symlink(), f"{tag}: {name} is a symbolic link"):
        return None
    if not led.check(path.is_file(), f"{tag}: score file {name} is missing (dangling reference)"):
        return None
    z = load_npz(path, led, tag, SCORE_KEYS)
    if z is None:
        return None
    valid, s_bal, s_full, thr = z["valid"], z["s_bal"], z["s_full"], z["threshold"]
    ids, bal, bl, full, fl = P["image_ids"], P["bal_pairs"], P["bal_labels"], P["full_pairs"], P["full_labels"]
    ok = led.check(valid.dtype == bool and valid.shape == (len(ids),), f"{tag}: 'valid' is not a bool array of length {len(ids)}")
    ok &= led.check(all(np.issubdtype(a.dtype, np.floating) for a in (s_bal, s_full)) and thr.shape == () and
                    np.issubdtype(thr.dtype, np.floating), f"{tag}: score arrays/threshold have the wrong dtype or shape")
    ok &= led.check(s_bal.shape == (len(bal),) and s_full.shape == (len(full),), f"{tag}: score array lengths do not match test_pairs.npz")
    if not ok:
        return None
    ok = led.check(bool(np.isfinite(s_bal).all() and np.isfinite(s_full).all() and np.isfinite(thr)), f"{tag}: non-finite scores or threshold")
    if not ok:
        return None
    led.check(float(np.abs(s_bal).max()) <= 1 + SCORE_TOL and float(np.abs(s_full).max()) <= 1 + SCORE_TOL,
              f"{tag}: a score lies outside the cosine-similarity range [-1, 1]")
    t = float(thr)
    led.check(close(t, r["threshold"]), f"{tag}: saved threshold {t} != row threshold {r['threshold']}")
    for sc, lb, sfx, acc_key, far_key, nimp in ((s_bal, bl, "", "accuracy", "far", "n_impostor_balanced"),
                                                 (s_full, fl, "_full", None, "far_full", "n_impostor_full")):
        pred = sc >= t
        led.check(close(pred[~lb].mean(), r[far_key]), f"{tag}: recomputed {far_key} differs from the row value")
        led.check(r[nimp] == int((~lb).sum()), f"{tag}: {nimp} {r[nimp]} != {int((~lb).sum())}")
        led.check(close(auc(sc, lb), r["auc" + sfx]), f"{tag}: recomputed auc{sfx} differs from the row value")
        if acc_key:
            led.check(close((pred == lb).mean(), r["accuracy"]), f"{tag}: recomputed accuracy differs from the row value")
            led.check(close(pred[lb].mean(), r["tar"]), f"{tag}: recomputed tar differs from the row value")
    led.check(r["n_genuine"] == int(bl.sum()), f"{tag}: n_genuine {r['n_genuine']} != {int(bl.sum())}")
    led.check(r["n_test_images"] == len(ids), f"{tag}: n_test_images {r['n_test_images']} != {len(ids)}")
    led.check(r["n_failed_embeddings"] == int((~valid).sum()), f"{tag}: n_failed_embeddings does not match the saved valid mask")
    for pr, sc, k in ((bal, s_bal, "n_failed_pairs_balanced"), (full, s_full, "n_failed_pairs_full")):
        touched = ~valid[pr[:, 0]] | ~valid[pr[:, 1]]
        led.check(r[k] == int(touched.sum()), f"{tag}: {k} {r[k]} != recomputed {int(touched.sum())}")
        led.check(bool((sc[touched] == 0).all()), f"{tag}: pairs containing a failed image are not scored 0")
    return s_bal, s_full


def run_checks(run, expect_per_model, expect_models, test_mode, led, partial=False):
    summary_p, rows_p, scores_dir = run / "verification_results.json", run / "verification_rows.jsonl", run / "scores"
    pairs_p = scores_dir / "test_pairs.npz"
    strict = not test_mode
    if strict and not led.check(not run.is_symlink(), f"run directory {run} is a symbolic link"):
        return                                   # nothing below may be read through a link
    scores_linked = strict and scores_dir.is_symlink()
    if scores_linked:
        led.check(False, "scores/ is a symbolic link")
    present = {}
    for p in (summary_p, rows_p, pairs_p):
        linked = (strict and p.is_symlink()) or (p is pairs_p and scores_linked)
        if linked and p is not pairs_p:
            led.check(False, f"{p.name} is a symbolic link")
        present[p] = False if linked else led.check(p.is_file(), f"required file is missing: {p}")
        if linked:
            led.skip(f"{p.name} not read because it is reached through a symbolic link")

    check_root(run, strict, led)

    # ---- stream
    recs = read_stream(rows_p, led) if present[rows_p] else []
    kinds = [r["type"] for r in recs]
    plan = recs[0] if recs and recs[0]["type"] == "plan" else None
    led.check(kinds.count("plan") == 1 and plan is not None, "stream must start with exactly one plan record")
    led.check("abort" not in kinds, "stream contains an abort record (the run was interrupted)")
    led.check("failure" not in kinds, f"stream contains {kinds.count('failure')} failure record(s)")
    end_ok = kinds.count("end") == 1 and bool(recs) and recs[-1]["type"] == "end" and recs[-1].get("status") == "COMPLETE"
    led.check(end_ok, "stream does not end with exactly one end record whose status is COMPLETE")
    srows = [r for r in recs if r["type"] == "row"]
    if plan is not None:
        pm, pc, pn = plan.get("models"), plan.get("conditions"), plan.get("n_rows_expected")
        plan_ok = (isinstance(pm, list) and isinstance(pc, list) and all(isinstance(x, str) for x in pm + pc)
                   and isint(pn) and pn == len(pm) * len(pc))
        led.check(plan_ok, "plan record is malformed or n_rows_expected != models x conditions")
        if plan_ok:
            led.check(len(set(pm)) == len(pm) and len(set(pc)) == len(pc) and bool(pm) and bool(pc) and all(x == x.strip() and x for x in pm + pc),
                      "plan declares a duplicate or empty model or condition")
        if plan_ok:
            led.check(len(srows) == pn, f"stream has {len(srows)} row records, the plan expects {pn}")
        else:
            pm = pc = pn = None
            led.skip("plan-based completeness checks skipped: the plan record is malformed")
    else:
        pm = pc = pn = None
        led.skip("plan-based completeness checks skipped: no usable plan record")
    if end_ok:
        led.check(recs[-1].get("n_rows") == len(srows) and recs[-1].get("n_rows_expected") == pn, "end record counts do not match the stream")

    # ---- summary
    data = load_json(summary_p, led, "verification_results.json") if present[summary_p] else None
    results = []
    if isinstance(data, dict):
        led.check(data.get("status") == "COMPLETE", f"summary status is {data.get('status')!r}, not COMPLETE")
        led.check(data.get("protocol") == "lfw_fold0_verification", "summary protocol is not lfw_fold0_verification")
        led.check(data.get("failures") == [], "summary lists failures")
        led.check(isinstance(data.get("results"), list), "summary has no results list")
        results = data["results"] if isinstance(data.get("results"), list) else []
        led.check(data.get("n_rows") == len(results) == data.get("n_rows_expected") == pn,
                  "summary n_rows / results / n_rows_expected / plan disagree")
        if test_mode:
            led.notes.append("TEST MODE: stub backbones and a non-official dataset are tolerated")
        else:
            led.check(data.get("stub_backbones") == [], f"stub backbones were used: {data.get('stub_backbones')}")
            led.check(data.get("full_lfw") is True, "summary does not say the dataset is the verified full official LFW")
            ds = data.get("dataset")
            if led.check(isinstance(ds, dict), "summary has no 'dataset' object"):
                led.check(ds.get("fingerprint_sha256") == OFFICIAL_LFW_FINGERPRINT,
                          f"dataset fingerprint {ds.get('fingerprint_sha256')!r} is not the registered official LFW fingerprint")
                led.check(ds.get("scope") == "official_lfw" and ds.get("counts_match_official_lfw") is True,
                          "dataset scope/counts_match_official_lfw do not say official LFW")
                led.check(ds.get("n_identities") == OFFICIAL_LFW_IDENTITIES and ds.get("n_images") == OFFICIAL_LFW_IMAGES,
                          f"dataset has {ds.get('n_identities')} identities / {ds.get('n_images')} images, "
                          f"expected {OFFICIAL_LFW_IDENTITIES} / {OFFICIAL_LFW_IMAGES}")
            check_summary_protocol(data, strict, led)
        strip = lambda rec: {k: v for k, v in rec.items() if k != "type"}      # stream records carry a "type" tag
        led.check(len(srows) == len(results) and all(json.dumps(strip(a), sort_keys=True) == json.dumps(b, sort_keys=True)
                                                      for a, b in zip(srows, results)),
                  "rows in the stream and in verification_results.json are not identical")
        if isinstance(data.get("results"), list):
            check_csv(run, results, strict, led)
        if data.get("synthetic_fixture"):
            led.notes.append("SYNTHETIC FIXTURE: this directory was produced by generate_synthetic_fixture.py, not by a benchmark run")
        so = data.get("shuffled_order")
        if isinstance(so, dict):
            led.notes.append(f"shuffled-order digest status vs the V25 reference: {so.get('status')!r} (reported, not enforced)")
    elif present[summary_p]:
        if data is not None:
            led.check(False, "verification_results.json is valid JSON but not a JSON object")
        led.skip("summary cross-checks skipped: verification_results.json is unusable")

    # ---- split manifest (identity-disjoint partitions, counts, sidecar); required in every mode
    man = check_manifest(run, strict, data if isinstance(data, dict) else None, led)

    # ---- rows against the plan
    dup = [k for k, n in Counter((r.get("model"), r.get("label")) for r in srows).items() if n > 1]
    led.check(not dup, f"duplicate (model, label) rows: {dup[:3]}")
    by_model = defaultdict(list)
    for r in srows:
        by_model[r.get("model")].append(r)
    if pm is not None and isinstance(pc, list):
        for m in pm:
            led.check({r.get("label") for r in by_model.get(m, [])} == set(pc), f"model {m!r}: its rows do not cover exactly the planned conditions")
        led.check(set(by_model) <= set(pm), "rows exist for a model that is not in the plan")
    for r in srows:
        check_condition(r, f"{r.get('model')}/{r.get('label')}", led)
    deep_set = {r.get("model") for r in srows if r.get("paradigm") == "deep_learning"}
    if pm is not None:
        led.check(deep_set <= set(pm), "deep-learning rows exist for a model that is not declared in the plan")
        if expect_models is not None:
            led.check(set(expect_models) <= set(pm), f"expected model(s) {sorted(set(expect_models) - set(pm))} are not declared in the plan")
        other = {m: len(by_model.get(m, [])) for m in pm if m not in deep_set}
        if other:
            led.notes.append(f"plan models without deep-learning rows (structure checked, metrics NOT verified): {other}")
    deep = [r for r in srows if r.get("paradigm") == "deep_learning"]
    led.check(bool(deep), "no deep-learning rows found: nothing to verify")
    deep_models = sorted({r.get("model") for r in deep})
    led.notes.append("deep rows per model: " + str({m: sum(1 for r in deep if r.get("model") == m) for m in deep_models}))
    if expect_models is not None:
        led.check(set(deep_models) == set(expect_models), f"deep models are {deep_models}, expected {sorted(expect_models)}")
    if strict and expect_models is not None:
        check_grid(srows, pc, expect_models, partial, led)
        if partial:
            led.notes.append("PARTIAL GRID: reduced smoke run; this is NOT a production verification")
    if expect_per_model is not None:
        for m in deep_models:
            led.check(sum(1 for r in deep if r.get("model") == m) == expect_per_model, f"{m}: row count != {expect_per_model}")
    for r in srows:
        if r.get("paradigm") != "deep_learning":
            led.check("scores_file" not in r, f"{r.get('model')}/{r.get('label')}: a non-deep row names a score file")
    led.check(len({r.get("scores_file") for r in deep}) == len(deep), "two rows name the same score file")
    if not test_mode:
        led.check(all(r.get("backbone") == "real" for r in deep), "a deep row does not use a real backbone")

    # ---- pairs and per-condition score files
    P = None
    if present[pairs_p]:
        P = load_npz(pairs_p, led, "test_pairs", PAIR_KEYS)
    pairs_ok = P is not None and check_pairs(P, led)
    clean, per_model_const = defaultdict(list), defaultdict(lambda: defaultdict(set))
    allgot = defaultdict(list)
    if pairs_ok:
        for r in deep:
            tag = f"{r.get('model')}/{r.get('label')}"
            try:
                if not check_row_fields(r, tag, led):
                    led.skip(f"{tag}: score checks skipped because its row fields are invalid")
                    continue
                got = check_score_file(r, tag, scores_dir, P, led)
                if got is None:
                    led.skip(f"{tag}: score-file consistency checks could not be completed")
                    continue
                for k in ("threshold", "val_balanced_accuracy", "n_test_images", "val_n_failed_embeddings", "val_detector_miss_rate"):
                    per_model_const[r["model"]][k].add("nan" if r[k] != r[k] else r[k])
                allgot[r["model"]].append((r["severity"], got))
                if r["severity"] == 0.0:
                    clean[r["model"]].append((tag, got, r))
            except Exception as e:                           # a checker bug must never turn into a pass
                led.check(False, f"{tag}: unexpected error while checking ({type(e).__name__}: {e})")
    elif deep:
        led.skip("per-condition score checks skipped: test_pairs.npz is missing or unusable")
    if pairs_ok and man is not None:             # the scored images are exactly the validated test partition
        got, want = set(map(str, P["image_ids"])), man["test_ids"]
        led.check(got == want, f"test_pairs.npz image ids are not the manifest test partition: {len(got - want)} not in the manifest, {len(want - got)} missing")
        if strict:
            led.check(len(got) == V1_IMAGES["test"] == len(P["image_ids"]), f"test_pairs.npz has {len(P['image_ids'])} images, official Fold 0 requires {V1_IMAGES['test']}")
    if isinstance(data, dict):
        check_pair_manifests(run, data, man, strict, P, pairs_ok, led)
    for m, consts in per_model_const.items():
        for k, vals in consts.items():
            led.check(len(vals) == 1, f"{m}: '{k}' is not constant across conditions ({sorted(vals)[:3]})")
    if man is not None:                          # validation failures can never exceed the validation image count
        nval = man["validation_images"]
        for m, consts in per_model_const.items():
            led.check(all(isinstance(v, int) and 0 <= v <= nval for v in consts["val_n_failed_embeddings"]),
                      f"{m}: val_n_failed_embeddings exceeds the {nval} validation images in the split manifest")
    for m, items in allgot.items():                # informational only: identical scores do not prove a perturbation was skipped
        base = next((g for sv, g in items if sv == 0.0), None)
        if base is not None:
            same = sum(1 for sv, g in items if sv != 0.0 and np.array_equal(g[0], base[0]) and np.array_equal(g[1], base[1]))
            if same:
                led.notes.append(f"{m}: {same} of {sum(1 for sv, _ in items if sv != 0.0)} non-clean conditions have scores identical to the clean baseline "
                                 "(informational; the run does not preserve perturbed images or per-condition perturbation parameters)")
    if pairs_ok and deep:
        report_failure_rates(deep, man["validation_images"] if man is not None else None, len(P["full_pairs"]), led)
    for m, items in clean.items():
        if len(items) < 2:
            led.notes.append(f"{m}: only {len(items)} severity-0.0 row, baseline-identity check not applicable")
            continue
        base_tag, (sb, sf), base = items[0]
        for tag, (b, f), row in items[1:]:
            led.check(np.array_equal(b, sb) and np.array_equal(f, sf) and close(row["accuracy"], base["accuracy"]),
                      f"{m}: severity-0.0 result differs between {base_tag} and {tag} (the clean baseline must be identical)")

    # ---- directory hygiene (the run directory itself is checked against the writer's allow-list by check_root)
    if scores_dir.is_dir() and not scores_linked:
        listed = {r.get("scores_file") for r in deep} | {"test_pairs.npz"}
        for p in sorted(scores_dir.iterdir(), key=lambda q: q.name):
            if p.name == ".DS_Store" and p.is_file() and not p.is_symlink():
                led.notes.append(".DS_Store (macOS Finder metadata) ignored in scores/")
                continue
            led.check(p.name in listed and p.is_file() and not p.is_symlink(), f"unexpected or orphan entry in scores/: {p.name}")


def main(argv):
    args, flags = [], {}
    it = iter(argv[1:])
    try:
        for a in it:
            if a in ("--expect-per-model", "--expect-models"):
                flags[a] = next(it)
            elif a in ("--test-mode", "--partial-grid"):
                flags[a] = True
            elif a.startswith("--"):
                raise ValueError(f"unknown option {a}")
            else:
                args.append(a)
        if len(args) != 1:
            raise ValueError("exactly one RUN_DIR is required")
        run = Path(args[0])
        if not run.is_dir():
            raise ValueError(f"{run} is not a directory")
        expect_n = int(flags["--expect-per-model"]) if "--expect-per-model" in flags else None
        expect_m = [m for m in flags["--expect-models"].split(",") if m] if "--expect-models" in flags else None
        if "--expect-models" in flags and not expect_m:
            raise ValueError("--expect-models needs at least one model name")
        if expect_n is not None and expect_n < 1:
            raise ValueError("--expect-per-model must be a positive integer")
        if not flags.get("--test-mode") and (expect_m is None or expect_n is None):
            raise ValueError("without --test-mode, both --expect-models and --expect-per-model are required")
        if flags.get("--partial-grid") and flags.get("--test-mode"):
            raise ValueError("--partial-grid cannot be combined with --test-mode")
        if not flags.get("--test-mode"):
            if flags.get("--partial-grid") and expect_n >= len(GRID):
                raise ValueError(f"--partial-grid is for reduced runs: --expect-per-model must be below {len(GRID)}")
            if not flags.get("--partial-grid") and expect_n != len(GRID):
                raise ValueError(f"a production run has {len(GRID)} conditions per model: use --expect-per-model {len(GRID)} "
                                 "(add --partial-grid only for a reduced real-model smoke run)")
    except (ValueError, StopIteration) as e:
        print(f"usage error: {e}\n{__doc__}")
        return 2
    led = Ledger()
    try:
        run_checks(run, expect_n, expect_m, bool(flags.get("--test-mode")), led, bool(flags.get("--partial-grid")))
    except Exception as e:                                   # never crash into a pass
        led.check(False, f"checker aborted unexpectedly ({type(e).__name__}: {e})")
    for n in led.notes:
        print("note:   ", n)
    for s in led.skips:
        print("SKIPPED:", s)
    for f in led.fails:
        print("FAIL:   ", f)
    label, code = verdict(led)
    print(f"summary: {led.passed} checks passed, {len(led.fails)} failed, {len(led.skips)} skipped")
    if code == 0:
        print("RESULT: ALL CHECKS PASSED")
    elif code == 1:
        print(f"RESULT: FAIL ({len(led.fails)} check(s) failed)")
    else:
        print(f"RESULT: {label} ({len(led.skips)} required check(s) skipped, no failures)")
    return code


if __name__ == "__main__":
    sys.exit(main(sys.argv))
