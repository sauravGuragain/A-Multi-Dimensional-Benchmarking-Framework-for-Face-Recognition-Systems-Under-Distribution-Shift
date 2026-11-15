#!/usr/bin/env python3
"""Regression harness for verify_run_npz.py (the read-only FaceEval-X Fold 0 run-directory integrity checker).

It generates SYNTHETIC fixtures with generate_synthetic_fixture.py, copies them for every scenario, corrupts the COPIES, runs
the checker on each copy and compares the exit code and the reported message with the expectation.  Nothing that a scenario
touches is a base fixture, a supplied real-pipeline stub run, or the checker itself; the harness hashes all of those before and
after and fails if any changed.  Synthetic success validates the checker's structural logic only; it does not validate real LFW
authenticity, real model execution or any scientific result.

Run:   python test_verify_run_npz_regressions.py --repo /path/to/FaceEval-X [--real-stubs DIR] [--keep]
       (or set FACEEVAL_REPO).  Exit status 0 only if every scenario behaved as expected.
--real-stubs DIR  optional; a directory holding two pipeline-produced stub runs, DIR/deep_only and DIR/mixed (run with the
                  real writer using stub backbones).  When absent those scenarios are reported as SKIPPED, never as passed.
Also usable from pytest:  FACEEVAL_REPO=... pytest test_verify_run_npz_regressions.py
"""
import argparse
import csv
import hashlib
import importlib.util
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
FLAGS = ("--expect-models", "facenet,arcface", "--expect-per-model", "96")


class Harness:
    def __init__(self, repo, checker, generator, real_stubs, keep):
        self.repo, self.checker, self.generator = str(repo), Path(checker), Path(generator)
        self.real_stubs = Path(real_stubs) if real_stubs else None
        self.keep = keep
        self.tmp = Path(tempfile.mkdtemp(prefix="verify_run_npz_tests_")).resolve()
        self.base, self.work = self.tmp / "base", self.tmp / "work"
        self.base.mkdir(); self.work.mkdir()
        self.n = self.bad = self.skipped = 0
        self.counter = 0
        self.failures, self.skips = [], []

    # ------------------------------------------------------------------ plumbing
    @staticmethod
    def tree_hash(p):
        h = hashlib.sha256()
        p = Path(p)
        if p.is_file():
            return hashlib.sha256(p.read_bytes()).hexdigest()
        for f in sorted(p.rglob("*")):
            h.update(str(f.relative_to(p)).encode())
            if f.is_symlink():
                h.update(b"->" + os.readlink(f).encode())
            elif f.is_file():
                h.update(f.read_bytes())
        return h.hexdigest()

    def gen(self, name, profile, *extra, out=None):
        out = out or (self.base / name)
        r = subprocess.run([sys.executable, str(self.generator), "--repo", self.repo, "--out", str(out), "--profile", profile, *extra],
                           capture_output=True, text=True)
        if r.returncode != 0:
            raise SystemExit(f"fixture generation failed for {name}: {r.stderr}")
        return out / "run"

    def run(self, d, *args):
        r = subprocess.run([sys.executable, "-I", str(self.checker), str(d), *args], capture_output=True, text=True)
        return r.returncode, r.stdout + r.stderr

    def expect(self, name, got, code, sub=None, absent=None):
        c, out = got
        self.n += 1
        ok = (c == code and (sub is None or sub in out) and (absent is None or absent not in out)
              and (("ALL CHECKS PASSED" in out) == (c == 0)) and "Traceback" not in out)
        if not ok:
            self.bad += 1
            self.failures.append(name)
        print(("ok       " if ok else "UNEXPECTED ") + f"{name}: exit {c} (expected {code})" + ("" if ok else f"; wanted {sub!r}\n{out[-1200:]}"))

    def skip(self, name, why):
        self.skipped += 1
        self.skips.append(name)
        print(f"SKIPPED  {name}: {why}")

    def copy(self, base_run, link_dir=False):
        """Disposable copy of a base run directory; returns (path_to_check, scenario_dir)."""
        self.counter += 1
        sc = self.work / f"s{self.counter:03d}"
        sc.mkdir()
        shutil.copytree(base_run, sc / "run", symlinks=True)
        return sc / "run", sc

    def scenario(self, name, base_run, mutate, sub, code=1, args=FLAGS, absent=None):
        d, sc = self.copy(base_run)
        try:
            mutate(d)
        except Exception as e:                                     # a harness bug must not look like a checker result
            self.n += 1; self.bad += 1; self.failures.append(name)
            print(f"UNEXPECTED {name}: harness error {e!r}")
            return
        self.expect(name, self.run(d, *args), code, sub, absent)
        shutil.rmtree(sc, ignore_errors=True)

    # ------------------------------------------------------------------ mutation helpers
    @staticmethod
    def load(d):
        return json.load(open(d / "verification_results.json")), [json.loads(l) for l in open(d / "verification_rows.jsonl")]

    @staticmethod
    def save(d, S, L):
        json.dump(S, open(d / "verification_results.json", "w"), indent=2, default=str)
        open(d / "verification_rows.jsonl", "w").write("".join(json.dumps(x) + "\n" for x in L))

    @staticmethod
    def write_csv(d, rows):
        fields = []
        for r in rows:
            for k in r:
                if k not in fields:
                    fields.append(k)
        with open(d / "verification_results.csv", "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=fields); w.writeheader(); w.writerows(rows)

    def edit_rows(self, d, fn, pred):
        S, L = self.load(d)
        for x in L:
            if x["type"] == "row" and pred(x): fn(x)
        for x in S["results"]:
            if pred(x): fn(x)
        self.save(d, S, L); self.write_csv(d, S["results"])

    def summ(self, d, fn):
        S, L = self.load(d); fn(S); self.save(d, S, L)

    def rewrite_manifest(self, d, edit):
        p = d / "split_manifest_fold0.tsv"; t = edit(p.read_text()); p.write_text(t)
        sha = hashlib.sha256(t.encode()).hexdigest()
        (d / "split_manifest_fold0.tsv.sha256").write_text(f"{sha}  split_manifest_fold0.tsv\n")
        self.summ(d, lambda S: S.__setitem__("manifest_sha256", sha))

    def set_tsv(self, d, part, edit):
        p = d / f"pairs_{part}_fold0.tsv"; t = edit(p.read_text()); p.write_text(t)
        self.summ(d, lambda S: S["pair_manifest_sha256"].__setitem__(part, hashlib.sha256(t.encode()).hexdigest()))

    @staticmethod
    def npz_edit(path, fn):
        with np.load(path) as z:
            a = {k: z[k] for k in z.files}
        fn(a); np.savez_compressed(path, **a)

    @staticmethod
    def link_out(d, name):
        """Move d/name outside the run directory and leave a symbolic link in its place."""
        out = d.parent / "outside"; out.mkdir(exist_ok=True)
        shutil.move(str(d / name), str(out / name))
        os.symlink(out / name, d / name)

    @staticmethod
    def tsv_rows(d, part):
        t = (d / f"pairs_{part}_fold0.tsv").read_text().split("\n")
        return t[0], [l.split("|") for l in t[1:-1]]

    # ------------------------------------------------------------------ sections
    def run_all(self):
        files = {"checker": self.checker, "generator": self.generator}
        P = self.gen("production", "production")
        SD = self.gen("stub-deep", "stub-deep")
        SM = self.gen("stub-mixed", "stub-mixed")
        SL = self.gen("slash", "production", "--slash-identity")
        bases = {"production": P.parent, "stub-deep": SD.parent, "stub-mixed": SM.parent, "slash": SL.parent}
        real = {}
        if self.real_stubs:
            real = {"deep": self.real_stubs / "deep_only", "mixed": self.real_stubs / "mixed"}
        before = {k: self.tree_hash(v) for k, v in {**files, **bases, **{"real-" + k: v for k, v in real.items()}}.items()}
        self.section_valid(P, SD, SM, SL, real)
        self.section_production(P)
        self.section_symlinks(P, SD)
        self.section_pairs(P, SD, SL)
        self.section_verdict()
        self.section_generator(P)
        self.section_legacy("generated stubs", {"deep": SD, "mixed": SM})
        if real:
            self.section_legacy("real-pipeline stubs", real)
        else:
            self.skip("legacy scenarios on real-pipeline stub runs", "--real-stubs not supplied")
        self.section_usage(SD)
        after = {k: self.tree_hash(v) for k, v in {**files, **bases, **{"real-" + k: v for k, v in real.items()}}.items()}
        unchanged = before == after
        self.n += 1
        if not unchanged:
            self.bad += 1; self.failures.append("immutability")
        print(("ok       " if unchanged else "UNEXPECTED ") + f"base fixtures, supplied stub runs, checker and generator unchanged by all scenarios "
              f"({len(before)} objects hashed before/after)")
        print(f"\nHARNESS RESULT: {self.n} scenarios, {self.n - self.bad} as expected, {self.bad} unexpected, {self.skipped} skipped")
        if self.failures:
            print("unexpected:", ", ".join(self.failures))
        if not self.keep:
            shutil.rmtree(self.tmp, ignore_errors=True)
        else:
            print("kept:", self.tmp)
        return self.bad == 0 and self.n > 0

    def section_valid(self, P, SD, SM, SL, real):
        self.expect("V1 valid synthetic production fixture (192 rows, 1,338 test images)", self.run(P, *FLAGS), 0, "detector miss rate")
        self.expect("V2 valid synthetic fixture with a '/' in an identity name", self.run(SL, *FLAGS), 0)
        self.expect("V3 synthetic stub-deep, --test-mode", self.run(SD, "--test-mode"), 0)
        self.expect("V4 synthetic stub-mixed (classical models in plan), --test-mode", self.run(SM, "--test-mode"), 0, "NOT verified")
        self.expect("V5 synthetic stub in strict mode fails (stub backbone, non-official dataset, 1,338 images)",
                    self.run(SD, "--expect-models", "facenet,arcface", "--expect-per-model", "4", "--partial-grid"), 1, "official Fold 0 requires 1338")
        self.expect("V6 synthetic fixture is announced as synthetic", self.run(P, *FLAGS), 0, "SYNTHETIC FIXTURE")
        for k, v in real.items():
            self.expect(f"V7 real-pipeline stub run '{k}', --test-mode", self.run(v, "--test-mode"), 0)
        if not real:
            self.skip("V7 real-pipeline stub runs in --test-mode", "--real-stubs not supplied")

    def section_production(self, P):
        S = self.scenario
        # 1 test coverage
        S("E1 one scored image is not in the manifest test partition", P, lambda d: self.npz_edit(d / "scores/test_pairs.npz", lambda z: z.update(
            image_ids=np.array(["tr_0000/tr_0000_0001.jpg"] + list(z["image_ids"][1:])))), "not the manifest test partition")
        S("E2 image_ids are not strings", P, lambda d: self.npz_edit(d / "scores/test_pairs.npz", lambda z: z.update(image_ids=np.arange(len(z["image_ids"])))), "not a string array")
        def perm(d):
            def f(z):
                a = z["image_ids"].copy(); a[[0, 1]] = a[[1, 0]]; z["image_ids"] = a
            self.npz_edit(d / "scores/test_pairs.npz", f)
        S("E3 image_ids permuted: index-to-image misalignment", P, perm, "misalignment", absent="not the manifest test partition")
        # 2 metadata
        for k, old, v in (("dataset_name", "lfw", "other"), ("fold", "0", "3"), ("seed", "42", "7")):
            S(f"M manifest {k}={v}", P, lambda d, k=k, old=old, v=v: self.rewrite_manifest(d, lambda t: t.replace(f"{k}={old}", f"{k}={v}", 1)), "protocol requires")
        S("S1 summary fold=1", P, lambda d: self.summ(d, lambda S_: S_.update(fold=1)), "summary fold is")
        S("S2 summary k_folds=5", P, lambda d: self.summ(d, lambda S_: S_.update(k_folds=5)), "summary k_folds is")
        S("S3 summary seed=7", P, lambda d: self.summ(d, lambda S_: S_.update(seed=7)), "summary seed is")
        S("S4 noise_seeding.master_seed=7", P, lambda d: self.summ(d, lambda S_: S_["noise_seeding"].update(master_seed=7)), "master_seed")
        S("S5 test pair_params seed=7", P, lambda d: self.summ(d, lambda S_: S_["pair_params"]["test"].update(seed=7)), "pair_params")
        S("S6 test pair_params n_genuine=700", P, lambda d: self.summ(d, lambda S_: S_["pair_params"]["test"].update(n_genuine=700, n_balanced=700)), "official Fold 0 requires")
        S("S7 validation pair_params n_pool=6000", P, lambda d: self.summ(d, lambda S_: S_["pair_params"]["validation"].update(n_pool=6000)), "official Fold 0 requires")
        S("S8 fingerprint altered", P, lambda d: self.summ(d, lambda S_: S_["dataset"].update(fingerprint_sha256="0" * 64)), "registered official LFW fingerprint")
        S("S9 full_lfw false", P, lambda d: self.summ(d, lambda S_: S_.update(full_lfw=False)), "full official LFW")
        S("S10 manifest partition counts wrong (one image moved)", P, lambda d: self.rewrite_manifest(d, lambda t: t.replace("\ntrain|tr_0000|", "\nvalidation|tr_0000|", 1)), "partition counts differ")
        S("S11 manifest missing", P, lambda d: (d / "split_manifest_fold0.tsv").unlink(), "cannot be read")
        S("S12 manifest identity in two partitions", P, lambda d: self.rewrite_manifest(d, lambda t: t.replace("\nvalidation|va_0001|", "\nvalidation|tr_0001|", 1)), "appear in both")
        S("S13 manifest malformed row", P, lambda d: self.rewrite_manifest(d, lambda t: t.replace("\ntrain|tr_0002|", "\ntrain|", 1)), "malformed manifest row")
        S("S14 manifest sidecar wrong", P, lambda d: (d / "split_manifest_fold0.tsv.sha256").write_text("0" * 64 + "  split_manifest_fold0.tsv\n"), "sidecar")
        # grid
        def drop(d):
            S_, L = self.load(d)
            i = [k for k, x in enumerate(L) if x["type"] == "row" and x["model"] == "arcface" and x["label"] == "gamma_s0.60"][0]
            del L[i]; S_["results"] = [r for r in S_["results"] if not (r["model"] == "arcface" and r["label"] == "gamma_s0.60")]
            L[-1]["n_rows"] = 191; S_["n_rows"] = 191; self.save(d, S_, L); self.write_csv(d, S_["results"])
        S("G1 one missing grid condition", P, drop, "row count")
        S("G2 plan lacks one grid condition", P, lambda d: (lambda S_, L: (L[0].__setitem__("conditions", L[0]["conditions"][:-1]), L[0].__setitem__("n_rows_expected", 190), self.save(d, S_, L)))(*self.load(d)), "plan")
        S("G3 production run needs --expect-per-model 96", P, lambda d: None, "96", code=2, args=("--expect-models", "facenet,arcface", "--expect-per-model", "4"))
        # csv
        def csv_edit(fn):
            def f(d):
                t = list(csv.reader(open(d / "verification_results.csv", newline=""))); fn(t)
                with open(d / "verification_results.csv", "w", newline="") as fh: csv.writer(fh).writerows(t)
            return f
        col = lambda t, k: t[0].index(k)
        S("C1 CSV accuracy altered", P, csv_edit(lambda t: t[5].__setitem__(col(t, "accuracy"), "0.999")), "differs from the JSON results at row")
        S("C2 CSV rows swapped", P, csv_edit(lambda t: t.__setitem__(slice(3, 5), [t[4], t[3]])), "differs from the JSON results at row")
        S("C3 CSV row deleted", P, csv_edit(lambda t: t.pop(7)), "has 191 rows")
        S("C4 CSV extra column", P, csv_edit(lambda t: [r.append("x") for r in t]), "header differs")
        S("C5 CSV NaN replaced by a number", P, csv_edit(lambda t: t[1].__setitem__(col(t, "detector_miss_rate"), "0.5")), "differs from the JSON results at row")
        S("C6 CSV NaN emptied", P, csv_edit(lambda t: t[1].__setitem__(col(t, "detector_miss_rate"), "")), "differs from the JSON results at row")
        S("C7 CSV missing", P, lambda d: (d / "verification_results.csv").unlink(), "verification_results.csv cannot be read")
        # filesystem allow-list
        S("R1 unexpected file in run directory", P, lambda d: (d / "notes.txt").write_text("x"), "unexpected entry in the run directory: notes.txt")
        S("R2 unexpected directory in run directory", P, lambda d: (d / "scores_old").mkdir(), "unexpected entry in the run directory: scores_old")
        S("R3 legitimate .DS_Store is ignored with a note", P, lambda d: ((d / ".DS_Store").write_bytes(b"x"), (d / "scores/.DS_Store").write_bytes(b"x")), ".DS_Store", code=0)
        # plan
        def plan(fn):
            return lambda d: (lambda S_, L: (fn(L[0]), self.save(d, S_, L)))(*self.load(d))
        S("P1 duplicate model in plan", P, plan(lambda p: p.update(models=["facenet", "facenet", "arcface"], n_rows_expected=288)), "duplicate or empty")
        S("P2 duplicate condition in plan", P, plan(lambda p: p["conditions"].__setitem__(1, p["conditions"][0])), "duplicate or empty")
        S("P3 deep model rows not declared in plan", P, plan(lambda p: p.update(models=["facenet"], n_rows_expected=96)), "not declared in the plan")
        S("P4 expected model absent from the run", P, lambda d: None, "expected", args=("--expect-models", "facenet,arcface,insightface", "--expect-per-model", "96"))
        # scores / detector rates / perturbation evidence
        S("SR1 a score above 1", P, lambda d: self.npz_edit(d / "scores/facenet__brightness__sev0.40.npz", lambda z: z["s_bal"].__setitem__(0, 1.5)), "cosine-similarity range")
        S("D1 detector_miss_rate 1.7", P, lambda d: self.edit_rows(d, lambda x: x.update(detector_miss_rate=1.7), lambda x: x["model"] == "arcface" and x["label"] == "gamma_s0.60"), "rate in [0, 1]")
        S("D2 val_detector_miss_rate not constant", P, lambda d: self.edit_rows(d, lambda x: x.update(val_detector_miss_rate=0.5), lambda x: x["model"] == "arcface" and x["label"] == "gamma_s0.60"), "not constant")
        def ident(d):
            sd = d / "scores"; shutil.copy(sd / "facenet__gaussian_blur__sev0.00.npz", sd / "facenet__gaussian_blur__sev1.00.npz")
            S_, _ = self.load(d); clean = [r for r in S_["results"] if r["model"] == "facenet" and r["label"] == "gaussian_blur_s0.00"][0]
            keep = ("model", "label", "perturbation", "severity", "scores_file")
            def f(x):
                for k, v in clean.items():
                    if k not in keep: x[k] = v
            self.edit_rows(d, f, lambda x: x["model"] == "facenet" and x["label"] == "gaussian_blur_s1.00")
        S("PN1 severity-1.0 scores identical to clean must NOT fail (informational note only)", P, ident, "identical to the clean baseline", code=0)

    def section_symlinks(self, P, SD):
        S = self.scenario
        T = ("--test-mode",)
        # intended --test-mode behaviour: layout symlinks tolerated; per-condition score-file symlinks rejected everywhere
        S("TS1 --test-mode tolerates a symlinked scores/", SD, lambda d: self.link_out(d, "scores"), None, code=0, args=T)
        S("TS2 --test-mode tolerates a symlinked CSV", SD, lambda d: self.link_out(d, "verification_results.csv"), None, code=0, args=T)
        S("TS3 --test-mode tolerates a symlinked pair TSV", SD, lambda d: self.link_out(d, "pairs_test_fold0.tsv"), None, code=0, args=T)
        S("TS4 --test-mode tolerates a symlinked manifest", SD, lambda d: self.link_out(d, "split_manifest_fold0.tsv"), None, code=0, args=T)
        S("TS5 --test-mode tolerates a symlinked summary", SD, lambda d: self.link_out(d, "verification_results.json"), None, code=0, args=T)
        S("TS6 --test-mode tolerates a symlinked stream", SD, lambda d: self.link_out(d, "verification_rows.jsonl"), None, code=0, args=T)
        def tsrun():
            d, sc = self.copy(SD)
            shutil.move(str(d), str(sc / "real")); os.symlink(sc / "real", sc / "linkrun")
            self.expect("TS7 --test-mode tolerates a symlinked run directory", self.run(sc / "linkrun", *T), 0)
        tsrun()
        def sf(d):
            f = sorted((d / "scores").glob("facenet__*.npz"))[0]
            outside = d.parent / "outside"; outside.mkdir(); shutil.copy(f, outside / f.name); f.unlink(); os.symlink(outside / f.name, f)
        S("TS8 --test-mode still rejects a symlinked per-condition score file", SD, sf, "symbolic link", args=T)
        # strict production mode keeps every symlink check
        S("SS1 strict rejects a symlinked scores/", P, lambda d: self.link_out(d, "scores"), "scores/ is a symbolic link")
        S("SS2 strict rejects a symlinked CSV", P, lambda d: self.link_out(d, "verification_results.csv"), "symbolic link")
        S("SS3 strict rejects a symlinked pair TSV", P, lambda d: self.link_out(d, "pairs_test_fold0.tsv"), "symbolic link")
        S("SS4 strict rejects a symlinked manifest", P, lambda d: self.link_out(d, "split_manifest_fold0.tsv"), "symbolic link")
        S("SS5 strict rejects a symlinked summary", P, lambda d: self.link_out(d, "verification_results.json"), "symbolic link")
        S("SS6 strict rejects a symlinked stream", P, lambda d: self.link_out(d, "verification_rows.jsonl"), "symbolic link")
        S("SS7 strict rejects a symlinked per-condition score file", P, sf, "symbolic link")
        d, sc = self.copy(P)
        shutil.move(str(d), str(sc / "real")); os.symlink(sc / "real", sc / "linkrun")
        self.expect("SS8 strict rejects a symlinked run directory", self.run(sc / "linkrun", *FLAGS), 1, "run directory")

    def section_pairs(self, P, SD, SL):
        S = self.scenario
        def first(d, part, label):
            _, rows = self.tsv_rows(d, part)
            return [r for r in rows if r[1] == label][0]
        def endpoint_out(part, own, other):
            def f(d):
                row = first(d, part, "genuine")
                self.set_tsv(d, part, lambda t: t.replace(row[2], other, 1))
            return f
        S("T1 pair TSV bytes altered, hash not updated", P, lambda d: (d / "pairs_test_fold0.tsv").write_text((d / "pairs_test_fold0.tsv").read_text().replace("rho=10", "rho=11", 1)), "SHA-256 differs")
        S("T2 pair TSV header altered, hash updated", P, lambda d: self.set_tsv(d, "test", lambda t: t.replace("cap=10", "cap=11", 1)), "header differs")
        S("T3 test pair endpoint outside its partition", P, endpoint_out("test", None, "tr_0000/tr_0000/tr_0000_0001.jpg"), "out-of-partition")
        S("T4 validation pair endpoint outside its partition", P, endpoint_out("validation", None, "te_0000/te_0000/te_0000_0001.jpg"), "out-of-partition")
        S("T5 pair TSV missing", P, lambda d: (d / "pairs_validation_fold0.tsv").unlink(), "pairs_validation_fold0.tsv cannot be read")
        S("T6 content_sha256.pairs altered", P, lambda d: self.summ(d, lambda S_: S_["content_sha256"]["pairs"].update(test="0" * 64)), "content_sha256.pairs")
        S("T7 content_sha256.split altered", P, lambda d: self.summ(d, lambda S_: S_["content_sha256"].update(split="0" * 64)), "content_sha256.split")
        # rules confirmed against faceeval.evaluation.pairs (build_pairs, build_genuine, manifest_rows)
        def dup_rank(d):
            def f(t):
                lines = t.split("\n"); imp = [i for i, l in enumerate(lines) if "|impostor|" in l]
                a, b = imp[3], imp[9]
                lines[b] = lines[b].rsplit("|", 1)[0] + "|" + lines[a].rsplit("|", 1)[1]
                return "\n".join(lines)
            self.set_tsv(d, "test", f)
        S("PM1 duplicated impostor acceptance rank", P, dup_rank, "acceptance ranks")
        def swap_ids(d):
            def f(t):
                lines = t.split("\n"); i = [k for k, l in enumerate(lines) if "|impostor|" in l][4]
                x = lines[i].split("|"); x[4], x[5] = x[5], x[4]; lines[i] = "|".join(x); return "\n".join(lines)
            self.set_tsv(d, "test", f)
        S("PM2 id_lo > id_hi (violates lower/higher image index order)", P, swap_ids, "inconsistent row")
        def wrong_count(d):
            _, rows = self.tsv_rows(d, "test")
            have = {frozenset((r[2], r[3])) for r in rows}
            imgs = [f"te_0000/te_0000_{k:04d}.jpg" for k in range(1, 12)]
            pair = next((a, b) for i, a in enumerate(imgs) for b in imgs[i + 1:] if frozenset((f"te_0000/{a}", f"te_0000/{b}")) not in have)
            victim = [r for r in rows if r[1] == "genuine" and r[4] != "te_0000"][0]
            new = f"te_0000/{pair[0]}", f"te_0000/{pair[1]}"
            self.set_tsv(d, "test", lambda t: t.replace("|".join(victim), "|".join([victim[0], "genuine", new[0], new[1], "te_0000", "te_0000", "genuine", ""]), 1))
        S("PM3 genuine pair moved to another identity (per-identity count rule broken)", P, wrong_count, "build_genuine rule")
        S("PM4 n_pool != rho * n_genuine", P, lambda d: (self.summ(d, lambda S_: S_["pair_params"]["test"].update(n_pool=7641)),
                                                        self.set_tsv(d, "test", lambda t: t.replace("n_pool=7640", "n_pool=7641", 1))), "rho * n_genuine")
        def dup_pair(d):
            def f(t):
                lines = t.split("\n"); imp = [i for i, l in enumerate(lines) if "|impostor|" in l]
                a, b = lines[imp[2]].split("|"), lines[imp[8]].split("|")
                b[2:6] = a[2:6]; lines[imp[8]] = "|".join(b); return "\n".join(lines)
            self.set_tsv(d, "test", f)
        S("PM5 the same pair appears twice", P, dup_pair, "duplicate")
        S("PM6 row numbering gap", P, lambda d: self.set_tsv(d, "test", lambda t: t.replace("\n00000005|", "\n00000500|", 1)), "malformed")
        S("PM7 pair TSV row with a wrong field count", P, lambda d: self.set_tsv(d, "test", lambda t: t.replace("|genuine|\n", "|genuine\n", 1)), "malformed")
        S("PM8 pair endpoints identical (same image twice)", P, lambda d: self.set_tsv(d, "test", lambda t: (lambda L: "\n".join(
            [l if "|genuine|" not in l or k != 3 else "|".join((lambda x: x[:3] + [x[2]] + x[4:])(l.split("|"))) for k, l in enumerate(L)]))(t.split("\n"))), "malformed")

    def section_verdict(self):
        spec = importlib.util.spec_from_file_location("checker_under_test", str(self.checker))
        m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
        def led(p, f, s):
            L = m.Ledger(); L.passed = p; L.fails = ["x"] * f; L.skips = ["y"] * s; return m.verdict(L)
        for name, got, want in (("verdict: passes only -> PASS/0", led(5, 0, 0), ("PASS", 0)),
                                ("verdict: passes + skipped check -> INCOMPLETE VERIFICATION/3", led(5, 0, 1), ("INCOMPLETE VERIFICATION", 3)),
                                ("verdict: failure + skip -> FAIL/1", led(5, 1, 1), ("FAIL", 1)),
                                ("verdict: nothing checked -> INCOMPLETE VERIFICATION/3", led(0, 0, 0), ("INCOMPLETE VERIFICATION", 3))):
            self.n += 1
            ok = got == want
            self.bad += not ok
            if not ok: self.failures.append(name)
            print(("ok       " if ok else "UNEXPECTED ") + f"{name}: {got}")

    def section_generator(self, P):
        env = lambda seed: {**os.environ, "PYTHONHASHSEED": str(seed)}
        a = self.base / "determinism_a"
        r = subprocess.run([sys.executable, str(self.generator), "--repo", self.repo, "--out", str(a), "--profile", "production"], capture_output=True, text=True, env=env(1))
        same = r.returncode == 0
        if same:
            for f in sorted(P.rglob("*")):
                g = a / "run" / f.relative_to(P)
                if f.is_dir():
                    same &= g.is_dir()
                elif f.suffix == ".npz":
                    with np.load(f) as x, np.load(g) as y:
                        same &= x.files == y.files and all(np.array_equal(x[k], y[k]) for k in x.files)
                else:
                    same &= g.is_file() and f.read_bytes() == g.read_bytes()
        self.n += 1; self.bad += not same
        if not same: self.failures.append("G1")
        print(("ok       " if same else "UNEXPECTED ") + "G1 generator output is identical across processes with different PYTHONHASHSEED (bytes; npz by array content)")
        def gen_rc(out, *extra, cwd=None):
            return subprocess.run([sys.executable, str(self.generator), "--repo", self.repo, "--out", str(out), "--profile", "stub-deep", *extra],
                                  capture_output=True, text=True, cwd=cwd)
        def check(name, cond):
            self.n += 1; self.bad += not cond
            if not cond: self.failures.append(name)
            print(("ok       " if cond else "UNEXPECTED ") + name)
        victim = self.work / "precious"; victim.mkdir(); (victim / "data.txt").write_text("keep")
        r = gen_rc(victim)
        check("G2 generator refuses a non-empty directory it did not create (contents untouched)", r.returncode == 2 and (victim / "data.txt").read_text() == "keep" and sorted(x.name for x in victim.iterdir()) == ["data.txt"])
        fx = self.work / "fx"; assert gen_rc(fx).returncode == 0
        r = gen_rc(fx)
        check("G3 generator refuses to overwrite its own fixture without --force", r.returncode == 2)
        (fx / "run" / "marker_in_run.txt").write_text("old")
        r = gen_rc(fx, "--force")
        check("G4 --force replaces only run/ of a fixture it created", r.returncode == 0 and not (fx / "run" / "marker_in_run.txt").exists() and (fx / "SYNTHETIC_FIXTURE.txt").is_file())
        (fx / "precious.txt").write_text("keep"); (fx / "run" / "keepme.txt").write_text("x")
        r = gen_rc(fx, "--force")
        check("G5 --force still refuses a fixture root holding foreign entries, and deletes nothing", r.returncode == 2 and (fx / "precious.txt").exists() and (fx / "run" / "keepme.txt").exists())
        r = gen_rc("/")
        check("G6 generator refuses the filesystem root", r.returncode == 2)
        r = gen_rc(Path.home())
        check("G7 generator refuses the home directory", r.returncode == 2)
        here = self.work / "cwdtest"; here.mkdir()
        r = gen_rc(".", cwd=here)
        check("G8 generator refuses the current directory", r.returncode == 2 and not any(here.iterdir()))
        lnk = self.work / "lnk"; os.symlink(self.work / "precious", lnk)
        r = gen_rc(lnk)
        check("G9 generator refuses a symbolic link as the output directory", r.returncode == 2)
        forged = self.work / "forged"; forged.mkdir(); (forged / "SYNTHETIC_FIXTURE.txt").write_text("not the real marker\n"); (forged / "run").mkdir(); (forged / "run" / "x").write_text("keep")
        r = gen_rc(forged, "--force")
        check("G10 generator refuses a directory whose marker is not the genuine marker", r.returncode == 2 and (forged / "run" / "x").exists())

    def section_legacy(self, label, src):
        SRC = {k: Path(v) for k, v in src.items()}
        pre = "L[" + label.split()[0] + "] "
        def lines(run): return [json.loads(l) for l in open(run / "verification_rows.jsonl")]
        def write_lines(run, recs): open(run / "verification_rows.jsonl", "w").write("".join(json.dumps(r) + "\n" for r in recs))
        def edit_rows(run, fn, pred=lambda r: r.get("paradigm") == "deep_learning", which=0):
            recs = lines(run); data = json.load(open(run / "verification_results.json")); n = -1
            for r in recs:
                if r["type"] == "row" and pred(r):
                    n += 1
                    if n == which:
                        idx = [j for j, x in enumerate([q for q in recs if q["type"] == "row"]) if x is r][0]
                        fn(r); fn(data["results"][idx]); break
            write_lines(run, recs); json.dump(data, open(run / "verification_results.json", "w"))
        def first_npz(run, prefix=""): return sorted((run / "scores").glob(prefix + "*__*.npz"))[0]
        npz_edit = self.npz_edit
        set_key = lambda k, v: (lambda r: r.__setitem__(k, v))
        del_key = lambda k: (lambda r: r.pop(k, None))
        rm = lambda p: (lambda run: (run / p).unlink())
        S = []
        def add(name, mut, sub, code=1, base="deep", args=("--test-mode",)): S.append((name, base, mut, list(args), code, sub))
        def m(r, f):
            d = json.load(open(r / "verification_results.json")); f(d); json.dump(d, open(r / "verification_results.json", "w"))
        add("summary missing", rm("verification_results.json"), "required file is missing")
        add("summary empty", lambda r: (r / "verification_results.json").write_text(""), "cannot be read as JSON")
        add("summary malformed", lambda r: (r / "verification_results.json").write_bytes((r / "verification_results.json").read_bytes()[:300]), "cannot be read as JSON")
        add("summary not a dict", lambda r: (r / "verification_results.json").write_text("[]"), "summary cross-checks skipped")
        add("summary INCOMPLETE", lambda r: m(r, lambda d: d.__setitem__("status", "INCOMPLETE")), "not COMPLETE")
        add("summary value altered", lambda r: m(r, lambda d: d["results"][0].__setitem__("accuracy", 0.5)), "not identical")
        add("summary n_rows altered", lambda r: m(r, lambda d: d.__setitem__("n_rows", 3)), "disagree")
        add("summary has failures", lambda r: m(r, lambda d: d.__setitem__("failures", [{"x": 1}])), "lists failures")
        add("summary wrong protocol", lambda r: m(r, lambda d: d.__setitem__("protocol", "other")), "protocol")
        add("stream missing", rm("verification_rows.jsonl"), "required file is missing")
        add("stream torn tail", lambda r: (r / "verification_rows.jsonl").write_bytes((r / "verification_rows.jsonl").read_bytes()[:-40]), "torn write")
        def blank(r):
            t = (r / "verification_rows.jsonl").read_text().split("\n"); t.insert(2, ""); (r / "verification_rows.jsonl").write_text("\n".join(t))
        add("stream blank line", blank, "not valid JSON")
        def app(r, s_):
            with open(r / "verification_rows.jsonl", "a") as f: f.write(s_)
        def ins(r, i, rec):
            L = lines(r); L.insert(i, rec); write_lines(r, L)
        add("stream non-dict line", lambda r: ins(r, 2, [1]), "known type")
        add("stream unknown type", lambda r: ins(r, 2, {"type": "zzz"}), "known type")
        add("stream no plan", lambda r: write_lines(r, lines(r)[1:]), "exactly one plan")
        add("stream duplicate plan", lambda r: ins(r, 1, lines(r)[0]), "exactly one plan")
        add("stream abort record", lambda r: (write_lines(r, lines(r)[:-1]), app(r, json.dumps({"type": "abort", "error": "x"}) + "\n")), "abort record")
        add("stream failure record", lambda r: ins(r, 3, {"type": "failure", "error": "x", "stage": "s", "model": "m", "protocol": "p"}), "failure record")
        add("stream missing end", lambda r: write_lines(r, lines(r)[:-1]), "end record")
        add("stream end INCOMPLETE", lambda r: (lambda L: (L[-1].__setitem__("status", "INCOMPLETE"), write_lines(r, L)))(lines(r)), "end record")
        add("stream duplicate end", lambda r: app(r, json.dumps(lines(r)[-1]) + "\n"), "end record")
        add("stream row deleted", lambda r: (lambda L: write_lines(r, L[:3] + L[4:]))(lines(r)), "plan expects")
        add("stream row duplicated", lambda r: (lambda L: write_lines(r, L[:3] + [L[2]] + L[3:]))(lines(r)), "duplicate")
        add("plan models not a list", lambda r: (lambda L: (L[0].__setitem__("models", "facenet"), write_lines(r, L)))(lines(r)), "plan record is malformed")
        add("plan n_rows_expected wrong", lambda r: (lambda L: (L[0].__setitem__("n_rows_expected", 99), write_lines(r, L)))(lines(r)), "plan record is malformed")
        add("row field missing", lambda r: edit_rows(r, del_key("accuracy")), "field 'accuracy' missing")
        add("row field wrong type", lambda r: edit_rows(r, set_key("accuracy", "0.9")), "field 'accuracy'")
        add("row field NaN", lambda r: edit_rows(r, set_key("auc", float("nan"))), "field 'auc'")
        add("row bool as int", lambda r: edit_rows(r, set_key("n_genuine", True)), "field 'n_genuine'")
        add("scores_file traversal ..", lambda r: edit_rows(r, set_key("scores_file", "../x.npz")), "plain *.npz")
        add("scores_file absolute", lambda r: edit_rows(r, set_key("scores_file", "/etc/passwd.npz")), "plain *.npz")
        add("scores_file subdir", lambda r: edit_rows(r, set_key("scores_file", "sub/x.npz")), "plain *.npz")
        add("scores_file test_pairs", lambda r: edit_rows(r, set_key("scores_file", "test_pairs.npz")), "plain *.npz")
        add("scores_file null", lambda r: edit_rows(r, set_key("scores_file", None)), "plain *.npz")
        def dupfile(r):
            names = [x["scores_file"] for x in lines(r) if x.get("scores_file")]
            edit_rows(r, set_key("scores_file", names[1]), which=0)
        add("two rows one score file", dupfile, "same score file")
        add("non-deep row names a file", lambda r: edit_rows(r, set_key("scores_file", "x.npz"), pred=lambda x: x.get("paradigm") == "traditional"), "non-deep row", base="mixed")
        def symlink(r):
            f = first_npz(r, "facenet"); g = first_npz(r, "arcface"); tmp = r / "scores" / "zz.npz"; shutil.copy(g, tmp); f.unlink(); f.symlink_to(tmp.name)
        add("score file is a symlink", symlink, "symbolic link")
        add("score file missing", lambda r: first_npz(r, "facenet").unlink(), "dangling")
        add("score file truncated", lambda r: (lambda f: f.write_bytes(f.read_bytes()[:max(1, f.stat().st_size // 2)]))(first_npz(r, "facenet")), "unreadable")
        add("score file empty", lambda r: first_npz(r, "arcface").write_bytes(b""), "unreadable")
        add("score file garbage", lambda r: first_npz(r, "arcface").write_bytes(b"not an archive" * 50), "unreadable")
        add("score file is .npy", lambda r: np.save(open(first_npz(r, "arcface"), "wb"), np.arange(5)), "not an .npz")
        add("score file extra array", lambda r: npz_edit(first_npz(r, "facenet"), lambda d: d.__setitem__("x", np.arange(3))), "has arrays")
        add("score file missing array", lambda r: npz_edit(first_npz(r, "facenet"), lambda d: d.pop("s_full")), "has arrays")
        add("valid as float", lambda r: npz_edit(first_npz(r, "facenet"), lambda d: d.__setitem__("valid", d["valid"].astype(float))), "'valid' is not a bool")
        add("s_bal wrong length", lambda r: npz_edit(first_npz(r, "facenet"), lambda d: d.__setitem__("s_bal", d["s_bal"][:-1])), "lengths do not match")
        add("s_bal int dtype", lambda r: npz_edit(first_npz(r, "facenet"), lambda d: d.__setitem__("s_bal", (d["s_bal"] * 100).astype(int))), "wrong dtype")
        def nan_score(d): d["s_bal"] = d["s_bal"].copy(); d["s_bal"][0] = np.nan
        add("NaN score", lambda r: npz_edit(first_npz(r, "facenet"), nan_score), "non-finite")
        def inf_score(d): d["s_full"] = d["s_full"].copy(); d["s_full"][3] = np.inf
        add("Inf score", lambda r: npz_edit(first_npz(r, "facenet"), inf_score), "non-finite")
        add("threshold mismatch", lambda r: npz_edit(first_npz(r, "facenet"), lambda d: d.__setitem__("threshold", np.float64(d["threshold"] + 0.01))), "saved threshold")
        add("threshold 1-D", lambda r: npz_edit(first_npz(r, "facenet"), lambda d: d.__setitem__("threshold", np.array([float(d["threshold"])]))), "wrong dtype or shape")
        add("scores reversed", lambda r: npz_edit(first_npz(r, "arcface"), lambda d: d.__setitem__("s_bal", d["s_bal"][::-1].copy())), "recomputed")
        def flip_valid(d): d["valid"] = d["valid"].copy(); d["valid"][0] = ~d["valid"][0]
        add("valid mask altered", lambda r: npz_edit(first_npz(r, "facenet"), flip_valid), "valid mask")
        tp = lambda r: r / "scores" / "test_pairs.npz"
        add("test_pairs missing", lambda r: tp(r).unlink(), "required file is missing")
        add("test_pairs truncated", lambda r: tp(r).write_bytes(tp(r).read_bytes()[:max(1, tp(r).stat().st_size // 2)]), "unreadable")
        add("test_pairs empty arrays", lambda r: npz_edit(tp(r), lambda d: d.update(bal_pairs=d["bal_pairs"][:0], bal_labels=d["bal_labels"][:0])), "empty")
        add("pair index out of range", lambda r: npz_edit(tp(r), lambda d: d.__setitem__("bal_pairs", d["bal_pairs"] + 100000)), "outside image_ids")
        add("pair index negative", lambda r: npz_edit(tp(r), lambda d: d.__setitem__("bal_pairs", d["bal_pairs"] - 100000)), "outside image_ids")
        add("single-class labels", lambda r: npz_edit(tp(r), lambda d: d.__setitem__("bal_labels", np.ones_like(d["bal_labels"]))), "single class")
        add("labels not bool", lambda r: npz_edit(tp(r), lambda d: d.__setitem__("full_labels", d["full_labels"].astype(int))), "wrong dtype")
        add("pairs float dtype", lambda r: npz_edit(tp(r), lambda d: d.__setitem__("bal_pairs", d["bal_pairs"].astype(float))), "wrong dtype")
        add("labels length mismatch", lambda r: npz_edit(tp(r), lambda d: d.__setitem__("full_labels", d["full_labels"][:-1])), "mis-shaped")
        add("test_pairs missing key", lambda r: npz_edit(tp(r), lambda d: d.pop("image_ids")), "has arrays")
        add("duplicate image ids", lambda r: npz_edit(tp(r), lambda d: d.__setitem__("image_ids", np.array(["a"] * len(d["image_ids"])))), "not unique")
        add("orphan npz", lambda r: shutil.copy(first_npz(r, "facenet"), r / "scores" / "facenet__orphan__sev0.50.npz"), "orphan")
        add("tmp file in run dir", lambda r: (r / "verification_results.json.tmp").write_text("x"), "unexpected entry in the run directory")
        add("tmp.npz in scores", lambda r: (r / "scores" / "facenet__x__sev0.10.tmp.npz").write_bytes(b"x"), "unexpected or orphan")
        add("extra file in scores", lambda r: (r / "scores" / "notes.txt").write_text("x"), "unexpected or orphan")
        add("subdir in scores", lambda r: (r / "scores" / "sub").mkdir(), "unexpected or orphan")
        def thr_edit(r):
            f = first_npz(r, "facenet"); npz_edit(f, lambda d: d.__setitem__("threshold", np.float64(d["threshold"] + 0.01)))
            edit_rows(r, lambda x: x.__setitem__("threshold", x["threshold"] + 0.01), pred=lambda x: x.get("model") == "facenet")
        add("threshold not constant across conditions", thr_edit, "not constant")
        add("--expect-per-model 96", lambda r: None, "row count != 96", args=("--test-mode", "--expect-per-model", "96"))
        add("--expect-models mismatch", lambda r: None, "expected", args=("--test-mode", "--expect-models", "facenet"))
        add("strict mode on a stub run reports the stub backbones", lambda r: None, "stub backbones were used", args=("--expect-models", "facenet,arcface", "--expect-per-model", "4", "--partial-grid"))
        add("missing --test-mode and no expect flags is a usage error", lambda r: None, "required", code=2, args=())
        def all_nondeep(r):
            for _ in range(8): edit_rows(r, set_key("paradigm", "traditional"), pred=lambda x: x.get("paradigm") == "deep_learning", which=0)
        add("no deep rows at all", all_nondeep, "no deep-learning rows", base="mixed")
        for name, base, mut, args, code, sub in S:
            run = None
            d, sc = self.copy(SRC[base])
            try:
                mut(d)
            except Exception as e:
                self.n += 1; self.bad += 1; self.failures.append(pre + name); print(f"UNEXPECTED {pre}{name}: harness error {e!r}"); continue
            self.expect(pre + name, self.run(d, *args), code, sub)
            shutil.rmtree(sc, ignore_errors=True)

    def section_usage(self, SD):
        for name, a in (("no arguments", []), ("missing directory", ["/nonexistent_dir_xyz"]), ("unknown option", [str(SD), "--bogus"]),
                        ("two directories", [str(SD), str(SD)]), ("non-numeric --expect-per-model", [str(SD), "--expect-per-model", "x"]),
                        ("--partial-grid with --test-mode", [str(SD), "--test-mode", "--partial-grid"]),
                        ("--expect-per-model 4 without --partial-grid in strict mode", [str(SD), "--expect-models", "facenet,arcface", "--expect-per-model", "4"])):
            r = subprocess.run([sys.executable, "-I", str(self.checker), *a], capture_output=True, text=True)
            self.n += 1
            ok = r.returncode == 2 and "ALL CHECKS PASSED" not in r.stdout
            self.bad += not ok
            if not ok: self.failures.append("usage: " + name)
            print(("ok       " if ok else "UNEXPECTED ") + f"usage error: {name}: exit {r.returncode} (expected 2)")


def run_all(repo, checker=None, generator=None, real_stubs=None, keep=False):
    h = Harness(repo, checker or HERE / "verify_run_npz.py", generator or HERE / "generate_synthetic_fixture.py", real_stubs, keep)
    return h.run_all()


def test_regressions():                                            # pytest entry point
    repo = os.environ.get("FACEEVAL_REPO")
    if not repo:
        import pytest
        pytest.skip("set FACEEVAL_REPO to a FaceEval-X checkout")
    assert run_all(repo, real_stubs=os.environ.get("FACEEVAL_REAL_STUBS"))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--repo", default=os.environ.get("FACEEVAL_REPO"), help="FaceEval-X checkout (or set FACEEVAL_REPO)")
    ap.add_argument("--checker", default=str(HERE / "verify_run_npz.py"))
    ap.add_argument("--generator", default=str(HERE / "generate_synthetic_fixture.py"))
    ap.add_argument("--real-stubs", default=None, help="directory holding pipeline-produced stub runs deep_only/ and mixed/ (optional)")
    ap.add_argument("--keep", action="store_true", help="keep the temporary work directory")
    a = ap.parse_args(argv)
    if not a.repo:
        ap.error("--repo (or FACEEVAL_REPO) is required")
    return 0 if run_all(a.repo, a.checker, a.generator, a.real_stubs, a.keep) else 1


if __name__ == "__main__":
    sys.exit(main())
