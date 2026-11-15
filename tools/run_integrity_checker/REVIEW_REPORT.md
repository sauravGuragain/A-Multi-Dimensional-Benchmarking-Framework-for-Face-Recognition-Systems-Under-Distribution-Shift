# Review report - FaceEval-X integrity checker

All verification below is CLOUD-SIDE (Anthropic cloud sandbox). Nothing was run on the user's Mac, no local files or real run
artifacts were verified, no real benchmark or model inference was executed, and nothing was committed, pushed or applied to any repo.

## Files
Base: original 583-line checker, SHA-256 3fb074a4b5ac19f2bbfd1b0e08f9fd261e7b605ab552d38410f0b39caa109254 (left unmodified in the cloud).
Final: verify_run_npz.py (839 lines); verify_run_npz.patch is the full diff against the base. See SHA256SUMS.txt for all hashes.

## Changes in this round
1. Test-mode symlink behaviour now matches the documentation: `--test-mode` follows symlinks for the run dir, `scores/` and top-level
   artifacts; strict mode rejects all of them; a symlinked per-condition score file is rejected in every mode.
2. Pair manifests cross-checked against the repo's actual writer (`manifest_rows`, `build_genuine`, `build_pairs`). Rules now enforced:
   endpoint identity resolved via a lookup of `identity/image_id` (so identities containing '/' are handled), a != b, id_lo <= id_hi and
   {id_lo,id_hi} equals the endpoints' identities, label consistent with same/different identity, pair uniqueness, contiguous numbering,
   n_pool == rho * n_genuine, impostor acceptance ranks exactly 0..K-1, per-identity genuine counts equal to the build_genuine rule
   (all pairs, or the cap of 10). Regression tests PM1-PM8 cover these.
3. Synthetic fixtures are announced in the checker output as synthetic.
4. generate_synthetic_fixture.py: deterministic SHA-256 seeds (not Python hash()), uses the repo's real pair/split writers, refuses
   unsafe output directories, only replaces a `run/` it created.
5. Harness rewritten as a standalone script.

## Cloud-side regression result (final run)
`HARNESS RESULT: 252 scenarios, 252 as expected, 0 unexpected, 0 skipped.`
Expected outcomes observed: 16 expected PASS (exit 0), 211 expected FAIL (exit 1), 10 expected usage error (exit 2);
the remaining 15 are generator-determinism/refusal checks, 4 verdict-logic checks and the immutability check (non-exit-code assertions).
Composition: valid fixtures (V1-V7), summary/protocol/manifest/grid/plan corruptions, CSV, root-entry, score-range, detector-rate,
strict and test-mode symlink cases (SS1-SS8, TS1-TS8), pair-manifest cases (T1-T7, PM1-PM8), 75 legacy scenarios run on generated stubs and
75 on the real-pipeline stub runs, usage errors, generator safety (G1-G10).
One first-run mismatch (a truncation scenario that was a no-op on small generated files) was a harness bug and was fixed; the final run is clean.
Immutability: the harness hashed 8 objects (base fixtures, supplied stub runs, checker, generator) before and after - unchanged. The original 583-line
checker and the real-pipeline stub runs were also confirmed byte-identical afterward.

## Metric coverage
Independently recomputed from saved scores: accuracy, tar, far, far_full, auc, auc_full, n_genuine, n_impostor_balanced/full,
n_failed_embeddings, n_failed_pairs_balanced/full, failed-pair scores == 0, saved-vs-row threshold, clean-baseline identity.
Consistency/range only (compared or sanity-bounded, not recomputed): threshold, val_balanced_accuracy and n_test_images constancy,
val_n_failed_embeddings <= validation count, detector rates in [0,1] or NaN, score range |s| <= 1.
NOT verified: eer, eer_full, frr_at_far_* (incl. _full), clean-validation threshold selection/optimality, true validation failure count,
embedding provenance, whether perturbations were applied (identical-to-clean scores give only an informational note), dataset authenticity
(fingerprint is self-reported), classical-model metrics.

## Remaining limitations (not production-proven)
- The checker has never been run against a real FaceNet/ArcFace run directory; only synthetic and stub runs were used.
- Synthetic IDs/scores validate checker structure only, not real LFW or real model execution.
- Protocol constants are hard-coded from the repo at the time of review and must be re-confirmed if the repo changes.
