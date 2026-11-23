// Run with: npm test   (Node's built-in test runner; no extra dependency)
import test from "node:test";
import assert from "node:assert/strict";
import { isCleanCondition, cleanBaseline, clusterValues, NEUTRAL_SEVERITY, EPS } from "./clean.js";

const row = (perturbation_type, severity, accuracy = 0.9) => ({
  perturbation_type,
  severity,
  accuracy,
});

test("unperturbed baseline row is clean", () => {
  assert.equal(isCleanCondition(row(null, null)), true);
  assert.equal(isCleanCondition(row(undefined, undefined)), true);
});

test("blur at severity 0 is clean; blur at 0.5 is not", () => {
  assert.equal(isCleanCondition(row("gaussian_blur", 0)), true);
  assert.equal(isCleanCondition(row("gaussian_blur", 0.5)), false);
  assert.equal(isCleanCondition(row("motion_blur", 0.0)), true);
});

test("brightness at 0.5 is clean (centred, shift 0)", () => {
  assert.equal(isCleanCondition(row("brightness", 0.5)), true);
});

test("gamma at 0.5 is clean (centred, gamma 1.0)", () => {
  assert.equal(isCleanCondition(row("gamma", 0.5)), true);
});

test("contrast has no exact neutral severity: 0.5 is NOT clean", () => {
  assert.equal("contrast" in NEUTRAL_SEVERITY, false);
  assert.equal(isCleanCondition(row("contrast", 0.5)), false); // scale 0.949
  assert.equal(isCleanCondition(row("contrast", 0.523)), false); // not a defined neutral point
});

test("severity 0.0 is the clean copy for the centred types too", () => {
  for (const t of ["brightness", "contrast", "gamma"]) {
    assert.equal(isCleanCondition(row(t, 0)), true, t);
  }
});

test("non-neutral conditions are not clean", () => {
  for (const [t, s] of [
    ["brightness", 0.2], ["brightness", 0.4], ["brightness", 0.6], ["brightness", 1.0],
    ["gamma", 0.4], ["gamma", 1.0], ["contrast", 0.2], ["contrast", 1.0],
    ["jpeg_compression", 0.5], ["rotation", 0.2], ["scaling", 1.0],
  ]) {
    assert.equal(isCleanCondition(row(t, s)), false, `${t}@${s}`);
  }
});

test("a perturbed row with unknown severity is not clean", () => {
  assert.equal(isCleanCondition(row("brightness", null)), false);
  assert.equal(isCleanCondition(row("gamma", undefined)), false);
  assert.equal(isCleanCondition(row("gaussian_blur", NaN)), false);
});

test("neutral severity of 0.5 is not applied to other types", () => {
  assert.equal(isCleanCondition(row("rotation", 0.5)), false);
});

test("equivalent clean rows are collapsed, not counted repeatedly", () => {
  const rows = [row(null, null, 0.8)];
  for (let i = 0; i < 16; i++) rows.push(row(`type_${i}`, 0, 0.8)); // 16 severity-0 copies
  rows.push(row("brightness", 0.5, 0.8), row("gamma", 0.5, 0.8));
  rows.push(row("gaussian_blur", 0.5, 0.3)); // perturbed: must not leak in
  const b = cleanBaseline(rows);
  assert.equal(b.rows, 19);
  assert.equal(b.distinct, 1);
  assert.equal(b.value, 0.8);
});

test("empty-string or non-null type is not treated as a baseline row", () => {
  assert.equal(isCleanCondition(row("", null)), false); // a spec exists, severity unknown
  assert.equal(isCleanCondition(row("", 0)), true); // severity 0.0 is still the clean copy
  assert.equal(isCleanCondition(row(0, null)), false);
});

test("a typeless row with a non-zero severity is not clean", () => {
  assert.equal(isCleanCondition(row(null, 0.4)), false);
  assert.equal(isCleanCondition(row(null, 0)), true);
});

test("severity compares with an absolute tolerance, not exact equality", () => {
  assert.equal(isCleanCondition(row("brightness", 0.5 + 1e-10)), true);
  assert.equal(isCleanCondition(row("brightness", 0.5 + 1e-6)), false);
});

// ---- tolerance clustering ---------------------------------------------------
const accRows = (vals) => vals.map((v) => row(null, null, v));
const permutations = (xs) =>
  xs.length <= 1
    ? [xs]
    : xs.flatMap((x, i) => permutations([...xs.slice(0, i), ...xs.slice(i + 1)]).map((p) => [x, ...p]));

test("values just inside, exactly at, and just outside the tolerance (exact binary arithmetic)", () => {
  const eps = 0.25; // all values below are exactly representable
  assert.deepEqual(clusterValues([1, 1.25 - 2 ** -20], eps).length, 1); // just below
  assert.deepEqual(clusterValues([1, 1.25], eps).length, 1); // exactly at: inclusive
  assert.deepEqual(clusterValues([1, 1.25 + 2 ** -20], eps).length, 2); // just above
});

test("default EPS: 0.9e-9 apart is one cluster, 1.1e-9 apart is two", () => {
  assert.equal(EPS, 1e-9);
  assert.equal(cleanBaseline(accRows([0.8, 0.8 + 0.9e-9])).distinct, 1);
  assert.equal(cleanBaseline(accRows([0.8, 0.8 + 1.1e-9])).distinct, 2);
});

test("values straddling a decimal rounding boundary are not split (old toFixed(9) bug)", () => {
  // 0.8000000004 and 0.8000000006 round to different 9-decimal strings but differ by 2e-10.
  const b = cleanBaseline(accRows([0.8000000004, 0.8000000006]));
  assert.equal(b.distinct, 1);
  assert.equal(b.value, 0.8000000004); // the cluster's smallest value, not an average
});

test("closeness is not chained: 0, 0.8e-9, 1.6e-9 give two clusters in any order", () => {
  const vals = [0.5, 0.5 + 0.8e-9, 0.5 + 1.6e-9];
  for (const p of permutations(vals)) {
    const cl = clusterValues(p);
    assert.equal(cl.length, 2);
    assert.equal(cl[0].length, 2);
    assert.equal(cl[1].length, 1);
  }
});

test("result is identical for every ordering of the input rows", () => {
  const vals = [0.7, 0.8, 0.8 + 5e-10, 0.8 + 3e-9];
  const expected = JSON.stringify(cleanBaseline(accRows(vals)));
  for (const p of permutations(vals)) {
    assert.equal(JSON.stringify(cleanBaseline(accRows(p))), expected);
  }
});

test("one cluster gives one value; rows, valid and clusters are counted separately", () => {
  const b = cleanBaseline([...accRows([0.8, 0.8, 0.8 + 1e-10]), row("gaussian_blur", 0, NaN), row("gamma", 0.5, null)]);
  assert.equal(b.rows, 5);
  assert.equal(b.valid, 3);
  assert.equal(b.distinct, 1);
  assert.equal(b.value, 0.8);
});

test("non-finite and missing accuracies are ignored, never zero", () => {
  const b = cleanBaseline([row(null, null, Infinity), row(null, null, -Infinity), row(null, null, NaN), row(null, null, "0.9")]);
  assert.equal(b.valid, 0);
  assert.equal(b.value, null);
  assert.equal(b.min, null);
  assert.equal(b.max, null);
  assert.equal(b.rows, 4);
});

test("disagreeing clusters: no value, min and max of observed values", () => {
  const b = cleanBaseline(accRows([0.8, 0.8 + 1e-10, 0.7, 0.9]));
  assert.equal(b.value, null);
  assert.equal(b.distinct, 3);
  assert.equal(b.min, 0.7);
  assert.equal(b.max, 0.9);
});

test("disagreeing clean rows give no single value, only min/max", () => {
  const b = cleanBaseline([row(null, null, 0.8), row("gaussian_blur", 0, 0.7)]);
  assert.equal(b.value, null);
  assert.equal(b.distinct, 2);
  assert.equal(b.min, 0.7);
  assert.equal(b.max, 0.8);
});

test("no clean rows, or clean rows without accuracy, give null (never 0)", () => {
  assert.equal(cleanBaseline([row("gaussian_blur", 0.5, 0.4)]).value, null);
  const b = cleanBaseline([row(null, null, null), { perturbation_type: "gaussian_blur", severity: 0 }]); // accuracy key absent
  assert.equal(b.value, null);
  assert.equal(b.min, null);
  assert.equal(b.rows, 2);
});
