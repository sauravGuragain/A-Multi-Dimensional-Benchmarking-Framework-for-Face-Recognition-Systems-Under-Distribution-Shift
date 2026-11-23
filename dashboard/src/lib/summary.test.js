import test from "node:test";
import assert from "node:assert/strict";
import { summarize, cleanCell, cleanTooltip } from "./summary.js";
import { fmt, pct, mean } from "./format.js";

const r = (model, type, sev, acc, extra = {}) => ({
  model_name: model, model_paradigm: "traditional", perturbation_type: type, severity: sev, accuracy: acc, auc: null, eer: null, ...extra,
});

test("clean accuracy and the all-condition mean are different quantities", () => {
  const rows = [r("m", null, null, 0.9), r("m", "gaussian_blur", 0, 0.9), r("m", "gaussian_blur", 0.5, 0.5), r("m", "gaussian_blur", 1, 0.1)];
  const [m] = summarize(rows);
  assert.equal(m.clean.value, 0.9); // two clean rows -> one baseline
  assert.equal(m.clean.rows, 2);
  assert.equal(m.conditions, 4);
  assert.ok(Math.abs(m.meanAccuracy - (0.9 + 0.9 + 0.5 + 0.1) / 4) < 1e-12); // every row counted once
  assert.notEqual(m.meanAccuracy, m.clean.value);
});

test("missing metrics stay null, never 0", () => {
  const [m] = summarize([r("m", null, null, null), r("m", "gaussian_blur", 0.5, undefined)]);
  assert.equal(m.meanAccuracy, null);
  assert.equal(m.meanAuc, null);
  assert.equal(m.meanEer, null);
  assert.equal(m.clean.value, null);
  assert.equal(cleanCell(m.clean), "—");
});

test("AUC/EER means ignore rows where they were not stored", () => {
  const [m] = summarize([r("m", null, null, 0.8, { auc: 0.9 }), r("m", "gamma", 0.2, 0.7, { auc: null })]);
  assert.equal(m.meanAuc, 0.9);
});

test("models are grouped per model and sorted by mean accuracy (missing last)", () => {
  const out = summarize([r("a", null, null, 0.5), r("b", null, null, 0.9), r("c", null, null, null)]);
  assert.deepEqual(out.map((x) => x.model), ["b", "a", "c"]);
});

test("clean cell: value, range, or dash; tooltip explains each", () => {
  const one = summarize([r("m", null, null, 0.8), r("m", "motion_blur", 0, 0.8)])[0].clean;
  assert.equal(cleanCell(one), "0.8000");
  assert.match(cleanTooltip(one), /counted once/);
  const two = summarize([r("m", null, null, 0.8), r("m", "motion_blur", 0, 0.7)])[0].clean;
  assert.equal(cleanCell(two), "0.7000–0.8000");
  assert.match(cleanTooltip(two), /2 groups/);
  const none = summarize([r("m", "motion_blur", 0.5, 0.7)])[0].clean;
  assert.equal(cleanCell(none), "—");
  assert.match(cleanTooltip(none), /No clean-equivalent rows/);
  const noAcc = summarize([r("m", null, null, null)])[0].clean;
  assert.match(cleanTooltip(noAcc), /none with a stored accuracy/);
});

test("formatters show a dash for missing values and never 0", () => {
  for (const v of [null, undefined, NaN, Infinity, "0.5"]) {
    assert.equal(fmt(v), "—");
    assert.equal(pct(v), "—");
  }
  assert.equal(fmt(0), "0.000"); // a real zero is shown as zero
  assert.equal(mean([null, undefined, NaN]), null);
  assert.equal(mean([0.5, null, 1]), 0.75);
});
