import { fmt, mean, isNum } from "./format.js";
import { cleanBaseline, EPS } from "./clean.js";

// One summary row per model from the stored (model x condition) results.
//   clean         - the collapsed clean baseline (see clean.js)
//   meanAccuracy  - mean over EVERY stored condition with an accuracy, clean and
//                   perturbed together; each stored clean row counts once here.
//                   It is a different quantity from the clean baseline.
// Missing values stay null (mean() ignores them); nothing becomes 0.
export function summarize(results) {
  const byModel = new Map();
  for (const r of results) {
    if (!byModel.has(r.model_name)) byModel.set(r.model_name, []);
    byModel.get(r.model_name).push(r);
  }
  return [...byModel.entries()]
    .map(([model, rs]) => ({
      model,
      paradigm: rs[0].model_paradigm,
      conditions: rs.length,
      clean: cleanBaseline(rs),
      meanAccuracy: mean(rs.map((r) => r.accuracy)),
      meanAuc: mean(rs.map((r) => r.auc)),
      meanEer: mean(rs.map((r) => r.eer)),
    }))
    .sort(
      (a, b) =>
        (isNum(b.meanAccuracy) ? b.meanAccuracy : -1) -
        (isNum(a.meanAccuracy) ? a.meanAccuracy : -1)
    );
}

// Text for the "Clean accuracy" cell: one value, a range if clean rows
// disagree, or a dash if there is no stored clean accuracy.
export function cleanCell(c, digits = 4) {
  if (c.value !== null) return fmt(c.value, digits);
  if (c.distinct > 1) return `${fmt(c.min, digits)}–${fmt(c.max, digits)}`;
  return "—";
}

export function cleanTooltip(c) {
  if (c.rows === 0) return "No clean-equivalent rows are stored for this model.";
  if (c.valid === 0) return `${c.rows} clean-equivalent row(s) found, none with a stored accuracy.`;
  if (c.distinct > 1) {
    return (
      `${c.valid} clean-equivalent accuracies fall into ${c.distinct} groups that differ by more than ` +
      `${EPS}; the range ${fmt(c.min, 4)}–${fmt(c.max, 4)} is shown instead of a single value.`
    );
  }
  return `${c.valid} clean-equivalent accuracy value(s) from ${c.rows} row(s), counted once.`;
}
