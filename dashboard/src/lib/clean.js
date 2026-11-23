// Which stored evaluation rows are the "clean" (unperturbed) condition.
//
// Sources (faceeval/perturbation/, docs/PROTOCOL.md section 7):
//  * Severity 0.0 is an exact copy of the image for EVERY perturbation type
//    (BasePerturbation.apply returns image.copy() before calling the type's
//    own code), so all severity-0.0 rows are the same clean condition.  This
//    holds for brightness, contrast and gamma too.  Note that the raw
//    parameters stored in a severity-0.0 spec (e.g. brightness
//    additive_shift = -0.4) are NOT applied, so rows are classified by
//    severity, never by raw parameters.
//  * Brightness and gamma are centred: severity 0.5 is exactly the identity
//    transform (additive shift 0; gamma 1.0).
//  * Contrast has no exact neutral severity.  It is log-interpolated, so its
//    scale is 0.949 at severity 0.5 and reaches 1.0 only near 0.523.  It is
//    therefore deliberately NOT listed here.
// tests/test_dashboard_clean_semantics.py checks this table against the code.
export const NEUTRAL_SEVERITY = {
  brightness: 0.5,
  gamma: 0.5,
};

// Absolute tolerance used to decide that two accuracy values are "the same".
export const EPS = 1e-9;
const near = (a, b, eps = EPS) => Math.abs(a - b) <= eps;
const isFiniteNumber = (v) => typeof v === "number" && Number.isFinite(v);

// Stored representation (api/routers/results.py): a row with NO perturbation
// has perturbation_type = null and severity = null.  A row whose spec exists
// has a string type (never null) and, normally, a numeric severity.
//
// A row is clean if
//   * it is an unperturbed baseline row (type null/undefined) with no severity,
//     or severity 0; or
//   * its severity is 0.0 (exact copy of the image for every type); or
//   * it sits exactly on the neutral severity of brightness or gamma.
// A perturbed row (any string type, including "") with an unknown severity is
// NOT clean, and neither is a typeless row that carries a non-zero severity.
export function isCleanCondition(row) {
  const type = row.perturbation_type;
  const s = row.severity;
  if (type === null || type === undefined) {
    return s === null || s === undefined || (isFiniteNumber(s) && near(s, 0));
  }
  if (!isFiniteNumber(s)) return false;
  if (near(s, 0)) return true;
  const neutral = NEUTRAL_SEVERITY[type];
  return neutral !== undefined && near(s, neutral);
}

// Deterministic anchor-based clustering of finite numbers.
//   1. Sort ascending (so the result never depends on input order).
//   2. The smallest unassigned value is the cluster ANCHOR.
//   3. Every following value v with v - anchor <= eps joins that cluster;
//      the first value beyond that starts a new cluster with itself as anchor.
// Every member is within eps of its own anchor.  Closeness is not chained:
// 0, 0.8e-9 and 1.6e-9 give two clusters ({0, 0.8e-9} and {1.6e-9}), never
// one.  Returns clusters as ascending arrays; non-finite values are ignored.
export function clusterValues(values, eps = EPS) {
  const sorted = values.filter(isFiniteNumber).sort((a, b) => a - b);
  const clusters = [];
  let anchor = null;
  for (const v of sorted) {
    if (anchor !== null && v - anchor <= eps) {
      clusters[clusters.length - 1].push(v);
    } else {
      clusters.push([v]);
      anchor = v;
    }
  }
  return clusters;
}

// Collapse a model's clean rows into ONE baseline so equivalent clean
// conditions (an unperturbed row plus up to 16 severity-0.0 rows) are not
// counted repeatedly.
//   rows     - number of clean-equivalent rows found
//   valid    - how many of them have a finite accuracy
//   distinct - number of accuracy clusters (see clusterValues)
//   value    - the cluster's smallest value if there is exactly ONE cluster,
//              else null (disagreeing rows are never averaged or invented)
//   min/max  - smallest/largest observed clean accuracy, or null if none
export function cleanBaseline(rows) {
  const clean = rows.filter(isCleanCondition);
  const accs = clean.map((r) => r.accuracy).filter(isFiniteNumber);
  const clusters = clusterValues(accs);
  return {
    rows: clean.length,
    valid: accs.length,
    distinct: clusters.length,
    value: clusters.length === 1 ? clusters[0][0] : null,
    min: clusters.length ? clusters[0][0] : null,
    max: clusters.length ? clusters[clusters.length - 1].at(-1) : null,
  };
}
