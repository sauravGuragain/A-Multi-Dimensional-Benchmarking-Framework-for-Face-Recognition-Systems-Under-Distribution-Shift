// Missing values are shown as an em dash, never as 0.
export const isNum = (v) => typeof v === "number" && Number.isFinite(v);
export const fmt = (v, digits = 3) => (isNum(v) ? v.toFixed(digits) : "—");
export const pct = (v, digits = 1) => (isNum(v) ? `${(v * 100).toFixed(digits)}%` : "—");
export const mean = (xs) => {
  const ok = xs.filter(isNum);
  return ok.length ? ok.reduce((a, b) => a + b, 0) / ok.length : null;
};
