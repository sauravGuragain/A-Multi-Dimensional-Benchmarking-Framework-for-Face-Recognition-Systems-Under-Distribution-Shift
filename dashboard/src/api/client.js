// All calls are GET (the API is read-only).  By default requests are relative
// ("/api/..."), so in development they go through the Vite proxy (see
// vite.config.js) and avoid cross-origin/CORS and localhost-vs-127.0.0.1 issues.
// Set VITE_API_BASE only if the API is served from a different origin.
const BASE = (import.meta.env.VITE_API_BASE || "").replace(/\/+$/, "");

const enc = encodeURIComponent;

// `status` 0 = the request never reached the API; 503 = the API answered but
// its data (the results database) is unavailable.
function apiError(message, status) {
  const e = new Error(message);
  e.status = status;
  return e;
}

async function request(path) {
  let res;
  try {
    res = await fetch(`${BASE}${path}`, { headers: { Accept: "application/json" } });
  } catch {
    throw apiError(
      "Cannot reach the API. Is the backend running? (see README: Optional: API and dashboard)",
      0
    );
  }
  if (!res.ok) {
    const body = await res.json().catch(() => ({}));
    const detail = typeof body.detail === "string" ? body.detail : null;
    if (detail) throw apiError(detail, res.status);
    if (res.status >= 500) {
      // The Vite proxy answers 500 when nothing is listening on the API port.
      throw apiError(
        `API unreachable or failing (HTTP ${res.status}). Is the backend running on 127.0.0.1:8000?`,
        res.status
      );
    }
    throw apiError(`${res.status} ${res.statusText}`, res.status);
  }
  return res.json();
}

function qs(params) {
  const s = new URLSearchParams(params).toString();
  return s ? `?${s}` : "";
}

export const api = {
  health: () => request("/api/health"),
  status: () => request("/api/status"),

  listRuns: (params = {}) => request(`/api/runs${qs(params)}`),
  getRun: (runId) => request(`/api/runs/${enc(runId)}`),

  listResults: (runId, params = {}) =>
    request(`/api/runs/${enc(runId)}/results${qs(params)}`),
  getAdcCurve: (runId, modelName, perturbationType) =>
    request(
      `/api/runs/${enc(runId)}/results/adc${qs({
        model_name: modelName,
        perturbation_type: perturbationType,
      })}`
    ),

  listFingerprints: (runId) => request(`/api/runs/${enc(runId)}/fingerprints`),
  getFingerprint: (runId, modelName) =>
    request(`/api/runs/${enc(runId)}/fingerprints/${enc(modelName)}`),
  compareFingerprints: (runId, modelA, modelB) =>
    request(
      `/api/runs/${enc(runId)}/fingerprints/compare/${enc(modelA)}/${enc(modelB)}`
    ),

  listRankings: (runId) => request(`/api/runs/${enc(runId)}/deployment/rankings`),
  getRankingForScenario: (runId, scenario) =>
    request(`/api/runs/${enc(runId)}/deployment/rankings/${enc(scenario)}`),
  getDecisionGuide: (runId) =>
    request(`/api/runs/${enc(runId)}/deployment/decision-guide`),

  listFigures: (subfolder = "") => request(`/api/figures${qs(subfolder ? { subfolder } : {})}`),
  // The API returns already-encoded relative URLs; this only adds the base.
  assetUrl: (url) => (url ? `${BASE}${url}` : null),

  listReportFiles: (runId) => request(`/api/runs/${enc(runId)}/reports/list`),

  listModels: () => request("/api/registry/models"),
  listPerturbations: () => request("/api/registry/perturbations"),
  listScenarios: () => request("/api/registry/scenarios"),
};

export default api;
