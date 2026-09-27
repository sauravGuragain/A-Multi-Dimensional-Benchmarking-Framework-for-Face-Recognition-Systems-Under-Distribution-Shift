const BASE = import.meta.env.VITE_API_BASE || "http://localhost:8000";

async function request(path, opts = {}) {
  const res = await fetch(`${BASE}${path}`, {
    headers: { "Content-Type": "application/json" },
    ...opts,
  });
  if (!res.ok) {
    const body = await res.json().catch(() => ({}));
    throw new Error(body.detail || `${res.status} ${res.statusText}`);
  }
  return res.json();
}

export const api = {
  health: () => request("/api/health"),

  listRuns: (params = {}) => {
    const qs = new URLSearchParams(params).toString();
    return request(`/api/runs${qs ? `?${qs}` : ""}`);
  },
  getRun: (runId) => request(`/api/runs/${runId}`),
  deleteRun: (runId) => request(`/api/runs/${runId}`, { method: "DELETE" }),

  listResults: (runId, params = {}) => {
    const qs = new URLSearchParams(params).toString();
    return request(`/api/runs/${runId}/results${qs ? `?${qs}` : ""}`);
  },
  getAdcCurve: (runId, modelName, perturbationType) =>
    request(
      `/api/runs/${runId}/results/adc?model_name=${encodeURIComponent(
        modelName
      )}&perturbation_type=${encodeURIComponent(perturbationType)}`
    ),

  listFingerprints: (runId) => request(`/api/runs/${runId}/fingerprints`),
  getFingerprint: (runId, modelName) =>
    request(`/api/runs/${runId}/fingerprints/${encodeURIComponent(modelName)}`),
  compareFingerprints: (runId, modelA, modelB) =>
    request(
      `/api/runs/${runId}/fingerprints/compare/${encodeURIComponent(
        modelA
      )}/${encodeURIComponent(modelB)}`
    ),

  listRankings: (runId) => request(`/api/runs/${runId}/deployment/rankings`),
  getRankingForScenario: (runId, scenario) =>
    request(`/api/runs/${runId}/deployment/rankings/${scenario}`),
  getDecisionGuide: (runId) =>
    request(`/api/runs/${runId}/deployment/decision-guide`),

  listFigures: (subfolder = "") =>
    request(`/api/figures${subfolder ? `?subfolder=${subfolder}` : ""}`),
  figureFileUrl: (relPath) => `${BASE}/api/figures/file/${relPath}`,

  listReportFiles: (runId) => request(`/api/runs/${runId}/reports/list`),
  reportDownloadUrl: (runId, filename) =>
    `${BASE}/api/runs/${runId}/reports/file/${filename}`,

  listModels: () => request("/api/registry/models"),
  listPerturbations: () => request("/api/registry/perturbations"),
  listScenarios: () => request("/api/registry/scenarios"),
};

export default api;
