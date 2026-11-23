import { PageHeader, Panel, Stat, Spinner, EmptyState, ErrorState, Badge, Notice } from "../components/PageHeader";
import api from "../api/client";
import useApi from "../api/useApi";
import { fmt } from "../lib/format";

const list = (v) => (Array.isArray(v) ? v : []);
const text = (v) => (v === undefined || v === null || v === "" ? "—" : String(v));

export default function Overview({ runId, unavailable = false }) {
  const run = useApi(() => api.getRun(runId), [runId], !!runId);
  // Rankings are optional on this page: a failure just hides the panel.
  const rk = useApi(() => api.listRankings(runId).catch(() => []), [runId], !!runId);

  const head = <PageHeader eyebrow="FaceEval-X" title="Overview" />;

  if (!runId) {
    return (
      <>
        <PageHeader eyebrow="FaceEval-X" title="Overview" description="No experiment run selected." />
        <Panel>
          {unavailable ? (
            <EmptyState
              title="Run list unavailable"
              description="The run list could not be loaded; see the error above. This does not mean there are no runs."
            />
          ) : (
            <EmptyState
              title="No runs available"
              description="The results database has no runs yet. Execute an experiment with faceeval.orchestration.ExperimentRunner, then reload this page."
            />
          )}
        </Panel>
      </>
    );
  }
  if (run.loading) return <>{head}<Panel><Spinner /></Panel></>;
  if (run.error) return <>{head}<Panel><ErrorState title="Could not load run" error={run.error} /></Panel></>;

  const r = run.data;
  const cfg = r.config || {};
  const trad = list(cfg.traditional_models);
  const deep = list(cfg.deep_models);
  const rankings = list(rk.data);
  const topModel = rankings[0]?.ranked_scores?.[0];

  return (
    <>
      <PageHeader
        eyebrow={cfg.experiment_name || "Experiment"}
        title="Experiment Overview"
        description={`Run ${r.run_id.slice(0, 12)} · status ${r.status}`}
        actions={<Badge tone={r.status === "completed" ? "good" : "neutral"}>{r.status}</Badge>}
      />

      {r.error_message && (
        <Panel><ErrorState title="Run reported an error" error={r.error_message} /></Panel>
      )}

      <div className="stat-grid">
        <Panel><Stat label="Models configured" value={trad.length + deep.length} /></Panel>
        <Panel><Stat label="Evaluation conditions" value={r.num_evaluation_results} /></Panel>
        <Panel><Stat label="Fingerprints" value={r.num_fingerprints} /></Panel>
        <Panel>
          <Stat label="Duration" value={fmt(r.duration_seconds, 1)} unit={r.duration_seconds != null ? "s" : undefined} />
        </Panel>
      </div>

      <Panel title="Configuration" subtitle="Reproducibility-relevant parameters, as stored with the run">
        <div className="config-grid">
          <ConfigRow label="Dataset(s)" value={list(cfg.dataset_names).join(", ") || "—"} />
          <ConfigRow label="Split strategy" value={text(cfg.split_strategy)} />
          <ConfigRow label="Random seed" value={text(cfg.random_seed)} />
          <ConfigRow label="Traditional models" value={trad.join(", ") || "—"} />
          <ConfigRow label="Deep learning models" value={deep.join(", ") || "—"} />
          <ConfigRow label="Config hash" value={cfg.config_hash ? String(cfg.config_hash).slice(0, 16) : "—"} mono />
        </div>
      </Panel>

      {topModel && (
        <Panel title="Highest composite deployment score" subtitle={`Scenario: ${String(rankings[0].scenario).replace(/_/g, " ")}`}>
          <Notice>
            A weighted composite. Any component with no data is set to a neutral 0.5 by the scorer, so
            this is a summary of the stored run, not a verified recommendation.
          </Notice>
          <div className="top-model-row">
            <span className="top-model-name">{topModel.model_name}</span>
            <span className="num top-model-score">{fmt(topModel.total_score)}</span>
            <span className="top-model-rec">{topModel.recommendation}</span>
          </div>
        </Panel>
      )}
    </>
  );
}

function ConfigRow({ label, value, mono }) {
  return (
    <div className="config-row">
      <span className="config-row-label">{label}</span>
      <span className={`config-row-value ${mono ? "num" : ""}`}>{value}</span>
    </div>
  );
}
