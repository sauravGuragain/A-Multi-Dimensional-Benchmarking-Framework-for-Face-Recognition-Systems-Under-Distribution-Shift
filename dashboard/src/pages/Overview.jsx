import { useEffect, useState } from "react";
import { PageHeader, Panel, Stat, Spinner, EmptyState, Badge } from "../components/PageHeader";
import api from "../api/client";

export default function Overview({ runId }) {
  const [run, setRun] = useState(null);
  const [rankings, setRankings] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  useEffect(() => {
    if (!runId) {
      setLoading(false);
      return;
    }
    setLoading(true);
    setError(null);
    Promise.all([api.getRun(runId), api.listRankings(runId).catch(() => [])])
      .then(([r, rk]) => {
        setRun(r);
        setRankings(rk);
      })
      .catch((e) => setError(e.message))
      .finally(() => setLoading(false));
  }, [runId]);

  if (!runId) {
    return (
      <>
        <PageHeader
          eyebrow="FaceEval-X"
          title="Overview"
          description="No experiment run selected."
        />
        <Panel>
          <EmptyState
            title="No runs available"
            description="Execute an experiment with faceeval.orchestration.ExperimentRunner, then refresh this dashboard to inspect results."
          />
        </Panel>
      </>
    );
  }

  if (loading) {
    return (
      <>
        <PageHeader eyebrow="FaceEval-X" title="Overview" />
        <Panel>
          <Spinner />
        </Panel>
      </>
    );
  }

  if (error) {
    return (
      <>
        <PageHeader eyebrow="FaceEval-X" title="Overview" />
        <Panel>
          <EmptyState title="Could not load run" description={error} />
        </Panel>
      </>
    );
  }

  if (!run) {
    return (
      <>
        <PageHeader eyebrow="FaceEval-X" title="Overview" />
        <Panel>
          <EmptyState
            title="Run data unavailable"
            description="The selected run could not be loaded. Try selecting it again from the run picker."
          />
        </Panel>
      </>
    );
  }

  const topModel = rankings[0]?.ranked_scores?.[0];

  return (
    <>
      <PageHeader
        eyebrow={run.config.experiment_name}
        title="Experiment Overview"
        description={`Run ${run.run_id.slice(0, 12)} · status ${run.status}`}
        actions={<Badge tone={run.status === "completed" ? "good" : "neutral"}>{run.status}</Badge>}
      />

      <div className="stat-grid">
        <Panel>
          <Stat label="Models evaluated" value={run.config.traditional_models.length + run.config.deep_models.length} />
        </Panel>
        <Panel>
          <Stat label="Evaluation conditions" value={run.num_evaluation_results} />
        </Panel>
        <Panel>
          <Stat label="Fingerprints" value={run.num_fingerprints} />
        </Panel>
        <Panel>
          <Stat
            label="Duration"
            value={run.duration_seconds ? run.duration_seconds.toFixed(1) : "—"}
            unit="s"
          />
        </Panel>
      </div>

      <Panel title="Configuration" subtitle="Reproducibility-relevant parameters">
        <div className="config-grid">
          <ConfigRow label="Dataset(s)" value={run.config.dataset_names.join(", ") || "—"} />
          <ConfigRow label="Split strategy" value={run.config.split_strategy} />
          <ConfigRow label="Random seed" value={run.config.random_seed} />
          <ConfigRow
            label="Traditional models"
            value={run.config.traditional_models.join(", ") || "—"}
          />
          <ConfigRow label="Deep learning models" value={run.config.deep_models.join(", ") || "—"} />
          <ConfigRow label="Config hash" value={run.config.config_hash?.slice(0, 16)} mono />
        </div>
      </Panel>

      {topModel && (
        <Panel
          title="Top-ranked model"
          subtitle={`Best overall for ${rankings[0].scenario.replace("_", " ")}`}
        >
          <div className="top-model-row">
            <span className="top-model-name">{topModel.model_name}</span>
            <span className="num top-model-score">{topModel.total_score.toFixed(3)}</span>
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
