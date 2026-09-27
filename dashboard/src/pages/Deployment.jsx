import { useEffect, useState } from "react";
import { PageHeader, Panel, Spinner, EmptyState, ParadigmTag } from "../components/PageHeader";
import ScoreBar, { ScoreBarLegend } from "../components/ScoreBar";
import api from "../api/client";

export default function Deployment({ runId }) {
  const [rankings, setRankings] = useState([]);
  const [scenario, setScenario] = useState(null);
  const [guide, setGuide] = useState(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    if (!runId) {
      setLoading(false);
      return;
    }
    setLoading(true);
    Promise.all([
      api.listRankings(runId),
      api.getDecisionGuide(runId).catch(() => null),
    ])
      .then(([rk, dg]) => {
        setRankings(rk);
        setGuide(dg);
        if (rk.length) setScenario(rk[0].scenario);
      })
      .finally(() => setLoading(false));
  }, [runId]);

  if (!runId) {
    return (
      <>
        <PageHeader eyebrow="FaceEval-X" title="Deployment Scoring" />
        <Panel>
          <EmptyState title="No run selected" />
        </Panel>
      </>
    );
  }

  if (loading) {
    return (
      <>
        <PageHeader eyebrow="FaceEval-X" title="Deployment Scoring" />
        <Panel>
          <Spinner />
        </Panel>
      </>
    );
  }

  const active = rankings.find((r) => r.scenario === scenario);
  const guideEntry = guide?.scenarios?.[scenario];

  return (
    <>
      <PageHeader
        eyebrow="Practitioner decision guide"
        title="Deployment Scoring"
        description="Context-weighted composite scores. Switch scenarios to see how the ranking changes with deployment priorities."
        actions={
          rankings.length > 0 && (
            <select
              className="inline-select"
              value={scenario || ""}
              onChange={(e) => setScenario(e.target.value)}
            >
              {rankings.map((r) => (
                <option key={r.scenario} value={r.scenario}>
                  {r.scenario.replace(/_/g, " ")}
                </option>
              ))}
            </select>
          )
        }
      />

      {!active ? (
        <Panel>
          <EmptyState
            title="No deployment rankings"
            description="Run the deployment scorer for this experiment to populate this view."
          />
        </Panel>
      ) : (
        <>
          {guideEntry && (
            <Panel
              title="Recommendation"
              subtitle={guideEntry.description}
            >
              <div className="rec-row">
                <div>
                  <span className="rec-model">{guideEntry.recommended_model.name}</span>
                  <ParadigmTag paradigm={guideEntry.recommended_model.paradigm} />
                </div>
                <span className="num rec-score">
                  {guideEntry.recommended_model.total_score.toFixed(3)}
                </span>
              </div>
              <p className="rec-text">{guideEntry.recommended_model.recommendation_text}</p>
              {guideEntry.warnings?.length > 0 && (
                <div className="rec-warnings">
                  {guideEntry.warnings.map((w, i) => (
                    <div key={i} className="rec-warning">
                      {w}
                    </div>
                  ))}
                </div>
              )}
            </Panel>
          )}

          <Panel
            title="Ranked models"
            subtitle={`Weighted score breakdown — ${scenario.replace(/_/g, " ")}`}
          >
            <div className="rank-list">
              {active.ranked_scores.map((s) => (
                <div key={s.model_name} className="rank-row">
                  <div className="rank-row-head">
                    <span className="rank-position">#{s.rank}</span>
                    <span className="rank-model-name">{s.model_name}</span>
                    <ParadigmTag paradigm={s.paradigm} />
                  </div>
                  <ScoreBar score={s} />
                </div>
              ))}
            </div>
            <ScoreBarLegend />
          </Panel>

          <Panel title="Scenario weights" subtitle="Importance assigned to each dimension for this scenario">
            <div className="weights-grid">
              {Object.entries(active.weights).map(([dim, w]) => (
                <div key={dim} className="weight-chip">
                  <span className="weight-chip-label">{dim.replace(/_/g, " ")}</span>
                  <span className="num weight-chip-value">{(w * 100).toFixed(0)}%</span>
                </div>
              ))}
            </div>
          </Panel>
        </>
      )}
    </>
  );
}
