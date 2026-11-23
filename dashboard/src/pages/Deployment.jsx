import { useState } from "react";
import { PageHeader, Panel, Spinner, EmptyState, ErrorState, ParadigmTag, Notice } from "../components/PageHeader";
import ScoreBar, { ScoreBarLegend } from "../components/ScoreBar";
import api from "../api/client";
import useApi from "../api/useApi";
import { fmt, isNum } from "../lib/format";

const label = (s) => String(s).replace(/_/g, " ");

export default function Deployment({ runId }) {
  const [chosen, setChosen] = useState(null);
  const rk = useApi(() => api.listRankings(runId), [runId], !!runId);
  // The decision guide is a bonus: if it fails the rankings still show.
  const dg = useApi(() => api.getDecisionGuide(runId).catch(() => null), [runId], !!runId);

  const head = <PageHeader eyebrow="FaceEval-X" title="Deployment Scoring" />;
  if (!runId) return <>{head}<Panel><EmptyState title="No run selected" /></Panel></>;
  if (rk.loading) return <>{head}<Panel><Spinner /></Panel></>;
  if (rk.error) return <>{head}<Panel><ErrorState title="Could not load rankings" error={rk.error} /></Panel></>;

  const rankings = rk.data || [];
  const scenario = rankings.some((r) => r.scenario === chosen) ? chosen : rankings[0]?.scenario;
  const active = rankings.find((r) => r.scenario === scenario);
  const guideEntry = dg.data?.scenarios?.[scenario];
  const rec = guideEntry?.recommended_model;

  return (
    <>
      <PageHeader
        eyebrow="Practitioner decision guide"
        title="Deployment Scoring"
        description="Context-weighted composite scores stored with this run. Switch scenarios to see how the ranking changes with the weights."
        actions={
          rankings.length > 0 && (
            <select className="inline-select" value={scenario || ""} onChange={(e) => setChosen(e.target.value)}>
              {rankings.map((r) => (
                <option key={r.scenario} value={r.scenario}>{label(r.scenario)}</option>
              ))}
            </select>
          )
        }
      />

      {!active ? (
        <Panel>
          <EmptyState
            title="No deployment rankings"
            description="This run has no stored deployment rankings."
          />
        </Panel>
      ) : (
        <>
          <Notice>
            Scores are weighted composites. The scorer sets any component it has no data for to a
            neutral 0.5, so a high score does not by itself show that a model was measured on every
            dimension. Treat this as a summary of the stored run, not an independent recommendation.
          </Notice>

          {rec && (
            <Panel title="Recommendation (generated text)" subtitle={guideEntry.description}>
              <div className="rec-row">
                <div>
                  <span className="rec-model">{rec.name}</span>
                  {rec.paradigm && <ParadigmTag paradigm={rec.paradigm} />}
                </div>
                <span className="num rec-score">{fmt(rec.total_score)}</span>
              </div>
              <p className="rec-text">{rec.recommendation_text}</p>
              {guideEntry.warnings?.length > 0 && (
                <div className="rec-warnings">
                  {guideEntry.warnings.map((w, i) => (
                    <div key={i} className="rec-warning">{w}</div>
                  ))}
                </div>
              )}
            </Panel>
          )}

          <Panel title="Ranked models" subtitle={`Weighted score breakdown — ${label(scenario)}`}>
            <div className="rank-list">
              {active.ranked_scores.map((s) => (
                <div key={s.model_name} className="rank-row">
                  <div className="rank-row-head">
                    <span className="rank-position">{isNum(s.rank) ? `#${s.rank}` : "—"}</span>
                    <span className="rank-model-name">{s.model_name}</span>
                    {s.paradigm && <ParadigmTag paradigm={s.paradigm} />}
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
                  <span className="weight-chip-label">{label(dim)}</span>
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
