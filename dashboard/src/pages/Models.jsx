import { useEffect, useState } from "react";
import { PageHeader, Panel, Spinner, EmptyState, ParadigmTag } from "../components/PageHeader";
import AdcChart from "../components/AdcChart";
import api from "../api/client";

export default function Models({ runId }) {
  const [results, setResults] = useState([]);
  const [perturbations, setPerturbations] = useState([]);
  const [selectedPt, setSelectedPt] = useState(null);
  const [adcSeries, setAdcSeries] = useState([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    if (!runId) {
      setLoading(false);
      return;
    }
    setLoading(true);
    api
      .listResults(runId)
      .then((r) => {
        setResults(r.results);
        const pts = [...new Set(r.results.map((x) => x.perturbation_type).filter(Boolean))];
        setPerturbations(pts);
        if (pts.length) setSelectedPt(pts[0]);
      })
      .finally(() => setLoading(false));
  }, [runId]);

  useEffect(() => {
    if (!runId || !selectedPt) return;
    const models = [...new Set(results.map((r) => r.model_name))];
    Promise.all(
      models.map((m) =>
        api
          .getAdcCurve(runId, m, selectedPt)
          .then((d) => ({ model_name: m, points: d.points }))
      )
    ).then(setAdcSeries);
  }, [runId, selectedPt, results]);

  if (!runId) {
    return (
      <>
        <PageHeader eyebrow="FaceEval-X" title="Model Comparison" />
        <Panel>
          <EmptyState title="No run selected" />
        </Panel>
      </>
    );
  }

  // Summarize: best clean accuracy per model
  const byModel = {};
  for (const r of results) {
    if (!byModel[r.model_name]) byModel[r.model_name] = { ...r, count: 0, accSum: 0 };
    byModel[r.model_name].count += 1;
    byModel[r.model_name].accSum += r.accuracy;
  }
  const rows = Object.values(byModel)
    .map((r) => ({ ...r, meanAccuracy: r.accSum / r.count }))
    .sort((a, b) => b.meanAccuracy - a.meanAccuracy);

  return (
    <>
      <PageHeader
        eyebrow="Benchmark"
        title="Model Comparison"
        description="Mean accuracy across all evaluated conditions, and accuracy-degradation behavior under one selected perturbation."
      />

      <Panel title="Mean accuracy by model" subtitle={`${results.length} (model × condition) evaluations`}>
        {loading ? (
          <Spinner />
        ) : rows.length === 0 ? (
          <EmptyState title="No evaluation results yet" />
        ) : (
          <table className="data-table">
            <thead>
              <tr>
                <th>Model</th>
                <th>Paradigm</th>
                <th>Mean accuracy</th>
                <th>AUC</th>
                <th>EER</th>
              </tr>
            </thead>
            <tbody>
              {rows.map((r) => (
                <tr key={r.model_name}>
                  <td>{r.model_name}</td>
                  <td>
                    <ParadigmTag paradigm={r.model_paradigm} />
                  </td>
                  <td className="num">{r.meanAccuracy.toFixed(4)}</td>
                  <td className="num">{r.auc.toFixed(4)}</td>
                  <td className="num">{r.eer.toFixed(4)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        )}
      </Panel>

      {perturbations.length > 0 && (
        <Panel
          title="Accuracy-Degradation Curve"
          subtitle="Accuracy vs. perturbation severity, by model"
          right={
            <select
              className="inline-select"
              value={selectedPt || ""}
              onChange={(e) => setSelectedPt(e.target.value)}
            >
              {perturbations.map((pt) => (
                <option key={pt} value={pt}>
                  {pt.replace(/_/g, " ")}
                </option>
              ))}
            </select>
          }
        >
          {adcSeries.length > 0 ? (
            <AdcChart series={adcSeries} perturbationLabel={selectedPt?.replace(/_/g, " ")} />
          ) : (
            <Spinner />
          )}
        </Panel>
      )}
    </>
  );
}
