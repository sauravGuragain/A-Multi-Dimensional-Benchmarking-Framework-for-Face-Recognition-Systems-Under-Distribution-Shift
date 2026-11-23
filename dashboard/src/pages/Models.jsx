import { useMemo, useState } from "react";
import { PageHeader, Panel, Spinner, EmptyState, ErrorState, ParadigmTag, Notice } from "../components/PageHeader";
import AdcChart from "../components/AdcChart";
import api from "../api/client";
import useApi from "../api/useApi";
import { fmt } from "../lib/format";
import { summarize, cleanCell, cleanTooltip } from "../lib/summary";

export default function Models({ runId }) {
  const [chosenPt, setChosenPt] = useState(null);
  const res = useApi(() => api.listResults(runId), [runId], !!runId);

  const results = res.data?.results || [];
  const perturbations = useMemo(
    () => [...new Set(results.map((x) => x.perturbation_type).filter(Boolean))],
    [results]
  );
  const models = useMemo(() => [...new Set(results.map((r) => r.model_name))], [results]);
  const selectedPt = perturbations.includes(chosenPt) ? chosenPt : perturbations[0] || null;
  const rows = useMemo(() => summarize(results), [results]);

  const adc = useApi(
    () =>
      Promise.all(
        models.map((m) =>
          api.getAdcCurve(runId, m, selectedPt).then((d) => ({ model_name: m, points: d.points }))
        )
      ),
    [runId, selectedPt, models.join("|")],
    !!runId && !!selectedPt && models.length > 0
  );

  const head = <PageHeader eyebrow="Benchmark" title="Model Comparison" />;
  if (!runId) return <>{head}<Panel><EmptyState title="No run selected" /></Panel></>;
  if (res.loading) return <>{head}<Panel><Spinner /></Panel></>;
  if (res.error) return <>{head}<Panel><ErrorState title="Could not load results" error={res.error} /></Panel></>;

  return (
    <>
      <PageHeader
        eyebrow="Benchmark"
        title="Model Comparison"
        description="Stored evaluation results for this run, and accuracy vs. severity for one perturbation."
      />

      <Panel title="Accuracy by model" subtitle={`${results.length} (model × condition) evaluations`}>
        {rows.length === 0 ? (
          <EmptyState
            title="No evaluation results in this run"
            description="This run has no stored evaluation results."
          />
        ) : (
          <>
            <Notice>
              Clean accuracy is the unperturbed condition: rows with no perturbation, severity 0.0 (an exact
              copy of the image for every perturbation type) and, for brightness and gamma only, severity 0.5
              (identity). Equivalent clean rows count once; if they disagree the range is shown. Contrast
              has no exact neutral severity, so it is not treated as clean at 0.5. "Mean (all)" averages
              every stored condition, clean and perturbed together (so the clean condition is counted in it
              once per stored clean row), and AUC/EER are means over the conditions where they were
              recorded. A dash means the value was not stored.
            </Notice>
            <table className="data-table">
              <thead>
                <tr>
                  <th>Model</th>
                  <th>Paradigm</th>
                  <th>Conditions</th>
                  <th>Clean accuracy</th>
                  <th>Mean accuracy (all)</th>
                  <th>Mean AUC</th>
                  <th>Mean EER</th>
                </tr>
              </thead>
              <tbody>
                {rows.map((r) => (
                  <tr key={r.model}>
                    <td>{r.model}</td>
                    <td><ParadigmTag paradigm={r.paradigm} /></td>
                    <td className="num">{r.conditions}</td>
                    <td className="num" title={cleanTooltip(r.clean)}>{cleanCell(r.clean)}</td>
                    <td className="num">{fmt(r.meanAccuracy, 4)}</td>
                    <td className="num">{fmt(r.meanAuc, 4)}</td>
                    <td className="num">{fmt(r.meanEer, 4)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </>
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
              onChange={(e) => setChosenPt(e.target.value)}
            >
              {perturbations.map((pt) => (
                <option key={pt} value={pt}>{pt.replace(/_/g, " ")}</option>
              ))}
            </select>
          }
        >
          {adc.loading ? (
            <Spinner />
          ) : adc.error ? (
            <ErrorState title="Could not load curve" error={adc.error} />
          ) : (adc.data || []).some((s) => s.points.length > 0) ? (
            <AdcChart
              series={adc.data.filter((s) => s.points.length > 0)}
              perturbationLabel={selectedPt?.replace(/_/g, " ")}
            />
          ) : (
            <EmptyState title="No complete points for this perturbation" />
          )}
        </Panel>
      )}
    </>
  );
}
