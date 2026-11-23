import { useState } from "react";
import { PageHeader, Panel, Spinner, EmptyState, ErrorState, ParadigmTag, Notice } from "../components/PageHeader";
import RadarChart from "../components/RadarChart";
import api from "../api/client";
import useApi from "../api/useApi";
import { fmt } from "../lib/format";

export default function Fingerprint({ runId }) {
  const [picked, setPicked] = useState(null); // null = default (first three)
  const res = useApi(() => api.listFingerprints(runId), [runId], !!runId);

  const head = <PageHeader eyebrow="Robustness profile" title="Behavioral Fingerprint" />;
  if (!runId) return <>{head}<Panel><EmptyState title="No run selected" /></Panel></>;
  if (res.loading) return <>{head}<Panel><Spinner /></Panel></>;
  if (res.error) return <>{head}<Panel><ErrorState title="Could not load fingerprints" error={res.error} /></Panel></>;

  const fingerprints = res.data?.fingerprints || [];
  const skipped = res.data?.skipped || 0;
  const known = new Set(fingerprints.map((f) => f.model_name));
  const selected = (picked || fingerprints.slice(0, 3).map((f) => f.model_name)).filter((n) => known.has(n));
  const activeFps = fingerprints.filter((f) => selected.includes(f.model_name));

  const toggle = (name) =>
    setPicked(selected.includes(name) ? selected.filter((n) => n !== name) : [...selected, name]);

  if (fingerprints.length === 0) {
    return (
      <>
        {head}
        <Panel>
          <EmptyState
            title="No fingerprints in this run"
            description={
              skipped
                ? `${skipped} stored fingerprint(s) were malformed and cannot be shown.`
                : "This run has no stored behavioral fingerprints."
            }
          />
        </Panel>
      </>
    );
  }

  return (
    <>
      <PageHeader
        eyebrow="Robustness profile"
        title="Behavioral Fingerprint"
        description="Per-dimension scores stored with the run (robustness per perturbation type, calibration, fairness, speed, memory, sample efficiency). Select up to 5 models to overlay."
      />
      <Notice>
        The fingerprint builder stores 0.0 for any perturbation type it had no curve for, so a zero on a
        robustness axis can mean "not evaluated" rather than "failed".
        {skipped > 0 && ` ${skipped} malformed fingerprint(s) were not shown.`}
      </Notice>

      <div className="fp-layout">
        <Panel title="Models" className="fp-picker">
          <div className="fp-model-list">
            {fingerprints.map((fp) => (
              <label key={fp.model_name} className="fp-model-row">
                <input
                  type="checkbox"
                  checked={selected.includes(fp.model_name)}
                  onChange={() => toggle(fp.model_name)}
                  disabled={!selected.includes(fp.model_name) && selected.length >= 5}
                />
                <span>{fp.model_name}</span>
                {fp.paradigm && <ParadigmTag paradigm={fp.paradigm} />}
              </label>
            ))}
          </div>
        </Panel>

        <Panel title="Radar comparison" className="fp-radar-panel">
          {activeFps.length > 0 ? (
            <RadarChart fingerprints={activeFps} />
          ) : (
            <EmptyState title="Select at least one model" />
          )}
        </Panel>
      </div>

      {activeFps.length > 0 && (
        <Panel title="Dimension scores" subtitle="Raw fingerprint values, 0–1 scale">
          <table className="data-table">
            <thead>
              <tr>
                <th>Dimension</th>
                {activeFps.map((fp) => (
                  <th key={fp.model_name} className="num">{fp.model_name}</th>
                ))}
              </tr>
            </thead>
            <tbody>
              {[...new Set(activeFps.flatMap((fp) => fp.dimension_names))].map((dim) => (
                <tr key={dim}>
                  <td>{dim.replace(/_/g, " ")}</td>
                  {activeFps.map((fp) => {
                    const idx = fp.dimension_names.indexOf(dim);
                    return (
                      <td key={fp.model_name} className="num">
                        {idx >= 0 ? fmt(fp.vector[idx]) : "—"}
                      </td>
                    );
                  })}
                </tr>
              ))}
            </tbody>
          </table>
        </Panel>
      )}
    </>
  );
}
