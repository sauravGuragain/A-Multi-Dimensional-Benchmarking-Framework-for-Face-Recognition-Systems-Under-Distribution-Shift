import { useEffect, useState } from "react";
import { PageHeader, Panel, Spinner, EmptyState, ParadigmTag } from "../components/PageHeader";
import RadarChart from "../components/RadarChart";
import api from "../api/client";

export default function Fingerprint({ runId }) {
  const [fingerprints, setFingerprints] = useState([]);
  const [selected, setSelected] = useState([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    if (!runId) {
      setLoading(false);
      return;
    }
    setLoading(true);
    api
      .listFingerprints(runId)
      .then((r) => {
        setFingerprints(r.fingerprints);
        setSelected(r.fingerprints.slice(0, 3).map((f) => f.model_name));
      })
      .finally(() => setLoading(false));
  }, [runId]);

  if (!runId) {
    return (
      <>
        <PageHeader eyebrow="FaceEval-X" title="Behavioral Fingerprint" />
        <Panel>
          <EmptyState title="No run selected" />
        </Panel>
      </>
    );
  }

  const toggle = (name) => {
    setSelected((prev) =>
      prev.includes(name) ? prev.filter((n) => n !== name) : [...prev, name]
    );
  };

  const activeFps = fingerprints.filter((f) => selected.includes(f.model_name));

  return (
    <>
      <PageHeader
        eyebrow="Robustness profile"
        title="Behavioral Fingerprint"
        description="Multi-dimensional comparison across robustness, calibration, fairness, speed, and memory. Select up to 5 models to overlay."
      />

      <div className="fp-layout">
        <Panel title="Models" className="fp-picker">
          {loading ? (
            <Spinner />
          ) : (
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
                  <ParadigmTag paradigm={fp.paradigm} />
                </label>
              ))}
            </div>
          )}
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
                  <th key={fp.model_name} className="num">
                    {fp.model_name}
                  </th>
                ))}
              </tr>
            </thead>
            <tbody>
              {activeFps[0].dimension_names.map((dim, di) => (
                <tr key={dim}>
                  <td>{dim.replace(/_/g, " ")}</td>
                  {activeFps.map((fp) => {
                    const idx = fp.dimension_names.indexOf(dim);
                    const v = idx >= 0 ? fp.vector[idx] : null;
                    return (
                      <td key={fp.model_name} className="num">
                        {v !== null ? v.toFixed(3) : "—"}
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
