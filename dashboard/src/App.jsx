import { useEffect, useState } from "react";
import { BrowserRouter, Routes, Route } from "react-router-dom";
import AppShell from "./components/AppShell";
import Overview from "./pages/Overview";
import Models from "./pages/Models";
import Fingerprint from "./pages/Fingerprint";
import Deployment from "./pages/Deployment";
import Figures from "./pages/Figures";
import api from "./api/client";
import "./pages/Pages.css";

export default function App() {
  const [runs, setRuns] = useState([]);
  const [runId, setRunId] = useState(
    () => localStorage.getItem("faceevalx.runId") || ""
  );

  useEffect(() => {
    api
      .listRuns({ limit: 50 })
      .then((r) => {
        setRuns(r.runs);
        const validIds = new Set(r.runs.map((run) => run.run_id));
        if ((!runId || !validIds.has(runId)) && r.runs.length) {
          // No cached run, or the cached run_id no longer exists on the
          // backend (e.g. stale localStorage from a previous database) —
          // fall back to the most recent available run instead of
          // leaving the UI pointed at a 404.
          setRunId(r.runs[0].run_id);
          localStorage.setItem("faceevalx.runId", r.runs[0].run_id);
        } else if (!r.runs.length) {
          setRunId("");
          localStorage.removeItem("faceevalx.runId");
        }
      })
      .catch(() => setRuns([]));
  }, []);

  const handleSelectRun = (id) => {
    setRunId(id);
    localStorage.setItem("faceevalx.runId", id);
  };

  return (
    <BrowserRouter>
      <AppShell runId={runId} runs={runs} onSelectRun={handleSelectRun}>
        <Routes>
          <Route path="/" element={<Overview runId={runId} />} />
          <Route path="/models" element={<Models runId={runId} />} />
          <Route path="/fingerprint" element={<Fingerprint runId={runId} />} />
          <Route path="/deployment" element={<Deployment runId={runId} />} />
          <Route path="/figures" element={<Figures runId={runId} />} />
        </Routes>
      </AppShell>
    </BrowserRouter>
  );
}
