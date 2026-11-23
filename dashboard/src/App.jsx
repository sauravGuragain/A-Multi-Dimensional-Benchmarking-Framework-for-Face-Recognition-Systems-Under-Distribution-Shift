import { useEffect, useState } from "react";
import { BrowserRouter, Routes, Route } from "react-router-dom";
import AppShell from "./components/AppShell";
import Overview from "./pages/Overview";
import Models from "./pages/Models";
import Fingerprint from "./pages/Fingerprint";
import Deployment from "./pages/Deployment";
import Figures from "./pages/Figures";
import { Spinner } from "./components/PageHeader";
import api from "./api/client";
import "./pages/Pages.css";

export default function App() {
  const [runs, setRuns] = useState([]);
  const [totalRuns, setTotalRuns] = useState(0);
  // "loading" | "ok" | { error: string }
  const [conn, setConn] = useState("loading");
  const [runId, setRunId] = useState(
    () => localStorage.getItem("faceevalx.runId") || ""
  );

  useEffect(() => {
    api
      .listRuns({ limit: 50 })
      .then((r) => {
        setConn("ok");
        setRuns(r.runs);
        setTotalRuns(typeof r.total === "number" ? r.total : r.runs.length);
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
      .catch((e) => {
        // Show the reason (API down, database missing, ...) instead of an
        // empty dashboard that looks like "no runs yet".
        setRuns([]);
        setTotalRuns(0);
        setConn({ error: e.message, status: e.status });
      });
  }, []);

  const handleSelectRun = (id) => {
    setRunId(id);
    localStorage.setItem("faceevalx.runId", id);
  };

  // Pages only receive a run id once the run list has loaded successfully, so a stale
  // id from localStorage is never requested; they are keyed by run id so that any
  // page-local selection (perturbation, scenario, models, lightbox) resets on a run switch.
  const activeRunId = conn === "ok" ? runId : "";

  return (
    <BrowserRouter>
      <AppShell runId={runId} runs={runs} totalRuns={totalRuns} conn={conn} onSelectRun={handleSelectRun}>
        {conn === "loading" ? (
          <Spinner />
        ) : (
          <Routes>
            <Route path="/" element={<Overview key={activeRunId} runId={activeRunId} unavailable={conn !== "ok"} />} />
            <Route path="/models" element={<Models key={activeRunId} runId={activeRunId} />} />
            <Route path="/fingerprint" element={<Fingerprint key={activeRunId} runId={activeRunId} />} />
            <Route path="/deployment" element={<Deployment key={activeRunId} runId={activeRunId} />} />
            <Route path="/figures" element={<Figures key={activeRunId} runId={activeRunId} />} />
          </Routes>
        )}
      </AppShell>
    </BrowserRouter>
  );
}
