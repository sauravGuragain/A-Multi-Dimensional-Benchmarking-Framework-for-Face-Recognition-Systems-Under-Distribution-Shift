import { NavLink } from "react-router-dom";
import "./AppShell.css";

const NAV_ITEMS = [
  { to: "/", label: "Overview", glyph: "◧" },
  { to: "/models", label: "Model Comparison", glyph: "◫" },
  { to: "/fingerprint", label: "Fingerprint", glyph: "◈" },
  { to: "/deployment", label: "Deployment", glyph: "▤" },
  { to: "/figures", label: "Figures", glyph: "▥" },
];

export default function AppShell({ runId, runs, totalRuns = 0, conn = "ok", onSelectRun, children }) {
  const failed = typeof conn === "object" && conn !== null;
  return (
    <div className="shell">
      <aside className="rail">
        <div className="rail-brand">
          <span className="rail-brand-mark">FX</span>
          <div className="rail-brand-text">
            <span className="rail-brand-name">FaceEval-X</span>
            <span className="rail-brand-sub">benchmark console</span>
          </div>
        </div>

        <div className="rail-run-picker">
          <label htmlFor="run-select" className="rail-label">
            Run
          </label>
          <select
            id="run-select"
            className="rail-select"
            value={runId || ""}
            onChange={(e) => onSelectRun(e.target.value)}
          >
            {runs.length === 0 && <option value="">{failed ? "Unavailable" : "No runs yet"}</option>}
            {runs.map((r) => (
              <option key={r.run_id} value={r.run_id}>
                {r.experiment_name} — {r.run_id.slice(0, 8)}
              </option>
            ))}
          </select>
          {totalRuns > runs.length && (
            <div className="rail-label" role="status">
              Showing the {runs.length} most recent of {totalRuns} runs
            </div>
          )}
        </div>

        <nav className="rail-nav">
          {NAV_ITEMS.map((item) => (
            <NavLink
              key={item.to}
              to={item.to}
              end={item.to === "/"}
              className={({ isActive }) =>
                "rail-link" + (isActive ? " rail-link-active" : "")
              }
            >
              <span className="rail-glyph">{item.glyph}</span>
              {item.label}
            </NavLink>
          ))}
        </nav>

        <div className="rail-foot">
          <span className="rail-foot-dot" style={failed ? { background: "#e5484d" } : undefined} />
          {failed
            ? conn.status === 503
              ? "API up · database unavailable"
              : "API unreachable"
            : conn === "loading"
            ? "Connecting…"
            : "API connected · read-only"}
        </div>
      </aside>

      <main className="main scroll-region">
        {failed && (
          <div className="empty-state empty-state-error" role="alert" style={{ marginBottom: 16 }}>
            <h3>{conn.status === 503 ? "Results database unavailable" : "Cannot load runs"}</h3>
            <p>{conn.error}</p>
          </div>
        )}
        {children}
      </main>
    </div>
  );
}
