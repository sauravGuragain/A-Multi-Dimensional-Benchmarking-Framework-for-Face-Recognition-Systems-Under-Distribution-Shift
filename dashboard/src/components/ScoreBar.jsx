import { useState } from "react";
import "./ScoreBar.css";

// Mirrors faceeval.visualization.style OKABE_ITO mapping used in the
// matplotlib figures, so the dashboard and the thesis PDF/PNG figures
// read as the same visual language.
const DIM_COLORS = {
  accuracy: "#5b8def",
  robustness: "#3dd68c",
  calibration: "#9b7bd6",
  fairness: "#f0a742",
  latency: "#e5484d",
  memory: "#4fb8c9",
  computational_cost: "#8893a8",
};

const DIM_LABELS = {
  accuracy: "Accuracy",
  robustness: "Robustness",
  calibration: "Calibration",
  fairness: "Fairness",
  latency: "Latency",
  memory: "Memory",
  computational_cost: "Comp. cost",
};

/**
 * Horizontal stacked bar showing the weighted-component breakdown of a
 * single deployment score. Hovering a segment reveals its raw component
 * score, weight, and contribution — the explainability artifact described
 * in the thesis Chapter 12 deployment scoring engine.
 */
export default function ScoreBar({ score, maxScore = 1.0, compact = false }) {
  const [hovered, setHovered] = useState(null);
  const dims = Object.keys(score.weighted_components);

  return (
    <div className={`scorebar ${compact ? "scorebar-compact" : ""}`}>
      <div className="scorebar-row">
        <div className="scorebar-track">
          {dims.map((dim) => {
            const weighted = score.weighted_components[dim] || 0;
            const widthPct = (weighted / maxScore) * 100;
            return (
              <div
                key={dim}
                className="scorebar-segment"
                style={{
                  width: `${widthPct}%`,
                  background: DIM_COLORS[dim] || "#666",
                  opacity: hovered && hovered !== dim ? 0.35 : 1,
                }}
                onMouseEnter={() => setHovered(dim)}
                onMouseLeave={() => setHovered(null)}
              />
            );
          })}
        </div>
        <span className="scorebar-total num">
          {score.total_score.toFixed(3)}
        </span>
      </div>

      {hovered && (
        <div className="scorebar-tooltip">
          <span
            className="scorebar-tooltip-dot"
            style={{ background: DIM_COLORS[hovered] }}
          />
          <span className="scorebar-tooltip-label">
            {DIM_LABELS[hovered] || hovered}
          </span>
          <span className="scorebar-tooltip-detail num">
            score {score.component_scores[hovered]?.toFixed(3)} × weight{" "}
            {(score.weighted_components[hovered] / score.component_scores[hovered] || 0).toFixed(2)}{" "}
            = {score.weighted_components[hovered]?.toFixed(3)}
          </span>
        </div>
      )}
    </div>
  );
}

export function ScoreBarLegend() {
  return (
    <div className="scorebar-legend">
      {Object.entries(DIM_LABELS).map(([dim, label]) => (
        <span key={dim} className="scorebar-legend-item">
          <span
            className="scorebar-legend-dot"
            style={{ background: DIM_COLORS[dim] }}
          />
          {label}
        </span>
      ))}
    </div>
  );
}
