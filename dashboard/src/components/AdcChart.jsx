import "./AdcChart.css";

const W = 480;
const H = 240;
const PAD = { top: 16, right: 16, bottom: 32, left: 40 };
const plotW = W - PAD.left - PAD.right;
const plotH = H - PAD.top - PAD.bottom;

const COLORS = ["#5b8def", "#f0a742", "#3dd68c", "#9b7bd6", "#e5484d"];

const x = (s) => PAD.left + s * plotW;
const y = (a) => PAD.top + (1 - a) * plotH;

/**
 * Accuracy-Degradation Curve chart: accuracy (y) vs. perturbation severity
 * (x) for one or more models. Mirrors faceeval.visualization.roc.plot_adc_curves.
 */
export default function AdcChart({ series, perturbationLabel }) {
  if (!series?.length) return null;

  return (
    <div className="adc-wrap">
      <svg viewBox={`0 0 ${W} ${H}`} className="adc-svg">
        {/* y gridlines */}
        {[0, 0.2, 0.4, 0.6, 0.8, 1.0].map((v) => (
          <g key={v}>
            <line
              x1={PAD.left}
              x2={W - PAD.right}
              y1={y(v)}
              y2={y(v)}
              className="adc-grid"
            />
            <text x={PAD.left - 8} y={y(v)} className="adc-axis-label" textAnchor="end" dominantBaseline="middle">
              {v.toFixed(1)}
            </text>
          </g>
        ))}
        {/* x gridlines */}
        {[0, 0.2, 0.4, 0.6, 0.8, 1.0].map((v) => (
          <text
            key={v}
            x={x(v)}
            y={H - PAD.bottom + 18}
            className="adc-axis-label"
            textAnchor="middle"
          >
            {v.toFixed(1)}
          </text>
        ))}

        {/* axes */}
        <line x1={PAD.left} y1={PAD.top} x2={PAD.left} y2={H - PAD.bottom} className="adc-axis" />
        <line x1={PAD.left} y1={H - PAD.bottom} x2={W - PAD.right} y2={H - PAD.bottom} className="adc-axis" />

        {/* curves */}
        {series.map((s, si) => {
          const color = COLORS[si % COLORS.length];
          const pts = s.points
            .slice()
            .sort((a, b) => a.severity - b.severity)
            .map((p) => `${x(p.severity)},${y(p.accuracy)}`)
            .join(" ");
          return (
            <g key={s.model_name}>
              <polyline points={pts} fill="none" stroke={color} strokeWidth={2} />
              {s.points.map((p, pi) => (
                <circle
                  key={pi}
                  cx={x(p.severity)}
                  cy={y(p.accuracy)}
                  r={3}
                  fill={color}
                  stroke="var(--ink-950)"
                  strokeWidth={1}
                />
              ))}
            </g>
          );
        })}

        <text x={(W) / 2} y={H - 2} className="adc-axis-title" textAnchor="middle">
          severity{perturbationLabel ? ` — ${perturbationLabel}` : ""}
        </text>
        <text
          x={-H / 2}
          y={12}
          className="adc-axis-title"
          textAnchor="middle"
          transform="rotate(-90)"
        >
          accuracy
        </text>
      </svg>

      <div className="adc-legend">
        {series.map((s, si) => (
          <span key={s.model_name} className="adc-legend-item">
            <span
              className="adc-legend-dot"
              style={{ background: COLORS[si % COLORS.length] }}
            />
            {s.model_name}
          </span>
        ))}
      </div>
    </div>
  );
}
