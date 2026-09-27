import "./RadarChart.css";

const SIZE = 360;
const CENTER = SIZE / 2;
const MAX_R = 130;
const RINGS = [0.2, 0.4, 0.6, 0.8, 1.0];

function point(angle, value) {
  const r = value * MAX_R;
  return [CENTER + r * Math.sin(angle), CENTER - r * Math.cos(angle)];
}

function shortLabel(dim) {
  const map = {
    robustness_gaussian_blur: "Blur",
    robustness_motion_blur: "Motion",
    robustness_gaussian_noise: "Noise",
    robustness_salt_pepper_noise: "S&P",
    robustness_speckle_noise: "Speckle",
    robustness_brightness: "Bright",
    robustness_contrast: "Contrast",
    robustness_gamma: "Gamma",
    robustness_rotation: "Rotate",
    robustness_scaling: "Scale",
    robustness_cropping: "Crop",
    robustness_random_occlusion: "Occl.",
    robustness_face_mask: "Mask",
    robustness_sunglasses: "Glasses",
    robustness_jpeg_compression: "JPEG",
    robustness_resolution_degradation: "Res ↓",
    calibration_quality: "Calib.",
    fairness_gap: "Fair",
    speed_score: "Speed",
    memory_efficiency: "Memory",
    sample_efficiency: "Sample Eff.",
  };
  return map[dim] || dim.replace("robustness_", "").replace(/_/g, " ");
}

const COLORS = ["#5b8def", "#f0a742", "#3dd68c", "#9b7bd6", "#e5484d"];

/**
 * Overlaid radar chart for comparing behavioral fingerprints across models.
 * Renders as raw SVG so the visual language matches the matplotlib
 * faceeval.visualization.radar output used in the thesis figures.
 */
export default function RadarChart({ fingerprints, dimensions }) {
  if (!fingerprints?.length) return null;

  const dims = dimensions || fingerprints[0].dimension_names.slice(0, 13);
  const n = dims.length;
  const angleStep = (2 * Math.PI) / n;

  return (
    <div className="radar-wrap">
      <svg viewBox={`0 0 ${SIZE} ${SIZE}`} className="radar-svg">
        {/* grid rings */}
        {RINGS.map((r) => (
          <polygon
            key={r}
            points={dims
              .map((_, i) => point(i * angleStep, r).join(","))
              .join(" ")}
            className="radar-ring"
          />
        ))}
        {/* spokes */}
        {dims.map((_, i) => {
          const [x, y] = point(i * angleStep, 1);
          return (
            <line
              key={i}
              x1={CENTER}
              y1={CENTER}
              x2={x}
              y2={y}
              className="radar-spoke"
            />
          );
        })}
        {/* data polygons */}
        {fingerprints.map((fp, fi) => {
          const dimIdx = Object.fromEntries(
            fp.dimension_names.map((d, i) => [d, i])
          );
          const pts = dims
            .map((d, i) => {
              const v = dimIdx[d] !== undefined ? fp.vector[dimIdx[d]] : 0;
              return point(i * angleStep, Math.max(0, Math.min(1, v))).join(",");
            })
            .join(" ");
          const color = COLORS[fi % COLORS.length];
          return (
            <g key={fp.model_name}>
              <polygon
                points={pts}
                fill={color}
                fillOpacity={0.1}
                stroke={color}
                strokeWidth={2}
                strokeDasharray={fp.paradigm === "deep_learning" ? "5,3" : "0"}
              />
            </g>
          );
        })}
        {/* labels */}
        {dims.map((d, i) => {
          const [x, y] = point(i * angleStep, 1.22);
          return (
            <text
              key={d}
              x={x}
              y={y}
              textAnchor="middle"
              dominantBaseline="middle"
              className="radar-label"
            >
              {shortLabel(d)}
            </text>
          );
        })}
      </svg>

      <div className="radar-legend">
        {fingerprints.map((fp, fi) => (
          <span key={fp.model_name} className="radar-legend-item">
            <span
              className="radar-legend-swatch"
              style={{
                background: COLORS[fi % COLORS.length],
                borderStyle: fp.paradigm === "deep_learning" ? "dashed" : "solid",
              }}
            />
            {fp.model_name}
          </span>
        ))}
      </div>
    </div>
  );
}
