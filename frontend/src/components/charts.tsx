// Small SVG chart primitives. Deliberately not a charting library: every
// chart in this app is a line or a bar group over at most ~100 points,
// the drift chart was already hand-rolled SVG in exactly this shape, and
// a dependency would add more bundle than the whole feature. Colors come
// from theme.ts's status palette so charts match the rest of the app
// rather than a library's defaults.
import { STATUS_HEX, type Status, typeScale } from "../theme";

const AXIS_COLOR = "#6b6d72";
// Matches --color-app-bg; used only as a text halo so labels stay legible
// where a line crosses them.
const PAGE_BG = "#121316";

export interface LinePoint {
  x: number;
  y: number;
}

export interface ReferenceLine {
  x: number;
  label: string;
  status: Status;
}

function scaler(min: number, max: number, from: number, to: number) {
  const span = max - min || 1;
  return (value: number) => from + ((value - min) / span) * (to - from);
}

export function LineChart({
  points,
  referenceLines = [],
  status = "neutral",
  ariaLabel,
  formatY,
  height = 180,
  width = 560,
  padding = 34,
}: {
  points: LinePoint[];
  referenceLines?: ReferenceLine[];
  status?: Status;
  ariaLabel: string;
  formatY?: (value: number) => string;
  height?: number;
  width?: number;
  padding?: number;
}) {
  if (points.length === 0) return null;

  const xs = points.map((p) => p.x);
  const ys = points.map((p) => p.y);
  const xMin = Math.min(...xs);
  const xMax = Math.max(...xs);
  const yMin = Math.min(...ys);
  const yMax = Math.max(...ys);

  const toX = scaler(xMin, xMax, padding, width - padding);
  const toY = scaler(yMin, yMax, height - padding, padding);

  const minPoint = points[ys.indexOf(yMin)];

  return (
    <svg viewBox={`0 0 ${width} ${height}`} className="w-full" role="img" aria-label={ariaLabel}>
      <line
        x1={padding}
        y1={height - padding}
        x2={width - padding}
        y2={height - padding}
        stroke={AXIS_COLOR}
        strokeWidth={1}
      />

      {referenceLines.map((ref) => (
        <g key={ref.label}>
          <line
            x1={toX(ref.x)}
            y1={padding - 6}
            x2={toX(ref.x)}
            y2={height - padding}
            stroke={STATUS_HEX[ref.status]}
            strokeWidth={1}
            strokeDasharray="3 3"
          />
          <text
            x={toX(ref.x)}
            y={padding - 10}
            fontSize={9}
            fill={STATUS_HEX[ref.status]}
            textAnchor="middle"
          >
            {ref.label}
          </text>
        </g>
      ))}

      <polyline
        points={points.map((p) => `${toX(p.x).toFixed(1)},${toY(p.y).toFixed(1)}`).join(" ")}
        fill="none"
        stroke={status === "neutral" ? "#8fa3ad" : STATUS_HEX[status]}
        strokeWidth={2}
      />

      {/* The minimum is the whole point of a cost curve — mark it. The
          label anchors inward at the edges so it never collides with the
          axis labels or clips outside the viewBox. */}
      <circle cx={toX(minPoint.x)} cy={toY(minPoint.y)} r={3.5} fill={STATUS_HEX.success} />
      <text
        x={toX(minPoint.x)}
        y={toY(minPoint.y) - 9}
        fontSize={9}
        fill={STATUS_HEX.success}
        // Halo, so a reference line crossing the label doesn't cut through
        // the digits.
        stroke={PAGE_BG}
        strokeWidth={3}
        paintOrder="stroke"
        textAnchor={
          minPoint.x === xMin ? "start" : minPoint.x === xMax ? "end" : "middle"
        }
      >
        {formatY ? formatY(minPoint.y) : minPoint.y.toFixed(2)}
      </text>

      <text x={padding} y={height - padding + 14} fontSize={9} fill={AXIS_COLOR}>
        {xMin}
      </text>
      <text x={width - padding} y={height - padding + 14} fontSize={9} fill={AXIS_COLOR} textAnchor="end">
        {xMax}
      </text>
    </svg>
  );
}

export interface Bar {
  label: string;
  value: number;
  status: Status;
}

// Horizontal bars — the comparisons here are a handful of labelled rates,
// which read better with the label alongside than rotated under an axis.
export function BarChart({
  bars,
  max,
  formatValue = (v) => v.toFixed(4),
  ariaLabel,
}: {
  bars: Bar[];
  max?: number;
  formatValue?: (value: number) => string;
  ariaLabel: string;
}) {
  const ceiling = max ?? (Math.max(...bars.map((b) => b.value), 0) * 1.1 || 1);

  return (
    <div role="img" aria-label={ariaLabel} className="space-y-2">
      {bars.map((bar) => (
        <div key={bar.label} className="flex items-center gap-3">
          <span className={`${typeScale.caption} w-48 shrink-0`}>{bar.label}</span>
          <div className="flex-1 h-3 rounded-full bg-app-rule overflow-hidden">
            <div
              className="h-full rounded-full"
              style={{
                width: `${Math.min(100, (bar.value / ceiling) * 100)}%`,
                backgroundColor: STATUS_HEX[bar.status],
              }}
            />
          </div>
          <span className="font-mono text-xs text-app-ink tabular-nums w-20 text-right">
            {formatValue(bar.value)}
          </span>
        </div>
      ))}
    </div>
  );
}
