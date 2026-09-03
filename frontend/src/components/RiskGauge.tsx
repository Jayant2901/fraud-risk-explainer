import {
  actionForScore,
  actionStatus,
  easing,
  focusRing,
  pressable,
  statusBadgeClass,
  statusDotClass,
  statusFillClass,
  typeScale,
} from "../theme";

// Preset scores for the interactive mode: the two real thresholds, their
// midpoint, and a clear-allow / clear-block anchor either side. Derived
// from the passed-in thresholds so this never drifts from the live
// decision boundary.
export function presetScores(reviewThreshold: number, blockThreshold: number): number[] {
  const presets = [
    10,
    reviewThreshold,
    Math.round((reviewThreshold + blockThreshold) / 2),
    blockThreshold,
    90,
  ];
  return [...new Set(presets)].sort((a, b) => a - b);
}

// Scores come back fractional (12.3), so don't round them to death —
// one decimal, with a whole number staying whole.
function formatScore(score: number): string {
  const rounded = Math.round(score * 10) / 10;
  return Number.isInteger(rounded) ? String(rounded) : rounded.toFixed(1);
}

export default function RiskGauge({
  score,
  reviewThreshold,
  blockThreshold,
  interactive,
  highlightThreshold,
  label = "Risk score",
}: {
  score: number;
  reviewThreshold: number;
  blockThreshold: number;
  interactive?: { onPick: (score: number) => void };
  // Draws attention to one threshold tick — Live Scoring flashes the tick
  // an escalation pushed the transaction past.
  highlightThreshold?: "review" | "block" | null;
  label?: string;
}) {
  const action = actionForScore(score, reviewThreshold, blockThreshold);
  const status = actionStatus[action];
  const clamped = Math.max(0, Math.min(100, score));

  return (
    <div>
      <div className="flex items-end justify-between gap-4">
        <div>
          <p className={typeScale.caption}>{label}</p>
          <p className="font-mono text-4xl font-bold text-app-ink mt-1 tabular-nums">
            {formatScore(score)}
            <span className="text-lg font-normal text-app-faint"> / 100</span>
          </p>
        </div>
        <div className={statusBadgeClass(status)}>
          <span className={statusDotClass(status)} />
          <span className="font-semibold">{action}</span>
        </div>
      </div>

      <div className="relative h-2 mt-4 rounded-full bg-app-rule overflow-hidden">
        <div
          className={`h-full rounded-full transition-all duration-500 ${statusFillClass(status)}`}
          style={{ width: `${clamped}%`, transitionTimingFunction: easing.standard }}
        />
      </div>

      <div className="relative h-4">
        {(
          [
            ["review", reviewThreshold],
            ["block", blockThreshold],
          ] as const
        ).map(([name, value]) => (
          <div
            key={name}
            className="absolute top-0 -translate-x-1/2 flex flex-col items-center"
            style={{ left: `${Math.max(0, Math.min(100, value))}%` }}
          >
            <span
              aria-hidden="true"
              className={`block w-px h-1.5 transition-colors duration-300 ${
                highlightThreshold === name ? "bg-app-ink" : "bg-app-faint"
              }`}
            />
            <span
              className={`font-mono text-[10px] tabular-nums transition-colors duration-300 ${
                highlightThreshold === name ? "text-app-ink font-semibold" : "text-app-faint"
              }`}
            >
              {value}
            </span>
          </div>
        ))}
      </div>

      {interactive && (
        <div className="flex flex-wrap gap-2 mt-4">
          {presetScores(reviewThreshold, blockThreshold).map((preset) => (
            <button
              key={preset}
              onClick={() => interactive.onPick(preset)}
              aria-pressed={score === preset}
              className={`font-mono text-xs tabular-nums rounded-lg px-3 py-1.5 border transition-colors ${pressable} ${focusRing} ${
                score === preset
                  ? "border-app-accent text-app-accent-soft bg-app-accent/10"
                  : "border-app-rule text-app-muted hover:text-app-ink"
              }`}
            >
              {preset}
            </button>
          ))}
        </div>
      )}
    </div>
  );
}
