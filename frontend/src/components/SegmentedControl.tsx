import { easing, focusRing, pressable } from "../theme";

export interface Segment {
  id: string;
  label: string;
}

// The app's tab nav. Keeps role="tablist"/role="tab"/aria-selected so
// keyboard and screen-reader behavior is identical to the plain <nav> of
// buttons this replaced.
export default function SegmentedControl({
  segments,
  activeId,
  onChange,
  ariaLabel,
}: {
  segments: Segment[];
  activeId: string;
  onChange: (id: string) => void;
  ariaLabel?: string;
}) {
  return (
    <div
      role="tablist"
      aria-label={ariaLabel}
      className="inline-flex bg-app-surface rounded-xl p-1 gap-1"
    >
      {segments.map((segment) => {
        const isActive = segment.id === activeId;
        return (
          <button
            key={segment.id}
            role="tab"
            aria-selected={isActive}
            onClick={() => onChange(segment.id)}
            style={{ transitionTimingFunction: easing.standard }}
            className={`rounded-lg px-4 py-1.5 text-sm font-medium transition-colors ${pressable} ${focusRing} ${
              isActive
                ? "bg-app-rule text-app-ink"
                : "text-app-faint hover:text-app-muted"
            }`}
          >
            {segment.label}
          </button>
        );
      })}
    </div>
  );
}
