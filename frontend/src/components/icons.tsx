// Minimal hand-rolled icon set (no icon library dependency) — stand-ins
// for the emoji this app used to render as functional UI. Each is a
// plain 16x16 stroke icon that inherits `currentColor`, so it picks up
// whatever text color class the caller applies.

interface IconProps {
  className?: string;
}

export function CheckIcon({ className = "h-3.5 w-3.5" }: IconProps) {
  return (
    <svg viewBox="0 0 16 16" fill="none" className={className} aria-hidden="true">
      <path d="M3 8.5L6.5 12L13 4.5" stroke="currentColor" strokeWidth={1.75} strokeLinecap="round" strokeLinejoin="round" />
    </svg>
  );
}

export function AlertTriangleIcon({ className = "h-3.5 w-3.5" }: IconProps) {
  return (
    <svg viewBox="0 0 16 16" fill="none" className={className} aria-hidden="true">
      <path
        d="M8 2.5L14.5 13.5H1.5L8 2.5Z"
        stroke="currentColor"
        strokeWidth={1.5}
        strokeLinejoin="round"
      />
      <path d="M8 6.5V9.5" stroke="currentColor" strokeWidth={1.5} strokeLinecap="round" />
      <circle cx="8" cy="11.5" r="0.9" fill="currentColor" />
    </svg>
  );
}

export function BlockIcon({ className = "h-3.5 w-3.5" }: IconProps) {
  return (
    <svg viewBox="0 0 16 16" fill="none" className={className} aria-hidden="true">
      <circle cx="8" cy="8" r="6" stroke="currentColor" strokeWidth={1.5} />
      <path d="M4.2 4.2L11.8 11.8" stroke="currentColor" strokeWidth={1.5} strokeLinecap="round" />
    </svg>
  );
}
