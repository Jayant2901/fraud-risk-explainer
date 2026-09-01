// Single source of truth for the small set of design decisions reused
// across every tab, so components stop each picking their own shade of
// slate/amber/emerald. Keep this file short — it's a handful of tokens,
// not a design framework.

// ---- Type scale --------------------------------------------------------
// pageTitle   : the one <h1> in the app header
// sectionTitle: a tab's own heading (e.g. "Human Review Queue")
// subTitle    : a heading for a sub-block within a tab
// body        : normal reading text
// caption     : metadata, labels, helper text, table cells
export const typeScale = {
  pageTitle: "text-xl font-semibold tracking-tight text-neutral-50",
  sectionTitle: "text-lg font-semibold text-neutral-50",
  subTitle: "text-sm font-semibold text-neutral-200",
  body: "text-sm text-neutral-300",
  caption: "text-xs text-neutral-400",
};

// ---- Status colors ------------------------------------------------------
// Two independent semantic scales share the same three underlying colors:
// decision actions (ALLOW/REVIEW/BLOCK) and entity escalation state
// (NORMAL/WATCH/ELEVATED). success=ALLOW/NORMAL, warning=REVIEW/WATCH,
// danger=BLOCK/ELEVATED.
export type Status = "success" | "warning" | "danger" | "neutral";

const STATUS_BADGE: Record<Status, string> = {
  success: "bg-emerald-500/10 border-emerald-500/30 text-emerald-300",
  warning: "bg-amber-500/10 border-amber-500/30 text-amber-300",
  danger: "bg-red-500/10 border-red-500/30 text-red-300",
  neutral: "bg-neutral-500/10 border-neutral-500/30 text-neutral-300",
};

const STATUS_DOT: Record<Status, string> = {
  success: "bg-emerald-400",
  warning: "bg-amber-400",
  danger: "bg-red-400",
  neutral: "bg-neutral-400",
};

const STATUS_TEXT: Record<Status, string> = {
  success: "text-emerald-300",
  warning: "text-amber-300",
  danger: "text-red-300",
  neutral: "text-neutral-300",
};

export function statusBadgeClass(status: Status): string {
  return `inline-flex items-center gap-1.5 border rounded-md px-2 py-0.5 text-xs font-medium ${STATUS_BADGE[status]}`;
}

export function statusDotClass(status: Status): string {
  return `inline-block h-2 w-2 rounded-full ${STATUS_DOT[status]}`;
}

export function statusTextClass(status: Status): string {
  return STATUS_TEXT[status];
}

export const actionStatus: Record<"ALLOW" | "REVIEW" | "BLOCK", Status> = {
  ALLOW: "success",
  REVIEW: "warning",
  BLOCK: "danger",
};

export const escalationStatus: Record<"NORMAL" | "WATCH" | "ELEVATED", Status> = {
  NORMAL: "success",
  WATCH: "warning",
  ELEVATED: "danger",
};

// ---- Surface elevation — one step lighter than the page background
// (bg-neutral-950), used for grouped content that needs to visually
// separate from the page without resorting to a border on every panel.
export const surface = "bg-neutral-900 rounded-xl";

// ---- Accent (indigo) — used for the one non-status highlight color,
// e.g. active tab, escalation-triggered badge, chart line.
export const accentText = "text-indigo-300";
export const accentBorder = "border-indigo-500/40";
export const accentBg = "bg-indigo-500/10";

// ---- Shared interactive-state classes ------------------------------------
// Applied to every clickable control so hover/focus-visible/disabled read
// the same everywhere.
export const focusRing =
  "focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-indigo-400 focus-visible:ring-offset-2 focus-visible:ring-offset-neutral-950";

export const buttonBase = `transition disabled:opacity-40 disabled:cursor-not-allowed ${focusRing}`;
