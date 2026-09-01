// Single source of truth for the small set of design decisions reused
// across every tab, so components stop each picking their own shade of
// slate/amber/emerald. Keep this file short — it's a handful of tokens,
// not a design framework.
//
// Color values themselves live as CSS custom properties registered in
// index.css's @theme block (app-bg/app-surface/app-ink/...) — this file
// only decides which token each semantic role uses.

// ---- Type scale --------------------------------------------------------
// pageTitle   : the one <h1> in the app header — the sole Orbitron/display use
// sectionTitle: a tab's own heading (e.g. "Human Review Queue")
// subTitle    : a heading for a sub-block within a tab
// body        : normal reading text
// caption     : metadata, labels, helper text, table cells
export const typeScale = {
  pageTitle: "font-heading font-semibold text-xl uppercase tracking-[.04em] text-app-ink",
  sectionTitle: "text-lg font-semibold text-app-ink",
  subTitle: "text-sm font-semibold text-app-ink",
  body: "text-sm text-app-muted",
  caption: "text-xs text-app-faint",
};

// ---- Status colors ------------------------------------------------------
// Two independent semantic scales share the same three underlying colors:
// decision actions (ALLOW/REVIEW/BLOCK) and entity escalation state
// (NORMAL/WATCH/ELEVATED). success=ALLOW/NORMAL, warning=REVIEW/WATCH,
// danger=BLOCK/ELEVATED. Warning/success keep the standard amber/emerald;
// danger uses the theme's muted terracotta rather than saturated red.
export type Status = "success" | "warning" | "danger" | "neutral";

const STATUS_BADGE: Record<Status, string> = {
  success: "bg-emerald-500/10 border-emerald-500/30 text-emerald-300",
  warning: "bg-amber-500/10 border-amber-500/30 text-amber-300",
  danger: "bg-app-danger/10 border-app-danger/30 text-app-danger",
  neutral: "bg-app-rule border-app-rule text-app-muted",
};

const STATUS_DOT: Record<Status, string> = {
  success: "bg-emerald-400",
  warning: "bg-amber-400",
  danger: "bg-app-danger",
  neutral: "bg-app-faint",
};

const STATUS_TEXT: Record<Status, string> = {
  success: "text-emerald-300",
  warning: "text-amber-300",
  danger: "text-app-danger",
  neutral: "text-app-muted",
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
// (bg-app-bg), used for grouped content that needs to visually separate
// from the page without resorting to a border on every panel.
export const surface = "bg-app-surface rounded-xl";

// ---- Accent — the one non-status highlight color, e.g. active tab,
// escalation-triggered badge, chart line.
export const accentText = "text-app-accent-soft";
export const accentBorder = "border-app-accent/40";
export const accentBg = "bg-app-accent/10";

// ---- Shared interactive-state classes ------------------------------------
// Applied to every clickable control so hover/focus-visible/disabled read
// the same everywhere.
export const focusRing =
  "focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-app-accent focus-visible:ring-offset-2 focus-visible:ring-offset-app-bg";

export const buttonBase = `transition disabled:opacity-40 disabled:cursor-not-allowed ${focusRing}`;

// Uppercase + wide tracking treatment applied to primary interactive
// buttons per the design reference.
export const buttonLabel = "uppercase tracking-[.08em]";
