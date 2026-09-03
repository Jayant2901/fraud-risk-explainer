// Single source of truth for the small set of design decisions reused
// across every tab, so components stop each picking their own shade of
// slate/amber/emerald. Keep this file short — it's a handful of tokens,
// not a design framework.
//
// Color values themselves live as CSS custom properties registered in
// index.css's @theme block (app-bg/app-surface/app-ink/sys-green/...) —
// this file only decides which token each semantic role uses.

// The app's UI font. San Francisco on Apple platforms, Inter (loaded in
// index.css) everywhere else. JetBrains Mono is still the right face for
// data-dense numeric values, applied per-value via `font-mono` rather
// than globally.
export const systemFont =
  "-apple-system, BlinkMacSystemFont, 'SF Pro Display', 'Inter', sans-serif";

// ---- Type scale --------------------------------------------------------
// pageTitle   : the one <h1> in the app header
// sectionTitle: a tab's own heading (e.g. "Human Review Queue")
// subTitle    : a heading for a sub-block within a tab
// body        : normal reading text
// caption     : metadata, labels, helper text, table cells
export const typeScale = {
  pageTitle: "font-semibold text-xl tracking-[-.01em] text-app-ink",
  sectionTitle: "text-lg font-semibold text-app-ink",
  subTitle: "text-sm font-semibold text-app-ink",
  body: "text-sm text-app-muted",
  caption: "text-xs text-app-faint",
};

// ---- Status colors ------------------------------------------------------
// Two independent semantic scales share the same three underlying colors:
// decision actions (ALLOW/REVIEW/BLOCK) and entity escalation state
// (NORMAL/WATCH/ELEVATED). success=ALLOW/NORMAL, warning=REVIEW/WATCH,
// danger=BLOCK/ELEVATED, all drawn from Apple's system palette.
export type Status = "success" | "warning" | "danger" | "neutral";

const STATUS_BADGE: Record<Status, string> = {
  success: "bg-sys-green/10 border-sys-green/30 text-sys-green",
  warning: "bg-sys-orange/10 border-sys-orange/30 text-sys-orange",
  danger: "bg-sys-red/10 border-sys-red/30 text-sys-red",
  neutral: "bg-app-rule border-app-rule text-app-muted",
};

const STATUS_DOT: Record<Status, string> = {
  success: "bg-sys-green",
  warning: "bg-sys-orange",
  danger: "bg-sys-red",
  neutral: "bg-app-faint",
};

const STATUS_TEXT: Record<Status, string> = {
  success: "text-sys-green",
  warning: "text-sys-orange",
  danger: "text-sys-red",
  neutral: "text-app-muted",
};

// Raw hex per status, for consumers that need a color value rather than a
// class — chart series in the Model Validation tab, mainly.
export const STATUS_HEX: Record<Status, string> = {
  success: "#32D74B",
  warning: "#FF9F0A",
  danger: "#FF453A",
  neutral: "#6b6d72",
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

export function statusFillClass(status: Status): string {
  return STATUS_DOT[status];
}

// A block-level message box (a "no report generated yet" notice, a
// caveat) in the same palette as the badges.
export function noticeClass(status: Status): string {
  return `border rounded-md px-3 py-2 ${STATUS_BADGE[status]}`;
}

export type Action = "ALLOW" | "REVIEW" | "BLOCK";

export const actionStatus: Record<Action, Status> = {
  ALLOW: "success",
  REVIEW: "warning",
  BLOCK: "danger",
};

export const escalationStatus: Record<"NORMAL" | "WATCH" | "ELEVATED", Status> = {
  NORMAL: "success",
  WATCH: "warning",
  ELEVATED: "danger",
};

// The deterministic gate's banding, mirrored for display. Same boundary
// semantics as src/decision_rules.py's decide_action (>= is the crossing),
// so the gauge and the backend never disagree about which side of a
// threshold a score sits on.
export function actionForScore(
  score: number,
  reviewThreshold: number,
  blockThreshold: number
): Action {
  if (score >= blockThreshold) return "BLOCK";
  if (score >= reviewThreshold) return "REVIEW";
  return "ALLOW";
}

// ---- Surface elevation — one step lighter than the page background
// (bg-app-bg), used for grouped content that needs to visually separate
// from the page without resorting to a border on every panel.
export const surface = "bg-app-surface rounded-xl";

// Translucent chrome. Deliberately reserved for the sticky app header —
// blur everywhere is what makes a UI look like a bad imitation of iOS
// rather than a considered one.
export const surfaceTranslucent = "bg-app-bg/70 backdrop-blur-xl backdrop-saturate-150";

// ---- Grouped list rows (label left, monospace value right) --------------
export const groupedRow = "flex items-baseline justify-between gap-4 py-2.5";
export const groupedRowLabel = "text-sm text-app-muted";
export const groupedRowValue = "font-mono text-sm text-app-ink tabular-nums";

// ---- Motion -------------------------------------------------------------
// standard: state transitions (color, width, opacity).
// spring:   entrance/arrival motion that should overshoot and settle.
export const easing = {
  standard: "cubic-bezier(.32,.72,0,1)",
  spring: "cubic-bezier(.34,1.56,.64,1)",
};

// The one piece of tactile feedback in an otherwise flat, dark UI.
export const pressable = "transition-transform duration-100 active:scale-[0.97]";

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
