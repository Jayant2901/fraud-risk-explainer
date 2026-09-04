export interface Factor {
  feature: string;
  label: string;
  value: string;
  contribution: number;
}

export interface EscalationState {
  entity_id?: string;
  state: "NORMAL" | "WATCH" | "ELEVATED";
  recent_verdict_count: number;
  recent_risky_count: number;
  avg_recent_risk_score?: number;
  recent_verdicts: string[];
}

export interface Verdict {
  explanation: string;
  action: "ALLOW" | "REVIEW" | "BLOCK";
  escalated_due_to_history: boolean;
  rationale: string;
}

export interface Decision {
  action: "ALLOW" | "REVIEW" | "BLOCK";
  escalated_due_to_history: boolean;
}

export interface ScoreResult {
  risk_score: number;
  above_threshold: boolean;
  top_factors: Factor[];
  escalation_before: EscalationState;
  decision: Decision;
  verdict_id: string;
}

export type ExplanationResult =
  | { status: "pending" }
  | { status: "ready"; verdict: Verdict };

export interface TxnSummary {
  index: number;
  TransactionAmt: number;
  TransactionDT: number;
  ProductCD: string;
}

export interface CostAnalysis {
  eval_report: string | null;
  defaults: { avg_fraud_loss: number; avg_fp_cost: number };
  params: { fraud_loss: number; fp_cost: number };
  headline_monthly_savings_estimate: number | null;
  headline_basis: string | null;
  // Per-threshold total cost over the real test set, recomputed by the
  // API for the fraud_loss/fp_cost this request asked for. Empty until
  // train_model.py has written models/cost_curve.json.
  cost_curve: { threshold: number; total_cost: number }[];
  decision_thresholds: { review: number; block: number };
  escalation_cutoffs: { watch: number; elevated: number };
  // null until train_model.py has been re-run to write it into
  // models/cost_summary.json.
  roc_auc: number | null;
}

export interface CostSensitivityCell {
  fraud_loss_multiplier: number;
  fp_cost_multiplier: number;
  avg_fraud_loss: number;
  avg_fp_cost: number;
  optimal_threshold: number;
  optimal_total_cost: number;
  estimated_savings_pct: number;
}

export interface CostSensitivity {
  base_fraud_loss: number;
  base_fp_cost: number;
  fraud_loss_multipliers: number[];
  fp_cost_multipliers: number[];
  grid: CostSensitivityCell[];
}

export interface CostSensitivityResult {
  sensitivity: CostSensitivity | null;
  message: string | null;
}

export interface ReviewNote {
  author: string;
  text: string;
  at: string;
}

export interface ReviewQueueItem {
  verdict_id: string;
  entity_id: string;
  txn_index: number;
  risk_score: number;
  decision: Decision;
  baseline_decision: Decision;
  escalated_due_to_history: boolean;
  disposition: "CONFIRMED_FRAUD" | "FALSE_POSITIVE" | null;
  disposed_at: string | null;
  created_at: string;
  notes: ReviewNote[];
}

export interface DriftBucket {
  bucket: number;
  n: number;
  n_fraud: number;
  roc_auc: number | null;
  precision: number;
  recall: number;
}

export interface DriftAnalysis {
  span_seconds: number;
  num_buckets: number;
  edges: number[];
  buckets: DriftBucket[];
}

export interface DriftAnalysisResult {
  drift: DriftAnalysis | null;
  message: string | null;
}

export interface BoundaryFragility {
  n_flagged: number;
  n_near_boundary: number;
  fraction_near_boundary: number;
}

export interface ConsistencyPair {
  status: "ok" | "insufficient_data";
  n_calls: number;
  n_excluded_fallback: number;
  n_valid: number;
  modal_action: "ALLOW" | "REVIEW" | "BLOCK" | null;
  self_consistency_rate: number | null;
  cross_agreement: boolean | null;
  band: string;
  row_index: number;
  risk_score: number;
  escalation_context: "NORMAL" | "ELEVATED";
  deterministic_action: "ALLOW" | "REVIEW" | "BLOCK";
}

export interface ConsistencyAnalysis {
  part_a_boundary_fragility: BoundaryFragility;
  part_b_pairs: ConsistencyPair[];
}

export interface ConsistencyAnalysisResult {
  consistency: ConsistencyAnalysis | null;
  message: string | null;
}

// Fields a person could reasonably fill in by hand for a transaction
// that doesn't exist in the cached historical sample — everything else
// the model expects (C1-C14, D-features, V-features, M-features,
// entity_prior_* features, ...) is left missing and handled by the
// backend exactly like RiskExplainer.score_transaction already handles
// any missing feature. TransactionAmt is the only required field.
export interface CustomTransactionRequest {
  TransactionAmt: number;
  ProductCD?: string;
  card4?: string;
  card6?: string;
  P_emaildomain?: string;
  R_emaildomain?: string;
  DeviceType?: string;
  addr1?: number;
  addr2?: number;
  hour_of_day?: number;
  attach_to_entity_id?: string | null;
}

export interface TextReportResult {
  report: string | null;
  message: string | null;
}

export interface StrategyMetrics {
  n_fraud: number;
  n_legit: number;
  flagged_fraud: number;
  flagged_legit: number;
  recall: number;
  false_flag_rate: number;
}

export interface EscalationSweepRow {
  watch_threshold: number;
  elevated_threshold: number;
  recall: number;
  false_flag_rate: number;
  cost: number;
}

// The structured twin of the escalation ablation text report — same
// numbers, same computation, shaped for charting. Null until
// src/escalation_ablation.py has been re-run to write it.
export interface EscalationAblationSummary {
  n_transactions: number;
  baseline: StrategyMetrics;
  adjusted: StrategyMetrics;
  flips: { n_flips: number; n_flips_fraud: number; precision: number };
  sweep: EscalationSweepRow[];
}

export interface EscalationAblationResult {
  report: string | null;
  summary: EscalationAblationSummary | null;
  message: string | null;
}

export interface ShadowActionPair {
  live_action: string;
  shadow_action: string;
  count: number;
}

// GET /api/shadow-comparison — how often a candidate model
// (SHADOW_MODEL_PATH) would have decided differently from the model
// actually in production, on transactions scored since the process (or
// the shared Redis counters) last reset.
export interface ShadowComparison {
  configured: boolean;
  total_scored: number;
  agreement_rate: number | null;
  action_pairs: ShadowActionPair[];
  message: string | null;
}

export interface ReviewQueueMetrics {
  total_disposed: number;
  confirmed_fraud_count: number;
  false_positive_count: number;
  overall_precision: number | null;
  escalated_count: number;
  escalated_precision: number | null;
  non_escalated_count: number;
  non_escalated_precision: number | null;
}

const BASE = "/api";

// Every /api/* route except /api/health requires this (see api/main.py's
// verify_api_key). Set in frontend/.env (copy from .env.example) — Vite
// only exposes VITE_-prefixed vars to client code, and only at build/dev-
// server time, so this is baked in per-environment, not runtime-secret.
const API_KEY = import.meta.env.VITE_API_KEY;

// FastAPI's HTTPException error body is {"detail": "..."} — a message
// written to be shown to a person (see e.g. api/main.py's
// SAMPLE_DATA_MISSING_DETAIL). ApiError carries that message on its own
// `detail` field so a caller can render it directly instead of the
// generic "{status} {statusText}: {raw body}" string, which is
// unreadable noise for an error that already has a human-written
// explanation attached.
export class ApiError extends Error {
  status: number;
  detail: string;

  constructor(status: number, detail: string) {
    super(detail);
    this.name = "ApiError";
    this.status = status;
    this.detail = detail;
  }
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const res = await fetch(`${BASE}${path}`, {
    ...init,
    headers: {
      "Content-Type": "application/json",
      ...(API_KEY ? { "X-API-Key": API_KEY } : {}),
      ...init?.headers,
    },
  });
  if (!res.ok) {
    const body = await res.text().catch(() => "");
    let detail = "";
    try {
      const parsed = JSON.parse(body);
      if (parsed && typeof parsed.detail === "string") detail = parsed.detail;
    } catch {
      // Not a {"detail": "..."} JSON body — fall through to the raw text.
    }
    if (!detail) detail = body || `${res.status} ${res.statusText}`;
    throw new ApiError(res.status, detail);
  }
  return res.json();
}

export const api = {
  listEntities: () => request<{ entities: string[] }>("/entities"),
  listTransactions: (entityId: string) =>
    request<{ entity_id: string; count: number; transactions: TxnSummary[] }>(
      `/entities/${encodeURIComponent(entityId)}/transactions`
    ),
  getEscalation: (entityId: string) =>
    request<EscalationState>(`/entities/${encodeURIComponent(entityId)}/escalation`),
  resetEntity: (entityId: string) =>
    request<{ status: string }>("/entities/reset", {
      method: "POST",
      body: JSON.stringify({ entity_id: entityId }),
    }),
  // idempotencyKey defaults to a fresh one per call. Pass an explicit key
  // (and reuse it) when wrapping this in retry logic, so a retried
  // request after a dropped response doesn't score + record the same
  // transaction twice.
  score: (entityId: string, txnIndex: number, idempotencyKey: string = crypto.randomUUID()) =>
    request<ScoreResult>("/score", {
      method: "POST",
      headers: { "Idempotency-Key": idempotencyKey },
      body: JSON.stringify({ entity_id: entityId, txn_index: txnIndex }),
    }),
  scoreCustom: (req: CustomTransactionRequest) =>
    request<ScoreResult>("/score-custom", {
      method: "POST",
      body: JSON.stringify(req),
    }),
  getExplanation: (verdictId: string) =>
    request<ExplanationResult>(`/explanations/${encodeURIComponent(verdictId)}`),
  // Server-Sent Events for one verdict: `decision` immediately, then
  // `explanation_delta` chunks as the model produces them, then a
  // terminal `explanation_complete` (or `error`). EventSource can't set
  // headers, so the API key rides as a query parameter — the same key
  // the X-API-Key header carries, over the same TLS connection.
  streamVerdict: (verdictId: string): EventSource =>
    new EventSource(
      `${BASE}/verdicts/${encodeURIComponent(verdictId)}/stream` +
        (API_KEY ? `?api_key=${encodeURIComponent(API_KEY)}` : "")
    ),
  costAnalysis: (fraudLoss: number, fpCost: number) =>
    request<CostAnalysis>(
      `/cost-analysis?fraud_loss=${fraudLoss}&fp_cost=${fpCost}`
    ),
  costSensitivity: () => request<CostSensitivityResult>("/cost-analysis/sensitivity"),
  listReviewQueue: () => request<{ items: ReviewQueueItem[] }>("/review-queue?status=pending"),
  disposeReviewItem: (verdictId: string, disposition: "CONFIRMED_FRAUD" | "FALSE_POSITIVE") =>
    request<ReviewQueueItem>(`/review-queue/${encodeURIComponent(verdictId)}/disposition`, {
      method: "POST",
      body: JSON.stringify({ disposition }),
    }),
  reviewQueueMetrics: () => request<ReviewQueueMetrics>("/review-queue/metrics"),
  addReviewNote: (verdictId: string, text: string, author = "Reviewer") =>
    request<ReviewNote>(`/review-queue/${encodeURIComponent(verdictId)}/notes`, {
      method: "POST",
      body: JSON.stringify({ author, text }),
    }),
  relatedReviewItems: (verdictId: string) =>
    request<{ items: ReviewQueueItem[] }>(`/review-queue/${encodeURIComponent(verdictId)}/related`),
  driftAnalysis: () => request<DriftAnalysisResult>("/drift-analysis"),
  consistencyAnalysis: () => request<ConsistencyAnalysisResult>("/consistency-analysis"),
  escalationAblation: () => request<EscalationAblationResult>("/escalation-ablation"),
  coldStartAnalysis: () => request<TextReportResult>("/cold-start-analysis"),
  shadowComparison: () => request<ShadowComparison>("/shadow-comparison"),
};
