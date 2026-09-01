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
}

const BASE = "/api";

// Every /api/* route except /api/health requires this (see api/main.py's
// verify_api_key). Set in frontend/.env (copy from .env.example) — Vite
// only exposes VITE_-prefixed vars to client code, and only at build/dev-
// server time, so this is baked in per-environment, not runtime-secret.
const API_KEY = import.meta.env.VITE_API_KEY;

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
    throw new Error(`${res.status} ${res.statusText}: ${body}`);
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
  getExplanation: (verdictId: string) =>
    request<ExplanationResult>(`/explanations/${encodeURIComponent(verdictId)}`),
  costAnalysis: (fraudLoss: number, fpCost: number) =>
    request<CostAnalysis>(
      `/cost-analysis?fraud_loss=${fraudLoss}&fp_cost=${fpCost}`
    ),
};
