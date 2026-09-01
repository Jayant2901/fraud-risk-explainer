import { useCallback, useEffect, useState } from "react";
import { api, type ReviewQueueItem, type ReviewQueueMetrics } from "../api/client";

const ACTION_STYLE: Record<string, string> = {
  REVIEW: "bg-amber-500/15 border-amber-500/40 text-amber-300",
  BLOCK: "bg-red-500/15 border-red-500/40 text-red-300",
};

function formatPct(p: number | null): string {
  return p === null ? "—" : `${(p * 100).toFixed(1)}%`;
}

export default function ReviewQueue() {
  const [items, setItems] = useState<ReviewQueueItem[]>([]);
  const [metrics, setMetrics] = useState<ReviewQueueMetrics | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [disposing, setDisposing] = useState<string | null>(null);

  const refresh = useCallback(async () => {
    try {
      const [queueRes, metricsRes] = await Promise.all([api.listReviewQueue(), api.reviewQueueMetrics()]);
      setItems(queueRes.items);
      setMetrics(metricsRes);
    } catch (e) {
      setError(String(e));
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    refresh();
  }, [refresh]);

  async function handleDispose(verdictId: string, disposition: "CONFIRMED_FRAUD" | "FALSE_POSITIVE") {
    setDisposing(verdictId);
    setError(null);
    try {
      await api.disposeReviewItem(verdictId, disposition);
      await refresh();
    } catch (e) {
      setError(String(e));
    } finally {
      setDisposing(null);
    }
  }

  return (
    <div className="space-y-6">
      <div>
        <h2 className="text-lg font-semibold text-slate-100">Human Review Queue</h2>
        <p className="text-sm text-slate-400 mt-1 max-w-2xl">
          Every REVIEW/BLOCK verdict from the Live Scoring tab lands here. Dispose each one as
          confirmed fraud or a false positive — that disposition closes the feedback loop between
          the model's decision and a confirmed outcome, and feeds the precision numbers below.
        </p>
      </div>

      {error && <p className="text-sm text-red-400">{error}</p>}

      {metrics && (
        <div className="bg-slate-900/60 border border-slate-800 rounded-xl p-4">
          <h3 className="text-sm font-semibold text-slate-300 mb-3">
            Reviewer precision so far ({metrics.total_disposed} disposed)
          </h3>
          {metrics.total_disposed === 0 ? (
            <p className="text-sm text-slate-500">
              No dispositions yet — confirm or dismiss items below to start building this up.
            </p>
          ) : (
            <div className="grid grid-cols-1 sm:grid-cols-3 gap-4 text-sm">
              <div>
                <p className="text-xs text-slate-400 uppercase tracking-wide">Overall</p>
                <p className="text-xl font-semibold text-slate-100">{formatPct(metrics.overall_precision)}</p>
              </div>
              <div>
                <p className="text-xs text-slate-400 uppercase tracking-wide">
                  Escalation-triggered ({metrics.escalated_count})
                </p>
                <p className="text-xl font-semibold text-indigo-300">{formatPct(metrics.escalated_precision)}</p>
              </div>
              <div>
                <p className="text-xs text-slate-400 uppercase tracking-wide">
                  Not escalated ({metrics.non_escalated_count})
                </p>
                <p className="text-xl font-semibold text-slate-100">{formatPct(metrics.non_escalated_precision)}</p>
              </div>
            </div>
          )}
        </div>
      )}

      <div>
        <h3 className="text-sm font-semibold text-slate-300 mb-2">
          Pending ({loading ? "…" : items.length})
        </h3>

        {!loading && items.length === 0 ? (
          <div className="bg-slate-900/40 border border-dashed border-slate-700 rounded-xl p-6 text-slate-400 text-sm">
            Nothing pending. Score a transaction in the Live Scoring tab that comes back REVIEW or
            BLOCK, and it'll show up here.
          </div>
        ) : (
          <ul className="space-y-3">
            {items.map((item) => (
              <li
                key={item.verdict_id}
                className="bg-slate-900/60 border border-slate-800 rounded-xl p-4 flex flex-col sm:flex-row sm:items-center gap-3"
              >
                <div className="flex-1 space-y-1">
                  <div className="flex items-center gap-2 flex-wrap">
                    <span className="text-sm font-medium text-slate-100">{item.entity_id}</span>
                    <span className="text-xs text-slate-500">txn #{item.txn_index}</span>
                    <span
                      className={`border rounded-md px-2 py-0.5 text-xs font-medium ${
                        ACTION_STYLE[item.decision.action] ?? ""
                      }`}
                    >
                      {item.decision.action}
                    </span>
                    {item.escalated_due_to_history && (
                      <span className="border border-indigo-500/40 bg-indigo-500/10 text-indigo-300 rounded-md px-2 py-0.5 text-xs">
                        escalated (baseline: {item.baseline_decision.action})
                      </span>
                    )}
                  </div>
                  <p className="text-xs text-slate-500">Risk score: {item.risk_score} / 100</p>
                </div>

                <div className="flex gap-2 shrink-0">
                  <button
                    onClick={() => handleDispose(item.verdict_id, "CONFIRMED_FRAUD")}
                    disabled={disposing === item.verdict_id}
                    className="text-xs font-medium px-3 py-1.5 rounded-md bg-red-500/15 border border-red-500/40 text-red-300 hover:bg-red-500/25 disabled:opacity-50 transition"
                  >
                    Confirm Fraud
                  </button>
                  <button
                    onClick={() => handleDispose(item.verdict_id, "FALSE_POSITIVE")}
                    disabled={disposing === item.verdict_id}
                    className="text-xs font-medium px-3 py-1.5 rounded-md bg-emerald-500/15 border border-emerald-500/40 text-emerald-300 hover:bg-emerald-500/25 disabled:opacity-50 transition"
                  >
                    Mark False Positive
                  </button>
                </div>
              </li>
            ))}
          </ul>
        )}
      </div>
    </div>
  );
}
