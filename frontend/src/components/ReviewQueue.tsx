import { useCallback, useEffect, useState } from "react";
import { api, type ReviewQueueItem, type ReviewQueueMetrics } from "../api/client";
import {
  accentBg,
  accentBorder,
  accentText,
  actionStatus,
  buttonBase,
  buttonLabel,
  statusBadgeClass,
  statusDotClass,
  typeScale,
} from "../theme";

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
        <h2 className={typeScale.sectionTitle}>Human Review Queue</h2>
        <p className={`${typeScale.body} mt-1 max-w-2xl`}>
          Every REVIEW/BLOCK verdict from the Live Scoring tab lands here. Dispose each one as
          confirmed fraud or a false positive — that disposition closes the feedback loop between
          the model's decision and a confirmed outcome, and feeds the precision numbers below.
        </p>
      </div>

      {error && <p className="text-sm text-app-danger">{error}</p>}

      {metrics && (
        <div>
          <h3 className={typeScale.subTitle}>
            Reviewer precision so far ({metrics.total_disposed} disposed)
          </h3>
          {metrics.total_disposed === 0 ? (
            <p className="text-sm text-app-faint mt-1.5">
              No dispositions yet — confirm or dismiss items below to start building this up.
            </p>
          ) : (
            <div className="grid grid-cols-1 sm:grid-cols-3 gap-4 text-sm mt-2 divide-y sm:divide-y-0 sm:divide-x divide-app-rule">
              <div className="sm:pr-4">
                <p className={`${typeScale.caption} uppercase tracking-wide`}>Overall</p>
                <p className="text-xl font-semibold text-app-ink tabular-nums">{formatPct(metrics.overall_precision)}</p>
              </div>
              <div className="sm:px-4 pt-3 sm:pt-0">
                <p className={`${typeScale.caption} uppercase tracking-wide`}>
                  Escalation-triggered ({metrics.escalated_count})
                </p>
                <p className={`text-xl font-semibold ${accentText} tabular-nums`}>{formatPct(metrics.escalated_precision)}</p>
              </div>
              <div className="sm:pl-4 pt-3 sm:pt-0">
                <p className={`${typeScale.caption} uppercase tracking-wide`}>
                  Not escalated ({metrics.non_escalated_count})
                </p>
                <p className="text-xl font-semibold text-app-ink tabular-nums">{formatPct(metrics.non_escalated_precision)}</p>
              </div>
            </div>
          )}
        </div>
      )}

      <div>
        <h3 className={typeScale.subTitle}>
          Pending ({loading ? "…" : items.length})
        </h3>

        {!loading && items.length === 0 ? (
          <div className="mt-2 border border-dashed border-app-rule rounded-xl p-6 text-app-faint text-sm">
            Nothing pending. Score a transaction in the Live Scoring tab that comes back REVIEW or
            BLOCK, and it'll show up here.
          </div>
        ) : (
          <ul className="mt-2 divide-y divide-app-rule border-t border-app-rule">
            {items.map((item) => (
              <li
                key={item.verdict_id}
                className="py-3 flex flex-col sm:flex-row sm:items-center gap-3"
              >
                <div className="flex-1 space-y-1">
                  <div className="flex items-center gap-2 flex-wrap">
                    <span className="text-sm font-medium text-app-ink">{item.entity_id}</span>
                    <span className={typeScale.caption}>txn #{item.txn_index}</span>
                    <span className={statusBadgeClass(actionStatus[item.decision.action])}>
                      <span className={statusDotClass(actionStatus[item.decision.action])} />
                      {item.decision.action}
                    </span>
                    {item.escalated_due_to_history && (
                      <span className={`border ${accentBorder} ${accentBg} ${accentText} rounded-md px-2 py-0.5 text-xs`}>
                        escalated (baseline: {item.baseline_decision.action})
                      </span>
                    )}
                  </div>
                  <p className={typeScale.caption}>Risk score: {item.risk_score} / 100</p>
                </div>

                <div className="flex gap-2 shrink-0">
                  <button
                    onClick={() => handleDispose(item.verdict_id, "CONFIRMED_FRAUD")}
                    disabled={disposing === item.verdict_id}
                    className={`text-xs font-medium px-3 py-1.5 rounded-md bg-app-danger/10 border border-app-danger/30 text-app-danger hover:bg-app-danger/20 ${buttonLabel} ${buttonBase}`}
                  >
                    Confirm Fraud
                  </button>
                  <button
                    onClick={() => handleDispose(item.verdict_id, "FALSE_POSITIVE")}
                    disabled={disposing === item.verdict_id}
                    className={`text-xs font-medium px-3 py-1.5 rounded-md bg-emerald-500/10 border border-emerald-500/30 text-emerald-300 hover:bg-emerald-500/20 ${buttonLabel} ${buttonBase}`}
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
