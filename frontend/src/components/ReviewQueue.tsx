import { useCallback, useEffect, useState } from "react";
import { api, type ReviewQueueItem, type ReviewQueueMetrics } from "../api/client";
import {
  accentBg,
  accentBorder,
  accentText,
  actionStatus,
  buttonBase,
  buttonLabel,
  focusRing,
  statusBadgeClass,
  statusDotClass,
  typeScale,
} from "../theme";

function formatPct(p: number | null): string {
  return p === null ? "—" : `${(p * 100).toFixed(1)}%`;
}

// Escalating visual urgency as a pending item ages, so an old,
// forgotten item stands out from one that just landed.
const AGE_BAND_AMBER_MS = 60 * 60 * 1000; // 1 hour
const AGE_BAND_RED_MS = 4 * 60 * 60 * 1000; // 4 hours

function ageLabel(createdAt: string): { text: string; className: string } {
  const ms = Date.now() - new Date(createdAt).getTime();
  const minutes = Math.max(0, Math.round(ms / 60000));
  const text = minutes < 60 ? `${minutes}m ago` : `${Math.round(minutes / 60)}h ago`;
  const className =
    ms >= AGE_BAND_RED_MS
      ? "text-app-danger"
      : ms >= AGE_BAND_AMBER_MS
        ? "text-amber-400"
        : "text-app-faint";
  return { text, className };
}

export default function ReviewQueue() {
  const [items, setItems] = useState<ReviewQueueItem[]>([]);
  const [metrics, setMetrics] = useState<ReviewQueueMetrics | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [disposing, setDisposing] = useState<string | null>(null);
  const [expandedNotes, setExpandedNotes] = useState<Set<string>>(new Set());
  const [noteDrafts, setNoteDrafts] = useState<Record<string, string>>({});
  const [addingNote, setAddingNote] = useState<string | null>(null);
  const [expandedRelated, setExpandedRelated] = useState<Set<string>>(new Set());
  const [relatedItems, setRelatedItems] = useState<Record<string, ReviewQueueItem[]>>({});

  const refresh = useCallback(async () => {
    try {
      const [queueRes, metricsRes] = await Promise.all([api.listReviewQueue(), api.reviewQueueMetrics()]);
      setItems(queueRes.items);
      setMetrics(metricsRes);

      // Prefetched (not lazy) so the "N other items for this entity"
      // count is visible up front, not only after the reviewer expands
      // it — one small GET per pending item, acceptable at this
      // project's demo scale (a handful of entities).
      const relatedPairs = await Promise.all(
        queueRes.items.map(async (item) => {
          try {
            const res = await api.relatedReviewItems(item.verdict_id);
            return [item.verdict_id, res.items] as const;
          } catch {
            return [item.verdict_id, []] as const;
          }
        })
      );
      setRelatedItems(Object.fromEntries(relatedPairs));
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

  function toggleNotes(verdictId: string) {
    setExpandedNotes((prev) => {
      const next = new Set(prev);
      if (next.has(verdictId)) next.delete(verdictId);
      else next.add(verdictId);
      return next;
    });
  }

  async function handleAddNote(verdictId: string) {
    const text = (noteDrafts[verdictId] ?? "").trim();
    if (!text) return;
    setAddingNote(verdictId);
    setError(null);
    try {
      const note = await api.addReviewNote(verdictId, text);
      setItems((prev) =>
        prev.map((i) => (i.verdict_id === verdictId ? { ...i, notes: [...i.notes, note] } : i))
      );
      setNoteDrafts((prev) => ({ ...prev, [verdictId]: "" }));
    } catch (e) {
      setError(String(e));
    } finally {
      setAddingNote(null);
    }
  }

  function toggleRelated(verdictId: string) {
    setExpandedRelated((prev) => {
      const next = new Set(prev);
      if (next.has(verdictId)) next.delete(verdictId);
      else next.add(verdictId);
      return next;
    });
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
              <li key={item.verdict_id} className="py-3 space-y-2">
                <div className="flex flex-col sm:flex-row sm:items-center gap-3">
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
                      <span className={`text-xs ${ageLabel(item.created_at).className}`}>
                        {ageLabel(item.created_at).text}
                      </span>
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
                </div>

                <div className="flex items-center gap-3 flex-wrap">
                  <button
                    onClick={() => toggleNotes(item.verdict_id)}
                    className={`text-xs text-app-faint hover:text-app-ink ${focusRing}`}
                  >
                    {item.notes.length > 0
                      ? `${item.notes.length} note${item.notes.length === 1 ? "" : "s"}`
                      : "Add note"}
                    {expandedNotes.has(item.verdict_id) ? " ▲" : " ▼"}
                  </button>
                  {(relatedItems[item.verdict_id]?.length ?? 0) > 0 && (
                    <button
                      onClick={() => toggleRelated(item.verdict_id)}
                      className={`text-xs text-app-faint hover:text-app-ink ${focusRing}`}
                    >
                      {relatedItems[item.verdict_id]!.length} other item
                      {relatedItems[item.verdict_id]!.length === 1 ? "" : "s"} for this entity
                      {expandedRelated.has(item.verdict_id) ? " ▲" : " ▼"}
                    </button>
                  )}
                </div>

                {expandedNotes.has(item.verdict_id) && (
                  <div className="pl-3 border-l-2 border-app-rule space-y-2">
                    {item.notes.length > 0 && (
                      <ul className="space-y-1.5">
                        {item.notes.map((note, idx) => (
                          <li key={idx} className="text-xs">
                            <span className="font-medium text-app-ink">{note.author}</span>{" "}
                            <span className="text-app-faint">{new Date(note.at).toLocaleString()}</span>
                            <p className="text-app-muted">{note.text}</p>
                          </li>
                        ))}
                      </ul>
                    )}
                    <div className="flex gap-2">
                      <label htmlFor={`note-${item.verdict_id}`} className="sr-only">
                        Add a note
                      </label>
                      <input
                        id={`note-${item.verdict_id}`}
                        type="text"
                        value={noteDrafts[item.verdict_id] ?? ""}
                        onChange={(e) =>
                          setNoteDrafts((prev) => ({ ...prev, [item.verdict_id]: e.target.value }))
                        }
                        placeholder="Add a note…"
                        className={`flex-1 bg-app-surface border border-app-rule rounded-md px-2 py-1 text-xs text-app-ink ${focusRing}`}
                      />
                      <button
                        onClick={() => handleAddNote(item.verdict_id)}
                        disabled={addingNote === item.verdict_id || !(noteDrafts[item.verdict_id] ?? "").trim()}
                        className={`text-xs font-medium px-2.5 py-1 rounded-md border border-app-accent text-app-accent-soft disabled:opacity-50 ${buttonLabel} ${buttonBase}`}
                      >
                        {addingNote === item.verdict_id ? "Adding…" : "Add"}
                      </button>
                    </div>
                  </div>
                )}

                {expandedRelated.has(item.verdict_id) && (relatedItems[item.verdict_id]?.length ?? 0) > 0 && (
                  <ul className="pl-3 border-l-2 border-app-rule space-y-1 text-xs">
                    {relatedItems[item.verdict_id]!.map((rel) => (
                      <li key={rel.verdict_id} className="text-app-muted flex items-center gap-2 flex-wrap">
                        <span className={statusBadgeClass(actionStatus[rel.decision.action])}>
                          <span className={statusDotClass(actionStatus[rel.decision.action])} />
                          {rel.decision.action}
                        </span>
                        <span>risk {rel.risk_score}</span>
                        <span>{rel.disposition ?? "pending"}</span>
                        <span className="text-app-faint">{new Date(rel.created_at).toLocaleString()}</span>
                      </li>
                    ))}
                  </ul>
                )}
              </li>
            ))}
          </ul>
        )}
      </div>
    </div>
  );
}
