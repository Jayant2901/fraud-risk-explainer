import { useEffect, useMemo, useState } from "react";
import { api, type EscalationState, type ExplanationResult, type ScoreResult, type TxnSummary } from "../api/client";

const STATE_DISPLAY: Record<string, { label: string; color: string }> = {
  NORMAL: { label: "🟢 NORMAL", color: "text-emerald-400" },
  WATCH: { label: "🟡 WATCH", color: "text-amber-400" },
  ELEVATED: { label: "🔴 ELEVATED", color: "text-red-400" },
};

const ACTION_STYLE: Record<string, string> = {
  ALLOW: "bg-emerald-500/15 border-emerald-500/40 text-emerald-300",
  REVIEW: "bg-amber-500/15 border-amber-500/40 text-amber-300",
  BLOCK: "bg-red-500/15 border-red-500/40 text-red-300",
};

function riskBand(score: number) {
  if (score >= 80) return { label: "HIGH RISK (model)", cls: "bg-red-500/15 border-red-500/40 text-red-300" };
  if (score >= 40) return { label: "MEDIUM RISK (model)", cls: "bg-amber-500/15 border-amber-500/40 text-amber-300" };
  return { label: "LOW RISK (model)", cls: "bg-emerald-500/15 border-emerald-500/40 text-emerald-300" };
}

export default function LiveScoring() {
  const [entities, setEntities] = useState<string[]>([]);
  const [selectedEntity, setSelectedEntity] = useState<string>("");
  const [txns, setTxns] = useState<TxnSummary[]>([]);
  const [txnIdx, setTxnIdx] = useState(0);
  const [escalation, setEscalation] = useState<EscalationState | null>(null);
  const [result, setResult] = useState<ScoreResult | null>(null);
  const [explanation, setExplanation] = useState<ExplanationResult | null>(null);
  const [loadingEntities, setLoadingEntities] = useState(true);
  const [loadingTxns, setLoadingTxns] = useState(false);
  const [scoring, setScoring] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    api
      .listEntities()
      .then((d) => {
        setEntities(d.entities);
        if (d.entities.length) setSelectedEntity(d.entities[0]);
      })
      .catch((e) => setError(String(e)))
      .finally(() => setLoadingEntities(false));
  }, []);

  useEffect(() => {
    if (!selectedEntity) return;
    setLoadingTxns(true);
    setResult(null);
    setTxnIdx(0);
    Promise.all([
      api.listTransactions(selectedEntity),
      api.getEscalation(selectedEntity),
    ])
      .then(([t, esc]) => {
        setTxns(t.transactions);
        setEscalation(esc);
      })
      .catch((e) => setError(String(e)))
      .finally(() => setLoadingTxns(false));
  }, [selectedEntity]);

  const currentTxn = useMemo(() => txns[txnIdx], [txns, txnIdx]);

  // The score/decision comes back immediately; the LLM explanation is
  // generated afterward in the background, so we poll for it separately
  // rather than blocking the scoring request on the LLM round-trip.
  useEffect(() => {
    if (!result) return;
    let cancelled = false;
    setExplanation({ status: "pending" });

    (async () => {
      for (let i = 0; i < 30 && !cancelled; i++) {
        try {
          const exp = await api.getExplanation(result.verdict_id);
          if (cancelled) return;
          if (exp.status === "ready") {
            setExplanation(exp);
            return;
          }
        } catch {
          // transient — keep polling
        }
        await new Promise((resolve) => setTimeout(resolve, 1000));
      }
    })();

    return () => {
      cancelled = true;
    };
  }, [result?.verdict_id]);

  async function handleReset() {
    if (!selectedEntity) return;
    await api.resetEntity(selectedEntity);
    const esc = await api.getEscalation(selectedEntity);
    setEscalation(esc);
    setResult(null);
    setExplanation(null);
  }

  async function handleScore() {
    if (!selectedEntity) return;
    setScoring(true);
    setError(null);
    try {
      const r = await api.score(selectedEntity, txnIdx);
      setResult(r);
      const esc = await api.getEscalation(selectedEntity);
      setEscalation(esc);
    } catch (e) {
      setError(String(e));
    } finally {
      setScoring(false);
    }
  }

  const stateInfo = escalation ? STATE_DISPLAY[escalation.state] ?? { label: escalation.state, color: "text-slate-300" } : null;

  return (
    <div className="grid grid-cols-1 lg:grid-cols-[280px_1fr] gap-6">
      <aside className="bg-slate-900/60 border border-slate-800 rounded-xl p-4 h-fit space-y-4">
        <h2 className="text-sm font-semibold text-slate-300 uppercase tracking-wide">Entity Session</h2>

        <div>
          <label className="block text-xs text-slate-400 mb-1">Entity (card/account fingerprint)</label>
          <select
            className="w-full bg-slate-800 border border-slate-700 rounded-md px-2 py-1.5 text-sm text-slate-100 disabled:opacity-50"
            value={selectedEntity}
            disabled={loadingEntities}
            onChange={(e) => setSelectedEntity(e.target.value)}
          >
            {entities.map((id) => (
              <option key={id} value={id}>
                {id}
              </option>
            ))}
          </select>
        </div>

        <p className="text-xs text-slate-500">
          {loadingTxns ? "Loading..." : `${txns.length} transactions in this entity's sequence`}
        </p>

        <div>
          <label className="block text-xs text-slate-400 mb-1">
            Transaction # in sequence: {txnIdx}
          </label>
          <input
            type="range"
            min={0}
            max={Math.max(txns.length - 1, 0)}
            value={txnIdx}
            onChange={(e) => setTxnIdx(Number(e.target.value))}
            disabled={!txns.length}
            className="w-full"
          />
          {currentTxn && (
            <p className="text-xs text-slate-500 mt-1">
              ₹{currentTxn.TransactionAmt.toFixed(2)} · {currentTxn.ProductCD}
            </p>
          )}
        </div>

        <button
          onClick={handleReset}
          className="w-full text-sm px-3 py-1.5 rounded-md border border-slate-700 text-slate-300 hover:bg-slate-800 transition"
        >
          Reset entity memory
        </button>

        <button
          onClick={handleScore}
          disabled={!txns.length || scoring}
          className="w-full text-sm font-medium px-3 py-2 rounded-md bg-indigo-500 hover:bg-indigo-400 disabled:opacity-50 disabled:cursor-not-allowed text-white transition"
        >
          {scoring ? "Scoring..." : "Score this transaction"}
        </button>

        {error && <p className="text-xs text-red-400 break-words">{error}</p>}
      </aside>

      <section>
        {!result ? (
          <div className="bg-slate-900/40 border border-dashed border-slate-700 rounded-xl p-6 text-slate-400 text-sm">
            Pick an entity and transaction number in the sidebar, then click{" "}
            <span className="text-slate-200 font-medium">Score this transaction</span>. Score several
            transactions from the same entity in sequence to see the escalation state build up.
          </div>
        ) : (
          <div className="grid grid-cols-1 md:grid-cols-[1fr_2fr] gap-6">
            <div className="space-y-5">
              <div className="bg-slate-900/60 border border-slate-800 rounded-xl p-4">
                <p className="text-xs text-slate-400 uppercase tracking-wide mb-1">Risk Score</p>
                <p className="text-3xl font-semibold text-slate-100">{result.risk_score} / 100</p>
                <div className={`mt-3 border rounded-md px-3 py-1.5 text-sm font-medium ${riskBand(result.risk_score).cls}`}>
                  {riskBand(result.risk_score).label}
                </div>
              </div>

              <div className="bg-slate-900/60 border border-slate-800 rounded-xl p-4">
                <h3 className="text-sm font-semibold text-slate-300 mb-2">Entity Escalation State</h3>
                <p className={`text-sm font-medium ${stateInfo?.color}`}>{stateInfo?.label}</p>
                <p className="text-xs text-slate-500 mt-1">
                  {result.escalation_before.recent_risky_count} risky verdicts in last{" "}
                  {result.escalation_before.recent_verdict_count} transactions for this entity
                </p>
                {result.escalation_before.recent_verdicts.length > 0 && (
                  <p className="text-xs text-slate-400 mt-2">
                    Recent verdicts: {result.escalation_before.recent_verdicts.join(" → ")}
                  </p>
                )}
              </div>

              <div className="bg-slate-900/60 border border-slate-800 rounded-xl p-4">
                <h3 className="text-sm font-semibold text-slate-300 mb-2">Top Contributing Factors</h3>
                <ul className="space-y-1.5 text-sm text-slate-300">
                  {result.top_factors.map((f) => (
                    <li key={f.feature}>
                      <span className="font-medium text-slate-100">{f.label}</span>: {f.value}{" "}
                      <span className="text-slate-500">
                        (impact: {f.contribution >= 0 ? "+" : ""}
                        {f.contribution.toFixed(3)})
                      </span>
                    </li>
                  ))}
                </ul>
              </div>
            </div>

            <div className="bg-slate-900/60 border border-slate-800 rounded-xl p-4 space-y-3">
              <h3 className="text-sm font-semibold text-slate-300">Automated Decision</h3>
              <div className={`border rounded-md px-3 py-2 text-sm font-medium ${ACTION_STYLE[result.decision.action] ?? ""}`}>
                Action: <span className="font-semibold">{result.decision.action}</span>
              </div>
              {result.decision.escalated_due_to_history && (
                <div className="border border-indigo-500/40 bg-indigo-500/10 text-indigo-300 rounded-md px-3 py-2 text-sm">
                  ⚠️ Action escalated due to this entity's recent risk trajectory.
                </div>
              )}
              <p className="text-xs text-slate-500">
                Decided synchronously from the score and rules — this is what actually gated the
                transaction, before any LLM call.
              </p>

              <div className="pt-2 border-t border-slate-800">
                <h4 className="text-sm font-semibold text-slate-300 mb-2">AI Reviewer Explanation</h4>
                {!explanation || explanation.status === "pending" ? (
                  <p className="text-sm text-slate-500 italic">Generating explanation…</p>
                ) : (
                  <div className="space-y-2">
                    {explanation.verdict.action !== result.decision.action && (
                      <div className="border border-amber-500/40 bg-amber-500/10 text-amber-300 rounded-md px-3 py-2 text-xs">
                        Note: the AI reviewer's suggested action ({explanation.verdict.action}) differs
                        from the automated decision above — the automated decision is authoritative
                        and already final; this is reviewer context only.
                      </div>
                    )}
                    <p className="text-sm text-slate-200">{explanation.verdict.explanation}</p>
                    <p className="text-xs text-slate-500">Rationale: {explanation.verdict.rationale}</p>
                  </div>
                )}
              </div>
            </div>
          </div>
        )}
      </section>
    </div>
  );
}
