import { useEffect, useMemo, useState } from "react";
import { api, type EscalationState, type ExplanationResult, type ScoreResult, type TxnSummary } from "../api/client";
import { AlertTriangleIcon } from "./icons";
import {
  accentBg,
  accentBorder,
  accentText,
  actionStatus,
  buttonBase,
  escalationStatus,
  focusRing,
  statusBadgeClass,
  statusDotClass,
  statusTextClass,
  typeScale,
} from "../theme";

function riskBand(score: number) {
  if (score >= 80) return { label: "HIGH RISK (model)", status: "danger" as const };
  if (score >= 40) return { label: "MEDIUM RISK (model)", status: "warning" as const };
  return { label: "LOW RISK (model)", status: "success" as const };
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

  const escStatus = escalation ? escalationStatus[escalation.state] : null;

  return (
    <div className="grid grid-cols-1 lg:grid-cols-[280px_1fr] gap-6">
      <aside className="space-y-4">
        <h2 className={`${typeScale.subTitle} uppercase tracking-wide text-neutral-400`}>Entity Session</h2>

        <div>
          <label htmlFor="entity-select" className={`block ${typeScale.caption} mb-1`}>
            Entity (card/account fingerprint)
          </label>
          <select
            id="entity-select"
            className={`w-full bg-neutral-900 border border-neutral-700 rounded-md px-2 py-1.5 text-sm text-neutral-100 disabled:opacity-50 ${focusRing}`}
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

        <p className={typeScale.caption}>
          {loadingTxns ? "Loading..." : `${txns.length} transactions in this entity's sequence`}
        </p>

        <div>
          <label htmlFor="txn-index-slider" className={`block ${typeScale.caption} mb-1`}>
            Transaction # in sequence: {txnIdx}
          </label>
          <input
            id="txn-index-slider"
            type="range"
            min={0}
            max={Math.max(txns.length - 1, 0)}
            value={txnIdx}
            onChange={(e) => setTxnIdx(Number(e.target.value))}
            disabled={!txns.length}
            className={`w-full accent-indigo-500 ${focusRing}`}
          />
          {currentTxn && (
            <p className={`${typeScale.caption} mt-1`}>
              ₹{currentTxn.TransactionAmt.toFixed(2)} · {currentTxn.ProductCD}
            </p>
          )}
        </div>

        <button
          onClick={handleReset}
          className={`w-full text-sm px-3 py-1.5 rounded-md border border-neutral-700 text-neutral-300 hover:bg-neutral-900 hover:border-neutral-600 ${buttonBase}`}
        >
          Reset entity memory
        </button>

        <button
          onClick={handleScore}
          disabled={!txns.length || scoring}
          className={`w-full text-sm font-medium px-3 py-2 rounded-md bg-indigo-500 hover:bg-indigo-400 text-white ${buttonBase}`}
        >
          {scoring ? "Scoring..." : "Score this transaction"}
        </button>

        {error && <p className="text-xs text-red-400 break-words">{error}</p>}
      </aside>

      <section>
        {!result ? (
          <div className="border border-dashed border-neutral-700 rounded-xl p-6 text-neutral-400 text-sm">
            Pick an entity and transaction number in the sidebar, then click{" "}
            <span className="text-neutral-200 font-medium">Score this transaction</span>. Score several
            transactions from the same entity in sequence to see the escalation state build up.
          </div>
        ) : (
          <div className="grid grid-cols-1 md:grid-cols-[1fr_2fr] gap-6">
            <div className="space-y-5">
              {/* Primary result — the one thing on this page that most needs
                  visual weight, so it gets a filled card instead of the
                  plain-outline treatment everything else uses. */}
              <div className="bg-neutral-900 border border-neutral-800 rounded-xl p-5">
                <p className={typeScale.caption}>Risk Score</p>
                <p className="text-4xl font-bold text-neutral-50 mt-1 tabular-nums">
                  {result.risk_score}
                  <span className="text-lg font-normal text-neutral-500"> / 100</span>
                </p>
                <div className={`mt-3 ${statusBadgeClass(riskBand(result.risk_score).status)}`}>
                  <span className={statusDotClass(riskBand(result.risk_score).status)} />
                  {riskBand(result.risk_score).label}
                </div>
              </div>

              <div>
                <h3 className={typeScale.subTitle}>Entity Escalation State</h3>
                {escStatus && (
                  <p className={`text-sm font-medium mt-1.5 flex items-center gap-1.5 ${statusTextClass(escStatus)}`}>
                    <span className={statusDotClass(escStatus)} />
                    {escalation?.state}
                  </p>
                )}
                <p className={`${typeScale.caption} mt-1`}>
                  {result.escalation_before.recent_risky_count} risky verdicts in last{" "}
                  {result.escalation_before.recent_verdict_count} transactions for this entity
                </p>
                {result.escalation_before.recent_verdicts.length > 0 && (
                  <p className={`${typeScale.caption} mt-2`}>
                    Recent verdicts: {result.escalation_before.recent_verdicts.join(" → ")}
                  </p>
                )}
              </div>

              <div>
                <h3 className={typeScale.subTitle}>Top Contributing Factors</h3>
                <ul className="space-y-1.5 text-sm text-neutral-300 mt-1.5">
                  {result.top_factors.map((f) => (
                    <li key={f.feature}>
                      <span className="font-medium text-neutral-100">{f.label}</span>: {f.value}{" "}
                      <span className="text-neutral-500">
                        (impact: {f.contribution >= 0 ? "+" : ""}
                        {f.contribution.toFixed(3)})
                      </span>
                    </li>
                  ))}
                </ul>
              </div>
            </div>

            <div className="bg-neutral-900 border border-neutral-800 rounded-xl p-5 space-y-3">
              <h3 className={typeScale.subTitle}>Automated Decision</h3>
              <div className={statusBadgeClass(actionStatus[result.decision.action])}>
                <span className={statusDotClass(actionStatus[result.decision.action])} />
                <span className="font-semibold">{result.decision.action}</span>
              </div>
              {result.decision.escalated_due_to_history && (
                <div className={`border ${accentBorder} ${accentBg} ${accentText} rounded-md px-3 py-2 text-sm flex items-start gap-2`}>
                  <AlertTriangleIcon className="h-4 w-4 mt-0.5 shrink-0" />
                  <span>Action escalated due to this entity's recent risk trajectory.</span>
                </div>
              )}
              <p className={typeScale.caption}>
                Decided synchronously from the score and rules — this is what actually gated the
                transaction, before any LLM call.
              </p>

              <div className="pt-3 border-t border-neutral-800">
                <h4 className={typeScale.subTitle}>AI Reviewer Explanation</h4>
                {!explanation || explanation.status === "pending" ? (
                  <p className="text-sm text-neutral-500 italic mt-1.5">Generating explanation…</p>
                ) : (
                  <div className="space-y-2 mt-1.5">
                    {explanation.verdict.action !== result.decision.action && (
                      <div className="border border-amber-500/30 bg-amber-500/10 text-amber-300 rounded-md px-3 py-2 text-xs flex items-start gap-2">
                        <AlertTriangleIcon className="h-3.5 w-3.5 mt-0.5 shrink-0" />
                        <span>
                          Note: the AI reviewer's suggested action ({explanation.verdict.action}) differs
                          from the automated decision above — the automated decision is authoritative
                          and already final; this is reviewer context only.
                        </span>
                      </div>
                    )}
                    <p className="text-sm text-neutral-200">{explanation.verdict.explanation}</p>
                    <p className={typeScale.caption}>Rationale: {explanation.verdict.rationale}</p>
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
