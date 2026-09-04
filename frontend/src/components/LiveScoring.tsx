import { useEffect, useMemo, useState } from "react";
import { api, type EscalationState, type ExplanationResult, type ScoreResult, type TxnSummary, type Verdict } from "../api/client";
import { AlertTriangleIcon } from "./icons";
import RiskGauge from "./RiskGauge";
import { useAnimatedNumber } from "../hooks";
import {
  accentBg,
  accentBorder,
  accentText,
  actionStatus,
  buttonBase,
  buttonLabel,
  easing,
  escalationStatus,
  focusRing,
  noticeClass,
  pressable,
  statusBadgeClass,
  statusDotClass,
  statusTextClass,
  surface,
  typeScale,
} from "../theme";

// Renders the explanation as it arrives. Phase 8 simulated this with a
// typewriter over already-complete text, which was the honest option
// while the transport was polling; the text now genuinely streams from
// the model (SSE + Gemini's streaming API), so the simulation is gone
// rather than layered on top of the real thing.
//
// While streaming, the raw accumulated text is shown — it is partial
// JSON mid-flight, so only the explanation field is extracted for
// display. Once the terminal event lands, the validated verdict replaces
// it. No animation of our own, so reduced-motion visitors see exactly
// the same thing: text appearing at the speed the model produces it.
function ExplanationView({
  streamedText,
  verdict,
}: {
  streamedText: string;
  verdict: Verdict | null;
}) {
  if (verdict) {
    return (
      <>
        <p className="text-sm text-app-ink">{verdict.explanation}</p>
        <p className={typeScale.caption}>Rationale: {verdict.rationale}</p>
      </>
    );
  }
  const partial = partialExplanation(streamedText);
  if (!partial) return null;
  return <p className="text-sm text-app-ink">{partial}</p>;
}

// The model streams a JSON object, so mid-flight text looks like
// `{"explanation": "The card ha`. This pulls out just the explanation
// value so far, rather than showing the reader raw JSON.
export function partialExplanation(raw: string): string {
  const match = raw.match(/"explanation"\s*:\s*"((?:[^"\\]|\\.)*)/);
  if (!match) return "";
  return match[1].replace(/\\"/g, '"').replace(/\\n/g, " ").replace(/\\\\/g, "\\");
}

type Mode = "replay" | "custom";
type PlaySpeed = 1 | 2 | 4;

// The 1x base interval and the hard floor below are both chosen so that
// NO speed setting can sustain a rate above the live /api/score limit
// (30/minute = one call per 2000ms): 1x/2x land comfortably under it,
// and 4x (6000/4 = 1500ms) gets clamped up to the floor rather than
// actually firing at 1500ms. This deliberately overrides this phase's
// "start at ~1.5s" suggestion, which — sustained — would itself exceed
// the limit (40 calls/minute); the "never exceed 30/minute" rule wins.
const PLAY_BASE_INTERVAL_MS = 6000;
const PLAY_MIN_INTERVAL_MS = 2100;

function playIntervalMs(speed: PlaySpeed): number {
  return Math.max(PLAY_MIN_INTERVAL_MS, PLAY_BASE_INTERVAL_MS / speed);
}

// A plausible legitimate-looking transaction, pre-filled so the custom
// form is immediately submittable rather than starting blank.
const DEFAULT_CUSTOM_FORM = {
  TransactionAmt: 100,
  ProductCD: "W",
  card4: "visa",
  card6: "debit",
  P_emaildomain: "gmail.com",
  R_emaildomain: "",
  DeviceType: "mobile",
  addr1: 300,
  addr2: 87,
  hour_of_day: 14,
};

export default function LiveScoring() {
  const [mode, setMode] = useState<Mode>("replay");
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
  const [customForm, setCustomForm] = useState(DEFAULT_CUSTOM_FORM);
  const [customAttachEntityId, setCustomAttachEntityId] = useState<string>("");
  const [playing, setPlaying] = useState(false);
  const [playSpeed, setPlaySpeed] = useState<PlaySpeed>(1);
  const [thresholds, setThresholds] = useState<{ review: number; block: number } | null>(null);
  const [escalationBeat, setEscalationBeat] = useState<"review" | "block" | null>(null);
  // Raw text accumulated from explanation_delta events, before the
  // validated verdict arrives.
  const [streamedText, setStreamedText] = useState("");

  // The gauge eases from the previous score to the new one, so a second
  // score reads as a move rather than a jump-cut.
  const animatedScore = useAnimatedNumber(result?.risk_score ?? 0, 450);

  // Fetched once for the whole session — the decision boundary doesn't
  // change between scores, so this must not be a per-score request.
  useEffect(() => {
    api
      .costAnalysis(5000, 150)
      .then((data) => setThresholds(data.decision_thresholds))
      .catch(() => {
        // The gauge falls back to a plain score readout; scoring itself
        // is unaffected.
      });
  }, []);

  // One brief beat per escalated result, highlighting the tick the
  // entity's history pushed this transaction past. Fires once and
  // settles — never loops.
  useEffect(() => {
    if (!result?.decision.escalated_due_to_history) {
      setEscalationBeat(null);
      return;
    }
    setEscalationBeat(result.decision.action === "BLOCK" ? "block" : "review");
    const timer = setTimeout(() => setEscalationBeat(null), 600);
    return () => clearTimeout(timer);
  }, [result?.verdict_id]);

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
    setPlaying(false);
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

  // The score/decision comes back immediately; the LLM explanation
  // arrives afterward over SSE, token by token, so the text builds up as
  // the model actually produces it. One open connection, no polling
  // loop — GET /api/explanations/{id} remains as the fallback below for
  // clients behind a proxy that buffers event streams.
  useEffect(() => {
    if (!result) return;
    const verdictId = result.verdict_id;
    let cancelled = false;
    let source: EventSource | null = null;
    let attempt = 0;

    setExplanation({ status: "pending" });
    setStreamedText("");

    async function pollOnce() {
      // Fallback transport. One shot per second until the verdict is
      // ready, exactly as this component behaved before streaming.
      for (let i = 0; i < 30 && !cancelled; i++) {
        try {
          const exp = await api.getExplanation(verdictId);
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
    }

    function connect() {
      source = api.streamVerdict(verdictId);

      source.addEventListener("explanation_delta", (event) => {
        if (cancelled) return;
        const { text } = JSON.parse((event as MessageEvent).data) as { text: string };
        setStreamedText((current) => current + text);
      });

      source.addEventListener("explanation_complete", (event) => {
        if (cancelled) return;
        const verdict = JSON.parse((event as MessageEvent).data) as Verdict;
        setExplanation({ status: "ready", verdict });
        source?.close();
      });

      source.addEventListener("error", (event) => {
        // A named "error" event from the server is terminal and carries
        // the same fallback verdict the polling path would have returned.
        const data = (event as MessageEvent).data;
        if (data && !cancelled) {
          setExplanation({ status: "ready", verdict: JSON.parse(data) as Verdict });
          source?.close();
          return;
        }
        // Otherwise it's a transport failure. Retry once, then give up on
        // SSE and poll — never leave the UI stuck because a proxy blocked
        // the stream.
        source?.close();
        if (cancelled) return;
        attempt += 1;
        if (attempt === 1) {
          connect();
        } else {
          void pollOnce();
        }
      });
    }

    connect();

    return () => {
      cancelled = true;
      source?.close();
    };
  }, [result?.verdict_id]);

  async function handleReset() {
    if (!selectedEntity) return;
    setPlaying(false);
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

  // "Replay at speed" demo mode: while playing, scores the current
  // transaction, then advances to the next one after a delay, until the
  // entity's transaction list is exhausted or the user pauses. Each tick
  // is a fresh setTimeout keyed on txnIdx (rather than one long-lived
  // setInterval) so its closure always sees the current txnIdx/entity,
  // and pausing/unmounting/switching entities cleanly cancels the
  // pending tick via the effect's own cleanup — no leaked timers.
  useEffect(() => {
    if (!playing || !txns.length) return;
    if (txnIdx >= txns.length) {
      setPlaying(false);
      return;
    }
    const timer = setTimeout(() => {
      void (async () => {
        await handleScore();
        setTxnIdx((i) => {
          const next = i + 1;
          if (next >= txns.length) setPlaying(false);
          return next;
        });
      })();
    }, playIntervalMs(playSpeed));
    return () => clearTimeout(timer);
  }, [playing, txnIdx, playSpeed, txns.length, selectedEntity]);

  async function handleScoreCustom() {
    setScoring(true);
    setError(null);
    try {
      const attach = customAttachEntityId || undefined;
      const r = await api.scoreCustom({
        TransactionAmt: customForm.TransactionAmt,
        ProductCD: customForm.ProductCD || undefined,
        card4: customForm.card4 || undefined,
        card6: customForm.card6 || undefined,
        P_emaildomain: customForm.P_emaildomain || undefined,
        R_emaildomain: customForm.R_emaildomain || undefined,
        DeviceType: customForm.DeviceType || undefined,
        addr1: customForm.addr1,
        addr2: customForm.addr2,
        hour_of_day: customForm.hour_of_day,
        attach_to_entity_id: attach,
      });
      setResult(r);
      if (attach) {
        const esc = await api.getEscalation(attach);
        setEscalation(esc);
      } else {
        // Nothing was recorded anywhere — the "before" state IS the
        // current state, so there's nothing further to refetch.
        setEscalation(r.escalation_before);
      }
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
        <h2 className={`${typeScale.subTitle} uppercase tracking-wide text-app-faint`}>Entity Session</h2>

        <div role="tablist" className="flex gap-1 border-b border-app-rule">
          <button
            role="tab"
            aria-selected={mode === "replay"}
            onClick={() => setMode("replay")}
            className={`px-2 py-1.5 text-xs font-medium border-b-2 ${focusRing} ${
              mode === "replay" ? "border-app-accent text-app-accent-soft" : "border-transparent text-app-faint hover:text-app-muted"
            }`}
          >
            Replay historical
          </button>
          <button
            role="tab"
            aria-selected={mode === "custom"}
            onClick={() => setMode("custom")}
            className={`px-2 py-1.5 text-xs font-medium border-b-2 ${focusRing} ${
              mode === "custom" ? "border-app-accent text-app-accent-soft" : "border-transparent text-app-faint hover:text-app-muted"
            }`}
          >
            Score custom
          </button>
        </div>

        {mode === "replay" ? (
          <>
            <div>
              <label htmlFor="entity-select" className={`block ${typeScale.caption} mb-1`}>
                Entity (card/account fingerprint)
              </label>
              <select
                id="entity-select"
                className={`w-full bg-app-surface border border-app-rule rounded-md px-2 py-1.5 text-sm text-app-ink disabled:opacity-50 ${focusRing}`}
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
                className={`w-full accent-app-accent ${focusRing}`}
              />
              {currentTxn && (
                <p className={`${typeScale.caption} mt-1`}>
                  ₹{currentTxn.TransactionAmt.toFixed(2)} · {currentTxn.ProductCD}
                </p>
              )}
            </div>

            <div className="flex items-center gap-2">
              <button
                onClick={() => setPlaying((p) => !p)}
                disabled={!txns.length}
                className={`text-sm font-medium px-3 py-1.5 rounded-md border border-app-accent text-app-accent-soft bg-transparent hover:bg-app-accent/10 disabled:opacity-50 ${pressable} ${buttonLabel} ${buttonBase}`}
              >
                {playing ? "Pause" : "Play"}
              </button>
              <label htmlFor="play-speed" className="sr-only">
                Playback speed
              </label>
              <select
                id="play-speed"
                value={playSpeed}
                onChange={(e) => setPlaySpeed(Number(e.target.value) as PlaySpeed)}
                className={`bg-app-surface border border-app-rule rounded-md px-2 py-1.5 text-xs text-app-ink ${focusRing}`}
              >
                <option value={1}>1x</option>
                <option value={2}>2x</option>
                <option value={4}>4x</option>
              </select>
              <p className={typeScale.caption}>steps automatically through the sequence</p>
            </div>

            <button
              onClick={handleReset}
              className={`w-full text-sm px-3 py-1.5 rounded-md border border-transparent text-app-muted hover:bg-app-ink/5 ${pressable} ${buttonLabel} ${buttonBase}`}
            >
              Reset entity memory
            </button>

            <button
              onClick={handleScore}
              disabled={!txns.length || scoring || playing}
              className={`w-full text-sm font-medium px-3 py-2 rounded-md border border-app-accent text-app-accent-soft bg-transparent hover:bg-app-accent/10 ${pressable} ${buttonLabel} ${buttonBase}`}
            >
              {scoring ? "Scoring..." : "Score this transaction"}
            </button>
          </>
        ) : (
          <>
            <div>
              <label htmlFor="custom-amount" className={`block ${typeScale.caption} mb-1`}>
                Transaction amount (₹) *
              </label>
              <input
                id="custom-amount"
                type="number"
                value={customForm.TransactionAmt}
                onChange={(e) => setCustomForm({ ...customForm, TransactionAmt: Number(e.target.value) })}
                className={`w-full bg-app-surface border border-app-rule rounded-md px-2 py-1.5 text-sm text-app-ink ${focusRing}`}
              />
            </div>

            <div>
              <label htmlFor="custom-product" className={`block ${typeScale.caption} mb-1`}>
                Product code
              </label>
              <input
                id="custom-product"
                type="text"
                value={customForm.ProductCD}
                onChange={(e) => setCustomForm({ ...customForm, ProductCD: e.target.value })}
                className={`w-full bg-app-surface border border-app-rule rounded-md px-2 py-1.5 text-sm text-app-ink ${focusRing}`}
              />
            </div>

            <div className="grid grid-cols-2 gap-2">
              <div>
                <label htmlFor="custom-card4" className={`block ${typeScale.caption} mb-1`}>
                  Card network
                </label>
                <input
                  id="custom-card4"
                  type="text"
                  value={customForm.card4}
                  onChange={(e) => setCustomForm({ ...customForm, card4: e.target.value })}
                  className={`w-full bg-app-surface border border-app-rule rounded-md px-2 py-1.5 text-sm text-app-ink ${focusRing}`}
                />
              </div>
              <div>
                <label htmlFor="custom-card6" className={`block ${typeScale.caption} mb-1`}>
                  Card type
                </label>
                <input
                  id="custom-card6"
                  type="text"
                  value={customForm.card6}
                  onChange={(e) => setCustomForm({ ...customForm, card6: e.target.value })}
                  className={`w-full bg-app-surface border border-app-rule rounded-md px-2 py-1.5 text-sm text-app-ink ${focusRing}`}
                />
              </div>
            </div>

            <div>
              <label htmlFor="custom-p-email" className={`block ${typeScale.caption} mb-1`}>
                Purchaser email domain
              </label>
              <input
                id="custom-p-email"
                type="text"
                value={customForm.P_emaildomain}
                onChange={(e) => setCustomForm({ ...customForm, P_emaildomain: e.target.value })}
                className={`w-full bg-app-surface border border-app-rule rounded-md px-2 py-1.5 text-sm text-app-ink ${focusRing}`}
              />
            </div>

            <div>
              <label htmlFor="custom-r-email" className={`block ${typeScale.caption} mb-1`}>
                Recipient email domain
              </label>
              <input
                id="custom-r-email"
                type="text"
                value={customForm.R_emaildomain}
                onChange={(e) => setCustomForm({ ...customForm, R_emaildomain: e.target.value })}
                className={`w-full bg-app-surface border border-app-rule rounded-md px-2 py-1.5 text-sm text-app-ink ${focusRing}`}
              />
            </div>

            <div>
              <label htmlFor="custom-device" className={`block ${typeScale.caption} mb-1`}>
                Device type
              </label>
              <input
                id="custom-device"
                type="text"
                value={customForm.DeviceType}
                onChange={(e) => setCustomForm({ ...customForm, DeviceType: e.target.value })}
                className={`w-full bg-app-surface border border-app-rule rounded-md px-2 py-1.5 text-sm text-app-ink ${focusRing}`}
              />
            </div>

            <div className="grid grid-cols-2 gap-2">
              <div>
                <label htmlFor="custom-addr1" className={`block ${typeScale.caption} mb-1`}>
                  Billing region
                </label>
                <input
                  id="custom-addr1"
                  type="number"
                  value={customForm.addr1}
                  onChange={(e) => setCustomForm({ ...customForm, addr1: Number(e.target.value) })}
                  className={`w-full bg-app-surface border border-app-rule rounded-md px-2 py-1.5 text-sm text-app-ink ${focusRing}`}
                />
              </div>
              <div>
                <label htmlFor="custom-addr2" className={`block ${typeScale.caption} mb-1`}>
                  Billing country
                </label>
                <input
                  id="custom-addr2"
                  type="number"
                  value={customForm.addr2}
                  onChange={(e) => setCustomForm({ ...customForm, addr2: Number(e.target.value) })}
                  className={`w-full bg-app-surface border border-app-rule rounded-md px-2 py-1.5 text-sm text-app-ink ${focusRing}`}
                />
              </div>
            </div>

            <div>
              <label htmlFor="custom-hour" className={`block ${typeScale.caption} mb-1`}>
                Hour of day (0-23)
              </label>
              <input
                id="custom-hour"
                type="number"
                min={0}
                max={23}
                value={customForm.hour_of_day}
                onChange={(e) => setCustomForm({ ...customForm, hour_of_day: Number(e.target.value) })}
                className={`w-full bg-app-surface border border-app-rule rounded-md px-2 py-1.5 text-sm text-app-ink ${focusRing}`}
              />
            </div>

            <div>
              <label htmlFor="custom-attach" className={`block ${typeScale.caption} mb-1`}>
                Attach to entity
              </label>
              <select
                id="custom-attach"
                className={`w-full bg-app-surface border border-app-rule rounded-md px-2 py-1.5 text-sm text-app-ink ${focusRing}`}
                value={customAttachEntityId}
                onChange={(e) => setCustomAttachEntityId(e.target.value)}
              >
                <option value="">None — brand new entity</option>
                {entities.map((id) => (
                  <option key={id} value={id}>
                    {id}
                  </option>
                ))}
              </select>
            </div>

            <button
              onClick={handleScoreCustom}
              disabled={scoring}
              className={`w-full text-sm font-medium px-3 py-2 rounded-md border border-app-accent text-app-accent-soft bg-transparent hover:bg-app-accent/10 ${pressable} ${buttonLabel} ${buttonBase}`}
            >
              {scoring ? "Scoring..." : "Score this transaction"}
            </button>
          </>
        )}

        {error && <p className="text-xs text-app-danger break-words">{error}</p>}
      </aside>

      <section>
        {!result ? (
          <div className="border border-dashed border-app-rule rounded-xl p-6 text-app-faint text-sm">
            {mode === "replay" ? (
              <>
                Pick an entity and transaction number in the sidebar, then click{" "}
                <span className="text-app-ink font-medium">Score this transaction</span>. Score several
                transactions from the same entity in sequence to see the escalation state build up.
              </>
            ) : (
              <>
                Fill in the fields for a transaction that doesn't exist in the historical sample, then
                click <span className="text-app-ink font-medium">Score this transaction</span>. Optionally
                attach it to an existing entity to see how its current escalation state would treat this
                new transaction.
              </>
            )}
          </div>
        ) : (
          <div className="grid grid-cols-1 md:grid-cols-[1fr_2fr] gap-6">
            <div className="space-y-5">
              {/* Primary result — the one thing on this page that most needs
                  visual weight, so it gets a filled card instead of the
                  plain-outline treatment everything else uses. */}
              <div className={`${surface} p-5`}>
                {thresholds ? (
                  <RiskGauge
                    score={animatedScore}
                    reviewThreshold={thresholds.review}
                    blockThreshold={thresholds.block}
                    highlightThreshold={escalationBeat}
                    label="Risk Score"
                  />
                ) : (
                  <>
                    <p className={typeScale.caption}>Risk Score</p>
                    <p className="font-mono text-4xl font-bold text-app-ink mt-1 tabular-nums">
                      {result.risk_score}
                      <span className="text-lg font-normal text-app-faint"> / 100</span>
                    </p>
                  </>
                )}
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
                <ul className="space-y-1.5 text-sm text-app-muted mt-1.5">
                  {result.top_factors.map((f) => (
                    <li key={f.feature}>
                      <span className="font-medium text-app-ink">{f.label}</span>: {f.value}{" "}
                      <span className="text-app-faint">
                        (impact: {f.contribution >= 0 ? "+" : ""}
                        {f.contribution.toFixed(3)})
                      </span>
                    </li>
                  ))}
                </ul>
              </div>
            </div>

            <div className={`${surface} p-5 space-y-3`}>
              <h3 className={typeScale.subTitle}>Automated Decision</h3>
              <div
                className={`${statusBadgeClass(actionStatus[result.decision.action])} ${
                  escalationBeat ? "ring-2 ring-app-accent" : ""
                }`}
                style={{ transition: `box-shadow 300ms ${easing.standard}` }}
              >
                <span className={statusDotClass(actionStatus[result.decision.action])} />
                <span className="font-semibold">{result.decision.action}</span>
              </div>
              {result.decision.escalated_due_to_history && (
                <div className={`border ${accentBorder} ${accentBg} ${accentText} rounded-md px-3 py-2 text-sm flex items-start gap-2`}>
                  <AlertTriangleIcon className="h-4 w-4 mt-0.5 shrink-0" />
                  <span>
                    Escalated due to this entity's recent history — the raw score alone would not
                    have reached {result.decision.action}.
                  </span>
                </div>
              )}
              <p className={typeScale.caption}>
                Decided synchronously from the score and rules — this is what actually gated the
                transaction, before any LLM call.
              </p>

              <div className="pt-3 border-t border-app-rule">
                <h4 className={typeScale.subTitle}>AI Reviewer Explanation</h4>
                {/* The pending state stays exactly as it was: it covers
                    the window before the first delta arrives. */}
                {(!explanation || explanation.status === "pending") && !streamedText ? (
                  <p className="text-sm text-app-faint italic mt-1.5">Generating explanation…</p>
                ) : (
                  <div className="space-y-2 mt-1.5">
                    <ExplanationView
                      streamedText={streamedText}
                      verdict={explanation?.status === "ready" ? explanation.verdict : null}
                    />
                    {explanation?.status === "ready" && explanation.verdict.action !== result.decision.action && (
                      <div className={`${noticeClass("warning")} text-xs flex items-start gap-2`}>
                        <AlertTriangleIcon className="h-3.5 w-3.5 mt-0.5 shrink-0" />
                        <span>
                          Note: the AI reviewer's suggested action ({explanation.verdict.action}) differs
                          from the automated decision above — the automated decision is authoritative
                          and already final; this is reviewer context only.
                        </span>
                      </div>
                    )}
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
