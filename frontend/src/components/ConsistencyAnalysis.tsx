import { useEffect, useState } from "react";
import { api, type ConsistencyAnalysis as ConsistencyAnalysisData } from "../api/client";
import { typeScale } from "../theme";

function formatPct(p: number | null): string {
  return p === null ? "—" : `${(p * 100).toFixed(0)}%`;
}

const TH_CLASS = `px-3 py-2 font-medium text-app-faint ${typeScale.caption}`;
const TD_CLASS = "px-3 py-2";

export default function ConsistencyAnalysis() {
  const [data, setData] = useState<ConsistencyAnalysisData | null>(null);
  const [message, setMessage] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    api
      .consistencyAnalysis()
      .then((res) => {
        setData(res.consistency);
        setMessage(res.message);
      })
      .catch((e) => setError(String(e)));
  }, []);

  return (
    <div className="space-y-6">
      <div>
        <h2 className={typeScale.sectionTitle}>Does This System Agree With Itself?</h2>
        <p className={`${typeScale.body} mt-1 max-w-2xl`}>
          Razorpay's own engineering blog names a concrete pain point in merchant risk review:
          different analysts reaching different conclusions on the identical case. This project has
          no human reviewers to test that with, but the LLM agent (<code>src/llm_agent.py</code>)
          plays an equivalent role — so it's asked to look at the same case repeatedly, and its
          case-level judgment consistency is measured directly.
        </p>
      </div>

      {error && <p className="text-sm text-app-danger">{error}</p>}

      {message && (
        <p className="text-sm text-amber-300 bg-amber-500/10 border border-amber-500/30 rounded-md px-3 py-2">
          {message}
        </p>
      )}

      {data && (
        <>
          <div>
            <h3 className={typeScale.subTitle}>
              Part A — Boundary fragility (deterministic rules, no LLM calls)
            </h3>
            <p className={`${typeScale.body} mt-1.5`}>
              {data.part_a_boundary_fragility.n_near_boundary.toLocaleString()} of{" "}
              {data.part_a_boundary_fragility.n_flagged.toLocaleString()} flagged transactions (
              {formatPct(data.part_a_boundary_fragility.fraction_near_boundary)}) sit within ±2
              points of a decision boundary (the model's real, cost-derived review/block
              thresholds — see Cost-Optimal Threshold below) — close calls where a couple of
              points of model noise could have gone the other way.
            </p>
          </div>

          <div>
            <h3 className={typeScale.subTitle}>
              Part B — LLM self-consistency and cross-agreement (real Gemini API calls)
            </h3>
            <div className="overflow-x-auto mt-2">
              <table className="text-xs text-app-muted w-full">
                <thead>
                  <tr className="border-b border-app-rule">
                    <th className={`${TH_CLASS} text-left`}>Band</th>
                    <th className={`${TH_CLASS} text-left`}>Escalation</th>
                    <th className={`${TH_CLASS} text-right`}>Risk score</th>
                    <th className={`${TH_CLASS} text-left`}>Status</th>
                    <th className={`${TH_CLASS} text-right`}>Valid / excl.</th>
                    <th className={`${TH_CLASS} text-left`}>Modal action</th>
                    <th className={`${TH_CLASS} text-right`}>Self-consistency</th>
                    <th className={`${TH_CLASS} text-left`}>Cross-agree</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-app-rule">
                  {data.part_b_pairs.map((pair, i) => (
                    <tr key={i}>
                      <td className={TD_CLASS}>{pair.band}</td>
                      <td className={TD_CLASS}>{pair.escalation_context}</td>
                      <td className={`${TD_CLASS} text-right tabular-nums`}>
                        {pair.risk_score.toFixed(1)}
                      </td>
                      <td className={TD_CLASS}>
                        {pair.status === "insufficient_data" ? "insufficient data" : "ok"}
                      </td>
                      <td className={`${TD_CLASS} text-right tabular-nums`}>
                        {pair.n_valid} / {pair.n_excluded_fallback}
                      </td>
                      <td className={TD_CLASS}>{pair.modal_action ?? "—"}</td>
                      <td className={`${TD_CLASS} text-right tabular-nums`}>
                        {formatPct(pair.self_consistency_rate)}
                      </td>
                      <td className={TD_CLASS}>
                        {pair.cross_agreement === null ? "—" : pair.cross_agreement ? "yes" : "no"}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        </>
      )}
    </div>
  );
}
