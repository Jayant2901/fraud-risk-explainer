import { useEffect, useState } from "react";
import { api, type ConsistencyAnalysis as ConsistencyAnalysisData } from "../api/client";

function formatPct(p: number | null): string {
  return p === null ? "—" : `${(p * 100).toFixed(0)}%`;
}

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
        <h2 className="text-lg font-semibold text-slate-100">Does This System Agree With Itself?</h2>
        <p className="text-sm text-slate-400 mt-1 max-w-2xl">
          Razorpay's own engineering blog names a concrete pain point in merchant risk review:
          different analysts reaching different conclusions on the identical case. This project has
          no human reviewers to test that with, but the LLM agent (<code>src/llm_agent.py</code>)
          plays an equivalent role — so it's asked to look at the same case repeatedly, and its
          case-level judgment consistency is measured directly.
        </p>
      </div>

      {error && <p className="text-sm text-red-400">{error}</p>}

      {message && (
        <p className="text-sm text-amber-400 bg-amber-500/10 border border-amber-500/30 rounded-md px-3 py-2">
          {message}
        </p>
      )}

      {data && (
        <>
          <div className="bg-slate-900/60 border border-slate-800 rounded-xl p-4">
            <h3 className="text-sm font-semibold text-slate-300 mb-2">
              Part A — Boundary fragility (deterministic rules, no LLM calls)
            </h3>
            <p className="text-sm text-slate-200">
              {data.part_a_boundary_fragility.n_near_boundary.toLocaleString()} of{" "}
              {data.part_a_boundary_fragility.n_flagged.toLocaleString()} flagged transactions (
              {formatPct(data.part_a_boundary_fragility.fraction_near_boundary)}) sit within ±2
              points of a decision boundary (40 or 80) — close calls where a couple of points of
              model noise could have gone the other way.
            </p>
          </div>

          <div>
            <h3 className="text-sm font-semibold text-slate-300 mb-2">
              Part B — LLM self-consistency and cross-agreement (real Gemini API calls)
            </h3>
            <div className="overflow-x-auto">
              <table className="text-xs text-slate-300 border-collapse w-full">
                <thead>
                  <tr>
                    <th className="border border-slate-700 bg-slate-800 px-3 py-1.5 text-left">Band</th>
                    <th className="border border-slate-700 bg-slate-800 px-3 py-1.5 text-left">Escalation</th>
                    <th className="border border-slate-700 bg-slate-800 px-3 py-1.5 text-right">Risk score</th>
                    <th className="border border-slate-700 bg-slate-800 px-3 py-1.5 text-left">Status</th>
                    <th className="border border-slate-700 bg-slate-800 px-3 py-1.5 text-right">Valid / excl.</th>
                    <th className="border border-slate-700 bg-slate-800 px-3 py-1.5 text-left">Modal action</th>
                    <th className="border border-slate-700 bg-slate-800 px-3 py-1.5 text-right">Self-consistency</th>
                    <th className="border border-slate-700 bg-slate-800 px-3 py-1.5 text-left">Cross-agree</th>
                  </tr>
                </thead>
                <tbody>
                  {data.part_b_pairs.map((pair, i) => (
                    <tr key={i}>
                      <td className="border border-slate-700 px-3 py-1.5">{pair.band}</td>
                      <td className="border border-slate-700 px-3 py-1.5">{pair.escalation_context}</td>
                      <td className="border border-slate-700 px-3 py-1.5 text-right tabular-nums">
                        {pair.risk_score.toFixed(1)}
                      </td>
                      <td className="border border-slate-700 px-3 py-1.5">
                        {pair.status === "insufficient_data" ? "insufficient data" : "ok"}
                      </td>
                      <td className="border border-slate-700 px-3 py-1.5 text-right tabular-nums">
                        {pair.n_valid} / {pair.n_excluded_fallback}
                      </td>
                      <td className="border border-slate-700 px-3 py-1.5">{pair.modal_action ?? "—"}</td>
                      <td className="border border-slate-700 px-3 py-1.5 text-right tabular-nums">
                        {formatPct(pair.self_consistency_rate)}
                      </td>
                      <td className="border border-slate-700 px-3 py-1.5">
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
