import { useEffect, useState } from "react";
import { api, type EscalationAblationSummary } from "../api/client";
import { BarChart, LineChart } from "./charts";
import { noticeClass, typeScale } from "../theme";

function formatRate(v: number): string {
  return v.toFixed(4);
}

export default function EscalationAblation() {
  const [report, setReport] = useState<string | null>(null);
  const [summary, setSummary] = useState<EscalationAblationSummary | null>(null);
  const [message, setMessage] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    api
      .escalationAblation()
      .then((res) => {
        setReport(res.report);
        setSummary(res.summary);
        setMessage(res.message);
      })
      .catch((e) => setError(String(e)));
  }, []);

  // Every watch candidate produces identical numbers for a given elevated
  // candidate (decide_action only branches on ELEVATED), so charting the
  // sweep by the elevated cutoff is the honest reduction — one point per
  // distinct cutoff, not nine overlapping ones.
  const sweepByElevated = summary
    ? [...new Map(summary.sweep.map((row) => [row.elevated_threshold, row])).values()].sort(
        (a, b) => a.elevated_threshold - b.elevated_threshold
      )
    : [];

  return (
    <div className="space-y-4">
      <p className={`${typeScale.body} max-w-2xl`}>
        Does watching an entity's recent verdict history and escalating borderline scores actually
        catch more fraud than the raw model score alone? <code>src/escalation_ablation.py</code>{" "}
        replays the real chronological test set two ways — baseline (no escalation) vs. what the
        live system does today — and also sweeps a small grid of candidate severity-weighted
        cutoffs to pick the WATCH/ELEVATED boundaries by the same cost-minimization principle
        used elsewhere in this project, not a guess.
      </p>

      {error && <p className="text-sm text-app-danger">{error}</p>}

      {message && <p className={`${noticeClass("warning")} text-sm`}>{message}</p>}

      {summary && (
        <div className="space-y-5">
          <div className="border border-app-rule rounded-xl p-4">
            <h4 className={typeScale.subTitle}>Escalation on vs. off</h4>
            <p className={`${typeScale.caption} mt-1`}>
              Over {summary.n_transactions.toLocaleString()} replayed transactions. Escalation buys
              recall and costs false flags — both bars move, which is the actual trade-off.
            </p>
            <div className="mt-3">
              <BarChart
                ariaLabel="Recall and false-flag rate, baseline versus escalation-adjusted"
                max={1}
                formatValue={formatRate}
                bars={[
                  {
                    label: "Recall — baseline",
                    value: summary.baseline.recall,
                    status: "neutral",
                  },
                  {
                    label: "Recall — with escalation",
                    value: summary.adjusted.recall,
                    status: "success",
                  },
                  {
                    label: "False flags — baseline",
                    value: summary.baseline.false_flag_rate,
                    status: "neutral",
                  },
                  {
                    label: "False flags — with escalation",
                    value: summary.adjusted.false_flag_rate,
                    status: "danger",
                  },
                ]}
              />
            </div>
            <p className={`${typeScale.caption} mt-3`}>
              Escalation-triggered flips: {summary.flips.n_flips.toLocaleString()}, of which{" "}
              {summary.flips.n_flips_fraud.toLocaleString()} were fraud — precision{" "}
              {formatRate(summary.flips.precision)}.
            </p>
          </div>

          {sweepByElevated.length > 1 && (
            <div className="border border-app-rule rounded-xl p-4 max-w-2xl">
              <h4 className={typeScale.subTitle}>Cost across candidate ELEVATED cutoffs</h4>
              <p className={`${typeScale.caption} mt-1`}>
                The swept grid, by total expected cost — the marked point is the cutoff the system
                actually uses. The WATCH cutoff isn't charted because it provably doesn't move any
                of these numbers: <code>decide_action()</code> only branches on ELEVATED.
              </p>
              <div className="mt-3">
                <LineChart
                  ariaLabel="Total cost across candidate elevated pressure cutoffs"
                  points={sweepByElevated.map((row) => ({
                    x: row.elevated_threshold,
                    y: row.cost,
                  }))}
                  formatY={(v) => `₹${Math.round(v).toLocaleString("en-IN")}`}
                />
              </div>
            </div>
          )}
        </div>
      )}

      {report && (
        <details className="border border-app-rule rounded-xl">
          <summary className={`${typeScale.caption} cursor-pointer px-4 py-3`}>
            Full text report (exact numbers)
          </summary>
          <pre className="px-4 pb-4 text-xs text-app-muted overflow-x-auto whitespace-pre">
            {report}
          </pre>
        </details>
      )}
    </div>
  );
}
