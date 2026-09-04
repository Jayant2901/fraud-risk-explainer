import { useEffect, useState } from "react";
import { api, type ShadowComparison } from "../api/client";
import { BarChart, type Bar } from "./charts";
import { noticeClass, typeScale } from "../theme";

function formatRate(v: number): string {
  return `${(v * 100).toFixed(1)}%`;
}

function barStatus(liveAction: string, shadowAction: string): Bar["status"] {
  if (liveAction === shadowAction) return "success";
  // A candidate that would ALLOW what the live model flags is the
  // dangerous direction (missed fraud); the reverse is only extra
  // review load — worth distinguishing at a glance, not just "disagree".
  if (shadowAction === "ALLOW" && liveAction !== "ALLOW") return "danger";
  return "warning";
}

export default function ShadowScoring() {
  const [comparison, setComparison] = useState<ShadowComparison | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    api
      .shadowComparison()
      .then(setComparison)
      .catch((e) => setError(String(e)));
  }, []);

  return (
    <div className="space-y-4">
      <p className={`${typeScale.body} max-w-2xl`}>
        Before promoting a candidate model (a retrain, e.g. with{" "}
        <code>train_model.py --with-feedback</code>), <code>SHADOW_MODEL_PATH</code> scores every
        live transaction with it too — silently, never gating anything — and{" "}
        <code>src/shadow_scoring.py</code> tracks how often its decision would have differed from
        the one that actually shipped.
      </p>

      {error && <p className="text-sm text-app-danger">{error}</p>}

      {comparison && !comparison.configured && (
        <p className={`${noticeClass("warning")} text-sm`}>{comparison.message}</p>
      )}

      {comparison?.configured && comparison.total_scored === 0 && (
        <p className={`${noticeClass("warning")} text-sm`}>
          A shadow model is configured but hasn't scored anything yet — score a transaction to see
          a comparison.
        </p>
      )}

      {comparison?.configured && comparison.total_scored > 0 && (
        <div className="space-y-4">
          <div className="border border-app-rule rounded-xl p-4">
            <h4 className={typeScale.subTitle}>Agreement with the live model</h4>
            <p className={`${typeScale.caption} mt-1`}>
              Over {comparison.total_scored.toLocaleString()} transactions scored by both models.
            </p>
            <p className="mt-3 font-mono text-2xl text-app-ink tabular-nums">
              {comparison.agreement_rate !== null ? formatRate(comparison.agreement_rate) : "—"}
            </p>
          </div>

          <div className="border border-app-rule rounded-xl p-4 max-w-2xl">
            <h4 className={typeScale.subTitle}>Where the decisions diverge</h4>
            <p className={`${typeScale.caption} mt-1`}>
              Live action → shadow action, by share of transactions scored. Red is the direction
              that matters most: the candidate would have allowed what the live model flagged.
            </p>
            <div className="mt-3">
              <BarChart
                ariaLabel="Live versus shadow model decision pairs, by count"
                formatValue={(v) => v.toLocaleString()}
                bars={comparison.action_pairs.map((pair) => ({
                  label: `${pair.live_action} → ${pair.shadow_action}`,
                  value: pair.count,
                  status: barStatus(pair.live_action, pair.shadow_action),
                }))}
              />
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
