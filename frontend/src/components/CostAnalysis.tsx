import { useEffect, useState } from "react";
import { api, type CostAnalysis as CostAnalysisData, type CostSensitivity, type DriftAnalysis } from "../api/client";
import { focusRing, typeScale } from "../theme";

const CHART_WIDTH = 480;
const CHART_HEIGHT = 140;
const CHART_PADDING = 24;

interface ChartPoint {
  bucket: number;
  rocAuc: number;
  x: number;
  y: number;
}

/** Maps each bucket's roc_auc onto SVG coordinates, auto-scaling the
 * y-axis to the data's own min/max (padded a little) so a small, real
 * spread is still visible rather than looking flat against a fixed
 * 0-1 axis. Buckets with no AUC (a single-class bucket) are skipped. */
function aucChartPoints(buckets: DriftAnalysis["buckets"]): ChartPoint[] {
  const withAuc = buckets.filter((b): b is typeof b & { roc_auc: number } => b.roc_auc !== null);
  if (withAuc.length === 0) return [];

  const aucs = withAuc.map((b) => b.roc_auc);
  const min = Math.min(...aucs);
  const max = Math.max(...aucs);
  const range = max - min || 0.01;
  const yPad = range * 0.2;

  const innerWidth = CHART_WIDTH - CHART_PADDING * 2;
  const innerHeight = CHART_HEIGHT - CHART_PADDING * 2;

  return withAuc.map((b, i) => {
    const x = CHART_PADDING + (withAuc.length === 1 ? 0 : (i / (withAuc.length - 1)) * innerWidth);
    const y = CHART_PADDING + innerHeight - ((b.roc_auc - (min - yPad)) / (range + 2 * yPad)) * innerHeight;
    return { bucket: b.bucket, rocAuc: b.roc_auc, x, y };
  });
}

export default function CostAnalysis() {
  const [fraudLoss, setFraudLoss] = useState(5000);
  const [fpCost, setFpCost] = useState(150);
  const [data, setData] = useState<CostAnalysisData | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [sensitivity, setSensitivity] = useState<CostSensitivity | null>(null);
  const [sensitivityMessage, setSensitivityMessage] = useState<string | null>(null);
  const [sensitivityError, setSensitivityError] = useState<string | null>(null);
  const [drift, setDrift] = useState<DriftAnalysis | null>(null);
  const [driftMessage, setDriftMessage] = useState<string | null>(null);
  const [driftError, setDriftError] = useState<string | null>(null);

  useEffect(() => {
    api
      .costAnalysis(fraudLoss, fpCost)
      .then(setData)
      .catch((e) => setError(String(e)));
  }, [fraudLoss, fpCost]);

  useEffect(() => {
    api
      .costSensitivity()
      .then((res) => {
        setSensitivity(res.sensitivity);
        setSensitivityMessage(res.message);
      })
      .catch((e) => setSensitivityError(String(e)));
  }, []);

  useEffect(() => {
    api
      .driftAnalysis()
      .then((res) => {
        setDrift(res.drift);
        setDriftMessage(res.message);
      })
      .catch((e) => setDriftError(String(e)));
  }, []);

  return (
    <div className="space-y-6">
      <div>
        <h2 className={typeScale.sectionTitle}>Cost-Optimal Threshold Analysis</h2>
        <p className={`${typeScale.body} mt-1`}>
          Translates the model's threshold choice into business cost, instead of reporting accuracy
          alone. Assumptions are adjustable below.
        </p>
      </div>

      <div className="grid grid-cols-1 sm:grid-cols-2 gap-4 max-w-xl">
        <div>
          <label htmlFor="avg-fraud-loss" className={`block ${typeScale.caption} mb-1`}>
            Assumed avg. fraud loss per missed fraud (₹)
          </label>
          <input
            id="avg-fraud-loss"
            type="number"
            step={100}
            value={fraudLoss}
            onChange={(e) => setFraudLoss(Number(e.target.value))}
            className={`w-full bg-neutral-900 border border-neutral-700 rounded-md px-2 py-1.5 text-sm text-neutral-100 ${focusRing}`}
          />
        </div>
        <div>
          <label htmlFor="avg-fp-cost" className={`block ${typeScale.caption} mb-1`}>
            Assumed cost per wrongly-flagged legit transaction (₹)
          </label>
          <input
            id="avg-fp-cost"
            type="number"
            step={10}
            value={fpCost}
            onChange={(e) => setFpCost(Number(e.target.value))}
            className={`w-full bg-neutral-900 border border-neutral-700 rounded-md px-2 py-1.5 text-sm text-neutral-100 ${focusRing}`}
          />
        </div>
      </div>

      {error && <p className="text-sm text-red-400">{error}</p>}

      {data && (
        data.eval_report ? (
          <pre className="border border-neutral-800 rounded-xl p-4 text-xs text-neutral-300 overflow-x-auto whitespace-pre">
            {data.eval_report}
          </pre>
        ) : (
          <p className="text-sm text-amber-300 bg-amber-500/10 border border-amber-500/30 rounded-md px-3 py-2">
            Run <code>python src/train_model.py</code> first to generate the eval report and cost
            curve.
          </p>
        )
      )}

      <div>
        <h3 className={typeScale.subTitle}>
          How sensitive is the optimal threshold to these assumptions?
        </h3>
        <p className={`${typeScale.caption} mt-1 max-w-2xl`}>
          The Rs {data?.defaults.avg_fraud_loss.toLocaleString() ?? "5,000"}/Rs{" "}
          {data?.defaults.avg_fp_cost.toLocaleString() ?? "150"} pair above is a single point
          estimate. This table sweeps both costs from 0.5x to 2x their defaults and shows the
          resulting cost-optimal threshold for each combination — rows are the assumed fraud loss,
          columns the assumed false-positive cost.
        </p>

        {sensitivityError && <p className="text-sm text-red-400 mt-2">{sensitivityError}</p>}

        {sensitivityMessage && (
          <p className="text-sm text-amber-300 bg-amber-500/10 border border-amber-500/30 rounded-md px-3 py-2 mt-2">
            {sensitivityMessage}
          </p>
        )}

        {sensitivity && (
          <div className="overflow-x-auto mt-3">
            <table className="text-xs text-neutral-300 w-full">
              <thead>
                <tr className="border-b border-neutral-700">
                  <th className="bg-neutral-900 px-3 py-1.5 text-left font-medium text-neutral-400">
                    fraud loss \ fp cost
                  </th>
                  {sensitivity.fp_cost_multipliers.map((fpMult) => (
                    <th
                      key={fpMult}
                      className="bg-neutral-900 px-3 py-1.5 text-right font-medium text-neutral-400"
                    >
                      ₹{Math.round(sensitivity.base_fp_cost * fpMult).toLocaleString()}
                    </th>
                  ))}
                </tr>
              </thead>
              <tbody className="divide-y divide-neutral-800">
                {sensitivity.fraud_loss_multipliers.map((flMult) => (
                  <tr key={flMult}>
                    <th className="px-3 py-1.5 text-left font-normal text-neutral-400 bg-neutral-900">
                      ₹{Math.round(sensitivity.base_fraud_loss * flMult).toLocaleString()}
                    </th>
                    {sensitivity.fp_cost_multipliers.map((fpMult) => {
                      const cell = sensitivity.grid.find(
                        (c) => c.fraud_loss_multiplier === flMult && c.fp_cost_multiplier === fpMult
                      );
                      return (
                        <td
                          key={fpMult}
                          className="px-3 py-1.5 text-right tabular-nums"
                        >
                          {cell ? cell.optimal_threshold : "—"}
                        </td>
                      );
                    })}
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </div>

      <div>
        <h3 className={typeScale.subTitle}>
          Does the model still perform well later in the test window?
        </h3>
        <p className={`${typeScale.caption} mt-1 max-w-2xl`}>
          A single AUC number from one split quietly assumes the model stays good forever. This
          buckets the same real test set by time and scores the already-trained model in each
          bucket separately — no retraining per bucket — to check for drift.
        </p>

        {driftError && <p className="text-sm text-red-400 mt-2">{driftError}</p>}

        {driftMessage && (
          <p className="text-sm text-amber-300 bg-amber-500/10 border border-amber-500/30 rounded-md px-3 py-2 mt-2">
            {driftMessage}
          </p>
        )}

        {drift && (() => {
          const points = aucChartPoints(drift.buckets);
          const aucs = points.map((p) => p.rocAuc);
          return (
            <div className="mt-3 border border-neutral-800 rounded-xl p-4">
              <svg
                viewBox={`0 0 ${CHART_WIDTH} ${CHART_HEIGHT}`}
                className="w-full max-w-lg"
                role="img"
                aria-label="ROC-AUC across time buckets of the test set"
              >
                <polyline
                  points={points.map((p) => `${p.x.toFixed(1)},${p.y.toFixed(1)}`).join(" ")}
                  fill="none"
                  stroke="#818cf8"
                  strokeWidth={2}
                />
                {points.map((p) => (
                  <g key={p.bucket}>
                    <circle cx={p.x} cy={p.y} r={3} fill="#818cf8" />
                    <text x={p.x} y={CHART_HEIGHT - 4} fontSize={9} fill="#94a3b8" textAnchor="middle">
                      {p.rocAuc.toFixed(3)}
                    </text>
                  </g>
                ))}
              </svg>
              {aucs.length > 0 && (
                <p className={`${typeScale.caption} mt-2`}>
                  {drift.buckets.length} buckets over a {(drift.span_seconds / 86400).toFixed(1)}-day test
                  window. ROC-AUC ranges from {Math.min(...aucs).toFixed(4)} to {Math.max(...aucs).toFixed(4)}{" "}
                  across buckets.
                </p>
              )}
            </div>
          );
        })()}
      </div>
    </div>
  );
}
