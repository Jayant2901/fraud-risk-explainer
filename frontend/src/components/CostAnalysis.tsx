import { useEffect, useState } from "react";
import { api, type CostAnalysis as CostAnalysisData, type CostSensitivity } from "../api/client";

export default function CostAnalysis() {
  const [fraudLoss, setFraudLoss] = useState(5000);
  const [fpCost, setFpCost] = useState(150);
  const [data, setData] = useState<CostAnalysisData | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [sensitivity, setSensitivity] = useState<CostSensitivity | null>(null);
  const [sensitivityMessage, setSensitivityMessage] = useState<string | null>(null);
  const [sensitivityError, setSensitivityError] = useState<string | null>(null);

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

  return (
    <div className="space-y-6">
      <div>
        <h2 className="text-lg font-semibold text-slate-100">Cost-Optimal Threshold Analysis</h2>
        <p className="text-sm text-slate-400 mt-1">
          Translates the model's threshold choice into business cost, instead of reporting accuracy
          alone. Assumptions are adjustable below.
        </p>
      </div>

      <div className="grid grid-cols-1 sm:grid-cols-2 gap-4 max-w-xl">
        <div>
          <label htmlFor="avg-fraud-loss" className="block text-xs text-slate-400 mb-1">
            Assumed avg. fraud loss per missed fraud (₹)
          </label>
          <input
            id="avg-fraud-loss"
            type="number"
            step={100}
            value={fraudLoss}
            onChange={(e) => setFraudLoss(Number(e.target.value))}
            className="w-full bg-slate-800 border border-slate-700 rounded-md px-2 py-1.5 text-sm text-slate-100"
          />
        </div>
        <div>
          <label htmlFor="avg-fp-cost" className="block text-xs text-slate-400 mb-1">
            Assumed cost per wrongly-flagged legit transaction (₹)
          </label>
          <input
            id="avg-fp-cost"
            type="number"
            step={10}
            value={fpCost}
            onChange={(e) => setFpCost(Number(e.target.value))}
            className="w-full bg-slate-800 border border-slate-700 rounded-md px-2 py-1.5 text-sm text-slate-100"
          />
        </div>
      </div>

      {error && <p className="text-sm text-red-400">{error}</p>}

      {data && (
        data.eval_report ? (
          <pre className="bg-slate-900/60 border border-slate-800 rounded-xl p-4 text-xs text-slate-300 overflow-x-auto whitespace-pre">
            {data.eval_report}
          </pre>
        ) : (
          <p className="text-sm text-amber-400 bg-amber-500/10 border border-amber-500/30 rounded-md px-3 py-2">
            Run <code>python src/train_model.py</code> first to generate the eval report and cost
            curve.
          </p>
        )
      )}

      <div>
        <h3 className="text-sm font-semibold text-slate-100">
          How sensitive is the optimal threshold to these assumptions?
        </h3>
        <p className="text-xs text-slate-400 mt-1 max-w-2xl">
          The Rs {data?.defaults.avg_fraud_loss.toLocaleString() ?? "5,000"}/Rs{" "}
          {data?.defaults.avg_fp_cost.toLocaleString() ?? "150"} pair above is a single point
          estimate. This table sweeps both costs from 0.5x to 2x their defaults and shows the
          resulting cost-optimal threshold for each combination — rows are the assumed fraud loss,
          columns the assumed false-positive cost.
        </p>

        {sensitivityError && <p className="text-sm text-red-400 mt-2">{sensitivityError}</p>}

        {sensitivityMessage && (
          <p className="text-sm text-amber-400 bg-amber-500/10 border border-amber-500/30 rounded-md px-3 py-2 mt-2">
            {sensitivityMessage}
          </p>
        )}

        {sensitivity && (
          <div className="overflow-x-auto mt-3">
            <table className="text-xs text-slate-300 border-collapse">
              <thead>
                <tr>
                  <th className="border border-slate-700 bg-slate-800 px-3 py-1.5 text-left">
                    fraud loss \ fp cost
                  </th>
                  {sensitivity.fp_cost_multipliers.map((fpMult) => (
                    <th
                      key={fpMult}
                      className="border border-slate-700 bg-slate-800 px-3 py-1.5 text-right"
                    >
                      ₹{Math.round(sensitivity.base_fp_cost * fpMult).toLocaleString()}
                    </th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {sensitivity.fraud_loss_multipliers.map((flMult) => (
                  <tr key={flMult}>
                    <th className="border border-slate-700 bg-slate-800 px-3 py-1.5 text-left font-normal">
                      ₹{Math.round(sensitivity.base_fraud_loss * flMult).toLocaleString()}
                    </th>
                    {sensitivity.fp_cost_multipliers.map((fpMult) => {
                      const cell = sensitivity.grid.find(
                        (c) => c.fraud_loss_multiplier === flMult && c.fp_cost_multiplier === fpMult
                      );
                      return (
                        <td
                          key={fpMult}
                          className="border border-slate-700 px-3 py-1.5 text-right tabular-nums"
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
    </div>
  );
}
