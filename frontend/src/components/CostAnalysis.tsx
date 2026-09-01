import { useEffect, useState } from "react";
import { api, type CostAnalysis as CostAnalysisData } from "../api/client";

export default function CostAnalysis() {
  const [fraudLoss, setFraudLoss] = useState(5000);
  const [fpCost, setFpCost] = useState(150);
  const [data, setData] = useState<CostAnalysisData | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    api
      .costAnalysis(fraudLoss, fpCost)
      .then(setData)
      .catch((e) => setError(String(e)));
  }, [fraudLoss, fpCost]);

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
    </div>
  );
}
