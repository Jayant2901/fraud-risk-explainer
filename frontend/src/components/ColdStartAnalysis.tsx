import { useEffect, useState } from "react";
import { api } from "../api/client";
import { noticeClass, typeScale } from "../theme";

export default function ColdStartAnalysis() {
  const [report, setReport] = useState<string | null>(null);
  const [message, setMessage] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    api
      .coldStartAnalysis()
      .then((res) => {
        setReport(res.report);
        setMessage(res.message);
      })
      .catch((e) => setError(String(e)));
  }, []);

  return (
    <div className="space-y-3">
      <p className={`${typeScale.body} max-w-2xl`}>
        A brand-new card/account with no history is the hardest, highest-volume real fraud case —
        <code>entity_prior_txn_count</code>/<code>entity_prior_fraud_rate</code> carry zero signal
        for exactly the rows that need it most. <code>src/graph_features.py</code> adds a causal
        (leakage-free) device/address graph signal aimed at that gap;{" "}
        <code>src/graph_features_ablation.py</code> measures whether it actually moves cold-start
        recall/precision, before vs. after, on the real test set's brand-new-entity rows only.
      </p>

      {error && <p className="text-sm text-app-danger">{error}</p>}

      {message && (
        <p className={`${noticeClass("warning")} text-sm`}>
          {message}
        </p>
      )}

      {report && (
        <pre className="border border-app-rule rounded-xl p-4 text-xs text-app-muted overflow-x-auto whitespace-pre">
          {report}
        </pre>
      )}
    </div>
  );
}
