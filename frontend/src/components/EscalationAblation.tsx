import { useEffect, useState } from "react";
import { api } from "../api/client";
import { typeScale } from "../theme";

export default function EscalationAblation() {
  const [report, setReport] = useState<string | null>(null);
  const [message, setMessage] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    api
      .escalationAblation()
      .then((res) => {
        setReport(res.report);
        setMessage(res.message);
      })
      .catch((e) => setError(String(e)));
  }, []);

  return (
    <div className="space-y-3">
      <p className={`${typeScale.body} max-w-2xl`}>
        Does watching an entity's recent verdict history and escalating borderline scores actually
        catch more fraud than the raw model score alone? <code>src/escalation_ablation.py</code>{" "}
        replays the real chronological test set two ways — baseline (no escalation) vs. what the
        live system does today — and also sweeps a small grid of candidate severity-weighted
        cutoffs to pick the WATCH/ELEVATED boundaries by the same cost-minimization principle
        used elsewhere in this project, not a guess.
      </p>

      {error && <p className="text-sm text-app-danger">{error}</p>}

      {message && (
        <p className="text-sm text-amber-300 bg-amber-500/10 border border-amber-500/30 rounded-md px-3 py-2">
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
